/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/exec/fuzzer/RowNumberFuzzer.h"
#include <boost/random/uniform_int_distribution.hpp>
#include <utility>
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int32(steps, 10, "Number of plans to generate and test.");

DEFINE_int32(
    duration_sec,
    0,
    "For how long it should run (in seconds). If zero, "
    "it executes exactly --steps iterations and exits.");

DEFINE_int32(
    batch_size,
    100,
    "The number of elements on each generated vector.");

DEFINE_int32(num_batches, 10, "The number of generated vectors.");

DEFINE_double(
    null_ratio,
    0.1,
    "Chance of adding a null value in a vector "
    "(expressed as double from 0 to 1).");

DEFINE_bool(enable_spill, true, "Whether to test plans with spilling enabled.");

DEFINE_int32(
    max_spill_level,
    -1,
    "Max spill level, -1 means random [0, 7], otherwise the actual level.");

DEFINE_bool(
    enable_oom_injection,
    false,
    "When enabled OOMs will randomly be triggered while executing query "
    "plans. The goal of this mode is to ensure unexpected exceptions "
    "aren't thrown and the process isn't killed in the process of cleaning "
    "up after failures. Therefore, results are not compared when this is "
    "enabled. Note that this option only works in debug builds.");

namespace facebook::velox::exec {
namespace {

class RowNumberFuzzer {
 public:
  explicit RowNumberFuzzer(
      size_t initialSeed,
      std::unique_ptr<test::ReferenceQueryRunner>);

  void go();

 private:
  static VectorFuzzer::Options getFuzzerOptions() {
    VectorFuzzer::Options opts;
    opts.vectorSize = FLAGS_batch_size;
    opts.stringVariableLength = true;
    opts.stringLength = 100;
    opts.nullRatio = FLAGS_null_ratio;
    return opts;
  }

  void seed(size_t seed) {
    currentSeed_ = seed;
    vectorFuzzer_.reSeed(seed);
    rng_.seed(currentSeed_);
  }

  void reSeed() {
    seed(rng_());
  }

  // Runs one test iteration from query plans generations, executions and result
  // verifications.
  void verify();

  int32_t randInt(int32_t min, int32_t max) {
    return boost::random::uniform_int_distribution<int32_t>(min, max)(rng_);
  }

  std::pair<std::vector<std::string>, std::vector<TypePtr>>
  generatePartitionKeys();

  std::vector<RowVectorPtr> generateInput(
      const std::vector<std::string>& keyNames,
      const std::vector<TypePtr>& keyTypes);

  void addPlansWithTableScan(
      const std::string& tableDir,
      const std::vector<std::string>& partitionKeys,
      const std::vector<RowVectorPtr>& input,
      std::vector<test::PlanWithSplits>& altPlans);

  // Makes the query plan with default settings in RowNumberFuzzer and value
  // inputs for both probe and build sides.
  //
  // NOTE: 'input' could either input rows with lazy
  // vectors or flatten ones.
  static test::PlanWithSplits makeDefaultPlan(
      const std::vector<std::string>& partitionKeys,
      const std::vector<RowVectorPtr>& input);

  static test::PlanWithSplits makePlanWithTableScan(
      const RowTypePtr& type,
      const std::vector<std::string>& partitionKeys,
      const std::vector<Split>& splits);

  FuzzerGenerator rng_;
  size_t currentSeed_{0};

  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool(
          "rowNumberFuzzer",
          memory::kMaxMemory,
          memory::MemoryReclaimer::create())};
  std::shared_ptr<memory::MemoryPool> pool_{rootPool_->addLeafChild(
      "rowNumberFuzzerLeaf",
      true,
      exec::MemoryReclaimer::create())};
  std::shared_ptr<memory::MemoryPool> writerPool_{rootPool_->addAggregateChild(
      "rowNumberFuzzerWriter",
      exec::MemoryReclaimer::create())};
  VectorFuzzer vectorFuzzer_;
  std::unique_ptr<test::ReferenceQueryRunner> referenceQueryRunner_;
};

RowNumberFuzzer::RowNumberFuzzer(
    size_t initialSeed,
    std::unique_ptr<test::ReferenceQueryRunner> referenceQueryRunner)
    : vectorFuzzer_{getFuzzerOptions(), pool_.get()},
      referenceQueryRunner_{std::move(referenceQueryRunner)} {
  test::setupReadWrite();
  seed(initialSeed);
}

template <typename T>
bool isDone(size_t i, T startTime) {
  if (FLAGS_duration_sec > 0) {
    std::chrono::duration<double> elapsed =
        std::chrono::system_clock::now() - startTime;
    return elapsed.count() >= FLAGS_duration_sec;
  }
  return i >= FLAGS_steps;
}

std::pair<std::vector<std::string>, std::vector<TypePtr>>
RowNumberFuzzer::generatePartitionKeys() {
  const auto numKeys = randInt(1, 3);
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto i = 0; i < numKeys; ++i) {
    names.push_back(fmt::format("c{}", i));
    types.push_back(vectorFuzzer_.randType(/*maxDepth=*/1));
  }
  return std::make_pair(names, types);
}

std::vector<RowVectorPtr> RowNumberFuzzer::generateInput(
    const std::vector<std::string>& keyNames,
    const std::vector<TypePtr>& keyTypes) {
  std::vector<std::string> names = keyNames;
  std::vector<TypePtr> types = keyTypes;
  // Add up to 3 payload columns.
  const auto numPayload = randInt(0, 3);
  for (auto i = 0; i < numPayload; ++i) {
    names.push_back(fmt::format("c{}", i + keyNames.size()));
    types.push_back(vectorFuzzer_.randType(/*maxDepth=*/2));
  }

  const auto inputType = ROW(std::move(names), std::move(types));
  std::vector<RowVectorPtr> input;
  input.reserve(FLAGS_num_batches);
  for (auto i = 0; i < FLAGS_num_batches; ++i) {
    input.push_back(vectorFuzzer_.fuzzInputRow(inputType));
  }

  return input;
}

test::PlanWithSplits RowNumberFuzzer::makeDefaultPlan(
    const std::vector<std::string>& partitionKeys,
    const std::vector<RowVectorPtr>& input) {
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<std::string> projectFields = partitionKeys;
  projectFields.emplace_back("row_number");
  auto plan = test::PlanBuilder()
                  .values(input)
                  .rowNumber(partitionKeys)
                  .project(projectFields)
                  .planNode();
  return test::PlanWithSplits{std::move(plan)};
}

test::PlanWithSplits RowNumberFuzzer::makePlanWithTableScan(
    const RowTypePtr& type,
    const std::vector<std::string>& partitionKeys,
    const std::vector<Split>& splits) {
  std::vector<std::string> projectFields = partitionKeys;
  projectFields.emplace_back("row_number");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId scanId;
  auto plan = test::PlanBuilder(planNodeIdGenerator)
                  .tableScan(type)
                  .rowNumber(partitionKeys)
                  .project(projectFields)
                  .planNode();
  return test::PlanWithSplits{plan, splits};
}

void RowNumberFuzzer::addPlansWithTableScan(
    const std::string& tableDir,
    const std::vector<std::string>& partitionKeys,
    const std::vector<RowVectorPtr>& input,
    std::vector<test::PlanWithSplits>& altPlans) {
  VELOX_CHECK(!tableDir.empty());

  if (!test::isTableScanSupported(input[0]->type())) {
    return;
  }

  const std::vector<Split> inputSplits = test::makeSplits(
      input, fmt::format("{}/row_number", tableDir), writerPool_);
  altPlans.push_back(makePlanWithTableScan(
      asRowType(input[0]->type()), partitionKeys, inputSplits));
}

void RowNumberFuzzer::verify() {
  const auto [keyNames, keyTypes] = generatePartitionKeys();
  const auto input = generateInput(keyNames, keyTypes);

  if (VLOG_IS_ON(1)) {
    // Flatten inputs.
    const auto flatInput = test::flatten(input);
    VLOG(1) << "Input: " << input[0]->toString();
    for (const auto& v : flatInput) {
      VLOG(1) << std::endl << v->toString(0, v->size());
    }
  }

  auto defaultPlan = makeDefaultPlan(keyNames, input);
  const auto expected =
      test::execute(defaultPlan, pool_, /*injectSpill=*/false, false);

  if (expected != nullptr) {
    if (!test::containsUnsupportedTypes(input[0]->type())) {
      auto [referenceResult, status] = test::computeReferenceResults(
          defaultPlan.plan, referenceQueryRunner_.get());
      if (referenceResult.has_value()) {
        VELOX_CHECK(
            test::assertEqualResults(
                referenceResult.value(),
                defaultPlan.plan->outputType(),
                {expected}),
            "Velox and Reference results don't match");
      }
    }
  }

  std::vector<test::PlanWithSplits> altPlans;
  altPlans.push_back(std::move(defaultPlan));

  const auto tableScanDir = exec::test::TempDirectoryPath::create();
  addPlansWithTableScan(tableScanDir->getPath(), keyNames, input, altPlans);

  for (auto i = 0; i < altPlans.size(); ++i) {
    LOG(INFO) << "Testing plan #" << i;
    auto actual = test::execute(
        altPlans[i], pool_, /*injectSpill=*/false, FLAGS_enable_oom_injection);
    if (actual != nullptr && expected != nullptr) {
      VELOX_CHECK(
          test::assertEqualResults({expected}, {actual}),
          "Logically equivalent plans produced different results");
    } else {
      VELOX_CHECK(
          FLAGS_enable_oom_injection, "Got unexpected nullptr for results");
    }

    if (FLAGS_enable_spill) {
      LOG(INFO) << "Testing plan #" << i << " with spilling";
      const auto fuzzMaxSpillLevel =
          FLAGS_max_spill_level == -1 ? randInt(0, 7) : FLAGS_max_spill_level;
      actual = test::execute(
          altPlans[i],
          pool_,
          /*=injectSpill=*/true,
          FLAGS_enable_oom_injection,
          "core::QueryConfig::kRowNumberSpillEnabled",
          fuzzMaxSpillLevel);
      if (actual != nullptr && expected != nullptr) {
        try {
          VELOX_CHECK(
              test::assertEqualResults({expected}, {actual}),
              "Logically equivalent plans produced different results");
        } catch (const VeloxException&) {
          LOG(ERROR) << "Expected\n"
                     << expected->toString(0, expected->size()) << "\nActual\n"
                     << actual->toString(0, actual->size());
          throw;
        }
      } else {
        VELOX_CHECK(
            FLAGS_enable_oom_injection, "Got unexpected nullptr for results");
      }
    }
  }
}

void RowNumberFuzzer::go() {
  VELOX_USER_CHECK(
      FLAGS_steps > 0 || FLAGS_duration_sec > 0,
      "Either --steps or --duration_sec needs to be greater than zero.");
  VELOX_USER_CHECK_GE(FLAGS_batch_size, 10, "Batch size must be at least 10.");

  const auto startTime = std::chrono::system_clock::now();
  size_t iteration = 0;

  while (!isDone(iteration, startTime)) {
    LOG(INFO) << "==============================> Started iteration "
              << iteration << " (seed: " << currentSeed_ << ")";
    verify();
    LOG(INFO) << "==============================> Done with iteration "
              << iteration;

    reSeed();
    ++iteration;
  }
}
} // namespace

void rowNumberFuzzer(
    size_t seed,
    std::unique_ptr<test::ReferenceQueryRunner> referenceQueryRunner) {
  RowNumberFuzzer(seed, std::move(referenceQueryRunner)).go();
}
} // namespace facebook::velox::exec
