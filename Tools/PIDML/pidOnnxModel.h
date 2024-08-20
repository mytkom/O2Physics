// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file pidONNXModel.h
/// \brief A class that wraps PID ML ONNX model. See README.md for more detailed instructions.
///
/// \author Maja Kabus <mkabus@cern.ch>

#ifndef TOOLS_PIDML_PIDONNXMODEL_H_
#define TOOLS_PIDML_PIDONNXMODEL_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <algorithm>
#include <map>
#include <type_traits>
#include <utility>
#include <memory>
#include <vector>
#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>
#else
#include <onnxruntime_cxx_api.h>
#endif

#include <arrow/builder.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type_traits.h>
#include <arrow/util/key_value_metadata.h>
#include "Framework/TableBuilder.h"
#include "Framework/Expressions.h"
#include "arrow/table.h"
#include "gandiva/tree_expr_builder.h"
#include "arrow/table.h"
#include <arrow/api.h>
#include <arrow/compute/api.h>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "CCDB/CcdbApi.h"
#include "Tools/PIDML/pidUtils.h"
#include "Common/DataModel/PIDResponse.h"

using namespace pidml::pidutils;

enum PidMLDetector {
  kTPCOnly = 0,
  kTPCTOF,
  kTPCTOFTRD,
  kNDetectors ///< number of available detectors configurations
};

namespace pidml_pt_cuts
{
// TODO: for now first limit wouldn't be used,
// network needs TPC, so we can either do not cut it by p or return 0.0f as prediction
constexpr std::array<double, kNDetectors> defaultModelPLimits({0.0, 0.5, 0.8});
} // namespace pidml_pt_cuts

// TODO: Copied from cefpTask, shall we put it in some common utils code?
namespace
{
bool readJsonFile(const std::string& config, rapidjson::Document& d)
{
  FILE* fp = fopen(config.data(), "rb");
  if (!fp) {
    LOG(error) << "Missing configuration json file: " << config;
    return false;
  }

  char readBuffer[65536];
  rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

  d.ParseStream(is);
  fclose(fp);
  return true;
}
} // namespace


struct PidONNXModel {
 public:
  PidONNXModel(std::string& localPath, std::string& ccdbPath, bool useCCDB, o2::ccdb::CcdbApi& ccdbApi, uint64_t timestamp,
               int pid, double minCertainty, const double* pLimits = &pidml_pt_cuts::defaultModelPLimits[0])
    : mPid(pid), mMinCertainty(minCertainty), mPLimits(pLimits, pLimits + kNDetectors)
  {
    assert(mPLimits.size() == kNDetectors);

    std::string modelFile;
    loadInputFiles(localPath, ccdbPath, useCCDB, ccdbApi, timestamp, pid, modelFile);

    Ort::SessionOptions sessionOptions;
    mEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "pid-onnx-inferer");
    LOG(info) << "Loading ONNX model from file: " << modelFile;
#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
    mSession.reset(new Ort::Experimental::Session{*mEnv, modelFile, sessionOptions});
#else
    mSession.reset(new Ort::Session{*mEnv, modelFile.c_str(), sessionOptions});
#endif
    LOG(info) << "ONNX model loaded";

#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
    mInputNames = mSession->GetInputNames();
    mInputShapes = mSession->GetInputShapes();
    mOutputNames = mSession->GetOutputNames();
    mOutputShapes = mSession->GetOutputShapes();
#else
    Ort::AllocatorWithDefaultOptions tmpAllocator;
    for (size_t i = 0; i < mSession->GetInputCount(); ++i) {
      mInputNames.push_back(mSession->GetInputNameAllocated(i, tmpAllocator).get());
    }
    for (size_t i = 0; i < mSession->GetInputCount(); ++i) {
      mInputShapes.emplace_back(mSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    for (size_t i = 0; i < mSession->GetOutputCount(); ++i) {
      mOutputNames.push_back(mSession->GetOutputNameAllocated(i, tmpAllocator).get());
    }
    for (size_t i = 0; i < mSession->GetOutputCount(); ++i) {
      mOutputShapes.emplace_back(mSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
#endif

    LOG(debug) << "Input Node Name/Shape (" << mInputNames.size() << "):";
    for (size_t i = 0; i < mInputNames.size(); i++) {
      LOG(debug) << "\t" << mInputNames[i] << " : " << printShape(mInputShapes[i]);
    }

    LOG(debug) << "Output Node Name/Shape (" << mOutputNames.size() << "):";
    for (size_t i = 0; i < mOutputNames.size(); i++) {
      LOG(debug) << "\t" << mOutputNames[i] << " : " << printShape(mOutputShapes[i]);
    }

    // Assume model has 1 input node and 1 output node.
    assert(mInputNames.size() == 1 && mOutputNames.size() == 1);
  }
  PidONNXModel() = default;
  PidONNXModel(PidONNXModel&&) = default;
  PidONNXModel& operator=(PidONNXModel&&) = default;
  PidONNXModel(const PidONNXModel&) = delete;
  PidONNXModel& operator=(const PidONNXModel&) = delete;
  ~PidONNXModel() = default;

  std::shared_ptr<arrow::ChunkedArray> convertToFloat32(const std::shared_ptr<arrow::ChunkedArray>& chunkedArray)
  {
    const uint32_t n = chunkedArray->num_chunks();
    arrow::ArrayVector chunks;
    arrow::compute::ExecContext exec_context(arrow::default_memory_pool());
    arrow::compute::CastOptions options;
    options.to_type = arrow::float32();

    for (int i = 0; i < n; ++i) {
      std::shared_ptr<arrow::Array> chunk = chunkedArray->chunk(i);

      auto result = arrow::compute::Cast(chunk, options, &exec_context);
      if (!result.ok()) {
        LOG(fatal) << "Error casting chunk: " << result.status();
        return nullptr;
      }

      auto floatChunk = result.ValueOrDie();
      chunks.push_back(floatChunk.make_array());
    }

    return std::make_shared<arrow::ChunkedArray>(chunks);
  }

  gandiva::NodePtr makeScalingNode(gandiva::NodePtr field_node, std::pair<float, float>& scalingParams)
  {
    gandiva::NodePtr literal_mean = gandiva::TreeExprBuilder::MakeLiteral(scalingParams.first);
    gandiva::NodePtr literal_stdev = gandiva::TreeExprBuilder::MakeLiteral(scalingParams.second);

    gandiva::NodePtr sub_node = gandiva::TreeExprBuilder::MakeFunction("subtract", {field_node, literal_mean}, arrow::float32());
    gandiva::NodePtr div_node = gandiva::TreeExprBuilder::MakeFunction("divide", {sub_node, literal_stdev}, arrow::float32());

    return div_node;
  }

  bool isNaNable(const std::string& label)
  {
    return label == "fTOFSignal" || label == "fBeta" || label == "fTRDSignal" || label == "fTRDPattern";
  }

  gandiva::NodePtr makeNaNableNode(gandiva::NodePtr field_node, const std::string& fieldName)
  {
    auto DetectorMapField = arrow::field("fDetectorMap", arrow::uint8());
    gandiva::NodePtr detector_map_node = gandiva::TreeExprBuilder::MakeField(DetectorMapField);
    gandiva::NodePtr zero_literal = gandiva::TreeExprBuilder::MakeLiteral(static_cast<uint8_t>(0U));
    gandiva::NodePtr quiet_nan_node = gandiva::TreeExprBuilder::MakeLiteral(std::numeric_limits<float>::quiet_NaN());

    gandiva::NodePtr final_node = field_node;

    if (fieldName == "fTOFSignal" || fieldName == "fBeta") {
      gandiva::NodePtr tof_mask = gandiva::TreeExprBuilder::MakeLiteral(o2::aod::track::TOF);
      gandiva::NodePtr has_tof_node = gandiva::TreeExprBuilder::MakeFunction("bitwise_and", {detector_map_node, tof_mask}, arrow::uint8());
      gandiva::NodePtr boolean_node = gandiva::TreeExprBuilder::MakeFunction("greater_than", {has_tof_node, zero_literal}, arrow::boolean());
      final_node = gandiva::TreeExprBuilder::MakeIf(boolean_node, final_node, quiet_nan_node, arrow::float32());
    } else if (fieldName == "fTRDSignal" || fieldName == "fTRDPattern") {
      gandiva::NodePtr trd_mask = gandiva::TreeExprBuilder::MakeLiteral(o2::aod::track::TRD);
      gandiva::NodePtr has_trd_node = gandiva::TreeExprBuilder::MakeFunction("bitwise_and", {detector_map_node, trd_mask}, arrow::uint8());
      gandiva::NodePtr boolean_node = gandiva::TreeExprBuilder::MakeFunction("greater_than", {has_trd_node, zero_literal}, arrow::boolean());
      final_node = gandiva::TreeExprBuilder::MakeIf(boolean_node, final_node, quiet_nan_node, arrow::float32());
    }

    return final_node;
  }

  void extendWithCastColumn(std::shared_ptr<arrow::Table>& table, std::shared_ptr<arrow::Field> field, std::shared_ptr<arrow::ChunkedArray> arr)
  {
    std::shared_ptr<arrow::ChunkedArray> float_arr = convertToFloat32(arr);

    auto result = table->AddColumn(table->num_columns(), field, float_arr);
    table = result.ValueOrDie();
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> getScaledChunkedArrays(std::shared_ptr<arrow::Table>& fullTable)
  {
    uint32_t n = mInputShapes[0][1];
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<uint32_t> returnIndices;
    std::vector<gandiva::ExpressionPtr> expressions;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> returnArrays;
    returnArrays.resize(n);

    std::shared_ptr<arrow::Table> table = fullTable;
    LOG(info) << "Schema: " << table->schema()->ToString();

    uint32_t returnIndex = 0U;
    uint32_t i = 0U;
    for (std::string& label : mTrainColumns) {
      auto scalingParamsEntry = mScalingParams.find(label);
      bool toNaNable = isNaNable(label);
      bool toScale = scalingParamsEntry != mScalingParams.end();

      LOG(info) << returnIndex << ". COLUMN NAME: " << label;

      auto arr = table->GetColumnByName(label);
      if (!toNaNable && !toScale) {
        returnArrays[returnIndex] = arr;
        returnIndex++;
        continue;
      }

      returnIndices.push_back(returnIndex);

      fields.push_back(arrow::field(label, arrow::float32()));

      if (arr->type() != arrow::float32()) {
        // FIXME: It seems that casting from uint8 to float32 haven't been supported yet in gandiva, so code below wouldn't work
        //    gandiva::NodePtr uncasted_field_node = gandiva::TreeExprBuilder::MakeField(fields[i]);
        //    field_node = gandiva::TreeExprBuilder::MakeFunction(o2::framework::expressions::upcastTo(arrow::Type::FLOAT), {field_node}, arrow::float32());
        // for this reason cast would be made using pure Apache Arrow functions

        extendWithCastColumn(table, fields[i], arr);
      }

      LOG(info) << i << ". SCALING COLUMN NAME: " << label;

      gandiva::NodePtr field_node = gandiva::TreeExprBuilder::MakeField(fields[i]);
      ;

      if (toScale) {
        field_node = makeScalingNode(field_node, scalingParamsEntry->second);
      }

      gandiva::NodePtr final_node = makeNaNableNode(field_node, label);

      std::shared_ptr<arrow::Field> field_result = arrow::field(label + "Scaled", arrow::float32());
      expressions.push_back(gandiva::TreeExprBuilder::MakeExpression(final_node, field_result));

      i++;
      returnIndex++;
    }

    n = i;

    std::shared_ptr<gandiva::Projector> projector;
    auto s = gandiva::Projector::Make(table->schema(), expressions, &projector);
    if (!s.ok()) {
      LOG(fatal) << "Cannot create projector: " << s.ToString();
    }

    arrow::TableBatchReader reader(*table);
    LOG(info) << "NUM ROWS: " << table->num_rows();
    std::shared_ptr<arrow::RecordBatch> batch;
    arrow::ArrayVector v;
    std::vector<arrow::ArrayVector> chunks;
    chunks.resize(n);

    while (true) {
      s = reader.ReadNext(&batch);
      if (!s.ok()) {
        LOG(fatal) << "Cannot read batches from source table to spawn: " << s.ToString();
      }
      if (batch == nullptr) {
        break;
      }
      try {
        s = projector->Evaluate(*batch, arrow::default_memory_pool(), &v);
        if (!s.ok()) {
          LOG(fatal) << "Cannot apply projector to source table, " << s.ToString();
        }
      } catch (std::exception& e) {
        LOG(fatal) << "Cannot apply projector to source table, exception caught: " << e.what();
      }

      for (auto i = 0U; i < n; ++i) {
        chunks[i].emplace_back(v.at(i));
      }
    }

    for (auto i = 0U; i < n; ++i) {
      returnArrays[returnIndices[i]] = std::make_shared<arrow::ChunkedArray>(chunks[i]);
    }

    return returnArrays;
  }

  template <typename T>
  std::vector<float> getModelOutputsGandiva(T& tracksTable)
  {
    // First rank of the expected model input is -1 which means that it is dynamic axis.
    // Axis is exported as dynamic to make it possible to run model inference with the batch of
    // tracks at once in the future (batch would need to have the same amount of quiet_NaNs in each row).
    // For now we hardcode 1.
    static constexpr int64_t batch_size = 1;
    auto input_shape = mInputShapes[0];
    input_shape[0] = batch_size;
    std::vector<float> outputs;

    auto tracksArrowTable = tracksTable.asArrowTable();
    auto arrays = getScaledChunkedArrays(tracksArrowTable);
    outputs.reserve(tracksTable.size());
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto it = tracksTable.begin();

    while (it != tracksTable.end()) {
      std::vector<float> input;
      input.reserve(arrays.size());
      for (uint32_t j = 0; j < arrays.size(); ++j) {
        // Get the first chunk
        auto scalar = arrays[j]->GetScalar(it.mRowIndex).ValueOrDie();
        if (scalar->type->id() == arrow::Type::FLOAT) {
          auto float_scalar = std::dynamic_pointer_cast<arrow::FloatScalar>(scalar);
          // Convert to float
          input.push_back(float_scalar->value);
        }
      }

      std::vector<Ort::Value> inputTensors;

#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
      inputTensors.emplace_back(Ort::Experimental::Value::CreateTensor<float>(input.data(), input.size(), input_shape));
#else
      inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), input_shape.data(), input_shape.size()));
#endif
      try {
#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
        auto outputTensors = mSession->Run(mInputNames, inputTensors, mOutputNames);
#else
        Ort::RunOptions runOptions;
        std::vector<const char*> inputNamesChar(mInputNames.size(), nullptr);
        std::transform(std::begin(mInputNames), std::end(mInputNames), std::begin(inputNamesChar),
                       [&](const std::string& str) { return str.c_str(); });

        std::vector<const char*> outputNamesChar(mOutputNames.size(), nullptr);
        std::transform(std::begin(mOutputNames), std::end(mOutputNames), std::begin(outputNamesChar),
                       [&](const std::string& str) { return str.c_str(); });
        auto outputTensors = mSession->Run(runOptions, inputNamesChar.data(), inputTensors.data(), inputTensors.size(), outputNamesChar.data(), outputNamesChar.size());
#endif

        // Double-check the dimensions of the output tensors
        // The number of output tensors is equal to the number of output nodes specified in the Run() call
        assert(outputTensors.size() == mOutputNames.size() && outputTensors[0].IsTensor());
        LOG(debug) << "output tensor shape: " << printShape(outputTensors[0].GetTensorTypeAndShapeInfo().GetShape());

        const float* output_value = outputTensors[0].GetTensorData<float>();
        float certainty = *output_value;
        outputs.push_back(certainty);
      } catch (const Ort::Exception& exception) {
        LOG(error) << "Error running model inference: " << exception.what();
      }

      it++;
    }

    return outputs; // unreachable code
  }

  template <typename Tb, typename T>
  float applyModel(const Tb& table, const T& track)
  {
    return getModelOutput(table, track);
  }

  template <typename Tb, typename T>
  bool applyModelBoolean(const Tb& table, const T& track)
  {
    return getModelOutput(table, track) >= mMinCertainty;
  }

  template <typename Tb>
  std::vector<float> batchApplyModel(const Tb& table)
  {
    std::vector<float> outputs;
    outputs.reserve(table.size());

    for (const auto& track : table) {
      outputs.push_back(applyModel(table, track));
    }

    return outputs;
  }

  template <typename Tb>
  std::vector<bool> batchApplyModelBoolean(const Tb& table)
  {
    std::vector<bool> outputs;
    outputs.reserve(table.size());

    for (const auto& track : table) {
      outputs.push_back(applyModelBoolean(table, track));
    }

    return outputs;
  }

  int mPid;
  double mMinCertainty;

 private:
  void getModelPaths(std::string const& path, std::string& modelDir, std::string& modelFile, std::string& modelPath, int pid, std::string const& ext)
  {
    modelDir = path;
    modelFile = "attention_model_";

    if (pid < 0) {
      modelFile += "0" + std::to_string(-pid);
    } else {
      modelFile += std::to_string(pid);
    }

    modelFile += ext;
    modelPath = modelDir + "/" + modelFile;
  }

  void downloadFromCCDB(o2::ccdb::CcdbApi& ccdbApi, std::string const& ccdbFile, uint64_t timestamp, std::string const& localDir, std::string const& localFile)
  {
    std::map<std::string, std::string> metadata;
    bool retrieveSuccess = ccdbApi.retrieveBlob(ccdbFile, localDir, metadata, timestamp, false, localFile);
    if (retrieveSuccess) {
      std::map<std::string, std::string> headers = ccdbApi.retrieveHeaders(ccdbFile, metadata, timestamp);
      LOG(info) << "Network file downloaded from: " << ccdbFile << " to: " << localDir << "/" << localFile;
    } else {
      LOG(fatal) << "Error encountered while fetching/loading the network from CCDB! Maybe the network doesn't exist yet for this run number/timestamp?";
    }
  }

  void loadInputFiles(std::string const& localPath, std::string const& ccdbPath, bool useCCDB, o2::ccdb::CcdbApi& ccdbApi, uint64_t timestamp, int pid, std::string& modelPath)
  {
    rapidjson::Document trainColumnsDoc;
    rapidjson::Document scalingParamsDoc;

    std::string localDir, localModelFile;
    std::string trainColumnsFile = "columns_for_training";
    std::string scalingParamsFile = "scaling_params";
    getModelPaths(localPath, localDir, localModelFile, modelPath, pid, ".onnx");
    std::string localTrainColumnsPath = localDir + "/" + trainColumnsFile + ".json";
    std::string localScalingParamsPath = localDir + "/" + scalingParamsFile + ".json";

    if (useCCDB) {
      std::string ccdbDir, ccdbModelFile, ccdbModelPath;
      getModelPaths(ccdbPath, ccdbDir, ccdbModelFile, ccdbModelPath, pid, "");
      std::string ccdbTrainColumnsPath = ccdbDir + "/" + trainColumnsFile;
      std::string ccdbScalingParamsPath = ccdbDir + "/" + scalingParamsFile;
      downloadFromCCDB(ccdbApi, ccdbModelPath, timestamp, localDir, localModelFile);
      downloadFromCCDB(ccdbApi, ccdbTrainColumnsPath, timestamp, localDir, "columns_for_training.json");
      downloadFromCCDB(ccdbApi, ccdbScalingParamsPath, timestamp, localDir, "scaling_params.json");
    }

    LOG(info) << "Using configuration files: " << localTrainColumnsPath << ", " << localScalingParamsPath;
    if (readJsonFile(localTrainColumnsPath, trainColumnsDoc)) {
      for (auto& param : trainColumnsDoc["columns_for_training"].GetArray()) {
        mTrainColumns.emplace_back(param.GetString());
      }
    }
    if (readJsonFile(localScalingParamsPath, scalingParamsDoc)) {
      for (auto& param : scalingParamsDoc["data"].GetArray()) {
        mScalingParams[param[0].GetString()] = std::make_pair(param[1].GetFloat(), param[2].GetFloat());
      }
    }
  }

  template <typename... P1, typename ...P2>
  static constexpr bool is_equal_size(o2::framework::pack<P1...>, o2::framework::pack<P2...>) {
    if constexpr (sizeof...(P1) == sizeof...(P2)) {
      return true;
    }

    return false;
  }

  static float scale(float value, const std::pair<float, float>& scalingParams) {
    return (value - scalingParams.first) / scalingParams.second;
  }

  template <typename T, typename C>
  typename C::type getPersistentValue(arrow::Table* table, const T& rowIterator)
  {
    auto colIterator = static_cast<C>(rowIterator).getIterator();
    uint64_t ci = colIterator.mCurrentChunk;
    uint64_t ai = *(colIterator.mCurrentPos) - colIterator.mFirstIndex;

    return std::static_pointer_cast<o2::soa::arrow_array_for_t<typename C::type>>(o2::soa::getIndexFromLabel(table, C::columnLabel())->chunk(ci))->raw_values()[ai];
  }

  template <typename T, typename Tb,  typename... C>
  std::vector<float> getValues(o2::framework::pack<C...>, const T& track, const Tb& table)
  {
    auto arrowTable = table.asArrowTable();
    std::vector<float> output;
    output.reserve(mTrainColumns.size());
    for (const std::string& columnLabel : mTrainColumns) {
      std::optional<std::pair<float, float>> scalingParams = std::nullopt;

      auto scalingParamsEntry = mScalingParams.find(columnLabel);
      if(scalingParamsEntry != mScalingParams.end()) {
        scalingParams = scalingParamsEntry->second;
      }

      bool isInPLimitTrd = inPLimit(track, mPLimits[kTPCTOFTRD]);
      bool isInPLimitTof = inPLimit(track, mPLimits[kTPCTOF]);
      bool isTrdMissing = trdMissing(track);
      bool isTofMissing = tofMissing(track);

      ([&]() {
        if constexpr (o2::soa::is_dynamic_v<C> && std::is_arithmetic_v<typename C::type>) {
          // check if bindings have the same size as lambda parameters (getter do not have additional parameters)
          if constexpr (is_equal_size(typename C::bindings_t{}, typename C::callable_t::args{})) {
            std::string label = C::columnLabel();

            // dynamic columns do not have "f" prefix in columnLabel() return string
            if (std::strcmp(&columnLabel[1], label.data())) {
              return;
            }

            float value = static_cast<float>(track.template getDynamicColumn<C>());

            if(scalingParams) {
              value = scale(value, scalingParams.value());
            }

            output.push_back(value);
          }
        } else if constexpr (o2::soa::is_persistent_v<C> && !o2::soa::is_index_column_v<C> && std::is_arithmetic_v<typename C::type> && !std::is_same_v<typename C::type, bool>) {
          std::string label = C::columnLabel();

          if (columnLabel != label) {
            return;
          }

          if constexpr (std::is_same_v<C, o2::aod::track::TRDSignal> || std::is_same_v<C, o2::aod::track::TRDPattern>) {
            if(isTrdMissing || !isInPLimitTrd) {
              output.push_back(std::numeric_limits<float>::quiet_NaN());
              return;
            }
          } else if constexpr (std::is_same_v<C, o2::aod::pidtofsignal::TOFSignal> || std::is_same_v<C, o2::aod::pidtofbeta::Beta>) {
            if(isTofMissing || !isInPLimitTof) {
              output.push_back(std::numeric_limits<float>::quiet_NaN());
              return;
            }
          }

          float value = static_cast<float>(getPersistentValue<T, C>(arrowTable.get(), track));

          if(scalingParams) {
            value = scale(value, scalingParams.value());
          }

          output.push_back(value);
        }
      }(),
      ...);
    }

    return output;
  }

  template <typename Tb, typename T>
  float getModelOutput(const Tb& table, const T& track)
  {
    // First rank of the expected model input is -1 which means that it is dynamic axis.
    // Axis is exported as dynamic to make it possible to run model inference with the batch of
    // tracks at once in the future (batch would need to have the same amount of quiet_NaNs in each row).
    // For now we hardcode 1.
    static constexpr int64_t batch_size = 1;
    auto input_shape = mInputShapes[0];
    input_shape[0] = batch_size;

    std::vector<float> inputTensorValues = getValues(typename Tb::table_t::columns{}, track, table);
    std::vector<Ort::Value> inputTensors;

#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
    inputTensors.emplace_back(Ort::Experimental::Value::CreateTensor<float>(inputTensorValues.data(), inputTensorValues.size(), input_shape));
#else
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(mem_info, inputTensorValues.data(), inputTensorValues.size(), input_shape.data(), input_shape.size()));
#endif

    // Double-check the dimensions of the input tensor
    assert(inputTensors[0].IsTensor() &&
           inputTensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
    LOG(debug) << "input tensor shape: " << printShape(inputTensors[0].GetTensorTypeAndShapeInfo().GetShape());

    try {
#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
      auto outputTensors = mSession->Run(mInputNames, inputTensors, mOutputNames);
#else
      Ort::RunOptions runOptions;
      std::vector<const char*> inputNamesChar(mInputNames.size(), nullptr);
      std::transform(std::begin(mInputNames), std::end(mInputNames), std::begin(inputNamesChar),
                     [&](const std::string& str) { return str.c_str(); });

      std::vector<const char*> outputNamesChar(mOutputNames.size(), nullptr);
      std::transform(std::begin(mOutputNames), std::end(mOutputNames), std::begin(outputNamesChar),
                     [&](const std::string& str) { return str.c_str(); });
      auto outputTensors = mSession->Run(runOptions, inputNamesChar.data(), inputTensors.data(), inputTensors.size(), outputNamesChar.data(), outputNamesChar.size());
#endif

      // Double-check the dimensions of the output tensors
      // The number of output tensors is equal to the number of output nodes specified in the Run() call
      assert(outputTensors.size() == mOutputNames.size() && outputTensors[0].IsTensor());
      LOG(debug) << "output tensor shape: " << printShape(outputTensors[0].GetTensorTypeAndShapeInfo().GetShape());

      const float* output_value = outputTensors[0].GetTensorData<float>();
      float certainty = *output_value;
      return certainty;
    } catch (const Ort::Exception& exception) {
      LOG(error) << "Error running model inference: " << exception.what();
    }
    return false; // unreachable code
  }

  // Pretty prints a shape dimension vector
  std::string printShape(const std::vector<int64_t>& v)
  {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
      ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
  }

  std::vector<std::string> mTrainColumns;
  std::map<std::string, std::pair<float, float>> mScalingParams;

  std::shared_ptr<Ort::Env> mEnv = nullptr;
  // No empty constructors for Session, we need a pointer
#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
  std::shared_ptr<Ort::Experimental::Session> mSession = nullptr;
#else
  std::shared_ptr<Ort::Session> mSession = nullptr;
#endif

  std::shared_ptr<Ort::IoBinding> mBinding = nullptr;

  std::vector<double> mPLimits;
  std::vector<std::string> mInputNames;
  std::vector<std::vector<int64_t>> mInputShapes;
  std::vector<std::string> mOutputNames;
  std::vector<std::vector<int64_t>> mOutputShapes;
};

#endif // TOOLS_PIDML_PIDONNXMODEL_H_
