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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <algorithm>
#include <map>
#include <utility>
#include <memory>
#include <vector>

#include "onnxruntime/core/session/experimental_onnxruntime_cxx_api.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

#include "CCDB/CcdbApi.h"

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
  PidONNXModel(std::string& localPath, std::string& ccdbPath, bool useCCDB, o2::ccdb::CcdbApi& ccdbApi, uint64_t timestamp, int pid, double minCertainty) : mUseCCDB(useCCDB), mPid(pid), mMinCertainty(minCertainty)
  {
    std::string modelFile;
    loadInputFiles(localPath, ccdbPath, useCCDB, ccdbApi, timestamp, pid, modelFile);

    Ort::SessionOptions sessionOptions;
    mEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_VERBOSE, "pid-onnx-inferer");
    LOG(info) << "Loading ONNX model from file: " << modelFile;
    mSession.reset(new Ort::Experimental::Session{*mEnv, modelFile, sessionOptions});
    LOG(info) << "ONNX model loaded";

    mInputNames = mSession->GetInputNames();
    mInputShapes = mSession->GetInputShapes();
    mOutputNames = mSession->GetOutputNames();
    mOutputShapes = mSession->GetOutputShapes();

    LOG(info) << "Input Node Name/Shape (" << mInputNames.size() << "):";
    for (size_t i = 0; i < mInputNames.size(); i++) {
      LOG(info) << "\t" << mInputNames[i] << " : " << printShape(mInputShapes[i]);
    }

    LOG(info) << "Output Node Name/Shape (" << mOutputNames.size() << "):";
    for (size_t i = 0; i < mOutputNames.size(); i++) {
      LOG(info) << "\t" << mOutputNames[i] << " : " << printShape(mOutputShapes[i]);
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

  template <typename T>
  float applyModel(const T& track)
  {
    return getModelOutput(track);
  }

  template <typename T>
  bool applyModelBoolean(const T& track)
  {
    return getModelOutput(track) >= mMinCertainty;
  }

  bool mUseCCDB;
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

  static constexpr float kTOFMissingSignal = -999.0f;
  static constexpr float kTRDMissingSignal = 0.0f;
  static constexpr float kEpsilon = 1e-10f;

  template <typename T>
  std::vector<std::vector<float>> createInputsSingle(const T& track)
  {
    // TODO: Hardcoded for now. Planning to implement RowView extension to get runtime access to selected columns
    // sign is short, trackType and tpcNClsShared uint8_t

    std::vector<std::vector<float>> inputValues;

    float scaledTPCSignal = (track.tpcSignal() - mScalingParams.at("fTPCSignal").first) / mScalingParams.at("fTPCSignal").second;

    inputValues.emplace_back(constructFeatureVector(0, scaledTPCSignal));


    if (TMath::Abs(track.trdSignal() - kTRDMissingSignal) >= kEpsilon) {
      float scaledTRDSignal = (track.trdSignal() - mScalingParams.at("fTRDSignal").first) / mScalingParams.at("fTRDSignal").second;
      // inputValues.push_back(scaledTRDSignal);
      // inputValues.push_back(track.trdPattern());
      std::cout << "WITH_TRD" << std::endl;
      inputValues.emplace_back(constructFeatureVector(1, scaledTRDSignal));
      inputValues.emplace_back(constructFeatureVector(2, track.trdPattern()));
    }
    // else {
    //   // inputValues.push_back(std::numeric_limits<float>::quiet_NaN());
    //   // inputValues.push_back(std::numeric_limits<float>::quiet_NaN());
    //   inputValues.push_back(-99.0f);
    //   inputValues.push_back(-99.0f);
    //   std::cout << "MISSING_TRD" << std::endl;
    // }

    if (TMath::Abs(track.tofSignal() - kTOFMissingSignal) >= kEpsilon) {
      float scaledTOFSignal = (track.tofSignal() - mScalingParams.at("fTOFSignal").first) / mScalingParams.at("fTOFSignal").second;
      float scaledBeta = (track.beta() - mScalingParams.at("fBeta").first) / mScalingParams.at("fBeta").second;
      // inputValues.push_back(scaledTOFSignal);
      // inputValues.push_back(scaledBeta);
      std::cout << "WITH_TOF" << std::endl;
      inputValues.emplace_back(constructFeatureVector(3, scaledTOFSignal));
      inputValues.emplace_back(constructFeatureVector(4, scaledBeta));
    }
    // else {
    //   // inputValues.push_back(std::numeric_limits<float>::quiet_NaN());
    //   // inputValues.push_back(std::numeric_limits<float>::quiet_NaN());
    //   inputValues.push_back(-99.0f);
    //   inputValues.push_back(-99.0f);
    //   std::cout << "MISSING_TOF" << std::endl;
    // }

    float scaledX = (track.x() - mScalingParams.at("fX").first) / mScalingParams.at("fX").second;
    float scaledY = (track.y() - mScalingParams.at("fY").first) / mScalingParams.at("fY").second;
    float scaledZ = (track.z() - mScalingParams.at("fZ").first) / mScalingParams.at("fZ").second;
    float scaledAlpha = (track.alpha() - mScalingParams.at("fAlpha").first) / mScalingParams.at("fAlpha").second;
    float scaledTPCNClsShared = (static_cast<float>(track.tpcNClsShared()) - mScalingParams.at("fTPCNClsShared").first) / mScalingParams.at("fTPCNClsShared").second;
    float scaledDcaXY = (track.dcaXY() - mScalingParams.at("fDcaXY").first) / mScalingParams.at("fDcaXY").second;
    float scaledDcaZ = (track.dcaZ() - mScalingParams.at("fDcaZ").first) / mScalingParams.at("fDcaZ").second;

    // inputValues.insert(inputValues.end(), {track.p(), track.pt(), track.px(), track.py(), track.pz(), static_cast<float>(track.sign()), scaledX, scaledY, scaledZ, scaledAlpha, static_cast<float>(track.trackType()), scaledTPCNClsShared, scaledDcaXY, scaledDcaZ});

    inputValues.emplace_back(constructFeatureVector(5, track.p()));
    inputValues.emplace_back(constructFeatureVector(6, track.pt()));
    inputValues.emplace_back(constructFeatureVector(7, track.px()));
    inputValues.emplace_back(constructFeatureVector(8, track.py()));
    inputValues.emplace_back(constructFeatureVector(9, track.pz()));
    inputValues.emplace_back(constructFeatureVector(10, static_cast<float>(track.sign())));
    inputValues.emplace_back(constructFeatureVector(11, scaledX));
    inputValues.emplace_back(constructFeatureVector(12, scaledY));
    inputValues.emplace_back(constructFeatureVector(13, scaledZ));
    inputValues.emplace_back(constructFeatureVector(14, scaledAlpha));
    inputValues.emplace_back(constructFeatureVector(15, static_cast<float>(track.trackType())));
    inputValues.emplace_back(constructFeatureVector(16, scaledTPCNClsShared));
    inputValues.emplace_back(constructFeatureVector(17, scaledDcaXY));
    inputValues.emplace_back(constructFeatureVector(18, scaledDcaZ));

    return inputValues;
  }

  static constexpr std::size_t featureCount = 19;
  // TODO: performance considerations, maybe some move/reference returning or vector of vectors reference as parameter and emplace_back here.
  std::vector<float> constructFeatureVector(int32_t i, float value) {
    std::vector<float> retVec(featureCount + 1, 0.0f);
    retVec[i] = 1.0f;
    retVec[featureCount] = value;
    return retVec;
  }

  // FIXME: Temporary solution, new networks will have sigmoid layer added
  float sigmoid(float x)
  {
    float value = std::max(-100.0f, std::min(100.0f, x));
    return 1.0f / (1.0f + std::exp(-value));
  }

  template <typename T>
  float getModelOutput(const T& track)
  {
    auto input_shape = mInputShapes[0];
    std::vector<std::vector<float>> inputTensorValues = createInputsSingle(track);
    std::vector<Ort::Value> inputTensors;

    std::vector<float> oneDimArray;
    for(auto& feature : inputTensorValues) {
      for(float f : feature) {
        oneDimArray.push_back(f);
      }
    }

    std::vector<int64_t> input_shape_vector;
    input_shape_vector.push_back(1);
    input_shape_vector.push_back(inputTensorValues.size());
    input_shape_vector.push_back(featureCount + 1);

    Ort::MemoryInfo memory_info{ nullptr };     // Used to allocate memory for input
    try {
      memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    }
    catch (Ort::Exception& oe) {
      std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
      return -1;
    }

    inputTensors.emplace_back(Ort::Experimental::Value::CreateTensor<float>((float*) oneDimArray.data(), oneDimArray.size(), input_shape_vector));

    // Double-check the dimensions of the input tensor
    assert(inputTensors[0].IsTensor() &&
           inputTensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
    LOG(info) << "input tensor shape: " << printShape(inputTensors[0].GetTensorTypeAndShapeInfo().GetShape());

    try {
      auto outputTensors = mSession->Run(mInputNames, inputTensors, mOutputNames);

      // Double-check the dimensions of the output tensors
      // The number of output tensors is equal to the number of output nodes specified in the Run() call
      assert(outputTensors.size() == mOutputNames.size() && outputTensors[0].IsTensor());
      LOG(info) << "output tensor shape: " << printShape(outputTensors[0].GetTensorTypeAndShapeInfo().GetShape());

      const float* output_value = outputTensors[0].GetTensorData<float>();
      float certainty = sigmoid(*output_value); // FIXME: Temporary, sigmoid will be added as network layer
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
  std::shared_ptr<Ort::Experimental::Session> mSession = nullptr;

  std::vector<std::string> mInputNames;
  std::vector<std::vector<int64_t>> mInputShapes;
  std::vector<std::string> mOutputNames;
  std::vector<std::vector<int64_t>> mOutputShapes;
};

#endif // TOOLS_PIDML_PIDONNXMODEL_H_
