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

/// \file mlModelGenHists
/// \brief Generate momentum TH1Fs for accepted mcParticles by ML model and for MC mcParticles.
///
/// \author Michał Olędzki <mioledzk@cern.ch>
/// \author Marek Mytkowski <mmytkows@cern.ch>

#include <Framework/AnalysisDataModel.h>
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "CCDB/CcdbApi.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/PIDResponse.h"
#include "Tools/PIDML/pidOnnxModel.h"

#include <string>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

#define EPS 0.00001f
#define ETA_CUT 0.8f
#define TOF_ACCEPTANCE_THRESHOLD (-999.0f)
#define TRD_ACCEPTANCE_THRESHOLD (0.0f)

// namespace o2::aod
// {
//   namespace mlpidresult
//   {
//     DECLARE_SOA_INDEX_COLUMN(Track, track);       //! Track index
//     DECLARE_SOA_COLUMN(Pid, pid, int);            //! Pid to be tested by the model
//     DECLARE_SOA_COLUMN(Accepted, accepted, bool); //! Whether the model accepted mcPart to be of given kind
//   } // namespace mlpidresult
//   DECLARE_SOA_TABLE(MlPidResults, "AOD", "MLPIDRESULTS", o2::soa::Index<>, mlpidresult::TrackId, mlpidresult::Pid, mlpidresult::Accepted);
// } // namespace o2::aod

struct MlModelGenHists {
  HistogramRegistry histos{"histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  PidONNXModel pidModel; // One instance per model, e.g., one per each pid to predict
  Configurable<uint32_t> cfgDetector{"detector", kTPCTOFTRD, "What detectors to use: 0: TPC only, 1: TPC + TOF, 2: TPC + TOF + TRD"};
  Configurable<int> cfgPid{"pid", 211, "PID to predict"};
  Configurable<double> cfgCertainty{"certainty", 0.5, "Min certainty of the model to accept given mcPart to be of given kind"};

  Configurable<std::string> cfgPathCCDB{"ccdb-path", "Users/m/mkabus/PIDML", "base path to the CCDB directory with ONNX models"};
  Configurable<std::string> cfgCCDBURL{"ccdb-url", "http://alice-ccdb.cern.ch", "URL of the CCDB repository"};
  Configurable<bool> cfgUseCCDB{"useCCDB", true, "Whether to autofetch ML model from CCDB. If false, local file will be used."};
  Configurable<std::string> cfgPathLocal{"local-path", "/home/mkabus/PIDML", "base path to the local directory with ONNX models"};

  Configurable<bool> cfgUseFixedTimestamp{"use-fixed-timestamp", false, "Whether to use fixed timestamp from configurable instead of timestamp calculated from the data"};
  Configurable<uint64_t> cfgTimestamp{"timestamp", 1524176895000, "Hardcoded timestamp for tests"};


  o2::ccdb::CcdbApi ccdbApi;
  int currentRunNumber = -1;

  // Produces<o2::aod::MlPidResults> pidMLResults;

  Filter trackFilter = requireGlobalTrackInFilter() &&
    (nabs(aod::pidtofbeta::beta - TOF_ACCEPTANCE_THRESHOLD) > EPS) &&
    (nabs(aod::pidtofsignal::tofSignal - TOF_ACCEPTANCE_THRESHOLD) > EPS) &&
    (nabs(aod::track::trdSignal - TRD_ACCEPTANCE_THRESHOLD) > EPS);
  // Minimum table requirements for sample model:
  // TPC signal (FullTracks), TOF signal (TOFSignal), TOF beta (pidTOFbeta), dcaXY and dcaZ (TracksDCA)
  // Filter on isGlobalTrack (TracksSelection)
  using BigTracks = soa::Filtered<soa::Join<aod::FullTracks, aod::TracksDCA, aod::pidTOFbeta, aod::TrackSelection, aod::TOFSignal, aod::McTrackLabels>>;

  void init(InitContext const&)
  {
    if (cfgUseCCDB) {
      ccdbApi.init(cfgCCDBURL);
    } else {
      pidModel = PidONNXModel(cfgPathLocal.value, cfgPathCCDB.value, cfgUseCCDB.value, ccdbApi, -1, cfgPid.value, static_cast<PidMLDetector>(cfgDetector.value), cfgCertainty.value);
    }

    const AxisSpec axisPt{50, 0, 3.1, "pt"};

    histos.add("hPtMCPositive", "hPtMCPositive", kTH1F, {axisPt});
    histos.add("hPtMCTracked", "hPtMCTracked", kTH1F, {axisPt});
    histos.add("hPtMLPositive", "hPtMLPositive", kTH1F, {axisPt});
    histos.add("hPtMLTruePositive", "hPtMLTruePositive", kTH1F, {axisPt});
  }

  void process(aod::Collisions const& collisions, BigTracks const& tracks, aod::BCsWithTimestamps const&, aod::McParticles const& mcParticles)
  {
    auto bc = collisions.iteratorAt(0).bc_as<aod::BCsWithTimestamps>();
    if (cfgUseCCDB && bc.runNumber() != currentRunNumber) {
      uint64_t timestamp = cfgUseFixedTimestamp ? cfgTimestamp.value : bc.timestamp();
      pidModel = PidONNXModel(cfgPathLocal.value, cfgPathCCDB.value, cfgUseCCDB.value, ccdbApi, timestamp, cfgPid.value, static_cast<PidMLDetector>(cfgDetector.value), cfgCertainty.value);
    }

    for (auto& mcPart : mcParticles) {
      // eta cut is included in requireGlobalTrackInFilter() so we cut it only here
      if(mcPart.isPhysicalPrimary() && TMath::Abs(mcPart.eta()) < ETA_CUT && mcPart.pdgCode() == pidModel.mPid) {
        histos.fill(HIST("hPtMCPositive"), mcPart.pt());
      }
    }

    for (auto& track : tracks) {
      if(track.has_mcParticle()) {
        auto mcPart = track.mcParticle();
        if(mcPart.isPhysicalPrimary()) {
          bool accepted = pidModel.applyModelBoolean(track);
          LOGF(info, "collision id: %d track id: %d accepted: %d p: %.3f; x: %.3f, y: %.3f, z: %.3f",
              track.collisionId(), track.index(), accepted, track.p(), track.x(), track.y(), track.z());

          if(mcPart.pdgCode() == pidModel.mPid) {
            histos.fill(HIST("hPtMCTracked"), mcPart.pt());
          }

          if(accepted) {
            if(mcPart.pdgCode() == pidModel.mPid) {
              histos.fill(HIST("hPtMLTruePositive"), mcPart.pt());
            }
            histos.fill(HIST("hPtMLPositive"), mcPart.pt());
          }
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<MlModelGenHists>(cfgc)};
}
