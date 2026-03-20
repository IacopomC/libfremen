#ifndef LIBFREMEN_FREMEN_MODEL_HPP
#define LIBFREMEN_FREMEN_MODEL_HPP

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace libfremen {

struct ChangePointConfig {
  float outlierThreshold = 0.35f;
  int outlierWindowSize = 32;
  float triggerRatio = 0.60f;
  int minMeasurementsBeforeDetection = 24;
};

struct AdaptiveBasisConfig {
  int maxOrder = 8;
  int refreshInterval = 8;
};

struct CellDebugStats {
  float gain = 0.5f;
  int measurements = 0;
  int64_t firstTime = -1;
  int64_t lastTime = -1;
  float outlierRatio = 0.0f;
  int changePointCount = 0;
  std::vector<float> activePeriods;
};

struct AddFrameResult {
  int updatedStates = 0;
  int ignoredUnknown = 0;
  int ignoredInvalid = 0;
  int changePointsTriggered = 0;
  std::vector<int> changedIndices;
};

class FremenCellModel {
 public:
  FremenCellModel();
  FremenCellModel(const std::vector<float>& candidatePeriods,
                  const ChangePointConfig& changePointConfig,
                  const AdaptiveBasisConfig& adaptiveConfig);

  bool addObservation(uint32_t timestamp, float state, bool* changePointTriggered = nullptr);
  float predict(uint32_t timestamp, int order) const;
  float entropy(uint32_t timestamp, int order) const;
  int evaluate(uint32_t timestamp, float state, int maxOrder, float* errors) const;

  void reset();
  CellDebugStats debugStats() const;

 private:
  struct Harmonic {
    float period = 1.0f;
    float realStates = 0.0f;
    float imagStates = 0.0f;
    float realBalance = 0.0f;
    float imagBalance = 0.0f;
    float amplitude = 0.0f;
    float phase = 0.0f;
  };

  void recomputeAmplitudesAndBasis();
  void registerOutlier(bool outlier);
  float currentOutlierRatio() const;

  std::vector<Harmonic> harmonics_;
  std::vector<int> activeIndices_;
  ChangePointConfig changePointConfig_;
  AdaptiveBasisConfig adaptiveConfig_;

  float gain_ = 0.5f;
  int measurements_ = 0;
  int64_t firstTime_ = -1;
  int64_t lastTime_ = -1;
  int changePointCount_ = 0;
  int addsSinceRefresh_ = 0;

  std::deque<int> outlierWindow_;
  int outlierCount_ = 0;
};

class FremenArrayModel {
 public:
  FremenArrayModel();

  void configure(const std::vector<float>& candidatePeriods,
                 const ChangePointConfig& changePointConfig,
                 const AdaptiveBasisConfig& adaptiveConfig);

  bool initialize(std::size_t stateCount);
  void reset();
  bool isInitialized() const;
  std::size_t size() const;

  AddFrameResult addFrame(uint32_t timestamp, const std::vector<int8_t>& states);
  int predictFrame(uint32_t timestamp, int order, std::vector<float>* probabilities) const;
  int entropyFrame(uint32_t timestamp, int order, std::vector<float>* entropies) const;
  int evaluateFrame(uint32_t timestamp,
                    const std::vector<int8_t>& states,
                    int maxOrder,
                    std::vector<float>* errors,
                    float* meanOutlierRatio = nullptr,
                    int* bestOrder = nullptr) const;

  int totalChangePointCount() const;
  float meanOutlierRatio() const;
  std::vector<float> representativeActivePeriods() const;

 private:
  std::vector<float> candidatePeriods_;
  ChangePointConfig changePointConfig_;
  AdaptiveBasisConfig adaptiveConfig_;
  std::vector<FremenCellModel> cells_;
};

std::vector<float> defaultCandidatePeriods();

}  // namespace libfremen

#endif