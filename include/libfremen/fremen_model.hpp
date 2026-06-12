// libfremen: a small, ROS-agnostic implementation of FreMEn (Frequency Map
// Enhancement) for long-term modeling of periodic binary/probabilistic states.
//
// The idea: a state that varies over time (e.g. "is this cell occupied?") is
// represented by its running mean plus a handful of Fourier harmonics at fixed
// candidate periods (a day, half a day, a week, ...). Coefficients are updated
// incrementally per observation, so the model needs O(#periods) memory and no
// stored history. From those coefficients it can predict the probability of the
// state at any (past or future) time, score its own uncertainty (entropy), and
// detect when the underlying pattern has changed (change-point -> reset).
//
// Two levels:
//   * FremenCellModel  - one independent state.
//   * FremenArrayModel - a fixed-size array of cells updated frame-by-frame.
#ifndef LIBFREMEN_FREMEN_MODEL_HPP
#define LIBFREMEN_FREMEN_MODEL_HPP

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace libfremen {

// Controls when a cell decides its periodic pattern has changed and resets.
struct ChangePointConfig {
  float outlierThreshold = 0.35f;       // |observation - prediction| above this counts as an outlier
  int outlierWindowSize = 32;           // number of recent observations tracked for the outlier ratio
  float triggerRatio = 0.60f;           // reset once this fraction of the window are outliers
  int minMeasurementsBeforeDetection = 24;  // don't detect change points until this many observations seen
};

// Controls the adaptive spectral basis (which harmonics are kept "active").
struct AdaptiveBasisConfig {
  int maxOrder = 8;        // keep at most this many strongest harmonics for prediction
  int refreshInterval = 8;  // re-rank/refresh the active harmonics every N observations
};

// Read-only snapshot of a cell's internal state, for inspection/diagnostics.
struct CellDebugStats {
  float gain = 0.5f;            // running mean of the state (the DC / order-0 term)
  int measurements = 0;        // number of observations absorbed since the last reset
  int64_t firstTime = -1;      // timestamp of the first observation (-1 if none)
  int64_t lastTime = -1;       // timestamp of the most recent observation (-1 if none)
  float outlierRatio = 0.0f;   // current fraction of outliers in the sliding window
  int changePointCount = 0;    // how many change points have fired over the cell's lifetime
  std::vector<float> activePeriods;  // periods (seconds) of the currently active harmonics
};

// Per-frame outcome of FremenArrayModel::addFrame.
struct AddFrameResult {
  int updatedStates = 0;          // cells that absorbed a valid observation
  int ignoredUnknown = 0;         // entries flagged unknown (state == -1)
  int ignoredInvalid = 0;         // entries out of the valid range
  int changePointsTriggered = 0;  // cells that reset this frame
  std::vector<int> changedIndices;  // indices of the cells that reset
};

// Models a single state's periodic behaviour from a stream of timestamped
// observations in [0, 1]. All updates are incremental (no history is stored).
class FremenCellModel {
 public:
  // Uses defaultCandidatePeriods() and default configs.
  FremenCellModel();
  FremenCellModel(const std::vector<float>& candidatePeriods,
                  const ChangePointConfig& changePointConfig,
                  const AdaptiveBasisConfig& adaptiveConfig);

  // Absorb one observation (state in [0, 1]) at the given timestamp (seconds).
  // Observations must arrive in strictly increasing time order; out-of-order or
  // out-of-range values are rejected (returns false). If a change point fires
  // while absorbing this observation, *changePointTriggered is set to true.
  bool addObservation(uint32_t timestamp, float state, bool* changePointTriggered = nullptr);

  // Predicted probability of the state at `timestamp`, using up to `order`
  // active harmonics on top of the mean. Result is clamped to [0, 1].
  float predict(uint32_t timestamp, int order) const;

  // Binary (Shannon) entropy, in bits, of the prediction at `timestamp`.
  float entropy(uint32_t timestamp, int order) const;

  // Fill errors[0..maxOrder] with |state - predict(timestamp, order)| for each
  // order and return the order with the smallest error. `errors` must hold at
  // least maxOrder + 1 floats. Returns -1 on invalid input.
  int evaluate(uint32_t timestamp, float state, int maxOrder, float* errors) const;

  // Forget all observations and zero the spectrum (config is kept).
  void reset();
  CellDebugStats debugStats() const;

 private:
  // One Fourier component at a fixed period. `*States` accumulate the raw
  // signal projection; `*Balance` accumulate the mean-weighted reference, so
  // (States - Balance) is the deviation from a constant signal. amplitude/phase
  // are the polar form derived from that deviation.
  struct Harmonic {
    float period = 1.0f;
    float realStates = 0.0f;
    float imagStates = 0.0f;
    float realBalance = 0.0f;
    float imagBalance = 0.0f;
    float amplitude = 0.0f;
    float phase = 0.0f;
  };

  // Recompute every harmonic's amplitude/phase and re-select the strongest
  // `maxOrder` of them into activeIndices_ (the adaptive basis).
  void recomputeAmplitudesAndBasis();
  void registerOutlier(bool outlier);     // push a hit/miss into the sliding window
  float currentOutlierRatio() const;      // outliers / window size

  std::vector<Harmonic> harmonics_;    // one per candidate period
  std::vector<int> activeIndices_;     // indices into harmonics_, strongest first
  ChangePointConfig changePointConfig_;
  AdaptiveBasisConfig adaptiveConfig_;

  float gain_ = 0.5f;          // running mean (order-0 term)
  int measurements_ = 0;
  int64_t firstTime_ = -1;
  int64_t lastTime_ = -1;
  int changePointCount_ = 0;
  int addsSinceRefresh_ = 0;   // observations since the last basis refresh

  std::deque<int> outlierWindow_;  // recent outlier flags (1/0), capped at outlierWindowSize
  int outlierCount_ = 0;           // cached sum of outlierWindow_
};

// A fixed-size array of independent FremenCellModel cells, updated together one
// frame at a time. Frame states are encoded as int8: -1 = unknown (skipped),
// 0..100 = percent occupied (mapped to a probability of 0.01 * value).
class FremenArrayModel {
 public:
  FremenArrayModel();

  // Set the candidate periods and configs applied to cells on the next
  // initialize(). Call before initialize().
  void configure(const std::vector<float>& candidatePeriods,
                 const ChangePointConfig& changePointConfig,
                 const AdaptiveBasisConfig& adaptiveConfig);

  // Allocate `stateCount` cells. Returns false for a zero count.
  bool initialize(std::size_t stateCount);
  void reset();                  // drop all cells (back to uninitialized)
  bool isInitialized() const;
  std::size_t size() const;      // number of cells

  // Absorb one frame of states (size must equal size()). See AddFrameResult.
  AddFrameResult addFrame(uint32_t timestamp, const std::vector<int8_t>& states);

  // Per-cell prediction / entropy for a frame. Output vector is resized to
  // size(); returns the cell count, or -1 if uninitialized / null output.
  int predictFrame(uint32_t timestamp, int order, std::vector<float>* probabilities) const;
  int entropyFrame(uint32_t timestamp, int order, std::vector<float>* entropies) const;

  // Mean per-order prediction error across all valid states in the frame.
  // errors is resized to maxOrder + 1. Optionally reports the mean outlier
  // ratio and the best (lowest-error) order. Returns the best order, or -1.
  int evaluateFrame(uint32_t timestamp,
                    const std::vector<int8_t>& states,
                    int maxOrder,
                    std::vector<float>* errors,
                    float* meanOutlierRatio = nullptr,
                    int* bestOrder = nullptr) const;

  int totalChangePointCount() const;   // summed over all cells
  float meanOutlierRatio() const;      // averaged over all cells
  // Active periods of the first cell, as a cheap representative of the array.
  std::vector<float> representativeActivePeriods() const;

 private:
  std::vector<float> candidatePeriods_;
  ChangePointConfig changePointConfig_;
  AdaptiveBasisConfig adaptiveConfig_;
  std::vector<FremenCellModel> cells_;
};

// Default candidate periods: the harmonics of a day (24h / k, k = 1..24) and of
// a week (7d / k, k = 1..8), de-duplicated and sorted. Units are seconds.
std::vector<float> defaultCandidatePeriods();

}  // namespace libfremen

#endif
