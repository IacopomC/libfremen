// Implementation of the incremental FreMEn model. See fremen_model.hpp for the
// high-level idea; comments here focus on the per-observation math.
#include "libfremen/fremen_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace libfremen {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Restrict a probability estimate to the valid [0, 1] range.
float clamp01(float value) {
  if (value < 0.0f) return 0.0f;
  if (value > 1.0f) return 1.0f;
  return value;
}

}  // namespace

std::vector<float> defaultCandidatePeriods() {
  std::vector<float> periods;
  periods.reserve(40);

  const float day = 24.0f * 3600.0f;
  const float week = 7.0f * day;

  // Daily harmonics (24h, 12h, 8h, ...) and weekly harmonics (7d, 3.5d, ...),
  // i.e. period / k. These cover the dominant human-activity periodicities.
  for (int harmonic = 1; harmonic <= 24; ++harmonic) {
    periods.push_back(day / static_cast<float>(harmonic));
  }
  for (int harmonic = 1; harmonic <= 8; ++harmonic) {
    periods.push_back(week / static_cast<float>(harmonic));
  }

  // Some daily/weekly harmonics coincide; keep one copy, sorted.
  std::sort(periods.begin(), periods.end());
  periods.erase(std::unique(periods.begin(), periods.end()), periods.end());
  return periods;
}

FremenCellModel::FremenCellModel()
    : FremenCellModel(defaultCandidatePeriods(), ChangePointConfig(), AdaptiveBasisConfig()) {}

FremenCellModel::FremenCellModel(const std::vector<float>& candidatePeriods,
                                 const ChangePointConfig& changePointConfig,
                                 const AdaptiveBasisConfig& adaptiveConfig)
    : changePointConfig_(changePointConfig), adaptiveConfig_(adaptiveConfig) {
  // Allocate one harmonic per candidate period (falling back to the defaults).
  const std::vector<float> periods = candidatePeriods.empty() ? defaultCandidatePeriods() : candidatePeriods;
  harmonics_.resize(periods.size());
  for (std::size_t index = 0; index < periods.size(); ++index) {
    harmonics_[index].period = periods[index];
  }
  recomputeAmplitudesAndBasis();
}

void FremenCellModel::reset() {
  // Forget every observation: clear the mean, counters, outlier window and all
  // accumulated spectral coefficients. The set of candidate periods is kept.
  gain_ = 0.5f;
  measurements_ = 0;
  firstTime_ = -1;
  lastTime_ = -1;
  addsSinceRefresh_ = 0;
  outlierWindow_.clear();
  outlierCount_ = 0;
  activeIndices_.clear();

  for (std::size_t index = 0; index < harmonics_.size(); ++index) {
    harmonics_[index].realStates = 0.0f;
    harmonics_[index].imagStates = 0.0f;
    harmonics_[index].realBalance = 0.0f;
    harmonics_[index].imagBalance = 0.0f;
    harmonics_[index].amplitude = 0.0f;
    harmonics_[index].phase = 0.0f;
  }
}

void FremenCellModel::registerOutlier(bool outlier) {
  if (changePointConfig_.outlierWindowSize <= 0) return;

  // Sliding window of the most recent outlier flags; outlierCount_ is kept in
  // sync so currentOutlierRatio() stays O(1).
  outlierWindow_.push_back(outlier ? 1 : 0);
  if (outlier) ++outlierCount_;

  while (static_cast<int>(outlierWindow_.size()) > changePointConfig_.outlierWindowSize) {
    outlierCount_ -= outlierWindow_.front();
    outlierWindow_.pop_front();
  }
}

float FremenCellModel::currentOutlierRatio() const {
  if (outlierWindow_.empty()) return 0.0f;
  return static_cast<float>(outlierCount_) / static_cast<float>(outlierWindow_.size());
}

void FremenCellModel::recomputeAmplitudesAndBasis() {
  // Total time span observed so far; used to gate which periods are resolvable.
  const int64_t duration = (firstTime_ >= 0 && lastTime_ >= 0) ? (lastTime_ - firstTime_) : 0;

  std::vector<std::pair<float, int> > ranking;
  ranking.reserve(harmonics_.size());

  for (std::size_t index = 0; index < harmonics_.size(); ++index) {
    Harmonic& harmonic = harmonics_[index];
    // (states - balance) is the projection of the signal onto this harmonic
    // with the running mean subtracted, so a constant signal yields ~0.
    const float re = harmonic.realStates - harmonic.realBalance;
    const float im = harmonic.imagStates - harmonic.imagBalance;

    // Only trust a period once we've seen clearly more than one of its cycles
    // (>= 1.5 periods); otherwise its amplitude is unreliable, so force it to 0.
    if (measurements_ > 0 && (1.5f * harmonic.period) <= static_cast<float>(duration)) {
      harmonic.amplitude = std::sqrt(re * re + im * im) / static_cast<float>(measurements_);
    } else {
      harmonic.amplitude = 0.0f;
    }
    harmonic.phase = std::atan2(im, re);

    ranking.push_back(std::make_pair(harmonic.amplitude, static_cast<int>(index)));
  }

  // Keep the strongest harmonics (largest amplitude) as the active basis,
  // capped at maxOrder and dropping any with zero amplitude.
  std::sort(
      ranking.begin(), ranking.end(), [](const std::pair<float, int>& left, const std::pair<float, int>& right) {
        return left.first > right.first;
      });

  activeIndices_.clear();
  const int keepCount = std::max(0, std::min(adaptiveConfig_.maxOrder, static_cast<int>(ranking.size())));
  for (int index = 0; index < keepCount; ++index) {
    if (ranking[index].first <= 0.0f) break;
    activeIndices_.push_back(ranking[index].second);
  }
}

bool FremenCellModel::addObservation(uint32_t timestamp, float state, bool* changePointTriggered) {
  if (changePointTriggered != nullptr) *changePointTriggered = false;
  if (state < 0.0f || state > 1.0f) return false;  // reject out-of-range probabilities

  // Observations must be strictly increasing in time.
  if (measurements_ > 0 && timestamp <= static_cast<uint32_t>(lastTime_)) {
    return false;
  }

  if (measurements_ > 0) {
    // Compare the new observation against the current prediction; a large
    // residual is an outlier. Enough outliers in the window => the pattern has
    // changed, so drop everything and start the model over.
    const int effectiveOrder = std::max(0, std::min(adaptiveConfig_.maxOrder, static_cast<int>(activeIndices_.size())));
    const float residual = std::fabs(state - predict(timestamp, effectiveOrder));
    const bool outlier = residual > changePointConfig_.outlierThreshold;
    registerOutlier(outlier);

    const bool detectionReady = measurements_ >= changePointConfig_.minMeasurementsBeforeDetection &&
                                static_cast<int>(outlierWindow_.size()) == changePointConfig_.outlierWindowSize;

    if (detectionReady && currentOutlierRatio() >= changePointConfig_.triggerRatio) {
      ++changePointCount_;
      reset();
      if (changePointTriggered != nullptr) *changePointTriggered = true;
    }
  }

  // First observation after construction/reset seeds the mean.
  if (measurements_ == 0) {
    firstTime_ = static_cast<int64_t>(timestamp);
    gain_ = state;
  }

  // Incrementally update the running mean (gain_).
  const float oldGain = gain_;
  gain_ = (gain_ * static_cast<float>(measurements_) + state) / static_cast<float>(measurements_ + 1);

  // The "balance" terms are the mean-weighted reference spectrum. Because the
  // mean just changed, rescale the existing balance so it stays consistent with
  // the new mean before this observation's contribution is added below.
  if (oldGain > std::numeric_limits<float>::epsilon()) {
    for (std::size_t index = 0; index < harmonics_.size(); ++index) {
      harmonics_[index].realBalance = gain_ * harmonics_[index].realBalance / oldGain;
      harmonics_[index].imagBalance = gain_ * harmonics_[index].imagBalance / oldGain;
    }
  } else {
    for (std::size_t index = 0; index < harmonics_.size(); ++index) {
      harmonics_[index].realBalance = 0.0f;
      harmonics_[index].imagBalance = 0.0f;
    }
  }

  // Project this observation onto every harmonic: accumulate the signal sums
  // (states) and the matching mean reference (balance) at this timestamp.
  for (std::size_t index = 0; index < harmonics_.size(); ++index) {
    const float angle = 2.0f * kPi * static_cast<float>(timestamp) / harmonics_[index].period;
    harmonics_[index].realStates += state * std::cos(angle);
    harmonics_[index].imagStates += state * std::sin(angle);
    harmonics_[index].realBalance += gain_ * std::cos(angle);
    harmonics_[index].imagBalance += gain_ * std::sin(angle);
  }

  ++measurements_;
  ++addsSinceRefresh_;
  lastTime_ = static_cast<int64_t>(timestamp);

  // Refresh the active basis periodically, and eagerly during the first few
  // observations while the spectrum is still settling.
  if (addsSinceRefresh_ >= std::max(1, adaptiveConfig_.refreshInterval) || measurements_ <= 4) {
    recomputeAmplitudesAndBasis();
    addsSinceRefresh_ = 0;
  }

  return true;
}

float FremenCellModel::predict(uint32_t timestamp, int order) const {
  // Start from the mean and add the contribution of the top `order` harmonics.
  float estimate = gain_;
  if (order < 0) order = 0;
  const int effectiveOrder = std::min(order, static_cast<int>(activeIndices_.size()));

  for (int index = 0; index < effectiveOrder; ++index) {
    const Harmonic& harmonic = harmonics_[activeIndices_[index]];
    const float angle = 2.0f * kPi * static_cast<float>(timestamp) / harmonic.period;
    // Factor 2 reconstructs the real signal from the one-sided spectrum.
    estimate += 2.0f * harmonic.amplitude * std::cos(angle - harmonic.phase);
  }
  return clamp01(estimate);
}

float FremenCellModel::entropy(uint32_t timestamp, int order) const {
  // Binary entropy of the predicted probability (0 at a confident 0 or 1).
  const float probability = predict(timestamp, order);
  if (probability <= 0.0f || probability >= 1.0f) return 0.0f;
  return -(probability * std::log2(probability) + (1.0f - probability) * std::log2(1.0f - probability));
}

int FremenCellModel::evaluate(uint32_t timestamp, float state, int maxOrder, float* errors) const {
  // Prediction error at each model order, plus the order that minimises it.
  if (errors == nullptr || maxOrder < 0) return -1;
  float bestError = std::numeric_limits<float>::infinity();
  int bestOrder = 0;

  for (int order = 0; order <= maxOrder; ++order) {
    errors[order] = std::fabs(state - predict(timestamp, order));
    if (errors[order] < bestError) {
      bestError = errors[order];
      bestOrder = order;
    }
  }
  return bestOrder;
}

CellDebugStats FremenCellModel::debugStats() const {
  CellDebugStats stats;
  stats.gain = gain_;
  stats.measurements = measurements_;
  stats.firstTime = firstTime_;
  stats.lastTime = lastTime_;
  stats.outlierRatio = currentOutlierRatio();
  stats.changePointCount = changePointCount_;
  stats.activePeriods.reserve(activeIndices_.size());
  for (std::size_t index = 0; index < activeIndices_.size(); ++index) {
    stats.activePeriods.push_back(harmonics_[activeIndices_[index]].period);
  }
  return stats;
}

FremenArrayModel::FremenArrayModel() {
  candidatePeriods_ = defaultCandidatePeriods();
}

void FremenArrayModel::configure(const std::vector<float>& candidatePeriods,
                                 const ChangePointConfig& changePointConfig,
                                 const AdaptiveBasisConfig& adaptiveConfig) {
  // Stored and applied to every cell at the next initialize().
  candidatePeriods_ = candidatePeriods.empty() ? defaultCandidatePeriods() : candidatePeriods;
  changePointConfig_ = changePointConfig;
  adaptiveConfig_ = adaptiveConfig;
}

bool FremenArrayModel::initialize(std::size_t stateCount) {
  if (stateCount == 0) return false;
  // One independent cell per state, all sharing the configured periods/configs.
  cells_.clear();
  cells_.reserve(stateCount);
  for (std::size_t index = 0; index < stateCount; ++index) {
    cells_.push_back(FremenCellModel(candidatePeriods_, changePointConfig_, adaptiveConfig_));
  }
  return true;
}

void FremenArrayModel::reset() {
  cells_.clear();
}

bool FremenArrayModel::isInitialized() const {
  return !cells_.empty();
}

std::size_t FremenArrayModel::size() const {
  return cells_.size();
}

AddFrameResult FremenArrayModel::addFrame(uint32_t timestamp, const std::vector<int8_t>& states) {
  AddFrameResult result;
  if (!isInitialized() || states.size() != cells_.size()) return result;

  for (std::size_t index = 0; index < states.size(); ++index) {
    // int8 encoding: -1 = unknown (skip), valid range is 0..100 (percent).
    const int8_t state = states[index];
    if (state == -1) {
      ++result.ignoredUnknown;
      continue;
    }
    if (state < -1 || state > 100) {
      ++result.ignoredInvalid;
      continue;
    }

    // Map percent -> probability in [0, 1] and feed the matching cell.
    bool changed = false;
    if (cells_[index].addObservation(timestamp, 0.01f * static_cast<float>(state), &changed)) {
      ++result.updatedStates;
      if (changed) {
        ++result.changePointsTriggered;
        result.changedIndices.push_back(static_cast<int>(index));
      }
    }
  }
  return result;
}

int FremenArrayModel::predictFrame(uint32_t timestamp, int order, std::vector<float>* probabilities) const {
  if (!isInitialized() || probabilities == nullptr) return -1;
  probabilities->assign(cells_.size(), 0.0f);
  for (std::size_t index = 0; index < cells_.size(); ++index) {
    (*probabilities)[index] = cells_[index].predict(timestamp, order);
  }
  return static_cast<int>(cells_.size());
}

int FremenArrayModel::entropyFrame(uint32_t timestamp, int order, std::vector<float>* entropies) const {
  if (!isInitialized() || entropies == nullptr) return -1;
  entropies->assign(cells_.size(), 0.0f);
  for (std::size_t index = 0; index < cells_.size(); ++index) {
    (*entropies)[index] = cells_[index].entropy(timestamp, order);
  }
  return static_cast<int>(cells_.size());
}

int FremenArrayModel::evaluateFrame(uint32_t timestamp,
                                    const std::vector<int8_t>& states,
                                    int maxOrder,
                                    std::vector<float>* errors,
                                    float* meanOutlierRatio,
                                    int* bestOrder) const {
  if (!isInitialized() || errors == nullptr || maxOrder < 0 || states.size() != cells_.size()) return -1;

  errors->assign(maxOrder + 1, 0.0f);
  std::vector<float> local(maxOrder + 1, 0.0f);

  // Sum each order's error over all valid (known, in-range) states.
  int validStates = 0;
  for (std::size_t index = 0; index < states.size(); ++index) {
    const int8_t state = states[index];
    if (state == -1 || state < -1 || state > 100) continue;
    ++validStates;
    cells_[index].evaluate(timestamp, 0.01f * static_cast<float>(state), maxOrder, local.data());
    for (int order = 0; order <= maxOrder; ++order) {
      (*errors)[order] += local[order];
    }
  }

  if (validStates == 0) return -1;
  // Turn the sums into mean errors.
  for (int order = 0; order <= maxOrder; ++order) {
    (*errors)[order] /= static_cast<float>(validStates);
  }

  // Pick the order with the smallest mean error.
  int selectedOrder = 0;
  float selectedError = std::numeric_limits<float>::infinity();
  for (int order = 0; order <= maxOrder; ++order) {
    if ((*errors)[order] < selectedError) {
      selectedError = (*errors)[order];
      selectedOrder = order;
    }
  }

  if (meanOutlierRatio != nullptr) {
    *meanOutlierRatio = this->meanOutlierRatio();
  }
  if (bestOrder != nullptr) {
    *bestOrder = selectedOrder;
  }

  return selectedOrder;
}

int FremenArrayModel::totalChangePointCount() const {
  int total = 0;
  for (std::size_t index = 0; index < cells_.size(); ++index) {
    total += cells_[index].debugStats().changePointCount;
  }
  return total;
}

float FremenArrayModel::meanOutlierRatio() const {
  if (cells_.empty()) return 0.0f;
  float sum = 0.0f;
  for (std::size_t index = 0; index < cells_.size(); ++index) {
    sum += cells_[index].debugStats().outlierRatio;
  }
  return sum / static_cast<float>(cells_.size());
}

std::vector<float> FremenArrayModel::representativeActivePeriods() const {
  if (cells_.empty()) return std::vector<float>();
  return cells_.front().debugStats().activePeriods;
}

}  // namespace libfremen
