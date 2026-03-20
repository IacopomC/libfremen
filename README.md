# libfremen (ROS-agnostic)

`libfremen` is a standalone C++ library for long-term periodic binary-state modeling.

It provides:

- Incremental FreMEn-style temporal modeling
- Online change-point detection from an outlier set
- Low-rate adaptive basis selection (dominant periodicities)
- Prediction, entropy, and online model-order evaluation

## Build standalone

```bash
cd libfremen
cmake -S . -B build
cmake --build build -j
```

## Main entry points

- `libfremen::FremenCellModel`
- `libfremen::FremenArrayModel`
- `libfremen::defaultCandidatePeriods()`