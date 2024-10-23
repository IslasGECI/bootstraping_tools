# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.2.0] - 2024-10-23
### Added
- Add abstract class `AbstractSeriesBootstrapper` as template for `LambdasBootstrapper` and `ProgressBootstrapper` classes
- Add class `Bootstrap_from_time_series_parametrizer` from `population_trend` repository

### Changed
- New argument `independent_variable `in class `Bootstrap_from_time_series_parametrizer`

### Removed
- Move class `LambdasBootstrapper` to `population_trend` repository
- Move class `ProgressBootstrapper` to `eradication_data_requirements` repository


## [3.1.0] - 2024-10-17
### Added
- Import class `Bootstrap_from_time_series` from package `population-trend`
### Changed
- Rename imported class `Bootstrap_from_time_series` to `LambdasBootstrapper`

## [3.0.0] - 2023-07-06


[unreleased]: https://github.com/IslasGECI/bootstrapping_tools/compare/v3.1.0...HEAD
[3.0.0]: https://github.com/IslasGECI/bootstrapping_tools/releases/tag/v3.0.0
