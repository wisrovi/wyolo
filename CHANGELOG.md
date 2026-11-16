# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.1] - 2026-11-16
### Fixed
- **Exhaustive Artifacts Isolation in CleanFolderExtra:** Enhanced `CleanFolderExtra` to rigorously walk and remove all individual files and subdirectories inside `extras/model_focus/` and other output directories, avoiding leftover heatmaps or predictions across executions on mounted Docker volumes.
- Bumped version to `2.3.1`.

## [1.0.1] - 2026-05-12

### Changed
- Consolidated training logic into `wyolo.trainer.trainer_wrapper`.
- Streamlined project structure by removing redundant core modules.
- Enhanced model logging to MLflow as artifacts.
- Improved GPU/CPU forcing logic.

## [1.0.0] - 2026-05-11

### Added
- Initial professional release.
- Comprehensive MLOps integration with MLflow and MinIO.
- Support for YOLOv8 and RT-DETR.
- CLI for automated training.
- DVC integration for dataset versioning.
- State-machine based worker implementation.
- Professional documentation and GitHub Actions workflows.
