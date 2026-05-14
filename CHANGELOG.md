# Changelog

All notable changes to **WavFlow** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-13

### Added
- Initial open-source release of **WavFlow**.
- Raw-waveform audio generation with flow matching (no VAE, direct *x*-prediction).
- Inference pipelines for **VT2A** (video + text), **T2A** (text-only), and **V2A** (video-only).
- Single-node and multi-node `torchrun` training scripts (`scripts/launch/`).
- Feature extraction pipelines for CLIP image / text and Synchformer features.
- Auto-download of external weights (CLIP, Synchformer) and local computation of `empty_string.pth` on first use.
- Project page (`docs/`) with demo videos, architecture figure, and benchmark comparisons.
- One-shot environment setup (`scripts/setup.sh`).
- 4-sample demo dataset under `training_samples/` for end-to-end smoke tests.
