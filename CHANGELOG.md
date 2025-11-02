# Changelog

## [2.0.0] - 2025-11-02

### Added
- Complete refactoring to information particle system
- 12-dimensional transparent feature framework
- SIF (Structure-Information-Function) quality metric
- Spherical topological mapping
- Lossless reconstruction mechanism
- Comprehensive experimental validation (6 experiments)
- Baseline comparison (Direct Storage, PCA)
- Ablation study validating all components
- Application examples (quality assessment, anomaly detection, visualization)
- Complete documentation (theory, experiments, user guide)

### Changed
- Core architecture redesigned for information structure analysis
- Feature system upgraded from implicit to explicit 12D representation
- Quality assessment mechanism standardized via SIF metric
- Documentation rewritten for professional clarity

### Technical
- Processing speed: ~1ms per 28×28 image
- Memory overhead: +10.8% for feature storage
- Perfect reconstruction: 100% (MSE=0) on 350+ test samples
- Time complexity: O(n + k²)
- Space complexity: O(k × (12 + s))

### Validated
- Multi-particle generation (1-28 particles)
- Cross-dataset generalization (MNIST, Fashion-MNIST)
- Superior to baselines (PCA, Direct Storage)
- Component necessity confirmed via ablation
- Theoretical completeness established

---

## [1.0.1] - 2025-11-01

### Performance
- 2.28x speedup through optimization
- Reduced GPU-CPU transfers
- Optimized similarity calculations
- Improved feature computation

### Fixed
- Device mismatch errors
- Dimension alignment issues
- Memory efficiency improvements

---

## [1.0.0] - 2025-10-31

### Initial Release
- Basic information-oriented system
- Information element extraction
- Information group construction
- Sphere mapping
- Reconstruction capability
