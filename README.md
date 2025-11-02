# Information Particle System

**Version**: 2.0.0  
**Author**: Beijing Qiuyishusheng Technology Center  
**License**: MIT

## Overview

Information Particle System is a framework for information structure analysis and quality assessment. The system discretizes continuous information into interpretable particles, each characterized by 12-dimensional transparent features.

## Core Concepts

### Information Particle
The fundamental unit of information representation, analogous to pixels in image processing. Each particle contains:
- 12-dimensional feature vector
- Structure-Information-Function (SIF) quality metric
- Original content for lossless reconstruction

### Feature Structure
```
[Time Dimension - 4D]
├── inner_time: Relative temporal position
├── outer_time: Absolute timestamp
├── time_flow: Rate of change
└── current_time: Current state marker

[Spatial Dimension - 3D]
├── spatial_x: Position coordinate
├── spatial_y: Content mean value
└── spatial_z: Content variance

[Structure Dimension - 4D]
├── density: Information density
├── connectivity: Inter-particle correlation
├── stability: Content stability
└── energy: Information magnitude

[Quality Metric - 1D]
└── sif_value: Composite quality score
```

### SIF Metric
Structure-Information-Function (SIF) value quantifies information quality:

```
SIF = 0.3 × Structure + 0.5 × Information + 0.2 × Function
```

Range: [0, 1], where higher values indicate richer information structure.

## Key Features

- **Lossless Reconstruction**: Perfect reconstruction with MSE=0
- **Transparent Features**: 12 interpretable dimensions
- **Quality Assessment**: Automated SIF scoring
- **Topological Mapping**: Spherical coordinate representation
- **Pure Mathematical**: Rule-based, no neural networks

## Installation

```bash
git clone https://github.com/changsheng137/information-sphere-system.git
cd information-sphere-system
pip install -r requirements.txt
```

## Quick Start

```python
from src.information_oriented_system import InformationOrientedSystemV2
import torch

# Initialize system
system = InformationOrientedSystemV2(particle_size=4)

# Process data
data = torch.randn(28, 28)
output = system.forward(data)

# Results
print(f"Particles: {output['num_particles']}")
print(f"Groups: {output['num_groups']}")
print(f"SIF Score: {output['avg_sif']:.4f}")

# Lossless reconstruction
reconstructed = system.reconstruct(output)
mse = torch.nn.functional.mse_loss(reconstructed, data)
print(f"MSE: {mse:.10f}")  # Expected: 0.0
```

## Architecture

```
Input Data [H×W]
    ↓
┌─────────────────────────┐
│ Particle Extraction     │
│ ├── Partitioning        │
│ ├── Feature Calculation │
│ └── SIF Computation     │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Group Construction      │
│ ├── Similarity Analysis │
│ └── Clustering          │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Spherical Mapping       │
│ └── Topology Projection │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Lossless Reconstruction │
└─────────────────────────┘
```

## Experimental Validation

### Reconstruction Quality
- Test samples: 350+ images
- Perfect reconstruction: 100% (MSE=0)
- Datasets: MNIST, Fashion-MNIST

### Baseline Comparison

| Method | MSE | Processing Time | Additional Info |
|--------|-----|-----------------|-----------------|
| Direct Storage | 0.0 | 0.03ms | None |
| PCA (64) | 0.0051 | 0.34ms | None |
| PCA (128) | 0.0009 | 0.10ms | None |
| **Ours** | **0.0** | **1.00ms** | **12D + SIF + Structure** |

### Component Validation
Ablation study confirms contribution of each component:
- Full system: Complete feature set
- Without SIF: No quality assessment
- Simplified features: Incomplete description
- Without grouping: No semantic structure
- Without connectivity: No topological information

## Applications

### 1. Quality Assessment
```python
quality = system.forward(image)['avg_sif']
# Interpret: 0.6+ high, 0.4-0.6 medium, <0.4 low
```

### 2. Anomaly Detection
```python
anomalies = [p for p in particles if p.sif_value < threshold]
```

### 3. Structure Visualization
```python
# Generate SIF distribution, connectivity network, sphere projection
```

## Performance

| Metric | Value |
|--------|-------|
| Processing Speed | ~1ms per 28×28 image |
| Memory Overhead | +10.8% (12D features) |
| Time Complexity | O(n + k²) |
| Space Complexity | O(k × (12 + s)) |

## Documentation

- [Theoretical Foundation](docs/THEORETICAL_FOUNDATION.md)
- [Experiment Report](docs/EXPERIMENT_REPORT.md)
- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API.md)

## Project Structure

```
information-sphere-v1.0/
├── src/
│   ├── information_particle_system.py    # Core system
│   └── information_oriented_system.py    # Main interface
├── experiments/
│   ├── baseline_comparison.py            # Baseline tests
│   └── ablation_study.py                 # Component validation
├── examples/
│   ├── basic_usage.py                    # Quick start
│   └── application_demo.py               # Application examples
├── docs/                                 # Documentation
└── tests/                                # Unit tests
```

## Citation

If you use this system in your research, please cite:

```
@software{qiuyishusheng2025information,
  title={Information Particle System: A Framework for Information Structure Analysis},
  author={Beijing Qiuyishusheng Technology Center},
  year={2025},
  version={2.0.0}
}
```

## License

MIT License - see LICENSE file

## Contact

- GitHub: [@changsheng137](https://github.com/changsheng137)
- Issues: [GitHub Issues](https://github.com/changsheng137/information-sphere-system/issues)

## Acknowledgments

Theoretical foundations:
- Shannon's Information Theory
- Cognitive Dimension Theory
- Time-Set Dimension Framework

---

**Beijing Qiuyishusheng Technology Center** © 2025
