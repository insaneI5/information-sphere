# User Guide

**Version**: 2.0.0  
**Author**: Beijing Qiuyishusheng Technology Center

---

## Quick Start

### Installation

```bash
git clone https://github.com/changsheng137/information-sphere-system.git
cd information-sphere-system
pip install -r requirements.txt
```

### Basic Usage

```python
from src.information_oriented_system import InformationOrientedSystemV2
import torch

# Initialize
system = InformationOrientedSystemV2(particle_size=4)

# Process data
data = torch.randn(28, 28)
output = system.forward(data)

# View results
print(f"Particles: {output['num_particles']}")
print(f"Groups: {output['num_groups']}")
print(f"SIF Score: {output['avg_sif']:.4f}")

# Reconstruct
reconstructed = system.reconstruct(output)
```

---

## System Configuration

### Parameters

```python
InformationOrientedSystemV2(
    particle_size=4,           # Particle size (affects granularity)
    similarity_threshold=0.5,  # Grouping threshold
    device='cpu'              # Computing device
)
```

**particle_size**: Determines number of particles
- Smaller → more particles → finer analysis → slower
- Larger → fewer particles → coarser analysis → faster
- Recommended: 4 for 28×28 images

**similarity_threshold**: Controls grouping sensitivity
- Higher → fewer groups → stricter clustering
- Lower → more groups → looser clustering
- Default: 0.5

---

## Understanding Output

### Output Structure

```python
output = {
    'particles': List[InformationParticle],  # Individual particles
    'groups': List[InformationGroup],        # Semantic groups
    'num_particles': int,                    # Particle count
    'num_groups': int,                       # Group count
    'avg_sif': float                         # Average quality
}
```

### Information Particle

Each particle contains:

```python
particle.sif_value       # Quality score [0, 1]
particle.connectivity    # Network connection [0, 1]
particle.energy         # Information magnitude
particle.density        # Information density
particle.stability      # Content stability
particle.raw_content    # Original data segment
```

### 12-Dimensional Features

```python
# Time (4D)
particle.inner_time      # Relative position
particle.outer_time      # Absolute timestamp
particle.time_flow       # Change rate
particle.current_time    # Current marker

# Space (3D)
particle.spatial_x       # X coordinate
particle.spatial_y       # Y coordinate (mean)
particle.spatial_z       # Z coordinate (variance)

# Structure (4D)
particle.density         # Information density
particle.connectivity    # Inter-particle correlation
particle.stability       # Content stability
particle.energy          # Information magnitude

# Quality (1D)
particle.sif_value       # Composite quality score
```

### Spherical Coordinates

```python
r, theta, phi = particle.get_sphere_coordinates()
```

- `r`: Radial distance (abstraction level)
- `theta`: Polar angle (complexity)
- `phi`: Azimuthal angle (semantic position)

---

## Applications

### 1. Quality Assessment

```python
def assess_quality(image):
    system = InformationOrientedSystemV2(particle_size=4)
    output = system.forward(image)
    return output['avg_sif']

quality = assess_quality(image)
if quality > 0.6:
    print("High quality")
elif quality > 0.4:
    print("Medium quality")
else:
    print("Low quality")
```

### 2. Anomaly Detection

```python
def detect_anomalies(image, threshold=0.35):
    system = InformationOrientedSystemV2(particle_size=4)
    output = system.forward(image)
    
    anomalies = []
    for p in output['particles']:
        if p.sif_value < threshold:
            anomalies.append({
                'index': p.sequence_index,
                'sif': p.sif_value,
                'position': (p.spatial_x, p.spatial_y)
            })
    return anomalies
```

### 3. Structure Visualization

```python
import matplotlib.pyplot as plt

def visualize_structure(image):
    system = InformationOrientedSystemV2(particle_size=4)
    output = system.forward(image)
    
    # SIF distribution
    sif_values = [p.sif_value for p in output['particles']]
    plt.hist(sif_values, bins=10)
    plt.xlabel('SIF Value')
    plt.ylabel('Count')
    plt.title(f'SIF Distribution (Avg: {output["avg_sif"]:.3f})')
    plt.show()
```

### 4. Batch Processing

```python
# Process multiple images
results = []
for image in image_list:
    output = system.forward(image)
    results.append({
        'sif': output['avg_sif'],
        'particles': output['num_particles'],
        'groups': output['num_groups']
    })

# Summary statistics
avg_sif = sum(r['sif'] for r in results) / len(results)
print(f"Average SIF: {avg_sif:.4f}")
```

---

## Performance Considerations

### Processing Speed

| Image Size | Particle Size | Time (ms) |
|:----------:|:-------------:|:---------:|
| 28×28 | 4 | ~1.0 |
| 28×28 | 7 | ~0.8 |
| 32×32 | 4 | ~1.2 |

### Memory Usage

For 28×28 image with 7 particles:
- Original: 3.14 KB
- Features: 0.34 KB
- Total: 3.48 KB (+10.8%)

### Optimization Tips

1. **Use GPU for large batches**
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
system = InformationOrientedSystemV2(device=device)
```

2. **Reuse system instance**
```python
system = InformationOrientedSystemV2()
for image in images:
    output = system.forward(image)  # Reuse, don't recreate
```

3. **Adjust particle_size for speed/accuracy trade-off**
```python
# Fast processing
system = InformationOrientedSystemV2(particle_size=14)

# Detailed analysis
system = InformationOrientedSystemV2(particle_size=1)
```

---

## Common Issues

### Q: How to choose particle_size?

**A**: Depends on application needs:
- Fine analysis: particle_size=1-2 (28 particles)
- Balanced: particle_size=4 (7 particles) - recommended
- Fast processing: particle_size=14 (2 particles)

### Q: What does SIF value represent?

**A**: SIF is a composite quality metric:
- 0.6+: High complexity/quality
- 0.4-0.6: Medium
- <0.4: Low complexity/simple content

### Q: Why is reconstruction always perfect?

**A**: System preserves raw_content in each particle, ensuring lossless reconstruction. The 12D features describe information structure, not replace content.

### Q: Can it process non-square images?

**A**: Yes, system handles arbitrary 2D data:
```python
data = torch.randn(28, 56)  # Non-square
output = system.forward(data)  # Works fine
```

### Q: What data types are supported?

**A**: Currently optimized for 2D data:
- Grayscale images [H, W]
- 2D matrices [H, W]
- Feature maps [H, W]

For other types:
- RGB images: Process each channel separately
- 1D sequences: Reshape to [1, L] or [L, 1]

---

## Advanced Usage

### Custom Feature Extraction

```python
# Extract specific features
for particle in output['particles']:
    time_features = [
        particle.inner_time,
        particle.outer_time,
        particle.time_flow,
        particle.current_time
    ]
    
    space_features = [
        particle.spatial_x,
        particle.spatial_y,
        particle.spatial_z
    ]
    
    # Use features for downstream tasks
```

### Group Analysis

```python
# Analyze information groups
for i, group in enumerate(output['groups']):
    print(f"Group {i}:")
    print(f"  Size: {len(group.particles)}")
    print(f"  Avg SIF: {group.avg_sif:.4f}")
    
    # Get group's spherical position
    r, theta, phi = group.get_sphere_coordinates()
    print(f"  Position: (r={r:.2f}, θ={theta:.2f}, φ={phi:.2f})")
```

### Connectivity Analysis

```python
# Build connectivity matrix
n = len(output['particles'])
connectivity_matrix = torch.zeros(n, n)

for i, pi in enumerate(output['particles']):
    for j, pj in enumerate(output['particles']):
        similarity = torch.cosine_similarity(
            pi.raw_content.flatten().unsqueeze(0),
            pj.raw_content.flatten().unsqueeze(0)
        )
        connectivity_matrix[i, j] = similarity

# Visualize network
import matplotlib.pyplot as plt
plt.imshow(connectivity_matrix, cmap='viridis')
plt.colorbar(label='Similarity')
plt.title('Particle Connectivity Network')
plt.show()
```

---

## API Reference

### InformationOrientedSystemV2

Main system class for information particle analysis.

#### Methods

**`__init__(particle_size=4, similarity_threshold=0.5, device='cpu')`**

Initialize system.

**`forward(data: torch.Tensor) -> Dict`**

Process data and extract information structure.

Parameters:
- `data` (Tensor): 2D input data [H, W]

Returns:
- `dict` with keys: particles, groups, num_particles, num_groups, avg_sif

**`reconstruct(output: Dict) -> torch.Tensor`**

Reconstruct original data from particles.

Parameters:
- `output` (dict): Output from forward()

Returns:
- `Tensor`: Reconstructed data [H, W]

---

## Examples

Complete examples available in `/examples` directory:
- `basic_usage.py`: Quick start guide
- `application_demo.py`: Four application scenarios
- See `/experiments` for validation scripts

---

## Support

- Documentation: `/docs`
- Issues: GitHub Issues
- Repository: https://github.com/changsheng137/information-sphere-system

---

**Beijing Qiuyishusheng Technology Center** © 2025

