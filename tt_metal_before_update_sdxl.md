# TT-Metal SDXL Implementation: Pre-Update Analysis

## Sampling Process Overview

### Core Sampling Mechanism
- Implemented in `run_tt_image_gen` function
- Iterates over a series of timesteps (typically 20-50 steps)
- Each iteration involves:
  1. UNet noise prediction
  2. Classifier-Free Guidance (CFG) application
  3. Scheduler latent update

### Unique TT-Metal Specifics
- Utilizes TTNN (Tenstorrent Neural Network) tensor operations
- Supports multi-device tensor sharding and replication
- Custom text encoders and UNet optimized for Tenstorrent hardware
- Supports parallel Classifier-Free Guidance across multiple devices

## Data Precision

### Primary Precision: `bfloat16`
- Most tensor conversions use `ttnn.bfloat16`
- Input tensors converted from `torch.float32` to `ttnn.bfloat16`

### Precision Strategy
- Computational pipeline predominantly uses bfloat16
- Intermediate computations may use float32
- Specific optimization for image quality noted in UNet implementation

### Precision Configuration
```python
# Typical tensor conversion
ttnn.from_torch(
    tensor,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    ...
)
```

### Rationale for Precision Choice
- Empirical testing showed bfloat16 + HiFi4 leads to better image quality
- Optimized for Tenstorrent's tensor processing capabilities

## Key Components
- Custom UNet implementation
- Specialized text encoders
- Tenstorrent-optimized scheduler
- Multi-device tensor processing support

## Performance Considerations
- Uses trace capture for performance optimization
- Supports both single and multi-device configurations
- Careful memory management with tensor allocation and deallocation

## Sampling Iteration Pseudocode
```python
for timestep in timesteps:
    # Predict noise with UNet
    noise_pred = unet(latent, timestep, conditioning)

    # Apply Classifier-Free Guidance
    noise_pred = apply_cfg(noise_pred, guidance_scale)

    # Update latent using scheduler
    latent = scheduler.step(noise_pred, latent)
```

## Future Optimization Opportunities
- Further precision tuning
- Enhanced multi-device scaling
- Continued performance improvements in tensor operations