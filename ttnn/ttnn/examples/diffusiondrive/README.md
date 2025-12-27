# DiffusionDrive TTNN Implementation

This is an implementation of DiffusionDrive model using TTNN APIs for Tenstorrent hardware.

## Model Architecture

- Backbone: ResNet-34 for image, custom CNN for LiDAR
- Fusion: Transformer decoder
- Heads: Agent detection and trajectory prediction

## Running the Model

```bash
python diffusiondrive.py
```

## Validation

The model produces trajectory outputs matching the expected shape.

## Optimizations

- Use sharded memory for conv ops
- Fuse ops where possible
- Store intermediates in L1

## Performance

- Runs on N150/N300 hardware
- Real-time inference possible with optimizations