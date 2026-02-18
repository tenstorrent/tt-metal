# E08: Tile Layout

Understand tiled vs row-major memory layouts and data formats.

## Goal

Learn about data layouts and formats:
- Understand tile layout (32x32 tiles) vs row-major
- Learn different data formats (BFloat16, Float16_b, BFloat8_b)
- Measure the impact of layout and format on performance

## Reference

- `docs/source/tt-metalium/tt_metal/advanced_topics/tiles.rst`

## Key Concepts

### Tile Layout
- Tensix works on 32x32 tiles natively
- Data stored as contiguous tiles, not contiguous rows
- Required for compute operations

### Row-Major Layout
- Standard C-style contiguous memory
- Elements contiguous along innermost dimension
- Some ops (concat, reshape) work better with row-major

### Data Formats
- **BFloat16**: Default for training (~3 decimal digits)
- **Float16_b**: Alternative 16-bit format
- **BFloat8_b**: 8-bit for inference (~2 decimal digits)
- Lower precision = faster but less accurate

## Common Pitfalls

1. **Dimension alignment** - Tile layout needs dimensions divisible by 32
2. **Format conversion overhead** - Minimize conversions
3. **Accuracy loss** - Lower precision loses accuracy in accumulation
