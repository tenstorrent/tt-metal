# E06: Tile Layout

Understand tiled vs row-major memory layouts and their performance implications.

## Goal

Compare performance between tiled and row-major tensor layouts using Tracy profiling.

## Key Concepts

- Row-major (RM) layout: Standard C-style contiguous memory
- Tile layout: 32x32 tiles optimized for tensix compute
- Layout conversion overhead vs compute efficiency

## Reference

- `docs/source/tt-metalium/tt_metal/advanced_topics/tiles.rst` - Tile format documentation

## Workflow

1. Implement matmul+add with both tile and row-major layouts
2. Profile both versions with Tracy
3. Compare and document the performance differences
