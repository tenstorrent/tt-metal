# TTNN Tensor-Manipulation Ops — AI-Generation Difficulty Ranking

This document inventories the tensor-manipulation (TM) operations exposed by `ttnn` (under `ttnn/cpp/ttnn/operations/`) and scores how hard each one is expected to be for an AI agent to generate end-to-end (host-side op + device kernels + Python binding) in the tt-metal stack.

## Methodology

Each op is given a **difficulty score from 1 (trivial) to 10 (very hard)**. The score is a qualitative judgement based on the following factors:

| Factor | Effect on score |
|---|---|
| Pure metadata change (no kernel) | Strongly lowers |
| Direct element-wise mapping with no coordinate math | Lowers |
| Well-known PyTorch / NumPy semantics with abundant reference examples | Lowers |
| Crosses tile boundaries (TILE layout coordinate math) | Raises |
| Sharding / multi-core coordination required | Raises |
| Multiple variants (row-major + tiled, sharded + interleaved) | Raises |
| Dynamic / data-dependent output shape | Raises |
| Irregular memory access (gather / scatter / index-driven) | Raises |
| Domain-specific semantics with limited public reference (e.g. MoE) | Raises |
| NoC-level orchestration of partial / streaming sharding | Raises |

The list is sorted **from easiest to hardest**.

## Kernel-less ops (metadata-only / pure allocation)

These ops do **not** require a device kernel — they either return a fresh allocation with no data initialization, or emit a new shape/stride descriptor over existing storage. They are excluded from the difficulty scoring below because they are essentially host-side bookkeeping.

| Op | Category | Why no kernel |
|---|---|---|
| `empty` | Creation | Pure tensor allocation; no data initialization. |
| `view` | Shape | Zero-cost metadata-only reshape over identical storage. |
| `reshape_view` | Shape | Same as `view`; emits a new shape descriptor over identical storage. |
| `squeeze` | Shape | Removes size-1 dims; metadata only. |
| `unsqueeze` | Shape | Inserts a size-1 dim; metadata only. |

## Op-by-op scoring (sorted easiest → hardest)

| # | Op | Score | Category | Why this score |
|---|---|---|---|---|
| 1 | `zeros` | 2 | Creation | Trivial fill with constant 0; standard creation pattern. |
| 2 | `ones` | 2 | Creation | Trivial fill with constant 1. |
| 3 | `full` | 2 | Creation | Trivial fill with a scalar. |
| 4 | `arange` | 2 | Creation | 1-D sequential fill with start/stop/step; very small kernel. |
| 5 | `clone` | 2 | Layout/Memory | Straight buffer copy with optional dtype/memory-config change. |
| 6 | `copy` | 2 | Layout/Memory | Straight src→dst copy. |
| 7 | `assign` | 2 | Layout/Memory | Variant of copy with optional dtype change. |
| 8 | `typecast` | 3 | Dtype | Per-element dtype conversion; SFPU-friendly, well-precedented. |
| 9 | `to_dtype` | 3 | Dtype | Thin wrapper over `typecast` semantics. |
| 10 | `fill_rm` | 3 | Padding/Filling | Region-fill in row-major; simple loops, no tile math. |
| 11 | `fill_implicit_tile_padding` | 3 | Padding/Filling | Fill the implicit padding region of a tile; localized to tile boundary. |
| 12 | `to_memory_config` | 3 | Property | Mostly orchestration over existing copy / sharding primitives. |
| 13 | `move` | 4 | Layout/Memory | Re-allocate then copy; orchestration-heavy but no new math. |
| 14 | `stack` | 4 | Concat/Split | Concat after `unsqueeze`; reuses concat machinery. |
| 15 | `chunk` | 4 | Shape | Even split along a dim; thin wrapper over `split`. |
| 16 | `narrow` | 4 | Slicing | Single-dim contiguous slice; simpler than full `slice`. |
| 17 | `transpose` | 4 | Reordering | Two-dim swap; well-trodden pattern, but tile-aware variants raise complexity. |
| 18 | `slice` | 5 | Slicing | N-D start/end/step; row-major + tile variants and stride math. |
| 19 | `concat` | 5 | Concat/Split | N-input concat along arbitrary dim; ok for row-major, harder when tiles are split mid-tile. |
| 20 | `split` | 5 | Concat/Split | Inverse of concat; size-list and even-split variants. |
| 21 | `pad` | 5 | Padding/Filling | N-D padding with arbitrary value; tile-aware version is non-trivial. |
| 22 | `repeat` | 5 | Repetition | Tile a tensor by an N-D repetition vector. |
| 23 | `repeat_interleave` | 5 | Repetition | Per-element repetition along a dim; index math is straightforward. |
| 24 | `expand` | 5 | Shape | Broadcast-expand (stride-zero in PyTorch) — must be materialized in tt-metal. |
| 25 | `bcast` | 5 | Sharding/Distribution | Binary broadcast (h/w/hw); well-defined patterns but multiple modes. |
| 26 | `to_layout` | 6 | Dtype/Layout | Dispatches between tilize / untilize and dtype conversion paths. |
| 27 | `tilize` | 6 | Tiling | Row-major → tile reblocking; mature reference kernels, but coordinate math is non-trivial. |
| 28 | `untilize` | 6 | Tiling | Tile → row-major reblocking; symmetric to `tilize`. |
| 29 | `tilize_with_zero_padding` | 6 | Tiling | `tilize` + zero-fill the padding region. |
| 30 | `roll` | 6 | Repetition | Cyclic shift with wraparound on multiple dims; index modulo math. |
| 31 | `indexed_fill` | 6 | Conditional | Conditional fill driven by a per-batch index tensor. |
| 32 | `tilize_with_val_padding` | 7 | Tiling | Tilize while padding to a target shape with an arbitrary value. |
| 33 | `untilize_with_unpadding` | 7 | Tiling | Untilize and crop to a target unpadded shape; reverse of the above. |
| 34 | `permute` | 7 | Reordering | General N-D permutation; tile-aware variants for swaps that cross tile boundaries are notably hard. |
| 35 | `gather` | 7 | Slicing/Indexing | Index-driven irregular reads along a dim; tile / sharding awareness compounds. |
| 36 | `scatter` | 7 | Slicing/Indexing | Index-driven irregular writes (with optional reduction); race-handling needed. |
| 37 | `scatter_add` | 7 | Conditional | Scatter with additive reduction — atomic-style accumulation. |
| 38 | `sort` | 8 | Property | Algorithmic op (returns values + indices); efficient parallel sort on tile-based memory is hard. |
| 39 | `fold` | 8 | Folding/Windowing | Sliding-window patches → image with stride/padding; complex coordinate transform. |
| 40 | `nonzero` | 8 | Special | Data-dependent output size; needs a host/device round-trip pattern. |
| 41 | `interleaved_to_sharded` | 8 | Sharding/Distribution | Multi-core sharded layout build-up; NoC + core-grid awareness. |
| 42 | `sharded_to_interleaved` | 8 | Sharding/Distribution | Symmetric to the above; gather across shards into a single interleaved buffer. |
| 43 | `reshard` | 9 | Sharding/Distribution | Arbitrary shard-config → shard-config; the most general case of the sharding triad. |
| 44 | `interleaved_to_sharded_partial` | 9 | Sharding/Distribution | Streaming / partial sharding — slice along a dim and feed shards incrementally. |
| 45 | `sharded_to_interleaved_partial` | 9 | Sharding/Distribution | Symmetric streaming gather; tightly coupled with consumer ops. |
| 46 | `moe_expert_token_remap` | 10 | Special/Expert | MoE-specific token remap driven by routing decisions; little public reference, complex semantics. |
| 47 | `moe_routing_remap` | 10 | Special/Expert | MoE-specific routing-weight remap for expert parallelism; same difficulty class. |

**Total: 52 ops** (47 scored + 5 kernel-less) spanning 14 logical categories. All host-side sources live under `ttnn/cpp/ttnn/operations/` (mostly the `data_movement/` subtree, with a few in `core/`, `copy/`, and `creation/`).

## Quick takeaways

- **Kernel-less ops** (separate section above): pure metadata/allocation ops that don't exercise the device at all. Excluded from scoring because there is no kernel to generate.
- **Bottom of the list (2–3, "free wins")**: simple creation, trivial copy / dtype conversion, and small region-fills. An AI can plausibly produce these from a one-line spec.
- **Middle of the list (4–6)**: standard data-movement ops with predictable patterns. Generation quality depends mostly on whether the agent correctly handles tile boundaries and dtype edge cases.
- **Upper-middle (7–8)**: ops where tile-aware coordinate math, irregular indexing, or data-dependent outputs dominate. These typically need iterative testing against a PyTorch reference.
- **Top of the list (9–10)**: sharding-aware ops with multi-core / NoC orchestration, and domain-specific MoE remaps. These are the most likely to require human-authored skeletons before an AI can finish them safely.
