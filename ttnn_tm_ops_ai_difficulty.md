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

## Op-by-op scoring (sorted easiest → hardest)

| # | Op | Score | Category | Why this score |
|---|---|---|---|---|
| 1 | `empty` | 1 | Creation | Pure tensor allocation; no kernel, no data initialization. |
| 2 | `view` | 1 | Shape | Zero-cost metadata-only reshape; no kernel. |
| 3 | `reshape_view` | 1 | Shape | Same as `view`; emits a new shape descriptor over identical storage. |
| 4 | `squeeze` | 1 | Shape | Removes size-1 dims; metadata only. |
| 5 | `unsqueeze` | 1 | Shape | Inserts a size-1 dim; metadata only. |
| 6 | `zeros` | 2 | Creation | Trivial fill with constant 0; standard creation pattern. |
| 7 | `ones` | 2 | Creation | Trivial fill with constant 1. |
| 8 | `full` | 2 | Creation | Trivial fill with a scalar. |
| 9 | `arange` | 2 | Creation | 1-D sequential fill with start/stop/step; very small kernel. |
| 10 | `clone` | 2 | Layout/Memory | Straight buffer copy with optional dtype/memory-config change. |
| 11 | `copy` | 2 | Layout/Memory | Straight src→dst copy. |
| 12 | `assign` | 2 | Layout/Memory | Variant of copy with optional dtype change. |
| 13 | `typecast` | 3 | Dtype | Per-element dtype conversion; SFPU-friendly, well-precedented. |
| 14 | `to_dtype` | 3 | Dtype | Thin wrapper over `typecast` semantics. |
| 15 | `fill_rm` | 3 | Padding/Filling | Region-fill in row-major; simple loops, no tile math. |
| 16 | `fill_implicit_tile_padding` | 3 | Padding/Filling | Fill the implicit padding region of a tile; localized to tile boundary. |
| 17 | `to_memory_config` | 3 | Property | Mostly orchestration over existing copy / sharding primitives. |
| 18 | `move` | 4 | Layout/Memory | Re-allocate then copy; orchestration-heavy but no new math. |
| 19 | `stack` | 4 | Concat/Split | Concat after `unsqueeze`; reuses concat machinery. |
| 20 | `chunk` | 4 | Shape | Even split along a dim; thin wrapper over `split`. |
| 21 | `narrow` | 4 | Slicing | Single-dim contiguous slice; simpler than full `slice`. |
| 22 | `transpose` | 4 | Reordering | Two-dim swap; well-trodden pattern, but tile-aware variants raise complexity. |
| 23 | `slice` | 5 | Slicing | N-D start/end/step; row-major + tile variants and stride math. |
| 24 | `concat` | 5 | Concat/Split | N-input concat along arbitrary dim; ok for row-major, harder when tiles are split mid-tile. |
| 25 | `split` | 5 | Concat/Split | Inverse of concat; size-list and even-split variants. |
| 26 | `pad` | 5 | Padding/Filling | N-D padding with arbitrary value; tile-aware version is non-trivial. |
| 27 | `repeat` | 5 | Repetition | Tile a tensor by an N-D repetition vector. |
| 28 | `repeat_interleave` | 5 | Repetition | Per-element repetition along a dim; index math is straightforward. |
| 29 | `expand` | 5 | Shape | Broadcast-expand (stride-zero in PyTorch) — must be materialized in tt-metal. |
| 30 | `bcast` | 5 | Sharding/Distribution | Binary broadcast (h/w/hw); well-defined patterns but multiple modes. |
| 31 | `to_layout` | 6 | Dtype/Layout | Dispatches between tilize / untilize and dtype conversion paths. |
| 32 | `tilize` | 6 | Tiling | Row-major → tile reblocking; mature reference kernels, but coordinate math is non-trivial. |
| 33 | `untilize` | 6 | Tiling | Tile → row-major reblocking; symmetric to `tilize`. |
| 34 | `tilize_with_zero_padding` | 6 | Tiling | `tilize` + zero-fill the padding region. |
| 35 | `roll` | 6 | Repetition | Cyclic shift with wraparound on multiple dims; index modulo math. |
| 36 | `indexed_fill` | 6 | Conditional | Conditional fill driven by a per-batch index tensor. |
| 37 | `tilize_with_val_padding` | 7 | Tiling | Tilize while padding to a target shape with an arbitrary value. |
| 38 | `untilize_with_unpadding` | 7 | Tiling | Untilize and crop to a target unpadded shape; reverse of the above. |
| 39 | `permute` | 7 | Reordering | General N-D permutation; tile-aware variants for swaps that cross tile boundaries are notably hard. |
| 40 | `gather` | 7 | Slicing/Indexing | Index-driven irregular reads along a dim; tile / sharding awareness compounds. |
| 41 | `scatter` | 7 | Slicing/Indexing | Index-driven irregular writes (with optional reduction); race-handling needed. |
| 42 | `scatter_add` | 7 | Conditional | Scatter with additive reduction — atomic-style accumulation. |
| 43 | `sort` | 8 | Property | Algorithmic op (returns values + indices); efficient parallel sort on tile-based memory is hard. |
| 44 | `fold` | 8 | Folding/Windowing | Sliding-window patches → image with stride/padding; complex coordinate transform. |
| 45 | `nonzero` | 8 | Special | Data-dependent output size; needs a host/device round-trip pattern. |
| 46 | `interleaved_to_sharded` | 8 | Sharding/Distribution | Multi-core sharded layout build-up; NoC + core-grid awareness. |
| 47 | `sharded_to_interleaved` | 8 | Sharding/Distribution | Symmetric to the above; gather across shards into a single interleaved buffer. |
| 48 | `reshard` | 9 | Sharding/Distribution | Arbitrary shard-config → shard-config; the most general case of the sharding triad. |
| 49 | `interleaved_to_sharded_partial` | 9 | Sharding/Distribution | Streaming / partial sharding — slice along a dim and feed shards incrementally. |
| 50 | `sharded_to_interleaved_partial` | 9 | Sharding/Distribution | Symmetric streaming gather; tightly coupled with consumer ops. |
| 51 | `moe_expert_token_remap` | 10 | Special/Expert | MoE-specific token remap driven by routing decisions; little public reference, complex semantics. |
| 52 | `moe_routing_remap` | 10 | Special/Expert | MoE-specific routing-weight remap for expert parallelism; same difficulty class. |

**Total: 52 ops** spanning 14 logical categories. All host-side sources live under `ttnn/cpp/ttnn/operations/` (mostly the `data_movement/` subtree, with a few in `core/`, `copy/`, and `creation/`).

## Quick takeaways

- **Bottom of the list (1–3, "free wins")**: shape/metadata ops, simple creation, and trivial copy / dtype conversion. An AI can plausibly produce these from a one-line spec.
- **Middle of the list (4–6)**: standard data-movement ops with predictable patterns. Generation quality depends mostly on whether the agent correctly handles tile boundaries and dtype edge cases.
- **Upper-middle (7–8)**: ops where tile-aware coordinate math, irregular indexing, or data-dependent outputs dominate. These typically need iterative testing against a PyTorch reference.
- **Top of the list (9–10)**: sharding-aware ops with multi-core / NoC orchestration, and domain-specific MoE remaps. These are the most likely to require human-authored skeletons before an AI can finish them safely.
