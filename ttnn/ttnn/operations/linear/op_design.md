# Operation Design: linear

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused matmul + optional bias add) |
| Goal | 2D matmul `output = input @ weight` with optional row-broadcast bias add `output += bias[0, :]`. Phase 0 baseline: single core, tile-aligned, DRAM-interleaved bf16. Forces the kernel-helper library's `matmul_block` path end-to-end (its own dataflow + compute kernels — NOT a Python composite over `ttnn.matmul`). |
| Math | `output[i, j] = sum_k(input[i, k] * weight[k, j]) [+ bias[0, j]]` |
| Mode | Derivative (greenfield op, follows the generic_op_template / toy_binary_in_place pattern) |
| References | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp` (matmul helper), `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp` (bias helper), `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` (canonical helper-using kernel), `.claude/references/generic_op_template/` (file/folder layout), `ttnn/ttnn/operations/toy_binary_in_place/` (host-side descriptor pattern). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | shape `[1, 1, M, K]`, bf16, TILE, DRAM-interleaved, M & K divisible by 32 | — | n/a (tensor) |
| `weight_tensor` | `ttnn.Tensor` | yes | shape `[1, 1, K, N]`, bf16, TILE, DRAM-interleaved, K & N divisible by 32 | — | n/a (tensor) |
| `bias` | `ttnn.Tensor` (kw-only) | no | shape `[1, 1, 32, N]`, bf16, TILE, DRAM-interleaved, N divisible by 32, N matches weight N | `None` | controls compute-kernel `has_bias` CT flag |

## Tensors

### Input tensor

| Property | Requirement |
|----------|-------------|
| Shape | `[1, 1, M, K]` |
| Dtype | `bfloat16` |
| Layout | `TILE_LAYOUT` |
| Memory | DRAM interleaved |
| Alignment | M, K divisible by 32 |

### Weight tensor

| Property | Requirement |
|----------|-------------|
| Shape | `[1, 1, K, N]` |
| Dtype | `bfloat16` |
| Layout | `TILE_LAYOUT` |
| Memory | DRAM interleaved |
| Alignment | K, N divisible by 32; K matches input K |

### Bias tensor (optional)

| Property | Requirement |
|----------|-------------|
| Shape | `[1, 1, 32, N]` (logical 1-row vector tile-padded to a single tile-row) |
| Dtype | `bfloat16` |
| Layout | `TILE_LAYOUT` |
| Memory | DRAM interleaved |
| Alignment | N divisible by 32; N matches weight N |
| Value layout | Row 0 of every tile holds the bias value; rows 1–31 zero-padded |

### Output

| Property | Value |
|----------|-------|
| Shape | `[1, 1, M, N]` |
| Dtype | `bfloat16` |
| Layout | `TILE_LAYOUT` |
| Memory | DRAM interleaved (default `ttnn.DRAM_MEMORY_CONFIG`) |

## Tile counts

All derived host-side from input shapes; carried into the program descriptor as compile-time args.

| Symbol | Formula | Meaning |
|--------|---------|---------|
| `Mt` | `M / 32` | input/output tile-rows |
| `Kt` | `K / 32` | inner-dim tile count |
| `Nt` | `N / 32` | output/weight tile-cols |

## Python-side validation

The `linear()` Python entry point must perform every check below before allocating the output tensor or building the descriptor. Failures raise `ValueError`. Validation is exhaustive — no tensor is allowed to reach `ttnn.generic_op` if any check would fail.

| # | Check | Applies to | Error |
|---|-------|------------|-------|
| 1 | `dtype == ttnn.bfloat16` | input, weight, bias (if present) | `ValueError("linear: <name> must be bfloat16")` |
| 2 | `layout == ttnn.TILE_LAYOUT` | input, weight, bias | `ValueError("linear: <name> must be TILE_LAYOUT")` |
| 3 | `len(shape) == 4` | input, weight, bias | `ValueError("linear: <name> must be rank 4")` |
| 4 | leading two dims == 1 | input, weight, bias | `ValueError("linear: <name> leading dims must be [1, 1, ...]")` |
| 5 | `input.shape[-1] == weight.shape[-2]` (K match) | input, weight | `ValueError("linear: input K (...) does not match weight K (...)")` |
| 6 | `bias.shape[-1] == weight.shape[-1]` (N match) | bias if present | `ValueError("linear: bias N (...) does not match weight N (...)")` |
| 7 | `bias.shape[-2] == 32` | bias if present | `ValueError("linear: bias height must be 32 (tile-padded single row)")` |
| 8 | M, K, N each divisible by 32 | derived from shapes | `ValueError("linear: M/K/N must be divisible by 32")` |

## Dataflow Strategy

Single-core, single-Tensix. No multicast, no inter-Tensix synchronization.

| Stage | Where | Format | Notes |
|-------|-------|--------|-------|
| Source | DRAM (interleaved) | tile pages | input, weight, optional bias |
| Reader → CB | NCRISC (NoC0) into L1 CBs | tile pages | `cb_input_tiles`, `cb_weight_tiles`, `cb_bias_tiles` (only if bias) |
| Compute | TRISCs (unpack/math/pack) | tile pages | matmul → optional bias add → output CB |
| Writer ← CB | BRISC (NoC1) from L1 CB | tile pages | `cb_output_tiles` → DRAM |
| Sink | DRAM (interleaved) | tile pages | output |

Reader scans all input pages in row-major tile order, all weight pages in row-major tile order, and (if bias) all bias pages in column-tile order — once each. Writer drains the output CB in row-major tile order.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | The entire `[Mt, Nt]` output tile grid. Phase 0 is single-core. |
| Grid | `ttnn.CoreRangeSet([CoreRange((0, 0), (0, 0))])` — single Tensix core (0, 0). |
| Per-core work | All `Mt * Nt` output tiles, all `Mt * Kt` input tiles, all `Kt * Nt` weight tiles, all `Nt` bias tiles. |
| Remainder | None — single core handles everything; tile alignment guaranteed by Python validation. |

## Matmul block parameterization

The `matmul_block` helper requires a `MatmulBlockShape`. Phase 0 uses the simplest valid configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `in0_num_subblocks` | `Mt` | One M-tile per subblock keeps DEST usage at 1 tile/subblock (well under the 8-tile bf16 limit). |
| `in1_num_subblocks` | `Nt` | One N-tile per subblock, same reasoning. |
| `out_subblock_h` | `1` | Trivial subblock geometry; output emerges in row-major tile order. |
| `out_subblock_w` | `1` | Same. |
| `in0_block_w` | `Kt` | Single K-block covers the entire inner dim — no spill/reload needed. |
| `num_k_blocks` | `1` | No K-blocking in Phase 0; refinement Phases will add it. |
| `batch` | `1` | No batching in Phase 0 (leading dims are `[1, 1, ...]`). |

Per-K-block tile counts the helper waits/pops on (verified against `matmul_block_helpers.inl:206-207`, `:450-464`):

- `in0_block_num_tiles = in0_num_subblocks * out_subblock_h * in0_block_w = Mt * 1 * Kt = Mt*Kt`
- `in1_block_num_tiles = in1_num_subblocks * out_subblock_w * in0_block_w = Nt * 1 * Kt = Kt*Nt`
- `out_block_num_tiles = in0_num_subblocks * in1_num_subblocks * out_subblock_h * out_subblock_w = Mt * Nt`

With `num_k_blocks = 1`, the helper waits/pops on each input CB exactly once (the single K-block IS the last K-block).

## Pack target selection

Selected at compile time by the host-side `has_bias` CT flag.

| Mode | `LastBlockTarget` | matmul packs to | Final writer reads from |
|------|------------------|-----------------|--------------------------|
| No bias | `LastBlockTarget::Out` | `cb_output_tiles` directly | `cb_output_tiles` |
| With bias | `LastBlockTarget::Interm` | `cb_partials` (intermediate) | `cb_output_tiles` (after `add_bias_bcast_rows` consumes partials) |

`OutputLayout::SubblockMajor` (default) is used in both modes. With `out_subblock_h = out_subblock_w = 1`, subblock-major iteration `(in0_sb, in1_sb)` for `in0_sb=0..Mt-1, in1_sb=0..Nt-1` produces tiles in row-major order — the writer reads `tile(m, n)` at linear page `m*Nt + n`, matching the natural DRAM-interleaved output tile ordering. No `RowMajor` layout override required.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_tiles` | 0 | `tile_size(bf16)` (= 2048 B) | `Mt * Kt` | bf16 | Reader (NCRISC) | Compute (matmul_block in0) | Filled once at start; popped fully when the single K-block finishes. |
| `cb_weight_tiles` | 1 | `tile_size(bf16)` | `Kt * Nt` | bf16 | Reader (NCRISC) | Compute (matmul_block in1) | Filled once at start; popped fully when the single K-block finishes. |
| `cb_bias_tiles` | 2 | `tile_size(bf16)` | `Nt` (only when bias present, else descriptor omitted) | bf16 | Reader (NCRISC) | Compute (`add_bias_bcast_rows`) | Filled once at start; compute waits once at start of bias phase, pops once at end (caller-managed lifecycle per `bias_add_helpers.hpp:93-97`). |
| `cb_partials` | 24 | `tile_size(bf16)` | `Mt * Nt` (only when bias present, else descriptor omitted) | bf16 | Compute (matmul_block, `LastBlockTarget::Interm`) | Compute (`add_bias_bcast_rows`) | Sized to FULL block of `Mt*Nt` tiles — sequential helpers don't pipeline (they each own all 3 TRISCs), so the entire matmul output must land before `add_bias_bcast_rows` starts consuming. |
| `cb_output_tiles` | 16 | `tile_size(bf16)` | `2` (double-buffered) | bf16 | Compute (matmul_block in no-bias mode, OR `add_bias_bcast_rows` in bias mode) | Writer (BRISC) | Streaming. Producer reserves 1 tile per subblock (= 1 tile/iter for `1×1` subblocks); writer drains in the same order. |

Page-size rationale: every CB carries TILE-layout bf16 tiles. `tile_size(bf16) = 32 * 32 * 2 = 2048 B`. All num_pages counts are in tile units.

CB sync verification (push count vs wait count):

| CB | Reader push | Compute wait | Compute pop | Compute push | Other consumer wait | Other consumer pop |
|----|------------|--------------|-------------|--------------|---------------------|---------------------|
| `cb_input_tiles` | `Mt*Kt` (one shot) | `Mt*Kt` (helper, single K-block) | `Mt*Kt` (helper, last block) | — | — | — |
| `cb_weight_tiles` | `Kt*Nt` (one shot) | `Kt*Nt` (helper, single K-block) | `Kt*Nt` (helper, last block) | — | — | — |
| `cb_bias_tiles` (bias mode) | `Nt` (one shot) | `Nt` (caller wait_front before bias helper) | `Nt` (caller pop_front after bias helper) | — | — | — |
| `cb_partials` (bias mode) | — | `Mt*Nt` (split as 1 tile/subblock-iter inside `add_bias_bcast_rows`) | `Mt*Nt` (1 tile/subblock-iter) | `Mt*Nt` (1 tile/subblock-iter from matmul_block) | — | — |
| `cb_output_tiles` | — | — | — | `Mt*Nt` total (1 tile/subblock-iter) | `Mt*Nt` (writer waits 1 tile/iter) | `Mt*Nt` (writer pops 1 tile/iter) |

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|------------------------|--------------------------|---------------------------|--------------|
| 0. Compute startup | raw_api | `compute_kernel_hw_startup` | `tt_metal/include/compute_kernel_api/common.h` (canonical entry) | called once with at least the matmul input/output CBs (`cb_input_tiles, cb_weight_tiles`, and the matmul pack target — `cb_output_tiles` no-bias, `cb_partials` bias) | n/a | n/a | Mandatory once-per-kernel init; all helpers require it (`binary_op_helpers.hpp:13-19` — same prerequisite applies to matmul/bias helpers). |
| 1. Matmul (no bias) | helper | `compute_kernel_lib::matmul_block` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:298-319` | `transpose=false, packer_l1_acc=false, last_block_target=LastBlockTarget::Out, layout=OutputLayout::SubblockMajor, init_mode=InitMode::Full, retain_in0=false, retain_in1=false`. Buffers: `in0_buf=cb_input_tiles, in1_buf=cb_weight_tiles, out_buf=cb_output_tiles, interm_buf=cb_output_tiles` (interm unused when `num_k_blocks==1`, see hpp:206-207). Shape: `MatmulBlockShape::of(Mt, Nt, 1, 1, Kt, 1)`. | `cb_input_tiles` (`Mt*Kt` tiles), `cb_weight_tiles` (`Kt*Nt` tiles) | `cb_output_tiles` (`Mt*Nt` tiles, 1 per subblock) | Helper owns wait/pop on inputs (`matmul_block_helpers.inl:206-207, :450-464`) and reserve/push on output (subblock-major pack). With `init_mode=Full` it also issues `mm_block_init` itself. |
| 1. Matmul (with bias) | helper | `compute_kernel_lib::matmul_block` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:298-319` | Same as above except `last_block_target=LastBlockTarget::Interm` (last block packs to `cb_partials` for the bias phase to consume — see hpp:265-278 example). Buffers: `in0_buf=cb_input_tiles, in1_buf=cb_weight_tiles, out_buf=cb_output_tiles` (unused on the last/only block since target=Interm), `interm_buf=cb_partials`. Shape unchanged. | `cb_input_tiles` (`Mt*Kt`), `cb_weight_tiles` (`Kt*Nt`) | `cb_partials` (`Mt*Nt` tiles in subblock-major order) | Same helper-managed lifecycle. |
| 2. Bias add (with bias only) | helper | `compute_kernel_lib::add_bias_bcast_rows` | `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp:168-179` | `broadcast=BiasBroadcast::RowBroadcast, output_layout=OutputLayout::SubblockMajor, PostBiasFn=NoPostBias`. Buffers: `partials_buf=cb_partials, bias_buf=cb_bias_tiles, out_buf=cb_output_tiles`. Shape: `BiasAddShape::of(Mt, Nt, 1, 1)` (out_row_width default 0 — derived from `out_subblock_w * in1_num_subblocks = Nt`, only consulted under `RowMajor` layout). `bias_offset=0`. | `cb_partials` (helper waits/pops `Mt*Nt`), `cb_bias_tiles` (caller-waited, helper indexes — see hpp:93-97, :115-122) | `cb_output_tiles` (`Mt*Nt` tiles, 1 per subblock) | Caller MUST `cb_wait_front(cb_bias_tiles, Nt)` before this call and `cb_pop_front(cb_bias_tiles, Nt)` after. Helper consumes partials and reserves/pushes output per subblock. Layout must match upstream `matmul_block` (both `SubblockMajor`). |

Every compute phase is helper-mapped. No raw-API fallback in the compute kernel beyond the mandatory `compute_kernel_hw_startup`.

## Reader CB push contract

(Implementer derives kernel arguments and exact `noc_async_read` calls from this table — these are the producer-side guarantees that make the CB sync table above hold.)

| CB | Total tiles pushed per program | Tile order pushed | Source page index for the i-th tile |
|----|--------------------------------|-------------------|--------------------------------------|
| `cb_input_tiles` | `Mt * Kt` | Tile-row-major: `(m=0,k=0), (m=0,k=1), …, (m=0,k=Kt-1), (m=1,k=0), …` | Page `m * Kt + k` of input DRAM buffer (natural DRAM-interleaved tile linear index for `[Mt, Kt]` tile grid). |
| `cb_weight_tiles` | `Kt * Nt` | Tile-row-major: `(k=0,n=0), (k=0,n=1), …, (k=0,n=Nt-1), (k=1,n=0), …` | Page `k * Nt + n` of weight DRAM buffer. |
| `cb_bias_tiles` (bias mode only) | `Nt` | Column order: `n=0..Nt-1` | Page `n` of bias DRAM buffer (bias is `[1, 1, 32, N]` = a single tile-row of `Nt` tiles). |

Reader push grouping is at the implementer's discretion (single bulk `cb_reserve_back / cb_push_back` after all reads complete is simplest for Phase 0; per-tile streaming is also valid as long as the totals match the wait counts). Bulk push aligns with the matmul helper's per-K-block `wait_front`.

## Writer CB pop contract

| CB | Tiles popped per program | Tile order popped | Destination page index |
|----|--------------------------|-------------------|------------------------|
| `cb_output_tiles` | `Mt * Nt` | Helper packs in order `(in0_sb=0..Mt-1, in1_sb=0..Nt-1)` with `out_subblock_h=out_subblock_w=1`, which equals tile-row-major `(m, n)` order | Page `m * Nt + n` of output DRAM buffer. |

Writer waits/pops one tile at a time (matches the helper's per-subblock `cb_reserve_back(out_buf, 1)` / `cb_push_back(out_buf, 1)` cadence under `SubblockMajor` layout with `1×1` subblocks).

## Compute Phases

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `compute_kernel_hw_startup(cb_input_tiles, cb_weight_tiles, <pack_target>)` | raw_api | n/a | n/a | n/a |
| 1a (no-bias) | `matmul_block<…, LastBlockTarget::Out, …>(in0=cb_input_tiles, in1=cb_weight_tiles, out=cb_output_tiles, interm=cb_output_tiles, MatmulBlockShape::of(Mt, Nt, 1, 1, Kt, 1))` | helper | `cb_input_tiles` (`Mt*Kt` tiles, fully populated by reader) → fully popped; `cb_weight_tiles` (`Kt*Nt`, fully populated) → fully popped | `cb_output_tiles` (`Mt*Nt` tiles streamed, 1 push per subblock) | After: input/weight CBs empty; `cb_output_tiles` is being drained by writer concurrently. End of compute kernel. |
| 1b (bias) | `matmul_block<…, LastBlockTarget::Interm, …>(in0=cb_input_tiles, in1=cb_weight_tiles, out=cb_output_tiles, interm=cb_partials, MatmulBlockShape::of(Mt, Nt, 1, 1, Kt, 1))` | helper | `cb_input_tiles` (`Mt*Kt`, populated) → fully popped; `cb_weight_tiles` (`Kt*Nt`, populated) → fully popped | `cb_partials` (`Mt*Nt` tiles streamed, 1 push per subblock) | After: input/weight empty; `cb_partials` holds all `Mt*Nt` tiles (sized to full block — phase 2 will drain it). `cb_bias_tiles` was untouched. |
| 2 (bias only) | `cb_wait_front(cb_bias_tiles, Nt)` | raw_api | `cb_bias_tiles` (Nt tiles, populated by reader) | n/a | `cb_bias_tiles` fronted with Nt tiles (no pop yet). |
| 3 (bias only) | `add_bias_bcast_rows<RowBroadcast, SubblockMajor>(partials=cb_partials, bias=cb_bias_tiles, out=cb_output_tiles, BiasAddShape::of(Mt, Nt, 1, 1))` | helper | `cb_partials` (`Mt*Nt` tiles fronted by phase 1b) → helper pops fully; `cb_bias_tiles` (Nt tiles fronted by phase 2, indexed only — caller still owns pop) | `cb_output_tiles` (`Mt*Nt` tiles streamed, 1 push per subblock) | After: `cb_partials` empty; `cb_bias_tiles` still fronted (helper does not pop); `cb_output_tiles` being drained by writer. |
| 4 (bias only) | `cb_pop_front(cb_bias_tiles, Nt)` | raw_api | `cb_bias_tiles` (Nt tiles fronted) | n/a | `cb_bias_tiles` empty. End of compute kernel. |

## Build Order

Incremental bring-up the implementer should follow:

| Step | What runs | Verification |
|------|-----------|--------------|
| 1 | Skeleton: reader pushes `cb_input_tiles` only, compute waits + immediately pops, writer is a no-op (zeroed output buffer). | Run with `tt-probe.sh --dev linear` on a 32×32×32 case. Confirm no hang. Output is all zeros. |
| 2 | Add weight read → `cb_weight_tiles`, compute pops both, writer streams a constant pattern (e.g. `pack_tile` of a known DST tile). | Output = constant pattern in row-major tile order. Sanity-check the writer's DRAM addressing. |
| 3 | Wire up `compute_kernel_hw_startup` + `matmul_block` no-bias path (`LastBlockTarget::Out`). Use deterministic input `torch.eye(M, K)` and weight `torch.eye(K, N)` so the expected output is identity. DPRINT a corner tile from compute on the last subblock to confirm. | Output ≈ identity slice for `M==K==N` shapes. PCC vs PyTorch matmul reference > 0.99. |
| 4 | Add the bias path: introduce `cb_bias_tiles` + `cb_partials` descriptors gated on `has_bias`; switch matmul template to `LastBlockTarget::Interm` and chain `add_bias_bcast_rows`. Verify with bias filled with a known scalar (e.g. 1.0 broadcast). | Output of `linear(zeros, zeros, bias=ones)` ≈ all 1s in row 0, then 1s in every row (bias broadcasts down). PCC vs PyTorch `x @ W + b[0]` > 0.99 on randn inputs. |
| 5 | Strip DPRINTs, confirm full acceptance test passes. | All cases in `tests/ttnn/unit_tests/operations/linear/test_linear.py` pass under `scripts/run_safe_pytest.sh`. |

DPRINT hint: under helper code, a `UNPACK(DPRINT << "k=" << k_block_idx ENDL());` inside any per-K-block hook (or simply at the top of compute before `matmul_block`) is enough to confirm the kernel is actually running. For tile-value sanity, add a tile-corner DPRINT in the writer between `cb_wait_front` and `noc_async_write`.

## Key Risks and Gotchas

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | `cb_partials` undersized → matmul fills it, no consumer yet, helper hangs on `cb_reserve_back`. | Sized to FULL `Mt * Nt` tiles in the descriptor. Documented in CB table. Sequential helpers cannot pipeline (`.claude/references/ttnn-cb-memory-fundamentals.md` "Intermediate CB Sizing"). |
| 2 | Bias CB lifecycle: `add_bias_bcast_rows` does NOT call `cb_wait_front` / `cb_pop_front` on `cb_bias_tiles` (`bias_add_helpers.hpp:93-97`). Forgetting the caller-side wait causes silent garbage reads; forgetting the pop is harmless for Phase 0 (program exits) but is sloppy. | Compute kernel issues `cb_wait_front(cb_bias_tiles, Nt)` before `add_bias_bcast_rows` and `cb_pop_front(cb_bias_tiles, Nt)` after. Documented in Compute Phases table (phases 2 and 4). |
| 3 | Bias tile expected layout: `RowBroadcast` mode uses `add_tiles_bcast_rows`, which broadcasts row 0 of the bias tile across all rows of the partial tile. The user's `[1, 1, 32, N]` tile-padded format puts bias values in row 0 with rows 1–31 zero — this matches exactly. | Validated explicitly in Python (check 7: `bias.shape[-2] == 32`). The 32 here is enforced because that's how `from_torch` of a `[1, 1, 1, N]` reference would be tile-padded; the user's contract makes the padding explicit. |
| 4 | `interm_buf` argument to `matmul_block` in the no-bias path. When `num_k_blocks == 1` the helper never reads from `interm_buf` (per hpp:204-206), but the function signature still requires a valid buffer reference. | Pass `cb_output_tiles` itself in that arg slot — same buffer type, guaranteed not read. Documented in API Mapping. |
| 5 | DEST register limit. With `out_subblock_h=out_subblock_w=1` each subblock holds 1 tile in DEST — far below the 8-tile bf16 limit (`dest_helpers.hpp` `DEST_AUTO_LIMIT`). No mitigation needed; constraint trivially satisfied. | n/a |
| 6 | Auto-discovery of the Python package: `ttnn/ttnn/operations/__init__.py` walks subpackages, so creating `linear/__init__.py` exposing `linear` is sufficient — no manual registration in any other file. | The implementer must create `linear/__init__.py` (a single `from .linear import linear` line, mirroring `toy_binary_in_place/__init__.py`). |
| 7 | `compute_kernel_hw_startup` must be called exactly once at the start of the compute kernel — never inside any loop or conditional that may execute multiple times. (`binary_op_helpers.hpp:13-19`) | Phase 0 runs the matmul exactly once per program, so the call sits at the top of `kernel_main`, before any helper. |
| 8 | Matching `OutputLayout` between `matmul_block` and `add_bias_bcast_rows`. Both must agree — otherwise the bias helper reads `cb_partials` in the wrong order. (`bias_add_helpers.hpp:81-90`) | Both fixed to `OutputLayout::SubblockMajor` in Phase 0. Documented in API Mapping. |

## File layout the implementer must produce

The design fully constrains the file tree the implementer creates. Listed here so nothing is left implicit.

```
ttnn/ttnn/operations/linear/
  __init__.py                       # exposes `linear`
  linear.py                         # Python entry point + validation
  linear_program_descriptor.py      # ProgramDescriptor builder
  kernels/
    reader.cpp                      # NCRISC: pushes input/weight (and bias if has_bias)
    compute.cpp                     # TRISC: compute_kernel_hw_startup → matmul_block → [add_bias_bcast_rows]
    writer.cpp                      # BRISC: drains cb_output_tiles to DRAM

tests/ttnn/unit_tests/operations/linear/
  test_linear.py                    # acceptance test (provided alongside this design)
```
