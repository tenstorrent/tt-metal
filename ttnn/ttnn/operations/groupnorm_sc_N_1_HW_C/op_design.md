# Operation Design: groupnorm_sc_N_1_HW_C

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (normalization, two-pass statistics + affine) |
| Goal | Single-core GroupNorm over (N, 1, H·W, C): per (batch, group) normalize over the H·W × (C/G) slab, then optional per-channel affine `* gamma + beta`. |
| Math | `y[n,0,s,c] = (x[n,0,s,c] − mean(n,g)) * rstd(n,g) * gamma[c] + beta[c]`, with `g = c // (C/G)`, `mean(n,g) = E[x]` over the H·W × Cg slab, `rstd(n,g) = 1/sqrt(E[(x−mean)²] + eps)` |
| Mode | Derivative of `ttnn/ttnn/operations/toy_variance` (streaming mean/centered-variance pattern), generalized to per-group SCALAR reduces + affine |
| References | `ttnn/ttnn/operations/toy_variance/kernels/compute.cpp`, `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp`, `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_*.hpp`, `eval/op_template.py`, `eval/golden_tests/groupnorm_sc_N_1_HW_C/feature_spec.py` |

Derived quantities used everywhere below: `Cg = C / num_groups`, `Wg = Cg / 32` (tiles per group along C), `Wt = C / 32` (total C tiles), `Ht = HW / 32` (tiles along H·W), `N_grp = HW * Cg` (elements per group).

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| num_groups | int | yes | ≥1, divides C; Phase-0: `Cg % 32 == 0` | — | CT (`Wg`, `inv_sqrt_n_bits` derive from it) |
| gamma | ttnn.Tensor or None | no | shape (1,1,1,C) | None | CT flag `HAS_GAMMA` |
| beta | ttnn.Tensor or None | no | shape (1,1,1,C); requires gamma (beta-only → NotImplementedError) | None | CT flag `HAS_BETA` |
| eps | float | no | > 0 | 1e-5 | CT (`eps_bits` f32 bit-pattern) |

Argument validation (ValueError, before registry gates): rank ≠ 4; dim[1] ≠ 1; `C % num_groups != 0`; gamma/beta shape ≠ (1,1,1,C).
Registry gates (NotImplementedError, via `validate()`): SUPPORTED per-axis then EXCLUSIONS. `(Cg % 32 != 0)` is gated by a `groups_aligned` tagger axis (SUPPORTED = `["aligned"]`) — refinement candidate, NOT structural.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | (N, 1, HW, C) |
| Dtype | bfloat16 (Phase 0); TARGET adds float32, bfloat8_b |
| Layout | TILE or ROW_MAJOR (RM converted host-side via `ttnn.to_layout` → kernel always sees TILE) |
| Memory | DRAM interleaved |

### Gamma / Beta (optional)

| Property | Requirement |
|----------|-------------|
| Shape | (1,1,1,C) → host `to_layout(TILE)` → 1×Wt padded tile row, data in tile row 0, padding rows zero |
| Dtype | bfloat16 (Phase 0) |
| Layout | ROW_MAJOR accepted at API; tilized host-side |

### Output

| Property | Value |
|----------|-------|
| Shape | (N, 1, HW, C) (same as input) |
| Dtype | input dtype (contract: output dtype == input dtype) |
| Layout | TILE_LAYOUT (always, regardless of input layout) |
| Memory | DRAM interleaved |

## Dataflow Strategy

Single Tensix core. No inter-core communication, no multicast, no semaphores.

The output tile grid is N × Ht × Wt. Each group is a column band of `Wg` tile columns; the group slab is `Ht × Wg` tiles. The kernel runs a per-(n,g) group loop; within a group, three streaming passes over the same slab tiles (reader re-reads from DRAM — slab residency in L1 is NOT required, so HW can be arbitrarily large):

```
              ┌─ pass 1 (mean):  reader → cb_input_tiles → reduce<SUM,SCALAR>(scaler 1/√N) → cb_mean (1 tile)
DRAM x ──3×──┤  pass 2 (var):   per tile-row b: sub<Scalar>(x − mean) → cb_centered → square → accumulate_reduce_block → cb_var
              └─ pass 3 (norm):  per tile-row b: sub<Scalar> → mul<Scalar rstd> → [mul<Row gamma>] → [add<Row beta>] → cb_output_tiles → writer → DRAM y
```

Between passes 2 and 3, `transform_in_place` converts var → rstd (add eps, rsqrt) in cb_var. `cb_mean` and `cb_var` persist across the group (HeldBulk operands), popped at group end. gamma/beta tiles (Wt each) are read once at program start and persist for the entire kernel.

All compute is FPU/SFPU on tiles; everything stays TILE format end-to-end (RM handled host-side). Reader is NCRISC, writer is BRISC; writer writes group tiles back via TensorAccessor at index `n·Ht·Wt + r·Wt + g·Wg + c`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one (n, g) group = Ht × Wg tile slab |
| Grid | 1 core (0,0) |
| Per-core work | N · num_groups groups, processed sequentially; per group 3 streaming passes of Ht·Wg tiles |
| Remainder | none (single core); Phase-0 shapes are tile-aligned, no partial tiles |

## Circular Buffers

All tile pages of input dtype unless noted. `pages(g/b)` = present only when gamma/beta given.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| cb_input_tiles | 0 | tile_size(in_dtype) | 2·Wg | in_dtype | reader | compute (pass 1/2/3, pop-per-tile) | per-tile stream, double-buffered row chunk |
| cb_gamma_tiles | 1 | tile_size(in_dtype) | Wt | in_dtype | reader (once) | compute (HeldBulk + base, Row bcast) | whole kernel; popped once at end |
| cb_beta_tiles | 2 | tile_size(in_dtype) | Wt | in_dtype | reader (once) | compute (HeldBulk + base, Row bcast) | whole kernel; popped once at end |
| cb_scaler | 8 | tile_size(bf16) | 1 | bfloat16 | reader (once, 1/√N_grp) | reduce (waits, never pops) | whole kernel; compute pops 1 at end |
| cb_output_tiles | 16 | tile_size(in_dtype) | 2·Wg | in_dtype | compute | writer | streaming, double-buffered |
| cb_mean | 24 | tile_size(in_dtype) | 1 | in_dtype | compute (pass 1) | compute (pass 2/3 HeldBulk) | per group; pop 1 at group end |
| cb_var | 25 | tile_size(in_dtype) | 1 | in_dtype | compute (pass 2 accum / transform) | compute (pass 3 HeldBulk) | per group; pop 1 at group end |
| cb_centered | 26 | tile_size(in_dtype) | 2·Wg | in_dtype | compute (sub) | compute (square / reduce / mul) | per row-chunk intermediate |
| cb_xhat | 27 | tile_size(in_dtype) | 2·Wg | in_dtype | compute (mul rstd) | compute (gamma mul) | per row-chunk; only when HAS_GAMMA |
| cb_scaled | 28 | tile_size(in_dtype) | 2·Wg | in_dtype | compute (gamma mul) | compute (beta add) | per row-chunk; only when HAS_GAMMA && HAS_BETA |

Sizing rationale: stream CBs are 2× one row-chunk (Wg tiles) for reader/compute and compute/writer overlap. cb_mean/cb_var are 1 page — `transform_in_place` pops before reserve (streaming_reduce_helpers.hpp:99–101) and the accumulator is push-1/pop-1 per block. gamma/beta hold the full Wt row because group g indexes tiles at base `g·Wg` (TileOffset::Set requires bulk-resident operand). Worst Phase-0 case (C=4096): gamma+beta 2·128 tiles ·2KB = 512KB + Wg=4 stream CBs ≈ negligible; G=1 C=4096 (Wg=128): 2·128-page stream CBs ×4 + gamma/beta ≈ 0.6MB — fits 1.5MB L1.

## API Mapping

All compute helpers live in `compute_kernel_lib` (alias `ckl`); shapes are `EltwiseShape::grid(...)` (eltwise_chain.hpp:116) / `ReduceInputBlockShape::of(...)` (reduce_helpers_compute.hpp:146). Caller (compute kernel) calls `compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles)` once at boot (reduce_helpers_compute.hpp:29–33, eltwise_chain.hpp:560–562). Helpers own all CB wait/pop/reserve/push edges per their declared lifecycle.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| scaler setup (reader) | helper | `prepare_reduce_scaler` | reduce_helpers_dataflow.hpp:65–67 | `<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_SCALAR>(1/sqrt(N_grp))` | — | cb_scaler | non-standard scaler (combines 1/N with SCALAR double-apply ⇒ 1/√N), exactly the documented use (lines 31–37). Pool-type-aware overload. |
| 1: mean | helper | `reduce<SUM, REDUCE_SCALAR>` | reduce_helpers_compute.hpp:400–415 | input_policy=WaitAndPopPerTile, `of(Ht, Wg)` | cb_input_tiles, cb_scaler | cb_mean | scaler pre-pushed; pops Ht·Wg input tiles; pushes 1 tile, valid at (0,0) |
| 2: center | helper | `sub` | eltwise_convenience.hpp:62–79 | `<cb_input_tiles, cb_mean, cb_centered, BroadcastDim::Scalar, ALife=Streaming, BLife=HeldBulk>` `grid(1, Wg)` | cb_input_tiles, cb_mean | cb_centered | HeldBulk = wait-1-no-pop (eltwise_chain.hpp:177); Scalar kind legal w/ HeldBulk (hpp:282–283) |
| 2: square | helper | `square` | eltwise_convenience.hpp:107–120 | `<cb_centered, cb_centered>` `grid(1, Wg)` | cb_centered | cb_centered | in-place x·x; 2·Wg pages cover pop-before-push window |
| 2: accumulate | helper | `accumulate_reduce_block<SUM, REDUCE_SCALAR>` | streaming_reduce_helpers.hpp:47–61 | `(cb_centered, cb_scaler, cb_var, of(1, Wg), b, Ht)` | cb_centered, cb_scaler | cb_var | per row b; helper owns Accumulate::at reload; caller sizes cb_var ≥1, pops at group end |
| rstd | helper | `transform_in_place` | streaming_reduce_helpers.hpp:110–111 | `(cb_var, λ(d){binop_with_scalar_tile_init(); add_unary_tile(d, EPS_BITS); rsqrt_tile_init(); rsqrt_tile(d);})` | cb_var | cb_var | lambda IS the documented seam for multi-instruction finalizers (lines 76–78, 103–105); raw `add_unary_tile`/`rsqrt_tile` are LLK calls inside it, not standalone raw phases |
| 3a: center | helper | `sub` | eltwise_convenience.hpp:62–79 | same as 2:center | cb_input_tiles, cb_mean | cb_centered | |
| 3b: scale rstd | helper | `mul` | eltwise_convenience.hpp:81–98 | `<cb_centered, cb_var, OUT3b, BroadcastDim::Scalar, Streaming, HeldBulk>` `grid(1, Wg)`; OUT3b = cb_xhat if HAS_GAMMA else cb_output_tiles | cb_centered, cb_var | cb_xhat / cb_output_tiles | |
| 3c: gamma (HAS_GAMMA) | helper | `eltwise_chain` + `BinaryFpu`/`PackTile` | eltwise_chain.hpp:576–577, 500–513, 535–541; ctor eltwise_chain.inl:753–755 | `eltwise_chain(grid(1, Wg), BinaryFpu<cb_xhat, cb_gamma_tiles, Mul, BroadcastDim::Row, Streaming, HeldBulk, Input, D0, OperandKind::Scalar, OperandKind::Row, Unset, Set>{0, g*Wg}, PackTile<OUT3c>{})`; OUT3c = cb_scaled if HAS_BETA else cb_output_tiles | cb_xhat, cb_gamma_tiles | cb_scaled / cb_output_tiles | Row kind+HeldBulk legal (hpp:287); TileOffset::Set legal w/ HeldBulk (hpp:317–320); chain waits g·Wg+Wg, never pops |
| 3d: beta (HAS_BETA) | helper | `eltwise_chain` + `BinaryFpu`/`PackTile` | same | `BinaryFpu<cb_scaled, cb_beta_tiles, Add, Row, Streaming, HeldBulk, Input, D0, Scalar, Row, Unset, Set>{0, g*Wg}` → `PackTile<cb_output_tiles>` | cb_scaled, cb_beta_tiles | cb_output_tiles | |
| reader I/O | raw_api | `noc_async_read_tile` + `TensorAccessor` | tech_reports/tensor_accessor/tensor_accessor.md | tile idx `n·Ht·Wt + r·Wt + g·Wg + c` | DRAM | cb_input_tiles / cb_gamma_tiles / cb_beta_tiles | no dataflow helper covers strided tile gather |
| writer I/O | raw_api | `noc_async_write_tile` + `TensorAccessor` | same | same index formula | cb_output_tiles | DRAM | same justification |

**Helpers considered and rejected**

| Candidate | File:Line | Concrete mismatch |
|-----------|-----------|-------------------|
| `tilize()` / `untilize()` | tilize_helpers.hpp / untilize_helpers.hpp | RM input and RM gamma/beta are converted host-side via `ttnn.to_layout`; output contract is always TILE. Kernel sees TILE only → nothing to (un)tilize. |
| `accumulate_reduce` (full loop) | streaming_reduce_helpers.hpp:79–92 | Pass 2 interleaves per-block sub/square before each reduce; header line 41 directs exactly this case to `accumulate_reduce_block`. Used for pass 2; pass 1 needs no accumulation (single reduce call streams the slab). |
| `calculate_and_prepare_reduce_scaler` | reduce_helpers_dataflow.hpp:94–101 | scaler 1/√(HW·Cg) is non-standard for SUM (standard = 1.0); doc (lines 31–37) mandates `prepare_reduce_scaler` for combined-factor scalers. |
| `unary<Rsqrt<>>` | eltwise_convenience.hpp:126–139, eltwise_math.hpp:36–37 | needs eps-add fused before rsqrt on a 1-page in-place CB; `transform_in_place` (streaming_reduce_helpers.hpp:103–105) is purpose-built for multi-instruction finalizers. |
| `DestReuseBinary` (fuse 3a–3d into one chain) | eltwise_chain.hpp:515–525 | no `BroadcastDim` parameter — DEST-reuse binary is elementwise-only; mean/rstd/gamma/beta operands all require Scalar/Row broadcast → stages must round-trip CBs. |
| `matmul_block` / `bias_add_helpers` / `reblock_untilize` | resp. headers | no matmul phase; bias helper consumes matmul `Interm` partials only (bias_add_helpers.hpp doc), no untilize phase exists. |

## Compute Phases

Per group (n,g); group loop `for n in 0..N-1, for g in 0..G-1` lives in all three kernels.

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|-------------------------|-------------------|----------------|
| 1 | mean = reduce SUM·(1/√N)² over slab | reduce<SUM,SCALAR> | cb_input_tiles (Ht·Wg, pop/tile); cb_scaler (1, no pop) | cb_mean (1) | cb_mean holds 1 tile until phase 6 |
| 2 | per b∈[0,Ht): sub Scalar → square → acc-reduce | sub, square, accumulate_reduce_block | cb_input_tiles (Wg pop); cb_mean (held); cb_centered (Wg push/pop) | cb_var (1, after b=Ht−1) | cb_centered drained; cb_var holds var |
| 3 | var → rstd: +eps, rsqrt | transform_in_place | cb_var (1) | cb_var (1) | cb_var holds rstd until phase 6 |
| 4 | per b: sub Scalar → mul Scalar rstd | sub, mul | cb_input_tiles (Wg pop); cb_mean, cb_var (held) | cb_xhat (or cb_output_tiles if !HAS_GAMMA) | |
| 5 | per b: ·gamma[Row,+g·Wg], +beta[Row,+g·Wg] | eltwise_chain ×2 | cb_xhat (Wg pop); cb_gamma/beta (held, idx g·Wg+c) | cb_output_tiles (Wg) | writer drains |
| 6 | group end | raw cb_pop_front | cb_mean(1), cb_var(1) | — | both empty for next group |
| 7 | kernel end | raw cb_pop_front | cb_scaler(1); cb_gamma/beta (Wt each, if present) | — | all drained |

CB sync: cb_input_tiles 3·Ht·Wg pushes = pops per group ✓; cb_output_tiles Ht·Wg per group ✓; cb_mean push 1 / waits(no-pop) / pop 1 ✓; cb_var push-pop balanced per block + final pop ✓; gamma/beta cumulative HeldBulk waits ≤ Wt resident, single end pop ✓.

## Kernel Argument Sketch (CT)

Implementer derives full layout. Compute CTs: `Ht, Wt, Wg, G, N, HAS_GAMMA, HAS_BETA, eps_bits`. Reader CTs: `Ht, Wt, Wg, G, N, HAS_GAMMA, HAS_BETA, inv_sqrt_n_bits` + TensorAccessorArgs(input[, gamma, beta]) last. Writer CTs: `Ht, Wt, Wg, G, N` + TensorAccessorArgs(output). RT: buffer addresses, one core.

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| 2/4 center | sub | full [H,W] | cb_mean (0,0) only (REDUCE_SCALAR out) | Scalar |
| 4 scale | mul | full | cb_var (0,0) only | Scalar |
| 5 gamma | mul | full | row 0 (host-tilized (1,C)) | Row |
| 5 beta | add | full | row 0 | Row |

## Key Risks and Gotchas

| Risk | Mitigation |
|------|------------|
| Pass 3 cannot run whole-slab per helper call (intermediate CBs would need full slab) | All pass-2/3 helper calls iterate `grid(1, Wg)` per tile row — design mandates row-chunked loops |
| Scaler must be 1/√N (REDUCE_SCALAR applies twice; reduce_helpers_dataflow.hpp:77–78) | `prepare_reduce_scaler<…SUM, REDUCE_SCALAR>(1/sqrt(HW·Cg))`, bf16 CB |
| cb_mean/cb_var must persist across pass 2/3 | HeldBulk wait-no-pop; explicit pops at group end |
| gamma TileOffset base: wait count = g·Wg+Wg grows per group | gamma/beta filled once with Wt tiles; never popped mid-kernel — cumulative waits safe |
| bf16 stat precision over large N_grp (≤4M elem) | acceptable at PCC 0.995; refinement: fp32 dest acc; centered-variance already avoids E[x²]−mean² cancellation |
| Output dtype must equal input dtype | all CBs in input dtype; output tensor allocated in input dtype |
| RM input / RM gamma/beta | host `ttnn.to_layout` before descriptor; output always TILE |
| `(Cg % 32)≠0` (SD regime) | excluded via `groups_aligned` tagger; future: per-group column masking + partial scalers (REDUCE_SCALAR lacks partial-scaler support — refactor stats to ROW+COL reduces) |
| `compute_kernel_hw_startup` once, never in group loop | first statement of compute kernel only |

## Structural impossibilities

`feature_spec.py` INVALID (5 entries) reviewed against this design — complete; no additional op-specific structural cells identified (C % num_groups ≠ 0 is excluded from the test universe by per-shape `num_groups` pairing in INPUTS).
