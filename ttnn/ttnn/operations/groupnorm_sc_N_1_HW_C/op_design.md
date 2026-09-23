# groupnorm_sc_N_1_HW_C — design note

GroupNorm over an `(N, 1, HW, C)` channel-last tensor, one device program (Python `generic_op`).

```
mean[n,g] = Σ_{h<HW, c∈g} x[n,h,c] / (HW·Cg)
var[n,g]  = Σ_{h<HW, c∈g} (x[n,h,c] − mean[n,g])² / (HW·Cg)          (centered, never E[x²] − mean²)
y[n,h,c]  = (x[n,h,c] − mean[n,g(c)]) · rsqrt(var[n,g(c)] + eps) · gamma[c] + beta[c]
Cg = C / G,  g(c) = floor(c / Cg)
```

Optionally `activation="silu"` applies SiLU in the apply pass.

## Supported inputs

| Item | Values |
|------|--------|
| dtype | bf16, fp32, bf8b (bf8b TILE only); every intermediate CB is fp32 |
| layout | TILE, ROW_MAJOR |
| placement | DRAM / L1 interleaved; L1 BLOCK_SHARDED (ROW_MAJOR or COL_MAJOR orientation, rectangle at (0,0)) |
| `in_place` | BLOCK_SHARDED only: the result overwrites the input shard |
| extents | any `N`, `HW`, `C`; `C % G == 0`; `ceil(G/32) ≤ MAX_GROUP_TILES` |
| gamma / beta | `(1,1,1,C)`, bf16 / fp32 / bf8b, TILE or ROW_MAJOR, same dtype/layout/placement for both |
| compute config | `fp32_dest_acc_en=True` required, `packer_l1_acc=False`; fidelity / approx / dst sync pass through |

## Algorithm

Groups may straddle 32-channel tile boundaries, so the op never reduces over channel lanes with a
tile reduce. Instead:

1. **Column sums.** Each core reduces its block down the rows (`reduce<SUM, REDUCE_COL>` with
   accumulation across chunks) into per-channel row-0 tiles.
2. **Membership matmul.** A 0/1 membership matrix `M` (K × Ng fp32 tiles, `M[c][g] = 1` iff channel
   `c` is in group `g`) is built on-device by the writer. `colsums @ M` gives the core's per-group
   partial sums. The writer also builds `Mᵀ` so that every matmul in the op — aggregation, root
   combine and expansion — is the same non-transposed `(1 × K') @ (K' × N)` body (one instantiation
   in the compute binary).
3. **Gather + multicast.** Every core unicasts its partial row into row `core_idx` of the root's
   `cb_gather` and bumps a monotone round counter on the root. The root waits for `num_active` rows,
   computes `inv_rows @ gather` (row 0 of `inv_rows` = 1/(HW·Cg), so the product is the mean / variance
   directly) and multicasts the Ng stat tiles to every core (`mcast_pipe` SenderPipe / ReceiverPipe,
   one sender per round). Two rounds per image: mean, then variance. One landing region suffices:
   a core sends round-r rows only after it received round r−1's broadcast, which the root sends only
   after its compute consumed the previous gather.
4. **Expansion.** `stat @ Mᵀ` expands a group statistic back to per-channel rows.
   Pass 2 subtracts the expanded mean and squares; pass 3 forms `rstd = rsqrt(var + eps)` on the Ng
   group tiles (SFPU cost scales with Ng, not K), expands it, multiplies by gamma, forms
   `shift = beta − mean·scale`, and applies `y = x·scale + shift` (fused FPU chain, DEST reuse).

### Block schedule (per image)

| Pass | Work |
|------|------|
| 1 | stage chunk → fp32 scratch; column sums; `colsums @ M` → partials; combine round 0 → group mean |
| 2 | expand mean; `(x − mean)²`; column sums; `@ M` → partials; combine round 1 → group variance |
| 3 | rstd, scale, shift; apply chunk → output (untilize for ROW_MAJOR) |

`two_pass` (interleaved streaming without `hw_mask`): passes 1 and 2 read the input once. Per chunk the
kernel accumulates `S = Σx` and `U = Σ(x − s)²` against a per-channel shift `s` (column means of the
first tile-row, via the same matmul with a 1/32 row tile); after round 0 it forms
`Σ(x − m)² = U + 2(s − m)(S − (n/2)(m + s))` per channel. Every large term is a sum of squares of
centered values, so there is no catastrophic cancellation. Input traffic drops from 4 to 3 tensor volumes.

## Work split and regimes

The grid is split `hw_splits × c_splits`: each core owns tile-rows `[r0, r0 + H_core)` of every image
and channel tiles `[t0, t0 + K)`, `K = Ct / c_splits` uniform (tilize/untilize take K as a template
argument). Images are an outer loop; every core participates in every image.

| Regime | When | Input traffic |
|--------|------|---------------|
| `resident_2d` | interleaved, the per-core block fits L1 next to the fixed CBs | read once |
| `streaming_2d` | interleaved, block does not fit | read 2× (`two_pass`) or 3× (`hw_mask`), in Q-tile-row chunks |
| `block_sharded_resident` | BLOCK_SHARDED: the shard is the block; split read off the shard spec | none (L1 → L1) |

**Split search** (interleaved): over the divisors `c_splits` of `Ct` with `K ≤ MAX_CORE_C_TILES` and a
fitting fixed footprint, minimize max tiles per core. TILE prefers an L1-resident split (halving the
chunk Q if that makes it fit); ROW_MAJOR prefers a wider K (stick slices are K·64 B) up to
`SPLIT_RM_STICK_K_TARGET`, within `SPLIT_COST_TOLERANCE_RM`. Ties: more `hw_splits`, then fewer cores.

**Block-sharded placement.** TILE shards back `cb_input_tiles` / `cb_output_tiles` directly
(zero-copy; `in_place` aliases the output index onto the input shard). ROW_MAJOR shards use the
**direct view** when exact: a `[rows, w]` shard with stick page `w·elem` bytes is byte-identical to a
row-major block of width `lcm(w, 32)`, so the tilize reads the shard in place and the untilize packs
straight into the output shard; block lane `j` is channel `c0 + j % w` (`c_period`), which only the
membership build and the affine-row fill see. Otherwise the reader stages sticks L1 → L1 into
`cb_input_sticks` and the writer copies the valid bytes back. Because the shards share L1 with the
CBs, the CB budget is `min(L1_CB_BUDGET_BYTES, lowest shard address − allocator base − margin)`.
A core whose shard misses image n sends a zero partial row and drains the broadcast stats.

## Non-aligned extents

* `C % 32 != 0` or shard width < K·32: per-core valid-lane pair `(c0, c_valid)`; membership rows and
  affine lanes beyond `c_valid` are zero; ROW_MAJOR reads / writes `c_valid·elem` bytes per stick
  into once-zeroed stick CBs.
* Partial tile-rows (`HW % 32 != 0`, RM shard heights not multiple of 32, N > 1 shards straddling
  images): `hw_mask` programs run pass 2 in `[head][body][tail]` segments. The writer copies the landed
  group-mean tiles with rows outside the image zeroed (`cb_masked_mean`); the expansion matmul turns
  them into full mean tiles, so the zero pad sticks give `(0 − 0)² = 0` with the same single chain.
  Counts always use the true HW. `geometry.hpp` is the shared per-image geometry for all three kernels.

## Circular buffers

All intermediates are fp32 (`T4` = 4096 B). `K` = channel tiles per core, `Q` = chunk tile-rows,
`D` = `STREAM_DEPTH`, `GT = ceil(num_active / 32)`.

| CB | Pages | Role |
|----|-------|------|
| `cb_input_tiles` (0) | resident: `Hmax·K`; streaming: `D·Q·K`; TILE shard: the shard | input tiles |
| `cb_input_sticks` (1) | `D·K` (RM direct view: the shard) | RM tilize source |
| `cb_scaler` (2) | 1 bf16 | reduce helper template contract (never read) |
| `cb_membership` / `cb_membership_t` (3 / 19) | `K·Ng` each | M and Mᵀ |
| `cb_gamma_rows` / `cb_beta_rows` (4 / 5) | `K` | row-0-valid affine tiles |
| `cb_inv_rows` (6) | `GT` | combine matmul in0 (row 0 = 1/n), root only |
| `cb_colsum_rows` (7) | `K` (`2K` with `two_pass`: S then U) | column sums / accumulator |
| `cb_partial_rows` (8) | `Ng` | per-core group partials |
| `cb_gather` (9) | `Ng·GT` | root landing region for partial rows |
| `cb_group_mean` / `cb_group_var` (10 / 11) | `Ng` | broadcast landing |
| `cb_mean_rows`, `cb_scale_rows`, `cb_shift_full` (12–14) | `K` | expanded per-channel operands |
| `cb_fp32_scratch` (15) | `Q·K` | staged / squared chunk |
| `cb_output_tiles` (16) | `D·Q·K`; TILE shard: the output shard | output tiles |
| `cb_output_sticks` (17) | `D·K` (RM direct view: the output shard) | RM untilize destination |
| `cb_stats_bcast` (18) | `Ng` | root combine result (multicast source) |
| `cb_group_rstd` (20) | `Ng` | rsqrt(var + eps) |
| `cb_masked_mean` (21) | `2·Ng` | `hw_mask` programs only |
| `cb_zero_row` (22) | 128 B | sharded: zero partial row |
| `cb_input_shard` / `cb_output_shard` (23 / 24) | the shards | RM staged path |
| `cb_inv32_row` (25) | 1 | `two_pass`: shift matmul in0 |

Semaphores: two monotone round counters on the root (mean / variance, target `(image+1)·num_active`,
never reset) plus the mcast_pipe ready / consumed pair.

## Parameters (`config.py`)

Read by attribute at call time so tests can monkeypatch them: `L1_CB_BUDGET_BYTES`,
`CHUNK_TILES_TARGET`, `MAX_CORE_C_TILES`, `MAX_GROUP_TILES`, `STREAM_DEPTH`,
`SPLIT_COST_TOLERANCE_TILE` / `_RM`, `SPLIT_RM_STICK_K_TARGET`, `L1_CB_SAFETY_MARGIN_BYTES`,
`STREAMING_TWO_PASS`, and the test knobs `FORCE_C_SPLITS`, `FORCE_STREAMING`.

## Precision

Stat CBs are fp32 and DEST is fp32; FPU operands are tf32-rounded, so the mean carries a relative error
of about 2⁻¹¹ per rounding — below bf16 input quantization (2⁻⁸). Stat CBs must never be bf16.

## Code-size constraint

The three kernels must fit the kernel-config ring in the `--dev` (watcher) build. This is why there
is one matmul body, one pass-2 chain, noinline wrappers around CB pops / (un)tilize / the affine
fill, and why weight / bf8b / `hw_mask` code is compiled only into programs that need it.
