# Operation Design: rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (normalization: reduce + eltwise scale) |
| Goal | Normalize each row of the last dimension by its root-mean-square, optionally scaled by a per-column `gamma`. Native TILE and ROW_MAJOR, native non-tile-aligned H and W, no host-side layout/padding workarounds. |
| Math | `out[..., h, w] = x[..., h, w] * rsqrt( (1/W) * Σ_{w'=0..W-1} x[..., h, w']² + ε ) * gamma[w]` |
| Mode | Derivative (generic_op / ProgramDescriptor) |
| References | `.claude/references/generic_op_template/`, `.claude/references/precision_convention.md`, `.claude/references/ttnn-cb-memory-fundamentals.md`, `.claude/references/ttnn-python-utility-bindings.md`, `eval/golden_tests/rms_norm/feature_spec.py`, `eval/op_template.py`, `tech_reports/tensor_accessor/tensor_accessor.md`, `ttnn/ttnn/operations/examples/master.md` |

Reduction is over the **last** dimension only. There is no `dim` parameter, so there is no reduce-direction table.

---

## 1. Blocking Model

**This section is decided first; Work Distribution (§5), Circular Buffers (§6) and Compute Phases (§8) are its realization.**

### 1.1 Axes of this op

The input is logically a matrix `(R, W)` where `R = prod(shape[:-1])` rows and `W = shape[-1]` columns.
In tile terms: `ht` tile-rows × `Wt` W-tiles.

| Axis | Character | Block-factor knob | Phase-1 value | Core-assignment | Later unlock |
|------|-----------|-------------------|---------------|-----------------|--------------|
| `ht` — tile-rows (all leading dims × H, collapsed) | **independent** — each row's RMS depends only on that row; no cross-row term anywhere in the math | `HT_BLOCK` (tile-rows per compute block) | `clamp(TILE_BLOCK_BUDGET // WT_CHUNK, 1, ht_total)` — coarsest that fits L1; = 16 for `W=64` bf16 TILE, = 1 when `Wt ≥ TILE_BLOCK_BUDGET` | **Split across the whole grid in phase 1.** `ttnn.split_work_to_cores(core_grid, ht_total, row_wise=True)`; each core owns a disjoint tile-row range and loops `NH_core = ceil(rows_core / HT_BLOCK)` blocks | **knob-turn** (raise `HT_BLOCK`, add cores; also the `HEIGHT_SHARDED` unlock — same logical shard, pinned + pre-placed) |
| `Wt` — W-tiles (the reduced axis) | **dependent** — one output scalar (`Σx²`) spans the whole axis; a partial from part of the axis is not a result | `WT_CHUNK` (W-tiles per compute chunk) | `min(Wt, TILE_BLOCK_BUDGET)` — coarsest that fits L1; = 32 (bf16 TILE), 16 (fp32 TILE / bf16 RM), 8 (fp32 RM) | **Single core in phase 1.** Each core owns the *whole* `W` of its rows; the reduce accumulates sequentially over `NW = ceil(Wt/WT_CHUNK)` chunks (cheap in-core accumulate, no NoC) | **scheme-change** (cross-core partial-sum combine + `1/rms` mcast) — **Lamp L1**; also the `WIDTH_SHARDED` / `BLOCK_SHARDED` unlock |
| `gamma` along `W` | **reuse-shared** — `gamma` is invariant along `ht`, the axis phase-1 splits across cores, so every core re-reads the identical `Wt` tiles (operand-reuse check, §1.4) | `WT_CHUNK` (shared with the input) + `GAMMA_RESIDENT` (read-once-and-hold flag) | `GAMMA_RESIDENT = 1` whenever `Wt · gamma_tile_bytes` fits the residual L1 budget (true up to `W = 4096` bf16); read once per core, held for the whole kernel | Replicated per core (each core reads the full `gamma`) | **scheme-change** (one injector core reads `gamma`, multicasts to the grid) — **Lamp L2** |

**Derived invariant (load-bearing, relied on by §8):** `WT_CHUNK = min(Wt, TILE_BLOCK_BUDGET)` and `HT_BLOCK = clamp(TILE_BLOCK_BUDGET // WT_CHUNK, 1, ht_total)` together guarantee

> `NW > 1  ⟹  HT_BLOCK == 1`

because `NW > 1` forces `WT_CHUNK == TILE_BLOCK_BUDGET`, hence `HT_BLOCK == 1`. Consequently a resident row-block is always a **flat contiguous `Wt`-tile strip** whenever chunk offsets are in play, so `TileOffset::Set(wc * WT_CHUNK)` indexing into it is always correct. The implementer must not break this invariant when re-tuning `TILE_BLOCK_BUDGET`.

### 1.2 The three knob families

**Block factors.** One source of truth: `TILE_BLOCK_BUDGET` (tiles per compute block). Everything else is *derived* from it — `WT_CHUNK`, `HT_BLOCK`, `NW`, `WT_LAST`, every block-scaled CB page count, every loop trip count. Turning the knob is a one-line host change.

```
BLOCK_CB_UNITS      = per (layout, gamma_present)   # §6.3 table — CB pages consumed per block tile
TILE_BLOCK_BUDGET   = pow2_floor( L1_BLOCK_BUDGET_BYTES // (BLOCK_CB_UNITS * tile_bytes) )   # >= 1
WT_CHUNK            = min(Wt, TILE_BLOCK_BUDGET)
HT_BLOCK            = clamp(TILE_BLOCK_BUDGET // WT_CHUNK, 1, ht_total)
NW                  = ceil(Wt / WT_CHUNK)
WT_LAST             = Wt - (NW - 1) * WT_CHUNK
```

`L1_BLOCK_BUDGET_BYTES = 512 * 1024` (phase-1 value, a named host constant). `pow2_floor` keeps chunk widths power-of-two so `WT_LAST == WT_CHUNK` for every power-of-two `W` in the suite (no short-last-chunk instantiation is exercised on the common shapes, but both widths are emitted as CT args so the general case is covered).

Resulting phase-1 block sizes (all **coarse**, none minimal):

| dtype | layout | gamma | `BLOCK_CB_UNITS` | `TILE_BLOCK_BUDGET` |
|---|---|---|---|---|
| bfloat16 | TILE | yes | 8 | 32 |
| bfloat16 | TILE | no | 6 | 32 |
| bfloat16 | ROW_MAJOR | yes | 11 | 16 |
| float32 | TILE | yes | 8 | 16 |
| float32 | ROW_MAJOR | yes | 11 | 8 |

Concrete blocks: `W=64` (`Wt=2`, bf16 TILE) → **16 tile-rows × 2 tiles = 32 tiles per helper call**, not one tile-row at a time. `W=1024` (`Wt=32`) → 1 × 32. `W=32768` (`Wt=1024`) → 1 × 32, `NW=32`.

Catalog evidence for the value: `examples/compute_block_size` measures ~320 ns fixed overhead **per helper pass** and 1.65× from tile-row-at-a-time → whole-block; `examples/double_buffer` measures the read curve flattening at 4–8 tiles per barrier and `block=256` OOMing L1. 32 tiles sits above the read plateau and well above the compute knee while leaving room for the resident buffers.

**Core-assignment.** Phase-1 is multi-core from day 1: the independent `ht` axis is spread over the full compute grid (§5). The grid itself is a parameter (`core_grid` runtime-derived from `device.compute_with_storage_grid_size()`), never an inlined core count.

**Buffer depths.** Per streaming CB, phase-1 minimal = 2 (double buffer), matching `examples/double_buffer` (depth 2 is the whole win; deeper is not measured to help).

| CB | Depth knob | Phase-1 value |
|---|---|---|
| `cb_input_tiles` (streaming regime) | `X_DEPTH` | 2 |
| `cb_input_rm` (RM) | `X_DEPTH` | 2 |
| `cb_output_tiles` (TILE) / `cb_output_rm` (RM) | `OUT_DEPTH` | 2 |
| `cb_gamma`, `cb_gamma_rm` (streaming regime) | `GAMMA_DEPTH` | 2 |
| `cb_x_squared`, `cb_scaled` | fixed at 1 block (sequential-helper intermediate — cannot pipeline) | 1 |
| `cb_scaler` | fixed | 1 page |

### 1.3 Residency fast-path (predicate-guarded, not unconditional)

`x` is needed twice: once to form `Σx²`, once to scale. The safe-at-any-size shape re-reads it (3 traffic units: read, read, write). When the row-block provably fits L1, holding it costs 2 units — a 1.5× traffic win on a DRAM-bound op. This is modelled as a **dual path on an explicit fits-in-L1 predicate**, streaming being the fallback:

```
small_cb_bytes    = cb_scaler + cb_partials + cb_rms_sum + cb_rms_recip            # §6
block_cb_bytes    = all block-scaled CBs at the phase-1 depths, streaming sizes    # §6.3
x_stream_bytes    = (IS_RM ? 1 : X_DEPTH) * HT_BLOCK * WT_CHUNK * tile_bytes       # cb_input_tiles, streaming
gamma_stream_bytes= GAMMA_DEPTH * WT_CHUNK * gamma_tile_bytes                      # cb_gamma, streaming
x_res_bytes       = HT_BLOCK * Wt * tile_bytes
gamma_res_bytes   = Wt * gamma_tile_bytes

base           = block_cb_bytes + small_cb_bytes
X_RESIDENT     = base - x_stream_bytes + x_res_bytes <= L1_CB_BUDGET_BYTES
total_after_x  = X_RESIDENT ? (base - x_stream_bytes + x_res_bytes) : base
GAMMA_RESIDENT = gamma_present and
                 total_after_x - gamma_stream_bytes + gamma_res_bytes <= L1_CB_BUDGET_BYTES
```

`L1_CB_BUDGET_BYTES = 1_100_000` (named host constant; Blackhole/Wormhole worker L1 is 1.5 MB, the remainder is firmware/stack/args headroom). The host **asserts** the final CB total against it; if the assert would fail, halve `L1_BLOCK_BUDGET_BYTES` and re-derive (documented fallback, same single source).

`X_RESIDENT` is evaluated **before** `GAMMA_RESIDENT` because it saves `ht_per_core · Wt` tile-reads versus `NH_core · Wt` for gamma, and `HT_BLOCK ≥ 1`.

Phase-1 reach (bf16 TILE): `X_RESIDENT` up to `Wt ≈ 256` (`W = 8192`), `GAMMA_RESIDENT` up to `Wt = 128` (`W = 4096`). `W = 16384 / 32768` (the `_WIDE` loose cases) fall to the streaming fallback and still run bounded.

### 1.4 Bandwidth ranking of the candidate splits (qualitative, no timings)

| Rank | Candidate split | Bytes moved / fan-out | Combine needed | Verdict |
|---|---|---|---|---|
| 1 | **`ht` (independent)** | `x` read once (resident) or twice (streaming) + `y` written once, all as whole contiguous tile pages; only redundancy is `gamma` (`Wt` tiles × num_cores) | none | **Phase-1 primary.** Zero cross-core traffic, zero synchronization, per-core reads are contiguous tile-page strips. |
| 2 | `Wt` (dependent) — each core owns a `W`-slice of every row | Same `x`/`y` bytes; `gamma` read **once total** (each core owns a disjoint slice — strictly better than rank 1 on gamma); adds one 1-tile partial gather + one 1-tile `1/rms` mcast **per tile-row** | cross-core reduce-root or two-stage grid reduce + mcast | **Lamp L1.** The only parallelism available when `ht_total < grid_size` (wide-decode: `ht_total = 1`). The combine payload is 1 tile per tile-row and `tensix_all_reduce_ring_transport` measures the semaphore skeleton at 215–1346 ns, so the combine is cheap relative to idling 100+ cores — but it is a different loop nest, so it is designed here and executed in a refinement. |
| 3 | leading dims (`N`,`C`) alone | Same as rank 1 but coarser and unable to fill the grid when `N·C` is small (`(1,1,H,W)` → 1 unit) | none | Subsumed by rank 1 (`ht` already folds `N·C·ceil(H/32)`). |

**Operand-reuse check** (mechanical, per (operand, chosen-split) pair):

| Operand | Varies along `ht` (the split axis)? | Consequence |
|---|---|---|
| `input_tensor` | yes | disjoint per core — no redundancy |
| `gamma` | **no** | every core re-reads the identical `Wt` tiles ⇒ **reuse-shared by construction of the split** ⇒ **Lamp L2: broadcast it** (one injector core reads `gamma` once, mcasts to the split's cores via `mcast_pipe`; `examples/shared_input_reuse` measures 1.71× for exactly this shape) |
| `cb_scaler` mask | no, and it is 1 tile generated in-kernel | negligible; no action |

### 1.5 Scheme committed by phase 1

> Split the **independent tile-row axis across the full compute grid**; each core owns whole rows and reduces the **dependent `W` axis sequentially in-core** over `NW` coarse chunks, holding `x` and `gamma` resident whenever a size predicate says they fit; every block factor, buffer depth and the grid itself is a parameter at its coarsest-fitting value, never an inlined constant.

### 1.6 Lamp — scheme-changes phase 1 deliberately leaves reachable

| # | Unlock | Class | Why phase-1 does not foreclose it |
|---|---|---|---|
| **L1** | **Cross-core `W`-split + partial-sum combine.** Each core reduces its `W`-slice into a *raw partial* tile; a **reduce-root** (small groups) or **two-stage grid reduce** (grid-filling / 1-tile payload) combines them; the root finalizes `rsqrt(mean+ε)` and **mcasts the single `1/rms` tile back**. Never all-gather the partials (`tensix_all_reduce`: unicast all-gather is 0.74×, the worst option measured). Unlocks `WIDTH_SHARDED` + `BLOCK_SHARDED`. | scheme-change | Pass A already writes a **raw partial-sum tile** into `cb_partials` (a CB *distinct* from the finalized `cb_rms_sum`), and the finalize is already a **separate phase** (§8 phase 4) acting on a 1-tile-per-tile-row CB. The cross-core combine slots between phase 3 and phase 4 without touching phases 1/2/5/6/7. |
| **L2** | **`gamma` broadcast.** One injector core reads `gamma` from DRAM once and mcasts `Wt` tiles to the grid instead of `num_cores` DRAM re-reads. | scheme-change (reuse) | In the `GAMMA_RESIDENT` regime `cb_gamma` is already a **separate, filled-once, persistent** CB. Swapping its filler from a per-core `TensorAccessor` read to a `ReceiverPipe::receive()` is a reader-only change; compute is untouched. |
| **L3** | **`HEIGHT_SHARDED` input.** The logical height-shard made physical: shard grid pins the core-assignment, shard height pins `HT_BLOCK`, reduction stays local. | knob-turn | Only `cb_input_tiles`' *placement* changes: it becomes `ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor)` — **zero-copy on the core's own L1 shard, no NoC read** — and the reader's input pass disappears. Compute phases are byte-identical because the resident regime already consumes a whole resident row-strip. |
| **L4** | `bfloat16` + `fp32_dest_acc_en=False` | knob-turn | `DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`) already derives the DEST cap from the build flag; no kernel change. |
| **L5** | `bfloat8_b` input/gamma | knob-turn | CB `data_format` is already derived from the tensor dtype; block sizing already keys off `ttnn.tile_size(dtype)`. |
| **L6** | Deeper buffers / bigger blocks | knob-turn | `X_DEPTH`/`OUT_DEPTH`/`GAMMA_DEPTH`/`L1_BLOCK_BUDGET_BYTES` are single-source host constants. |

---

## 2. Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2; `bfloat16` \| `float32`; `TILE_LAYOUT` \| `ROW_MAJOR_LAYOUT`; `INTERLEAVED` | — | — |
| `gamma` | `Optional[ttnn.Tensor]` | no (kw-only) | shape `(1,1,1,W)` with `W == input.shape[-1]`; `bfloat16` \| `float32`; `TILE_LAYOUT` \| `ROW_MAJOR_LAYOUT` | `None` | — |
| `epsilon` | `float` | no (kw-only) | `> 0` | `1e-6` | RT (float bits, compute kernel) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no (kw-only) | `fp32_dest_acc_en` must be `True` in phase 0; `math_fidelity` / `math_approx_mode` accepted at any value and never gated | `default_compute_kernel_config()` | passed as-is to `ttnn.ComputeConfigDescriptor` on the compute kernel descriptor |
| `memory_config` | `Optional[ttnn.MemoryConfig]` | no (kw-only) | phase 1: `None` or interleaved | `None` | — |
| `program_config` | `Optional[Any]` | no (kw-only) | phase 1: `None` | `None` | — |

**Why `memory_config` / `program_config` are in the signature.** `eval/golden_tests/rms_norm/axes.py:24-27` declares `classify_call(input_tensor, *, gamma, epsilon, compute_kernel_config, memory_config, program_config)` and `eval/golden_tests/rms_norm/helpers.py:225-228` forwards `memory_config=` for every sharded cell. Accepting them keyword-only (and routing them through `validate()`) makes an unsupported sharded cell raise `SupportRefusal` → strict-xfail, instead of a `TypeError` → hard error. All five required call patterns are unaffected.

**Validation contract.** `validate()` raises `ValueError` / `RuntimeError` for (a) `input_tensor` rank < 2 — the message **must contain the substring `rank`**; (b) `gamma.shape[-1] != input_tensor.shape[-1]` — the message **must contain the substring `gamma`**. The acceptance test matches on those substrings (`test_rms_norm.py::test_rms_norm_rejects_rank_1`, `::test_rms_norm_rejects_gamma_width_mismatch`). Unsupported *axis values* (a sharded `memory_layout`, `bfloat8_b`, `fp32_dest_acc_en=False`) raise `SupportRefusal` subclasses from `ttnn/ttnn/operations/_op_contract.py` instead — these are support gates, not argument errors.

**Precision contract** (`.claude/references/precision_convention.md:61-85`): the op exports exactly one factory

```python
def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
```

`None` resolves through it; the golden axis tagger reads the same factory (`axes.py:40-43`). `float32` + `fp32_dest_acc_en=False` is refused via **op-side `EXCLUSIONS`** (strict-xfail), not `feature_spec.INVALID`.

## 3. Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2, `(..., H, W)`; `H`, `W` arbitrary (no 32-multiple requirement) |
| Dtype | `bfloat16`, `float32` (phase 1); `bfloat8_b` is TARGET-only (L5) |
| Layout | `TILE` or `ROW_MAJOR`, both native |
| Memory | interleaved DRAM (phase 1); `HEIGHT/WIDTH/BLOCK_SHARDED` are TARGET (L1/L3) |

### Gamma

| Property | Requirement |
|----------|-------------|
| Shape | `(1, 1, 1, W)`; `gamma.shape[-1] == input.shape[-1]` (validated) |
| Dtype | `bfloat16`, `float32` — **independent of the input dtype** (mixed precision supported; the FPU multiply reads srcA at the input format and srcB at the gamma format, reconfig handled by `BinaryDataFormatReconfig::Input`) |
| Layout | `ROW_MAJOR` (single stick of `W` elements) or `TILE` (tile-padded to `(1,1,32,W)`, row 0 valid) — both native |
| Absent | `gamma_dtype`/`gamma_layout` canonicalize to the string sentinel `"none"`; `"none"` is always legal and `validate()` never refuses it |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | identical to input |
| Layout | identical to input (`TILE` in → `TILE` out; `ROW_MAJOR` in → `ROW_MAJOR` out) |
| Memory | interleaved DRAM (phase 1), allocated with `ttnn.allocate_tensor_on_device` |

## 4. Dataflow Strategy

### 4.1 Phase-1 path (per core)

```
                          TILE input                              ROW_MAJOR input
DRAM  --TensorAccessor-->  cb_input_tiles  (tiles)      DRAM --read_sticks_for_tilize--> cb_input_rm (row pages)
                                 |                                        |  compute: tilize()
                                 |                                 cb_input_tiles (tiles)
                                 v                                        v
   compute:  square -> cb_x_squared -> reduce(SUM,REDUCE_ROW,AccumulateViaAdd) -> cb_partials/cb_rms_sum
             AddUnary(eps) + Rsqrt chain -> cb_rms_recip                (1 tile per tile-row, col-0 valid)
             mul<BroadcastDim::Col>(x, 1/rms) -> cb_scaled
             mul<BroadcastDim::Row>(scaled, gamma) -> cb_output_tiles
                                 |                                        |  compute: untilize()
                                 v                                 cb_output_rm (row pages)
DRAM  <--TensorAccessor--  cb_output_tiles                DRAM <--write_sticks_after_untilize--
```

Format at each stage:

| Stage | Format |
|---|---|
| DRAM `x` | tiles (TILE) / sticks of `W` elements (RM), input dtype |
| `cb_input_rm` | row-major page = one stick-chunk of `WT_CHUNK*32` elements (RM only) |
| `cb_input_tiles` | 32×32 tiles, input dtype |
| `cb_x_squared` | tiles, input dtype (bf16 `x²` is safe: bf16 shares fp32's exponent range; `examples/row_reduce_accumulate` measures bf16 *input* to the reduce as near-free while bf16 *accumulation* is the cost — and the accumulation is fp32 DEST here) |
| `cb_partials`, `cb_rms_sum`, `cb_rms_recip` | tiles, **Float32** (accumulator accuracy; `AccumulateReloadMode::CopySeedPairs` is the default and is safe for any accumulator CB format, `reduce_helpers_compute.hpp:312`) |
| `cb_scaler` | one **bfloat16** tile (0/1 mask or 1.0 scaler) |
| `cb_gamma` | tiles, gamma dtype (row 0 valid) |
| `cb_scaled`, `cb_output_tiles` | tiles, input dtype |
| `cb_output_rm` | row-major page = one stick-chunk (RM only) |

Reader on **NCRISC** (`ReaderConfigDescriptor`, NoC0), writer on **BRISC** (`WriterConfigDescriptor`, NoC1) — the default assignment, which `examples/noc_placement` measures as the fast pairing (read·NoC0 / write·NoC1, 4.3–4.8× over the mirrored assignment).

### 4.2 Tensix-to-Tensix contract for the *unlocked* L1 scheme (designed now, not built in phase 1)

Phase 1 issues no inter-Tensix traffic. The `W`-split scheme it must not foreclose:

| Element | Contract |
|---|---|
| Topology | A **line of `C_w` cores along one grid row** owns one tile-row's `W`; `Wt_c = ceil(Wt / C_w)` tiles each. Multiple grid rows process different tile-rows in parallel (this is exactly `BLOCK_SHARDED` geometry; `C_w = num_cores` with one grid row is `WIDTH_SHARDED`). |
| Partial production | Each core runs §8 phases 1–3 unchanged on its own `Wt_c` slice, ending with a **raw partial `Σx²` tile** in `cb_partials` (`Accumulate::at`, never `at_last` — the finalize is deferred). |
| Combine | **Reduce-root** for `C_w ≤ 4`; **two-stage grid reduce** when the grid is full or the payload is 1 tile/core (the RMSNorm case) — `examples/tensix_all_reduce` measures grid two-stage as the only steady variant under grid contention (1.45–1.60× over root, <1% variance) and measures **unicast all-gather at 0.74×, the worst option**. Never all-gather. |
| Finalize | Root core runs §8 phase 4 (`AddUnary(eps)` + `Rsqrt`) on the combined tile with `n_reduced = W` (the **grand total**, not the per-core slice count). |
| Broadcast back | Root **multicasts the single `1/rms` tile** to the line via `mcast_pipe`: host emits the wire with `Mcast1D`/`McastConfig` (`kernel_lib/host/mcast_host.hpp:64,76,145`), sender uses `SenderPipe::send()` (`mcast_pipe.hpp:169,188`), receivers `ReceiverPipe::receive()` (`mcast_pipe.hpp:247,265`). |
| Synchronization | Two `ttnn.SemaphoreDescriptor`s (data-ready, consumer-ready) declared over the line's `CoreRangeSet`; `McastArgs<>` (`mcast_pipe.hpp:319`) parses them off CT/RT args. |
| Ordering | Per tile-row: all partials must land before the root finalizes; the root's mcast must land before any core enters phase 5. Both edges are the `mcast_pipe` handshake; the semaphore skeleton itself costs 215–1346 ns (`examples/tensix_all_reduce_ring_transport`). Never route the combine as a ring/serpentine spanning two grid rows (~47 µs trap, same source). |
| `gamma` under this scheme | Each core owns a **disjoint** `gamma` slice → gamma is read exactly once in total; Lamp L2 becomes unnecessary in the `W`-split regime. |

### 4.3 Sharded placement, per scheme (the lamp expressed as a TARGET axis)

| `memory_layout` | Logical shard it makes physical | Class | Data access at the boundary |
|---|---|---|---|
| `INTERLEAVED` | — (phase 1) | — | `TensorAccessor` reads/writes the core's row strip from DRAM |
| `HEIGHT_SHARDED` | the phase-1 `ht` split, pinned + pre-placed | **knob-turn** | **Native: `ttnn.cb_descriptor_from_sharded_tensor` backs `cb_input_tiles` on the resident L1 shard — zero-copy, no NoC read, no reader input pass.** Re-reading a local shard through a `TensorAccessor` is explicitly *not* the design. |
| `WIDTH_SHARDED` | the `W`-split, one grid row | **scheme-change** (L1) | Same zero-copy CB aliasing per core, plus the cross-core combine of §4.2 |
| `BLOCK_SHARDED` | `ht` split × `W` split | **scheme-change** (L1) | Same, with the combine confined to each grid row |

## 5. Work Distribution

This is §1's core-assignment made concrete.

| Field | Value |
|-------|-------|
| Work unit | one **row-block** = `HT_BLOCK` tile-rows × full `W` |
| Grid | `core_grid = device.compute_with_storage_grid_size()` (a parameter, never an inlined count); `num_cores = min(core_grid.x * core_grid.y, ht_total)` |
| Split call | `ttnn.split_work_to_cores(core_grid, ht_total, row_wise=True)` → `(num_cores, all_cores, group_1, group_2, rows_g1, rows_g2)` (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:466-498`) |
| `row_wise` | **`True`, not the default `False`.** `examples/noc_placement` measures the column line that `row_wise=False` produces at 2.91× slower than a row line once reads are batched. |
| Per-core work | core `i` owns tile-rows `[start_row_i, start_row_i + rows_i)`; loops `NH_core = ceil(rows_i / HT_BLOCK)` row-blocks; the last row-block may be short (`ht_this = min(HT_BLOCK, rows_i - hb*HT_BLOCK)`), passed as a **runtime** shape to every helper |
| Remainder | handled by `split_work_to_cores`' two core groups (`rows_g1` / `rows_g2`); both groups run the same kernels with different `rows_i` / `start_row_i` runtime args |

### 5.1 Tile geometry — alignment-aware, per-image, from the start

| Layout | `ht_total` (tile-rows) | `Wt` | Rationale |
|---|---|---|---|
| `TILE` | `prod(shape[:-2]) * ceil(H / 32)` | `ceil(W / 32)` | In TILE layout **each `(H,W)` image is tile-padded independently** — it is `prod(leading) * ceil(H/32)`, **never** `floor(prod(leading)*H / 32)` |
| `ROW_MAJOR` | `ceil(prod(shape[:-1]) / 32)` | `ceil(W / 32)` | RM has no per-image padding: the buffer is a flat list of `prod(shape[:-1])` sticks, grouped into 32-stick tile-rows. Grouping across image boundaries is numerically sound because rows are independent. |

Every formula uses `ceil`. `W_valid_in_last_tile = W - (Wt-1)*32` (in `[1,32]`); `has_partial_w = (W % 32 != 0)`.

### 5.2 Regime-selection function (pinned; regime-pinned tests required)

The kernels are specialized on four compile-time booleans. **These predicates are the contract — the implementer must implement exactly these, and the acceptance test pins one shape per reachable regime.**

| Flag | Predicate | Effect |
|---|---|---|
| `IS_RM` | `input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT` | adds `cb_input_rm`/`cb_output_rm` + `tilize`/`untilize` phases |
| `HAS_GAMMA` | `gamma is not None` | adds `cb_gamma` + phase 6; when false phase 5 packs straight into `cb_output_tiles` |
| `IS_RM_GAMMA` | `HAS_GAMMA and gamma.layout == ttnn.ROW_MAJOR_LAYOUT` — **independent of `IS_RM`** (a TILE activation may carry an RM weight and vice versa) | adds `cb_gamma_rm` + phase 0a (`tilize` of the single gamma stick) |
| `X_RESIDENT` | §1.3 predicate | 1 reader pass instead of 2; `cb_input_tiles` sized `HT_BLOCK*Wt`; `InputLifecycle::CallerManaged` + `TileOffset::Set` |
| `GAMMA_RESIDENT` | §1.3 predicate (requires `HAS_GAMMA`) | gamma read once per core; `cb_gamma` sized `Wt`, `CallerManaged` + `TileOffset::Set` |

Plus one CT boolean derived from the shape: `HAS_PARTIAL_W = (W % 32 != 0)` (selects the mask scaler and the `ReducePartialScaler` argument).

Regimes that must be pinned by an explicit test (a regime that only triggers on some grids can pass on one device and fail on another):

| Regime | Pinning shape (bf16) |
|---|---|
| TILE, gamma, `X_RESIDENT`, `GAMMA_RESIDENT`, `NW == 1` | `(1,1,64,128)` |
| TILE, gamma, `X_RESIDENT`, `GAMMA_RESIDENT`, `NW > 1` | `(1,1,32,4096)` |
| TILE, gamma, `X_RESIDENT`, **not** `GAMMA_RESIDENT` | `(1,1,32,8192)` |
| TILE, gamma, **neither** resident (streaming fallback) | `(1,1,32,16384)` |
| TILE, no gamma | `(1,1,64,128)` with `gamma=None` |
| TILE, `HAS_PARTIAL_W` | `(1,1,32,50)` |
| TILE, `H % 32 != 0` | `(1,1,17,64)` |
| RM, gamma | `(1,1,64,128)` RM |
| RM, `HAS_PARTIAL_W` + `H % 32 != 0` | `(1,1,17,50)` RM |
| `HT_BLOCK > 1` (narrow-W multi-row block) | `(1,1,2048,64)` |

## 6. Circular Buffers

`H` = `HT_BLOCK`, `C` = `WT_CHUNK`, `B` = `H*C` (the block, in tiles). All page counts are functions of the knobs; **no CB's unconditional size is a function of a whole-op dimension.** The two `Wt`-sized CBs (`cb_input_tiles` when `X_RESIDENT`, `cb_gamma` when `GAMMA_RESIDENT`) are **predicate-guarded residents with a streaming fallback**, exactly the sanctioned fast-path shape.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_tiles` | 0 | `tile_bytes` | `X_RESIDENT ? H*Wt : (IS_RM ? B : X_DEPTH*B)` | input dtype | `IS_RM ? compute : reader` | compute | per row-block (resident) / streaming |
| `cb_gamma` | 1 | `gamma_tile_bytes` | `HAS_GAMMA ? (GAMMA_RESIDENT ? Wt : GAMMA_DEPTH*C) : 0` | gamma dtype | `IS_RM(gamma) ? compute : reader` | compute | whole kernel (resident) / per chunk |
| `cb_scaler` | 2 | `2048` (bf16 tile) | `1` | **Float16_b** | reader | compute | whole kernel |
| `cb_input_rm` | 3 | `C*32*elem_bytes` | `IS_RM ? X_DEPTH*H*32 : 0` | input dtype | reader | compute | streaming (RM only) |
| `cb_gamma_rm` | 4 | `C*32*gamma_elem_bytes` | `IS_RM_GAMMA ? GAMMA_DEPTH : 0` | gamma dtype | reader | compute | streaming (RM gamma only) |
| `cb_output_tiles` | 16 | `tile_bytes` | `IS_RM ? B : OUT_DEPTH*B` | input dtype | compute | `IS_RM ? compute : writer` | streaming |
| `cb_output_rm` | 17 | `C*32*elem_bytes` | `IS_RM ? OUT_DEPTH*H*32 : 0` | input dtype | compute | writer | streaming (RM only) |
| `cb_x_squared` | 24 | `tile_bytes` | `B` | input dtype | compute | compute | per chunk |
| `cb_partials` | 25 | `4096` (fp32 tile) | `2*H` | **Float32** | compute | compute | per row-block (only when `NW > 1`) |
| `cb_rms_sum` | 26 | `4096` (fp32 tile) | `H` | **Float32** | compute | compute | per row-block |
| `cb_rms_recip` | 27 | `4096` (fp32 tile) | `H` | **Float32** | compute | compute | per row-block (held across all `NW` chunks of pass B) |
| `cb_scaled` | 28 | `tile_bytes` | `HAS_GAMMA ? B : 0` | input dtype | compute | compute | per chunk |

### 6.1 Ownership (exactly one producer thread, one consumer thread)

Every row above names exactly one producer and one consumer. Notes on the non-obvious ones:

- **`cb_input_tiles` in the RM regime** is produced by compute's `tilize()` and consumed by compute's `square`/`mul` — one producer thread, one consumer thread (both `compute`), which satisfies the invariant. In the TILE regime the producer is the reader.
- **`cb_output_tiles` in the RM regime** is compute→compute (feeds `untilize()`); in the TILE regime compute→writer.
- **`cb_partials` / `cb_rms_sum` / `cb_rms_recip`** are compute→compute accumulators. `cb_partials` is both the `reduce()` output CB and its `AccumulationConfig::cb_accumulator` for non-final chunks — that is a single-thread read-modify-write, not two owners.
- There is **no** CB read by both a dataflow kernel and compute. `cb_input_tiles` is never written back into.

### 6.2 Sync ledger (push count = wait count)

| CB | Producer pushes (per row-block) | Consumer waits (per row-block) |
|---|---|---|
| `cb_input_tiles` | `X_RESIDENT ? H*Wt : 2*H*Wt` (`NW` chunk pushes × 1 or 2 passes) | pass A `H*Wt`, pass B `H*Wt`; resident regime waits `H*Wt` once (CallerManaged) and pops `H*Wt` once |
| `cb_gamma` | `GAMMA_RESIDENT ? Wt (once per kernel) : NH_core*NW*C` | matching |
| `cb_scaler` | `1` (once per kernel, never popped) | `HeldBulk`-style: waited by `reduce()`, never popped |
| `cb_x_squared` | `NW * B` | `NW * B` |
| `cb_partials` | `(NW-1) * H` | `(NW-1) * H` |
| `cb_rms_sum` | `H` | `H` |
| `cb_rms_recip` | `H` | waited `H` per chunk (`HeldBulk`), **popped `H` once** by the compute kernel after the chunk loop |
| `cb_scaled` | `NW * B` | `NW * B` |
| `cb_output_tiles` | `NW * B` | `NW * B` |

### 6.3 `BLOCK_CB_UNITS` — CB pages per block tile (the divisor in §1.2)

| Contribution | TILE | ROW_MAJOR |
|---|---|---|
| `cb_input_tiles` | `X_DEPTH` = 2 | 1 (compute→compute) |
| `cb_input_rm` | 0 | `X_DEPTH` = 2 |
| `cb_x_squared` | 1 | 1 |
| `cb_scaled` (gamma only) | 1 | 1 |
| `cb_output_tiles` | `OUT_DEPTH` = 2 | 1 |
| `cb_output_rm` | 0 | `OUT_DEPTH` = 2 |
| `cb_gamma` (gamma only, streaming) | 2 | 2 |
| `cb_gamma_rm` (`IS_RM_GAMMA` only — orthogonal to `IS_RM`) | 1 | 1 |
| **Total, gamma present** | **8** (7 with a TILE gamma) | **11** (10 with a TILE gamma) |
| **Total, no gamma** | **6** | **8** |

`cb_gamma_rm` is charged a full unit for safety; its real cost is `GAMMA_DEPTH` pages of `C*32*gamma_elem_bytes` (one *stick chunk*, i.e. 1/32 of a tile-row) — a few KB. Always use the conservative unit count so `TILE_BLOCK_BUDGET` never overshoots.

## 7. API Mapping

Every mechanism has a verified file:line reference. `ckl = compute_kernel_lib`, `dkl = dataflow_kernel_lib`.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| boot | raw_api | `compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles)` | mandated by `reduce_helpers_compute.hpp:30-34`, `eltwise_chain.hpp:27-42` | — | — | — | Exactly once, first statement of `MAIN()`. Never inside a loop. |
| reader: scaler | helper | `dkl::prepare_reduce_mask<cb_scaler, ReduceDim::REDUCE_ROW>(W_valid_in_last_tile)` | `reduce_helpers_dataflow.hpp:73-74` | `cb_scaler`, `REDUCE_ROW` | — | `cb_scaler` | Only when `HAS_PARTIAL_W`. Reduce-dim-aware overload; fills row-0 in the layout `AccumulateViaAdd` consumes on the last tile. |
| reader: scaler | helper | `dkl::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()` | `reduce_helpers_dataflow.hpp:97-99` | `cb_scaler`, `SUM`, `REDUCE_ROW` | — | `cb_scaler` | Only when `!HAS_PARTIAL_W`. **Pool-type-aware overload** (required); `cb_scaler` is **Float16_b**. |
| reader (RM) | helper | `dkl::read_sticks_for_tilize<cb_input_rm, TilizeGranularity::ROW>(acc, ht_this*32, chunk_row_bytes, start_page, wc*C*32*elem)` | `tilize_helpers_dataflow.hpp:87-93` | `cb_input_rm`, `ROW` | DRAM | `cb_input_rm` | `byte_offset_within_page` is the documented wide-W chunking hook (`:79-85`) — this is precisely the `WT_CHUNK` knob on the read side. Handles non-aligned W (pads L1 stride) and non-aligned H (pushes full pages, stale rows). |
| reader (RM gamma) | helper | `dkl::read_sticks_for_tilize<cb_gamma_rm, TilizeGranularity::ROW>(gacc, 1, chunk_row_bytes, 0, wc*C*32*gelem)` | `tilize_helpers_dataflow.hpp:87-93` | `cb_gamma_rm`, `ROW` | DRAM | `cb_gamma_rm` | `total_num_rows=1`; `:64-65` documents the reduced-L1 path for `< 32` rows. |
| reader (TILE) | raw_api | `TensorAccessor` + `noc_async_read_tile` / `noc_async_read_barrier` | `tech_reports/tensor_accessor/tensor_accessor.md` | page = tile | DRAM | `cb_input_tiles` / `cb_gamma` | Whole tile pages, one barrier per chunk (`examples/tile_reorder`: whole-page transfers ≥ face-scatter; `examples/double_buffer`: batch 4–8 tiles per barrier — `WT_CHUNK ≥ 8` satisfies this). *Helpers considered and rejected:* `read_sticks_for_tilize` (`tilize_helpers_dataflow.hpp:87`) is stick-indexed and feeds the tilize helper — it cannot express a tile-page read of an already-tiled tensor. |
| 1 (RM) | helper | `ckl::tilize<C, cb_input_rm, cb_input_tiles>(ht_this, ht_this*32)` | `tilize_helpers.hpp:187-197` | `block_width_tiles = C` (**the W block knob**, CT); `WT_LAST` instantiation for the short last chunk; `Fp32Mode::Fast` (default — `:47-71` explains why "max precision" still wants Fast: every FPU consumer re-truncates through SrcA/SrcB) | `cb_input_rm` | `cb_input_tiles` | Asymmetric mode (`total_input_pages` = rows) pairs with `TilizeGranularity::ROW`. |
| 2 | helper | `ckl::square<cb_input_tiles, cb_x_squared, XLife, OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input, PackTileReconfig::Output, OperandKind::Block>(EltwiseShape::grid(ht_this, wt_this))` | `eltwise_convenience.hpp:107-120`; `EltwiseShape::grid` `eltwise_chain.hpp:119` | `XLife = X_RESIDENT ? CallerManaged : Bulk`; the `EltwiseShape` **is** the (`HT_BLOCK`,`WT_CHUNK`) block knob | `cb_input_tiles` | `cb_x_squared` | FPU `x*x` via the same-buffer path (waits/pops once, `:101-105`). `OperandKind::Block` + `Bulk`/`CallerManaged` is a legal cell (`eltwise_chain.hpp:363-368`). |
| 3 (non-final chunk) | helper | `ckl::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_x_squared, cb_scaler, cb_partials, ReduceInputPolicy::BulkWaitBulkPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, ReduceAlgorithm::AccumulateViaAdd>(ReduceInputBlockShape::of(ht_this, wt_this, 1), ReduceInputMemoryLayout::contiguous(), Accumulate::at(cb_partials, wc), NoOp{})` | `reduce_helpers_compute.hpp:522-538`; `::of` `:215`; `Accumulate::at` `:327`; algorithm enum `:152` (semantics `:125-150`) | `AccumulateViaAdd` is the measured-fast datapath (`examples/row_reduce_accumulate`: 2.93× at 32 t; `examples/reduce_block`: 5.35×; crossover at `Wt ≥ 4`). `BulkWaitBulkPop` is **mandatory** for `Accumulate` (`:139,:148`) | `cb_x_squared` | `cb_partials` | Only emitted when `NW > 1`. Accumulator holds the **raw** partial sum. |
| 3 (final chunk) | helper | `ckl::reduce_mean<ReduceDim::REDUCE_ROW, cb_x_squared, cb_scaler, cb_rms_sum, BulkWaitBulkPop, INPUT_AND_OUTPUT, AccumulateViaAdd>(ReduceInputBlockShape::of(ht_this, wt_this, 1), /*n_reduced=*/W, ReduceInputMemoryLayout::contiguous(), Accumulate::at_last(cb_partials, NW-1), partial)` | `reduce_helpers_compute.hpp:576-590`; `at_last` `:332`; `ReducePartialScaler::partial_mask` `:263` | `n_reduced = W` — the **true element count** (`W`, not `Wt*32`), the grand total across all chunks (`:551-553`). `partial = HAS_PARTIAL_W ? ReducePartialScaler::partial_mask(W_valid_in_last_tile, 0) : ReducePartialScaler::none()` | `cb_x_squared` | `cb_rms_sum` | When `NW == 1` pass `NoAccumulation{}` instead. Partial composes with Accumulate under the default `CopySeedPairs` reload (`:148-150`, default at `:312`). |
| 4 | helper | `ckl::eltwise_chain(EltwiseShape::tiles(ht_this), CopyTile<cb_rms_sum, Dst::D0, InputLifecycle::Streaming>{}, AddUnary<Dst::D0>{eps_bits}, Rsqrt<>{}, PackTile<cb_rms_recip, OutputLifecycle::Streaming>{})` | `eltwise_chain.hpp:710-711`; `CopyTile` `:621-628`; `PackTile` `:666-674`; `AddUnary` `eltwise_scalar.hpp:26` (impl `eltwise_scalar.inl:56`); `Rsqrt` `eltwise_math.hpp:38` | `eps_bits` = `epsilon` float bits as `uint32_t` (RT arg); one dst-sync window for both SFPU ops | `cb_rms_sum` | `cb_rms_recip` | `examples/compute_fusion`: fusing the reduce epilogue on the SFPU is a win (1.01–1.12×); the FPU consumer that follows must **not** try to keep it in DEST (0.82×) — hence the L1 hand-off to phase 5. |
| 5 | helper | `ckl::mul<cb_input_tiles, cb_rms_recip, OutCb5, BroadcastDim::Col, XLifeB, InputLifecycle::HeldBulk, OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input, PackTileReconfig::Output, OperandKind::Block, RmsKind>(EltwiseShape::grid(ht_this, wt_this))` | `eltwise_convenience.hpp:81-98`; `BroadcastDim` `eltwise_chain.hpp:529-534`; `OperandKind` `:346-351` | `OutCb5 = HAS_GAMMA ? cb_scaled : cb_output_tiles`; `RmsKind = (HT_BLOCK > 1) ? OperandKind::Col : OperandKind::Scalar`; `XLifeB = X_RESIDENT ? CallerManaged : Bulk` | `cb_input_tiles`, `cb_rms_recip` | `cb_scaled` / `cb_output_tiles` | **`BroadcastDim::Col` is correct**: `eltwise_chain.hpp:526-528` states a `REDUCE_ROW` result is column-shaped and broadcasts back across columns via `Col`. `Col`/`Scalar` + `HeldBulk` is a legal cell (`:375-381`). |
| 6 (gamma) | helper | `ckl::mul<cb_scaled, cb_gamma, cb_output_tiles, BroadcastDim::Row, InputLifecycle::Streaming, GammaLife, OutputLifecycle::Streaming, BinaryDataFormatReconfig::Input, PackTileReconfig::Output, OperandKind::Block, GammaKind>(EltwiseShape::grid(ht_this, wt_this))` | `eltwise_convenience.hpp:81-98` | `GammaKind = (HT_BLOCK > 1) ? OperandKind::Row : OperandKind::Block`; `GammaLife = GAMMA_RESIDENT ? CallerManaged : Bulk` | `cb_scaled`, `cb_gamma` | `cb_output_tiles` | `BroadcastDim::Row` broadcasts operand **B** (`gamma`, row-0 valid) down the rows. `Row` kind requires the 2-D grid shape + a persistent lifecycle (`:378-381`) — satisfied. |
| 7 (RM) | helper | `ckl::untilize<C, cb_output_tiles, cb_output_rm>(ht_this)` | `untilize_helpers.hpp:145-154` | `block_width_tiles = C` (CT, the W block knob); `WT_LAST` instantiation for the short last chunk | `cb_output_tiles` | `cb_output_rm` | Symmetric tile pages on input, row pages on output. |
| writer (RM) | helper | `dkl::write_sticks_after_untilize<cb_output_rm>(acc, ht_this*32, chunk_row_bytes, start_page, wc*C*32*elem)` | `tilize_helpers_dataflow.hpp:129-135` | — | `cb_output_rm` | DRAM | Writes only valid sticks on a short last tile-row (`:104-105`) and skips L1 W-padding (`:102-103`) — this is what makes non-aligned H/W native with no host slice. |
| writer (TILE) | raw_api | `TensorAccessor` + `noc_async_write_tile` / barrier | `tech_reports/tensor_accessor/tensor_accessor.md` | page = tile | `cb_output_tiles` | DRAM | *Helpers considered and rejected:* `write_sticks_after_untilize` (`tilize_helpers_dataflow.hpp:129`) writes **sticks** produced by the untilize helper; the TILE path never untilizes, so its input contract cannot be met. |
| host | binding | `ttnn.split_work_to_cores(core_grid, ht_total, row_wise=True)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:466-498` | — | — | — | Two core groups; both get the same kernels. |
| host | binding | `ttnn.tile_size(dtype)` | `ttnn/cpp/ttnn-nanobind/tensor.cpp:249-262` | — | — | — | `tile_bytes` for intermediate CBs that have no backing tensor. |
| host | binding | `ttnn.ComputeConfigDescriptor` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:627-686` | — | — | — | Passed through as `config=compute_kernel_config` after `validate()` (ctor default is `fp32_dest_acc_en=False` — the factory must set `True` explicitly). |
| host (L3 lamp) | binding | `ttnn.cb_descriptor_from_sharded_tensor(cb_index, tensor, ...)` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517-521` | — | — | — | Not used in phase 1; the sanctioned zero-copy path for a resident local shard. |
| — | constant | `ckl::DEST_AUTO_LIMIT` | `dest_helpers.hpp:103` | — | — | — | 4 tiles at `fp32_dest_acc_en=True` + half-sync. Every chain here uses `block_size = 1` (D0, or D0+D1) — well inside the cap; never hardcode `8`. |

### 7.1 Helpers considered and rejected

| Considered helper | File:Line of the mismatch | Concrete reason |
|---|---|---|
| `ckl::accumulate_reduce` / `accumulate_reduce_block` (the streaming chunked-reduce wrappers) | `streaming_reduce_helpers.inl:32-40` and `:42-50` | They call `reduce<pool, rdim, in_policy, reconfig_mode>(cb_in, cb_scaler, cb_acc, block_shape, ml, accumulate, post, partial)` — **8 runtime arguments and 4 template arguments**. The only `reduce()` declaration (`reduce_helpers_compute.hpp:522-538`, definition `reduce_helpers_compute.inl:704-720`) takes the three CB ids as **template** NTTPs (params 3–5) and **5 runtime arguments**; the wrapper's 3rd/4th template arguments are the `ReduceInputPolicy` / `ReduceDataFormatReconfigMode` enums, which cannot bind to `uint32_t input_dfb_id` / `scaler_dfb_id`. The call cannot instantiate. Independently, `accumulate_reduce_block` hardwires `Accumulate::at` (`streaming_reduce_helpers.inl:38,48`) and never `Accumulate::at_last`, so even if it compiled it could not drive the `AccumulateViaAdd` datapath, whose finalize is gated on `is_last()` (`reduce_helpers_compute.hpp:314-319,346-347`). The explicit chunk loop in §8 is the pattern `reduce_mean`'s own docstring prescribes (`reduce_helpers_compute.hpp:564-574`). |
| `ckl::transform_in_place` (for the `rsqrt(mean+ε)` finalize) | `streaming_reduce_helpers.inl:74-102` (`constexpr uint32_t onetile = 1;` and the single `cb_wait_front(cb, onetile)`) | Processes **exactly one tile per call** and only in-place on a single CB. Phase 4 finalizes `HT_BLOCK` tiles (up to 16 for narrow-W blocks); using it would mean `HT_BLOCK` separate dst-sync windows plus a raw-lambda SFPU body. The `eltwise_chain` in phase 4 does all `HT_BLOCK` tiles in one call with typed `AddUnary`/`Rsqrt` elements and no in-place 1-page hazard. |
| `ckl::unary<Rsqrt<>, ...>` (`eltwise_convenience.hpp:126-139`) for phase 4 | `eltwise_convenience.hpp:135-139` — the wrapper emits exactly `CopyTile → SfpuOp → PackTile` | It composes **one** SFPU op; the finalize needs two (`AddUnary(ε)` then `Rsqrt`) sharing one dst-sync window. `eltwise_chain` directly is the documented escape hatch for multi-op chains (`eltwise_convenience.hpp:29-30`). |
| `ckl::DestReuseBinary` to fuse phases 5+6 into one dst window | `eltwise_chain.hpp:646-656` — the template has **no `BroadcastDim` parameter** | Phase 5 needs `BroadcastDim::Col` (per-row `1/rms`) and phase 6 needs `BroadcastDim::Row` (per-column `gamma`); a single chain can carry only one `BinaryFpu` broadcast, and `DestReuseBinary` cannot broadcast at all, so `gamma` would have to be materialized as `Wt` full-valid tiles. Independently, `examples/compute_fusion` measures DEST-reuse **losing** for an FPU consumer (0.82× on the isolated combine; the L1 round-trip is 1.22× faster), so the two-`mul` form is also the faster shape. |
| `ckl::reduce` with the default `ReduceAlgorithm::Auto` / `ReduceTile` | `reduce_helpers_compute.hpp:117-119` — `Auto` "always resolves to ReduceTile" today | `examples/row_reduce_accumulate` and `examples/reduce_accumulate` both measure `ReduceTile` losing from `Wt ≥ 4` (2.87–2.93× on WH, 5.35× on BH at 32 t). Every phase-1 chunk is ≥ 8 tiles wide except for `W ≤ 96`, so `AccumulateViaAdd` is pinned explicitly rather than left to `Auto`. |
| `ckl::reduce<PoolType::AVG, ...>` instead of `reduce_mean` | `reduce_helpers_compute.hpp:543-553` | `AVG`'s divisor is derived from tile geometry, which is wrong for a non-tile-aligned `W` and cannot compose across `Accumulate` chunks (stated verbatim at `:543-546`). `reduce_mean`'s caller-supplied `n_reduced = W` is the only correct form. Also `AccumulateViaAdd` is `SUM`-only (`:135-136`). |

## 8. Compute Phases

Per core: `for hb in 0..NH_core-1` over row-blocks (`ht_this = min(HT_BLOCK, rows_i - hb*HT_BLOCK)`); inside, two chunk loops over `wc in 0..NW-1` (`wt_this = (wc == NW-1) ? WT_LAST : WT_CHUNK`).

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|--------------------------|-------------------|----------------|
| 0 | `compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles)` | raw | — | — | Once per kernel, before everything. |
| 0a | *(`IS_RM_GAMMA`)* `tilize<C\|WT_LAST, cb_gamma_rm, cb_gamma>(1, 1)`, once per chunk. `GAMMA_RESIDENT` → all `NW` calls hoisted to kernel start with `InitUninitMode::InitOnly`/`Neither`/`UninitOnly` (`tilize_helpers.hpp:29-38`) to amortise the LLK init; otherwise emitted inline before phase 6 of each chunk. | yes | `cb_gamma_rm` (1 row page) | `cb_gamma` (`Wt` resident, or `wt_this` streaming) | Resident: `cb_gamma` holds `Wt` tiles for the **whole kernel**, never popped. Only row 0 of each tile is meaningful and only row 0 is ever read (`BroadcastDim::Row`), so the stale rows 1–31 the 1-row tilize leaves behind are harmless (risk R10). |
| 1 | *(RM)* `tilize<C|WT_LAST, cb_input_rm, cb_input_tiles>(ht_this, ht_this*32)` | yes | `cb_input_rm` (`ht_this*32` row pages) | `cb_input_tiles` (`ht_this*wt_this`) | Runs once per chunk in pass A; **and again in pass B iff `!X_RESIDENT`**. |
| 2 | `square` → `x²` | yes | `cb_input_tiles` (`ht_this*wt_this`; `CallerManaged` if resident, `Bulk` otherwise) | `cb_x_squared` (`ht_this*wt_this`) | `cb_x_squared` full-block (sequential-helper intermediate). `cb_input_tiles` retained iff `X_RESIDENT`. |
| 3 | `reduce<SUM,REDUCE_ROW,AccumulateViaAdd>` with `Accumulate::at(cb_partials, wc)` for `wc < NW-1`; `reduce_mean<...>` with `Accumulate::at_last(cb_partials, NW-1)` + `partial` on the last chunk (or `NoAccumulation{}` when `NW==1`) | yes | `cb_x_squared` (`ht_this*wt_this`, `BulkWaitBulkPop`) | `cb_partials` (`ht_this`) / `cb_rms_sum` (`ht_this`) | `cb_x_squared` drained. After the chunk loop `cb_rms_sum` holds `ht_this` tiles of `mean(x²)` (col-0 valid) and `cb_partials` is drained (see §9 risk R3). |
| 4 | `eltwise_chain(tiles(ht_this), CopyTile(cb_rms_sum) → AddUnary(ε) → Rsqrt → PackTile(cb_rms_recip))` | yes | `cb_rms_sum` (`ht_this`) | `cb_rms_recip` (`ht_this`) | `cb_rms_sum` drained. `cb_rms_recip` holds `1/rms` per tile-row (col-0 valid) and must **survive all `NW` chunks of pass B**. |
| 5 | `mul<BroadcastDim::Col>`: `x * (1/rms)` | yes | `cb_input_tiles` (`ht_this*wt_this`), `cb_rms_recip` (`HeldBulk`, `Col`/`Scalar`) | `cb_scaled` (`ht_this*wt_this`) or `cb_output_tiles` when `!HAS_GAMMA` | `cb_rms_recip` **not popped** by the chain. |
| 6 | *(gamma)* `mul<BroadcastDim::Row>`: `scaled * gamma` | yes | `cb_scaled` (`ht_this*wt_this`), `cb_gamma` (`wt_this`; `CallerManaged`+`TileOffset` if resident, `Bulk` otherwise) | `cb_output_tiles` (`ht_this*wt_this`) | `cb_scaled` drained. |
| 7 | *(RM)* `untilize<C|WT_LAST, cb_output_tiles, cb_output_rm>(ht_this)` | yes | `cb_output_tiles` (`ht_this*wt_this`) | `cb_output_rm` (`ht_this*32` row pages) | — |
| 8 | end of row-block: `cb_pop_front(cb_rms_recip, ht_this)`; if `X_RESIDENT` `cb_pop_front(cb_input_tiles, ht_this*Wt)` | raw (CB plumbing, not compute) | — | — | All per-row-block CBs empty. `cb_gamma` (resident) and `cb_scaler` persist. |

Pass structure: **phases 1–3 form pass A** (all `NW` chunks), **phase 4 runs once**, **phases 1(iff not resident)+5–7 form pass B** (all `NW` chunks).

Reader pass count: `X_RESIDENT ? 1 : 2` per row-block. This is the single structural difference between the two residency regimes.

## 9. Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| 2 | `square` (`BinaryFpu<cb_input_tiles, cb_input_tiles, Mul>`) | 2D `[H,W]` → All | same buffer → All | `None` |
| 5 | `mul` `x * (1/rms)` | `cb_input_tiles` 2D `[H,W]` → All | `cb_rms_recip` = `REDUCE_ROW` output → **Col0** | **`Col`** (`eltwise_chain.hpp:526-528`: a `REDUCE_ROW` result is column-shaped and broadcasts back across columns via `Col`) |
| 6 | `mul` `scaled * gamma` | `cb_scaled` 2D `[H,W]` → All | `cb_gamma` 1D `[W]` → **Row0** | **`Row`** |

## 10. Key Risks and Gotchas

| # | Risk | Mitigation / requirement |
|---|---|---|
| R1 | **CBs that must hold a full block.** `cb_x_squared` and `cb_scaled` sit between two *sequential* helper calls on the same compute thread. Sizing them below `HT_BLOCK*WT_CHUNK` deadlocks: the producer must push the whole block before the consumer's first `cb_wait_front` runs. | Both are sized exactly `B = HT_BLOCK*WT_CHUNK` pages in §6. Do not "optimize" them to 2. |
| R2 | **`cb_rms_recip` must survive the whole pass-B chunk loop.** `InputLifecycle::HeldBulk` waits but never pops (`eltwise_chain.hpp:216`). | Compute pops `ht_this` pages explicitly at phase 8. If this pop is missed the CB fills after `H`/`ht_this` row-blocks and the kernel hangs on `cb_reserve_back`. |
| R3 | **`cb_partials` drain.** `reduce()`'s `Accumulate` path both reads and writes the accumulator CB. Whether the finalizing `at_last` call leaves the accumulator popped is not stated in the header. | `cb_partials` is sized `2*HT_BLOCK` (not `HT_BLOCK`) so a one-block lag cannot deadlock, and the implementer must **verify the front is empty at the end of each row-block** (add an explicit `cb_pop_front(cb_partials, ht_this)` if not) before running multi-row-block shapes such as `(1,1,2048,256)`. |
| R4 | **Scaler CB must be bfloat16 and the pool-type-aware overload.** | `cb_scaler` format is `Float16_b`, 1 page. `HAS_PARTIAL_W` → `prepare_reduce_mask<cb, REDUCE_ROW>` (`reduce_helpers_dataflow.hpp:73`); otherwise `calculate_and_prepare_reduce_scaler<cb, SUM, REDUCE_ROW>` (`:97`). Never the legacy single-template-arg `prepare_reduce_scaler<cb>`. Emit **exactly one** tile and never pop it. |
| R5 | **`n_reduced` is `W`, not `Wt*32`.** Using the padded count silently scales every output by `W/(Wt*32)` on non-aligned shapes — a PCC failure that only appears on `w_non_aligned` cells. | `reduce_mean(..., n_reduced=W, ...)`, passed as a runtime arg, on the **finalizing** chunk only (`reduce_helpers_compute.hpp:551-553`). |
| R6 | **Tile-padding content.** The `W`-mask multiplies padding lanes by 0. `Inf * 0 = NaN`. | The design relies on tile padding being finite (ttnn zero-fills TILE padding; the RM reader path leaves *stale L1* in the H-padding rows — harmless because `REDUCE_ROW` is per-row and `write_sticks_after_untilize` never writes those rows back, `tilize_helpers_dataflow.hpp:104-105`). If a future path can produce `Inf`/`NaN` padding, the mask must move to an explicit pre-reduce multiply. |
| R7 | **`HT_BLOCK` vs `TileOffset` invariant.** `TileOffset::Set(wc*WT_CHUNK)` into a resident buffer is only valid because `NW > 1 ⟹ HT_BLOCK == 1` (§1.1). | Re-tuning `TILE_BLOCK_BUDGET` must preserve the invariant; the host should `assert not (NW > 1 and HT_BLOCK > 1)`. |
| R8 | **Data that must persist across phases.** `cb_gamma` (`GAMMA_RESIDENT`) and `cb_scaler` live for the whole kernel and are **never popped**. `cb_input_tiles` (`X_RESIDENT`) lives across pass A **and** pass B of one row-block. | Use `InputLifecycle::CallerManaged` for these (the chain neither waits nor pops) with an explicit compute-side `cb_wait_front` once and `cb_pop_front` at the documented point. |
| R9 | **DEST budget under `fp32_dest_acc_en=True`.** Half-sync + fp32 → 4 tiles (`dest_helpers.hpp:88-96`). | Every chain uses `EltwiseShape` `block_size = 1` → 1 DEST lane (phase 4's `CopyTile`+SFPU chain shares D0). `AccumulateViaAdd` uses one DST per output tile and processes one row at a time. Never write a literal `8`; use `DEST_AUTO_LIMIT` if a bound is needed. |
| R10 | **RM gamma tilization.** `tilize` on a **1-row** input produces `Wt` tiles whose rows 1–31 are stale. | Correct here because `BroadcastDim::Row` reads only row 0. Do not reuse `cb_gamma` for anything that reads other rows. |
| R11 | **`ceil`, never `//`.** `ht_total` for TILE is `prod(shape[:-2]) * ceil(H/32)` — per image. `floor(prod*H/32)` silently drops the last partial tile-row of every image on `h_non_aligned` cells. | §5.1. |
| R12 | **Reconfig cost.** `ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT` is the safe default and is required here because `cb_x_squared` (input dtype) → `cb_partials`/`cb_rms_sum` (fp32) is a genuine format change. | Keep it. `examples/compute_block_size/report_reconfig_ablation.md` measures ~110–150 ns per reconfig — real, but only removable when formats truly never change, which is not the case on this boundary. |

## 11. Feature-spec alignment

`eval/golden_tests/rms_norm/feature_spec.py` is authoritative and read-only for this design.

| TARGET axis | Phase-1 intent | Gap → refinement |
|---|---|---|
| `dtype` | `float32`, `bfloat16` | `bfloat8_b` → L5 (knob-turn) |
| `fp32_dest_acc_en` | `True` | `False` for `bfloat16` → L4 (knob-turn); `float32`+`False` → op-side `EXCLUSIONS` (never supported) |
| `layout` | `TILE`, `ROW_MAJOR` (both native) | — |
| `alignment` | `tile_aligned`, `w_non_aligned`, `h_non_aligned` (all three native) | — |
| `rank` | 2, 3, 4 (all leading dims collapse into `ht_total`) | — |
| `gamma_mode` | `gamma`, `no_gamma` | — |
| `gamma_dtype` | `float32`, `bfloat16`, `"none"` — **`"none"` is always legal** | `bfloat8_b` → L5 |
| `gamma_layout` | `TILE`, `ROW_MAJOR`, `"none"` — **`"none"` is always legal** | — |
| `memory_layout` | `INTERLEAVED` | `HEIGHT_SHARDED` → **L3, knob-turn** (placement only: `cb_descriptor_from_sharded_tensor`, reduction stays local); `WIDTH_SHARDED` / `BLOCK_SHARDED` → **L1, scheme-change** (dependent axis split, partials combined across cores) |

Required `INPUT_TAGGERS` (implementer, in the op module — imported by `axes.py:22` and `test_golden.py:24-25`):
`tag_alignment(inputs, axes) -> "tile_aligned" | "w_non_aligned" | "h_non_aligned"` (W-not-divisible-by-32 wins over H) and `tag_rank(inputs, axes) -> len(inputs[0])`.
The op module must export exactly: `rms_norm`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `default_compute_kernel_config`.

### Structural impossibilities

No candidates beyond those already in `feature_spec.py`'s `INVALID`. Note for the implementer (not an `INVALID` candidate): `{dtype: float32, fp32_dest_acc_en: False}` is a **refusable** cell and belongs in the op's `EXCLUSIONS` (strict-xfail), per `.claude/references/precision_convention.md:31-39` — not in `INVALID`.
