# Operation Requirements: groupnorm_sc_N_1_HW_C_b

## Definition
- **Formula**: for each image `n` and group `g` (channels `[g·C/G, (g+1)·C/G)`), over all `HW` rows and the group's channels:
  `mean = E[x]`, `var = E[(x − mean)²]`, `y = (x − mean) / sqrt(var + eps) · γ + β` (γ/β optional, per channel).
- **PyTorch Reference**: `eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers.pytorch_groupnorm_sc_N_1_HW_C(input_tensor, num_groups, *, gamma=None, beta=None, eps=1e-5)` — fp32 `torch.nn.functional.group_norm` on the `(N, C, HW)` permute, returned in the input dtype.
- **Import Path**: `from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b`
- **Function Signature**:
  `groupnorm_sc_N_1_HW_C_b(input_tensor: ttnn.Tensor, num_groups: int, *, gamma: ttnn.Tensor = None, beta: ttnn.Tensor = None, eps: float = 1e-5, in_place: bool = False) -> ttnn.Tensor`
  - The input is 4D `(N, 1, HW, C)`; γ/β are `(1, 1, 1, C)` in DRAM.
  - There is no `memory_config`, `core_grid` or `compute_kernel_config` parameter; this is a prompt contract.
  - Output placement:
    - DRAM-interleaved input gives a DRAM-interleaved TILE output.
    - BLOCK_SHARDED input gives an output with the same layout and the same shard spec.
  - The output dtype equals the input dtype.
  - One `ttnn.generic_op` dispatch per call.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Queue shape**: the op prompt (`eval/prompts/groupnorm_sc_N_1_HW_C_b.txt`, PREFERRED REFINEMENT QUEUE) fixes Refinements 1–7, in this order and unmerged. Measured perf refinements come after Refinement 7 (8–10), and the runner adds its own trailing perf rounds after Refinement 2. This replaces the default 2:1 cadence.
> The perf-anchor rule still holds: Refinements 1 + 2 land the full contract of the first perf target (`unet_rm_ip_*`: BLOCK_SHARDED 8×8, ROW_MAJOR, in place, bf16, γ+β bf16 ROW_MAJOR, fixed HiFi3 + fp32 DEST).
> **Every refinement**:
> - re-runs the acceptance suite with `--dev`. Kernel text sits close to the 70 688 B kernel-config ring (see the deviation table in `l1_ledger.md`). If it overflows, split the compute kernel by CT-specialized role; do not drop `--dev`.
> - keeps `l1_ledger.md` current.
> - records the device-ns of each PERF FOCUS case it touches in `changelog.md`.

**TARGET − SUPPORTED coverage**:

| Axis | Missing value | Where it lands |
|---|---|---|
| `memory_layout` | `BLOCK_SHARDED` | Refinement 1 |
| `shard_fit` | `even` | Refinement 1 |
| `shard_fit` | `ragged` | Refinement 3 |
| `shard_fit` | `illegal` | structural INVALID |
| `in_place` | `True` | Refinement 2. On interleaved input it is structural INVALID / `ValueError`. |
| `dtype` | `float32` | Refinement 4 (DRAM). Sharded is scope-parked INVALID. |
| `dtype` | `bfloat8_b` | Refinement 7 (TILE, DRAM). RM is INVALID; sharded is scope-parked. |
| `affine_dtype` | `float32` | Refinement 4 |
| `affine_dtype` | `bfloat8_b` | Refinement 7 |
| `affine_layout` | `TILE` | Refinement 4 |
| `alignment` | `hw_non_aligned` | Refinement 5 |
| `alignment` | `c_non_aligned` | Refinement 6 |

`layout` and `affine` are already complete.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]. RM input is tilized inside the kernel; the output is always TILE (DRAM contract).
- **SUPPORTED shape-derived axes**:
  - `alignment = tile_aligned`
  - `memory_layout = INTERLEAVED`
  - `shard_fit = none`
  - Partial channels per group (`(C/G) % 32 ≠ 0`, e.g. Cg = 4, 10, 20, 30, 36) are supported, with no gate.
- **SUPPORTED op-specific axes**:
  - `affine ∈ {gamma_beta, gamma_only, no_affine}`
  - `affine_dtype ∈ {bfloat16, "none"}`
  - `affine_layout ∈ {ROW_MAJOR, "none"}`
  - `in_place = False`
- **Cores**: full grid. The split is `n × xs × ht` with a gather → root → `ttnn.Mcast` combine. Every perf shape plans 104–110 of 110 cores.
- **Compute config**: fixed HiFi3 + `fp32_dest_acc_en`.
- **Golden baseline**: 399 `supported_pass`, 0 `supported_fail` / `xpass_drift` / `xfail_wrong_mode`. That is 434 / 47 152 collected cells passing; the rest are xfail_expected 4365 and invalid_skipped 42 345. All 9 DRAM PERF FOCUS cases pass at `PERF_TOLERANCE` and are 1.96×–25× faster than native (see `changelog.md`).

### [x] Refinement 1 — L1 BLOCK_SHARDED input, out of place (even shards, ROW_MAJOR + TILE, bf16), plus the variance precision fix

**Goal**:
- Add `ttnn.TensorMemoryLayout.BLOCK_SHARDED` to `SUPPORTED["memory_layout"]` and `"even"` to `SUPPORTED["shard_fit"]`, keeping `in_place = False`.
- Cover both layouts, bf16 input and γ/β bf16 ROW_MAJOR, on the model grids 8×8, 4×8 and 4×4.
- **Output contract:**
  - same layout as the input (RM in → RM out);
  - the same BLOCK_SHARDED L1 memory config with an identical shard spec;
  - output dtype = input dtype.
- **Shard handling is native and zero-copy.** A core's own shard *is* its block:
  - Input CBs are backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor`).
  - Output CBs are backed on the output shard.
  - **No** `TensorAccessor` read or write of a core's own shard, no DRAM round trip, no manipulation-op wrapper. An accessor read of the local shard counts as not implemented, even if the golden cells pass.
- **Combine.** The unit is one image's grid column: grid rows split `N·HW`, grid columns split `C`. The gather goes up the column to its root, and the stats mcast goes back down it in `ColumnMajor` mcast groups. For `N > 1`, each column holds `gy / N` shards per image, and each image combines separately.
- **RM slices that are not tile-wide** (e.g. 40 channels = Cg 10 × 4 groups) are tilized to `Wt_core` tiles. The pad channels are masked by the group map, which already works relative to the slice start.
- **Fix the one-pass variance cancellation.** Phase 0 computes `var = E[x²] − mean²`, which loses precision as `(mean/std)²` grows; at mean/std = 10, rel-RMS is 0.027 against the 0.02 limit. Golden `test_sharded_offset_heavy[oop-mean5/mean10]` activates with this refinement and must pass.
  - Lever: `.claude/references/numerical_stability_analysis_reference.md` §3.3 / §4.2, and the design's deferred `two_pass_centered_variance` regime.
  - The DRAM path shares the kernel, so the fix lands there too.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- **Anchor.** R1 + R2 together land the first perf target's full contract (R8: sharded RM in place). R1 also lands R9's (sharded TILE out of place). Build the sharded path to perf-conformance, not as a stub:
  - one block = the whole resident shard for TILE (`K = B`, borrowed CBs);
  - batched combine;
  - no per-tile handshakes.
- **Skill scope.** `/memory-layouts` covers only the **RM untilize-to-RM-output-shard** leg. Phase 0's outputs are TILE only. Sharding *placement* has no skill: use the CB-placement pattern named above.
- **L1 budget.** `l1_cb_budget(device)` already caps the budget at the lowest live L1 buffer, so the input and output shards are excluded from the CB budget. Borrowed CBs are not static CB space. Re-derive `K_max` for RM, which tilizes into resident tiles and re-tilizes streamed chunks from the shard in pass 2.
- **Precision fix: the formulation must not cost DRAM traffic.** A full second centered pass re-reads the `ρ·X` non-resident bytes on the streaming DRAM cases (ρ up to 0.87 on mochi), roughly +30% kernel time there.
  - Options that keep traffic flat:
    - centering only where the data is resident (shards, DRAM `K = B`);
    - a shifted one-pass `Σ(x−k)`, `Σ(x−k)²` with a per-channel pilot shift `k` per core, combined at the root with the shift correction (Chan-style);
    - reducing the TF32 truncation at the cancellation point (`ReduceFp32Mode::Accurate`, hi/lo split of the matmul operands).
  - Pick by measurement.
  - **Done-when requires no regression** (beyond 2–3% noise) on the 9 DRAM perf cases.
  - Remove the strict xfail on `offset_mean10` in `test_groupnorm_sc_N_1_HW_C_b_precision_baseline.py`; it will XPASS.
- **Program cache.** `test_sharded_program_cache_fresh_address[*-oop]` needs the shard-backed CB addresses patched on a cache hit. A new tensor at a fresh address must not reuse the stale address.
- **TILE geometries.** Model TILE geometries that are not even (`4096x960`, `1024x1920`, `4096x1152` on 8×8) are `ragged` and belong to R3. TILE-illegal ones are INVALID.

**Done when**:
- Every `memory_layout=BLOCK_SHARDED, shard_fit=even, in_place=False` cell passes. Today there are 114 such `xfail_expected` cells.
- The sharded TILE loose cases pass at `PERF_TOLERANCE`, with device-ns recorded in `changelog.md`:
  - `unet_tile_1024x1280`
  - `unet_tile_4096x640_4x8`
  - `vae_tile_16384x512`
- These regression tests pass:
  - `test_sharded_offset_heavy[oop-mean5]`
  - `test_sharded_offset_heavy[oop-mean10]`
  - `test_sharded_program_cache_fresh_address[rm-oop]`
  - `test_sharded_program_cache_fresh_address[tile-oop]`
- The precision baseline's `offset_mean10` passes without its xfail mark.
- DRAM cells stay green, and the 9 DRAM perf cases do not regress.
- `verify_supported` has no loud category.

**Outcome**: Done (2026-11-25).
- Native zero-copy BLOCK_SHARDED (even, out of place) for TILE and RM. 114/114 cells pass in `--dev` and plain mode, as do the 4 R1 regression tests and the 3 sharded TILE loose cases (1.61× / 2.96× / 2.55× faster than native).
- Shifted-moment variance fix: `offset_mean10` RMS 0.0273 → 0.0077, xfail removed.
- 9 DRAM perf cases: −3.1%…+0.9% of Phase 0 (flux1 1M×64 +0.9%, inside noise). `verify_supported` is clean.
- Finding: the non-contract `mid_*` DRAM bench shapes (~50–120 µs, latency-bound) carry the fixed pilot / seed / shift cost, +2.9…+10.6%.
- Details are in `changelog.md`.

### [ ] Refinement 2 — in_place=True on block-sharded input (both layouts), including the 16384x640 / 16384x960 L1 fit

**Goal**:
- Add `True` to `SUPPORTED["in_place"]` for `BLOCK_SHARDED` input, in both layouts. In-place on interleaved input stays a `ValueError`.
- The returned tensor lives at the input's buffer address, which the harness checks with `buffer_address()`. The input's contents are overwritten. There is no second output buffer.
- The SDXL U-Net/refiner resnet norms (RM, 8×8) must fit L1, including the two L1-budget cases:
  - `unet_rm_ip_16384x640`: shard 2048×80
  - `unet_rm_ip_16384x960`: shard 2048×120 RM, 480 KB
- Design CBs to stream within the shard, not to duplicate it.

**Implementation skill**: /memory-budget-metal

**Verifier notes**:
- **Blocking class.** Knob-turn (aliasing), per `op_design.md` "TARGET placement values as regime rows". The output-shard CB *is* the input-shard CB's bytes.
- **Write ordering.**
  - Pass 2 may write chunk j only after chunk j has been consumed.
  - For RM, keep the tilized tiles resident (`K = B`) wherever `K_max ≥ B`, so pass 2 untilizes from resident tiles into the shard rows they came from.
  - Where `K < B`, a streamed chunk must be re-tilized from the shard *before* its rows are overwritten.
- **L1 sizing.** At 16384×960 the tilized block is 64 tile-rows × 4 tiles × 2 KB = 512 KB, on top of the 480 KB shard and `F_fixed`. Compute the `K_max` and the regime explicitly and record the result in `l1_ledger.md`.
- **Order.** Depends on R1's shard-backed CBs; land R1 first.

**Done when**:
- Every `BLOCK_SHARDED, shard_fit=even, in_place=True` cell passes. Today there are 120 such `xfail_expected` cells.
- All 9 RM in-place loose cases pass at `PERF_TOLERANCE`, with device-ns recorded:
  - `unet_rm_ip_1024x1280`
  - `unet_rm_ip_16384x320`
  - `unet_rm_ip_4096x320`
  - `unet_rm_ip_4096x640`
  - `unet_rm_ip_1024x640`
  - `refiner_rm_ip_256x1536`
  - `refiner_rm_ip_16384x384`
  - `unet_rm_ip_16384x640`
  - `unet_rm_ip_16384x960`
- These regression tests pass:
  - `test_sharded_offset_heavy[inplace-mean5]`
  - `test_sharded_offset_heavy[inplace-mean10]`
  - `test_sharded_program_cache_fresh_address[rm-inplace]`
  - `test_sharded_program_cache_fresh_address[tile-inplace]`
- R1 and DRAM cells stay green.

### [ ] Refinement 3 — ragged shard geometries (padded last row/column, shards straddling images, groups straddling core columns)

**Goal**: add `"ragged"` to `SUPPORTED["shard_fit"]` for both `in_place` values. This covers three geometry cases:
- **Padded last shard row or column**, e.g. RM 586×40 with a last row of 580, or RM 512×112 with a last column of 80.
- **Shards straddling two images**, e.g. `(2,1,256,1280)` on `(8,3)`.
- **Groups straddling core columns**:
  - `(1,1,1024,1280)` on `(7,8)`;
  - the TILE-ragged model shapes `4096x960`, `1024x1920` and `4096x1152` on 8×8, where TILE padding makes the shard 128 channels wide while `Cg = 30`.

**Verifier notes**:
- **Blocking class: mixed.** Padded edges are a **knob-turn**: the Phase 0 ragged-chunk mechanism (nominal `T`, valid rows only) plus the map's channel bound. The other two are **scheme-changes**:
  - **Straddled images.** A core contributes to two units, so there are two combine rounds with different roots and two stats sets applied to disjoint row ranges.
  - **Straddled groups.** This is the design's deferred `cross_column_group_combine`. Each core holds a partial span. Partial sums must combine across column neighbours as well as down the column: a 2-D gather (row-neighbour exchange of the straddling lanes plus the column gather), then the mcast back to every core that holds part of the group.
- **No skill covers the topology.** State and build it here. `/padding-in-ttnn` (shard padding) is useful background for the padded-edge part.
- **Order.** Depends on R1 + R2: it extends their shard-backed CBs and combine.

**Done when**:
- Every `BLOCK_SHARDED, shard_fit=ragged` cell passes in both `in_place` modes. Today that is 33 + 33 `xfail_expected` cells.
- R1/R2 cells and the sharded perf cases do not regress.

### [ ] Refinement 4 — fp32 input (DRAM interleaved), fp32 γ/β, and TILE-layout γ/β

**Goal**: add three values to SUPPORTED:
- `ttnn.float32` to `SUPPORTED["dtype"]`, for DRAM-interleaved input in TILE and RM. Sharded fp32 is scope-parked INVALID.
- `ttnn.float32` to `SUPPORTED["affine_dtype"]`.
- `ttnn.TILE_LAYOUT` to `SUPPORTED["affine_layout"]`, for both bf16 and fp32 γ/β.

The output is fp32 for fp32 input. Statistics and intermediates stay Float32 with fp32 DEST. CB page formats and sizes follow the dtype.

**Implementation skill**: /numeric-formats-metal, /memory-layouts

**Verifier notes**:
- **No `compute_kernel_config` parameter.** The op prompt forbids it, so the skill's config-exposure step does not apply. HiFi3 + `fp32_dest_acc_en` stays fixed. Bundle dtype and intermediate-CB precision only.
- **Single source of page sizes.** Replace the `BF16_TILE_BYTES` literals in the descriptor (ring, group B via `group_b_bytes`, output view, map, γ/β rows) with one dtype-derived tile-size source, and re-derive `footprint_bytes` / `K_max`. An fp32 ring halves the resident chunks.
- **fp32 precision path.**
  - The fp32 RM tilize and the fp32 x operand must not lose precision silently: use `UnpackToDestFp32` where the skill says it applies.
  - Check pass-2 `x·a` against `TOLERANCES[fp32]` = (0.999, 0.01). With HiFi3, fp32 x enters SrcA as TF32.
- **TILE γ/β.** Read directly as tiles. Row 0 carries the values, matching `BroadcastDim::Row`. This replaces the RM staging copy for that layout. `/memory-layouts` covers this leg.
- **Grid fill.** Keep the full-grid split on the new dtype path.

**Done when**:
- Every `xfail_expected` cell whose only unsupported axes are among `{dtype=FLOAT32, affine_dtype=FLOAT32, affine_layout=TILE}` passes. Today that is 1 950 tile-aligned cells.
- bf16 Phase 0 / R1–R3 cells stay green, and the DRAM perf cases do not regress.

### [ ] Refinement 5 — hw_non_aligned (HW % 32 ≠ 0)

**Goal**: add `"hw_non_aligned"` to `SUPPORTED["alignment"]` for DRAM-interleaved input, in both layouts and every dtype supported at that point. This includes the real Mochi per-device shape `(4,1,6360,768)`, where `HW % 32 = 24`.
- The last tile-row of each image is partly valid. Its padding (TILE padding contents are unspecified; RM has no rows there) must not enter Σx or Σx².
- `inv_n` uses the true `HW·Cg`.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- **Mechanism** (from the design): the chunk containing an image's last tile-row uses `ReducePartialScaler::with_partial()` on both the x reduce and the x² reduce (`reduce_helpers_compute.hpp`). `/partial-scaler-reduce` has the exact scaler-pair helpers.
- **RM.** Read only the valid sticks and zero the pad rows before tilize.
- **Per-image row count.** `Ht = ceil(HW/32)` per image is already the Phase 0 row unit, so N > 1 keeps per-image padding.
- **Kernel text.** Watch the `--dev` text budget: it adds one reduce variant.
- **Order.** After R4, so the fp32 non-aligned cells land here too.

**Done when**:
- Every `alignment=hw_non_aligned` cell passes whose other axes are in SUPPORTED at that point (bf16 / fp32 input; bf16 / fp32 / TILE γ/β).
- Earlier cells stay green.

### [ ] Refinement 6 — c_non_aligned (C % 32 ≠ 0)

**Goal**: add `"c_non_aligned"` to `SUPPORTED["alignment"]` for DRAM-interleaved input, in both layouts and every dtype supported at that point. Examples: `C = 17, 47, 50, 100`, and `C = 48` with `G = 2` (Cg = 24).
- Pad channels of the last tile column never enter a group, and the output's valid channels are exact.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- **Mechanism** (from the design): the group map's channel bound masks the pad channels. `S = ceil(...)` spans, so the last slice is partial.
- **`_choose_gx` constraint.** It must keep every slice boundary group-aligned: `lcm(Cg, 32)` need not divide the padded C.
- **RM alignment.**
  - RM sticks are `C·2 B`, not 64 B multiples, so stick-segment reads must use the accessor's aligned page.
  - γ/β RM staging reads `C·2 B`.
  - Check DRAM read alignment.
- **Order.** After R5 (shared non-aligned edge handling).

**Done when**:
- Every `alignment=c_non_aligned` cell passes whose other axes are in SUPPORTED at that point.
- Earlier cells stay green.

### [ ] Refinement 7 — bfloat8_b input / gamma / beta (TILE)

**Goal**:
- Add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` (TILE only; bf8b+RM is INVALID) and to `SUPPORTED["affine_dtype"]` (TILE γ/β).
- Output is bf8b, since output dtype = input dtype.
- Statistics, x² and channel vectors stay Float32. Only the input ring, the output pack and the γ/β pages change format.
- Cells that fail out of the box (typically bf8b + non-aligned) go to `EXCLUSIONS`, not to a refinement of their own.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Reuses R4's dtype-derived page sizes. The bf8b tile is 1 088 B, which changes `K_max`.
- Scope: DRAM-interleaved only, since sharded bf8b is scope-parked INVALID.

**Done when**:
- Every `dtype=BFLOAT8_B` and `affine_dtype=BFLOAT8_B` cell passes against `TOLERANCES[bf8b]` = (0.99, 0.10), or is in `EXCLUSIONS` with a reason.
- Earlier cells stay green.

### [ ] Refinement 8 — Speed up the SDXL U-Net / refiner resnet norms (block-sharded RM, in place, 8×8)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags these nine cases as PERF FOCUS. All are bf16 with γ+β bf16 RM, BLOCK_SHARDED ROW_MAJOR, in place on 8×8, held to `PERF_TOLERANCE` (0.999 / 0.01). Native BH time is in `MEASURED_NS_BLACKHOLE`:

| Case | Native (ns) |
|---|---|
| `unet_rm_ip_1024x1280` | 41 026 |
| `unet_rm_ip_16384x320` | 256 041 |
| `unet_rm_ip_4096x320` | 81 969 |
| `unet_rm_ip_4096x640` | 88 043 |
| `unet_rm_ip_1024x640` | 40 638 |
| `refiner_rm_ip_256x1536` | 30 251 |
| `refiner_rm_ip_16384x384` | 253 781 |
| `unet_rm_ip_16384x640` | 320 169 |
| `unet_rm_ip_16384x960` | 345 006 |

Take R2's measured numbers as "before" and reduce device-ns toward or below native, using the relevant patterns in `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.
- **Small shards** (1024×N: 128-row shards, ≤ 4 tile-rows × ≤ 5 tiles per core) are combine-latency-bound: the grid-sync lamp in `op_design.md`, the gather up the column and the stats mcast back. See `tensix_all_reduce` / `mcast_topology`.
- **Large shards** (16384×N) are bound by the tilize → stats → untilize compute and its per-call overheads; see `compute_block_size` and `compute_fusion`. Block size is the whole shard.
- Fidelity (`HiFi3 → HiFi2`) is a lamp only if `PERF_TOLERANCE` still holds.

**Done when**:
- Measured device-ns (median of the trial loop) improves on the flagged cases, with before and after per case recorded in `changelog.md`.
- `PERF_TOLERANCE` still holds and the golden suite is green.
- No regression across the config-spanning guard set: one representative per kernel path × layout × placement (DRAM TILE, DRAM RM, BS TILE out of place, BS RM out of place, BS TILE in place, BS RM in place).

### [ ] Refinement 9 — Speed up the SDXL attention / VAE-mid norms (block-sharded TILE, out of place)

**Type**: perf

**Goal**: three flagged cases, all bf16 with γ+β, eps 1e-6, at `PERF_TOLERANCE`:

| Case | Grid | Native (ns) |
|---|---|---|
| `unet_tile_1024x1280` | 8×8 | 39 669 |
| `unet_tile_4096x640_4x8` | 4×8 | 136 060 |
| `vae_tile_16384x512` | 8×8 | 143 179 |

The input shard is consumed zero-copy, so these cells have no DRAM traffic at all. The time is the two compute passes plus the column combine. Optimize them toward native using `master.md`:
- the combine round trip (`tensix_all_reduce`, `mcast_topology`);
- the per-shard helper-call granularity (`compute_block_size`, `reduce_block` / `reduce_accumulate`);
- the pass-2 fusion lamp (`compute_fusion`).

No SUPPORTED change.

**Done when**: measured device-ns improves on the flagged cases, with before and after per case in `changelog.md`. `PERF_TOLERANCE` holds, golden is green, and there is no regression on the guard set (as in Refinement 8).

### [ ] Refinement 10 — Speed up the DRAM-streaming VAE / Flux / Mochi norms

**Type**: perf

**Goal**: nine flagged DRAM TILE cases at `PERF_TOLERANCE`. Phase 0 already beats native (1.96×–25×); the target is the DRAM roofline, since they achieve roughly 53–73% of nominal bandwidth today. Phase 0 numbers are in `changelog.md`.

| Case | Note |
|---|---|
| `vae_dram_1048576x128` | |
| `vae_dram_65536x512` | |
| `vae_dram_262144x256` | |
| `vae_dram_262144x512` | |
| `vae_dram_1048576x256` | |
| `refiner_dram_16384x1152` | |
| `flux1_dram_16384x256` | |
| `flux1_dram_1048576x64` | |
| `mochi_dram_8x407040x128` | 104 of 110 cores |

Relevant levers from `master.md` and the design's perf lamps:
- **Residency.** The streaming cases re-read `ρ` = 0.11–0.87 of the input.
  - Co-tune `CHUNK_TILES_TARGET` and `D_IN`. The T-scaled stream and group-B terms are 256 KB of `F_fixed`.
  - Alias `cb_sum_acc` / `cb_sumsq_acc` onto `cb_chan_scale` / group C as lockstep views and drop the publish copy. This frees `2·W` Float32 pages and is the ledger finding recorded on the `cb_sum_acc` row.
  - Consider the `cb_x_squared` Float16_b lamp only if `PERF_TOLERANCE` still holds.
- **DRAM bank contention.** The height split on Blackhole starts many cores on the same bank: see `bank_stagger`. Also `trid_double_issue` and `noc_placement`.
- **Variable `cpi` per unit**, so that `P % U ≠ 0` stops idling cores (mochi uses 104/110).

No SUPPORTED change.

**Done when**: measured device-ns improves on the flagged DRAM cases, with before and after per case in `changelog.md`. `PERF_TOLERANCE` holds, golden is green, and there is no regression on the guard set (as in Refinement 8, including the DRAM RM representative).
