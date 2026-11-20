# Operation Requirements: conv2d_3x3

## Definition
- **Formula**: `out[h_out*W_out + w_out, n] = act( bias[n] + sum_{kh,kw in 0..2, c < C_in} in[h_in*W + w_in, c] * w[n, c, kh, kw] )`, with `h_in = h_out*s + kh*d - top` and `w_in = w_out*s + kw*d - left`. `in = 0` outside `[0,H) x [0,W)`. `act in {identity, relu}`. Padding is an int `p` (= `(p, p, p, p)`) or `(top, bottom, left, right)`. `H_out = (H + top + bottom - 2d - 1)//s + 1`, `W_out = (W + left + right - 2d - 1)//s + 1`.
- **PyTorch Reference**: `eval/golden_tests/conv2d_3x3/helpers.py:pytorch_conv2d_3x3`:
  ```python
  def pytorch_conv2d_3x3(x, w, *, input_height, input_width, bias=None, padding=0, stride=1, dilation=1, activation=None):
      H, W, c_in = input_height, input_width, x.shape[-1]
      xn = x.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
      out = torch.nn.functional.conv2d(xn, w.float(), None if bias is None else bias.float(),
                                       stride=stride, padding=padding, dilation=dilation)
      out = torch.relu(out) if activation == "relu" else out
      return out.permute(0, 2, 3, 1).reshape(1, 1, -1, out.shape[1]).to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_weights, prepare_conv2d_bias, default_compute_kernel_config`
- **Function Signature**:
  ```python
  conv2d_3x3(input_tensor: ttnn.Tensor,            # [1, 1, H*W, C_in]
             weight_tensor: ttnn.Tensor,           # prepare_conv2d_weights output, [1, 1, K_p, C_out] TILE
             *, input_height: int, input_width: int,
             bias: ttnn.Tensor | None = None,      # prepare_conv2d_bias output, [1, 1, 1, C_out], dtype = weight dtype
             padding: int | tuple[int, int, int, int] = 0,   # int p or (top, bottom, left, right)
             stride: int = 1, dilation: int = 1,
             activation: str | None = None,        # None | "relu"
             compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None) -> ttnn.Tensor
             # -> [1, 1, H_out*W_out, C_out], TILE, dtype = input dtype
  prepare_conv2d_weights(torch_weight, *, stride=1, padding=0, dilation=1, input_dtype, input_layout,
                         weight_dtype, input_memory_config, device) -> ttnn.Tensor
  prepare_conv2d_bias(torch_bias, *, <same keywords>) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **One native dispatch (run mandate)**: every path dispatches exactly one device program. A `ttnn.to_layout` / `ttnn.tilize` / `ttnn.untilize` / `ttnn.to_memory_config` / `ttnn.fill_implicit_tile_padding` wrapper around the op is therefore not available even as a partial-tick fallback. Layout, placement and padding are handled in-kernel or the refinement ships `[ ]`.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED weight_dtype**: [bfloat16, bfloat8_b] (bf8_b and the mixed bf16 x bf8_b cell promoted by the verifier after a gate-lifted probe passed)
- **SUPPORTED layout**: [ROW_MAJOR]
- **SUPPORTED memory_layout**: [INTERLEAVED] (DRAM)
- **SUPPORTED shape-derived axes**: channel_alignment = tile_aligned; hw_alignment = hw_aligned; stride_mode ∈ {unit, strided}; dilation_mode ∈ {unit, dilated}; padding_mode ∈ {valid, same} (strided/dilated promoted by the verifier; the kernel already took s, d as CT args)
- **SUPPORTED op-specific axes**: bias_mode ∈ {with_bias, no_bias}; activation = none; fp32_dest_acc_en = True
- **Cores**: M (output pixels, tile-rows) split row-major over the full live grid (regime R0), one M-block per core on every INPUT; per-core weight stream
- **Compute config**: caller fidelity/approx honored; fp32 DEST (gated True); packer-L1 fp32 accumulation over 3 kh K-blocks
- **Golden baseline**: 64 / 64 supported cells passing (1399 xfail_expected, 480 invalid_skipped, 0 loud) per `golden_results/verifier_report.json`

### [x] Refinement 1 — TILE / bf8_b activation, fused ReLU, unaligned C_in (interleaved path)

**Goal**: add `ttnn.TILE_LAYOUT` to `SUPPORTED["layout"]`, `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, `"relu"` to `SUPPORTED["activation"]`, and `"c_in_unaligned"` to `SUPPORTED["channel_alignment"]` on the DRAM-interleaved path. This moves all 416 INTERLEAVED xfail cells to passing: every dtype x weight_dtype x layout x activation x alignment combination, including the mixed bf8_b-activation x bf16-weight cells.

**Implementation skill**: /memory-layouts, /numeric-formats-metal

**Verifier notes**:
- **TILE input** is op_design regime **R3** and stays in-kernel. The M-block's footprint tile-rows (bands rounded out to 32-pixel tile rows) are read per band, with one trid fence per band, never one barrier per tile. Compute untilizes them into `cb_halo_slab` (bf16 RM), which becomes a compute->reader CB. Gather, tilize and matmul are unchanged.
  - TILE `C_in_p = roundup32(C_in)`. `per_pixel_channels`/`padded_k` already return it, so prepared weights already match.
- **bf8_b activation**:
  - `cb_im2col_tiles` and `cb_output_tiles` take the activation dtype. The untilize source CB uses the input dtype.
  - Re-quantizing im2col rows to bf8_b is exact: tap boundaries are multiples of `C_in_p` (32-aligned on TILE), so every 16-element exponent group stays inside one pixel's channels, in both layouts.
  - Mixed cells must pass in this pass. Use EXCLUSIONS only for `severity=bug` cells, never for precision misses.
- **ReLU**: MUST apply only to the final value (prompt rule).
  - With bias: `pack_relu_config(ReluConfig::zero())` immediately before `add_bias_bcast_rows`, then `none()` after it.
  - Without bias: `LastBlockTarget::OutWithRelu`.
  - The helper sets PACK_RELU and never clears it (lessons file, 2026-11-17 planner entry). Clear it after every `matmul_block` call and again at the top of the tilize `PreKBlockFn`. Otherwise the next M-block's tilize relu-clamps activations and the next K-block's partials.
- **c_in_unaligned**: lanes `C_in..C_in_p` are not data.
  - On RM interleaved they are DRAM page-alignment padding. On TILE they are tile padding. Weight rows there are zero, so `0 x Inf/NaN = NaN`.
  - The gate-lifted probe passes today only because `from_torch` writes full aligned pages (`tt_metal/impl/buffers/dispatch.cpp:108-109`). An input written by an upstream device op leaves stale bytes there (verification_report.md, probe 003).
  - Zero the lanes in-kernel, e.g.:
    - read only `C_in * 2` B per pixel into slab rows whose lanes were zeroed once at kernel start, or
    - zero-fill each tap's lane tail from `MEM_ZEROS_BASE` (512 B chunks, mechanism cap 6).
  - Either way the kernel needs the real `C_in` as a CT arg.
  - Add a regression whose input lanes hold NaN, e.g. a TILE input built with NaN tile padding, and assert a finite output.
  - The kh-block K-pad zero-fill already exists.
- **Performance bar**: this path is on Refinement 3's critical path. Keep Phase 0's full-grid M split and per-band read batching on the TILE leg. Do not tilize or untilize per tile.
- **Ordering**: before Refinement 2. Its TILE untilize stage and ReLU epilogue are what the 13 TILE bf8_b HEIGHT LOOSE layers and stem.conv1 need.

**Done when**: every golden `test_op` cell with `memory_layout=INTERLEAVED` passes (0 xfail on the interleaved path). The 9 `test_regression.py` C_in = 8 tap-shift cases run and pass. The NaN-lane regression passes. The acceptance and precision-baseline suites stay green.

**Result**: 480 / 480 interleaved `test_op` cells pass (160 INVALID skipped), all 33 `test_regression.py` cases run and pass (the C_in = 8 tap tests use C_out = 8, i.e. `both_unaligned`; `c_out_unaligned` / `both_unaligned` were promoted after a gate-lifted probe, no kernel change). NaN-lane tests pass, but on Blackhole the matmul FPU returns 0 for 0 x NaN / 0 x Inf, so they cannot fail there; the lane exclusion is kept as specified.

### [x] Refinement 2 — HEIGHT_SHARDED input and output (regime R2), all layouts

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` to `SUPPORTED["memory_layout"]` for every layout/dtype in SUPPORTED after Refinement 1 (RM bf16, TILE bf16, TILE bf8_b). This moves the 494 HEIGHT_SHARDED xfail cells to passing, including 14 Panoptic LOOSE_CASES: stem.conv1 (RM bf16, C_in = 8, stride 2) and the 13 TILE bf8_b HEIGHT layers.

**Verifier notes**:
- No implementation skill covers sharded placement. `/memory-layouts` is RM/TILE layout, not placement.
- Build per op_design **R2**:
  - **Grid and block**:
    - The grid is the input shard grid, read from `input_tensor.shard_spec()`. Never assume a grid (prompt MUST).
    - One M-block is the whole output shard (sharded block default), clamped by `select_blocking`.
    - The L1 budget must subtract the resident input and output shard bytes.
  - **Output (native placement is mandatory)**:
    - HEIGHT shard `[roundup32(ceil(M / n_cores)), C_out_p]` on the same grid.
    - `cb_output_tiles` is backed on the output shard via `ttnn.cb_descriptor_from_sharded_tensor`, so packs land in place and the writer moves no output. Routing output through TensorAccessor writes or DRAM does not count as implementing HEIGHT_SHARDED.
  - **Input**: a TensorAccessor over the sharded input reads page runs from ANY owner core.
    - `num_contiguous_pages` coalesces when `contiguous_page_stride() == 1`, which is the sharded case. One NoC read per run replaces Phase 0's one DRAM read per pixel, the measured R0 bottleneck on large M.
    - Shards are not row-aligned, and the halo reaches `d*W + d` rows, several shards away for ASPP-like reach (prompt MUST).
    - The core's own shard portion enters the slab through the same L1->L1 path. That is a layout transform (the slab is kh-banded and contiguous across shard edges), not an interleaved re-read. Zero-copy consumption of the resident shard is lamp P8 (perf). The input never round-trips DRAM.
- **stem.conv1**: shard `[4767, 8]` on 110 cores, so `C_in_p = 8` (16 B rows, L1-aligned), `K_p = 96` (kh-block pad 24 -> 32 zero-filled), stride 2. It needs Refinement 1's c_in_unaligned lane rules and ReLU.
- **L1 risk shapes**:
  - stem.conv3 (38 tile-rows per core, C_out = 128) is expected to need `bm < core_m_tiles` once shards occupy L1.
  - `semantic decoder.res2.fuse_conv.0` (C_in_p = 288, slab ~480 KB) needs `bk` at its 9-tile tap floor. The solve handles both cases if the shard bytes are subtracted.
- Update `l1_ledger.md`: `cb_output_tiles` capacity becomes the shard, and shard bytes enter the budget.
- **Performance bar**: Refinement 3 measures exactly this path. Fill the shard grid, batch halo reads per band, and keep the `cb_weights` quantum (`bk*bn`) and depth unchanged so the multicast receiver drops in.

**Done when**: every golden `test_op` cell with `memory_layout=HEIGHT_SHARDED` passes. All 14 HEIGHT LOOSE_CASES pass at their pinned 110-core geometry. Both dispatches hit the program cache. No regression on the interleaved path.

**Result**:
- 480 / 480 HEIGHT `test_op` cells pass (160 INVALID skipped). 14 / 14 HEIGHT LOOSE_CASES pass at the pinned geometry, both dispatches, cache hit. Interleaved `test_op` 480 / 480 and `test_regression.py` 33 / 33 are unchanged.
- The output CB is backed on the output shard and the writer moves no output.
- The L1 budget subtracts 2 x (input + output shard). A cache-hit re-dispatch re-checks CBs against the live shards, and the harness keeps the first dispatch's pair alive.
- On a sharded output, `bm` is restricted to divisors of the shard's tile-rows.
- `semantic decoder.res2.fuse_conv.0` solves to bm 5 of 10 and streams its weights twice (868 us). bm 10 / bk 9 measured 506 us but misses the 2-pair L1 budget by 2 KB. This is Refinement 3's starting point.

### [x] Refinement 3 — Weight multicast for the HEIGHT-sharded Panoptic layers

**Type**: perf

**Goal**:
- **Target**: no LOOSE case carries an `attention:` perf flag, so the region is free-selected. It is the HEIGHT-sharded TILE bf8_b Panoptic layers in their exact LOOSE config (bf8_b x bf8_b, fp32 DEST, HiFi4 as pinned, with bias, ReLU). The most-impacted shape is `semantic decoder.res2.fuse_conv.0`: `(128, 256, 288) -> 256`, 103 cores, 320-row shards.
- **Why weights dominate**: every core streams the full prepared weight (705 KB bf8_b), about 72 MB of DRAM reads per call. Phase 0 measured a full-grid weight stream saturating at ~190 GB/s aggregate (32x64x128: 73 of 98 us was weights), so these layers are expected to be weight-read-bound.
- **Lever**: the weights are reuse-shared across the M split (op_design regime **R1**, a `deferred` row; the prompt's "one reader core multicasting each block"). Build the read-once + multicast variant:
  - Use the multicast helper: host `ttnn.Mcast` + `attach()`, kernel `McastArgs` (`mcast/kernel/mcast_args.hpp`).
  - One injector (or a rotating one) reads each `bk x bn` K-block once and multicasts it into every core's `cb_weights` slot. Receivers push exactly as the Phase 0 writer does.
  - Catalog: `shared_input_reuse` (⭐⭐⭐: read once + mcast, double-buffer the injector), `mcast_topology` (a 1-D work split needs a 2-D rectangle multicast), `noc_placement`.
- **Measure**: validate by building the variant and measuring device-ns. Do not infer the win from a weights-stubbed ablation. No SUPPORTED change.

**Done when**:
- Measured device-ns improves on `semantic decoder.res2.fuse_conv.0` and on the other weight-dominated HEIGHT layers.
- The golden suite, including LOOSE_CASES, stays green.
- No regression across the config-spanning guard set, one representative per kernel path x layout x placement:
  - R0 RM bf16 interleaved (e.g. 32x64x128 -> 128),
  - R0 TILE bf8_b interleaved,
  - HEIGHT RM (stem.conv1),
  - HEIGHT TILE (e.g. `res3.{1,2,3}.conv2`),
  - the few-core deep-K `8x16x2048 -> 256`, where an extra semaphore round per K-block can cost more than it saves.

**Outcome**:
- **Measured** (DEVICE KERNEL DURATION, R0 per-core stream -> R1 multicast):
  - fuse_conv.0: 864 -> 466 us.
  - Other weight-dominated HEIGHT layers: semantic res2.fuse_conv.1 746 -> 397 us, semantic res3.fuse_conv.0 392 -> 145 us, semantic res3.fuse_conv.1 312 -> 110 us, instance res3.fuse_conv.0 198 -> 92 us, res3.x 79 -> 41 us, instance res2.fuse_conv.0 188 -> 154 us.
  - All 14 HEIGHT layers: 3961 -> 2539 us.
  - Guard set: 32x64x128 RM 98 -> 26 us, TILE bf8_b 98 -> 24 us. stem.conv1 214 -> 214 us. res3.x as above. Deep-K 4-core TILE bf16 465 -> 289 us: multicast wins even there.
- **Bottleneck now**: a weights-stubbed ablation matches the R1 number on every HEIGHT target (fuse_conv.0 466 vs 466 us). Those layers are compute-thread-bound (HiFi4 bf8_b matmul + tilize/untilize); stem / res2.x are bound by the slab fill / gather.
- **Next lever, not taken**: injector double-buffering (read round r+1 during round r's multicast). It would recover at most the 21 us still exposed on the 4-core deep-K shape and 1-3 us on small interleaved shapes, and none on the HEIGHT targets.
- **bm re-check**: the bm 10 vs 5 choice was re-measured; the coarser bm is now slower (fuse_conv.0 497 vs 466 us), so the L1 budget rule stays.

### [x] Refinement 4 — BLOCK_SHARDED: K split across core columns + row reduce-scatter (regime R4)

**Goal**: add `ttnn.TensorMemoryLayout.BLOCK_SHARDED` to `SUPPORTED["memory_layout"]` for every layout/dtype/activation in SUPPORTED. This moves the 489 BLOCK_SHARDED xfail cells to passing, including the 9 BLOCK LOOSE_CASES: res4.0, `res4.{1..5}` and its LoFi twin, res5.0/1/2 (d = 2, 4, 8), aspp.convs.1/2/3 (C_in up to 2048, d up to 18).

**Verifier notes**:
- This is a **scheme-change** and stands alone. The caller cut the dependent axis (c, inside K) across `n_cols` core columns, so partial sums cross cores. No implementation skill covers this class. Required topology per op_design **R4**:
  - **Per-core work**: each core builds the slab of its own channel slice (halo along its core column, from any owner, same rules as Refinement 2). It gathers its K-slice against its column's weight K-slice and produces fp32 partials `[bm, tensor_n_tiles]`.
  - **Weight layout**: `prepare_conv2d_weights` lays the weight out per column, rows ordered `(col, kh, kw, c_slice)`, keyed on `input_memory_config` (prompt MUST). `padded_k` already computes `n_cols * 3 * roundup32(3 * C_slice_p)`.
  - **Combine**: a row-local reduce-scatter. After its last K-block each core NoC-writes partial N-slice y into column y's per-peer slot CB and increments column y's semaphore. Column y waits for `n_cols - 1` arrivals, sums in fp32 (DEST), adds bias, applies ReLU, then packs into its BLOCK output shard `[roundup32(ceil(M / n_rows)), roundup32(ceil(C_out / n_cols))]`, backed on the output shard.
  - **ReLU never touches a partial** (prompt MUST).
  - Catalog for the combine: `tensix_all_reduce_compute` (accumulate directly into DST, init once per batch) and `tensix_all_reduce`.
- R4b (activation all-gather, no partial combine) is a perf alternative for the shapes where the full-C_in slab fits. It is not a substitute. R4 is the only scheme that fits aspp (its full-C_in RM slab is 3.3 MB).
- **Ordering**: hardest remaining refinement, placed right after the perf-1 slot. Land the combine on the small golden BLOCK cells first, then the LOOSE shapes.
- **LoFi twin**: runs at `math_fidelity=LoFi` with tolerance `(0.996, 0.08)`. Fidelity is honored, never gated.

**Done when**: every golden `test_op` cell with `memory_layout=BLOCK_SHARDED` passes and all 9 BLOCK LOOSE_CASES pass at their pinned grids. No ReLU is applied before the combine; a test with a negative-heavy partial distribution would expose it. No regression on interleaved or HEIGHT.

**Result**:
- 480 / 480 BLOCK `test_op` cells pass (160 INVALID skipped). 9 / 9 BLOCK LOOSE_CASES pass at their pinned grids, both dispatches, cache hit, and clean under `--dev`.
- No regression: HEIGHT LOOSE 14 / 14, interleaved + HEIGHT unit suites and `test_regression.py` 176 / 176, guard-set device time unchanged.
- ReLU after the combine only: a test whose channel slices produce large cancelling partials fails PCC when ReLU is armed on the partials, and passes as shipped.
- Built as the design's R4:
  - each core gathers its channel slice against its column's weight K-slice (rows `(slice, kh, kw, c)`);
  - the writer takes the fp32 partials (`caller_owns_pack_target`) and scatters N-slice y to column y of the row;
  - column y sums the `k_cols` slices in fp32 DEST (`reduce` AccumulateViaAdd + Skip), adds the bias slice, applies ReLU and packs its output shard in place.
- The scatter runs on both NoCs (writer: own and lower columns on NoC1; reader: higher columns on NoC0). Measured: res4.x 86.8 -> 78.7 us, res5.1/5.2 281 -> 266 us.
- Device time, Panoptic config:
  - res4.x 78.7 us (LoFi 60.2 us), res4.0 81.3 us;
  - res5.0 / 5.1 / 5.2: 210.5 / 266.3 / 265.9 us;
  - aspp.convs.1 / 2 / 3: 400.8 / 401.5 / 205.2 us.
- Weights are not on the critical path of any BLOCK LOOSE layer (stubbed: at most 13 us), so the per-column weight multicast was left to Refinement 5. The remaining scatter on res4.x is ~12.6 us; the rest is compute and the gather path. Details in changelog.md.

### [x] Refinement 5 — Speed up the BLOCK-sharded res4 layer

**Type**: perf

**Goal**:
- **Target**: free-selected (no flagged LOOSE case): `res4.{1..5}.conv2 @ LoFi`, i.e. `(32, 64, 256) -> 256`, 8x8 BLOCK grid, `[256, 32]` shards. It is the most repeated Panoptic 3x3 layer (x5), in Panoptic's real compute config.
- **Costs after Refinement 4**: a ~229 KB fp32 partial combine per core, and every core of a column streaming that column's weight K-slice from DRAM.
- **Levers** (pick from `ttnn/ttnn/operations/examples/master.md`):
  - multicast each column's weight K-slice down the column. The slice is reuse-shared along the column's M split (`mcast_topology`: 1-D channels on a 2-D split; `shared_input_reuse`).
  - the R4b activation all-gather where the full-C_in slab fits: res4.x exchanges ~92 KB of activation instead of 229 KB of partials (op_design R4b row).
  - the combine's DST accumulation shape (`tensix_all_reduce_compute`).
- **Scope**: one ⭐⭐⭐ lever per phase. No SUPPORTED change.

**Done when**: measured device-ns improves on `res4.{1..5}.conv2 @ LoFi` with its `(0.996, 0.08)` tolerance still holding. The golden suite stays green. No regression across the guard set (Refinement 3's set plus one BLOCK representative per layout).

**Outcome**:
- **Measured** (DEVICE KERNEL DURATION, same session, both levers off -> on):
  - res4.{1..5}.conv2 @ LoFi: 60.3 -> 35.2 us. The tap-major gather alone gave 41.7 us; the column weight multicast took it to 36.0 us.
  - res4.x @ HiFi4: 78.8 -> 54.1 us. res5.0: 211.3 -> 141.5 us. res4.0 (stride 2, gather lever gated off): 81.5 us, unchanged.
  - Output is bit-identical to the old path. res4.x LoFi PCC 0.99954, rel-RMS 0.0186.
  - Nothing regressed across the 41-shape perf harness (worst +0.5%). Shared-path wins elsewhere include HEIGHT stem.conv2 264 -> 126 us, stem.conv3 331 -> 202 us, res2.x 76.5 -> 38.1 us, center_head.1 79 -> 45.5 us, and interleaved 128x256x64 TILE bf8_b 93.8 -> 52.4 us.
- **Bound found**: the reader's im2col gather, at one 192 B L1 read per output pixel per kh (768 per core, 37 us). Compute's tilize waited on it. The suggested levers did not target it:
  - weights were hidden behind the gather;
  - R4b multiplies the gather and tilize by `k_cols`;
  - the combine's add is ~1.4 us.
- **Levers built**:
  - Tap-major im2col: one read per tap per image-row run, about 100 reads per core.
  - With the gather fast, the per-core weight stream was exposed (5 MB from DRAM on 64 cores, about 21 us; stubbing it gave 41.7 -> 36.1 us), so the named column multicast was built next. It is one channel with one group per grid column and measured 36.0 us, equal to the stub.
- **Bottleneck now** (LoFi):
  - The partial scatter: stubbing its writes gives 35.2 -> 23.6 us (HiFi4 54.1 -> 42.4 us). The all-to-all puts 512 KB per direction on the middle link of each 8-core row.
  - Then a compute-bound matmul phase of ~18 us (tilize ~8 us + matmul ~10 us), then the 3.4 us slab untilize.
- **Next, not taken** (each changes the combine protocol, beyond one ⭐⭐⭐ lever per phase):
  - Pipeline the scatter under the last K-block, with per-row-group partial pushes.
  - Use a reduce-then-forward topology that cuts the middle-link load.
  - bf16 partials would halve the bytes, but that is a precision change.
- **Gather gates**: res5.1/5.2 and aspp.convs.1/2/3 still use the per-pixel gather.
  - res5.1/5.2: L1-solved `bk` 3 splits a 2-tile tap. A tap-aligned `bk` preference in `select_blocking` would enable it.
  - aspp: the last channel slice is short, and the one shared compute kernel fixes the tilize width. Per-slice-width compute kernels would enable it.

### [x] Refinement 6 — Speed up stem.conv1

**Type**: perf

**Goal**:
- **Target**: free-selected: `stem.conv1`, i.e. `(512, 1024, 8) -> 64`, stride 2, HEIGHT RM bf16 x bf8_b weights, 110 cores, `[4767, 8]` shards. It is the largest spatial layer.
- **Bound**: fixed cost, not bytes.
  - `tensor_kkh_tiles = 1`, so every M-block pays 3 tilize inits, 3 matmul inits and 3 packer-L1 passes over `bm x bn` fp32 tiles for a single K tile each.
  - The gather issues one fixed-size 48 B read per output pixel per kh.
- **Levers** (several ⭐/⭐⭐ ones fit one phase):
  - lamp P1: `bk = tensor_k_tiles`, one K-block per M-block, allowed by mechanism cap 3; catalog `compute_block_size`.
  - lamp P7: `set_read_state` / `read_with_state` for the constant-size combined read.
  - lamp P2: `im2col_rm_depth_rows = block_m_tiles`.
  - lamp P8: gather in-shard taps straight from the shard-backed CB.
  - `trid_double_issue` / `split_reader`, once the gather is classified issue-bound. Use the lessons-file method: same reads, local-L1 source.
- **Scope**: no SUPPORTED change.

**Done when**: measured device-ns improves on stem.conv1 with the golden suite green and no regression across the guard set (Refinement 3's set).

**Outcome**:
- **Measured** (DEVICE KERNEL DURATION, Blackhole p150b): stem.conv1 214.5 -> 24.7 us (8.7x). The bound was per-pixel work, not bytes: 3648 x 48 B gather reads per core, a per-row loop and 114 one-tile tilizes. Stubbing the gather reads gave 128 us; matmul `SKIP_COMPUTE` changed nothing.
- **Levers built**:
  - face-direct im2col: stride * C_in_p == 16 makes each 16x16 face one contiguous slab run, so the reader writes in0 tiles directly and the tilize goes (99.5 us with P1);
  - P1 whole-K K-block;
  - a lean interior path (63.7 us);
  - P7 stateful reads (60.6 us);
  - the slab fill split by NoC, with the writer reading higher-index shards on NoC1 (53.2 us);
  - the gather split across reader and writer (36.3 us);
  - in0 streamed per 2-tile-row group, with the bias add per 5 groups (24.7 us).
- **Not built**: P2 (no `cb_im2col_rm` on this path) and P8 (the slow cores' footprints lie in other shards).
- **Guard set**: within noise of a same-session baseline. aspp.convs.1 +1.0%, center_head.1 -2%, every other shape within ±2%.
- **Bottleneck now**: co-bound within ~2 us.
  - Compute is 24.0 us with the face reads stubbed: HiFi4 matmul ~11 us, bias add 3.3 us, ~0.36 us fixed cost per streamed matmul call.
  - The gather is ~17 us on two RISCs at ~84 cycles per 512 B face read, plus ~4 us of exposed slab fill.
- **Next, not taken**: a progressive per-subblock in0 wait inside one `matmul_block` call (a shared-helper change) plus a bias fold into the matmul. Each alone moves the wall by at most ~2 us, because the other half then binds.

### [x] Refinement 7 — Per-side padding (padding_mode "asymmetric") for FLUX.2, interleaved path

**Goal**:
- Add `"asymmetric"` to `SUPPORTED["padding_mode"]`: padding given as `(top, bottom, left, right)` whose sides differ, each side 0 or `dilation`. The tagger and `output_hw` already classify and size it (`conv2d_3x3.py`: `padding_sides`, `canonical_padding`, `tag_padding_mode`); the kernels do not handle it yet, so every such cell is refused today.
- Required: every asymmetric cell on DRAM-interleaved input, i.e. what FLUX.2 runs:
  - 72 interleaved `test_op` cells from three INPUTS:
    - `(18, 32, 32) -> 32, (0, 0, 1, 1)`;
    - `(34, 64, 64) -> 128, (0, 0, 1, 1)`;
    - `(32, 32, 64) -> 64, (0, 1, 0, 1)`, stride 2.
  - 20 FLUX.2 LOOSE_CASES:
    - the 3 encoder downsamplers (`(0, 1, 0, 1)`, stride 2, up to 1024x1024 x 128);
    - the 17 Blackhole (2, 4) per-device decoder convs (`(0, 0, 1, 1)`, H/2 + 2 rows, bf16 TILE, HiFi4, bias), which are FLUX.2's `[perf]` cases.
- Sharded x asymmetric (144 `test_op` cells): optional. No model needs it; EXCLUSIONS are allowed (prompt rule). If excluded, say so in the result and tick `[x]` — the goal is the interleaved path.

**Implementation skill**: /padding-in-ttnn (only for the tile / page padding it disambiguates; this padding is algorithmic)

**Verifier notes**:
- **Where padding enters**: `Conv3x3Geometry.padding` (an int today) feeds the tap coordinates, the path gates and the slab layout in `conv2d_3x3_program_descriptor.py`:
  - coordinates: `_block_first_tap` (`(ho*s - p)*W + wo*s - p`) and the in-image row checks near lines 693 and 1353;
  - slab gaps: `slab_direct_layout` (`over` / `gap`);
  - path gates on `padding == dilation`: `padded_slab_capable`, the s2-taps, kw-fold and slab-direct eligibility, around lines 934-1029 and 1985;
  - kernel CT args / defines: `PAD_SLAB_PAD`, `SD_PAD`, and the positional padding args around lines 3027 and 3186.
- **The algebra**:
  - Per-side padding changes the coordinates only through `top` (rows) and `left` (columns): `h_in = h_out*s + kh*d - top`, `w_in = w_out*s + kw*d - left`.
  - `bottom` and `right` change only `H_out` / `W_out` (already right in `output_hw`) and how far the last row's / column's taps reach past the image edge (the right-edge zero reach in `slab_direct_layout`'s `over`).
  - Zero fill stays "outside `[0, H) x [0, W)`".
  - `(0, 0, 1, 1)` is "valid" vertically and "same" horizontally.
- **Shape it into the geometry**: carry `pad_top` / `pad_left` (and the bottom / right reach where a layout needs it) instead of one `padding`. Keep every int-padding call on exactly today's path.
- **Path selection**: a fast path whose gate is `padding == dilation` may either learn the per-side form (preferred where it is only coordinate algebra) or stay closed for asymmetric calls. The general path must be correct for any side in `{0, d}`.
- **Performance bar** (so step 4 does not start from a slow fallback):
  - Each FLUX.2 `[perf]` case must run within ~5% of its `measured_ns_blackhole` in `feature_spec.py`. That number is the same conv on the H/2 interior rows with symmetric padding 1 — same output, same MACs — so asymmetric padding must reuse the symmetric path's speed, not a per-pixel fallback.
  - The harness that produced those numbers is `tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_flux_conv3x3_compare.py` (run with `scripts/run_safe_pytest.sh --profile --run-all`, select cases with `FLUX_CMP_CASES`). Its conv2d_3x3 side currently runs the symmetric proxy; switch it to the real `(0, 0, 1, 1)` / `(0, 1, 0, 1)` call once supported.
- **Panoptic must not move**: every Panoptic call is int padding.
  - Its 45 LOOSE_CASES must stay green and their outputs bit-identical.
  - Their device time must stay within noise (±2%). Measure the 22 cases with `tests/ttnn/unit_tests/operations/conv2d_3x3_vs_ttnn/test_panoptic_conv3x3_compare.py` before and after (`--profile`, `CONV_CMP_CASES` 0-10 and 11-21).
  - In a worktree, `export TT_METAL_HOME=$PWD PYTHONPATH=$PWD` first, or the run silently tests the main checkout.

**Done when**:
- The 72 interleaved asymmetric `test_op` cells and the 20 asymmetric FLUX.2 LOOSE_CASES pass, both dispatches, with a program-cache hit.
- Every previously passing cell still passes (golden was 1604 passed / 236 xfailed / 576 skipped before this refinement).
- The FLUX.2 `[perf]` cases meet the performance bar.
- The Panoptic `[perf]` cases are within noise.

**Outcome**: per-side padding is coordinate algebra on the interleaved walk (rows by `top`, columns by `left`); sharded x asymmetric is EXCLUSIONS. All 17 FLUX.2 bh2x4 `[perf]` cases, real `(0, 0, 1, 1)` call, median of 3: -0.44% to +0.72% of `measured_ns_blackhole` (e.g. up3.res 1026x2048 32->128 3584587 vs 3583793 ns, up2.upsample 1026x2048 64->256 9657214 vs 9659221 ns), PCC 0.999999. Panoptic: the 45 LOOSE outputs are bit-identical (sha256) and the 22 compare cases move -1.7% to +1.75% (same kernels; noise). Not covered by a bar: the encoder downsamplers `(0, 1, 0, 1)` s2 are 13-17% slower than the symmetric proxy at 512² x 256, 1024² x 256 and 2048² x 128 (flat elsewhere), because left pad 0 closes S2_TAPS and they take the strided window. Next: teach S2_TAPS the left-pad-0 layout (kw = 0 even pixels, kw = 1 odd, kw = 2 even shifted one with the next input tile's pixel 0 in place of the previous tile's pixel 31). Not done here: it is a new DEST/packer move pattern in raw LLK, not coordinate algebra, and no FLUX.2 perf case runs it.
