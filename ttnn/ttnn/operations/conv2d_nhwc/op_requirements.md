# Operation Requirements: conv2d_nhwc

## Definition
- **Formula**: `out[n,ho,wo,co] = act( Σ_{kh,kw,ci} in[n, ho·s − p + d·kh, wo·s − p + d·kw, g(co)·C_in/G + ci] · W[co, ci, kh, kw] + bias[co] )`, zero outside the image, `act ∈ {identity, relu}`
- **PyTorch Reference**: `torch.nn.functional.conv2d(x.permute(0,3,1,2), w, b, stride=s, padding=p, dilation=d, groups=G).permute(0,2,3,1)` (then `relu` if `activation="relu"`)
- **Import Path**: `from ttnn.operations.conv2d_nhwc import conv2d_nhwc, prepare_conv2d_weights, prepare_conv2d_bias`
- **Function Signature**:
  - `conv2d_nhwc(input_tensor, weight_tensor, kernel_size, *, bias=None, padding=0, stride=1, groups=1, dilation=1, input_spatial=None, activation=None) -> ttnn.Tensor`
  - `prepare_conv2d_weights(torch_weight, *, kernel_size, stride=1, padding=0, groups=1, dilation=1, input_dtype, weight_dtype, input_memory_config, device) -> ttnn.Tensor`
  - `prepare_conv2d_bias(torch_bias, *, kernel_size, stride=1, padding=0, groups=1, dilation=1, input_dtype, weight_dtype, input_memory_config, device) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **One dispatch**: every path must stay exactly one `ttnn.generic_op` program per call (no `ttnn.to_layout` / `ttnn.to_memory_config` / `ttnn.fold` / reshard dispatches around the kernel).

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32]; **weight_dtype**: [float32]
- **SUPPORTED layout**: [ROW_MAJOR]
- **SUPPORTED shape-derived axes**: channel_alignment = tile_aligned; stride_mode ∈ {unit, strided} (strided added by the verifier — zero kernel change); groups_mode = dense; dilation_mode = unit
- **SUPPORTED op-specific axes**: bias_mode ∈ {with_bias, no_bias}; activation = none; memory_layout = INTERLEAVED (DRAM in, DRAM out)
- **Cores**: 2-D (output-pixel tile-rows × output-channel tiles) split over the whole compute grid (regime R0); window column clip (R10) and hard-cap L1 fallback added by the verifier
- **Compute config**: module constants `MATH_FIDELITY = HiFi4`, `FP32_DEST_ACC_EN = True` (single source; `DST_TILES` and the accumulator page format derive from it)
- **Golden baseline**: 172 supported_pass, 0 supported_fail / xpass_drift / xfail_wrong_mode; 2262 xfail_expected; 1104 invalid_skipped (per `verifier_report.json`)

### [x] Refinement 1 — BLOCK_SHARDED + WIDTH_SHARDED placement (regimes R4 / R5)

**Goal**: add `ttnn.TensorMemoryLayout.BLOCK_SHARDED` and `ttnn.TensorMemoryLayout.WIDTH_SHARDED` to `SUPPORTED["memory_layout"]`, natively. Output keeps the input's TensorMemoryLayout in L1. This is a **scheme-change**: the input-channel (K) axis is split over grid x, so each core's im2col slice has to reach every core in its grid row.
- **Input**: each core's (stick range, channel slice) shard is consumed in place through a shard-backed CB (`ttnn.cb_descriptor_from_sharded_tensor`). Never read a core's own shard back through a `TensorAccessor`. The halo rows of the window come over the NoC from the same-column neighbours' shards (BLOCK only; WIDTH holds every stick). Derive the per-core map from `input_tensor.shard_spec()`. Shards are ragged and can span several grid rows (harness `_build_memory_config`, and the pinned `extras` geometries such as 10×10 and 8×8).
- **K exchange**: each core gathers and tilizes the im2col of its own channel slice and multicasts it along its grid row (`ttnn.Mcast` + `McastArgs`). Every core accumulates all S slices' K-blocks against its own column's weight slice. A K-block is one (slice, kh) band.
- **prepare_conv2d_weights / prepare_conv2d_bias** must derive the plan from `input_memory_config` (they never see the layout). K order is `[slice][kh][kw × ci_slice]`, with each (slice, kh) band padded to whole tiles (**technique 6**: slices that are not whole tiles, e.g. SDXL's 40- and 240-channel slices). Columns are padded to `S·ceil(N_t/S)` tiles, with the trailing weight and bias tiles zero-filled (**technique 5**: C_out tiles that don't split evenly, e.g. 10 tiles over 8 slices, or `C_out = 4`).
- **Output**: a shard-backed CB on the output tensor, `ceil(N_t/S)` tiles wide per slice. Together the shards are wider than C_out; ttnn accepts that.

**Verifier notes**:
- Hardest structural refinement in the queue, so it goes first. It is the first half of the perf-1 contract: the flagged SDXL level-1 3×3 is BLOCK, 10×10 grid, shard (410, 64), bf16 activations × bf8b weights. Refinement 2 lands the other half.
- Build it on the fp32 test surface: the golden BLOCK/WIDTH INPUTS rows run at fp32 too. Implement techniques 5 and 6 here, in general form. The bf16 model-placement cases that exercise them (`_MODEL_PLACEMENT_CASES`, 40/240-channel slices) become runnable when Refinement 2 lands, and its Done-when checks them.
- Blocking (Blocking Model):
  - The shard pins m. Use the sharded default (one block = the whole resident shard), clamped only by a mechanism cap.
  - Pick K-block counts by modelled cost, not first fit (**technique 8**: per-K-block multicast handshakes dominate at small K-blocks).
  - Watch the footprint against the perf case's `achievable_l1_peak_bytes` (618,992 B). A blocking that fills L1 is a regression for a model.
- Build the path performant, not correct-only. Refinement 3 optimizes exactly this path:
  - weights read once per grid column, or multicast down it;
  - one multicast per K-block, double-buffered;
  - batched barriers on both dataflow halves.
- No implementation skill covers cross-core sharding. Measured multicast how-to: `master.md` → `mcast_topology` and `shared_input_reuse`.

**Done when**: every golden BLOCK_SHARDED and WIDTH_SHARDED cell that is otherwise inside SUPPORTED (fp32, tile-aligned, dense, unit dilation, activation none) passes, with the output in the input's TensorMemoryLayout. The interleaved cells stay green. `verify_supported` shows 0 in every loud category.

### [x] Refinement 2 — Numerical configurability (bf16 activations, bf16 / bf8b weights, mixed precision) + fused ReLU

**Goal**:
- Add `ttnn.bfloat16` to `SUPPORTED["dtype"]`, and `ttnn.bfloat16` and `ttnn.bfloat8_b` to `SUPPORTED["weight_dtype"]`.
- Pipe mixed precision (`dtype != weight_dtype`) through the matmul. It must **not** go to EXCLUSIONS: the perf-1 contract is bf16 × bf8b.
- Choose `math_fidelity` / `fp32_dest_acc_en` per (dtype, weight_dtype) through the single-source module constants. The public signature is fixed by the prompt, so the policy lives in the op. Do not add a kwarg that changes the exact signature.
- Fix intermediate-CB formats: partials page format follows `fp32_dest_acc_en`. bf16 → `elem_bytes = 2`, granule 8 elements, so `tap_width` and `band_tiles` change and `prepare_*` must use the same `conv_geometry`.
- Add `"relu"` to `SUPPORTED["activation"]` via **technique 4**: `pack_relu_config(ReluConfig::zero())` only around the final output pack (the bias pack, or the untilize pack when there is no bias), cleared before the next block's tilize and partial packs. Cover the interleaved and BLOCK/WIDTH paths.

Cells that fail out of the box go to `EXCLUSIONS`, not their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Second half of the perf-1 contract (bf16 activations, bf8b weights, bias, the fidelity the op picks). When Refinement 3 starts, that exact config must be in SUPPORTED on the BLOCK path from Refinement 1. Apply the dtypes to **every** built path (interleaved R0 + BLOCK/WIDTH), not only R0.
- Fidelity: the references ran the same dtype pair at HiFi2 (SDXL) and LoFi (ResNet50). Pick the policy by measuring PCC against `helpers.TOLERANCES` (bf16: PCC ≥ 0.995, rel-RMS ≤ 0.05), and record it in the changelog.
- **bf8b activations** are TILE-only (INVALID × ROW_MAJOR). Their only new code path is the bf8b → bf16 untilize (technique 1), so `SUPPORTED["dtype"] += bfloat8_b` is filed with the TILE refinement (Refinement 7), not here.
- ReLU is bundled here because it is a pack-config change on the same compute kernel this refinement re-derives formats in.

**Done when**:
- Every golden cell inside the new SUPPORTED set passes on the interleaved and BLOCK/WIDTH paths: all `(dtype ∈ {fp32, bf16}) × (weight_dtype ∈ {fp32, bf16, bf8b})` combinations, plus relu.
- The bf16 BLOCK `_MODEL_PLACEMENT_CASES` (8×8 grid; 40/240-channel slices; 10 or 20 output tiles over 8 slices; conv_out `C_out = 4` stays xfail via `c_out_unaligned` until Refinement 8) pass.
- The flagged SDXL level-1 3×3 BLOCK loose case passes its precision gate.
- 0 loud categories.

### [x] Refinement 3 — Speed up the perf-flagged SDXL level-1 3×3 (BLOCK)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags SDXL UNet level-1 3×3 as the first mandatory perf target (`attention: PERF FOCUS`):
- Shape: `((1, 64, 64, 640), 640, k=3, p=1, s=1, g=1, d=1, BLOCK_SHARDED)`, bf16 ROW_MAJOR activations pre-sharded at shard (410, 64) on a 10×10 grid, bf8b weights, bias on.
- `extras.achievable_ns = 149,049` (production ttnn.conv2d at 1350 MHz; scale by the measured AICLK).
- `extras.achievable_l1_peak_bytes = 618,992`.

Optimize THIS config (no proxy) toward `achievable_ns` while staying at or under the L1 peak, using the relevant patterns in `ttnn/ttnn/operations/examples/master.md`. Likely candidates for this profile:
- **Blocking by cost (technique 8).** Fewer, larger K-blocks to amortize the per-K-block multicast handshake, weight fetch and partial reload, co-tuned with bm, within the footprint target.
- `mcast_topology` / `shared_input_reuse`: weights multicast down each grid column.
- `double_buffer` / `trid_double_issue`: multicast and weight depth.
- `matmul_output_subblock`: subblock vs DEST capacity under the chosen `fp32_dest_acc_en`.
- `compute_block_size`: one tilize/untilize init per block, not per tile-row.

No SUPPORTED change.

**Done when**:
- Measured device-ns on the flagged shape improves toward `achievable_ns`, with `device_l1_peak_bytes` ≤ 618,992 (report both).
- Its precision gate still holds and the golden suite is green.
- No regression across the config-spanning guard set: one representative per distinct kernel path × layout × placement (interleaved R0, BLOCK, WIDTH; with/without bias; relu).

**Outcome**:
- Measured: 883,338 → 258,975 cycles/core (654 → 192 µs at 1350 MHz, 3.41×) against `achievable_ns` 149,049. `device_l1_peak_bytes` is 590,336 (≤ 618,992). The precision gate and golden BLOCK/WIDTH slice are green, and no guard shape regressed.
- What changed: interleaved slice K order, send-depth tilize schedule (replaces the 468 KB prefetch), run-based stateful gather, and weight stream lead.
- Bottleneck now: compute alone is 194k cycles (unpack-bound 1×2 subblock, because bm = 13 is prime). The row act exchange alone is ~174k (transfer-bound, 2.9k cycles per 78 KB block). Together they give 258k because the overlap is imperfect (likely L1 contention).
- Next would be bk = 6 or 2×2 subblocks. Both exceed the L1 target at bm = 13. A two-RISC act exchange (two blocks in flight) or a gated LoFi / bf16-DEST policy are the alternatives. Not done: they are either over the L1 target or need a new precision gate / RISC split beyond this refinement's scope.

### [x] Refinement 4 — Depthwise convolution (regime R11)

**Goal**: add `"depthwise"` to `SUPPORTED["groups_mode"]` (`groups == C_in`, `C_out` a multiple of `C_in`). This is a **scheme-change in compute**: a block-diagonal matmul would waste `C_in`× the FLOPs.
- The output tile (pixels × channels) is computed as `Σ_taps act_tap ⊙ w_tap`: an eltwise multiply with a row-broadcast of each tap's per-channel weights, accumulated in DEST (`eltwise_chain` / `BinaryFpu` with `DestAccumulation`).
- The gather emits per-tap tiles instead of im2col rows.
- `prepare_conv2d_weights` lays the weights out as `kH·kW` rows of per-channel weights (tile-row-broadcastable), and `C_out = m·C_in` multipliers are handled.
- Cover the interleaved and BLOCK/WIDTH placements. For WIDTH/BLOCK, depthwise needs **no** K exchange: each channel slice is self-contained, so no row multicast.

**Verifier notes**:
- Ordered here (hardest remaining generality) because it adds a second compute structure. The lighter axes that follow (dilation, channel alignment, TILE) then extend both structures once, instead of being retrofitted.
- Tile-aligned channel counts only. Half-tile and 1.5-tile depthwise (16 / 48 channels) is `both_unaligned` and lands with Refinement 8.
- No implementation skill covers this. `master.md` → `eltwise_l1_vs_dest_accumulate` has the DEST-accumulation syntax.
- Strided depthwise (MobileNet dw2/4/6/8) is in scope: stride is already supported.

**Done when**: every golden depthwise cell otherwise inside SUPPORTED passes on interleaved and BLOCK/WIDTH (the 1024-channel WIDTH depthwise INPUTS row included), including the strided ones and the bf16 × bf8b depthwise loose case. The dense paths are unregressed and 0 loud categories.

**Outcome**: landed. All golden depthwise cells inside SUPPORTED pass (interleaved, WIDTH 1024, strided, 5×5, bf16 × bf8b loose), plus the 99 newly reachable depthwise pytorch-sweep cells. The channel multiplier (C_out = m·C_in, m > 1) is a typed refusal: no golden or translated cell uses it (finding, not a follow-up). Perf (BH, bf16 × bf8b, per-core kernel max): 112×112×32 77 µs, 56×56×128 35 µs, 112×112×64 s2 56 µs, WIDTH 7×7×1024 10 µs, BLOCK 28×28×256 15 µs. The interleaved shapes are bound by the DRAM window fetch: one 64–256 B page per stick, with each core re-reading its halo rows (~3× at 1.1 output rows per core). `DW_CORE_M_TILES_MIN` (taller per-core blocks) is a live knob at its no-op default: 7 gives −16% on 112×112×32 but +70% on 56×56×128.

### [ ] Refinement 5 — HEIGHT_SHARDED placement (regime R3) + weight multicast (R1)

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` to `SUPPORTED["memory_layout"]`, natively.
- The input shard (a contiguous, ragged stick run, possibly shorter than one image row and straddling images for N > 1) is consumed in place through a shard-backed CB. The halo sticks the window needs come from the neighbouring shards over the NoC, one remote read per owner run.
- The im2col gather code is reused; only the window fill changes.
- Weights are read once from DRAM and multicast to every core (regime R1), on the same `cb_weight_tiles` / K-block cadence.
- The output goes into a HEIGHT-sharded L1 output through a shard-backed CB. The output geometry is the op's choice: keep the input core set, with `bm·32` rows per core rounded to whole tiles.
- Cover the dense and depthwise compute paths.

**Verifier notes**:
- Prerequisite for Refinement 6, whose four flagged ResNet50 cases are HEIGHT, and for Refinement 7 (TILE on a HEIGHT shard: technique 1; the 1×1 in-place path: technique 3).
- Sharded block default: one block = the whole shard.
- Halo traffic must be one NoC read per owner run, not per stick.
- No skill covers this placement. Weight multicast via `ttnn.Mcast` + `McastArgs` (`master.md` → `shared_input_reuse`).
- The same machinery is what R7 (interleaved → in-op HEIGHT placement) reuses later, so keep the window fill a separable stage.

**Done when**: every golden HEIGHT_SHARDED cell otherwise inside SUPPORTED passes, including:
- ragged split, odd W, a shard shorter than one image row, N > 1 straddling images;
- strided, pointwise and depthwise cells;
- relu;
- the flagged ResNet50 HEIGHT loose cases at bf16 × bf8b (precision only).

The output is HEIGHT-sharded and there are 0 loud categories.

### [ ] Refinement 6 — Speed up the perf-flagged ResNet50 HEIGHT convs

**Type**: perf

**Goal**: the four `attention: PERF FOCUS` ResNet50 HEIGHT cases in `feature_spec._SHARDED_PERF_CASES` (batch 16, bf16 ROW_MAJOR pre-sharded, bf8b weights, bias, reference LoFi), each against its `extras.achievable_ns` and `achievable_l1_peak_bytes`:

| Case | Shape | Cores | achievable_ns | achievable_l1_peak_bytes |
|---|---|---|---|---|
| layer1 3×3 | 56×56×64→64 | 105 | 28,166 | 723,456 |
| layer2 downsample | 56×56×128→128, s2 | 110 | 30,396 | 670,576 |
| layer2 3×3 | 28×28×128→128 | 98 | 19,970 | 450,928 |
| 1×1 s2 shortcut | 56×56×64→256 | 110 | 15,010 | 317,040 |

Optimize these exact configs (taken in list order; split into a follow-up if two need different levers) using `ttnn/ttnn/operations/examples/master.md`. Likely candidates:
- `compute_block_size`: whole-shard blocks, one init per block;
- `double_buffer` / `trid_double_issue`: the gather and the weight multicast;
- the gather's per-(pixel, band) L1→L1 copy count;
- `bank_stagger`: the weight DRAM read;
- the per-tile-row output path.

No SUPPORTED change.

**Done when**:
- Measured device-ns improves on each flagged case toward its `achievable_ns`, with `device_l1_peak_bytes` reported against `achievable_l1_peak_bytes`.
- Their precision gates hold and the golden suite is green.
- No regression across the config-spanning guard set (interleaved R0, BLOCK, WIDTH, HEIGHT; dense and depthwise; with/without bias; relu).

### [ ] Refinement 7 — TILE activations (+ bf8b activations): techniques 1, 2, 3 / regime R9

**Goal**: add `ttnn.TILE_LAYOUT` to `SUPPORTED["layout"]` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, in-kernel. The output takes the input's layout. A TILE input is accepted 4-D only when W and W_out are tile multiples; otherwise it is refused.
- **Sharded (technique 1)**: every core untilizes its own TILE shard (zero-copy shard-backed CB of tiles) and the halo tile-rows its window reads, once per kernel, into row-major L1. bf8b untilizes to bf16. The gather / tilize / matmul then run unchanged. The output is packed as tiles straight into the TILE output shard; the TileRowMajor partials are already the shard's tile order.
- **1×1 stride-1 (technique 3 / R9)**: the shard's tiles are the matmul in0 directly. HEIGHT multiplies them in place; BLOCK/WIDTH multicast the shard tiles. No untilize, gather or tilize.
- **DRAM interleaved TILE (technique 2)**: the op reads its TILE rows plus halo inside the **same** program and runs the sharded TILE path, choosing the placement prepare's K order matches. Output goes back to DRAM interleaved. Never `ttnn.to_layout` / `ttnn.untilize`: unavailable in some builds, and a second dispatch.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Hard dependency on Refinements 1 and 5: TILE rides on the HEIGHT/BLOCK/WIDTH paths.
- Ranked above channel alignment because it reshapes the data-access boundary of every path.
- bf8b activations are bundled here (not in Refinement 2) because their only new code is the bf8b → bf16 untilize.
- The ResNet50 model-placement cases (bf8b TILE + fused ReLU, 1×1 and 3×3 on HEIGHT and BLOCK) are the acceptance anchors.
- If the DRAM-TILE in-program path is more than one focused pass, partial-tick `[~]` with the sharded TILE paths done and file `Refinement 7b` for technique 2. Do **not** ship a `ttnn.to_layout` / `ttnn.to_memory_config` wrapper: it violates the one-dispatch rule.

**Done when**:
- Every golden TILE cell otherwise inside SUPPORTED passes (HEIGHT / BLOCK / WIDTH / DRAM-interleaved, 1×1 and 3×3, strided, relu), and so do all bf8b-activation cells.
- The ResNet50 bf8b TILE and SDXL DRAM-TILE `_MODEL_PLACEMENT_CASES` pass (conv_in `C_in = 4` once Refinement 8 lands).
- 0 loud categories.

### [ ] Refinement 8 — Channel alignment, dilation, grouped convs

**Goal**: add `"c_in_unaligned"`, `"c_out_unaligned"`, `"both_unaligned"` to `SUPPORTED["channel_alignment"]`, `"dilated"` to `SUPPORTED["dilation_mode"]`, and `"grouped"` to `SUPPORTED["groups_mode"]`, natively on every built path (interleaved, HEIGHT, BLOCK, WIDTH; ROW_MAJOR and TILE; dense and depthwise).
- **Channel alignment**: `tap_width` and `band_tiles` already encode the K padding. The reader zero-fills each window stick's pad lanes once (a zero weight times a stale NaN is NaN). The writer writes only C_out columns: the last n-slice's partial stick, and the RM output page is `C_out·elem_bytes`. C_out tile padding is zero in prepare.
- **Dilation**: a kernel-row band becomes `kW` runs of `tap_width` at stride `d`. The window bounds already take `d`.
- **Grouped**:
  - Tile-aligned per-group channels: offset each n-block's K range to its group's band.
  - Per-group channel counts that are not tiles (ResNeXt g=32 with 4 channels/group; AlexNet 48): prepare builds the zero-padded block-diagonal K order. Make the waste explicit in the changelog.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- The lightest axes, so they come last and extend the structures above once. Channel alignment is the K-dim / N-dim masking rule from the skill's non-aligned section, applied in the gather and the writer.
- Unlocks SDXL conv_in (`C_in = 4`) and conv_out (`C_out = 4`), plus the C_in=3 ResNet / AlexNet first layers (k=7 s=2, k=11 s=4).
- Bundled as one refinement because all three are gather / prepare-K-order changes on the same code.

**Done when**: every golden cell with these axis values that is otherwise inside SUPPORTED passes (the atrous + grouped loose case, the half-tile depthwise cells, the `C_in = 1` / `C_in = 3` starters). 0 loud categories.

### [ ] Refinement 9 — Speed up the perf-flagged ResNet50 BLOCK convs

**Type**: perf

**Goal**: the three `attention: PERF FOCUS` ResNet50 BLOCK cases (batch 16, bf16 ROW_MAJOR pre-sharded, bf8b weights, bias, reference LoFi):

| Case | Shape | Grid | Shard | achievable_ns | achievable_l1_peak_bytes |
|---|---|---|---|---|---|
| layer3 downsample | 28×28×256→256, s2 | 8×10 | (1280, 32) | 49,247 | 370,416 |
| layer3 3×3 | 14×14×256→256 | 8×10 | (314, 32) | 47,881 | 238,640 |
| layer4 3×3 | 7×7×512→512 | 8×9 | (96, 64) | 35,705 | 161,296 |

Optimize these exact configs toward `achievable_ns` under their footprint targets. The generated op was measured at 1.2–4.2× production's footprint on BLOCK cases, so the footprint is half the goal here. Use `ttnn/ttnn/operations/examples/master.md`:
- one-channel-tile slices: the K-block / handshake balance (technique 8);
- `mcast_topology`;
- the `trid_double_issue` L1-for-depth trade;
- `matmul_output_subblock`.

No SUPPORTED change.

**Done when**:
- Measured device-ns improves on each flagged case, with `device_l1_peak_bytes` at or under (or reported against) its target.
- Precision gates hold and the golden suite is green.
- No regression across the config-spanning guard set (one representative per kernel path × layout × placement).
