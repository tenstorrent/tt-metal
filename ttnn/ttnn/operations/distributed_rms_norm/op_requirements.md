# Operation Requirements: distributed_rms_norm

## Definition
- **Formula**: for each mesh group of G devices along `cluster_axis` (device p holds `x[p]` of shape `(..., M, W_local)`):
  `ss[..., m] = Σ_{g<G} Σ_{w<W_local} x[g][..., m, w]²`,
  `out[p][..., m, w] = x[p][..., m, w] / sqrt(ss[..., m] / (G·W_local) + ε) · γ[p][w]` (γ ≡ 1 when absent).
  The divisor is the true global width `G·W_local`, never a padded width.
- **PyTorch Reference**:
  ```python
  def distributed_rms_norm_ref(x_stacked, gamma_stacked, cluster_axis, eps):
      # x_stacked: (R, C, ..., M, W_local) — what device (r, c) holds; gamma_stacked: (R, C, 1, 1, 1, W_local) or None
      xs = x_stacked.float()
      ss = (xs * xs).sum(dim=-1, keepdim=True).sum(dim=cluster_axis, keepdim=True)  # sum over the group
      W = x_stacked.shape[cluster_axis] * x_stacked.shape[-1]
      out = xs * torch.rsqrt(ss / W + eps)
      return out * gamma_stacked.float() if gamma_stacked is not None else out
  ```
  (golden: `eval/golden_tests/distributed_rms_norm/helpers.pytorch_distributed_rms_norm`)
- **Import Path**: `from ttnn.operations.distributed_rms_norm import distributed_rms_norm`
- **Function Signature**:
  ```python
  distributed_rms_norm(
      input_tensor: ttnn.Tensor, *, cluster_axis: int, gamma: ttnn.Tensor | None = None, epsilon: float = 1e-6,
      topology: ttnn.Topology = ttnn.Topology.Linear, num_links: int | None = None,
      compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None,
      memory_config: ttnn.MemoryConfig | None = None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Test-running notes for every refinement (verifier findings, see `verification_report.md`)**:
> - (Superseded by Refinement 1b: the acceptance file now takes its meshes from the root `mesh_device` fixture and runs under the default inline precompile, whole directory in one session, as the harness gate runs it.) Previously: run the acceptance file with `--no-precompile`, split by mesh shape (`-k "not test_ring and not size_one"`, `-k "test_ring and 1x4"`, `-k "(test_ring and 4x1) or test_ring_fp32"`, `-k size_one`). Re-opening a fabric mesh inside one process under `--dev` intermittently fails at `open_mesh_device` ("active ethernet core … to become active again"; reproduced with stock `ttnn.all_gather`, so it is infrastructure), and the acceptance file opens its meshes inside test bodies, which the inline precompile pass cannot collect.
> - (Superseded by Refinement 1b: graph capture now sees this op's CBs, so the golden suite runs with the default `EVAL_CAPTURE_L1=1`.) Previously: run the golden suite with `EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh …`. The harness's per-cell L1 capture cannot see this op's CBs: a per-chip (heterogeneous) MeshWorkload compiles on the mesh thread pool and `GraphTracker` is thread-local, so every cached call reports `peak_total=0` → `L1ProfilingError`.
> - Any new path with `num_waves > 1` or with `R_b ≥ 2` and `nch ≥ 2` must stay covered by `test_distributed_rms_norm_verifier_regressions.py` (alternating inputs; a stale cross-call partial is invisible when an input repeats).

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (EXCLUSION `{dtype: float32, fp32_dest_acc_en: False}`)
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = tile_aligned only; rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {TILE, "none"}; memory_layout = INTERLEAVED; cluster_axis ∈ {0, 1}; topology ∈ {Linear, Ring}; num_links ∈ {None, 1, 2}
- **Cores**: whole compute grid minus `2·L_act` port cores. R1-X (row-parallel, `split_work_to_cores` row-wise semantics) or R3-X (W-slice row-groups + in-chip root combine), selected per shape; stats all-gather over Fabric by dedicated port cores
- **Compute config**: caller's `math_fidelity` / `math_approx_mode` honoured; `fp32_dest_acc_en=True` only; `cb_stats_slots` UnpackToDestFp32
- **Golden baseline**: see `verification_report.md` → Verifier CLI Summary (supported_fail = xpass_drift = xfail_wrong_mode = 0)

### [x] Refinement 1 — Sharded placements: HEIGHT_SHARDED and WIDTH_SHARDED, native zero-copy

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` and `ttnn.TensorMemoryLayout.WIDTH_SHARDED` to `SUPPORTED["memory_layout"]`, with the output written into the caller's sharded `memory_config` (the golden suite passes the input's own memory config). Native in-kernel support only: `cb_x_tiles` and `cb_output` are placed on the input / output shards with `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy). A core's own shard is never re-read through a `TensorAccessor`.
- HEIGHT_SHARDED → regime R1-X over the shard grid: each shard is whole local rows, the per-row partial stays on its core, and the shard is the block (`R` = the shard's tile-rows, clamped only by the L1 solve).
- WIDTH_SHARDED → regime R3-X with the row-group = the shard grid, `G_w` = the number of shard columns, `S` = the shard width. Every core holds all M rows of its W-slice, so a row-group holds all rows: the in-chip combine already exists (base op R3-sharded), but the root finalizes every row. Measure whether the design's deferred **R3-RS** owners (each member finalizes `R/G_w` rows; base Refinement 6 measured 2.1–2.5× on tall shards) is needed to keep the root off the critical path at M ≥ 640.

**Verifier notes**:
- Hardest-first: this is the most structurally invasive generality item (new CB placement, regime selection driven by the shard spec, the exchange's per-row-group segment layout derived from shard geometry). R2 / R4 extend whatever structure this lands.
- **Port-core placement is the open design question.** `eval.sharding.auto_shard_config` fills the grid from (0, 0), so on larger shapes the shard grid covers the row-0 port cores. Either place the `2·L_act` port cores outside the shard grid when free cores exist, or fold the port role into a shard core's data-movement kernel (blackhole-fabric rule 7: senders on NoC1). A port core cannot simply share a core with the reader / compute / writer trio (RISCV_0 is taken).
- No implementation skill covers placement / sharding (`/memory-layouts` is RM/TILE). The local-shard mechanism is `ttnn.cb_descriptor_from_sharded_tensor` (base op Refinement 4 "Sharded TILE (native, zero-copy)"); the cross-core combine for WIDTH reuses the R3-X gather + inv distribution already in `distributed_rms_norm_reader.cpp`.
- Keep the persistent landing-scratch cache bucketed (`_landing_bucket`) — sharded inputs shrink the L1 budget further, and the solve already subtracts the scratch.

**Done when**: every golden `memory_layout ∈ {HEIGHT_SHARDED, WIDTH_SHARDED}` cell that is tile-aligned, with bf16 / fp32 input under `fp32_dest_acc_en=True`, passes (both topologies, both cluster axes, both link counts); `verify_supported` shows no loud category; acceptance and regression files green; the sharded paths read no own-shard bytes over the NoC (check the reader).
**Outcome**: landed native (cb_x_tiles / cb_output on the shards; the reader's sharded load_block issues no NoC read). Golden: all 1152 valid supported sharded cells run — 1124–1128 pass, the rest skip INFEASIBLE_L1 (fp32 512 KB–1 MB shards), 0 fail; translated width-sharded cells with a different output grid pass via the streaming writer. R3-RS measurement (traced, slowest device, 1×4 ring, 2 links, bf16): WIDTH 640×1792 with a single root = 171 µs vs 27.9 µs interleaved, so the root WAS the critical path; with R3-RS owners (20 owners × 1 row, both links) 26.3 µs. HEIGHT 640×1792 20.0 µs, 2048×1280 22.3 µs (interleaved 43.0), 4096×1024 26.1 µs. Remaining headroom: tall WIDTH (2048×1280 63 µs, 4096×1024 139 µs) is L1-bound to several blocks (a 1-tile slice carries 4 fp32 intermediate tiles per row), each paying an exchange round + ~6 helper inits; next levers: cross-block pipelining (square/collapse of block b+1 under block b's exchange, base op combine_pipeline) and 16-bit intermediates (Refinement 4).
```

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression.
**Outcome**: neither violation was an op defect. (1) Bullet 2: under the gate's inline precompile, the acceptance file's own `_MeshCache` (it opened fabric meshes inside test bodies) left the real-pass mesh dropping every host write — input, a replicated `ones` tensor, and output all read back 0 (rel-RMS 1.0) — while the root-fixture sharded file was unaffected. The acceptance file now takes its meshes from the root `mesh_device` fixture (`device_params` = FABRIC_2D_TORUS_XY + router payload; system-mesh tests parametrized to the system shape, ring tests to their place's shape); bodies and assertions are unchanged. Whole directory, default precompile: 118 passed, 1 skipped (no system axis of size ≥ 3 without a wrap link). (2) Bullet 3: 593 of the 593 golden failures (563 interleaved) were `L1ProfilingError peak_total=0`, the same as at Phase 0 (16 / 615 passed there). Graph tracking is thread-local, and a heterogeneous MeshWorkload (one program per chip) compiled and allocated its CBs on the mesh thread pool, so the capture saw no CB. Once that was fixed, the capture over-counted 4–16×: every program reported its CBs to every sub-device, and the per-core extractor summed over the bare (x, y) across chips. Fixed in infra (C++ rebuild): `MeshWorkloadImpl::compile` compiles on the calling thread while a graph capture is active; each program of a multi-program workload reports its CBs only to its own range's devices (`ProgramImpl::set_cb_tracking_devices`); `extract_resource_usage_per_core` accounts per (device, core). Full golden suite (default L1 capture + precompile): 1957 passed / 1 failed / 0 hangs of 47,025. The harness's `evaluate_completion` on that result: bullets 1, 2 and 3 pass, 1921 / 1945 responsible cells, no regression. The 1 failure is translated `test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc`: an expected-raise test where the op correctly refuses before dispatch, so the capture sees no allocation. It is not in the gate's golden set.
### [x] Refinement 2 — Row-major gamma sticks + non-tile-aligned shapes (W and H)

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["gamma_layout"]` and `"w_non_aligned"`, `"h_non_aligned"` to `SUPPORTED["alignment"]`, for every placement in SUPPORTED at the time (interleaved, and the sharded placements from Refinement 1), natively in the kernels:
- RM γ sticks `(1, 1, ceil(W_local/32), 32)`: stick r is row 0 of γ tile r. The writer reads each stick as two 16-element segments into face 0 / face 1 row 0 of `cb_gamma_tiles` page r (base op `GAMMA_DIRECT_READ = SEGMENTS`); `Row` broadcast reads row 0 only, so nothing else changes.
- `w_non_aligned`: the reader zeroes the W-pad columns of the last valid x tile of each row before the square pass (the tile padding of an interleaved / sharded input is not guaranteed zero), so pad columns never enter Σx²; the divisor is already `G·W_local` (logical). The writer still stores whole tiles.
- `h_non_aligned`: rows are independent; pad rows are computed and land in the output's tile padding. Verify the M_t geometry (`ceil` per image) on rank-3/4 shapes with batch > 1.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- One refinement because the γ stick reader and the W-pad masking are both reader/writer data-access changes on the same tiles; ROW_MAJOR here applies to the **weight** only (the input `layout` axis is TILE-only in TARGET).
- Lands after Refinement 1 so the masking is written once for interleaved and sharded readers.
- Never wrap the call in `ttnn.to_layout` / `ttnn.tilize` / `ttnn.fill_implicit_tile_padding` (one native dispatch per call is mandatory). If the masking needs more than a focused pass, partial-tick (`[~]`) with `gamma_layout = ROW_MAJOR` landed and file the alignment remainder as `Refinement 2b`.

**Done when**: all golden cells with `gamma_layout = ROW_MAJOR` and/or `alignment ∈ {w_non_aligned, h_non_aligned}` within the SUPPORTED placements and dtypes pass on both topologies; `verify_supported` clean.
**Outcome**: landed natively for interleaved, HEIGHT_SHARDED and WIDTH_SHARDED. The writer reads each RM γ stick whole and re-lays elements 16..31 into face 1 row 0. The reader zeroes the W-pad columns of tile Wt-1, in L1 for interleaved inputs and in the resident shard for sharded ones. H-pad rows need no change. Golden stratified slice: 186/186 (all 180 shape × placement × γ-layout groups). Translated + regression: 0 failed.

### [x] Refinement 3 — Speed up the perf-flagged 640×1792 ring case (knob co-tune)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` is the mandatory perf target: per-device `(1, 1, 640, 1792)`, bf16 x and TILE bf16 γ, `fp32_dest_acc_en=True` (HiFi4), INTERLEAVED, `cluster_axis=1`, Ring, `num_links=2`, on the 4-device ring (QB2: 1×4 under FABRIC_2D_TORUS_XY). Goal: move it toward `target_us = 14.1` µs (roofline 10.6 µs; baseline composite 92.1 µs). Phase-0 measured state: see `verification_report.md` → Performance. Optimize **this exact config** with the cheap knobs the planner exposed (no new topology), using `ttnn/ttnn/operations/examples/master.md`:
- **Waves (lamp L1, ⭐⭐ `double_buffer`)**: `BLOCK_ROWS_CAP` so `num_waves ∈ {2, 4}`: wave w's exchange overlaps wave w+1's DRAM reads (`x_depth = 2`). The waves path is now correct (verifier fix + regression test).
- **R3 group shape (lamp L2, ⭐⭐ `compute_block_size` / `tensix_all_reduce`)**: `R3_ROWS_PER_GROUP ∈ {1, 2, 4}`, `R3_MAX_GROUP_CORES ∈ {8, 14, 24}`: fewer, larger in-chip gathers vs. more parallel roots.
- **Port placement (lamp L3)**: Ethernet-adjacent port cores derived on the host (no probe program; blackhole-fabric rule 6 measured 48.5 vs 30.5–40.6 GB/s by placement).
- **NoC balance (lamp L8, ⭐⭐ `noc_placement`)**: reader/writer NoC pairing on the R3 row-groups.
No SUPPORTED change.

**Done when**: measured device-ns (traced, steady-state calls, `test_distributed_rms_norm_perf.py` under `--profile`, the slowest device of the group) improves on the 640×1792 case at its exact config, its golden PCC / RMS gate still holds, the golden suite is green, and there is no regression across the config-spanning guard set (one representative per distinct kernel path × layout × placement: at least R1-X 2048×1280, R3-X decode 32×1792, a γ-less cell, an fp32 cell, Linear and Ring, and the placements landed by Refinements 1–2).
**Outcome**: 640×1792 ring went from 27.3 to 22.3–22.8 µs (traced, slowest device), with no guard-set regression
(changelog table: interleaved R3-X 1024×2048 41 → 33–34 µs and 2×4×128×512 19.5 → 18.6 µs, everything else within
noise).
- **What landed**: L1 is now a real overlap. With the wave look-ahead (`WAVE_PIPELINE`), block w+1's load, combine and
  publish run under block w's exchange, and the port advances sends and notifies independently. The plan splits
  interleaved R3-X row-groups into `R3_MIN_WAVES = 2` blocks. On its own, 2 waves was 30.9 µs because the reader ran
  them back to back.
- **Nulls / losses, kept at their defaults**: L2 R3 group shape (flat or worse). L3 port placement (`PORT_X0`, flat
  within ±0.3 µs: the packets are latency-bound). L8 NoC swap (`DM_NOCS` (1, 0): 32.2 µs).
- **Bottleneck now**:
  - Wave 0's latency chain: its stats land about 13 µs after launch, including up to 4 µs of inter-chip launch skew at
    the ready fence. Its inv reaches the members at about 16.5 µs.
  - Then both waves' output DRAM writes, 2.3 MB at about 300 GB/s (about 7.5 µs), with a spatial tail: the cores far
    from the DRAM columns finish last.
- **What next, and why not here**:
  - Refinement 5's transport (direct multi-hop sends, per-entry notify, inv multicast) shortens the chain. It is a
    scheme change, out of this knob refinement's scope.
  - More than 2 waves needs more rows per row-group, and 4-row groups measured 28–34 µs: fewer cores and a longer
    in-chip chain.

### [x] Refinement 4 — Numerical configurability: bfloat8_b input and 16-bit DEST

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and `False` to `SUPPORTED["fp32_dest_acc_en"]` (bf16 / bf8b only; the `{float32, False}` EXCLUSION stays). Derive every intermediate CB page format from the DEST width in one place (today `INTERM_FORMAT = float32` is a module constant): `cb_sumsq_partial`, `cb_partial_col`, `cb_gather`, `cb_device_partial`, `cb_inv_root`, `cb_inv_rms`, `cb_x_gamma` follow DEST; **`cb_stats_slots` stays Float32 with `UnpackToDestFp32`** — the exchanged partials are fp32 on the wire and when summed whatever the flag (prompt requirement). `cb_x_tiles` / `cb_output` take the bf8b page format. The compact 128 B wire rows and the column-0 word moves in the reader assume 4-byte words in a Float32 tile — keep every tile the reader touches Float32 (or re-derive `col0_word` per format). `FinalizeCol0` / `AddCol0` already round to bf16 when `!DST_ACCUM_MODE`.

**Goal**: fix the hard violation from Refinement 1 so the completion gate's three bullets hold.

**Verifier notes**:
- Cells that fail out of the box (base-op precedent: `{dtype: bfloat8_b, alignment: w_non_aligned}` — W-pad zeroing inside a Bfp8_b tile is exponent-section work) go to `EXCLUSIONS` as structural capability gaps, not a separate refinement.
- Lands after Refinement 2 so the W-pad mask exists when deciding the bf8b × w_non_aligned exclusion.
- The fp32 output bias seen in the precision baseline (median got/true 0.9991, rel-RMS 1.1e-3, within the 5e-3 gate) comes from FPU TF32 operand truncation; this refinement should not regress it.

**Done when**: every golden cell with `dtype = bfloat8_b` or `fp32_dest_acc_en = False` passes or sits in a justified EXCLUSION; the precision matrix (dtype × DEST width × fidelity) is recorded in the changelog; `verify_supported` clean.

### [x] Refinement 5 — Exchange transport scheme for the perf-flagged case (T3)

**Type**: perf

**Goal**: the same flagged case as Refinement 3 (`(1, 1, 640, 1792)`, bf16, TILE γ, fp32 DEST, INTERLEAVED, axis 1, Ring, 2 links, toward `target_us = 14.1`). After the knob co-tune, attack the latency-bound stats exchange itself (⭐⭐⭐ `tensix_all_reduce_ring_transport`, `tensix_all_reduce` in `master.md`):
- **Direct multi-hop unicast (lamp L6)**: on the Ring, send each partial straight to every peer by destination node (first-hop direction from `ttnn.get_eth_forwarding_direction`), removing the `floor(G/2) − 1` relay-kernel turnarounds of the store-and-forward schedule. Linear stays on line neighbours.
- **Per-entry notify (lamp L4)**: ports notify roots per landed entry and roots add slots as they land (fixed g order: wait for slot g before adding it, so `inv` stays bit-identical group-wide).
- **inv distribution (lamp L5)**: multicast `inv` over hole-free row-group rectangles (port cores excluded by grid construction) instead of per-member unicast.
No SUPPORTED change.

**Verifier notes**: a scheme-change (new transport topology), so it stands alone and comes after the T1/T2 knobs. Keep the `SEM_OWN_COUNTED` wave gate (or an equivalent per-wave count) in any reworked publish path, and keep the alternating-input regression test green.

**Done when**: measured device-ns improves on the flagged case beyond the Refinement 3 result, golden PCC / RMS holds, results stay bit-identical run to run (`test_regression.py::test_deterministic`), the golden suite is green, and there is no regression across the same guard set as Refinement 3.
**Outcome**: all three levers landed, on by default as knobs (`DIRECT_RING_SENDS`, `NOTIFY_PER_ENTRY`, `INV_MCAST`;
inv multicast over the new single-rectangle R3-X row-groups).
- **Measured** (traced, slowest device, median of 6 calls, full six-shape runs):
  - 640×1792: 22.6–23.2 → 21.5–22.0 µs. Measured alone, 22.2–22.5 → 21.8–22.4 µs.
  - decode 32×1792: 11.4–11.6 → 10.8–11.1 µs.
  - 2×4×128×512: 17.9–18.2 → 17.3–17.7 µs.
  - 1024×2048: 32.7–33.4 → 31.6–32.0 µs.
  - R1-X and sharded: within noise or better.
- **Bottleneck now**: not the transport.
  - The chips launch with 2–4 µs of skew, growing about 0.4 µs per traced call. The last-launched chip finds every peer
    slot already landed (landed wait 0.13 µs), and the others' durations track its self-paced chain.
  - That chain is load → combine → publish → front(1) → back(0) (slot reads + expand + finalize + inv about 2.2 µs)
    → about 8 µs of output DRAM writes at about 270 GB/s.
- **Next, and why not here**: move the self-paced chain, not the transport.
  - Run back(0) before front(1) when wave 0's slots have already landed. The reader and compute would have to agree on
    the order (a CB-signalled schedule switch), which is a scheduling change outside this transport refinement.
  - Raise the output write rate: the far-from-DRAM tail cores finish last.
