# Changelog: distributed_rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-10-20
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  - One fused `ttnn.generic_op` (MeshProgramDescriptor) per call, regime R1 resident-flat from `op_design.md`.
  - Per chip: a worker grid of row-groups × W-chunks from the host solve, and 2 port cores per link exchanging the fp32 per-row partials over Fabric 2D. Ring or Linear schedule, with a ready fence and re-armed arrival counters.
  - One leader per row-group combines `G · num_w_chunks` partials and fans inv_rms out to its row-group.
  - x stays resident in L1 (one DRAM read), and the `x ⊙ gamma` stall-shadow is computed before the stats arrive.
- **SUPPORTED at Phase 0** (after the verifier's drift fixes):
  - dtype = [float32, bfloat16, bfloat8_b]
  - fp32_dest_acc_en = [True]
  - layout = [TILE]
  - alignment = [tile_aligned, h_non_aligned]
  - rank = [2, 3, 4]
  - gamma_mode = [gamma, no_gamma]
  - gamma_dtype = [float32, bfloat16, "none"]
  - gamma_layout = [TILE, "none"]
  - memory_layout = [INTERLEAVED]
  - cluster_axis = [0, 1]
  - topology = [Linear, Ring]
  - num_links = [1, 2, None]
  - EXCLUSIONS: `{float32, fp32_dest_acc_en=False}`
- **Accuracy achieved** (4 shapes × {bf16, fp32}, 2×2 mesh, G = 2, bf16 gamma, `test_distributed_rms_norm_precision_baseline.py`):
  - bf16: PCC ≥ 0.999998, max_abs_err ≤ 0.049, rel_rms_err ≈ 0.0020, ≤ 1 ULP.
  - fp32: PCC 1.000000, max_abs_err ≤ 0.032, rel_rms_err ≈ 0.0014. The got/true median is 0.9988, a −0.12% TF32-truncation bias of the FPU multiply operands (probe-confirmed precision effect, not a scale bug).
- **Golden suite at Phase 0**: 1283 supported_pass, 8 supported_fail, 11790 xfail_expected, 33696 invalid_skipped, 0 xfail_wrong_mode (`verifier_report.json`, `EVAL_CAPTURE_L1=0`). The runner reports 1320 passed / 8 failed / 0 hangs.
  - The 8 failures are the op's L1 refusal at per-device (2,1,2048,4096) (refinement 1).
  - The one xpass_drift is a negative-test false positive.
- **Perf** (1×4 ring, G = 4, 2 links, bf16; last-launched chip, untraced): 640×1792 22.2 µs, 32×1792 10.5 µs, 2048×1280 48.7 µs. Targets: 22.2 / 10.9 / 37.6 µs; composite baselines: 92.1 / 87.2 / 93.0 µs.
- **Issues encountered**:
  - **Harness, not op**: the runner's default L1 capture fails every fabric-mesh test with `L1ProfilingError: peak_total=0`, because graph capture records no CB allocation on the mesh; native `ttnn.add` reproduces it. Golden runs use `EVAL_CAPTURE_L1=0`.
  - **Drift fixes (verifier)**:
    - `num_links: None` added, because the harness records the raw default kwarg;
    - `h_non_aligned` and `bfloat8_b` promoted after a validate-bypass probe and golden confirmation.
  - **Code-review fixes**: removed the dead `HALF_ROW_BYTES` constant; documented `L1_BUDGET` as a conservative host constant and corrected the ledger's symbol table.
  - **Deferred to refinement 1**: the collapsed blocks-per-core / `rows_per_round` knob.
- **Tests added**:
  - `test_distributed_rms_norm.py` (acceptance, planner)
  - `test_distributed_rms_norm_precision_baseline.py`
  - verifier probes under `probes/` (`probe_unlock.py`, `probe_ratio.py`, `probe_l1_capture.py`)

## Refinement 1 — Multi-block per-core schedule (L1 fit for tall × wide shapes)
- Date: 2026-10-20
- What was done:
  - **Solve**: the Phase 0 one-block-per-core search runs first, unchanged, and wins whenever any pair fits. Otherwise it searches `(block_h, num_w_chunks)` with `num_rounds = ceil(num_row_groups / num_slots)` row-blocks per core (row-block b → slot b % num_slots, round b // num_slots), keyed by per-core tiles, then rounds. New host knob `MIN_ROUNDS` (default 1) replaces the dead `ROWS_PER_ROUND` and can force multi-round.
  - **Kernels (all four)**: loop over `num_rounds` rounds; each round's tile rows are derived in-kernel from `row0 + r·round_stride` and `tensor_h_tiles` (CT). An empty last round still counts for the port/leader counters. One board region per round (`num_rounds·G·slot_bytes`), so rounds need no fence between them.
  - **Round order**: with gamma, x(r) is consumed by the stall shadow, so round r+1's load/sumsq/publish run before round r's combine/normalize, and round r+1's exchange overlaps round r's normalize/store. `cb_inv_row`/`cb_inv_col` hold 2 rounds on that path; the leader fans out into half r%2. Without gamma, rounds run back to back.
  - **Synchronization**: the port-local and board-ready counters have several sources, so they are counted per round parity (two new semaphores; a worker is at most one round ahead). Arrival and inv-ready have a single source each and stay cumulative. All counters are re-armed at the end of the call.
  - **CBs**: `cb_x_block` pushes its whole capacity per round (compute pops the pad pages unread), so a ragged W-chunk keeps every round's block at the FIFO base. Gamma is read once per call and popped once at the end.
  - Reused: every CB, helper call and block operation, the board/port protocol and the fence. Added: the round loop, per-round board regions, parity semaphores, and inv double-buffering (gamma multi-round only).
- Accuracy achieved: acceptance PCC thresholds (bf16 ≥ 0.995, fp32 ≥ 0.999) met on (2,1,2048,4096) bf16+gamma, (1,1,8192,4096) bf16 ± gamma, (1,1,4096,4096) fp32+gamma, and forced 2–3 rounds on 640×1792, 200×1312 and 2×4×64×256 (gamma/no gamma, Linear and Ring mocks). The precision baseline is unchanged (8/8 pass).
- Golden test progress: the 8 previously failing `test_distributed_norm_allclose[*-8192-2048-2]` cells pass, and the LOOSE cases pass. The full suite was not re-run; the harness re-runs it.
- Perf (last-launched chip, untraced, 1×4 ring, 2 links, bf16): 640×1792 21.1 µs (was 22.2), 32×1792 10.1 µs (was 10.5), 2048×1280 46.7 µs (was 48.7). All three still solve to one round.
- Issues encountered: the first multi-round version raced intermittently on the gamma path (PCC 0.985 on one device). Cumulative multi-source counters let a worker's round r+1 increment satisfy round r's threshold; parity counters fixed it.
- Tests added: `test_distributed_rms_norm_multiround.py` (tall shapes, forced rounds on Linear and Ring, back-to-back bit identity with 3 rounds).

## Refinement 1b — Multi-block per-core schedule (debug: fix gate violations)
- Date: 2026-10-20
- What was done: the gate failed because 1051 SUPPORTED golden cells raised `L1ProfilingError: peak_total=0` under the harness's default L1 capture. The Phase 0 baseline failed the same way, so this was not a Refinement 1 kernel regression. Fixes:
  - `tt_metal/distributed/mesh_workload.cpp`: while a graph capture is active, compile heterogeneous MeshWorkloads on the calling thread. `GraphTracker` processors are `thread_local`, so CB allocations made on pool workers were never recorded.
  - `ttnn/core/graph/graph_trace_utils.cpp` (`extract_resource_usage_per_core`):
    - per-core program L1 is keyed by (device, core);
    - an exact duplicate CB/DFB/scratchpad report (device, address, size, cores) is counted once per epoch;
    - placeholder address-0 reports are never deduplicated.
    Before this, one core was summed across chips and across every program of a mesh workload (16× for this op on 4 chips), and large shapes would have tripped the harness's 2 MB sanity bound.
  - `distributed_rms_norm_program_descriptor.py::_board`: one persistent board per (mesh, port cores), grow-only (allocate the new one, then free the old). The size-keyed cache leaked one L1 board per distinct size until a later program's CBs collided with it, which hit the 8 `*-8192-2048-2` cells in a full run.
  - Unit tests: the fabric mesh fixtures are registered once in the directory `conftest.py`. Per-module imports created two session caches and TT_FATAL'd when the directory ran in one session.
- Accuracy achieved: unchanged. Acceptance and precision-baseline thresholds hold (bf16 PCC ≥ 0.995, fp32 ≥ 0.999).
- Golden test progress: full suite, `EVAL_CAPTURE_L1=1` (harness default): 1327 passed / 1 failed / 0 errors / 0 hangs (was 41 passed / 1078 failed). The failure is the negative test `test_pre_all_gather_non_welford_fp32_requires_fp32_dest_acc`: the op correctly refuses before allocating, so peak = 0 and the harness raises.
- Issues encountered: the harness artifact above. Its fix belongs in `eval/l1_profiling.py`, which this run must not modify.
- Tests added: none. `tests/ttnn/unit_tests/operations/distributed_rms_norm/` passes as one session: 98 passed, 1 skipped. `test_graph_trace_utils.py`: 19/19 pass.

## Refinement 2 — Sharded input/output placement (HEIGHT_SHARDED + WIDTH_SHARDED)
- Date: 2026-10-20
- What was done:
  - **SUPPORTED**: `memory_layout` += HEIGHT_SHARDED, WIDTH_SHARDED (L1, ROW_MAJOR orientation). The output follows a sharded input (the default for `memory_config=None`); an interleaved output is also accepted. A sharded output must equal the input's memory config.
  - **Native consumption**: `cb_x_block` = `ttnn.cb_descriptor_from_sharded_tensor` on the input (zero-copy). A round's load is a push of its pages, with no NoC read. A sharded output backs `cb_out_tiles`: compute packs straight into the shard and the writer stores nothing.
  - **Workers come from the shard spec** (`plan_sharded`):
    - HEIGHT (R9): one row-slot per shard, `num_w_chunks = 1`; rounds walk the shard's rows contiguously, with a per-core `[row0, row_end)` replacing `tensor_h_tiles` in every kernel.
    - WIDTH (R10): one W-chunk per shard and one row-slot; rounds walk all rows. The padded last shard of `Split.PAD` (8192 wide → 86 × 3-tile shards) computes its whole width: the reader zero-fills its pad columns once, and its gamma pad pages.
  - **Ports** go on the last `2·num_links` cores of the grid, past the row-major shard grid. The op refuses with a clear error if the shard grid reaches them.
  - **Pre-combine (R3)** for the WIDTH board cap. Tall WIDTH shapes would need `G·tensor_h_tiles·num_w_chunks·128 B` of board (up to 1 MiB), so the solve switches to a leader-side pre-combine (`PRE_COMBINE` knob, auto):
    - workers write their partial half-rows into the leader's `cb_pregather` (row = chunk, two round halves, per-parity `sem_pre_ready`);
    - the leader reduces them (REDUCE_COL, Accurate) into a second `cb_partial_row` push and publishes one vector per row.
  - **Streamed gamma** for wide HEIGHT shards (`GAMMA_STREAM` knob, auto). Resident gamma plus the fp32 stall shadow do not fit next to the resident x and out shards (4096/8192-wide rows), so:
    - gamma slivers are re-read per tile row in `gamma_batch` batches;
    - normalize is one fused pass, `(x · gamma[Row]) · inv_full` (`UnaryBcast` Col expands inv in place; `DestReuseBinary`);
    - no `cb_xg`; rounds run in the no-shadow order.
  - **Sharded L1 solve**: it budgets the real room from `ttnn.get_memory_view` (allocator base → lowest live L1 buffer, with the output allocated first) and charges board growth, or re-sizes a stale larger board. Search order: no pre-combine before pre-combine; resident gamma before streamed (batch 8 → 1); then the largest `block_h`.
  - **Board**: one persistent board per mesh, following the current port set (new before old is freed). An L1 board reserves its address range on every core, so a second board for the sharded port set cost every core up to 256 KiB.
  - Reused: every block operation, the round loop, the exchange/fence protocol, the leader combine and fan-out, and the interleaved solve (unchanged). Added: the sharded CB descriptors, `plan_sharded`, `cb_pregather` + 2 semaphores, the streamed-gamma normalize and `cb_inv_full` (alias group B).
- Accuracy achieved: acceptance PCC thresholds (bf16 ≥ 0.995, fp32 ≥ 0.999) on HEIGHT {640×1792, 4096×1024, 100×512, 2×4×128×512, 128×4096, 32×8192} and WIDTH {32×1792, 640×1792, 4096×1024, 32×8192, 2×4×128×512}, bf16 / fp32, ± gamma. Forced modes (pre-combine with 1 and 3 rounds, streamed gamma, multi-round HEIGHT with an empty last round) pass, and back-to-back calls are bit-identical. The golden tolerances hold on every target cell.
- Golden test progress:
  - sharded target cells: 2136 / 2160 pass, 24 `infeasible_skipped` (fp32 32×8192 HEIGHT, single-core 2 × 1 MiB shards, harness-side);
  - `test_sharded_distributed_layernorm` passes; `test_rms_fuse*` still xfail on ROW_MAJOR gamma (refinement 4);
  - interleaved regression slice (LOOSE cases, ring mock, 640×1792 / 4×1×512×512 / 100×512 cells): 135 passed.
- Issues encountered:
  1. **No-pop reduce packs linearly.** `reduce(WaitUpfrontNoPop)` bulk-reserves its outputs and packs page `o` linearly from the write pointer. On the pre-combine leader, the `cb_partial_row` stream interleaves a short last-round partial with a later full combined block, so the pack ran past the FIFO end (PCC 0.4). Fixed by padding every pre-combine push to `block_h`.
  2. **CB region clash.** `L1_BUDGET` ignores live L1 buffers, so the CB region clashed with L1 buffers when shards were large. Fixed by the measured budget.
  3. **L1 fragmentation.** The persistent board and global semaphores can sit mid-L1 after earlier calls. The x and out shards then land lower, and the solve halves the streamed-gamma batch to fit.
- Tests added: `test_distributed_rms_norm_sharded.py` (60 cases: shapes × dtype × gamma, forced knobs, back-to-back), `test_distributed_rms_norm_sharded_debug.py` (per-tile-row error map, kept for future debugging).

## Refinement 2b — Sharded input/output placement (debug: fix gate violations)
- Date: 2026-10-20
- What was done:
  - **Why the gate failed.** 207 prior-passing INTERLEAVED cells (plus 4 ring-mock cells) failed with `L1ProfilingError: peak_total=18446744073709…`, an underflow. The kernels were correct.
  - **What triggered it.** Refinement 2's single persistent board follows the port set. A port-set change (`num_links` 1↔2, interleaved↔sharded ports) therefore reallocates the board and frees the old one inside the next call's graph capture.
  - **Where it broke.** `extract_resource_usage_per_core` subtracted every L1 deallocation from its `size_t` resident base. A buffer allocated before the capture was never added to that base, so freeing it wrapped the base.
  - **Fix.** The reducer now only releases L1 buffers whose allocation it recorded in this capture (map keyed by buffer node id). Deallocations of buffers allocated before the capture are ignored.
  - Op code and kernels are unchanged.
- Accuracy achieved: unchanged. Every golden cell passes at its golden tolerance.
- Golden test progress: full suite with `EVAL_CAPTURE_L1=1`, run as 4 memory_layout chunks: 3409 passed, 0 failed, 0 hangs (INTERLEAVED 1083, HEIGHT_SHARDED 996, WIDTH_SHARDED 1080, other 250). The harness run had 3041 passed and 214 failed.
- Issues encountered: the one-session full suite exceeds the agent tool's 10-minute command cap, so it was run in 4 chunks that together cover every cell.
- Tests added: none. `tests/ttnn/unit_tests/base_functionality/test_graph_trace_utils.py` passes 19/19.

## Refinement 3 — Speed up the perf-flagged profile (640 × 1792, G = 4 ring)
- Date: 2026-10-20
- **Method**: every number is on-device kernel time from the realtime program profiler (one record per chip program), `tests/.../test_distributed_rms_norm_perf.py`, on a 1×4 ring stand-in (QuietBox 2, FABRIC_2D_TORUS_XY), bf16, untraced.
  - Reported value: the **last-launched chip** (the minimum over the 4 chips), median of 3 calls. The earlier chips wait at the Fabric exchange for it, so the minimum excludes host dispatch skew.
  - Traced replays still skew the chips by 5–8 µs on this box, so traced numbers were not used.
  - Stage breakdowns: `--profile` with per-round device zones, summarized by `perf_zones.py`.
- **What was done**:
  1. **Ready credit instead of a start-of-call fence** (port + reader).
     - The global `ready` semaphore is now 1 at rest. A port consumes the credit when its call starts.
     - The board's owner returns the credit at the end of the call. It waits until every leader of the link has read the board (`sem_gather_done`) and the receiving port has re-armed its arrival counter (`sem_rx_free`).
     - The next call's sends therefore need no round trip to the peer. Before, the last-launched chip paid ~2 µs: Fabric connection open plus the ready's round trip.
     - Board (re)allocation now drains the mesh first (host `synchronize_device`), because a peer may write into a board before this chip reaches the call.
  2. **Fan-out**: one 1088 B write per tile row per member (face0 row 0 through face1 row 0) instead of 2 × 64 B. This halves the issue count.
  3. **Auto exchange rounds** (`AUTO_ROUNDS`, `MIN_ROUND_TILES = 5`; gamma path only).
     - Among the plans with the fewest per-core tiles, the solve takes the most rounds whose per-round block is still ≥ 5 tiles.
     - Round r+1's load, sumsq and exchange then overlap round r's normalize and store, and round r's store overlaps the later rounds' reads.
     - Perf focus: 2 rounds of 1×6 tiles over 10 chunks × 10 row-slots (was 1 × 12 over 5 × 20).
     - Without gamma, rounds run back to back and measured 8–12 µs slower, so the plain solve stays there.
  4. **DRAM stagger** (`DRAM_STAGGER`, on): tile row t's reads and stores start at column `t % w_act`. Without it, every core of a round opens on the same few DRAM banks.
  5. **Bug fix (latent since refinement 1)**: the writer popped a `w_act` batch straddling the `cb_out_tiles` FIFO end in one `cb_pop_front`. That call only wraps on an exact landing, so tiles of later rows were never stored.
     - Only ragged multi-round plans hit it (e.g. 2048×2048 with 13 chunks of 5 tiles, the last holding 4). The auto-rounds solve now picks such plans; the batch is now popped in two parts.
  6. Knobs measured and parked:
     - `OUT_DEPTH` is now clamped to the core's rows across rounds, not to `block_h`; 1/2/4 rows measured flat.
     - `STORE_FLUSH` (off): release `cb_out` pages after a flush instead of a barrier; flat.
     - `COMBINE_AHEAD` (off): the leader combines round r+1 before normalize(r). It needs its own 2-round `cb_inv_row_out`. 640×1792 was flat and 2048×1280 went 38.7 → 47.6 µs.
  7. Tried and reverted (correct, but measured flat or slower, and not knob-shaped):
     - A non-blocking port scheduler (own(r+1) ahead of relay(r), per-parity arrival counters).
     - A poll-scheduled reader.
     - Moving publish and fan-out from the writer (NoC1) to the reader (NoC0). NoC0 is as congested by the DRAM reads as NoC1 is by the stores: 640×1792 21.7 → 22.5 µs.
  - Reused: every block operation, CB, helper call and the round machinery from refinement 1. Added: the credit semaphores, `plan_rounds`, the stagger, the split pop, and the knobs.
- **Perf** (last-launched chip, µs, before → after, 1×4 ring, bf16):
  - gamma, 2 links:
    - **640×1792 21.65 → 18.87** (target 22.2, roofline 16.7)
    - 32×1792 10.26 → 8.13; 2048×1280 47.2 → 37.9; 640×1536 19.2 → 17.7
    - 2048×2048 63.3 → 59.1; 4096×1024 64.5 → 54.9; 4×1×512×512 21.9 → 17.8
  - gamma, 1 link: 640×1792 23.3 → 21.2; 2048×1280 44.1 → 39.7; 2048×2048 64.7 → 58.7; 4096×1024 68.7 → 59.8.
  - No gamma: 32×1792 10.45 → 8.1; 640×1792 20.9 → 19.9. Every other case is within ±1% (e.g. 2048×1280 1-link 42.1 → 42.6).
- **What binds now** (640×1792 profile):
  - DRAM: x reads take ~7.5 µs (~310 GB/s) and the stores' tail ~3 µs. `ttnn.clone` of the same tensor takes 12.4 µs on one chip (read + write, ~370 GB/s), the practical DRAM floor.
  - The last round's chain on top of that: round-1 arrivals ~2.5 µs (a neighbour's port sends its round-1 slot only after relaying this chip's round-0 slot), gather 0.75, combine ~1, fan-out ~1.1 (NoC1 shared with the stores), normalize ~0.7.
- Accuracy achieved: the unit directory passes (194 passed, 3 skipped), including the acceptance PCC thresholds (bf16 ≥ 0.995, fp32 ≥ 0.999) and the unchanged precision baseline.
- Golden test progress: targeted slice = `test_op` INTERLEAVED gamma cells on 640×1792, 2048×2048 and 4096×1024, plus `test_op_loose`, `test_ring_mock` and the 2048-2048 / 4096-1024 `test_distributed_norm_allclose` cells: 131 passed, 216 xfailed, 0 failed (360 topology-skipped).
- Issues encountered: the latent `cb_out_tiles` straddle-pop bug above, found when the new plans failed (PCC ≈ 0; the error map showed only the ragged chunk's columns of later rounds unwritten).
- Tests added:
  - `test_distributed_rms_norm_perf.py`: the per-chip perf matrix, 7 shapes × gamma/no gamma × 1/2 links; knob overrides via `DRN_KNOBS`.
  - `test_distributed_rms_norm_rounds_debug.py`: tile error map of ragged multi-round plans; 2048×2048 at 1/2/4/8 rounds and a small ragged case.
  - `perf_zones.py`: device-zone summarizer.
  - `probes/probe_003.py`: the `ttnn.clone` DRAM reference.

## Refinement 4 — Non-tile-aligned W_local + ROW_MAJOR gamma
- Date: 2026-10-20
- What was done: `w_non_aligned` added to `SUPPORTED["alignment"]` and `ROW_MAJOR_LAYOUT` added to `SUPPORTED["gamma_layout"]`. Both are handled natively in the reader. No host wrapper, no new CB, no new kernel file. Compute, writer and port kernels are unchanged.
  - **W mask**: the worker whose chunk ends at `tensor_w_tiles` zeroes columns `[W_local % 32, 32)` of its last tile column before the push. Interleaved inputs get this per round after the read barrier. Sharded inputs get it once up front, next to the existing pad-tile zero fill. Whole pad faces are cleared with `noc.async_write_zeros`. Only the partial face rows (at most 32 rows of fewer than 16 elements) use RISC stores. bfloat8_b clears mantissa bytes behind the 64 B exponent header (a zero mantissa is zero whatever the exponent). The mean still uses `1/W_global` with the true width. The output's pad columns are don't-care. Pad rows are still row-independent end to end, so H+W non-aligned works too.
  - **ROW_MAJOR gamma**: stick `c` (one 32-element DRAM page, read whole so the read stays aligned) lands at the start of `cb_gamma` page `c`, which is face0 row 0. After the barrier its second half is moved locally to face1 row 0 with an L1-to-L1 NoC read. The `cb_gamma` page stride is now the tile size (`4 * g_face_bytes`) rather than the DRAM page size, so the TILE path is unchanged byte for byte.
  - Reused: `push_gamma` (resident and streamed), the sharded pad-zero pass, the `cb_gamma` / `cb_x_block` CBs. Added: 3 reader CT args (`x_last_cols`, `x_elem_bytes`, `gamma_rm`), `mask_x_tile`, and `X_ELEM_BYTES` on the host.
- Accuracy achieved: unit tests use pcc ≥ 0.995 (bf16), 0.999 (fp32) and 0.99 (bf8b) with the input tile padding poisoned to 1000. The golden slice passed at its `TOLERANCES` (bf16 pcc 0.999 / rel-rms 0.01, fp32 0.9999 / 0.005, bf8b 0.995 / 0.04). Shapes: (32,17), (1,1,17,50), (4,128,47), (1,1,64,1000), (2,1,128,100), (1,1,32,120/240), WIDTH-sharded 2x1 and 2x2.
- Golden test progress: targeted slice (every `w_non_aligned` cell, and ROW_MAJOR-gamma cells with bf16 input on every shape, all at cluster_axis=1 / Linear / 1 link / fp32 acc): 450 passed, 0 failed (1062 skipped as INVALID gamma sentinels). Translated `odd_hidden_dim`, `*_non_tile_aligned_width` and `memory_layouts`: 10/10 passed.
- Issues encountered: poisoning a bfloat8_b input's tile padding changes the shared exponent of valid values in the input itself, which is not an op error. The new test therefore poisons only bf16/fp32. The golden suite does not poison bfloat8_b.
- Tests added: `tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_nonaligned.py` (66 cases: poisoned W padding × {none, TILE, RM} gamma × {bf16, fp32, bf8b}, aligned RM gamma bf16/fp32, WIDTH-sharded non-aligned × TILE/RM gamma).

## Refinement 5 — Numerical configurability: bfloat8_b + fp32_dest_acc_en=False
- Date: 2026-10-20
- What was done: `False` was added to `SUPPORTED["fp32_dest_acc_en"]`. `bfloat8_b` was already in `SUPPORTED["dtype"]` (its fp32-DEST corner shipped earlier, since the x/out CB formats derive from `tile.get_tile_size(dtype)`). `{float32, False}` stays in EXCLUSIONS and still raises `ExcludedCell`.
  - Descriptor: `cb_xg` uses a Float16_b page when DEST is 16-bit (audit 3: Float32 would store no extra bits). `UnpackToDestFp32` on `cb_sumsq`, `cb_sumsq_t`, `cb_gather`, `cb_inv_row` and `cb_pregather` is gated on `fp32_dest_acc_en`. The exchanged partials, the board, `cb_gather` and `cb_inv_row*` keep Float32 pages in both modes (fp32 on the wire, per the prompt; see the ledger). The solve keeps charging `cb_xg` at 4 KB/tile, an upper bound, so the block plan does not depend on the DEST mode.
  - Compute kernel (one line of logic): the three `reduce` calls use `ReduceFp32Mode::Accurate` only when `DST_ACCUM_MODE` is on, otherwise `Fast`. Root cause: the helper's SFPU-path finalize (`sfpu_reduce_init/sfpu_reduce<..., reduce_format>`) reads DEST in the input CB's format (Float32). A 16-bit DEST does not hold that format, so the output was garbage (all-ones input gave 0.898 / 1.21 / 1000 / inf per row, PCC ≈ 0.15). With a 16-bit DEST the sums are 16-bit anyway, so the FPU path loses nothing. Helper gap, not fixed here: the SFPU-path finalize should key its format on `DST_ACCUM_MODE`, as the AccumulateViaAdd path already does (`dst_fmt`).
  - Reused: every CB, helper call and descriptor path. Added: the DEST-mode branch in the `cb_xg` format, the gated unpack-mode loop, and `kReduceFp32Mode`.
- Accuracy achieved (precision baseline, HiFi4, bf16 gamma, 2x2 mesh, cluster_axis=1). Results are the same across shapes 32x32, 32x1792, 640x1792 and 2048x1280:
  - bf16 + fp32 acc: PCC 0.999998, rel-RMS 0.0020, ULP mean 0.2 (unchanged from before)
  - fp32 + fp32 acc: PCC 0.9999998, rel-RMS 0.0014 (unchanged)
  - bf8b + fp32 acc: PCC 0.99992, rel-RMS 0.012
  - **bf16 + bf16 acc**: PCC 0.999992–0.999993, rel-RMS 0.0039–0.0040, ULP mean 0.5, got/true median 1.0002–1.0016, max abs 0.04–0.14
  - **bf8b + bf16 acc**: PCC 0.99992, rel-RMS 0.0135–0.0141, max abs 0.10–0.28
- Golden test progress: a random sample of 80 buildable `test_op` cells with `fp32_dest_acc_en=False` (bf16 / bf8b, every layout, alignment, topology and link count): 80/80 passed. One sampled `{float32, False}` cell: XFAIL (ExcludedCell). Regression: `test_distributed_rms_norm.py` 69 passed, 1 skipped; precision baseline 20/20.
- Issues encountered: the SFPU reduce / 16-bit DEST format mismatch above, found with an all-ones deterministic probe.
- Tests added:
  - `test_distributed_rms_norm_precision_baseline.py` now covers the (dtype, fp32_dest_acc_en) corners: bf16/fp32/bf8b with fp32 acc, and bf16/bf8b with bf16 acc.
  - `test_distributed_rms_norm_bf16acc_debug.py` (new): all-ones and per-row ratio map, each in both DEST modes.

## Refinement 6 — Second perf pass on the perf-flagged profile (structural levers)
- Date: 2026-10-20
- **Method**: same as refinement 3. On-device kernel time from the realtime program profiler, using `test_distributed_rms_norm_perf.py` on the 1×4 ring stand-in (QuietBox 2), untraced.
  - Reported value: the last-launched chip (minimum over chips), median of 3 calls.
  - A/B baselines were re-measured in the same session by temporarily restoring the refinement-5 descriptor and reader.
  - Stage breakdowns: `--profile` + `perf_zones.py`.
- **What was done**, one structural lever at a time:
  1. **Exchange/compute overlap**: already the default (refinement 3, `AUTO_ROUNDS`). This pass let the solve trade 100 one-tile-row workers for wider multi-round blocks when that measures faster:
     - **Tile slack** (`TILE_SLACK = 0.25`). Among the plans within 25% of the fewest per-core tiles, `plan_rounds` now prefers, in order:
       1. per-round blocks ≥ `MIN_ROUND_TILES`;
       2. the fewest gather tiles `ceil(G·C/32)`;
       3. several rounds over one;
       4. the fewest tiles;
       5. the most rounds.
     - **Balanced row-slots**: every multi-round candidate uses `ceil(num_row_groups / num_rounds)` slots, so no round is under-filled.
     - Perf focus: 2 rounds of 1×7 over 8 chunks × 10 row-slots. G·C = 32 means one gather tile, so the leader's zero, board read and combine are halved. The previous plan was 1×6 over 10 chunks, with two gather tiles.
     - Sweep that found it (640×1792, 2 balanced rounds, chunk width): 6 tiles → 18.9 µs; 7 → 18.0; 8 → 18.2; 12 → 18.8; 14 → 19.9; 19 → 25.4.
     - Ragged 15-slot rounds measured 20.4 µs; 3 or more rounds of ≤ 5 tiles measured 27–31 µs.
     - The slack ordering is "several rounds over one, then fewest tiles". Ranking more rounds above fewer tiles made 4096×1024 2-link (60.3 vs 54.9) and 2048×1280 1-link slower, so that order was rejected.
  2. **Gamma broadcast (regime R4)**: built and measured, then parked at `GAMMA_MCAST = False`.
     - Structure: a `McastFamily` (mcast_pipe) with one group per W-chunk column.
     - The sender issues its chunk's sliver reads ahead of its own `x(0)` reads, then multicasts its `cb_gamma` pages to the column. No stage CB, no local copies.
     - Placement when on is 2-D: worker (rg, wc) on core (1+wc, rg), ports in grid column 0. Each chunk column is one physical column (one rectangle) and each row-group one grid row. The broadcast applies only when the plan fits that box.
     - Data-ready is a **Counter** with no handshake: one block per call lands in an empty `cb_gamma`.
     - The first hang came from a Flag `ReceiverPipe`, whose constructor sets the flag INVALID and so cleared a flag that had already landed. A receiver may construct its pipe after the sender's signal, so the Counter is required. Each receiver re-arms it after its single receive.
     - Measured, 640×1792 ring 2-link (3 × 3 runs): on ≈ 19.05 µs vs off ≈ 18.88 µs (flat to 1% slower). At fixed column placement: 18.73 vs 19.06. 2048×1280 and 2048×2048: within noise.
     - Earlier shapes, each measured:
       - one sender reading every column's slivers: 20.9 µs (112 serial sliver reads, `R_gsend` 4.4 µs, gating every core's `x(1)`);
       - chunk-major row-major placement with ≤ 3 rectangles per column: placement alone +0.6 µs.
     - The parked path is pinned by a test (TILE and ROW_MAJOR gamma, ragged non-aligned chunk, back-to-back calls).
  3. **Local pre-combine (R3)** on interleaved, via the existing knob `PRE_COMBINE=True`: 640×1792 18.8 → 26–27 µs. The serial gather and reduce ahead of the first send outweigh the saved Fabric bytes. It stays at auto (board cap only).
  - Reused:
    - the round machinery, every CB and helper, `fill`/`push_gamma`;
    - the `mcast_pipe` host and device helpers (`McastFamily`, `McastArgs`; `attach_absent` when off).
  - Added: `TILE_SLACK`, balanced slots, `GAMMA_MCAST` with `column_placement` and `gamma_mcast_family`, and the issue/finish split of the gamma reads.
- **Perf** (last-launched chip, µs; refinement-5 code → now, same session):
  - bf16 gamma, 2 links:
    - **640×1792 18.79 → 18.06** (refinement-3 record 18.87; target 22.2, roofline 16.7)
    - 640×1536 18.04 → 16.62
    - 2048×2048 59.1 → 54.3
    - 2048×1280 39.4 → 38.6
    - 4×1×512×512 17.84 → 17.49
    - 4096×1024 54.9 → 54.6
    - 32×1792 8.13 → 8.13
  - bf16 gamma, 1 link: 640×1792 21.43 → 20.28; 640×1536 20.2 → 19.1; 4×512×512 21.0 → 20.5; 2048×1280 40.1–40.5 → 39.8–40.0; 2048×2048 58.7 → 59.3 (noise); 4096×1024 59.8 → 58.8.
  - bf8b gamma: 640×1792 15.87 → 15.23 (2 links) and 18.32 → 17.57 (1 link); 2048×1280 unchanged (31.7 / 32.6–33.1).
  - No gamma: plans unchanged (32×1792 8.1, 640×1792 19.9).
  - Sharded: `plan_sharded` is untouched; only the gamma reads were split into issue and finish (same NoC operations).
- **What binds now**: the same structure as refinement 3.
  - DRAM reads of x (~7.5 µs) plus the store tail.
  - Then the last round's exchange chain: round-1 arrivals behind the ports' in-order relay, plus gather, combine, fan-out and normalize.
  - The gather tile is now halved; the arrival serialization is untouched.
- Accuracy achieved: the unit directory passes (259 passed, 3 skipped), including the acceptance PCC (bf16 ≥ 0.995, fp32 ≥ 0.999, bf8b ≥ 0.99) and the precision-baseline test.
- Golden test progress: targeted slice = `test_op_loose`, `test_ring_mock`, and the `test_op` INTERLEAVED gamma cells on 640×1792 plus bf16 2048×2048: 163 passed, 24 xfailed, 0 failed (176 topology-skipped).
- Issues encountered:
  - The Flag-receiver hang above.
  - `McastArgs` faces called from a non-template lambda do not discard the absent branch (compile error with the knob off), so they moved into function templates.
- Tests added:
  - `test_distributed_rms_norm_gamma_mcast.py` (knob forced on: 640×1792 and 320×1000, TILE and ROW_MAJOR gamma, back-to-back).
  - The perf test takes a `DRN_DTYPE` override (bf8b A/B).

## Perf 1 — Perf tournament round 1 (compute on the exchange's critical chain)
- Date: 2026-10-21
- **Method**: on-device kernel time from the realtime program profiler, `test_distributed_rms_norm_perf.py`, 1×4 ring stand-in (QuietBox 2, FABRIC_2D_TORUS_XY), untraced, last-launched chip. Each cell is the median of 3 calls; before and after were measured in the same session. Stage breakdowns come from `--profile` + `perf_zones.py`, per-TRISC.
- **Focus (LOOSE perf case, exact config)**: per-device (1,1,640,1792), bf16 x, bf16 TILE gamma, fp32_dest_acc_en=True, HiFi4, INTERLEAVED, cluster_axis 1, Ring, 2 links. Every knob is in SUPPORTED. Plan: 2 rounds of 1×7 tiles over 8 chunks × 10 row-slots.
- **Instrumentation (permanent)**: added `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` (`MaybeDeviceZoneScope`, which was missing from the tree). All four kernels moved to it.
  - New compute zones split wait from work: `C_xwait` / `C_sumsq`, `C_xg`, `C_precombine`, `C_gwait` / `C_combine`, `C_invwait` / `C_norm`. The waits are hoisted onto unpack, so the work zones on unpack exclude them.
  - New writer zones: `W_owait` / `W_store`, one per round to stay within the marker budget.
### Measured breakdown (focus shape, before)
- Whole op: **18.4–18.5 µs**.
- Cumulative ablation (stub the payload, keep the sync):

  | ablation | µs |
  |---|---|
  | full | 18.5 |
  | − store | 15.7 |
  | − store − x read | 12.6 |
  | − store − x read − compute | **10.2** (exchange / sync / program floor) |

- Solo ablations: − compute alone 13.5 (**−5.0 µs**); − x read alone 15.9.
- Per-TRISC compute, h = 1, w = 7. Math minima include no waits; the unpack numbers exclude the hoisted wait.
  - sumsq: unpack ≈1.0, math ≈1.8, pack ≈1.6 µs.
  - leader combine: unpack 0.2, **math ≈1.3**, pack 0.2 µs.
  - normalize: math ≈1.3, pack ≈2.4 µs.
- All three run once per round on the exchange chain: x(r) → sumsq → publish → fabric → gather → combine → fan-out → normalize → store.
- **Ranked**:
  1. Compute on the critical chain, ≈5 µs. Latency-bound: no DM roofline applies, so there is headroom.
  2. x DRAM read, ≈3 µs, and 3. store, ≈2.8 µs. Both are near the measured practical DRAM floor (`ttnn.clone` ≈370 GB/s, refinement 3), so little headroom.
  4. The exchange / sync floor, ≈10.2 µs.
### Portfolio floated (cap: 2 experiments)
- **A. Fused sumsq in one DEST window (tested).** x·x accumulated in DEST → `transpose_dest` → SFPU column fold → one pack. This replaces 3 passes and 2 L1 round trips.
- **B. Leader combine epilogue scoped to row 0 (tested).** VectorMode::R × 2 iterations for ×1/W, +eps, rsqrt, per `examples/sfpu_tile_scope`.
- C. Move the inv transpose into the leader's DEST (`transpose_dest`) and fan out a column tile, so workers drop normalize's transpose. Not tested (cap).
- D. Fan out inv_rms by multicast to the row-group rectangle (writer; `W_fan` is ≈0.75–0.95 µs per round). Not tested (cap).
- E. DRAM read / store levers. Not floated: gated by the DRAM floor, and the stagger was already tried in refinement 3.
### Per-idea verdicts
- **A — WIN** (`perf_experiments/fused_sumsq_dest/`, single-core bench, BH):
  - h=1, w=7, bf16, fp32 DEST: 1442 → **1040 ns** kernel; stage ≈1030 → ≈630 ns.
  - Bit-identical to the 3-pass path in all 54 fp32-DEST cells (x bf16 / fp32 / bf8b; h 1/2/4; w 1–56). It saves 330–420 ns per tile row-group and was faster in every cell.
  - 16-bit DEST: still faster, but the SFPU fold showed a −3 to −4e-3 relative bias on the sums against today's FPU (Fast) reduce. fp32 x was worse in all 18 cells; mean rel error averaged 8.5e-3 against 5.0e-3. This is a measured precision regression, so it is an **`incorrect`** exception.
  - Menu, not graduated: row_tdest 1131 ns; row_tface 1093 ns (needs a writer change, and is wrong with a 16-bit DEST).
- **B — WIN** (`perf_experiments/combine_sfpu_scope/`):
  - h=1, n_gather=1, fp32 DEST: combine **1670 → 785 ns** e2e, math 1607 → 714.
  - Row 0 is bit-exact in every cell: both DEST modes, n_gather 1/2, h 1–8, speedup 1.76–3.55×.
  - Menu, not graduated: `fused_r_iter2` (one raw-sfpi pass) 765 ns, +2–8%, bit-exact. `fused_mad_r_iter2` 742 ns is NOT bit-exact (single rounding of x·a+b), so it is recorded as an option with that cost.
### Graduated
- **A**: the op's sumsq path for **every fp32_dest_acc_en=True cell** (all dtypes, layouts, placements, pre-combine, streamed gamma, any h, w). `if constexpr (DST_ACCUM_MODE) { fused } else { 3-pass }`.
  - **Carve-out**: fp32_dest_acc_en=False keeps the 3-pass helper path. It was earned by the measured 16-bit-DEST bias above.
  - The 3-pass code therefore stays, as that carve-out's path. `cb_sumsq` / `cb_sumsq_t` are idle in fp32-DEST mode; they share alias group A with `cb_inv_row_out`, so they cost no L1. `compute_kernel_hw_startup` now keys on `cb_partial_row`.
- **B**: the combine post-op everywhere (both DEST modes), with no carve-out. The full-tile `mul_unary_tile` / `add_unary_tile` / `rsqrt_tile` calls were deleted.
  - The tile's unused rows now differ: tile row 16 (row 0 of faces 2 and 3) holds the raw sum. Nothing reads it; the fan-out ships face0 plus face1 row 0, and transpose + Col broadcast read row 0 only.
- The instrumentation zones `C_sumsq` and `C_combine` cover the new paths.
### Whole op, before → after (µs, last-launched chip, median of 3, same session)
- **gamma, 2 links**:

  | shape | before | after |
  |---|---|---|
  | **640×1792** | **18.43** | **16.90** (−8.3%; target 22.2, roofline 16.7) |
  | 32×1792 | 8.48 | 7.19 |
  | 2048×1280 | 38.95 | 37.94 |
  | 640×1536 | 17.06 | 15.77 |
  | 2048×2048 | 55.17 | 51.73 |
  | 4096×1024 | 55.51 | 52.38 |
  | 4×1×512×512 | 18.01 | 16.79 |

- **gamma, 1 link**: 640×1792 20.61 → 19.25; 32×1792 8.31 → 7.20; 2048×1280 40.65 → 38.68; 640×1536 19.30 → 18.26; 2048×2048 58.96 → 57.76 (noise band); 4096×1024 61.72 → 57.49; 4×512×512 20.68 → 19.65.
- **no gamma, 2 links**: 640×1792 20.09 → 18.75; 32×1792 8.38 → 7.15; 2048×1280 47.29 → 42.20; 640×1536 18.05 → 16.77; 2048×2048 63.33 → 58.76; 4096×1024 64.54 → 59.75; 4×512×512 21.44 → 19.27.
- **no gamma, 1 link**: 640×1792 21.2 → 20.1; 32×1792 8.28 → 7.06; 2048×1280 42.59 → 40.42; 2048×2048 63.2 → 59.8; 4096×1024 68.81 → 64.28; 4×512×512 23.02 → 20.81.
- **Guard set**: all 28 cells are faster or flat (gamma / no gamma × 1 / 2 links × 7 shapes). No regression, so no further carve-outs.
### Correctness
- Golden suite, run in 7 chunks (each layout × DEST mode, plus the non-layout tests):
  - INTERLEAVED: True 2163 passed; False 1440 passed / 720 xfailed.
  - HEIGHT_SHARDED: True 2020 passed; False 1440 / 720 xfailed.
  - WIDTH_SHARDED: True 2160; False 1440 / 720 xfailed.
  - Other tests: 301 passed / 12 xfailed / **2 failed**. The two failures are `test_translated.py::test_ds_distributed_norm[random_weights-prefill-{32768,131072}-0.98]`, the host L1 refusal "working set exceeds L1 for every block (streaming regime deferred)". They **fail identically on the pre-Perf-1 kernel**, so they are a pre-existing streaming-regime generality gap, not a perf regression. They are reported upstream as unfinished generality work.
- Unit tests: precision baseline + multiround, 41 passed.
### What binds now (focus, profiled)
- Round 0's own slot is ready about 0.6 µs earlier.
- The chain is now the fabric relay / arrival (`P_relaywait`), gather, fan-out (`W_fan` 0.7–0.95 µs, unicast to 8 members), and normalize (`C_norm` ≈2.7 µs including its transpose).
- Round-2 candidates: C and D above, and the normalize transpose / pack.
### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `ckl::eltwise_chain` (BinaryFpu Mul, PerRow) + `ckl::reduce<SUM, REDUCE_COL, …, Accurate>` (sumsq 2a–2c) | capability | The chain has no in-DEST transpose element (`TransposeDest<Dst>`) and no SFPU-reduce element. `ckl::reduce` only reduces tiles it unpacks from a CB itself, so it cannot fold a tile already in DEST. Together these force the x² sum to round-trip L1 twice (pack S → transpose pass → reduce pass). | 1442 (kernel; stage ≈1030) | 1040 (stage ≈630) | `distributed_rms_norm_compute.cpp:199` |
| `mul_unary_tile` / `add_unary_tile` / `rsqrt_tile` (compute API, inside `ckl::reduce`'s `post_reduce_op`) | ergonomics | The API hard-codes `VectorMode::RC` × 8 iterations, with no `vector_mode` / iterations parameter, while the LLK supports both. Scoping to row 0 therefore means calling `SFPU_UNARY_CALL` with the stock `ckernel::sfpu` bodies plus `TRISC_MATH`-guarded `ckernel_sfpu_*` includes. A row-0 scope on kernel_lib's unary elements, or a vector-mode + iterations parameter on the API, would close it. | 1670 | 785 | `distributed_rms_norm_compute.cpp:337` |

## Perf 2 — Perf tournament round 2 (normalize transpose + leader fan-out on the exchange chain)
- Date: 2026-10-21
- **Method**: same as Perf 1. On-device kernel time comes from the realtime program profiler (`test_distributed_rms_norm_perf.py`), on the 1×4 ring stand-in (QuietBox 2, FABRIC_2D_TORUS_XY), untraced, last-launched chip, median of 3 calls; before and after were measured in the same session. Stage numbers come from `--profile` per TRISC.
- **Focus (LOOSE perf case, exact config)**: per-device (1,1,640,1792), bf16 x, bf16 TILE gamma, fp32_dest_acc_en=True, HiFi4, INTERLEAVED, cluster_axis 1, Ring, 2 links. Every knob is in SUPPORTED. The plan is unchanged: 2 rounds of 1×7 tiles, 8 chunks × 10 row-slots.
- **Instrumentation**: the existing zones are reused and cover the new paths: `W_fan` (multicast fan-out) and `C_norm` (srcA transpose). Nothing was added or removed.
### Measured breakdown (focus, before; re-measured on the Perf-1 code)
- Whole op 17.2 µs.
- **Cumulative ablation**: the payload was stubbed and the sync kept, using temporary `DRN_ABLATE` defines that were not committed.

  | ablation | µs |
  |---|---|
  | full | 17.2 |
  | − store | 14.4 |
  | − store − x read | 11.4 |
  | − store − x read − compute | **10.1** (exchange / sync / program floor; every payload stubbed at once) |

  - Solo: − compute 13.5; − x read 14.5.
  - Pairs: − compute − x read 10.8; − compute − store 10.9.
  - So compute and the store are balanced on the tail, and compute is still the biggest single lever.
- **Per TRISC, per round**, unpack-thread work (the waits are hoisted, so these are pure work):
  - normalize (`C_norm`) ≈ 790 ns;
  - the same 7-tile multiply without the transpose (`C_xg#1`) ≈ 500 ns, so the inv transpose costs ≈ 270 ns on every worker's critical chain.
  - Leader fan-out `W_fan` ≈ 755 ns per round, on the exchange chain: 8 unicasts + barrier + 8 semaphore increments.
- **Ranked**:
  1. Compute on the chain after inv arrives (normalize's transpose + multiply).
  2. The leader fan-out.
  3. The x read and the store tail. These sit near the DRAM floor (Perf 1 / refinement 3), so they were not targeted.
### Portfolio (cap: 2 experiments)
- **C — tested.** Remove the worker's per-round inv transpose L1 round trip from normalize. Options: leader-side `transpose_dest` plus a column-vector fan-out, worker in-DEST fusion, or anything else exact.
- **D — tested.** Leader inv_rms fan-out by NoC multicast to the row-group's rectangles (the reuse-shared broadcast idea applied to the fan-out), with the reader's twin semaphore semantics.
- Not floated:
  - streaming the store per tile to overlap normalize (the tail is DRAM-bandwidth-bound across 80 cores; low expected headroom);
  - x read / store levers (DRAM floor).
### Per-idea verdicts
- **C — WIN** (`perf_experiments/norm_no_transpose/`, single core, BH):
  - **Winner `srca_t`**: stop tagging `CB_INV_ROW` as `UnpackToDestFp32`. The worker kernel is unchanged, but its `transpose_tile` now transposes in the unpacker (srcA) instead of unpack-to-DEST + 32-bit `transpose_dest`.
  - Worker normalize kernel at h=1, w=7: **1333 → 1207 ns**. h=2: 2046 → 1730. h=8: 6283 → 4970.
  - **Bit-exact** in all 64 cells (h 1/2/4/8 × w 1/4/7/16 × gamma / no-gamma × both DEST modes). The only consumer is the Col broadcast's srcB (Tf32), and fp32 → Tf32 is idempotent.
  - With a 16-bit DEST it is the same program (that CB was only tagged with fp32 DEST), so flat.
  - **Domain: everywhere.**
  - Menu, not graduated:
    - (a) leader `transpose_dest` + a 3 KB column-vector fan-out. Worker −277 ns, but leader +154 ns and fan-out +91 ns, so ≈ −32 ns net at h=1. It is a measured regression at h ≥ 4 (≈ +450 ns at h=8) and with a 16-bit DEST.
    - (b) worker DEST-reuse was not built: the ELWMUL dest-reuse overwrites the DEST source tile, so it would need one in-DEST copy per output tile.
    - Raw `TRNSPSRCB` was rejected: it is 16-bit only and loses Tf32 bits.
- **D — WIN** (`perf_experiments/inv_fanout_mcast/`):
  - Closed-loop bench with 10 concurrent leaders on the real placement. The metric is leader issue → last member sees the signal.
  - **`mcast_hybrid` with the leader at the row-group's last row-major core: 1044 → 420 ns** at C=8, h=1, 1088 B. Leader busy time 894 → 287 ns.
  - Hybrid = per rectangle, a linked multicast per tile row, then a multicast SET of a scratch cell holding r + 1 into `sem_inv_ready`. It uses the same NoC / VC / path, so the flag lands after the data with no barrier. A 1-core rectangle gets ordered unicasts instead.
  - Leader placement is part of the change: with the old first-core leader, multicast is a **measured regression** with 10 concurrent leaders (C=8 h=1 1044 → 1145; C=4 / C=2 worse). Each multicast routes around the NoC1 ring to the rectangle's high corner.
  - With leader = last core, hybrid is ≥ 1.57× faster than unicast in every cell of C 2/4/8 × h 1/2/4 × 1088 / 3072 B, in both simple and column placement.
  - Bit-exact data movement: every member was checked word-for-word each round.
  - `mcast_pipe` itself: Counter 1318 ns (leader last) / 2314 ns (leader first); Flag mode cannot express the pattern (see the bypass table).
### Graduated (both, no conflict: C touches the compute config, D the writer + leader placement)
- **C**: `CB_INV_ROW` dropped from the UnpackToDestFp32 list. This is the one unqualified path, with no carve-out. The CB is no longer listed with the transposes' lossless reloads, and the comment states why.
- **D**: the leader fan-out is the multicast hybrid everywhere, and the unicast + barrier + per-member increment code is **deleted**.
  - **Leader** = the member with the largest logical (y, x) in its row-group, i.e. the NoC1-start corner of its own rectangle. This applies on every placement: simple, column (parked gamma mcast) and sharded. The leader role (board gather, combine, pre-combine target, port leader lists) moves with it.
  - **Rectangles**: `_fanout_rects` decomposes any member set into per-row contiguous x runs, merging consecutive rows that have the same run. Every core set is expressible this way, so there is **no "inexpressible" carve-out**: a non-contiguous shard group just yields more rectangles.
  - **Signalling**: new semaphore `SEM_INV_SIG` (9) is the leader's scratch. The reader's `noc_semaphore_wait_min(inv_ready, r + 1)` and its end-of-call reset are unchanged, because a set to r + 1 is the same monotone sequence as the old increments.
  - **Carve-outs: none.**
### Whole op, before → after (µs, last-launched chip, median of 3, same session)
- **Focus 640×1792 gamma 2 links: 17.30 → 16.79** (−2.9%; target 22.2, roofline 16.7).
- Profiled focus:
  - `W_fan` 755 → 272 ns per round;
  - `C_norm` unpack ≈ 790 → ≈ 605–630 ns per round;
  - best-chip span 17.23 → 16.88 µs (profiler on).
- **Guard set**:
  - **gamma, 2 links**:

    | shape | before | after |
    |---|---|---|
    | 640×1792 | 17.30 | 16.79 |
    | 32×1792 | 7.17 | 6.74 |
    | 2048×1280 | 37.83 | 34.40 |
    | 640×1536 | 15.60 | 15.72 (+0.8%, within this cell's spread) |
    | 2048×2048 | 51.33 | 50.63 |
    | 4096×1024 | 53.13 | 52.69 |
    | 4×512×512 | 17.24 | 16.31 |

    - The 640×1536 runs spanned 15.38–16.33 µs across the two sessions.
  - **gamma, 1 link**: 640×1792 19.24 → 18.74; 32×1792 7.15 → 6.67; 2048×1280 38.96 → 38.47; 640×1536 18.08 → 17.48; 2048×2048 57.50 → 56.52; 4096×1024 58.72 → 56.91; 4×512×512 19.71 → 19.26.
  - **no gamma, 2 links**: 640×1792 18.86 → 18.31; 32×1792 7.18 → 6.63; 2048×1280 42.49 → 41.53; 640×1536 16.64 → 16.06; 2048×2048 58.22 → 57.23; 4096×1024 59.80 → 57.51; 4×512×512 19.33 → 18.56.
  - **no gamma, 1 link**: 640×1792 20.12 → 19.45; 32×1792 7.19 → 6.62; 2048×1280 40.23 → 39.57; 640×1536 17.80 → 17.12; 2048×2048 60.29 → 58.60; 4096×1024 64.44 → 62.20; 4×512×512 20.56 → 19.81.
  - Result: 27 of 28 cells are faster; one is flat within noise. **No material regression → no carve-outs.**
### Correctness
- **Golden suite**:
  - INTERLEAVED: True 2160 passed (2 × 1080); False 1440 passed / 720 xfailed.
  - HEIGHT_SHARDED: True 2020 passed; False 1440 / 720 xfailed.
  - WIDTH_SHARDED: True 2160 passed; False 1440 / 720 xfailed.
  - Other tests: 304 passed / 12 xfailed / 2 failed. The 2 failures are the same pre-existing `test_translated.py::test_ds_distributed_norm[...prefill-{32768,131072}...]` streaming-regime L1 refusals as Perf 1.
- **Pre-existing, order-dependent L1 clash (found here, not caused by this round)**:
  - Running WIDTH_SHARDED split by num_links first hits "Statically allocated circular buffers ... clash with L1 buffers" at `4x1x512x512 ... WIDTH_SHARDED-num_links=2`, and later cells then cascade-fail.
  - The **Perf-1 code fails identically** with the same ordering: same cell, same addresses (646528 / 904192).
  - In collection order (both link counts in one chunk, as in Perf 1), WIDTH_SHARDED is fully green.
  - Reported upstream as a sharded-path L1-budget / stale-allocation bug, not a perf regression.
- **Unit tests** (acceptance, gamma mcast including the column placement with the moved leader, multiround, precision baseline, non-aligned, sharded): 239 passed, 3 skipped.
### What binds now (focus)
- The fabric relay / arrival (`P_relaywait`), the leader's board gather + combine (`C_combine#1` math ≈ 1.3–2.5 µs, including gather waits), the normalize multiply (≈ 0.6 µs per round) and the DRAM store tail (`W_store#1` ≈ 0.6–2 µs, bandwidth-bound across 80 cores).
- 16.79 µs is about 0.1 µs above the 16.7 µs roofline.
### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `mcast_pipe` (`SenderPipe` / `McastArgs`) | capability | **Counter** signal: a write barrier per rectangle before a non-posted atomic increment-multicast, plus a separate local bump of the sender's own cell. **Flag** signal: resets the receiver's cell on every receive (needs clear-before-next-set back-pressure that pipelined rounds don't have) and sources the SET from the sender's own ready cell (the leader's own reader sees "ready" before its loopback data lands). Missing: a no-barrier "data then SET from a scratch cell holding a caller value, same VC" signal mode, and per-group multi-rectangle sends with a unicast fallback for 1-core rectangles. | 1318 (Counter, leader last; 2314 leader first) | 420 | `kernels/distributed_rms_norm_writer.cpp:188` |

- C graduated as a host config change (`distributed_rms_norm_program_descriptor.py:883`), so it is not a bypass. The worker's transpose was already raw (`transpose_block_in_place`, Perf 1 table / kernel header).
