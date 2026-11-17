# Changelog: rms_norm

## Phase 0 — Core Implementation
- **Date**: 2026-11-17
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Verifier added the `memory_config` kwarg to the entry point (golden harness passes it on sharded cells).
- **SUPPORTED at Phase 0**: dtype=[float32, bfloat16], fp32_dest_acc_en=[True], layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2,3,4], gamma_mode=[gamma, no_gamma], gamma_dtype=[float32, bfloat16, none], gamma_layout=[ROW_MAJOR, none], memory_layout=[INTERLEAVED]
- **Accuracy achieved**: PCC≥0.9999984 (bf16), ≥0.9999997 (fp32); max_abs_err 0.0155–0.039 (bf16), rel RMS 7.5e-4–1.9e-3 (measured on 4 shapes × 2 dtypes via test_rms_norm_precision_baseline.py)
- **Golden suite at Phase 0**: 436 / 436 supported cells passing (`verifier_report.json`); 6481 xfail_expected; 0 supported_fail / xpass_drift / xfail_wrong_mode
- **Issues encountered**: 4786 sharded cells failed with TypeError for the missing `memory_config` kwarg — fixed.
- **Tests added**: test_rms_norm.py, test_rms_norm_regimes.py, test_rms_norm_precision_baseline.py

## Refinement 1 — Precision surface + TILE gamma (perf-flagged-profile contract, part 1)
- Date: 2026-11-17
- What was done: SUPPORTED gained fp32_dest_acc_en=False, dtype bfloat8_b, gamma_dtype bfloat8_b, gamma_layout TILE. EXCLUSIONS: fp32 input + fp32_dest=False; bf8b + RM / non-aligned; bf8b gamma + RM gamma. Intermediates (cb_sq/cb_xn/cb_stat) are bf16 when DEST is 16-bit, fp32 otherwise. TILE gamma: reader reads gamma tiles directly (row 0 broadcast row), no scatter, no gamma stick scratch CB; multi-core split and segment batching unchanged. 1/W is folded into the reduce scaler when DEST is 16-bit.
- Accuracy achieved: perf-profile loose cases (interleaved) 107/107 pass; PCC >= 0.9995.
- Golden test progress: targeted slices green (loose INTERLEAVED 107/107; sampled bf8b / fp32_dest=False shapes 56 passed, rest xfail/skip as expected).
- Issues encountered: with a 16-bit DEST, sequential reduce_tile accumulation of sum-of-squares is biased high once the accumulator dwarfs the per-tile increment (all-ones W=5120 gave 0.859 instead of 1.0; independent of G_CAP, CB dtype, fidelity). Fix: for Wt > BF16_DEST_MAX_WT (64, W > 2048) the descriptor promotes the compute config to fp32_dest_acc_en=True (intermediates fp32). Finding / next lever: a bf16-only multi-level reduction tree (reduce ~4 tiles, add partials pairwise) would restore true 16-bit DEST on wide rows; perf refinements should measure the cost of the fp32 promotion.
- Tests added: tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_bf16dest_debug_probe.py (debug probe script), probes/.

## Refinement 2 — W-split across cores with cross-core partial combine (perf-flagged-profile contract, part 2)
- Date: 2026-11-17
- What was done: new W-split regime behind the host predicate `pick_w_split` (n_w = divisor of Wt minimizing estimated per-core work, taken only if < 0.75x the row-split work; n_w == 1 keeps the previous path byte-identical). Row group g (n_r groups) owns contiguous tile-rows; its n_w cores each own a W slice (x read once, slice-sized CBs/blocking via `pick_blocking(Wt_slice)`). Topology: compute accumulates the slice's partial in `cb_stat` -> copies it to `cb_psend` -> writer RISC unicasts it to every core of the group (incl. itself) into `cb_part` slot [row][slice] (2 parity halves + 2 cumulative semaphores) -> writer pushes the gathered block -> compute sums the n_w tiles per row with an L1-accumulating `PackTile` into `cb_stat` -> existing inv_rms chain -> pass 2 on the slice. Reader/writer index tiles/stick-bytes by the slice offset; the W-pad zeroing only applies on the last slice. New CBs/semaphores recorded in `l1_ledger.md`.
- Accuracy achieved: all prior unit tests (122) + 65 new W-split cases pass (PCC thresholds of test_rms_norm: 0.999 fp32 / 0.995 bf16) on shapes 32x{1024,5120,7168}, 1x16384, 32x32768, 64x12288, 96x4000 (ragged W), 2x32x3072, TILE+RM, gamma/no gamma.
- Device ns (bf16, HiFi2, fp32_dest=False, RM gamma; before -> after): 32x5120 73.3k->20.5k, 32x1024 23.0k->9.3k, 32x7168 104k->25.9k, 1x16384 272k->35.0k, 64x12288 203k->34.8k, 32x32768 555k->54.3k.
- Golden test progress: not run in full (harness gate); unit regression net green.
- Issues encountered: pass-1 `Accumulate` pushes intermediate partials to its output CB each segment, so pointing it directly at the writer-consumed CB sent half-sums; fixed by the cb_stat -> cb_psend hand-off copy. Limitation: n_w must divide Wt (prime Wt stays on the row-split path). Unicast all-gather used instead of `ttnn.Mcast` (groups are arbitrary core lists; n_w tiny payloads).
- Tests added: test_rms_norm_wsplit.py (core-count predicate + correctness), test_rms_norm_wsplit_perf.py (profile probe; RMS_WSPLIT=0 disables).

## Refinement 3 — Speed up the perf-flagged profile
- Date: 2026-11-17
- What was done: added perf probe test_rms_norm_perf_r3.py (env overrides RMS_GCAP / RMS_ODEPTH for the module-level G_CAP / OUT_DEPTH knobs). Measured G_CAP 16 vs 32 and OUT_DEPTH 2 vs 4 on 8192x{1024,2304,5120,7168} and 32x{2304,5120}: all within noise (8192x5120 572k / 572k / 569k ns). Prefill is at ~300 GB/s effective DRAM bandwidth (bandwidth-bound); decode already W-split. No kernel change; knobs stay at defaults.
- Accuracy achieved: PCC > 0.9995 on the 6 probe shapes (asserted in probe)
- Golden test progress: not run (no functional change)
- Issues encountered: None. Finding: remaining levers (gamma mcast, ragged-tail Sb for prime Wt, fused scale chain) target non-binding stages for these shapes.
- Tests added: test_rms_norm_perf_r3.py

## Refinement 4 — HEIGHT_SHARDED in place + output `memory_config`
- Date: 2026-11-17
- What was done: `HEIGHT_SHARDED` added to SUPPORTED["memory_layout"] (TILE layout). Sharded input: `cb_x` is backed by the input shard (`cb_descriptor_from_sharded_tensor`), the reader publishes the whole shard with one reserve/push (no NoC read, no accessor; W pad zeroed in place); blocking is forced resident. Sharded output: `cb_out_tiles` is backed by the output shard, compute packs straight into it and the writer does nothing. Core plan comes from the shard grid (core i owns tile-rows [i*sh, (i+1)*sh), ragged last shard handled by rows=0/partial); W-split is off. `memory_config` is honored (default: sharded input -> output inherits its spec, otherwise DRAM); input-sharded/output-interleaved and the reverse also work. Under L1 pressure (in+out shards > 900KB, e.g. Wt=344) the compute window cap drops to 2 tiles. ROW_MAJOR x HEIGHT_SHARDED is in EXCLUSIONS (follow-up 4a).
- Accuracy achieved: golden TILE HEIGHT cells (fp32/bf16/bf8b, gamma variants, W/H non-aligned, 3D/2D, prime tile counts, wide 11008) pass at the harness tolerances.
- Golden test progress: targeted slices only (64x128, ragged/non-aligned/wide resilience shapes) all PASSED; RM cells XFAIL (excluded).
- Issues encountered: remote JIT precompile server fails to compile (env), ran with --no-precompile; wide shards (Wt=344) clashed CBs with the two resident shards until the window cap was reduced.
- Tests added: tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_height_sharded.py

## Refinement 4a — ROW_MAJOR HEIGHT_SHARDED (in-place stick shards)
- Date: 2026-11-17
- What was done: lifted the RM x HEIGHT_SHARDED EXCLUSION (and the RM sharded-output gate in validate()). RM rms_norm is per-stick, so for an RM shard the core plan groups tile-rows as 32 consecutive sticks of the core's shard (shard-flat, image-independent; ragged last tile-row via per-core `core_sticks` runtime arg -> existing H zero-fill / valid-row writes). Reader/writer RM paths gained `ROW_FLAT` (flat stick geometry) and `IN/OUT_SHARD_RM` (stick pitch = shard_w*es, addressed at the local shard base instead of via a TensorAccessor); W-padded sticks reuse the existing W-tail zero-fill / valid-byte writes. Reused: the whole RM tilize/untilize compute path, CB layout, blocking. Added: 3 CT args per DM kernel, 2 RT args, flat-stick branches.
- Limitation / finding: RM shards are NOT backed zero-copy by a CB (the shard stick pitch shard_w*es differs from the tilize/untilize page width Wt*32*es when W%32 != 0, and tilize/untilize have no partial-row mode); the reader/writer move sticks between the shard and the tilize CBs with local-L1 NoC reads/writes. A zero-copy variant for W%32==0 and shard_h%32==0 is a possible perf follow-up.
- Accuracy achieved: bf16/fp32, gamma/no-gamma, sharded/interleaved output, shapes (1,1,64,128) [1-stick shards], (2,1,96,160), (1,1,333,1000), (3,1,555,1025), (1,1,256,512), (4097,64) pass at rtol=0.05 atol=0.1.
- Golden test progress: slice `HEIGHT_SHARDED and ROW_MAJOR and not bfloat8 and tile_aligned and rank`: 316 passed, 100 xfailed (golden-excluded cells), 0 failed. Unit regression test_rms_norm.py 86 passed, height_sharded TILE 24 passed.
- Issues encountered: writer RT arg offset when N_W == 1 (no peer coords are passed) caused garbage output; fixed.
- Tests added: tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_height_sharded_rm.py

## Refinement 5 — WIDTH_SHARDED and BLOCK_SHARDED with cross-core Σx² combine
- Date: 2026-11-17
- What was done: `WIDTH_SHARDED` / `BLOCK_SHARDED` added to SUPPORTED["memory_layout"] (TILE and ROW_MAJOR; output inherits the shard spec, or follows `memory_config`). Topology = Refinement 2's W-split exchange with the slice pre-placed in L1: each shard is a W slice, the cores of one shard-grid row (WIDTH: all cores; BLOCK: one grid row) form a row group; partial Σx² per row block -> writer unicasts to every core of the group -> compute sums (L1-accumulating pack) -> inv_rms -> pass 2 on the resident shard. TILE shards back cb_x / cb_out zero-copy; RM shards are read/written by stick at the shard pitch (shard width need not be a multiple of 32). Ragged geometry: per-core `slice_valid_elems` / `slice_elem0` RT args; pad tile columns past W are zeroed (one contiguous NoC zero-write per row), pad tiles are never read/written on interleaved sides, gamma tiles/stick segments past W are zeroed/clamped, RM gamma reads start at a 64B-aligned-down address. Rb is capped so the gathered-partials CB stays <= PART_BUDGET. Reused: whole W-split exchange/compute, HEIGHT-shard zero-copy CBs, RM shard stick addressing. Added: CorePlan dataclass + group id, shard->(group,slice) plan, W_PAD CT arg, 2 RT args per DM kernel.
- Bug found/fixed: cb_stat is re-pushed in place (Σ -> combine -> inv_rms); a short tail block (rb < Rb) walked the write pointer past the CB end and the front-relative Col-broadcast read of its stat tiles wrapped (bf16 NaN rows). cb_stat is now 2x (3x with the combine) Rb pages when Rb > 1 (also fixes the latent non-split tail case).
- Accuracy achieved: harness tolerances (PCC>=0.9995 loose perf cases) on WIDTH/BLOCK perf cases (32x{1024,2304,5120,7168} width; 640x{6144,7168}, 7168x{1024,2048}, 8192x1024 block) and resilience shapes, TILE+RM, gamma TILE/RM.
- Golden test progress: slice `(WIDTH or BLOCK) and loose`: 191 passed / 1 failed; slice `test_op ... BFLOAT16 and w_non_aligned`: 420 passed, 120 xfail, 0 failed. Unit regression (height sharded TILE/RM, wsplit, test_rms_norm) 219 passed.
- Issues encountered: 13x777x1023 WIDTH_SHARDED (bf16, fp32_dest=False) fails with a static-CB/L1 clash: its input and output shards are 650KB each (325 tile-rows x 1 tile, 32 slices) leaving no room for the 2-parity partial buffer (32 peers x 2KB x 2). Finding / next lever: send only the valid column of each partial (or a single parity half with an ack) to shrink cb_part under L1 pressure. Remote JIT precompile server fails (env) so the precompile pass is best-effort.
- Tests added: tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_width_block_sharded.py (56 cases: TILE/RM x WIDTH/BLOCK x ragged shapes x gamma)

## Refinement 6 — Speed up the sharded perf cases
- Date: 2026-11-17
- What was done: (1) W-split/shard row-group exchange: when the group is a logical rectangle, cores unicast partials to the leader (peer 0) which multicasts the assembled block once per physically-contiguous sub-rectangle (Blackhole worker columns have NoC holes; all-to-all overlapping multicasts hang, so gather+single-sender mcast; NOC1 start/end swapped); 2 extra semaphores; non-rectangular groups keep the unicast all-gather. (2) inv_rms chain: custom `AddEpsCol`/`RsqrtCol` ops with `VectorMode::C` (faces 0/2 = the valid column of a reduce(ROW) tile), MulUnary skipped when 1/W is folded into the scaler (CT arg). (3) Reader publishes the resident shard before loading gamma; compute waits for gamma lazily before first pass 2. (4) `SHARD_G_CAP`, `MCAST_CHUNK_TILES`, `MCAST_MAX_RECTS`, `USE_MCAST` knobs.
- Device ns (before -> after): WIDTH 32x1024 6.0k->4.8k, 32x2304 9.2k->7.0k, 32x5120 22.5k->15.3k, 32x7168 19.5k->11.6k; BLOCK 640x6144 32.5k->23.4k, 640x7168 36.5k->28.5k, 7168x1024 88.9k->73.6k, 7168x2048 136.8k->108.1k, 8192x1024 122.6k->102.0k.
- Accuracy achieved: PCC>0.9995 asserted in the perf probe on all 9 cases.
- Golden test progress: slice `(WIDTH or BLOCK) and loose and {sizes}`: 17 passed; unit: width/block 56, wsplit+height 85, test_rms_norm 86 passed.
- Issues encountered: first mcast attempts hung (non-worker nodes inside the rectangle; NOC1 coordinate order; then all-to-all multicast). Finding: remaining gap is compute-side per-row SFPU + per-tile FPU cost; SHARD_G_CAP > 16 gave only ~3%.
- Tests added: test_rms_norm_sharded_perf_r6.py (profile probe, RMS_SGCAP / RMS_NOGAMMA env).
