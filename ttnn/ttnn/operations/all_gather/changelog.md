# all_gather — changelog

## 2026-07-02 — Phase 0 verify (code review + hardening + golden verification)

- **Date**: 2026-07-02
- **What was done**: verification pass over the Phase 2 implementation — code review, golden
  suite + `eval.verify_supported`, precision baseline, refinement queue.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], layout=[TILE, ROW_MAJOR],
  topology=[Linear], gather_dim=[-4], alignment=[tile_aligned, non_tile_aligned]. Unchanged by
  the verifier (no `xpass_drift` — SUPPORTED does not under-claim).
- **Accuracy achieved**: bit-exact identity gather — PCC=1.0, max_abs_err=0.0, mean_abs_err=0.0,
  rel_rms_err=0.0 across dtype {bf16, f32} × layout {TILE, ROW_MAJOR} × alignment {tile-aligned
  (256²), non-tile-aligned (48×64)} (8 cells; golden `metrics_plugin` +
  `test_all_gather_precision_baseline.py`).
- **Golden suite at Phase 0**: curated 56-case subset of the 384-case cartesian (function-scoped
  `mesh_device` + 900 s runner cap make a full run infeasible on sim). Result: supported_pass=8,
  xfail_expected=40, invalid_skipped=8; **supported_fail=0, xpass_drift=0, xfail_wrong_mode=0**
  (all loud categories clean). `verifier_report.json` →
  `generated/all_gather_verify/verifier_report.json`.
- **Issues encountered / fixed**: **1 correctness fix** — WAR hazard on the relay/backward-seed
  `cb_relay_pages` slot. The fabric payload send is `NON_BLOCKING`
  (`edm_fabric_utils.hpp:send_chunk_from_address` issues `noc_async_write(src→EDM)` with no
  flush), so after `write_page` the read of `src` is still in flight; the writer popped the slot
  immediately and the reader (NCRISC) refilled it, racing the fabric read. The forward-seed
  self-copy path already guarded this with `noc_async_write_barrier()`; the backward-seed and
  both relay loops did not. Added the barrier before `cb_pop_front` in those paths (a plain
  `noc_async_write_barrier()` flushes the fabric NoC — same primitive the helper's `drain()`
  uses). The sim executes NoC writes synchronously so it never exposed the race (all cells were
  bit-exact regardless); re-verified relay-heavy 256² + program_cache + output_tensor after the
  fix: still bit-exact, no hang.
- **Tests added**: none new — existing `test_all_gather.py` (22 cases) and
  `test_all_gather_precision_baseline.py` (8 cases) already cover the supported rectangle;
  reviewed and found sufficient (no shape/param gap uncovered).
- **Refinements queued** (`op_requirements.md`): R1 gather_dim {-3,-2,-1} strided concat, R2 Ring
  topology, R3 bfloat8_b — all verifier-authored (no inventory skill applies).

## 2026-07-02 — Phase 2: initial implementation (gather_dim=0, Linear)

Fresh implementation of the self-contained Python all_gather CCL op
(`ttnn.generic_op` + `ttnn.MeshProgramDescriptor`), per `op_design.md`. Does NOT
wrap or dispatch to any existing all_gather / all_gather_async op.

### What was built

- **Algorithm** — bidirectional store-and-forward ring on a 1-D line of N devices.
  Every device runs two worker cores (forward `(0,0)` dir 0 → i+1; backward `(0,1)`
  dir 1 → i-1), each with a reader (NCRISC) + writer (BRISC). No compute kernel
  (pure data movement).
- **Store-and-forward through output DRAM** — a writer fabric-`write_page`s a block
  directly into the downstream device's persistent output DRAM at the block's
  canonical page range (`out_page(c,p) = c*pages_per_shard + p`); the downstream
  reader detects arrival via the counting semaphore and reads the block back out of
  its own output DRAM to forward one more hop. The forward writer also self-copies
  its own shard into its own output block (local `noc_async_write`, every device).
- **Two op-internal `GlobalSemaphore`s** (parked on the descriptor, created once per
  mesh_device + one `synchronize_device`, cached): `barrier_sem` (N-party startup
  barrier via `arm_multicast_inc`, each core reaches `ring_size-1`) and
  `counting_sem` (store-and-forward flow control, one `inc` per block, `Flush`-ordered
  so the block's writes land before the inc). Receivers reset their `counting_sem`
  after the last wait; every writer resets its `barrier_sem` after the barrier —
  cache-reuse re-arm.
- **Fabric egress via the CCL helper** (`ccl_helpers_dataflow.hpp`:
  `FabricStreamSender → FabricStream → {MulticastIncChannel, UnicastWriteChannel,
  AtomicIncChannel}`). The op owns only what the helper banner says it does not:
  the local self-copy, the concat addressing, the barrier/counting WAIT halves, the
  semaphore resets, and the receive ingress (`noc_async_read`).
- **Host assembly** mirrors `point_to_point_program_descriptor.py`: one
  `ProgramDescriptor` per device coordinate on the line; per-direction 1-D routes via
  `ccl_dm_route`; fabric-connection RT args via `setup_fabric_connection` laid out as
  `append_ccl_fabric_rt_args`; worker-core virtual NOC coords via
  `worker_core_from_logical_core` for the barrier-multicast + counting-inc targets.
- **Registry declarations** — `INPUT_TAGGERS` (alignment), `SUPPORTED`
  (dtype {bf16, fp32}, layout {TILE, ROW_MAJOR}, topology {Linear}, gather_dim {-4},
  alignment {both}), `EXCLUSIONS` (empty), and `validate()` (canonicalizes gather_dim
  to negative before the axis gate). Re-exported from the package `__init__.py`.

### Accuracy / test results

**ALL 22 acceptance tests PASS** (`test_all_gather.py`) on the WH T3K all-MMIO sim,
mesh `(1, 8)`, `FABRIC_1D`, via `scripts/run_multidevice_sim_pytest.py --op all_gather`:

- `test_all_gather_gather_dim_0`: 20/20 — {bf16, fp32} × {TILE, ROW_MAJOR} × 5 shard
  shapes (single-tile, multi-tile, non-square, multi-batch, non-tile-aligned). PCC
  meets the per-dtype thresholds (bf16 ≥ 0.995, fp32 ≥ 0.999); gather is identity so
  PCC ~1.0.
- `test_all_gather_program_cache`: PASS — second (cached) call re-gathers correctly
  (semaphore re-arm verified).
- `test_all_gather_output_tensor`: PASS — writes into the supplied replicated output
  and returns the same handle.

Issues encountered: **None** — passed on the first implementation.

### Advisory deviations (from op_design.md)

- **CB sizing**: design suggested `cb_relay_pages` = `2 * pages_per_packet` (2 pages).
  Implemented as a constant 4-page streaming double-buffer with page-granular
  wait/pop. Bounded, constant L1 (independent of shard size); the CB sync invariant
  (push == pop per direction) holds.
- **Flow-control granularity**: one counting `inc` per whole block, rather than the
  design's `chunks_per_sync` sub-block granularity. Functionally identical for all
  test shards (design default `chunks_per_sync = pages_per_shard` for shards ≤ 160
  pages ⇒ one inc/block).

### Scope / follow-ups (refinement candidates vs golden TARGET)

- `topology`: Ring (design's noted extension — wraparound slice-walk, single
  direction). SUPPORTED = [Linear] only.
- `gather_dim`: strided concat for -3 / -2 / -1 (page-contiguous only for -4/outermost).
- `dtype`: bfloat8_b (pure byte movement should carry it, needs verification).
