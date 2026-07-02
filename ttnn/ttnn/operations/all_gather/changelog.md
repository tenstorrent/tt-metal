# all_gather — changelog

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
