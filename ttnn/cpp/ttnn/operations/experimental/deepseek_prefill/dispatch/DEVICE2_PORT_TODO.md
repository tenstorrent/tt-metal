# deepseek_prefill/dispatch — Device 2.0 port: status and remaining work

Working notes for finishing the Device 2.0 port of the dispatch data-movement kernels and preparing the op
for Metal 2.0. Delete this file once the port is complete.

Kernels: `device/kernels/dataflow/{reader_worker_dispatch,writer_worker_dispatch,writer_sender_dispatch}.cpp`.
`device/kernels/compute/untilize_dispatch.cpp` and `dispatch_plan.hpp` need no changes.
The gemma4 copies under `models/demos/gemma4/custom_ops/moe_dispatch_prefill/kernels/` are separate files and are
not part of this port.

## Done (this branch)

Mapped per `docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md`. Each
new call lowers to the same low-level NoC primitive as the old one (WH/BH):

| Legacy | Device 2.0 |
|---|---|
| `cb_reserve_back/push_back/wait_front/pop_front`, `get_write_ptr/get_read_ptr` | `CircularBuffer` methods |
| `noc_async_read_page(id, accessor, l1)` | `noc.async_read(accessor, cb, accessor.get_aligned_page_size(), {.page_id}, {.offset_bytes})` |
| `noc_async_write_one_packet_with_trid(src, get_noc_addr(x,y,a), n, trid)` | `noc.async_write<NocOptions::TXN_ID, NOC_MAX_BURST_SIZE>(CoreLocalMem, UnicastEndpoint{}, n, {}, {x,y,a}, {.trid})` |
| `noc_async_write_barrier_with_trid(trid)` | `noc.async_write_barrier<NocOptions::TXN_ID>({.trid})` |
| `noc_async_{read,write,atomic,full}_barrier`, `noc_async_writes_flushed` | `Noc` methods |
| `get_semaphore` + `noc_semaphore_set/wait_min` (reader turn semaphore) | `Semaphore<>::set/wait_min` |
| raw `reinterpret_cast<[volatile] tt_l1_ptr T*>` L1 pointers | `CoreLocalMem<[volatile] T>` (volatility kept) |

## Still legacy (not ported)

1. **Posted remote semaphore increments** — `noc_semaphore_inc<true>(...)`:
   `writer_worker_dispatch.cpp` (data_avail, sparse and non-sparse paths) and `writer_sender_dispatch.cpp`
   (ring space_avail). `Semaphore::up()` is non-posted, so swapping changes timing and what
   `async_atomic_barrier()` waits on. Needs a decision (non-posted `up()` + barrier, or a posted Device 2.0 primitive)
   and perf numbers. Same open question as moe_gate_mm (#63938).
2. **`noc_async_write_set_trid(0)`** in `writer_worker_dispatch.cpp` — no Device 2.0 sticky-trid reset.
   `async_write<TXN_ID>` still leaves the trid in the write cmd buffer, so the reset is still required.
3. **`get_semaphore()` used as a raw address** — the cross_addr mailbox (worker writer + sender) and the data_avail /
   space_avail NoC addresses feeding item 1.
4. **`noc_inline_dw_write`** of the three ring base addresses in the sender handshake. `Noc::inline_dw_write`
   (`api/dataflow/noc.h`) exists but is not in the migration guide; check that `INLINE_L1` / the default destination type
   match the legacy call before swapping.
5. **Sender init/exit semaphores** (`init_semaphore_address`, `exit_semaphore_address`) — raw global-semaphore
   addresses with `noc_semaphore_wait/set`. `Semaphore<>` takes an ID, so this needs a host-side change.
6. **Sender `ring_data_avail_ptr[]` poll** — array of raw volatile pointers with a manual `invalidate_l1_cache()`.
   Candidate: `Semaphore<>(ring_data_avail_id[s]).value()` (it invalidates the cache itself).
7. **Fabric packet-header pointers** — raw pointers required by the fabric API; probably stays.

Next step: run the Metal 2.0 pre-port audit (`/metal2-audit experimental/deepseek_prefill/dispatch`) to see which of
these block the Metal 2.0 port.

## What has been tested

On a single Wormhole n150 only:
- Local dispatch→combine round trip (the `_round_trip` helper from `test_dispatch_combine_local.py`, 1x1 mesh,
  fabric disabled): bit-exact for 2 seeds, program cache reused. The pytest itself cannot run on one chip:
  its `FABRIC_1D` setup needs ≥ 2 chips.
- JIT compile of every kernel variant, original vs. ported, with the captured JIT command: reader {tile,
  row_major, +padding, +fp8, fabric 1D}, worker writer {base, padding, fp8, sparse-mcast combos}, sender {local,
  fabric 1D, 1D + sparse-mcast, fabric 2D}. All compile with no new warnings.

**None of the cross-device paths have run on hardware.**

## To test on a multi-device machine

Use `TT_METAL_RUNTIME_ROOT=$PWD` (or re-run the build) so the JIT compiles the repo sources, not stale
`build_Release/libexec` copies. Run once with `TT_METAL_WATCHER=1` as well (NoC sanitizer, trid range checks).

```bash
T=models/demos/deepseek_v3_d_p/tests
pytest $T/op_unit_tests/test_dispatch_combine_local.py                  # local-1x1, local-1x4 (QB2)
pytest $T/op_unit_tests/test_prefill_dispatch.py                        # 4x1 / 8x1 / 8x4 fabric2d; tile + row_major, bf16 + fp8_scaled
pytest $T/op_unit_tests/test_ttnn_dispatch_combine.py
pytest $T/op_unit_tests/test_dispatch_fabric2d.py                       # includes padding_config (HAS_PADDING_CONFIG)
pytest $T/op_unit_tests/test_dispatch_combine_l1_small_semaphores.py
pytest $T/pcc/test_ttnn_moe.py                                          # end-to-end MoE incl. padding_config
pytest $T/perf/test_dispatch_fabric2d_perf.py                           # compare against main
```

Paths that need coverage (check that the selected configs hit each one):
- Cross-device worker-writer path: TRID writes to the sender's c_4/c_5/c_6 and the sentinel slot.
- `SPARSE_MCAST_DISPATCH` (1D ring, tile input): grouped route_info + payload, sender sparse multicast.
- `FP8_SCALED` with row-major input: scale tail through the local, legacy cross-device and sparse paths.
- Fabric 1D and fabric 2D sender handshakes (init/exit semaphores), multi-hop 2D routes.
- Non-owner baton ring (`core_id != 0`): offsets pulled from and pushed to the owner core.

Perf: dispatch latency should match `main`. The TRID writes keep the one-packet path
(`max_page_size = NOC_MAX_BURST_SIZE`), so no change is expected.
