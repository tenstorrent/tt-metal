# posted_writes: bake-off of posted DRAM tile writes and the set_state issue path for `store_rows`

Box: WH B0 n150, 64 Tensix cores, AICLK 1000 MHz (1 cycle = 1 ns). Metric: `DEVICE KERNEL DURATION [ns]`
for the whole tilize op. Each cell is the median of 4 `--profile` runs.

## Variants (`gen_variants.py` builds `kernels_<name>/` from `../../kernels`; only the writer's `store_rows` changes)
| variant | tile write | CB-slot release / kernel end |
|---|---|---|
| baseline | `noc_async_write` (any-length path), non-posted | `noc_async_writes_flushed` / `noc_async_write_barrier` |
| onepkt | `noc_async_write_one_packet`, non-posted | same as baseline |
| onepkt_posted | `noc_async_write_one_packet<posted=true>` | `noc_async_posted_writes_flushed` / same |
| posted | `noc_async_write<posted=true>` | `noc_async_posted_writes_flushed` / same |
| state | `ncrisc_noc_write_set_state` once per quantum, then per tile: coordinate register + `ncrisc_noc_write_with_state` (4 register writes instead of 6), non-posted | same as baseline |
| state_posted | same issue path, posted | posted flushed / same |

## Run
```
TILIZE_PW=baseline,onepkt,onepkt_posted,posted,state,state_posted TILIZE_PW_CHAIN=0 \
  scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/tilize/test_tilize_perf1_posted_writes.py
pytest --collect-only -q <same test> | grep test_pw > ids.txt ; python table.py ids.txt <ops csv> [...]
```
A plain run (no `--profile`) is the bit-exact gate. `test_chain` runs tilize and then `ttnn.add(out, out)`
40 times back to back, with fresh inputs and reused buffers.

## Result: NULL. The posted variants are also unsafe as the op's final output.
Focus case [1,1,16384,64]: baseline 24078, onepkt 24348, onepkt_posted 24040, posted 24072, state 24322,
state_posted 23380 ns. Every variant is within the ±3% noise band on every shape with at least 2k tiles.
Posted writes are only faster on latency-bound shapes: [1,1,128,64] at -12% (about 300 ns) and [1,1,32,2048]
at -5%. The zones show where that time comes from. Posted writes remove only `writer_barrier`, the wait for
the final write acks (about 300-400 cycles), and leave `writer_issue` and `writer_flush` unchanged. Those two
are limited by injection back-pressure, not by ack traffic. So the only thing a posted write saves is the
guarantee that the data has landed.

Why posted writes are unsafe (there is no way to know a posted DRAM write has landed):
- `noc_nonblocking_api.h` (wormhole):488 — posted clears `NOC_CMD_RESP_MARKED`, so no ack is sent. :582-583 —
  the only posted counter is `NIU_MST_POSTED_WR_REQ_SENT`, which counts writes that have left the Tensix core. :593-594 — the ack counter exists only for non-posted writes.
- `dataflow_api.h`:1822-1844 — `noc_async_posted_writes_flushed` waits for the writes "to depart, but will
  not wait for them to complete". No posted barrier exists.
- `noc.h` (wormhole):24-27 — the NoC orders transactions only when they are linked and go to the same destination. :62-66 —
  a non-posted ack is the way "to ensure the writes are flushed".
- `brisck.cc`:85-95 and `brisc.cc`:549-563 — after `kernel_main`, the firmware only ASSERTs, and only in debug
  builds. For posted writes the ASSERT checks "sent". `brisc.cc`:575/590 then signals done with a posted inline
  write to the dispatcher (`firmware_common.h`:250-259). No barrier sits in between.
- `cq_dispatch_subordinate.cpp`:263-278 `wait_for_workers` only counts done signals, so it does not flush DRAM
  before the next program starts.

The chain test passed for every variant. This is expected, because the race window (hundreds of ns) is far
shorter than the time to launch the next program. Passing it does not prove the posted writes are safe.
