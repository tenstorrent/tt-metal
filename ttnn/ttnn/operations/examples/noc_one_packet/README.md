# noc_one_packet — is `noc_async_write`'s page-size template argument a perf lever? (no) and what is (destination spread)

**Difficulty:** ⭐⭐ T2 · **Concept(s):** (1) the `max_page_size` template argument on `noc_async_write`; (2) destination concurrency of a page stream
**First profiled on:** `bh-49-special-mstaletovic-for-reservation-60064` · BH · 1350 MHz · 2026-08-17

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem

`noc_async_write` is declared with a defaulted template parameter:

```cpp
template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, ...>
inline void noc_async_write(uint32_t src, uint64_t dst, uint32_t size, ...) {
    if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
        noc_async_write_one_packet<...>(src, dst, size, ...);   // cheap path
    } else {
        ncrisc_noc_fast_write_any_len<...>(...);                // generic path
    }
}
```

The default is `NOC_MAX_BURST_SIZE + 1`, so a **plain call always compiles to the generic
multi-burst path** — even for a page that is provably one packet (a burst is 16 KB on this
part). It is tempting to read that as free performance left on the table: name the size as
the template argument and get the cheap path. This example asks whether that is true, and
measures what actually governs the cost of a stream of page writes.

## What this isolates — and how

- **Concept:** the per-call *issue* cost of a NoC write (which path the compiler picks),
  and separately, how many distinct destinations the stream targets.
- **Isolation setup:** Tensix↔Tensix NoC — **no compute**, input *and* output both
  L1-sharded so no DRAM bandwidth or bank contention is in the measurement, no circular
  buffers, no semaphores. One data-movement kernel per core walks its own input shard and
  writes it page by page into another core's output shard, then barriers once. The issuing
  RISC-V is therefore the *entire* kernel, so its per-call cost lands directly in the
  device kernel duration rather than hiding behind something else.
- **Why it's kernel-level:** both the template argument and the destination order are
  chosen by whoever writes the kernel loop; neither changes the data, the layout, or the
  bytes moved.

## The methods being compared

| Variant | What it does | Why it should differ |
|---|---|---|
| `generic` *(baseline)* | `noc_async_write(src, dst, PAGE_BYTES)` | the default — generic multi-burst path |
| `one_packet` | `noc_async_write<PAGE_BYTES>(src, dst, PAGE_BYTES)` | selects the one-packet path: fewer registers programmed, no chunk-loop test |
| `generic_runtime_size` | plain call, size arrives as a **runtime** arg | control: the generic path's chunk loop and size arithmetic genuinely survive into the binary, so this shows what a compile-time size is worth on its own |

Orthogonal axis, same bytes and same transaction count either way:

| `dest` | What it does |
|---|---|
| `single` | every page to ONE destination core |
| `spread` | page `p` to the `p`-th destination round-robin around the ring |

## Headline numbers (BH p150b, 8 cores, bf16, 32 pages/core, median of 3)

**The template argument is a null.** `generic` vs `one_packet` is **1.00× (0.99–1.02×)** at
every page size from 64 B to 8 KB, on both destination modes. The `generic_runtime_size`
control is flat too.

**Destination spread is a real 1.41×**, holding bytes, transaction count and issue path fixed:

| page_bytes | `single` ns/write | `spread` ns/write | speedup |
|---|---|---|---|
| 64 | 35.4 | 37.3 | 0.95× |
| 256 | 41.2 | 37.2 | **1.11×** |
| 1024 | 119.6 | 91.6 | **1.31×** |
| 2048 | 228.9 | 165.4 | **1.38×** |
| 4096 | 450.4 | 318.5 | **1.41×** |

Full tables, both axes, all three variants: [`report.md`](report.md).

## Why

**Why the template argument does nothing.** For a single-packet page the two paths differ
by a handful of register writes and a statically-false loop test. That software cost is
dwarfed by the transaction: at 2048 B a write costs 229 ns, and both paths spend nearly all
of it inside `while (!noc_cmd_buf_ready(...))`, waiting for the write command buffer to be
released — which happens only once the NIU has taken the previous transfer. Choosing a
cheaper *issue* sequence cannot help when issue is not what you are waiting for. The
runtime-size control confirms it: even with the chunk loop genuinely present, the cost is
unchanged.

**Why destination spread does.** Writes aimed at one core serialise at that core's receive
NIU — the sender's next transfer cannot progress until the previous one is accepted. Writes
aimed at different cores drain concurrently, so more bytes stay in flight. The win grows
with page size because the serialised portion grows with payload, and it **inverts below
~256 B**, where per-transfer fixed cost dominates and the extra per-page address
computation stops being free.

**The transferable lesson:** a stream of page writes is limited by how fast destinations
will *accept* transfers, not by how cheaply the sender can *issue* them. Optimise the
former (spread destinations, keep pages ≥ ~256 B); do not expect the latter to pay.

## CLI — measure your own shapes/params

```bash
# default sweep: 3 variants x 2 destination modes x page sizes
python -m ttnn.operations.examples.noc_one_packet

# your own page sizes / page counts / core count
python -m ttnn.operations.examples.noc_one_packet --page-elems 512,2048 --pages-per-core 64 --cores 16

# steady-state instead of per-launch latency
python -m ttnn.operations.examples.noc_one_packet --kernel-iters 50

# one variant only
python -m ttnn.operations.examples.noc_one_packet --variant one_packet
```

Or drive the tests directly:

```bash
# correctness: every variant x destination mode is a bit-exact permuted copy
scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/examples/test_noc_one_packet.py::test_noc_one_packet_correctness

# the measured table
scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/examples/test_noc_one_packet.py::test_noc_one_packet_device_perf
```

## Caveats

- Numbers are BH p150b only; the crossover page size is a property of the part.
- `pages/core = 32` throughout the headline table. Raise `--pages-per-core` to confirm the
  per-write costs are flat in count (they are, on this part).
- Correctness is bit-exact (it is a copy); perf is reported and never asserted, so a noisy
  or inverted result is data, not a test failure.
