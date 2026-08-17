# noc_one_packet — measured report

Pure L1→L1 page copy, no compute. Each of 8 cores writes its input shard page by page
into another core's output shard, then barriers once. The issuing RISC-V is the entire
kernel, so its per-call cost lands directly in `DEVICE KERNEL DURATION [ns]`.

Two axes:

* **`variant`** — how the write is issued:
  * `generic` — `noc_async_write(src, dst, PAGE_BYTES)`; the defaulted
    `max_page_size = NOC_MAX_BURST_SIZE + 1` selects the generic multi-burst path.
  * `one_packet` — `noc_async_write<PAGE_BYTES>(...)`, selecting the one-packet path.
  * `generic_runtime_size` — control: the size arrives as a **runtime** arg, so the
    generic path's chunk loop and size arithmetic cannot be folded at compile time.
* **`dest`** — where the pages go:
  * `single` — all pages to ONE destination core.
  * `spread` — page `p` to the `p`-th destination round-robin around the ring.

Correctness is bit-exact against a torch permutation for every variant × axis
combination (30 cases). Perf is reported, never asserted.

---

## Blackhole p150b — box `bh-49-special-mstaletovic-for-reservation-60064`

8 cores, row-major placement, bfloat16, `pages/core = 32`, `kernel-iters = 1`,
3 trials (median), spread = (max−min)/median.

| dest | page_bytes | variant | kernel ns | spread | ns/write | vs generic |
|---|---|---|---|---|---|---|
| single | 64 | generic | 1133 | 3.2% | 35.4 | — |
| single | 64 | one_packet | 1146 | 1.0% | 35.8 | 0.99× |
| single | 64 | generic_runtime_size | 1181 | 0.2% | 36.9 | 0.96× |
| single | 256 | generic | 1320 | 0.9% | 41.2 | — |
| single | 256 | one_packet | 1320 | 2.7% | 41.2 | 1.00× |
| single | 256 | generic_runtime_size | 1290 | 1.6% | 40.3 | 1.02× |
| single | 1024 | generic | 3827 | 1.3% | 119.6 | — |
| single | 1024 | one_packet | 3859 | 0.4% | 120.6 | 0.99× |
| single | 2048 | generic | 7324 | 0.4% | 228.9 | — |
| single | 2048 | one_packet | 7277 | 0.5% | 227.4 | 1.01× |
| single | 4096 | generic | 14412 | 0.4% | 450.4 | — |
| single | 4096 | one_packet | 14401 | 0.4% | 450.0 | 1.00× |
| **spread** | 64 | generic | 1195 | 2.9% | 37.3 | — |
| **spread** | 256 | generic | 1189 | 2.6% | **37.2** | — |
| **spread** | 1024 | generic | 2930 | 3.1% | **91.6** | — |
| **spread** | 2048 | generic | 5294 | 0.4% | **165.4** | — |
| **spread** | 4096 | generic | 10191 | 0.5% | **318.5** | — |

(`one_packet` and `generic_runtime_size` under `spread` are within 0.95–1.02× of
`generic` at every page size; omitted above for brevity, present in the test output.)

---

## Result 1 — the `max_page_size` template argument is a NULL here

`generic` vs `one_packet` is **1.00× (range 0.99–1.02×)** at every page size, on both
destination modes. The `generic_runtime_size` control is also flat, which is the
informative part: even when the size is *not* a compile-time constant — so the generic
path's chunk loop and size arithmetic genuinely survive into the binary — the cost does
not move.

**Mechanism.** For a page that is one packet, the two paths differ by a handful of
register writes and a statically-false loop test. That software cost is dwarfed by the
NoC transaction itself: at 2048 B the copy costs 229 ns/write (`single`), and both paths
spend essentially all of it inside the `while (!noc_cmd_buf_ready(...))` spin waiting for
the command buffer to come back. Selecting a cheaper *issue* sequence does not help when
issue is not what you are waiting for.

**So: do not reach for the template argument as a perf lever on this class of transfer.**
It is still the more honest way to express a known-single-packet write, and it costs
nothing — but it is a readability choice, not a speed one.

## Result 2 — destination concurrency IS a lever (up to 1.41×)

Holding bytes, transaction count and variant fixed, and changing only *which core each
page goes to*:

| page_bytes | single (ns/write) | spread (ns/write) | speedup |
|---|---|---|---|
| 64 | 35.4 | 37.3 | 0.95× |
| 256 | 41.2 | 37.2 | **1.11×** |
| 1024 | 119.6 | 91.6 | **1.31×** |
| 2048 | 228.9 | 165.4 | **1.38×** |
| 4096 | 450.4 | 318.5 | **1.41×** |

**Mechanism.** Writes aimed at one destination serialise at that core's receive NIU: the
issuing core's next transfer cannot make progress until the previous one has been taken.
Writes aimed at *different* destinations drain concurrently, so the sender keeps more
bytes in flight. The gain grows with page size because the serialised portion grows with
payload — and it inverts below ~256 B, where per-transfer fixed cost dominates and the
extra address computation per page is no longer free.

**Read this as: in a gather/scatter, prefer many destinations over one.** The crossover
is around 256 B on this part; measure yours.
