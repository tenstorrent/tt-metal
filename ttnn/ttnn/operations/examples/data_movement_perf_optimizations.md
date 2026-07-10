# Data-movement performance optimizations — optimal vs non-optimal

A cross-codebase checklist of what separates an **optimal** data-movement op from a
**non-optimal** one in tt-metal. The recurring theme:

> **Optimal** = keep data in L1, move large coalesced transactions, overlap read/write
> streams, and specialize at compile time.
> **Non-optimal** = stream small pages through DRAM with generic runtime address-gen and a
> barrier per transaction.

Every item has a code pointer (file + line on this branch). The two authoritative deep
references are the tech reports:
- `tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md` — DRAM bandwidth + NoC
  congestion (the theory behind sections A and B).
- `tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md`
  — host-dispatch overlap (section E).

The dataflow kernel API surface is `tt_metal/hw/inc/api/dataflow/dataflow_api.h`; the
host-side patterns live under `ttnn/cpp/ttnn/operations/data_movement/`. Line numbers cited for
the surviving data-movement ops (untilize, tilize, transpose, concat, move) are exemplary — the
same patterns recur across factories.

---

## A. Core / grid placement — the biggest single lever

1. **Spread worker cores across the DRAM-facing axis, not down one axis.** DRAM banks sit in a
   small number of columns (WH: banks across a 12×1 DRAM grid), so a line of readers stacked on
   one axis piles their traffic onto the same shared NoC links and stops scaling; spreading the
   line across the bank-facing axis keeps routes disjoint. The `split_work_to_cores` core order
   (`row_wise`) is the knob that decides which line you get.
   - `tt_metal/api/tt-metalium/work_split.hpp:46` (`split_work_to_cores`).
   - Theory: `Saturating_DRAM_bandwidth.md:4-13` (single-bank saturation, reader placement).

2. **Launch only on cores that actually hold data.** Returns exactly the cores with shards (and
   maps each DRAM bank to its NoC-optimal worker); launching on empty cores wastes cores and can
   fault.
   - `get_optimal_worker_cores_for_sharded_tensor()` — `ttnn/core/tensor/tensor_utils.cpp:54`.
   - Consumers on this branch: `untilize/device/factories/untilize_multi_core_program_factory.cpp:330`,
     `pad/device/pad_rm_sharded_width_only_program_factory.cpp`.

3. **Reader adjacent to its DRAM bank; one reader ↔ one bank.** Placing the reader next to its
   bank makes responses take one NoC hop and keeps routes from overlapping. Multiple readers per
   bank congests.
   - `Saturating_DRAM_bandwidth.md:4-13`.

4. **Cliff-core specialization** — split into "full" cores + one remainder ("cliff") core so work
   divides evenly; skip generating the cliff kernel when empty.
   - `work_split.hpp:46`; `untilize_multi_core_program_factory.cpp:132` ("No need to double buffer
     if the core is only processing a single block"), cliff handling at `:396-400`.

---

## B. Transaction shape & the NoC (kernel level)

5. **Coalesce into whole-page transactions; don't scatter sub-tile faces.** One large NoC write
   beats many small ones — bigger transactions hit higher achieved DRAM bandwidth. A barrier after
   every tiny sub-transaction pays full NoC latency with no overlap.
   - `Saturating_DRAM_bandwidth.md:11-13`; dispatch on size at `dataflow_api.h:566` (see #6).

6. **Hit the one-packet fast path.** `noc_async_read`/`write` dispatch on page size: transfers
   ≤ `NOC_MAX_BURST_SIZE` (**512 B** on WH) take the cheap single-packet path; larger go through
   the slow `_any_len` loop.
   - `dataflow_api.h:551,566`; `NOC_MAX_BURST_SIZE` — `.../wormhole/noc/noc_parameters.h:219`.

7. **Keep many transactions in flight — one barrier per *block*, not per transaction.** Issue a
   block of reads, then a single `noc_async_read_barrier()`. One-read-per-barrier is latency-bound;
   the barrier idles DRAM between blocks.
   - `Saturating_DRAM_bandwidth.md:11`.

8. **Best practice — transaction-ID (trid) double-issue.** Tag each block and barrier only on the
   *previous* id, so there's always ≥1 request in flight — no DRAM bubble.
   - `dataflow_api.h:2366` (`noc_async_read_set_trid`) + the trid barrier/with-state family;
     `Saturating_DRAM_bandwidth.md:11-13`.

9. **Split streams across NoCs.** Reader on **NoC0**, writer on **NoC1**, so read and write
   streams overlap instead of contending. (Reads prefer NOC_0 +x/+y; writes NOC_1 −x/−y.)
   - VC defaults: `tt_metal/hw/inc/internal/dataflow/dataflow_api_common.h:62-63`
     (`NOC_UNICAST_WRITE_VC 1`, `NOC_MULTICAST_WRITE_VC 4` on WH).
   - `preferred_noc_for_dram_read/write` — `tt-metalium/kernel_types.hpp`.

10. **Per-reader VC assignment** to break first-come-first-serve serialization when readers share a
    route — the NoC round-robins instead.
    - `Saturating_DRAM_bandwidth.md`; `vc`/`use_vc` params on `dataflow_api.h` read/write.

11. **Alignment** — 32 B DRAM-read / 16 B DRAM-write; misaligned transactions get split or do
    read-modify-write. Address generators auto-align page size.
    - `.../wormhole/noc/noc_parameters.h:295-296`;
      `tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:289` (`aligned_page_size`).

12. **Multicast instead of N unicasts** for broadcast — one write fans out to a rectangle of
    receivers.
    - `dataflow_api.h:932` (`noc_async_write_multicast`);
      `tech_reports/prog_examples/multicast/multicast.md`.

13. **`set_state`/`with_state` stateful transfers** to configure the command buffer once when
    issuing many same-shape transfers to varying addresses.
    - `dataflow_api.h:594,627` (one-packet set-state / with-state).

---

## C. Buffering & data residency (host + kernel)

14. **Zero-copy: alias the circular buffer directly onto the shard buffer** (L1↔L1). The reader
    "just pushes" instead of copying. Requires input *and* output in L1 — any DRAM side forces a
    NoC copy.
    - `untilize_multi_core_program_factory.cpp:103-116` (`cb_backing_buffer = src0_buffer`,
      "zero-copy backed CB (fast production path)");
      `tilize/device/tilize_device_operation.cpp:22` (`can_use_sharded_optimized_factories`);
      `untilize/device/untilize_device_operation.cpp:315-316` ("identical factories use backed CBs
      (zero-copy)").

15. **Prefer sharded (L1-resident) over interleaved for DRAM-bound ops** — sharding gives each
    reader its own bank and hits >92% BW vs interleaved congestion.
    - `Saturating_DRAM_bandwidth.md`.

16. **Double-buffer CBs (depth 2) — but only when it pays.** Depth-2 lets one block fill while
    another drains; single-block cores skip it to save L1.
    - `untilize_multi_core_program_factory.cpp:132`; `concat/device/concat_program_factory.cpp:111`
      ("Depth=2 is a prefetch optimization; fall back to depth=1 when it would overflow L1");
      `transpose/device/transpose_wh_sharded_rm_program_factory.cpp:117,157` (`// double buffer`).

17. **In-place / no-copy when buffers don't overlap** (move op) — only copy through a CB when
    regions actually overlap.
    - `move/move.cpp:69,89-92` (`non_overlap`), `:107` (`MULTI_CORE_OVERLAP` only when overlapping),
      `:148` (`MULTI_CORE_SHARDED`).

---

## D. Compile-time specialization & program caching (host level)

18. **Bake `TensorAccessorArgs` as compile-time args** so address generation is unrolled per buffer
    type, not computed at runtime.
    - `untilize_multi_core_program_factory.cpp:175` (reader), `:209` (writer).

19. **Pass only buffer base addresses as runtime args** so the program caches and only the address
    is patched on re-run.
    - `untilize_multi_core_program_factory.cpp:330,335,396` (`src0_buffer` / `dst_buffer` as runtime
      args).

20. **Layout / special-case factory selection** — pick the specialized factory by layout match and
    fall back to the generic streaming factory only when nothing matches.
    - `untilize/device/untilize_device_operation.cpp:285` (`select_program_factory`), `:310`
      ("Optimized special case … neither input nor output is sharded"), `:315-316` (identical L1
      sharded specs → zero-copy), `:346-349` (`threshold_row_block` heuristic).

21. **Precompute per-core indexing host-side** (`tile_start_index`, block counts) so kernels don't
    recompute shard addresses; use `InterleavedAddrGenFast` (shifts, not multiplies) for pow2 pages.
    - `untilize_multi_core_program_factory.cpp:335,396` (`tile_start_index` per core);
      `dataflow_api_addrgen.h:349` (`InterleavedAddrGenFast`).

---

## E. Host-dispatch overlap (whole-model level)

22. **Metal Trace** to remove per-op host dispatch overhead; **multiple command queues + events**
    to overlap input I/O (CQ1) with execution (CQ0).
    - `AdvancedPerformanceOptimizationsForModels.md:33` (Metal Trace), `:157,161` (Multiple Command
      Queues).

---

## Compact optimal-vs-not table

| Dimension | Non-optimal | Optimal |
|---|---|---|
| Core placement | line stacked on one axis | spread across bank-facing axis; only cores with data |
| Transaction size | many <512 B sub-transactions | coalesced whole pages, one-packet ≤512 B |
| Barriers | one per transaction | one per block → trid double-issue |
| Streams | shared NoC | reader NoC0 / writer NoC1, per-reader VCs |
| Residency | stream through DRAM interleaved | L1 sharded, CB aliased to shard (zero-copy) |
| Args | runtime address-gen | compile-time `TensorAccessorArgs`, cached program |

---

## Notes

- Numbers/effects behind sections A–B are grounded in
  `tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md` (>92% DRAM BW achieved on
  Wormhole). Use `/perf-roofline-dm` to turn a proposed transfer scheme into a predicted target and
  `/perf-measure` to measure the real number on device.
- This checklist is the source of Mode-A candidate levers for `/perf-roofline-dm` — walk it when
  enumerating competing algorithm ideas, then estimate each on your own transfers.
- Some ops referenced by earlier drafts (`interleaved_to_sharded`, `sharded_to_interleaved`,
  `reshard`) are absent on this branch (nuked for agent eval); the surviving `untilize` / `tilize` /
  `transpose` / `concat` / `move` factories illustrate the same host-side patterns.
