# L1 Ledger: clone

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes are the ones in `op_design.md` → Blocking Model: `page` (block extent `core_pages`, streamed through a window of `ring_entries_per_thread` per thread pair) and `in_page` (block extent `entry_bytes`, never split). `outer`, `rows` and TILE `cols` fold into `page`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_staged_pages` (DFB `"staged_pages"`, entry = `entry_bytes`) | `ring_entries_per_thread · R` (Phase 0: 4 · 1 = 4) | Ideal peak: `ring_entries_per_thread · R` entries simultaneously in flight, which is the goal (reads landing ahead of writes draining). Correctness needs 1. | `{page: streams → window ring_entries_per_thread per thread pair (× R pairs); in_page: spans → entry_bytes (one whole page per entry); outer/rows/cols: fold into page, streams}` | `input.dtype` (Float16_b in Phase 0; Float32 for TARGET float32). A dataflow-only payload: moved by the NoC, never unpacked or packed, never in DEST, so audit 2's DEST-width rule has no DEST to compare against. Its width is the tensor's own, by R1. | reader | writer | the whole program (one block per cluster) | **Shares with nothing; it is the only buffer.** Capacity > the correctness minimum of 1 entry, and the gap is pipelining: ≥ 2 implicit-sync transaction-id slices, so the ISR credits one slice while the next is on the NoC, and ≥ 2 reads in flight per slice (catalog `double_buffer`). Not aliasable: there is no other buffer with a disjoint lifetime. It is not removable, because a NoC read lands in the issuing cluster's L1 and a write leaves from it, so a staging slot is needed for any non-local endpoint (the ring-free form is the `owner_local` regime, deferred). |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `page_bytes` | `input.buffer_page_size()`: tile bytes (TILE) or `W · elem_bytes` (ROW_MAJOR) | `1 ≤ page_bytes ≤ 65,536` | Host refusal `entry_bytes > NOC_MAX_BURST_BYTES` → `NotImplementedError` (`noc_parameters.h:395-396`; `dataflow_buffer.cpp:560`). INPUTS max: 16,384 (bf16 W=8192); TARGET max: 32,768 (fp32 W=8192). |
| `entry_bytes` | `align_up(page_bytes, ttnn.get_l1_alignment())` | `page_bytes ≤ entry_bytes ≤ min(input.buffer_aligned_page_size(), output.buffer_aligned_page_size())`, and `≤ 65,536` | By construction (both buffers' alignments ≥ L1 alignment), asserted on host. Same refusal as above. |
| `R` (= W) | `THREAD_PLAN` reader / writer thread count | `R ∈ {1, 2, 3}` (R + W ≤ 6, R = W) | Host validation of `THREAD_PLAN` (Phase 0: 1) |
| `ring_entries_per_thread` | depth knob | even, `2 ≤ ring_entries_per_thread ≤ min(RING_ENTRIES_PER_THREAD, floor(1,048,560 / (entry_bytes · R)))` | Host clamp against the DFB ring cap (target page *Cluster model*), `ring_entries_per_thread_for()`. Phase 0: 4. (Verifier: the earlier bound `min(8, …)` named a cap the code does not apply; the knob itself is the upper bound.) |
| `core_pages` | this cluster's contiguous page count | `≤ ceil(tensor_pages / num_cores)` (rounded up to whole rounds of R) | `ttnn.split_work_to_cores`. Appears in no capacity: the ring streams over it. |
| L1 budget | `ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes` | value on quasar: queried (≤ 4,194,304 B per cluster; the target page does not give the unreserved figure) | Not used in any predicate: the footprint is bounded by the DFB ring cap (1,048,560 B) first. |

## Total per-core footprint

`footprint = ring_entries_per_thread · R · entry_bytes`

- It scales linearly with the depth knob (`ring_entries_per_thread`), the thread knob (`R`) and the page geometry (`entry_bytes`, set by dtype, layout and, for ROW_MAJOR, W).
- It does not scale with `core_pages`, `tensor_pages` or any op dimension beyond one page.
- Phase 0 on quasar: 8,192 B for any bf16 TILE input; ≤ 65,536 B for bf16 ROW_MAJOR on INPUTS (W = 8192).
- Worst legal case: 8 × 3 × 65,536 clamped to the ring cap = 1,048,560 B.
- This is far under 4 MiB, so no budget solve, safety fraction or blocking search exists in this design (inventory = 1 buffer, the minimum for a staged NoC copy).

## Data-movement budget

For the chosen split (contiguous page ranges, `interleaved_page_stream`). T = `tensor_pages · entry_bytes`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 if in DRAM, 0 if in L1 | Every page is read exactly once, into the ring of the cluster that owns its page range. No second pass and no re-read: nothing in the op needs a page twice. | L1 input: T over the NoC from its bank (except the ~1/n_banks of pages whose bank is the reading cluster). DRAM input: none beyond the DRAM read. |
| output | 1 if in DRAM, 0 if in L1 | Every page is written exactly once, from the ring. | L1 output: T over the NoC to its bank (except the locally owned fraction). |

Totals per tier for the traced calls on quasar (bf16):

| Call | DRAM bytes | Cross-core NoC bytes |
|------|-----------|----------------------|
| `[1,1,1600,512]` TILE, DRAM → L1 (largest) | 1,638,400 (read) | ~1,638,400 (write to L1 banks) |
| `[1,1,400,512]` TILE, DRAM → L1 | 425,984 | ~425,984 |
| `[1,1,400,64]` RM, L1 → L1 | 0 | ~2 × 51,200 (read from banks + write to banks) |

> Cheapest-traffic split considered: bank-affine page sets (`owner_local`: the core owning a page's L1 bank reads the input page directly into the output slot, or writes its local input page out). Delta: −T cross-core NoC bytes per L1 endpoint, with no change to DRAM bytes: −1,638,400 B NoC on `[1,1,1600,512]` DRAM → L1, and −51,200 B on `[1,1,400,64]` L1 → L1. Implemented: contiguous page ranges (`interleaved_page_stream`). Deferred because the host had no L1-bank → core map to assign pages by bank, and the built regime is required in any case for DRAM → DRAM (in TARGET), where the two splits tie. The built structure keeps it reachable: a host-built per-core page assignment passed as runtime args, `TensorAccessor` page-id addressing in the kernels (which yields the bank address an owner check needs), and per-regime kernel selection behind one `validate()` and one `generic_op` dispatch.

> **Verifier note (Phase 0 audit).** The "no binding exposes the bank → core map" half of the deferral does not hold as written: `ttnn.reports.get_buffer_pages(device)` returns `BufferPageInfo` with `core_x`, `core_y`, `bank_id` and `page_index` per page (`ttnn/cpp/ttnn-nanobind/reports.cpp:38-52`; not yet run on Quasar), and a kernel can also tell a local page from the NoC address `TensorAccessor` gives for it. The other half (DRAM → DRAM needs the built regime anyway) stands. The cheaper split is therefore queued as a measured perf refinement (`op_requirements.md`, Refinement 2), gated on the cross-core NoC bytes above.
