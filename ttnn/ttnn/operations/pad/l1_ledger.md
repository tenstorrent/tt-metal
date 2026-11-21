# L1 Ledger: pad

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Target for worked values: quasar on ttsim
(budget `ttnn.get_max_worker_l1_unreserved_size()` = 3,963,328 B, `l1_reserve_bytes` = 0). Thread plan
(R, C, W) = (1, 0, 1), so `T = max(R, W) = 1`.

Block axes of the Blocking Model: **data rows** (extent `block_rows`), **pad rows** (extent
`pad_rows_per_barrier`), **row elements** (extent = whole row, `W_o`).

Inventory: two rings, both the minimum for this program.
- One buffer carries data rows between the reader (which assembles them) and the writer. NoC transfers cannot go DRAM→DRAM, and a DM kernel cannot self-loop a ring on Gen2 (`validation/resource/dfb.cpp:637-653`).
- One read-only template row is the source of every pad-row write.
- There is no intermediate. Each data row is assembled in the entry the writer drains: the read lands in place, and the margins are written in place. This is the "pack into destination" pattern applied to a re-layout.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `dfb_out_rows` | `T · ring_blocks · block_rows` entries of `entry_bytes = align_up(W_o·elem_bytes, l1_alignment)` | `rows_resident` (`ring_blocks = 1`): `block_rows` = the core's whole data assignment, all live at once (read in flight, then drained). `rows_streamed` (`ring_blocks = 2`): `2 · block_rows` (block k draining while k+1 fills). | `{data rows: spans → block_rows · ring_blocks, pad rows: streams → none (pad rows never enter this ring), row elements: spans → whole row (entry_bytes)}` | input dtype, raw bytes (Float16_b for bf16 / Float32 for fp32). Dataflow-only payload, never through DEST; there is no compute kernel and no `fp32_dest_acc_en`. | reader | writer | whole kernel | **Not shared with `dfb_pad_row`: the lifetimes overlap.** The template is live for the whole kernel and is read while data blocks stream. Capacity equals the live set in both regimes. The `entry_bytes − out_row_bytes` gap (< 16 B per entry) is L1 ring-alignment padding, and it is also what makes each implicit write stay within its page's aligned slot. |
| `dfb_pad_row` | `T` entries of `entry_bytes` (Phase 0: 1) | 1 row per writer thread, for the whole kernel. It is a constant: it does not scale with any block extent. | `{data rows: streams → none (independent of block_rows), pad rows: streams → none (one template serves every pad row; pad_rows_per_barrier writes read the same bytes), row elements: spans → whole row (entry_bytes)}` | input dtype, raw bytes. Dataflow-only. | reader | writer | whole kernel (cores with `pad_count > 0` only; still allocated on every engaged core, since one spec serves the program) | **Not aliased with `dfb_out_rows`: the lifetimes overlap** (see above). **Cannot be a slice of a `dfb_out_rows` entry:** a data entry holds input data in its middle, and pad rows need a full row of value. **Cannot be dropped when `P = 0` globally:** a DFB needs ≥ 1 entry. It costs `entry_bytes` (≤ 16 KiB for bf16 on INPUTS), so it is kept for program uniformity. |

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `elem_bytes` | 2 (bf16) or 4 (fp32) | `validate()` on `dtype` |
| `W_o` | `entry_bytes ≤ ring_byte_cap` | host refusal (`NotImplementedError`) otherwise. On INPUTS, max `W_o = 8192` → 16,384 B bf16 / 32,768 B fp32. |
| `entry_bytes` | `align_up(W_o·elem_bytes, ttnn.get_l1_alignment())` ≤ `output.buffer_aligned_page_size()` | host assert. It always holds, because the allocator alignment (16 L1 / 64 DRAM) is ≥ 16. |
| `T` | `max(R, W)`, `R + W ≤ 6`, `max % min == 0` | `PAD_THREADS` host constant, validated on the host |
| `budget` | `ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes` (3,963,328 B on quasar/ttsim) | run-time query |
| `ring_byte_cap` | `min(1,048,560, budget − T·entry_bytes)` | target page ring cap; the query above |
| `rows_fit` | `min(255, ring_byte_cap // entry_bytes) // T` | implicit-sync descriptor cap (`dataflow_buffer.cpp:1351-1360`) and the ring cap |
| `block_rows` | `rows_resident`: `= max_core_data_rows / T ≤ rows_fit`. `rows_streamed`: `= rows_fit // buffer_depth`. Override: any value ≥ 1 with `T·buffer_depth·block_rows ≤ 255`. | the regime selection function in `op_design.md` |
| `ring_blocks` | 1 (`rows_resident`) or `buffer_depth` = 2 (`rows_streamed`) | same |
| `max_core_data_rows` | `ceil(D / num_cores)`, unbounded in the shape | it only enters a capacity through `block_rows`, which the predicate caps at `rows_fit` |

## Total per-core footprint

```
L1 = T · entry_bytes · (1 + ring_blocks · block_rows)
   ≤ T · entry_bytes + min(1,048,560, budget − T·entry_bytes)      (by the rows_fit clamp)
```

How each term scales:
- Both terms scale with `W_o · elem_bytes` (the row extent, unsplit).
- The ring term also scales with `block_rows · ring_blocks` (the data-row block knob and the depth knob).
- The template term scales with nothing else.

Worked values (quasar):

| Shape | Ring term | Template term |
|-------|-----------|---------------|
| YOLO stem | 96 B | 48 B |
| `(2,4,128,512)`+N | 32,768 B | 1,024 B |
| `(1,1,2048,256)`+H | 32,768 B | 512 B |
| `(1,1,32,4096)`+W | 16,384 B | 16,384 B |

The worst case on INPUTS is 64 entries × 512 B = 32 KiB. Every INPUTS case is below 1 % of the budget, so no budget solve binds on INPUTS.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 (exactly `D · W · elem_bytes`, logical row bytes only) | each data row is read once, into the ring entry it is written from; nothing is re-read | none |
| output | 1 (`D · entry_bytes + P · W_o · elem_bytes`; the < 16 B per data row above the logical row lands in the page's own alignment slack) | data rows are assembled in L1 and written once. Pad rows are written from the per-core template, which is built by stores and never read from DRAM. | none |
| `value` (scalar) | 0 | an RT arg; the template and the margins are built in L1 | none |

Totals per tier:
- **DRAM:** `D·W·e` read, plus about `O·W_o·e` written. Here `O = D + P` output rows and `e = elem_bytes`.
- **Cross-core:** 0 B.
- **Core-local stores (first lap only):** `W_o·e` per core for the template, plus `ring_blocks·block_rows·(bw+aw)·e` per core for the margins.

> Cheapest-traffic split considered: data rows and pad rows as two lists over the full grid. It has the minimum
> crossings (input once, output once, no cross-core traffic). Implemented: the same split. The implemented split
> is the cheapest; the only tie on bytes, contiguous output-row ranges, is rejected on load balance alone.

## Implementation notes (ttnn-implementer, phase 0)

- Inventory and capacities as above; no CB added, resized, merged or deleted.
- `block_rows` in `rows_resident` is `ceil(max_core_data_rows / T)` (one op per tile counter of a STRIDED ring;
  T = 1 at the Phase 0 plan, so the worked values are unchanged).
- `dfb_pad_row` holds `T` entries; `W` of them are pushed (one per writer thread). With R > W the counters of
  readers `p >= W` stay empty, which the ring allows.
- Data-movement budget unchanged: input 1 crossing, output 1 crossing, 0 B cross-core.
