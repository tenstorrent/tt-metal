# L1 Ledger: slice

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` → Blocking Model: `out_page` (the block axis, extent `block_pages`), and the folded source axes `a0`, `a1`, `h`, `w`. Worked values are for `quasar` (8 × 4 grid, unreserved budget 3,963,328 B).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_slice_pages` | `RING_DEPTH_BLOCKS × block_pages` (Phase 0: `2 × block_pages`; `block_pages = round_up(max(1, min(ceil(max_core_pages / RING_DEPTH_BLOCKS), block_pages_cap)), max(R, W))`) | `RING_DEPTH_BLOCKS × block_pages` pages: one block being filled by the reader while the previous one drains through the writer. Equals capacity: the second block is the double buffer, which is the only overlap a relay has. Capacity never exceeds the core's assignment rounded up to whole blocks (`block_pages ≤ ceil(max_core_pages / 2)`), so no slot is allocated that no page passes through, except on a 1-page core (ring of 2, 1 used — the txn-batch divisibility needs an even ring) | `{out_page: spans → RING_DEPTH_BLOCKS × block_pages, a0: streams (folded into out_page), a1: streams (folded into out_page), h: streams (folded into out_page), w: streams (folded into out_page; RM: inside the page)}` | the input dtype as metadata (bf16 in Phase 0), `entry_size = input.buffer_aligned_page_size()`; raw bytes moved by the NoC only — never unpacked, never through DEST, so audit 2 (page format vs DEST width) does not apply: the payload is dataflow-only | reader | writer | whole program | Shares with nothing — it is the only buffer of the op. A relay needs one staging buffer (a core reads only into its own L1 and writes only out of it); no second buffer exists to alias, pack into or transform in place |

**Implementation note (implementer, Phase 0).** Inventory unchanged: one ring. Its *protocol* depends on
`entry_bytes`: implicit sync for pages up to one 8 KiB NoC packet (`IMPLICIT_SYNC_MAX_ENTRY_BYTES`, the
overlay's `NOC_OVERLAY_MAX_BYTES_IN_PACKET`), explicit sync with block-batched reserve/push and wait/pop
(one barrier per block per side) above it, because a multi-packet implicit-sync entry miscounts on the
DM0 interrupt path (measured: `tile counter underflowed` at 8,256 B and above once a block holds 2 entries).
Capacity, block size and traffic are the same either way.

## Symbol table

| Symbol | Bound | Predicate / source establishing it |
|--------|-------|-------------------------------------|
| `entry_bytes` | `≤ NOC_MAX_BURST_BYTES = 65,536` | validate refuses larger pages (mechanism cap; `tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:394-396`). Phase 0 worked range: 128 B (RM `W = 47`, DRAM) … 16,832 B (RM `W = 8400`, DRAM); 2,048 B per bf16 tile |
| `max_core_pages` | `= ceil(num_out_pages / num_cores_engaged)` (`per_core_1` of `ttnn.split_work_to_cores`) | the split; `num_out_pages = prod(out_page_shape) = output.buffer_num_pages()` |
| `ring_cap_bytes` | `= min(DFB_RING_MAX_BYTES, ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES)` = `min(1,048,560, 3,963,328 − 0)` = 1,048,560 on this target | `DFB_RING_MAX_BYTES` from the target page's cluster model; the budget is the run-time query, never a literal |
| `block_pages_cap` | `= floor(ring_cap_bytes / (RING_DEPTH_BLOCKS × entry_bytes))` ≥ 7 for any page under the burst cap (1,048,560 / (2 × 65,536) = 7.99) | derived; validate refuses `block_pages_cap < 1` (unreachable under the burst cap) |
| `RING_DEPTH_BLOCKS` | 2 (Phase 0) | host constant |
| `R`, `W` | 1, 1 (Phase 0, `DM_THREAD_PLAN`); `R + W ≤ 6`, `R % W == 0 or W % R == 0` | target page thread plan; DFB guide STRIDED rules |
| `L1_RESERVE_BYTES` | 0 (Phase 0) | host constant (the ring is the op's only L1 allocation besides the tensors the allocator places) |

**Total per-core footprint:** `RING_DEPTH_BLOCKS × block_pages × entry_bytes ≤ ring_cap_bytes`. Scales with `block_pages` (block knob), `RING_DEPTH_BLOCKS` (depth knob) and `entry_bytes` (dtype × W for RM, dtype for TILE); independent of the tensor's size beyond `max_core_pages`, which `block_pages` clamps. Phase 0 worked values: 768 B (`[4,128,47]` RM) … 67,328 B (`[1,144,8400]` rows 64–144, RM) — at most 1.7 % of the budget.

## Data-movement budget

Tier = the input's buffer (DRAM, or L1 banks for an L1 input); the output is in the same tier (R7).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 crossing of the block's pages only (`num_out_pages × entry_bytes`); pages outside the block: 0 | each output page has exactly one source page, read once by the one core that owns that output page; nothing is re-read because no operand is shared and no footprint overlaps | none |
| `output` | 1 (`num_out_pages × entry_bytes`) | each output page written once by its owning core | none |

**Totals per tier:** read `num_out_pages × entry_bytes` + write `num_out_pages × entry_bytes` in the input's tier; 0 cross-core (no combine, no multicast). Above the logical minimum (`2 × output elements × elem_bytes`) only by RM row alignment pads (< 64 B per row in DRAM, < 16 B in L1) and by TILE padding inside the output's edge tiles, which the output must hold anyway. Example, `[1,144,8400]` RM DRAM rows 64–144: 80 × 16,832 = 1,346,560 B read + 1,346,560 B written.

> Cheapest-traffic split considered: contiguous `out_page` ranges over the full grid — every candidate split (by plane, by page row, by tile-column) moves exactly the same bytes, since no operand is shared and there is no dependent axis to combine. Implemented: contiguous `out_page` ranges, which is the cheapest (tied) and the only one that fills the grid on every INPUTS shape. Nothing deferred on traffic grounds.

## Verifier audit (Phase 0)

- **Currency**: one DFB declared (`slice_pages`, `entry_size = input.buffer_aligned_page_size()`, `num_entries = RING_DEPTH_BLOCKS × block_pages`), one row here; expressions match `slice_program_artifacts.py`. The explicit-sync path adds no buffer; its per-hart batch `thread_block_pages = block_pages / max(R, W)` is derived from the same `block_pages` (single source).
- **Capacity vs live set**: equal by construction. Over-direction finding, not fixed in place: `block_pages = ceil(max_core_pages / 2)` makes the ring span the busiest core's whole share (up to the 1 MiB cap), so the live set is chosen to be the core's share, not a fixed streaming window; on the implicit path the ISR credits half the ring per batch, so the writer starts only after half the core's pages have landed. Bounded (≤ 1 MiB, ≤ 1.7 % of the budget on INPUTS), so no OOM risk; folded into the first perf refinement (op_requirements.md Refinement 3: cap `block_pages` independently of the share). Under direction: none (one axis, spanned by a capacity that scales with it).
- **Page format vs DEST**: n/a — payload is dataflow-only, never through DEST (confirmed: no compute kernel).
- **Disjoint lifetimes**: n/a — one buffer.
- **Bounds**: every symbol bounded (`entry_bytes ≤ 65,536` by validate, `block_pages ≤ block_pages_cap`), total closed-form.
- **Traffic budget**: consistent with the implemented split (read once, write once, no cross-core); cheapest split implemented (all candidate splits tie on bytes).
