# Metal 2.0 Port Report — `experimental/quasar/reshape_view`

**Outcome: SUCCESSFUL CAPITULATION (STOP).** Both factories blocked on a Metal 2.0 expressibility gap that is **under active design**. No code changed (plan + report only). Per recipe [§When the discipline doesn't fit] and [patterns catalog — Sync-free and single-ended CBs], a grounded stop is the correct, complete deliverable here.

## Successful failure
- **Op / factories:** `reshape_view` — `ReshapeViewRMProgramFactory` and `ReshapeViewTiledProgramFactory` (both).
- **Blocking construct:** **DM single-ended producer CB** (real `reserve_back`/`push_back`, no FIFO consumer; read by raw pointer as L1 scratch). All reshape kernels are dataflow (DM) kernels:
  - RM factory — `rm_reshape_interleaved.cpp`: `cb_in0`/`cb_in1` (and `src2`/`src3` in dual-kernel mode). `reserve_back`+`get_write_ptr`+`push_back`, then the write-ptr is used as raw L1 scratch for `enhanced_noc_async_read/write` / `tt_memmove`. No `wait_front`/`pop_front`.
  - Tiled factory — `writer_reshape_tiled.cpp`: `working_cb`. `reserve_back`+`get_write_ptr`(scratch)+`push_back`, no consumer. (`mapping_cb`/`input_cb` are proper reader→writer FIFOs — fine; `working_cb` is the blocker.)
- **Why mechanical conversion fails:** Metal 2.0's `DataflowBufferSpec` validator requires ≥1 PRODUCER **and** ≥1 CONSUMER on distinct kernels. The catalog's fork for a CB lacking a usable producer/consumer pair:
  - compute kernel → self-loop (INTRA). N/A — these are DM kernels.
  - DM **sync-free** (zero FIFO ops) → fabricate a consumer on another kernel. **N/A** — these CBs DO drive FIFO producer ops (`reserve_back`/`push_back`), so they are not pure scratch; the cardinal safety rule forbids fabricating a consumer unless the CB drives *zero* FIFO sync.
  - DM **single-ended producer** (our case) → **STOP, surface to API owner.** "Under active design… likely resolution is an op-owner rewrite (a DM kernel can write its output tensor directly), but it is not finalized. We deliberately do not enshrine the [fabricate-consumer] hack."
- **Off-rules change that would have been needed (sketch):** an op-owner rewrite replacing the scratch-CB-as-L1-window idiom with either (a) a real producer/consumer split across kernels, or (b) the forthcoming Metal 2.0 kernel-scratchpad / local-`TensorAccessor` construct (not yet landed). Not a porter-scope change.

## Recipe note (friction → for the doc maintainer)
The **audit** (`port_op_to_metal2_audit.md`, DFB endpoint legality / TensorAccessor sync-free gate) classifies sync-free **and** single-ended CBs together as **FYI-P** ("port applies the interim workaround"), which reads as non-gating. But the **port recipe / patterns catalog** then forks them: a **DM single-ended producer specifically STOPs** (no workaround). So an op can audit GREEN yet capitulate at construction on exactly this CB shape. This op is that case: my audit marked `working_cb` (and, by the same logic, the RM scratch CBs) FYI-P interim-workaround, but the catalog's DM-single-ended-producer rule makes it a STOP. Suggest the audit's SPSC/sync-free subject distinguish **DM single-ended producer (GATE)** from **sync-free / compute single-ended (FYI-P)** so the gate fires at audit time rather than surfacing mid-port.

## Handoff points
- Route to the API owner / Metal 2.0 framework team: the **DM single-ended producer** end state (per catalog, under active design). `reshape_view` is parked until that lands (or an op-owner rewrite of the scratch-CB idiom).

## Open items / remaining work
- `reshape_view` is **not portable today**. Re-attempt once the DM single-ended-producer construct lands or the op is rewritten.
- Device-2.0 status unchanged (kernels already migrated). Custom `compute_program_hash` deletion was **not** applied (no port landed).
