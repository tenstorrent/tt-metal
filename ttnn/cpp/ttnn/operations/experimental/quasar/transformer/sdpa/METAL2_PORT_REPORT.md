# Metal 2.0 Port Report — `experimental/quasar/transformer/sdpa` (SDPA + JointSDPA)

## Outcome

**PORTED** (port-completion). The three compliance gaps the audit/gaps-analysis found in the
already-~85%-ported quasar SDPA fork are closed. The op now carries **zero** `CircularBuffer`
references and no legacy `circular_buffer.h` include. Kernel-only change: no host `.cpp`/`.hpp`
touched.

**Verification (WH n150, `wormhole_b0`, Watcher on = `TT_METAL_WATCHER=10`, `TT_METAL_LLK_ASSERTS=1`):**
- `ops/test_scaled_dot_product_attention.py` — **3 passed** (seq 128 / 512 / 1024).
- `ops/test_chunked_scaled_dot_product_attention.py` — **3 passed** (chunked/paged path,
  `-DIS_CHUNKED=1`, exercises the `dataflow_common.hpp` page-table area).
- Kernels JIT-recompiled on each run (confirming the edited sources are what ran); no Watcher trips,
  no `0xdeadc0de`, no JIT compile error. `prototype_ops` mirror `ops` and were not separately run;
  `graph_ops` are graph-capture (not device numerics). JointSDPA: kernels JIT-compiled clean via the
  shared headers, but **no runtime numerical test exists in the fork suite** (open item below).

Scope was the four shared kernel helper headers only; the host factories and the six top-level
kernel entry points were already on Metal 2.0 (prior commit `c1eaea9f196`).

## Provenance

- **Recipe docs (this port):** `385e3f7a90d 2026-09-02 docs(metal_2.0): the conditional-binding pattern covers tensors and semaphores too`
- **Audit docs (inherited):** `385e3f7a90d 2026-09-02` (see `METAL2_PREPORT_AUDIT.md`)

## TTNN ProgramFactory

No factory or device-op-class edits. Both factories were already realized as
`ProgramSpecFactoryConcept` (`ProgramSpec` / `DataflowBufferSpec` / `TensorParameter` /
`KernelSpec` / `SemaphoreSpec`). Custom hash: not touched. Pybind: no `create_descriptor` binding
existed to remove. This section is short because the host side needed nothing — the success case.

## What changed (the three gaps)

- **Gap 1 — CircularBuffer → DataflowBuffer (whitelist rule 1).** 176 `CircularBuffer` refs across
  the 4 helper headers swapped to `DataflowBuffer`; each header's `#include "api/dataflow/circular_buffer.h"`
  swapped to `dataflow_buffer.h`. Pure type-token swap — the only methods used (`reserve_back` /
  `push_back` / `wait_front` / `pop_front` / `get_write_ptr` / `get_read_ptr`) map 1:1, and the
  helpers' `uint32_t cb_id` params feed `DataflowBuffer(uint16_t)` directly (established kernel_lib
  pattern). Files: `compute_common.hpp` (78), `compute_streaming.hpp` (75), `dataflow_common.hpp`
  (20), `windowed_mask_gen.hpp` (3).
- **Gap 2 — `cb_push_back_hold_wr_ptr` cursor surgery (whitelist §D).**
  `compute_streaming.hpp:97`. The raw `LocalCBInterface.fifo_wr_ptr -= / +=` mutations were replaced
  by `DataflowBuffer::evil_set_write_ptr`. Reads stay on the sanctioned `get_local_cb_interface()`
  in raw 16B units; only the write is redirected. Verified byte-exact (see "Correctness reasoning").
- **Gap 3 — dead helper removed.** `read_page_table_for_batch` (`dataflow_common.hpp:77`) deleted
  (unreferenced; its one call site was already inlined in `reader_interleaved.cpp`). This removed
  the last `TensorAccessorArgs` + raw-address form in the op. The now-stale comment in
  `reader_interleaved.cpp:383` that named the deleted helper was updated to describe the rationale
  without pointing at the removed symbol.

## Correctness reasoning (why this is a true no-behavior-change on WH/BH)

- **`get_write_ptr()` semantics are identical on Gen1.** `DataflowBuffer::get_write_ptr()` returns
  `fifo_wr_ptr + L1_UNCACHED_OFFSET`, and `L1_UNCACHED_OFFSET` is `MEM_L1_UNCACHED_BASE` **only** on
  `ARCH_QUASAR && COMPILE_FOR_DM`; it is `0` on WH/BH and on all TRISC. So on the Gen1 target the
  value matches the legacy `CircularBuffer::get_write_ptr()` (raw `fifo_wr_ptr`) exactly. On Quasar
  DM it correctly returns the uncached alias for local L1 pointer deref — the Gen1/Gen2 divergence
  the audit flagged as the reason the fork needed the swap.
- **`evil_set_write_ptr` is byte-exact with the legacy field write.** It assigns
  `local_dfb_interface_.fifo_wr_ptr = addr` with no shift/offset, and the DFB's `local_dfb_interface_`
  is a **reference** bound to `get_local_cb_interface(id)` — the same object the legacy code mutated.
  Keeping the arithmetic in raw interface units (not the byte-valued `get_entry_size()` getters)
  preserves the exact computation.

## Handoff points

None. No out-of-op edits, no `sem::`/`tensor::` boundary crossing, no kernel-lib gap, no removed
pybind surface. The 6 pure-geometry headers included from main-tree `transformer/sdpa/device/kernels/`
were left untouched (out of scope; no CB idioms).

## Successes

- **CB→DFB whitelist §D (cursor surgery) + the `dataflow_buffer.h` header comments** were exactly
  what Gap 2 needed. Reading the `evil_set_write_ptr` / `get_write_ptr` / `L1_UNCACHED_OFFSET`
  definitions at the field (per "go to the headers first") settled the unit/offset question
  definitively and prevented a plausible silent-corruption error (using the byte-valued size
  getters, or assuming `get_write_ptr()` was raw on all arches).

## Friction

- **Gaps:** The recipe is written for a from-scratch, host-heavy port; it has **no shape for a
  port-*completion*** where the host and entry kernels are already Metal 2.0 and only shared kernel
  *helper* headers remain on the Device-2.0 `CircularBuffer` wrapper. Most of the recipe (spec
  construction, `hw_config`, `opt_level`, tensor bindings, legality-check forcing) was N/A. The
  kernel-side whitelist (rule 1, rule 7, §D) and the CB→DFB API whitelist carried the whole port.
- **Confusion / near-miss:** the `get_write_ptr()` `L1_UNCACHED_OFFSET` delta between `CircularBuffer`
  and `DataflowBuffer` is a real semantic difference that only collapses to a no-op because the
  offset is 0 on Gen1. A porter who swapped types without checking would ship a latent Quasar bug
  invisible on WH/BH tests. Worth a one-line note in the CB→DFB whitelist §C that the DFB peek adds
  the uncached offset on Quasar DM.

## Scope decision — `cb_*` local-variable naming deliberately NOT renamed

Whitelist rule 1 allows a *limited* `cb_* → dfb_*` variable rename "to keep the kernel readable."
The four helper headers still hold **hundreds** of kernel-internal `cb_*` locals / params /
constants (`cb_qkt_im`, `cb_mask_in`, `out_cb`, `cb_id`, `cb_exp_max_diff`, …) inherited from the
pre-fork legacy code. These were **not** renamed, deliberately:

- They are **not** one of the three compliance gaps the audit found — the rule-1 *requirement*
  (no `CircularBuffer` **type**/reference, no `circular_buffer.h`) is fully met (0 hits).
- The **dangerous** self-audit case — a `cb_`-prefixed **`DataflowBufferSpec` name** escaping to the
  generated header as `dfb::cb_*` — is **absent**: every DFB spec name is already clean
  (`dfb::q_in`, `dfb::k_in`, `dfb::attention_sink`, …). These `cb_*` are purely local.
- Renaming them is a **1000+-line, purely-cosmetic churn** through the core flash-attention compute
  loop, which would bury the actual compliance diff in review and carry real typo-bug risk across
  hundreds of sites for zero behavioral or Gen2-correctness benefit.

Recommended as a **separate readability-only cleanup PR** if desired. The `run_safe_pytest`/Watcher
baseline in this PR is the natural guard for it.

## Open items for downstream

- **Quasar-uplift debt — `cb_push_back_hold_wr_ptr` (`compute_streaming.hpp`).** Now Metal-2.0-legal
  via `evil_set_write_ptr`, but `evil_set_*` is a WH/BH-only escape hatch with no Quasar equivalent.
  This "hold-wr" scheme (push tiles visible to UNPACK, then rewind the write cursor) needs a
  credit-based or separate-intermediate-buffer rewrite before it can run on Gen2. Flagged in a
  self-contained comment at the site.
- **JointSDPA has no runtime test in the fork's suite.** Only prefill SDPA is exercised on device
  (`models/experimental/llama32_1b_quasar/tests/ops|prototype_ops`). The JointSDPA factory shares
  all four converted helper headers, so the build JIT-compiles its kernels (`joint_reader` /
  `joint_writer` / `compute/joint_sdpa`), but there is no numerical no-regression check for it. A
  direct `joint_scaled_dot_product_attention` test would close this.
- **Fork ↔ main coupling (not port work).** The fork still includes 6 headers from main-tree
  `transformer/sdpa/device/kernels/`. Pure geometry, no compliance debt, but a self-containedness
  decision for the fork owner.
