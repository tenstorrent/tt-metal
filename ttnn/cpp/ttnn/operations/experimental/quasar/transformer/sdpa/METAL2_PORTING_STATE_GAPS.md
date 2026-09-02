# Metal 2.0 Port-State Compliance Gaps — `experimental/quasar/transformer/sdpa`

> Companion to `METAL2_PREPORT_AUDIT.md`. The audit answers *"was this op portable?"*
> (yes — GREEN). **This file answers the question you actually asked:** *where is the
> existing port (fork `c1eaea9f196`) not yet Metal 2.0 compliant?* Findings are measured
> against the **port recipe's kernel-side whitelist** (`ai/port/metal2_port.md`) and the
> **CB→DFB API whitelist** (`ai/shared/cb_dfb_api_whitelist.md`), not against the
> feasibility gates.
>
> **Recipe docs:** `385e3f7a90d 2026-09-02`

## TL;DR

The port is **~85% done and structurally correct**. The work left is confined to the **four
shared kernel helper headers**, and it is **not all one kind of change** — there are three
distinct items (Gaps 1–3 below). Everything else was checked against the whitelist and is clean
(see "Verified clean" — the list is bounded, not open-ended).

- ✅ **Host factory** — fully Metal 2.0: `ProgramSpec` / `KernelSpec` / `DataflowBufferSpec` /
  `TensorParameter` / `SemaphoreSpec`. No `CBDescriptor`, no address RTAs, clean Case-1 bindings.
- ✅ **Six top-level kernel entry points** — fully ported: `reader_interleaved`, `writer_interleaved`,
  `joint_reader`, `joint_writer`, `compute/sdpa`, `compute/joint_sdpa` — all on `dfb::` / `tensor::` /
  `sem::` / `get_arg(args::…)`, **0** `CircularBuffer`. Metadata reads correctly use the
  `get_tile_size(dfb::name)` **constexpr token form** (rule 7 exception) — compliant.
- ❌ **Gap 1 — the bulk:** four helper headers still on the Device-2.0 `CircularBuffer` wrapper
  (**176 references** + `#include "api/dataflow/circular_buffer.h"`). Whitelist **rule 1**.
- ❌ **Gap 2 — a *different* rule, not a type swap:** `cb_push_back_hold_wr_ptr`
  (`compute_streaming.hpp:97`) mutates `LocalCBInterface` FIFO cursor fields directly
  (`intf.fifo_wr_ptr -= … / += …`). Whitelist **§D** requires `evil_set_write_ptr`, not a plain
  `CircularBuffer`→`DataflowBuffer` swap. **Live** (called at `:1476`, streaming-v2 path from
  `sdpa.cpp:142`).
- ❌ **Gap 3 — dead code carrying legacy idioms:** `read_page_table_for_batch`
  (`dataflow_common.hpp:77`) — unreferenced, drop it.

All three live in the same four unported headers, so **one PR still closes them** — but whoever
does it must treat Gap 2 as §D cursor surgery, not fold it into the bulk `sed`.

It passes tests because on WH/BH the Device-2.0 `CircularBuffer` wrapper and the Metal-2.0
`DataflowBuffer` are functional synonyms. They **diverge on Gen2/Quasar** — the fork's entire
reason to exist — so these gaps defeat the fork's stated purpose if left unclosed. (Gap 2's
`evil_set_*` is itself Gen1-only — unavailable on Quasar — so even after conversion it is flagged
Quasar-uplift debt, i.e. this "hold-wr" trick needs a real refactor before Quasar, not just an API rename.)

---

## Gap 1 (primary) — shared helper headers left on `CircularBuffer` (whitelist rule 1)

**What the whitelist requires.** Kernel-side whitelist rule 1: `CircularBuffer` → `DataflowBuffer`,
1:1 method mapping; `#include "api/dataflow/circular_buffer.h"` → `api/dataflow/dataflow_buffer.h`.
"This transition is **total**. Post-port, *no* `CircularBuffer` references survive."

**What the port did.** The transition was applied to the callers (top-level kernels) but the
callee helper headers were left on the wrapper. The callers pass the `dfb::name → uint32_t`
implicit conversion (a Gen1-only shim, rule 2) into `uint32_t cb_id` helper params, and each
helper reconstructs a Device-2.0 `CircularBuffer` from it:

```cpp
// device/kernels/dataflow/dataflow_common.hpp:30
inline void fill_zeros_async(const Noc& noc, uint32_t cb_id, uint32_t tile_bytes, uint32_t offset_bytes = 0) {
    CircularBuffer cb(cb_id);                    // ❌ rule 1: should be DataflowBuffer dfb(cb_id);
    noc.async_write_zeros(cb, tile_bytes, {.offset_bytes = offset_bytes});
}
```

```cpp
// device/kernels/compute/compute_streaming.hpp:85
CircularBuffer(cb_id).push_back(num_tiles);      // ❌ should be DataflowBuffer(cb_id).push_back(...)
```

**Why rule 2's shim does not sanction this.** The `dfb::name → uint32_t` conversion exists to
bridge into call sites "that aren't on Metal 2.0 — LLK primitives and shared/other-op helpers
(escapes) that can't all port at once." These helpers are **the fork's own vendored copies inside
the op directory** — they *can* be ported, so using the shim to reach them is the incomplete-port
smell, not a sanctioned escape. Rule 1 is unconditional for in-directory code.

**Surface — 176 references, 4 headers, 4 legacy includes:**

| File | `CircularBuffer` refs | `circular_buffer.h` include | Role |
|---|---|---|---|
| `device/kernels/compute/compute_common.hpp` | 78 | line 32 | compute helpers (flash-attn reduce/exp/matmul steps) |
| `device/kernels/compute/compute_streaming.hpp` | 75 | line 29 | streaming QK·T / QK·V compute loop |
| `device/kernels/dataflow/dataflow_common.hpp` | 20 | line 16 | reader/writer DM helpers (fill-zeros, tile copy, page-table) |
| `device/kernels/dataflow/windowed_mask_gen.hpp` | 3 | line 19 | windowed mask generation |
| **total** | **176** | 4 | |

*(Regenerate the exact line list with:
`grep -rn "CircularBuffer" device/kernels/` from the op root.)*

**Fix (per the whitelist, no behavior change on WH/BH):**
1. In each of the 4 headers, replace `#include "api/dataflow/circular_buffer.h"` with
   `#include "api/dataflow/dataflow_buffer.h"`.
2. Replace every `CircularBuffer cb(id)` / `CircularBuffer(id).method()` with `DataflowBuffer`
   (the canonical FIFO methods `reserve_back`/`push_back`/`wait_front`/`pop_front` map 1:1).
3. Rename locals `cb_*` → `dfb_*` where it aids readability (rule 1's limited rename allowance) —
   but note several helpers take a bare `uint32_t cb_id` param; either keep the param name or
   rename to `dfb_id`, consistently.
4. Any tile/format metadata these helpers read by cb-id free function (`get_tile_size(cb_id)`,
   etc.) moves onto the DFB object getter (rule 7 / whitelist §A) — **except** a value the
   legacy line declared `constexpr`, which keeps the free-function form with the token. Do not
   demote `constexpr`→`const`.
5. Confirm `DataflowBuffer` is constructible from a plain `uint32_t` id at these call sites
   (the callers already produce a `uint32_t` via the token shim). If a helper is better served
   taking a `DataflowBuffer&` / `DataflowBuffer` directly, that is the cleaner end-state, but a
   `uint32_t`-id constructor keeps the diff minimal.

**Verification target:** `grep -rn "CircularBuffer" device/kernels/` returns **0** hits, and
`grep -rn "circular_buffer.h" device/kernels/` returns **0** hits. Then rebuild + rerun the
fork's tests (they should be byte-for-byte unchanged on WH/BH).

---

## Gap 2 — raw `LocalCBInterface` cursor mutation (whitelist §D, not a type swap)

**Where.** `cb_push_back_hold_wr_ptr` (`device/kernels/compute/compute_streaming.hpp:97`):

```cpp
ALWI void cb_push_back_hold_wr_ptr(uint32_t cb_id, uint32_t num_tiles) {
    CircularBuffer(cb_id).push_back(num_tiles);          // ← Gap 1 (type swap)
    PACK(({
        auto& intf = get_local_cb_interface(cb_id);
        intf.fifo_wr_ptr -= num_tiles * intf.fifo_page_size;   // ❌ raw FIFO cursor field write
        uint32_t fifo_start = intf.fifo_limit - intf.fifo_size;
        if (intf.fifo_wr_ptr < fifo_start) {
            intf.fifo_wr_ptr += intf.fifo_size;                // ❌ raw FIFO cursor field write
        }
    }));
}
```

**Why it is a separate gap.** This is a **hold-wr** cursor rewind — it makes each row visible to
UNPACK without advancing the write pointer (see the doc comment at `:1318`). The whitelist is
explicit that this is *not* covered by the `CircularBuffer`→`DataflowBuffer` type swap:

- Kernel-side whitelist rule 1 (cursor-surgery note): "A kernel that mutates FIFO pointers
  directly — `get_local_cb_interface(cb).fifo_wr_ptr = …`, rewind / jump / **hold-wr** — maps to
  the DFB `evil_set_*` setters (whitelist §D)… **never** leave a raw `LocalCBInterface` field
  write in place."
- CB→DFB whitelist §D: `fifo_wr_ptr` assign / `+=` / wrap → `evil_set_write_ptr`; "Kernel keeps
  arithmetic; setter takes absolute cursor — **never** leave `LocalCBInterface` field writes."

So the getter `get_local_cb_interface(cb_id)` itself is sanctioned (it is one of the two
Device-2.0-blessed cb-id free functions), but **writing its `.fifo_wr_ptr` field is not**.

**Live, not dead.** Called at `compute_streaming.hpp:1476` (`cb_push_back_hold_wr_ptr(cb_qkt_im, row_tiles)`)
inside the streaming-v2 QK·T path, reached from `compute/sdpa.cpp:142`.

**Fix (whitelist §D).** After the helper's CB is a `DataflowBuffer dfb(cb_id)`:

```cpp
uint32_t wr = dfb.get_write_ptr();                 // public peek to snapshot
wr -= num_tiles * dfb.get_entry_size();            // keep the existing arithmetic
uint32_t fifo_start = /* limit - size, via §B size getters */;
if (wr < fifo_start) wr += /* fifo_size, §B getter */;
dfb.evil_set_write_ptr(wr);                        // the only mutate
```

Pull `fifo_page_size` / `fifo_limit` / `fifo_size` from the §B size getters
(`get_entry_size()` / span / total-size APIs) rather than the raw interface fields.

**Quasar caveat — flag it.** `evil_set_write_ptr` is **Gen1-only** (the CB→DFB whitelist §D row
says "Not available on Quasar"). So converting it to `evil_set_*` makes it Metal-2.0-*legal* but
leaves it as **Quasar-uplift debt**: this hold-wr trick will need a genuine refactor (credit-based
or a different intermediate-CB scheme) before the fork can run on its Gen2 target. Record it in the
port report as such — don't let the `evil_set_*` rename read as "done for Quasar."

---

## Gap 3 (minor) — dead helper carrying legacy idioms

`read_page_table_for_batch` (`device/kernels/dataflow/dataflow_common.hpp:77`) still takes a
`TensorAccessorArgs` + raw `page_table_addr` + a page-size 3rd arg and wraps a `CircularBuffer`.
It is **unreferenced** in the fork — its single call site was inlined at
`reader_interleaved.cpp:383` precisely because "a `tensor::` binding token cannot cross into that
shared header." So it is dead code that also carries three separate legacy idioms (raw-address
arg, `TensorAccessorArgs`, `CircularBuffer`).

**Fix:** delete `read_page_table_for_batch`. (If a future caller needs it, it should be rewritten
against a `tensor::` binding + `DataflowBuffer`, like the inlined version already is.) Rolls into
Gap 1's `CircularBuffer`-count reduction.

---

## Verified clean (checked against the whitelist — so the gap list is bounded)

Each of these was scanned; none is a gap. This is what lets me say Gaps 1–3 are the *whole* list,
not "the ones I happened to notice":

- **Named arguments (rule 4):** complete. `grep` for `get_compile_time_arg_val` / `get_arg_val` /
  `get_common_arg_val` across `device/kernels/` returns **0** — args are all `get_arg(args::…)`.
- **Varargs (rule 4):** none retained — no `get_vararg` / `get_common_vararg` /
  `get_compile_time_vararg`.
- **Metadata via object / token (rule 7):** the ported kernels read tile size as
  `get_tile_size(dfb::name)` — the sanctioned **constexpr token form** (whitelist §A exception),
  not a raw `uint32_t`. Compliant. (The helper-internal metadata reads move onto the DFB object as
  part of Gap 1.)
- **Raw pointers in args (rule 5):** none. All tensors flow through `TensorParameter`; no
  `buffer()->address()` in args; no `get_bank_base_address` in a compute kernel (no blocked
  Case-2 compute pattern).
- **`TensorAccessorArgs` (rule 3):** no live construction — only in comments and the dead helper
  (Gap 3). The one 3rd-arg page-size site was correctly dropped in the ported reader.
- **Unportable CB APIs:** no `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`.
- **Generated headers:** kernels do not wrongly `#include` `kernel_args_generated.h` /
  `kernel_bindings_generated.h`.
- **Raw `noc_async_*` transfers:** the 2 `noc_async_` hits in the reader are **comments**, not
  calls — transfers use the Device-2.0 `Noc` API (`noc.async_read/write`). (Even a leftover
  `noc_async_*` transfer would be allowed by the recipe as minimal-diff; there are none anyway.)
- **Host imperative remnants:** none — no `SetRuntimeArgs` / `CreateSemaphore` /
  `CreateCircularBuffer` / raw `Program&` / `CreateKernel`. Fully declarative `ProgramSpec`.
- **Appendix A features:** none (GlobalCircularBuffer / GlobalSemaphore / `address_offset`).
- **`.id` extraction to keep legacy form (rule 7):** none.

**Not a gap, but worth recording — cross-tree includes (fork hygiene).** The fork `#include`s 6
headers directly from `transformer/sdpa/device/kernels/` (`windowed_loop_geometry.hpp`,
`q_chunk_remapping.hpp`, `dataflow/chunked_prefill_utils.hpp`, `sdpa_streaming_qktv.hpp`,
`sliding_window_geometry.hpp`, `sliding_window_work_plan.hpp`). All are **pure geometry /
work-plan headers with 0 `CircularBuffer` / legacy-arg idioms** → no compliance debt, but they
**couple the fork to main**. Deliberate vendor-or-share decision, not a Metal 2.0 blocker.

---

## Bottom line

**No — it is not only `CircularBuffer`→`DataflowBuffer`.** One PR still closes everything, but it
carries three distinct changes, and they are not interchangeable:

1. **Gap 1 (bulk):** convert the four helper headers `CircularBuffer` → `DataflowBuffer` +
   `circular_buffer.h` → `dataflow_buffer.h` (176 refs). Mostly mechanical.
2. **Gap 2 (delicate):** rewrite `cb_push_back_hold_wr_ptr`'s raw `LocalCBInterface` cursor writes
   as `evil_set_write_ptr` (whitelist §D). **Not** a `sed`; flag as Quasar-uplift debt.
3. **Gap 3 (trivial):** delete the dead `read_page_table_for_batch` helper.

No host changes, no top-level kernel changes, and — on WH/BH — no behavioral change. Verify with
`grep -rn "CircularBuffer\|circular_buffer.h\|fifo_wr_ptr" device/kernels/` → **0**. The payoff is
Gen2/Quasar readiness (Gap 2 remains a real refactor for that target, even once it is Metal-2.0-legal).
