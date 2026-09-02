# Metal 2.0 Port Compliance Gaps — `experimental/quasar/transformer/sdpa_decode`

> Where the landed port (commit `cafa17411f3`) does **not** yet meet Metal 2.0 kernel-side
> compliance. The feasibility audit (`METAL2_PREPORT_AUDIT.md`) is GREEN — the op is portable and the
> *host* side (factory, specs, bindings) is correct. The gaps below are all on the **kernel** side,
> and all trace to the same root cause: two donor helper headers were **vendored into the op but not
> converted** from the legacy `CircularBuffer` API to `DataflowBuffer`.
>
> **Recipe basis:** kernel-side whitelist rule 1 (`metal2_port.md`) — *"This transition is total.
> Post-port, no `CircularBuffer` references survive… A grep for `CircularBuffer` and `CBDescriptor`
> across the op directory at the end of the port should return zero hits in code."* The op directory
> currently returns **98** hits.

## Verdict

**PORTED BUT NON-COMPLIANT.** Two in-scope, transitively-included kernel headers are on the legacy
CircularBuffer API. Both are compiled into the instantiated kernels and exercised on the live
execution path, so this is not cosmetic dead code — it is the legacy CB idiom running in production
under new-API syntax.

| # | Severity | File | Issue |
|---|---|---|---|
| G1 | **High** | `device/kernels/compute/compute_common.hpp` | Entirely unported — 78 `CircularBuffer` sites, stale `circular_buffer.h` include, 0 `DataflowBuffer` |
| G2 | **High** | `device/kernels/dataflow/sdpa_dataflow_common.hpp` | Partially ported — 20 `CircularBuffer` sites remain, stale `circular_buffer.h` include retained |

Everything else in the op — `reader_decode_all.cpp`, `writer_decode_all.cpp`,
`sdpa_flash_decode.cpp`, `dataflow_common.hpp`, and the whole host factory — is compliant:
`CircularBuffer` = 0, correct includes, `dfb::`/`tensor::`/`sem::` handles, multi-binding + self-loop
+ borrowed-DFB + `get_bank_base_address` bridge all correctly applied.

---

## G1 — `compute/compute_common.hpp` is unported (High)

**What's there.** The file is a **byte-identical vendored copy** of the legacy sdpa-prefill
`transformer/sdpa/device/kernels/compute/compute_common.hpp` (the only diff is a 5-line
"vendored copy… **unmodified on the port branch**" comment the port itself added). It contains:

- **78 `CircularBuffer` construction/use sites** — `compute_common.hpp:56` (`CircularBuffer cb_in0(in0);`)
  through `:2072` (`CircularBuffer(alias_prev_max).pop_front(...)`).
- The forbidden include: **`compute_common.hpp:31` — `#include "api/dataflow/circular_buffer.h"`**
  (kernel-side whitelist: a port `#include`s only `experimental/kernel_args.h` and
  `api/dataflow/dataflow_buffer.h`; the `circular_buffer.h` include must drop with rule 1's sweep).
- Zero `DataflowBuffer`.

**Why it's live, not dead.** The instantiated compute kernel `sdpa_flash_decode.cpp` (factory
`.source = "compute/sdpa_flash_decode.cpp"`, `device/sdpa_decode_program_factory.cpp:1112`) includes
it (`sdpa_flash_decode.cpp:30`) and calls its helpers on the main flash-attention loop — e.g.
`reduce_c<…, dfb_qk_im, dfb_identity_scale_in, …>(…)`, `sub_exp_block<scale_fp32>(dfb_prev_max,
dfb_cur_max, dfb_exp_max_diff, …)`, `mul_block_inplace(dfb_prev_sum, dfb_exp_max_diff, …)`,
`matmul_blocks(…)`, `max_block<…>(…)`. The kernel passes `dfb::name` handles which implicitly convert
to `uint32_t` CB ids; the helper then **reconstructs a `CircularBuffer` from that id**
(`void max_block(uint32_t in0, uint32_t in1, uint32_t out_cb, …) { CircularBuffer cb_in0(in0); … }`).
So the ported kernel is a thin DFB shell over a legacy-CB compute library.

**Compliance rules violated.**
- Rule 1 (CB→DFB is total; no `CircularBuffer` survives; the `circular_buffer.h` include drops).
- The `#include` restriction (only the two sanctioned headers may be added; `circular_buffer.h` is not one).
- Rule 7 (DFB tile/format metadata via the object) — these helpers read metadata off `CircularBuffer`
  objects, which must move onto `DataflowBuffer` getters.

**Required fix.** Convert `compute_common.hpp` the same way the main kernels were converted:
`CircularBuffer` → `DataflowBuffer` (`cb_*` locals → `dfb_*`), drop the `circular_buffer.h` include,
route tile/format metadata through DFB member getters (or keep the `get_*(dfb::name)` free-function
form only where the legacy value was `constexpr`, per whitelist §A). The helper signatures currently
take `uint32_t <name>_cb`; per rule 2 the `dfb::name → uint32_t` conversion at the call sites is
sanctioned, so the signatures can keep `uint32_t` ids and only the *bodies* need converting — the
minimal-diff path. Confirm the call sites in `sdpa_flash_decode.cpp` still pass `dfb::` handles
(they do).

---

## G2 — `dataflow/sdpa_dataflow_common.hpp` is partially ported (High)

**What's there.** A vendored copy of the legacy sdpa-prefill
`transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp`. The port added exactly **one** Metal
2.0 element — an additive `read_page_table_for_batch(Noc, DataflowBuffer&, …)` overload (the 3
`DataflowBuffer` occurrences) plus a `dataflow_buffer.h` include — and left the rest legacy:

- **20 `CircularBuffer` sites remain** — `sdpa_dataflow_common.hpp:31` (`CircularBuffer cb(cb_id);`)
  through `:1733`. These are the mask-generation / fill / gather helpers
  (`fill_tile`, mask writers, etc.).
- The forbidden include is **still present**: **`sdpa_dataflow_common.hpp:15` —
  `#include "api/dataflow/circular_buffer.h"`** (the port *added* `dataflow_buffer.h` at `:16` but did
  not remove the legacy `circular_buffer.h` at `:15`, so both coexist).

**Why it's live.** `reader_decode_all.cpp` / `writer_decode_all.cpp` → `dataflow_common.hpp`
(`:18` includes `sdpa_dataflow_common.hpp`). The mask/fill helpers here are used by the mask-writer
path.

**Compliance rules violated.** Same as G1: rule 1 (residual `CircularBuffer`), the `#include`
restriction (stale `circular_buffer.h`), and rule 7 for any metadata reads.

**Required fix.** Finish the conversion the port started: convert the remaining 20 `CircularBuffer`
sites to `DataflowBuffer`, then remove the now-redundant `circular_buffer.h` include at `:15`
(`dataflow_buffer.h` at `:16` stays). The single already-ported overload is the template to follow.

---

## Not gaps (checked, clean)

- **Host factory** (`sdpa_decode_program_factory.cpp`) — no `CBDescriptor` / `CreateCircularBuffer` /
  `SetRuntimeArgs`; `DataflowBufferSpec` + typed bindings throughout; multi-binding flag,
  `borrowed_from`, self-loop, and `get_bank_base_address` Case-2 bridge all present and correct.
- **`reader_decode_all.cpp`, `writer_decode_all.cpp`, `sdpa_flash_decode.cpp`,
  `dataflow_common.hpp`** — `CircularBuffer` = 0; includes limited to `dataflow_api.h` +
  `dataflow_buffer.h` + `kernel_args.h`; no self-included generated headers; no
  `get_cb_tiles_acked_ptr` / `get_local_cb_interface` / raw `cb_*` free-function calls.
- **Transitive main-tree helpers** (`q_chunk_remapping.hpp`, `chunked_prefill_utils.hpp`,
  `sliding_window_geometry.hpp`) — `CircularBuffer` = 0. Out of the op dir and CB-free, so no fix
  needed there.

## Lower-priority / optional (post-port passes, not rule-1 violations)

These are *not* compliance failures; they are the optional post-port cleanup passes
(`post_port/style/`, `post_port/semantic/`). Listed for completeness; they should run only **after**
G1/G2 are closed and the op is green again:
- **Sync-free DFBs → `Scratchpad` / `LocalTensorAccessor`** (`style/sync_free_dfbs.md`) — candidates
  such as scratch/identity buffers, if any qualify after the CB→DFB conversion exposes them.
- **DM self-loop DFBs** (`semantic/dm_self_loop_dfbs.md`) — assess once the kernels are on DFB.

## How to verify the fix

After converting G1 and G2, the whole-op grep must come back clean:

```bash
grep -rnE "CircularBuffer|circular_buffer\.h" \
  ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/
# expect: zero hits
```

Then rebuild and run the sentinel set (WH): `tests/ttnn/unit_tests/operations/sdpa/`
`{test_sdpa_decode.py, test_paged_sdpa_decode_flexible_geometry.py, test_bounded_sliding_kv_cache.py}`
plus the nightly `sdpa/{test_sdpa_decode,_cache,_sink}.py` — green before and after (behaviour is
preserved; this is a syntax swap).
