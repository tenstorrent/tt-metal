# Metal 2.0 Port Brief — `experimental/quasar/transformer/sdpa`

> Audit cleared all feasibility gates. **But this op is already ported** (fork
> `c1eaea9f196`), and the existing port is **not yet fully compliant**. So this brief is
> not a from-scratch work order — it is the reference for *what a complete port looks like*,
> to be read alongside **`METAL2_PORTING_STATE_GAPS.md`**, which lists exactly what remains.
> The full gate record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `385e3f7a90d 2026-09-02 docs(metal_2.0): the conditional-binding pattern covers tensors and semaphores too`

## TTNN factory analysis

The op targets — and already sits on — `ProgramSpecFactoryConcept`.

- **Current concept:** `MetalV2` (host factory already ported: `ProgramSpec` / `KernelSpec` /
  `DataflowBufferSpec` / `TensorParameter` / `SemaphoreSpec`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` — achieved on the host side.
- **Gate-cleared, confirmed absent:** non-`none` `TensorParameter relaxation` · `get_dynamic_runtime_args`.

## Construct — status

**Tensor bindings** (per binding) — all **Case 1** (via `TensorAccessor` / borrowed-DFB),
bound as `TensorParameter` host-side, kernel builds `TensorAccessor(tensor::name)`. **Done.**

- `q_in`, `k_in`, `v_in`, `out` — required.
- `mask_in`, `page_table`, `attention_sink`, `cu_window_seqlens`, `windowed_q_token_offset` — conditional.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** page-table page-size override — **dropped** in the ported reader
(`reader_interleaved.cpp:388`). *(The dead helper `read_page_table_for_batch` still carries
the legacy 3-arg form — drop candidate; see gaps file.)*

**CB endpoints:** self-loop + 1P+1C + conditional DFBs — **expressed** in the host
`DataflowBufferSpec` group. **Done.**

## Remaining work — the compliance gap (see `METAL2_PORTING_STATE_GAPS.md`)

The kernel-side CB→DFB transition (**kernel-side whitelist rule 1**) is **incomplete**. It was
applied to the six top-level kernel entry points (0 `CircularBuffer`), but **not** to the four
shared kernel helper headers, which remain on the Device-2.0 `CircularBuffer` wrapper:

| Header | `CircularBuffer` refs | `circular_buffer.h` include |
|---|---|---|
| `device/kernels/compute/compute_common.hpp` | 78 | yes |
| `device/kernels/compute/compute_streaming.hpp` | 75 | yes |
| `device/kernels/dataflow/dataflow_common.hpp` | 20 | yes |
| `device/kernels/dataflow/windowed_mask_gen.hpp` | 3 | yes |
| **total** | **176** | 4 |

Per rule 1, a complete port leaves **zero** `CircularBuffer` references in code and swaps
`api/dataflow/circular_buffer.h` → `api/dataflow/dataflow_buffer.h`. These helpers take a
`uint32_t cb_id` (fed the `dfb::name → uint32_t` shim by the ported callers) and wrap it in
`CircularBuffer cb(cb_id)`; they should construct `DataflowBuffer` instead, and route
tile/format metadata through the DFB object getters (rule 7). On WH/BH the two are synonyms
(why tests pass); on the fork's Gen2/Quasar target they diverge and the `uint32_t`-token shim
is Gen1-only — so this is real debt for the fork's purpose.

## Watch for

- **It is not only the type swap.** Alongside the 176 `CircularBuffer` refs (Gap 1), the helpers
  carry a **§D cursor-surgery** site — `cb_push_back_hold_wr_ptr` (`compute_streaming.hpp:97`)
  mutates `LocalCBInterface.fifo_wr_ptr` directly (`-=` / `+=`). That must become
  `DataflowBuffer::evil_set_write_ptr` (snapshot with `get_write_ptr()`), **not** a plain swap —
  and `evil_set_*` is Gen1-only, so flag it as Quasar-uplift debt. See Gaps file, Gap 2.
- **Shared helpers, not top-level kernels, are the work.** The entry-point kernels are done;
  do not re-touch them. Convert the four helper headers in place (they are the fork's own
  vendored copies — no cross-op fork coordination needed, unlike a main-tree shared kernel).
- **`constexpr` metadata trap (rule 7 / whitelist §A).** Several helpers read tile/format
  metadata. A value the legacy line declared `constexpr` keeps the free-function form with the
  token (`get_tile_size(dfb::in)`); a runtime value moves to the DFB getter (`dfb.get_tile_size()`).
  Do **not** demote `constexpr`→`const` to make a getter fit.
- **Cross-tree includes (fork hygiene, not compliance):** 6 pure-geometry headers are pulled
  from main-tree `transformer/sdpa/device/kernels/`. They are Metal-2.0-agnostic (no CB), so
  they need no port work, but they do couple the fork to main.
- **Dead code:** `read_page_table_for_batch` (`dataflow_common.hpp:77`) is unreferenced — drop it.
