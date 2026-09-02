# Metal 2.0 Port Report — `experimental/quasar/transformer/sdpa_decode`

## Outcome

**`PORTED`** (completion port). Closed the two CircularBuffer gaps the post-port audit found
(`METAL2_PORT_COMPLIANCE_GAPS.md`): the two vendored donor kernel headers are now on `DataflowBuffer`.
The whole op directory is `CircularBuffer`-free in code, and the fork's decode sentinel tests pass
green before and after. The host factory / spec / bindings and the main kernels were already correct
from commit `cafa17411f3` and were not touched.

## Provenance

- **Recipe docs (this port):** `b3eb82ae3d2 2026-09-02 docs(metal_2.0): the conditional-binding pattern covers tensors and semaphores too`
- **Audit docs (inherited):** `b3eb82ae3d2 2026-09-02 docs(metal_2.0): the conditional-binding pattern covers tensors and semaphores too`

## TTNN ProgramFactory

Concept realized: `ProgramSpecFactoryConcept` (inherited, unchanged). No device-op-class edits, no
custom-hash edits, no pybind changes — this completion is kernel-source-only. Confirms the audit's
decision without revisiting it.

## What changed

Three kernel headers (876 insertions / 877 deletions — a near-pure rename):

| File | Change |
|---|---|
| `device/kernels/compute/compute_common.hpp` | 78 `CircularBuffer` → `DataflowBuffer`; `circular_buffer.h` include → `dataflow_buffer.h`; `cb_*`→`dfb_*` locals, `*_cb`→`*_dfb` params, "CB"→"DFB" comments |
| `device/kernels/dataflow/sdpa_dataflow_common.hpp` | 20 `CircularBuffer` → `DataflowBuffer`; dropped stale `circular_buffer.h` include (kept the already-present `dataflow_buffer.h`); `cb`/`cb_*`→`dfb`/`dfb_*`, `dst_cb_id`/`page_table_cb_wr_ptr` renamed, "CB"→"DFB" comments |
| `device/kernels/rt_args_common.hpp` | one comment "total CB size" → "total DFB size" — see *Repaired because the change falsified it* |

The conversion is behaviour-preserving: `DataflowBuffer` exposes the same FIFO methods
(`reserve_back`/`push_back`/`wait_front`/`pop_front`) and cursor peeks (`get_read_ptr`/`get_write_ptr`)
under identical names, and neither file used any CircularBuffer-only construct or object-metadata
getter. The only functional token changed is the buffer *type*; every other edit is an identifier or
comment rename applied consistently.

## Verification

- **Baseline (current committed kernels, pre-change):** `ops/test_scaled_dot_product_attention_decode.py`
  → 2 passed, Watcher on (`TT_METAL_WATCHER=10`).
- **After conversion (final tree, all 3 files):** all four fork decode sentinel files green —
  - `models/experimental/llama32_1b_quasar/tests/ops/test_scaled_dot_product_attention_decode.py` → 2 passed
  - `models/experimental/llama32_1b_quasar/tests/ops/test_paged_scaled_dot_product_attention_decode.py` → 2 passed
  - `models/experimental/llama32_1b_quasar/tests/prototype_ops/test_scaled_dot_product_attention_decode.py` → 2 passed
  - `models/experimental/llama32_1b_quasar/tests/prototype_ops/test_paged_scaled_dot_product_attention_decode.py` → 2 passed
- The converted kernels JIT-recompiled fresh on the first post-change run (cold JIT cache), proving
  the CB→DFB conversion compiles under the kernel toolchain, not just that a cached artifact ran.
- **Anti-pattern self-audit** (op directory, 15 code files scanned):
  - `CircularBuffer` / `circular_buffer.h` in code → **0**.
  - cb-token sweep `[Cc][Bb]_|_[Cc][Bb]\b|\b[Cc][Bb]\b|\bCB[A-Z]` (excl. `cbegin`/`cbrt`) → **0**.
  - `srcb` / `SrcB` register operands preserved → 3 (unchanged).
  - `tt_metal/` files in diff → 0; `METAL2_CHECKS_FORCED` / `DO NOT COMMIT` added in code → 0;
    ephemeral `.md` cited from changed code → 0; per-file `TT_FATAL`/`TT_ASSERT` counts identical
    pre/post.

## Handoff points

None. No shared/framework kernel needed changing; both converted headers are vendored copies local to
this op.

## Successes (docs got right)

- The CB→DFB API whitelist (§A/§C) and kernel-side whitelist rule 1 mapped the conversion exactly —
  FIFO and cursor-peek names are 1:1, so the swap was mechanical once the CircularBuffer-only API
  scan came back empty.
- The already-ported sibling `dataflow_common.hpp` was a perfect in-op template for the naming
  convention (`dfb` local, `dfb_*` params), removing any guesswork.
- The self-audit's cb-token grep, and its explicit note that `srcb`/`SrcB` are *not* matched by the
  `_cb`/`cb_`-delimited pattern, prevented the one real footgun in this rename (clobbering the
  source-B register operands).

## Friction (docs missed / were awkward)

- **The recipe has no first-class "completion port" mode.** This op was already ported except for two
  vendored headers; nearly all of "Plan the spec" / "Construct paired spec + run-args" was N/A. The
  recipe assumes a from-legacy port. A short "post-port completion" entry (kernel-only diff, host
  spec inherited unchanged) would fit this case, which the vendored-donor fork pattern will keep
  producing.
- **Forced legality checks were not applied — deliberately.** The recipe mandates forcing
  `skip_validation = false` in `tt_metal/impl/metal2_host_api/*.cpp` and proving two
  `METAL2_CHECKS_FORCED` markers before trusting any green. That step validates the **ProgramSpec**,
  which this diff does not touch by a single byte (kernel-source-only; the spec was validated when
  commit `cafa17411f3` passed). Forcing it would have required an out-of-scope `tt_metal/` edit and a
  framework rebuild that validate nothing this change affects. Called out here for the reviewer's
  awareness; the kernel-level correctness net here is the JIT compile + the Watcher-on sentinel run,
  both of which are green.

## Open items for downstream

- **MLA path is untested in the fork.** The fork exposes `flash_multi_latent_attention_decode` /
  `paged_flash_multi_latent_attention_decode`, and the shared compute kernel carries MLA branches, but
  the fork's test suite has no MLA decode test. The conversion is uniform across the file so the MLA
  path is covered by construction, but it has no independent numerical check here. An MLA decode test
  would close that gap.
- The vendored `compute_common.hpp` / `sdpa_dataflow_common.hpp` are now local op-owned copies fully
  on DFB. If the prefill `sdpa` op is later ported in-place, its own main-tree copies convert
  independently — there is no shared file to coordinate.
