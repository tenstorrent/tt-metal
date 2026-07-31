# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard`

## Outcome

**`PORTED`** — all 8 program-factory variants (5 factory types) of `ReshardDeviceOperation`
converted from the `descriptor` concept to `MetalV2FactoryConcept`, together with all 9 kernel
sources they bind. Build and test execution were explicitly deferred to the invoker, so the
port is **not yet build- or test-verified**; the recommended verification set is at the bottom
of this report.

| Factory | Status |
|---|---|
| `ReshardSameWidthFactory<true>` / `<false>` | ported |
| `ReshardSameHeightFactory<true>` / `<false>` | ported |
| `ReshardGenericFactory` | ported (both runtime-selected kernel sources) |
| `NdReshardCopyPagesFactory` | ported |
| `NdReshardCopyLocalShardFactory<true>` / `<false>` | ported |

No factory was left on the legacy concept, so `AllFactoriesValid` sees a uniform
`MetalV2FactoryConcept` variant.

## Review round

PR [#51495](https://github.com/tenstorrent/tt-metal/pull/51495) — Copilot raised 4 comments
(1 🔴, 3 🟡). All 4 accepted and addressed; none required a scope decision to be escalated.

| # | Sev | Finding | Disposition |
|---|---|---|---|
| 1 | 🔴 | Borrowed output DFB can exceed the tensor's packed size and be rejected at spec time | **Fixed, and widened** — the same defect was latent in the same-width and same-height borrowed DFBs, which the review did not flag; all three are clamped. Root cause recorded as a framework gap under *Handoff points*. |
| 2 | 🟡 | Live `DPRINT` carried into a production kernel | **Fixed** — removed, with the now-dead debug include. Overrides the port's initial scope-discipline call; see the DPRINT entry under *Open items for downstream*. |
| 3 | 🟡 | `METAL2_PORT_BRIEF.md` still asserts the stale Quasar co-borrower claim | **Fixed by annotation, not rewrite** — see below. |
| 4 | 🟡 | `METAL2_PREPORT_AUDIT.md` carries the same stale claim | **Fixed by annotation, not rewrite** — see below. |

On 3 and 4: the brief and the pre-port audit are the **audit's** artifacts and the historical record
of what the audit concluded, and this port is downstream of them — silently rewriting their findings
would falsify that record and detach them from the audit's own status tables and provenance line.
The reviewer's actual concern is that stale *actionable* guidance can wrongly expand a future port's
scope, which a correction note answers just as well. Each file therefore keeps its original text and
gains a dated `⚠ CORRECTION (port, 2026-07-31)` block at the stale bullet, stating what is wrong,
how it was verified, and where the evidence lives. The audit's note also carries a process
suggestion for the readiness-sheet owner (derive the co-borrower list from a grep at audit time,
since a sibling op forking its kernels silently invalidates the sheet).

Finding 1 is the one worth dwelling on: it was a **latent launch-time `TT_FATAL` on valid shapes**,
not a style issue, and it was reachable from the op's own test suite. The port's own reasoning had
gone wrong in a specific, generalisable way — see the second half of the Quasar-template entry under
*Confusion*.

## Provenance

`git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
prints nothing in the reshard checkout — the doc tree is not tracked there. Run from the
`Port_Recipe` checkout that `/localdev/edwinlee/metal2_port.md` symlinks into, it gives:

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

Recipe and audit ran against the same doc revision.

## TTNN ProgramFactory

### Concept realized

**`MetalV2FactoryConcept`**, as the audit decided. No re-decision, nothing surfaced back to the
invoker. Each factory's `create_descriptor` became
`static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const ReshardParams&, const ReshardInputs&, Tensor&)`;
the five factory headers dropped `#include <tt-metalium/program_descriptors.hpp>` and gained
`#include "ttnn/metal_v2_artifacts.hpp"`. `op_owned_tensors` is left default-empty everywhere.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — `ReshardDeviceOperation` never had one
  (confirms the reconciled readiness sheet; this was the column whose inconsistency RED'd the
  prior audit).
- **Pybind entry points removed:** none — `reshard_nanobind.cpp` binds only `ttnn::reshard`; no
  `create_descriptor` was exposed. `reshard_device_operation.{hpp,cpp}`,
  `reshard.{cpp,hpp}` and `reshard_nanobind.{cpp,hpp}` are **untouched** by this diff.
- **Pybind-hook-only factory parameter dropped:** none.

The success case: zero device-op-class edits.

### Open items

- **Factories still use `ttnn::Tensor`, not `MeshTensor`, in their bodies.** `ttnn_factory.md`
  ("Extracting the tensor") recommends extracting the `MeshTensor` at the top of the factory and
  working with it throughout. These factories were left on `ttnn::Tensor` because their bodies
  are dense with TTNN-level queries that `MeshTensor` does not expose — `shard_spec()`,
  `element_size()`, `logical_shape()`, `layout()`, `memory_config()`, `buffer()` — so converting
  would be a substantial refactor of code the port otherwise leaves alone. `.mesh_tensor()` is
  reached only at the `TensorArgument` construction sites (one or two calls per factory).
  Deliberate scope-discipline deviation, flagged here rather than silently taken.
- **Relaxation candidates:** none identified. Every `TensorParameter` is strict, matching the
  audit (`TensorParameter relaxation: none`). The borrowed-DFB parameters (`local` / `output`)
  are the only ones where a relaxation would even be conceivable, and nothing in the kernels
  suggests they tolerate a spec mismatch.
- **No capability gaps hit.** No op-owned tensors, no op-owned `GlobalSemaphore`s, no
  multi-program / per-coord variation. The op is genuinely single-program.

## Handoff points

No capitulation, no boundary-rule assumption violation, no kernel-lib gap, no removed pybind
surface. **One framework gap**, below.

### Framework gap — the spec-time borrowed-DFB size check is not shard-aware

*Owner: Metal 2.0 host-API team. Reporter: this port. Surfaced by PR review (Copilot, 🔴).*

**The check.** `program_spec.cpp:1541-1580` validates a borrowed-memory DFB with

```cpp
const size_t dfb_bytes    = dfb.entry_size * dfb.num_entries;
const size_t tensor_bytes = tensor_spec.compute_packed_buffer_size_bytes();
TT_FATAL(dfb_bytes <= tensor_bytes, ...);
```

**The problem.** A borrowed DFB is a **per-node** resource, but
`compute_packed_buffer_size_bytes()` is a **whole-tensor** quantity derived from the physical
(padded logical) shape — `tensor_layout.cpp:257-274` — with no core-count multiplier, no shard
padding, and no L1 page alignment. (`compute_consumed_memory_bytes_per_bank` is the shard-aware
method, and is not the one used.) So the two sides of the comparison are not commensurable, and
the check is neither an upper nor a lower bound:

- **Normally far too lenient** — a shard of an N-core tensor is ~1/N of the packed size. The
  framework's own comment concedes this: *"a DFB can pass here against the full-tensor size and
  still fail per-bank later. By design."*
- **Sometimes too strict**, which is what bit. When a shard covers most/all of the tensor *and* is
  padded, its real size legitimately exceeds the packed logical size. A tiled `[32, 96]` output
  with a `(32, 128)` shard on one core is 4 tiles of shard vs 3 tiles of packed tensor →
  `TT_FATAL` before launch, on a perfectly valid tensor. The same-width unaligned path reaches the
  same failure by a second route: the padded per-row stride (`local_unit_size_padded > unit_size`)
  rather than a padded shard shape.

**Why this is a regression against legacy, not porter error.** Legacy's equivalent check was
`CircularBufferConfig::set_total_size` against `buffer.aligned_size_per_bank()`
(`circular_buffer_config.cpp:193-231`) — the *real* allocation, which is large enough. Metal 2.0
still applies that per-bank check, but only at attach time
(`program_run_args.cpp:460-468`). The spec-time check is an additional, differently-based gate with
no legacy counterpart, and it is the one that rejects valid programs.

**What the port did.** Clamped each of the three borrowed DFBs to
`min(shard-derived bytes, tensor packed bytes)`. Behaviour-neutral: all three are **address-only**
(every binding kernel uses `get_write_ptr()` / `get_read_ptr()` + offset and issues zero FIFO ops —
verified across all nine kernels), so nothing reads `entry_size` / `num_entries` and `get_*_ptr()`
returns the borrowed base regardless of declared capacity. Sites:
`reshard_program_factory_generic.cpp` (`output_shard`),
`reshard_program_factory_same_width.cpp` (`shard`),
`reshard_program_factory_same_height.cpp` (`shard`).

**Requested fix.** Compare against a shard-aware bound at spec time (the `TensorSpec` can already
produce one via `compute_consumed_memory_bytes_per_bank`), or drop the spec-time size check for
borrowed DFBs and rely on the precise attach-time per-bank check that already exists. Until then,
every op with a borrowed DFB on a padded shard must carry this clamp — the sibling
`experimental/quasar/reshard` independently arrived at the same workaround
(`experimental/quasar/reshard/device/reshard_program_factory_generic.cpp:817-831`), which is
evidence the gap is generic rather than reshard-specific.

**Ticket-ready summary.** *Metal 2.0's spec-time borrowed-DFB size check compares a per-node DFB
size against a whole-tensor, padding-blind packed size, rejecting valid programs whose shard is
padded beyond the logical tensor. Two independent ops have had to clamp their DFB size to work
around it. Either make the bound shard-aware or defer entirely to the existing attach-time per-bank
check.*

Specifically, on the things the recipe warns could force a stop:

- **`sem::` / `tensor::` never crosses out of the op.** The op uses no semaphores at all, and
  every `tensor::` handle is consumed inside the reshard kernels themselves. No out-of-op call
  site demanded a named handle.
- **Case 2 bindings are all in data-movement kernels**, so the sanctioned
  `TensorAccessor::get_bank_base_address()` bridge is available. There are no compute kernels in
  this op at all, so the compute-kernel Case 2 block never applies.
- **No host-computed `base + offset` folded into an address arg.** The audit's Offset-base-pointer
  gate held: every offset is added kernel-side. The one construct that *looked* like a fold —
  `ReshardGenericFactory` back-patching arg index `grid.x + grid.y` with the raw `input_buffer`
  pointer over the `input_buffer->address()` the `detail` helpers had already written
  (`reshard_program_factory_generic.cpp:779-798` pre-port) — was a clean-base `Buffer*` binding,
  not an offset fold. The helpers no longer emit the slot and the back-patch loop is deleted.
- **No GlobalCircularBuffer, no `CBDescriptor::address_offset`, no CTA varargs, no aliased CBs,
  no dead CBs, no `get_cb_tiles_acked_ptr`/`get_cb_tiles_received_ptr`, no cursor surgery.**

## Successes

- **[Caution: Avoid varargs unless absolutely necessary](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  fired exactly as intended, on the op it uses as its worked example.** The generic factory's
  reader is the catalog's named case, and the trap it describes is real: the legacy kernel reads
  `input_shard_addr`, `num_output_pages`, `num_ranges`, `output_page_offset` through a running
  `get_arg_val<uint32_t>(arg_index++)` immediately followed by the genuine per-range vararg loop
  (`reshard_reader.cpp:23-26` pre-port), and the mechanical instinct is to sweep all four into
  `get_vararg`. The catalog's rule — *distinct field read once → named, regardless of legacy
  offset* — split them correctly: the three scalars became named RTAs and the address became a
  `TensorBinding`. Same rule resolved the same shape in `reshard_same_width_reader.cpp:24-26`
  and `reshard_same_height_reader.cpp:18-22`.
- **The mirror half of that caution also fired.** `reshard_reader.cpp:41-42,60-61` reads
  `get_arg_val(start_x_index)` where `start_x_index` was unpacked from the *data* — genuinely
  un-nameable. Keeping the physical-core-coordinate table as the leading block of the same
  per-node vararg vector meant the kernel's data-selected indexing (`get_vararg(start_x_index)`,
  `get_vararg(y_offset + start_y_index)`) and its payload cursor
  (`arg_index = num_x_cores + num_y_cores`) both survived *unchanged* — the cleanest possible
  diff for the hardest kernel in the op.
- **[Pattern: Two-toucher DFB → assign 1P+1C](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
  is exactly right, and the "guard against stacking" warning stopped a wrong turn.** Four of the
  op's five DFBs are the dual-instance work-split shape. Before reading the pattern the
  temptation was real: both instances of `reshard_same_width_reader.cpp` call
  `dfb_scratch.get_write_ptr()` *and* `get_read_ptr()` in the same body, which reads like a
  one-toucher self-loop. The endpoint-assignment census (`get_*_ptr` is a role-free public peek;
  two distinct KernelSpecs are available; 2 touchers with ≤1 locked per role → 1P+1C) resolved
  it to 1P+1C without a self-loop and without the multi-binding flag. Re-derived rather than
  transcribed, and it agreed with the brief on all five DFBs.
- **[Pattern: Conditional / optional DFB bindings](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings),
  specifically its "Promote a CTA gate to a define" sub-section.** The same-width scratch DFB
  exists only when `unaligned && local_is_output`, and the legacy kernel gated it with
  `if constexpr (unaligned)` on a CTA (`reshard_same_width_reader.cpp:38` pre-port) — precisely
  the case that fails name lookup on `dfb::scratch` in the aligned build. The sub-section's
  "watch the emission target … the define must be fed to every kernel that references the
  conditionally-bound DFB" is why `UNALIGNED` goes on **both** KernelSpecs, not just the
  PRODUCER one.
- **[Hardware configuration](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#hardware-configuration)'s
  "match on the *values*, not the role name" caught a real trap.** `NdReshardCopyLocalShardFactory`
  uses a raw `DataMovementConfigDescriptor{RISCV_0, RISCV_0_default}` /
  `{RISCV_1, RISCV_1_default}` pair. `NOC::RISCV_0_default == NOC_0` and
  `NOC::RISCV_1_default == NOC_1`, so the resolved triples are `(RISCV_0, NOC_0)` and
  `(RISCV_1, NOC_1)` — which match **neither** the reader default `(RISCV_1, NOC_0)` nor the
  writer default `(RISCV_0, NOC_1)`. The names look like defaults; the values are custom. Both
  are replicated field-for-field with an explicit `DataMovementGen1Config`. Reaching for the
  "close" helper here would have swapped both NOC assignments — the exact silent regression the
  section exists to prevent. Every other kernel in the op resolves to a genuine reader/writer
  default and uses the arch-agnostic TTNN helper.

## Friction

### Gaps

- **The kernel-side `#include` rule doesn't account for a Case 2 binding.** The
  [kernel-side whitelist](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist)
  opens with "the port **adds exactly two headers**: `experimental/kernel_args.h` and
  `api/dataflow/dataflow_buffer.h`", and adds that types the kernel keeps using unchanged —
  naming `TensorAccessor` explicitly — "come from the same headers before *and* after, so you
  neither add nor touch their includes (don't go hunting for them)." That holds for a kernel that
  *already* used `TensorAccessor`. It does not hold for a **Case 2** kernel that had none: the
  four raw-NoC reshard kernels (`reshard_same_width_{reader,writer}.cpp`,
  `reshard_same_height_{reader,writer}.cpp`) plus the two generic readers reach the tensor only
  through `AllocatorBank` / `UnicastEndpoint`, so binding `tensor::remote` / `tensor::input` for
  the `get_bank_base_address()` bridge required **adding `api/tensor/tensor_accessor.h`** — a
  third header the rule doesn't sanction. Suggested fix: add a clause to whitelist rule 5 noting
  that a Case 2 binding in a kernel with no prior `TensorAccessor` use adds
  `api/tensor/tensor_accessor.h`, and soften the "exactly two headers" phrasing accordingly.
- **No guidance on per-node-varying vararg counts.** `KernelAdvancedOptions::num_runtime_varargs`
  is a single per-kernel number, and the per-node override
  (`num_runtime_varargs_per_node`) is `[[deprecated]]`. Every vararg-carrying kernel in this op
  has a per-core-varying tail length (transfer / segment / range counts differ per core), so all
  three factories zero-pad each node's vararg vector up to the kernel's maximum. That is safe
  here — each kernel's loop is bounded by a named count RTA, so padding words are never read, and
  the peak per-node arg count is unchanged from legacy because the longest node already carried
  the max — but the recipe and the varargs Caution say nothing about it, and the reasoning
  (why padding is safe, and why it isn't a regression) had to be derived. Suggested fix: a short
  "varargs with a per-node-varying count → pad to the per-kernel maximum; the named count RTA
  must bound the kernel's loop" note in
  [Caution: Avoid varargs](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary).
  It is also worth stating that aggregate-initializing `KernelAdvancedOptions` at all touches a
  `[[deprecated]]` member, which may need `-Wno-deprecated` scrutiny in a `-Werror` build.
- **`entry_size` / `num_entries` from a legacy `total_size` + `page_size` pair is left to the
  porter.** The [migration guide's `DataflowBufferSpec` section](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#dataflowbufferspec)
  shows the mapping only for a legacy CB whose `total_size` was *written* as
  `num_pages * page_size`. Four of this op's five CBs compute `total_size` from an unrelated
  expression (e.g. generic: `output_shard_shape[0] * output_shard_shape[1] * element_size` with
  `page_size = output_buffer->page_size()`), so the port has to derive `num_entries =
  total_size / page_size` and satisfy itself that the division is exact. It is — the legacy
  `CircularBufferConfig::set_page_size` hard-requires `total_size % page_size == 0`
  (`tt_metal/impl/buffers/circular_buffer_config.cpp:180-187`) — but confirming that meant
  reading framework source. Suggested fix: state the rule (`entry_size = legacy page_size`,
  `num_entries = legacy total_size / page_size`, exact by the legacy CB invariant) in that
  section.
- **Borrowed-DFB sizing is under-documented, and the missing half caused the port's only real bug.**
  The recipe says a borrowed DFB's backing address refreshes from the `TensorArgument` and that no
  `dfb_run_overrides` entry is needed, but says nothing about the *sizing* checks — of which there
  are **two**, on different bases, and only one is a legacy analog:
  - *Attach time*, `dfb_total_bytes <= buffer->aligned_size_per_bank()`
    (`program_run_args.cpp:460-468`) — byte-identical to legacy's `total_size <= max_size_` with
    `max_size_ = buffer.aligned_size_per_bank()` (`circular_buffer_config.cpp:193-231`). Preserving
    `entry_size * num_entries == legacy total_size` preserves this outcome, which is the conclusion
    the port drew and stopped at.
  - *Spec time*, `dfb_bytes <= tensor_spec.compute_packed_buffer_size_bytes()`
    (`program_spec.cpp:1541-1580`) — **no legacy counterpart**, and it compares a per-node DFB
    against a whole-tensor padding-blind size, so preserving legacy parity is *not* sufficient to
    pass it. This is what the port missed, and it is a latent launch-time `TT_FATAL` on valid padded
    shards; see *Handoff points* for the full analysis and the clamp.

  Suggested fix, in two parts. **Docs:** the borrowed-memory paragraph should name both checks, say
  which is the legacy analog, and warn that the spec-time one is not shard-aware — so a porter knows
  legacy parity alone does not clear it. A worked line would do it: *"size your borrowed DFB from the
  shard, then clamp to the backing tensor's packed size; the spec-time check is whole-tensor and
  padding-blind."* **Framework:** make the spec-time bound shard-aware or drop it (see *Handoff
  points*), at which point the doc note and every op's clamp can go away.
- **Nothing says a borrowed-only `TensorParameter` needs no kernel `TensorBinding`.** Three DFBs
  here borrow from a `TensorParameter` no kernel binds (`local` in same-width / same-height,
  `output` in generic). Whether that trips the "TensorParameter is defined but not bound by any
  kernel" validator is a natural worry, and the answer is only in framework source
  (`program_spec.cpp:533-543` registers a `borrowed_from` reference as a use). The already-ported
  Quasar reshard hedges by adding a redundant `TensorBinding` for its borrowed parameter on both
  kernels — an unused `tensor::` accessor in each kernel. One sentence in the borrowed-memory
  paragraph would retire that hedge.

### Confusion

- **The brief's cross-op "port-together set" is stale, and it is the kind of stale that changes
  the port's scope.** Both the brief ("Cross-op / shared kernels (port-together set)") and the
  audit's Heads-ups state that the six shared kernels in
  `data_movement/sharded/device/kernels/dataflow/` are co-borrowed by
  `experimental/quasar/reshard/` and that the shared kernels plus **both** consuming ops must be
  ported as one unit. Taken at face value that pulls a second whole op into the diff. It is not
  true: the Quasar reshard carries its own private copies of all nine kernels under
  `experimental/quasar/reshard/device/kernels/` and instantiates only those paths, and — a second
  staleness — all five of its factories are *already* on `create_program_artifacts`, not the
  "SameWidth/SameHeight/CopyLocal still legacy device-op" the audit reports. Verified by
  extracting every `kernels/` string literal from `experimental/quasar/reshard/device/*.cpp` and
  by a repo-wide grep for consumers of the shared paths, which returns only the three reshard
  factories ported here. The
  [shared-dataflow-kernel Caution](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel)
  prescribes exactly the check that resolved it (`grep -rl <kernel-filename> ttnn/cpp/ttnn/operations/`,
  then in-place if every consumer is in the bundled set) — so the catalog was right and the brief
  was out of date. Suggested fix: have the audit derive the co-borrower list from a grep at audit
  time rather than from the readiness sheet, since a sibling op forking its kernels silently
  invalidates the sheet's answer.
- **The recipe's "don't lean on already-ported ops" warning is even more warranted than it
  reads.** For this op the temptation was unusually strong: `experimental/quasar/reshard/` is a
  *complete Metal 2.0 port of the same op*, factory for factory. It is also wrong in ways that
  would have shipped as silent numerics bugs had it been used as a template — it is an older
  snapshot of these kernels, from before the unaligned/padded-stride fix landed on the legacy
  side. Concretely: its `reshard_same_width_writer.cpp` has **no unaligned path at all**; its
  same-width DFB is sized `entry_size = unit_size` where the current legacy CB uses
  `local_unit_size_padded` (with a comment asserting parity that no longer holds, and a
  `(void)total_size;` to silence the now-unused variable); and both same-width kernels carry
  leftover `WATCHER_RING_BUFFER_PUSH` debug instrumentation and an
  `#include "api/debug/ring_buffer.h"` marked "remove after". Every value in this port was taken
  from the current legacy source instead. Worth promoting from a general caution to a named
  failure mode: *a ported sibling can be a stale fork of the very kernel you are porting; diff it
  against the legacy source before trusting any value in it.*

  **But the inverse error is just as real, and this port made it.** Having established that the
  Quasar copy was a stale fork, I stopped mining it — and it turned out to contain one thing this
  port genuinely needed: the borrowed-DFB size clamp (its
  `reshard_program_factory_generic.cpp:817-831`, with a comment describing the padded-shard failure
  in detail). That is not staleness, it is a framework gap the Quasar porter hit first and solved
  correctly; skipping it cost a 🔴 review finding and a latent launch-time `TT_FATAL` in three
  factories (see *Handoff points*). So the guidance needs both halves: a ported sibling's **values**
  are untrustworthy without a diff against current legacy, but its **workarounds** are evidence of
  real framework gaps and deserve to be understood before being discarded. A cheap discriminator:
  a divergence with a comment explaining *why* is a candidate finding; a divergence with no
  explanation is probably drift.
- **The shared reference docs were hard to locate.** `metal2_port.md` links its four companions
  as `../shared/*.md`, but the invoker supplies only the absolute path to `metal2_port.md`, which
  in this workspace is a symlink into a *different* checkout
  (`Port_Recipe/docs/source/.../ai/port/metal2_port.md`). A shallow search of the workspace and
  of the reshard checkout's own `docs/` tree found nothing, and the port ran a while on
  `metal2_port.md` plus framework headers alone before the companions turned up by resolving the
  symlink. Suggested fix: have the recipe's "Reference material" list resolve its own location
  (e.g. "these live beside this file, at `<dirname>/../shared/`"), or have the invoker pass the
  `ai/` directory rather than the single file.

## Open items for downstream

- **Cross-op kernel touches — in-place modification, no fork.** Six kernel sources outside the
  op directory were modified:
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_reader.cpp`
  - `…/reshard_reader_diff_width.cpp`
  - `…/reshard_same_width_reader.cpp`
  - `…/reshard_same_width_writer.cpp`
  - `…/reshard_same_height_reader.cpp`
  - `…/reshard_same_height_writer.cpp`

  **Path taken: in-place**, per the shared-dataflow-kernel Caution's option 1. The bundled
  consumer set is the complete consumer set: `reshard_program_factory_generic.cpp`,
  `reshard_program_factory_same_width.cpp`, `reshard_program_factory_same_height.cpp` — all in
  this PR. **Remaining unmigrated consumers: none.** No `_metal2` fork was created and no legacy
  copy remains, so there is no sunset checklist and no drift-discipline burden. The sibling ops in
  the same shared pool (`interleaved_to_sharded`, `sharded_to_interleaved`) use *different*
  kernels in that directory (`reader_unary_sharded.cpp`, `writer_unary_sharded.cpp`, …) and are
  untouched.
- **RTA → CRTA candidates (deliberately not converted — changes dispatch semantics).** Several
  named RTAs carry the same value on every node and would dispatch more efficiently as common
  runtime args:
  - `reshard_same_height`: `total_num_sticks`, `local_stride_bytes`, `remote_stride_bytes` (all
    three are node-invariant; only `num_segments` genuinely varies per core).
  - `reshard_generic`: the physical-core-coordinate table is byte-identical on every node *and*
    on both kernels, but is replicated into each node's `runtime_varargs`. It is a natural
    `common_runtime_varargs` block — the kernel would read it with `get_common_vararg(...)` and
    the per-node vararg vector would shrink to just the range payload. Kept per-node to preserve
    legacy dispatch semantics exactly.
- **Host-side perf left alone.** `ReshardGenericFactory` calls
  `detail::get_core_page_ranges…()` **inside** its per-core loop
  (`reshard_program_factory_generic.cpp:722,736`), rebuilding the entire output-core → page-range
  map once per core — O(cores) redundant work on a map that does not depend on the loop variable.
  Hoisting it out of the loop is behavior-preserving and would be a clear win, but it is unrelated
  to the Metal 2.0 transformation. Preserved verbatim.
- **Pre-existing findings the audit flagged, confirmed untouched by this port** (all routed to the
  ops team, all left exactly as-is):
  - **Unreachable code** in `is_valid_for_legacy_reshard` (`reshard_device_operation.cpp:39-50`):
    the unconditional `return` at line 39 makes the whole row-major shard-width block at 41-50
    dead. In the device-operation class, so off-limits regardless.
  - **Live `DPRINT`** in a shipping kernel: `reshard_same_width_reader.cpp` carried
    `DPRINT("addr: {}\n", addr);` in the unaligned reader path, plus two commented-out
    `print_bf16_pages` calls. The port initially carried all of it across verbatim, on the
    scope-discipline reading that removing a pre-existing debug print is an unrelated cleanup that
    belongs in the report rather than the diff. **PR review overrode that** (see *Review round*
    below): the repo's own PR-review rules for `ttnn/**` list "no debug print statements in
    production code … remove leftover debugging artifacts before merge" as 🟡 IMPORTANT
    (`.github/instructions/ttnn.instructions.md:18`), and it applies to files this PR touches. The
    live `DPRINT` and the now-dead `#include "api/debug/dprint_pages.h"` are removed. The two
    **commented-out** `print_bf16_pages` lines are kept — they are inert, and deleting comments
    cuts against whitelist rule 8; whether to keep them is the op owner's call.
  - **Dead RTA read** `num_output_pages` in `reshard_reader.cpp` and
    `reshard_reader_diff_width.cpp` — unpacked, never referenced. Faithfully preserved as a named
    RTA (declared in the schema, set per node, read into an unused local), so the port stays a
    pure syntax swap. Dropping it is a one-line cleanup for a separate PR: it would also let the
    host stop emitting the value, which is genuinely used host-side only to seed kernel 1's
    `output_page_offset`.
  - **CB index 16** was the generic factory's hardcoded `dst_cb_index`. Metal 2.0 assigns DFB ids
    itself, so the specific slot number is gone. No behavioral consequence on Gen1 (a DFB lowers
    to a plain CB and the index only picks a slot), but noted in case anything downstream ever
    depended on that particular index.
- **Test coverage note.** `tests/nightly/t3000/ccl/test_slice_reshard_async.py` exercises reshard
  inside a multi-device CCL flow and was **not** included in the recommended verification set
  below (it needs a T3000). Worth a run before merge on hardware that has one, since it is the
  only coverage of reshard under a mesh workload.

---

## Recommended verification

Not run by the porter (build and test were deferred to the invoker). Minimum set, in the
recipe's recommended order:

```bash
# 1. Build (Metal + all TTNN test binaries)
./build_metal.sh --build-tests

# 2. Primary correctness — covers all five factory types
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_reshard.py -x -q
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nd_reshard.py -x -q

# 3. Program-cache behaviour specifically (the cache-hit UpdateTensorArgs path)
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_reshard.py \
    -x -q -k "program_cache"

# 4. Adjacent coverage that calls ttnn.reshard
pytest tests/ttnn/unit_tests/operations/data_movement/test_core.py -x -q -k "reshard"
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py -x -q \
    -k "test_llama_mlp_width_sharded_to_interleaved_pcc_err"
```

(Step 4's `test_sharded.py` case is the only test in that file that actually calls
`ttnn.reshard` — its module-level `run_reshard_test` helper at
`test_sharded.py:1346` is dead code, never invoked.)

Factory-to-test mapping, so a failure points at the right factory:

| Factory | Covering tests |
|---|---|
| `ReshardSameWidthFactory` (HEIGHT→HEIGHT) | `test_reshard`, `test_reshard_rn50`, `test_reshard_aligned_channels_height_sharded`, `test_reshard_unaligned_channels_height_sharded` (the `unaligned` / scratch-DFB path), `test_reshard_variant3_skip_height_sharded` |
| `ReshardSameHeightFactory` (row-major WIDTH→WIDTH) | `test_sd_reshard`, `test_reshard_sentencebert_embeddings{,_full}` |
| `ReshardGenericFactory` | `test_reshard`, `test_reshard_diff_width` (the `reshard_reader_diff_width.cpp` source), `test_reshard_interleaved_to_block_sharded`, `test_core.py::test_reshard_conv` |
| `NdReshardCopyPagesFactory` (DRAM→DRAM) | `test_reshard_dram_to_dram`, `test_DRAM_nd_reshard` |
| `NdReshardCopyLocalShardFactory` | `test_nd_reshard`, `test_nd_reshard_different_input_output_grid`, `test_reshard_between_L1_and_DRAM`, `test_dram_reshard{,_with_program_cache}` |

`test_reshard_diff_width` and `test_reshard_unaligned_channels_height_sharded` are the two most
load-bearing cases in the set: the first is the only coverage of the second runtime-selected
generic kernel source, the second the only coverage of the conditional scratch DFB and the
`UNALIGNED` define.

Not included, and why:

- `tests/nightly/t3000/ccl/test_slice_reshard_async.py` — needs a T3000 (see the note above).
- `tests/sweep_framework/sweeps/model_traced/reshard_model_traced.py` — sweep harness, long
  runtime; worth a pass in CI rather than locally.
- `tests/tt_metal/tt_metal/data_movement/reshard_hardcoded/` — a tt_metal-level test with its own
  private copy of `reshard_reader.cpp`; unaffected by this port.
