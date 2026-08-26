# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/transformer/sdpa`

Scope: `SDPAOperation` (`SDPAProgramFactory`) + `JointSDPADeviceOperation` (`JointSDPAProgramFactory`),
the two DeviceOperations the audit cleared GREEN. The five RED DeviceOperations (Sparse, SparseMSA,
RingDistributed, RingJoint, ExpRingJoint) were not touched.

## Outcome
**PORTED** — both factories converted to `ProgramSpecFactoryConcept`. The full confirmed baseline passes
with the Metal 2.0 host-side legality checks forced on (`METAL2_CHECKS_FORCED` markers observed in every
run, then reverted before commit):

| Suite | Result |
|---|---|
| `unit/.../test_sdpa_prefill.py` | 8 passed, 2 skipped, 0 failed |
| `unit/.../test_windowed_sdpa.py` | 40 passed, 0 failed |
| `nightly/.../test_sdpa_chunked.py` | 36 passed, 0 failed |
| `nightly/.../test_sdpa_prefill.py` | 787 passed, 208 skipped, 0 failed |
| `nightly/.../test_sdpa_joint.py` | 168 passed, 32 skipped, 0 failed |

Paths exercised: causal, non-causal (KV-chain store-and-forward + mcast), provided-mask, generated/
lightweight/padded masks, sliding-window, windowed (block-diagonal), chunked/paged + flexible-chunked +
HMA geometry override, MLA, attention-sink, streaming vs standard compute, and program-cache (cache-hit
tensor refresh). The `cb_*` → `dfb_*` kernel-local rename (self-audit item 4) was applied to the three
SDPA forks + joint compute and re-validated (prefill 8, windowed 40, chunked 34, joint 48; 0 failed).

## Provenance
- **Recipe docs (this port):** `0094ee1fd60 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `d6087d9353f 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory
### Concept realized
Both on `ProgramSpecFactoryConcept` (as the audit chose). Neither op has `override_runtime_arguments`,
so the framework refreshes tensor bindings on cache hits and each factory implements a single
`create_program_artifacts`. Cache-hit correctness verified by the `*_program_cache` tests.
### Device-op-class edits
- Pybind entry points removed: **none** (`sdpa_nanobind.cpp` binds no `create_descriptor`).
- Custom `compute_program_hash`: **none** (both) — nothing to preserve.
- Only forced edit: the factory method signature `create_descriptor` → `create_program_artifacts` in each
  `*_device_operation.hpp` + factory `.cpp`, plus a `#include "ttnn/metal_v2_artifacts.hpp"`. Nested
  `program_factory_t` already existed (no direct-descriptor exception 3).

## Handoff points
- **`RingDistributedSDPADeviceOperation` binds the forked SDPA kernels.** The three SDPA kernels
  (`reader_interleaved.cpp`, `writer_interleaved.cpp`, `compute/sdpa.cpp`) are also bound by
  `ring_distributed_sdpa_program_factory.cpp` (a RED/blocked factory), so they were forked (rung 2) rather
  than converted in place — see Open items. No cross-team API change is needed for this port.

## Successes
- **Shared-kernel census caught a real lending relationship.** A path-exact
  `grep -rln "transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp" ttnn/cpp/ttnn/operations`
  showed `ring_distributed_sdpa_program_factory.cpp` also binds the SDPA kernels. The substring form was a
  false-positive trap: `ring_joint_reader.cpp` contains `joint_reader.cpp` as a substring, which made the
  JointSDPA kernels look shared when they are exclusive. Following
  [Caution: Porting a shared kernel](../shared/port_patterns.md#caution-porting-a-shared-kernel) rung 2
  (create `_metal2` forks beside the originals) is what keeps RingDistributed building.
- **Conditional / optional bindings pattern held across a very large kernel.** SDPA's reader has ~9
  conditionally-bound resources (mask, page_table, chunk_start_idx ×2, attention_sink, and the 3 KV-chain
  semaphores) plus KV-chain runtime args. `#ifdef`-gating the token references + named-arg reads, with the
  gate moved from a CTA to a `KernelSpec::compiler_options.defines` `#define`, worked exactly as
  [Pattern: Conditional / optional DFB bindings](../shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  describes — including the `#ifdef`-gating of the *named RTAs* in the non-causal chain block.
- **Placeholder token-alias for template-discarded paths.** Where a conditional DFB token flows into a
  template that discards it on the off-path (`windowed_generate_if_enabled<false, ...>`, `sdpa_joint<..,
  use_joint_mask=false, ..>`), aliasing the token to a bound placeholder (`dfb::out` / `dfb::q_in`) under
  `#else` kept the call sites clean without binding the resource — the same trick the legacy factory used
  with numeric CB indices (`cu_window_seqlens = q_in`).
- **`get_bank_base_address` never needed** — all bindings were Case 1 (`TensorAccessor`), as the audit said.
- **fp32 `unpack_modes` required-entry rule fired as documented.** SDPA derives `qk_im`/`sum_A`/`sum_B` as
  `fp32_dest_acc_en ? Float32 : Float16_b`; with `enable_32_bit_dest` on, the validator requires an explicit
  `unpack_modes` entry for each. Added `{qk_im, sum_A, sum_B} → UnpackToSrc` (legacy default) gated on
  `fp32_dest_acc_en` ([Hardware configuration](../port/metal2_port.md#compute-kernels)).

## Friction
### Gaps
- **`Semaphore<>(sem::name)` construction is under-documented.** The recipe says `sem::name` does *not*
  convert to `uint32_t` and is "consumed inside the op's own kernels"; the KV-chain reader constructs
  `Semaphore<>(sem::valid)` etc. This works (confirmed against `data_movement/sort` kernels and the build),
  because the generated `sem::name` is accepted by `Semaphore`'s `uint32_t` constructor — but the recipe's
  boundary note reads as if it might not. A one-line "in-op `Semaphore<>(sem::name)` is the intended
  consumption form" would remove the doubt.
- **In-directory shared *helper headers* aren't the shared-*kernel* Caution case, but the recipe's
  CircularBuffer sweep doesn't carve them out.** `dataflow_common.hpp` / `compute_common.hpp` /
  `compute_streaming.hpp` are `#include`d by SDPA/Joint *and* the blocked ops; they take `uint32_t` cb-ids
  and read no args, so they are boundary-bridged and left untouched (`dfb::name → uint32_t`). But the
  self-audit's "grep CircularBuffer across the op directory → zero" cannot hold, because those helpers keep
  `CircularBuffer`. Scoping the sweep to *converted* files (done here) is the right reading; the recipe
  could say so explicitly.

### Confusion
- **The `-Wno-deprecated-declarations` host flag made the `SemaphoreAdvancedOptions::initial_value`
  deprecation a non-issue.** The field is `[[deprecated]]`, but the ttnn host build sets
  `-Wno-deprecated-declarations`, so setting `valid`'s non-zero initial value needs no `#pragma`. (Left a
  descriptive comment instead.)
- **`KernelSpec::CompilerOptions` is a nested type, not a namespace-level alias.** `CompilerOptions::Defines`
  does not resolve even with `using namespace tt::tt_metal::experimental;` — must write
  `KernelSpec::CompilerOptions::Defines`. Cost one build iteration.

## Open items for downstream
- **Shared kernel forks (rung 2).** SDPA's `reader_interleaved.cpp`, `writer_interleaved.cpp`,
  `compute/sdpa.cpp` are lent to `RingDistributedSDPADeviceOperation`
  (`ring_distributed_sdpa_program_factory.cpp`). This port **created** the forks beside each original
  (`reader_interleaved_metal2.cpp`, `writer_interleaved_metal2.cpp`, `compute/sdpa_metal2.cpp`) and added the
  pointer comment to each legacy original. Remaining unmigrated consumer:
  **`RingDistributedSDPADeviceOperation`**. Sunset the fork (delete the legacy copy, rename the fork onto its
  name) once RingDistributed migrates.
- **In-directory shared helper headers left with legacy `CircularBuffer`** (`dataflow_common.hpp`,
  `compute_common.hpp`, `compute_streaming.hpp`, `generate_bcast_scalar.hpp`). Boundary-bridged, shared with
  the blocked ops; not converted. The now-uninstantiated `read_page_table_for_batch` template in
  `dataflow_common.hpp` (its only caller, the SDPA reader, now inlines the page-table read against the
  `page_table` TensorBinding) is dead-but-harmless; it stays for the blocked ops that may adopt it later.
- **`sdpa_interleaved_cb_ids.hpp`** (`CBIds` struct) is now unused by the ported SDPA factory (CB indices are
  DFB bindings). It is still `#include`d by the blocked `ring_distributed`/`sparse`/`ring_joint` factories, so
  it is left in place.
- **Attention-sink prefill path is compile-covered but lightly runtime-covered** in the confirmed baseline
  (the dedicated sink sweeps are decode-op tests, out of scope). The prefill baseline's
  `test_sdpa_with_attention_sink` cases exercise it.
