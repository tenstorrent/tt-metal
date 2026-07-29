# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/normalization/batch_norm`

Companion to `METAL2_PORT_PLAN.md` (structural decisions), `METAL2_PORT_BRIEF.md` (audit input) and
`METAL2_PREPORT_AUDIT.md` (full audit record).

## Outcome

**`PORTED`** — both factories converted in one change and the confirmed test set passes with no regression.

- `BatchNormOperation::BatchNormFactory` → `ProgramSpecFactoryConcept`, with **both** of its selectable
  compute sources (`batch_norm_kernel.cpp`, `batch_norm_sfpu_kernel.cpp`).
- `RunningStatistics::RunningStatisticsProgramFactory` → `ProgramSpecFactoryConcept`, with **both** of its
  selectable compute sources (`running_statistics_kernel.cpp`, `running_statistics_sfpu_kernel.cpp`).

Nothing is left for a later pass: the op has exactly these two factories and all 8 kernel sources converted.

## Verification

### Build

`./build_metal.sh --build-tests` — **SUCCESS**, no new warnings. (`-Werror` is on for both host and kernel
builds; the kernel flags add `-Wno-unused-variable`, which is what lets a CTA that is only referenced inside
an `#ifdef`-excluded block stay declared.)

### Tests

The confirmed baseline from [Locate and confirm the op's tests] — `tests/ttnn/unit_tests/operations/fused/test_batch_norm.py`
and `tests/ttnn/unit_tests/operations/fused/test_batch_norm_program_cache.py`, confirmed with the invoker as
complete (no C++ gtest exists for this op) — run **before** and **after** the port:

| | passed | xfailed | failed |
|---|---|---|---|
| pre-port (baseline) | 1560 | 786 | 0 |
| post-port | 1560 | 786 | 0 |

**No regression: identical counts, zero failures.** The `xfailed` block is the pre-existing sharded-memory-config
set (`Input tensors to batch norm must be interleaved`), unchanged by the port.

### Coverage — all four compute sources and every conditional binding path were actually exercised

Confirmed from the JIT cache rather than assumed, since half this port's risk sits in config-selected paths
that a green test run could silently skip:

| kernel source | compiled variants | conditional path evidence |
|---|---|---|
| `batch_norm_sfpu_kernel.cpp` | 35 | 13 variants bind `dfb::output_final` → **both** typecast-on and typecast-off |
| `batch_norm_kernel.cpp` (FPU) | 2 | reached via the explicit `fp32_dest_acc_en=False` parametrization |
| `running_statistics_sfpu_kernel.cpp` | 20 | 2 variants bind `dfb::writer_updated_mean` / `writer_updated_var` → **both** stat-typecast-on and -off |
| `running_statistics_kernel.cpp` (FPU) | 1 | same `fp32_dest_acc_en=False` path |
| `writer_batch_norm.cpp` | 69 | 12 bind `tensor::weight`, 12 bind `tensor::bias` → **both** present and absent, i.e. both sides of every `#ifdef` this port introduced |
| `writer_running_statistics.cpp` | 42 | 8 bind `tensor::old_running_mean`, 8 bind `tensor::old_running_var` → both present and absent |

### Hardware config — empirical before/after diff, not just a read-through

The recipe's [Hardware configuration](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#hardware-configuration)
section asks for a before/after diff, because these are silent perf/precision settings with no test net. Both
runs' JIT artifacts were still on disk, so the diff was done **on the generated device config** rather than by
inspecting the factory source. For each compiled variant I read `chlkc_descriptors.h` and built a signature
from, **per logical DFB**, `unpack_src_format`, `unpack_dst_format`, `unpack_tile_size`, plus the kernel-wide
`MATH_FIDELITY`, `APPROX`, `DST_ACCUM_MODE` and `DST_SYNC_MODE`. Post-port DFB ids are framework-assigned, so
each post-port signature was un-permuted back into legacy CB-index order using the variant's own
`kernel_bindings_generated.h`.

| kernel source | pre-port | post-port | post-port configs byte-identical to a pre-port config |
|---|---|---|---|
| `batch_norm_sfpu_kernel.cpp` | 35 variants → 25 distinct configs | 35 → 25 | **25/25** |
| `batch_norm_kernel.cpp` | 2 → 1 | 2 → 1 | **1/1** |
| `running_statistics_sfpu_kernel.cpp` | 20 → 17 | 20 → 17 | **17/17** |
| `running_statistics_kernel.cpp` | 1 → 1 | 1 → 1 | **1/1** |

An exact bijection in every case — same variant count, same distinct-config count, and every post-port config
identical to its pre-port twin. This is the direct check on the two fields that fail silently:

- **`unpack_modes`** — `unpack_dst_format[]` is precisely where `unpack_to_dest_mode` lands in the JIT
  (`tt_metal/jit_build/data_format.cpp:212-217`), so an entry re-keyed to the wrong DFB, dropped, or inverted
  would show up as a differing array. None did. *Worth recording for the next porter:* the JIT applies an
  `UnpackToDest` entry **only when the DFB's format is Float32** and otherwise ignores it, so an entry on a
  bfloat16 DFB is inert — which is why this op's `fp32_dest_acc_en=true` + all-bfloat16 configs come out
  identical either way, and why translating the legacy list verbatim (including its inert entries) is the
  safe choice.
- **`dst_full_sync_en` → `double_buffer_dest`** — the inverted field surfaces as
  `#define DST_SYNC_MODE DstSync::SyncHalf` (double-buffered) vs `SyncFull`. Every variant reproduced the
  legacy value, confirming `ttnn::to_compute_hardware_config` applied the inversion in the direction the op
  needs.

### Anti-pattern self-audit

Every checklist item from the recipe's
[Anti-pattern self-audit](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#anti-pattern-self-audit),
each verified by grep over the op directory:

- [x] **No `tensor.buffer()->address()` survived** — zero hits for `address()` op-wide.
- [x] **No magic-number CB indices in CTAs** — zero hits for `CBIndex` / `CBDescriptor` / `NUM_CIRCULAR_BUFFERS`;
  the only `compile_time_args` values are the 4 + 3 scalar predicates and format codes listed in the plan.
- [x] **No `TensorAccessorArgs<N>()` survived** in any of the 8 kernels — zero hits, and zero for
  `get_compile_time_arg_val` / `get_arg_val` / `get_common_arg_val` as well.
- [x] **Conditional DFB bindings follow the pattern** — every conditional binding (`output_final`,
  `writer_updated_mean`, `writer_updated_var`) is conditionally bound on the host, carries a matching
  `compiler_options.defines` entry, and is `#ifdef`-gated kernel-side at both the alias and every use. The
  four conditional *tensor* bindings likewise. No binding was made unconditional as a workaround; the
  unconditional DFB bindings (`weight`, `bias`, `old_running_*`, `temp_1`, `output`) are unconditional
  because the *legacy allocation* was, not to dodge name lookup.
- [x] **No `.id` extraction at LLK call sites** — zero hits; `dfb::name` (or a `constexpr` alias of it) is
  passed directly everywhere, and no temporary `DataflowBuffer` is constructed to fetch an id.
- [x] **No CTA→RTA demotion in compute kernels** — nothing moved from CTA to RTA. (The reverse also did not
  happen: the four gate CTAs became `defines`, which is the sanctioned promotion, not a demotion.)
- [x] **No unnecessary multi-binding flag, and never stacked with a self-loop** — zero hits for
  `allow_instance_multi_binding`. Census re-derived independently; no DFB reaches ≥3 touchers or has two
  kernels locked to the same FIFO role. No DFB is both self-looped and multi-bound.
- [x] **All CTAs are named** — both factories' `compile_time_args` are `Table<std::string, uint32_t>`
  literals; there is no positional CTA vector left in either file.
- [x] **No nameable argument smuggled into varargs** — zero `get_vararg` / `num_runtime_varargs` /
  `num_common_runtime_varargs` anywhere. Every argument in all 8 kernels is a distinct field read once, so
  all are named; there was no genuine indexed collection to justify a vararg.
- [x] **Every `hw_config` reproduces the legacy op's resolved values** — see the empirical diff above. DM:
  all four DM kernels resolve to the reader / writer defaults and take the matching arch-agnostic helper.
  Compute: 44/44 generated configs identical, including `bfp_pack_precision_mode` (left default, matching
  the legacy default) and `unpack_modes`.

Also swept, per the whitelist's "this transition is total" clause: **zero** `CircularBuffer` / `CBDescriptor`
references survive in code on either side of the op — the only textual mentions are in the two factories'
header comments and the `unpack_modes` translation comment, which describe what the legacy code *was*.


## Provenance

- **Recipe docs (this port):** `40b61b016a1 2026-07-29 docs(metal_2.0): fix stale API symbol names across the porting docs`
- **Audit docs (inherited):** `40b61b016a1 2026-07-29 docs(metal_2.0): fix stale API symbol names across the porting docs`
  *(copied verbatim from `METAL2_PORT_BRIEF.md`; the audit was re-run against the same doc revision this
  port ran against)*

## TTNN ProgramFactory

### Concept realized

**`ProgramSpecFactoryConcept`** on both factories — exactly the concept the audit chose; no re-decision, and
nothing surfaced during the port that argued against it.

Each factory now exposes a single
`static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)`
and returns `{.spec = …, .run_params = …}` with `op_owned_tensors` defaulted. `AllFactoriesValid` is satisfied
on both one-alternative `program_factory_t` variants.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** **none** — neither device-op defined one, so there was nothing
  to delete. (`BatchNormOperation::operation_attributes_t::to_hash()` at
  `device/batch_norm_device_operation.cpp:121-123` was left **completely untouched**, per the brief's explicit
  instruction. It is not a `compute_program_hash` and the readiness sheet does not score it as one.)
- **Pybind entry points removed:** **none.** `batch_norm_nanobind.cpp:70-84` binds only `&ttnn::batch_norm` —
  no factory entry point was ever exposed, so the port's removal of `create_descriptor` is not a user-visible
  API surface change and nothing in the nanobind file needed touching.
- **Factory-parameter unwind (exception 3):** not applicable — neither `create_descriptor` carried a
  pybind-hook-only parameter.

The only forced device-op-*header* edits were the two the concept flip requires, one per DOp:
`create_descriptor` → `create_program_artifacts` in the factory struct
(`device/batch_norm_device_operation.hpp:38-43`, `device/running_statistics_device_operation.hpp:35-40`),
and swapping the now-unused `#include <tt-metalium/program_descriptors.hpp>` for
`#include "ttnn/metal_v2_artifacts.hpp"`. No `validate*`, `invoke`, `compute_output_specs`, attribute-parsing
or `TT_FATAL` line was changed in either device-op.

### Open items

- **Relaxation candidates: none applied, none identified as needed.** All 11 `TensorParameter`s stay strict.
  No kernel uses `ArgConfig::Runtime*` (`RuntimeTensorShape` / `RuntimeShardShape` / `RuntimeBankCoords`), so
  the migration guide's pre-migration check does not fire and there is no legacy relaxation to mirror. The
  op's dataflow kernels *are* written shape-agnostically (they walk tiles by page id), so
  `dynamic_tensor_shape` would probably widen cache equivalence — but that is a deliberate correctness call
  for the op owner, not a port-time one, and the op's shape/stride RTAs are recomputed per program anyway, so
  the win looks small.
- **Capabilities not yet on this concept that the op would benefit from: none.** Single program, no op-owned
  tensors, no op-owned `GlobalSemaphore`s (the op has no semaphores at all), no per-coord variation.
- **Friction with the concept fit: none.** The two `create_descriptor` bodies mapped onto
  `create_program_artifacts` without any structural pressure.

## Handoff points

### 1. Three latent defects in `running_statistics_kernel.cpp` — ops team, **not** fixed in this diff

Tagged "op correctness: non-default compute path." Carried forward verbatim from the audit's *Misc anomalies*
1–3, re-confirmed against the current tree during the port, and **ported byte-for-byte unchanged** (the
recipe's "do not 'fix' the legacy kernel" rule). Recorded here because the port is the natural moment to
route them, and because a reviewer reading the new kernel will see the odd FIFO shape and should know it is
pre-existing.

All three are **masked today**: `batch_norm_utils.cpp:27` sets `default_fp32_acc = true`, so the SFPU sibling
is the default compute source. `running_statistics_kernel.cpp` is reached only when a caller explicitly
passes `fp32_dest_acc_en = false` **and** every tensor is bfloat16.

1. **`push_back` with no matching `reserve_back`** — `device/kernels/compute/running_statistics_kernel.cpp:56-58`.
   The kernel packs and pushes the output tile without ever calling `dfb_out0_obj.reserve_back(1)`. The
   writer does `wait_front` / `pop_front` on that DFB
   (`device/kernels/dataflow/writer_running_statistics.cpp:144,148`), so the FIFO has a consumer but no
   producer-side back-pressure: with `num_tiles` greater than the DFB depth (2 entries) the producer can
   outrun the consumer and overwrite unread data. The SFPU sibling does it correctly
   (`running_statistics_sfpu_kernel.cpp:113` reserves, `:312` pushes).
2. **Nested `tile_regs_acquire()`** — `running_statistics_kernel.cpp:39-57`. The outer
   `tile_regs_acquire()` / `commit` / `wait` / `release` bracket wraps calls to
   `sub_tiles_to_cb` / `mul_tiles_to_cb` / `add_tiles_to_cb`, **each of which runs its own full
   acquire→commit→wait→release cycle** (`ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp:86-93` and
   siblings). The `pack_tile(0, dfb_out0)` at `:56` therefore packs whatever DST reg 0 holds after the last
   inner release, not a value this bracket produced.
3. **Output packed from an undefined DST when both running stats are absent** —
   `running_statistics_kernel.cpp:56`. With `old_running_mean_has_value == false` and
   `old_running_var_has_value == false` both `if constexpr` blocks are elided, yet `pack_tile(0, dfb_out0)`
   still runs and pushes a tile of uninitialised DST content to the op's output tensor. The SFPU variant has
   the same structural hole (`running_statistics_sfpu_kernel.cpp:113` reserves and `:312` pushes with nothing
   packed in between). Reachable: `ttnn::batch_norm` calls `ttnn::prim::running_statistics` whenever
   `training == true` (`batch_norm.cpp:130-133`) and the nanobind docstring makes both running-stat tensors
   optional in training mode.

### 2. Boundary-rule assumption violations

**None.** No call site outside the op directory required a `sem::name` or `tensor::name` handle. Every
out-of-op callee this op invokes takes either a raw `uint32_t l1_write_ptr` or a `uint32_t cb_id`, both of
which `dfb::name` crosses via its `constexpr operator uint32_t()`. Concretely, the three donor headers were
consumed unchanged:

| Donor | Call sites | What crosses |
|---|---|---|
| `eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` | `reader_batch_norm.cpp:47,49`, `writer_batch_norm.cpp:81,83,89,91,105,107,120,122`, `reader_running_statistics.cpp:63,65`, `writer_running_statistics.cpp:95,97,124,126` | `dfb.get_write_ptr()` — a raw L1 address, no handle |
| `ttnn/kernel/dataflow/cb_fill_helpers.hpp` | `reader_running_statistics.cpp:56` | `dfb::one` → `uint32_t cb_id` |
| `ttnn/kernel/compute/dest_format_helpers.hpp` | `batch_norm_kernel.cpp:50,57,86,98,103,116,121`, `running_statistics_kernel.cpp:44-53`, `running_statistics_sfpu_kernel.cpp` (throughout) | `dfb::name` (or a `constexpr` alias of it) → `uint32_t icb` |

### 3. Kernel-lib gaps

**None.** No shared helper or LLK was incompatible with Metal 2.0 binding semantics, and **no file outside
the op's own directory was modified.**

### 4. Framework gaps

**None.** Every framework capability the port needed was on this branch: `ProgramSpecFactoryConcept`,
`ProgramArtifacts`, `TensorBindingToken`, the `DFBAccessor → uint32_t` conversion, `AddRuntimeArgsForNode`,
the `ttnn::create_{reader,writer}_datamovement_config` / `ttnn::to_compute_hardware_config` helpers, and
`ComputeGen1Config::unpack_modes`. No audit-time UNSUPPORTED flag existed to bite (all four Appendix A
entries were `N/A`).

### 5. Removed pybind surface

**None** — see *Device-op-class edits* above.

## Successes

- **[Two-toucher DFB → assign 1P+1C](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
  and its "re-derive, don't transcribe" clause fired correctly, and produced full agreement.** I ran the
  endpoint census myself from the kernel bodies rather than transcribing the brief's table, over all
  config-selected paths (weight/bias present-absent × running-stat present-absent × typecast on/off × SFPU /
  FPU compute source). Every row matched the brief, including the three cases where a DFB's disposition
  *flips with config* — batch-norm `output` (`device/batch_norm_program_factory.cpp:206-217`, 1:1 without
  typecast, self-loop with) and running-statistics `updated_running_mean` / `updated_running_var`
  (`device/running_statistics_program_factory.cpp:243-254`, same flip). Having to derive those independently
  is what surfaced the clean way to express them (next bullet); a transcription would have hidden it.
- **The "sync-free / single-ended → self-loop" hard gate stopped a wrong call.** My first instinct on
  batch-norm `temp_1` was to bind it conditionally, since it is untouched when weight *and* bias are both
  absent. The gate's *count the distinct kernels* framing plus the brief's runtime-ternary warning made clear
  that `dfb_affine_or_out` / `dfb_scaled_output`
  (`device/kernels/compute/batch_norm_sfpu_kernel.cpp:43-44`) can resolve to `temp_1` on *any* path at
  runtime, so it must stay bound unconditionally and self-looped
  (`device/batch_norm_program_factory.cpp:410-413`). Narrowing it would have been a spec/kernel mismatch that
  compiles.
- **The recipe's insistence on a before/after `hw_config` diff paid off as a *confidence* win, and the diff
  turned out to be far more mechanisable than the section implies.** Because both runs' JIT artifacts sit in
  `~/.cache/tt-metal-cache/<build>/kernels/<kernel>/<config-hash>/`, the comparison can be done on the
  **generated device config** instead of by eyeballing the factory: 44 distinct configs across the four
  compute sources, all byte-identical (table in *Verification*). That is a much stronger statement than "I
  translated each field carefully," and it covers exactly the two silent fields the section warns about. This
  seems worth writing into the recipe as a concrete technique — see the Gap entry below.
- **The recipe's `unpack_modes` warning ("three things change at once") is calibrated correctly, and reading
  the validator up front was the right move.** Before writing a line of the hardware config I read
  `tt_metal/impl/metal2_host_api/program_spec.cpp:921-1072` (per the recipe's *go to the headers first*
  advice, applied to the validator) and checked all four rules against every config this op can take. That
  turned up the one case I would otherwise have got wrong by reasoning from the recipe's prose alone: this op
  can legitimately run `fp32_dest_acc_en = true` with **all-bfloat16** DFBs (`batch_norm_utils.cpp:27` makes
  `fp32_dest_acc_en` default-true regardless of dtype), which reads like the "≤16-bit format +
  UnpackToDest → REJECTED on Gen1" row. It is not: that row is gated on `enable_32_bit_dest == false`, and
  the entries here only exist when it is true. Had I "defensively" dropped those entries to avoid the
  imagined rejection, I would have silently flipped the precision/perf tradeoff on the op's *default* path
  with no test signal — exactly the failure the section warns about.
- **[Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)**,
  specifically its *"Promote a CTA gate to a define"* paragraph, named the exact trap this op sets four
  times over. The legacy writers gate their optional-tensor reads with `if constexpr (weight_has_value)` on a
  **CTA** — which still name-looks-up `tensor::weight` in the discarded branch, and that token does not exist
  when the host declares no `TensorParameter`. Promoting each gate to a `#define` +`#ifdef`
  (`device/kernels/dataflow/writer_batch_norm.cpp:53-62,103-131`,
  `device/kernels/dataflow/writer_running_statistics.cpp:47-56,81-143`) was the fix, and the paragraph's
  "watch the emission target" note is why I checked that only the *writers* reference those tensors (the
  compute kernels don't bind tensors at all, so they keep plain CTAs).
- **The [scope-discipline](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#scope-discipline)
  "improvements are routed, not suppressed" framing held the diff tight** against three separate pulls: the
  `running_statistics_kernel.cpp` defects above (obvious one-line fixes, left alone), the
  `b_num_tiles_per_cb` no-op alias (`device/batch_norm_program_factory.cpp:205`, kept), and the
  loop-invariant `packed_scalar_eps` recomputed per node
  (`device/batch_norm_program_factory.cpp:110-112`, kept inside the loop). All three are one edit away and
  none is in this diff.

## Friction

### Gaps

- **The recipe and the brief disagree on which data-movement helper to call, and neither flags the other.**
  The brief's hardware-config item 0 says to use the Metal-layer
  `CreateReaderGen1DataMovementConfig()` / `CreateWriterGen1DataMovementConfig()`; the recipe's
  [Data movement kernels](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#data-movement-kernels)
  section says to prefer the TTNN wrappers `ttnn::create_reader_datamovement_config(arch)` /
  `create_writer_datamovement_config(arch)` *because they also supply the Gen2 branch*, and explicitly notes
  that the migration guide's use of the Metal-layer names is "identical on Gen1." **Resolved in favour of the
  recipe** (it outranks the brief on port mechanics, and the reasoning is substantive rather than stylistic):
  `device/batch_norm_program_factory.cpp:320,381` and
  `device/running_statistics_program_factory.cpp:340,414`. The two are byte-identical on Gen1, so nothing is
  at stake here — but an auditor writing a brief has no signal that naming the Metal-layer helper will read
  as a contradiction. **Suggestion:** have the audit doc tell auditors to name the *TTNN* helper for a TTNN
  op, or to defer the choice to the recipe rather than naming a function at all.

- **Nothing in the docs covers a legacy factory that assigns "which CB the writer drains" on the host — the
  shape that makes half this op's conditional bindings evaporate.** Legacy sets
  `writer_output_cb = needs_output_typecast ? c_9 : c_2` (and `writer_updated_m_cb` / `writer_updated_v_cb`
  likewise) and hands the result to the writer as one CTA. The natural-but-wrong port is to bind both DFBs
  and `#ifdef` the writer, which is what I started to do. The clean expression is to keep **one**
  `DFBBinding` under the writer's own accessor name and let the *host* vary only its `dfb_spec_name`
  (`device/batch_norm_program_factory.cpp:373-376`,
  `device/running_statistics_program_factory.cpp:454-461`) — the writer kernel then needs no `#ifdef` at all
  and its diff is a single line. This is arguably a corollary of "accessor names are kernel-local," but that
  point is made in the migration guide only as an aside ("independent of the producer's name") and never
  connected to config-varying targets. The
  [Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
  entry's "Two forms — kernel-side and host-side" paragraph is the closest match and is about a *different*
  thing (one CB under two names, not two CBs under one name). **Suggestion:** add a short pattern —
  *"Config-selected DFB target: vary the `dfb_spec_name`, not the kernel"* — with the sibling note that the
  *other* end (a kernel that names **both** the staging and the writer-facing DFB, as these compute kernels
  do on the typecast path) is where the `#ifdef` genuinely belongs.

- **The conditional-binding pattern has no worked shape for "the token is needed as a function argument on
  the taken path only."** The pattern's example gates a `constexpr` alias and the *uses*, which works when the
  uses are statements. Here the conditionally-bound DFB (`output_final`, `writer_updated_mean`,
  `writer_updated_var`) is passed as a plain `uint32_t` **parameter** to a template whose body elides every
  use under `if constexpr (NeedsTypecast)` — so the argument must still be *spellable* on the untaken path.
  What works is an `#ifdef`-selected alias whose `#else` arm names an always-bound DFB and is inert
  (`device/kernels/compute/batch_norm_sfpu_kernel.cpp:198-209`,
  `device/kernels/compute/running_statistics_sfpu_kernel.cpp:45-66`) — which happens to reproduce legacy's
  own `writer_output_cb = output_tensor_cb` assignment exactly, so it is faithful rather than a workaround.
  It took a while to convince myself the inert `#else` arm was not the "bind unconditionally to dodge the
  name-lookup problem" anti-pattern the recipe's stop-signal list warns about. It isn't — the *binding* stays
  conditional and no extra L1 is allocated; only a compile-time integer is aliased. **Suggestion:** state
  that distinction explicitly in the pattern (conditional *binding* is the invariant; a conditionally-aliased
  *handle* whose untaken arm names an already-bound DFB is fine), because the stop-signal wording reads as
  though any `#else` fallback is suspect.

- **"Diff before against after" is asked for but never operationalised, and there is a mechanical way to do
  it.** The [Hardware configuration](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#hardware-configuration)
  section says the before/after comparison "is yours to make" and leaves it at that, which in practice invites
  a careful re-read of one's own field mapping — the weakest possible check, since a mis-mapping and its
  justification come from the same reasoning. The stronger check is to diff the **JIT-generated**
  `chlkc_descriptors.h` between a pre-port and a post-port test run: both sets of artifacts coexist under
  `~/.cache/tt-metal-cache/<build-hash>/kernels/<kernel-name>/<config-hash>/`, and
  `unpack_dst_format[]` / `DST_ACCUM_MODE` / `DST_SYNC_MODE` / `MATH_FIDELITY` / `APPROX` are exactly the
  fields `hw_config` controls. The one wrinkle worth documenting is that post-port DFB ids are
  framework-assigned, so the arrays are **permuted** relative to the legacy CB indices and must be
  un-permuted via the variant's own `kernel_bindings_generated.h` before comparing (that mapping is right
  there in the artifact directory, so it is cheap). **Suggestion:** add this as a short "how to actually do
  the diff" subsection — it converts the section's most safety-critical instruction from a discipline into a
  check, and it costs nothing because the porter has already run the tests twice.

- **The dead-RTA guidance has no rule for the count constants that encode the dead args.** The brief tells
  the porter that dropping unread RTAs is behaviour-preserving, and mentions that the idle-core zero-fill
  constants "shrink with them." In Metal 2.0 they don't shrink — they **disappear**, because idle nodes now
  receive the same *named* arguments as working nodes (the framework requires every named RTA on every node
  the kernel runs on), so there is no count to zero-fill against. That is a slightly larger restructure of
  the legacy per-core loop than "drop two args" suggests: the legacy `continue`-and-zero-fill branch becomes
  an `in_work_group` flag with the values zeroed at the emission point
  (`device/batch_norm_program_factory.cpp:100-152`,
  `device/running_statistics_program_factory.cpp:105-150`). Worth one line in the recipe's `KernelRunArgs`
  bullet, since every op with an `all_device_cores` placement and a `split_work_to_cores` work split hits it.

### Confusion

- **`get_arg(args::x) == 1` on a CTA reads like it might not be constant-folded.** The legacy kernels write
  `constexpr uint32_t weight_has_value = get_compile_time_arg_val(0) == 1;`. The named form
  `constexpr uint32_t weight_has_value = get_arg(args::weight_has_value) == 1;` is fine — `get_arg(CtaVal<T>)`
  is `constexpr` and the header says so — but the recipe's worked examples only ever show
  `constexpr auto x = get_arg(args::x);` with no expression around it, so it took a header read
  (`tt_metal/hw/inc/experimental/kernel_args.h:82-85`) to be sure a comparison in the initialiser still
  yields a compile-time constant. One example with a predicate would settle it.
- **Which generated header carries what, and that you must not include either, is stated in three places but
  never next to the "add exactly two headers" rule** that a porter is actually working from. I re-read the
  whitelist's rule preamble, the migration guide's *Kernel Argument Retrieval Syntax* section, and the
  recipe's build-failure table before being confident that `experimental/kernel_args.h` alone is enough to
  get `dfb::`, `tensor::` **and** `args::` (it is — the framework injects `kernel_bindings_generated.h` and
  `kernel_args_generated.h` via `<kernel_includes.hpp>`). A one-line parenthetical on whitelist rule 3 would
  save that round trip.
- **`Table` vs `Group` bit once, exactly where the recipe predicted.** I reached for `push_back` on a
  `CompileTimeArgs` while building the per-compute-source CTA table
  (`device/batch_norm_program_factory.cpp:437-446`). The recipe's *"`Table`s are maps, not vectors"* note
  covers this precisely and the fix was `insert({name, value})`; noting it only to confirm the warning earns
  its place — the two types sit side by side in the same struct and look interchangeable.

## Open items for downstream

- **Shared kernel touches: none.** The op owns all 8 kernel sources and no factory outside this directory
  binds any of them (census re-run during the port: the only out-of-directory hits are in
  `ttnn/ttnn.egg-info/SOURCES.txt`, a build artifact). **No `_metal2` fork was reused and none was created**,
  no pointer comment was added to any peer file, and there is no sunset list. The two factories share no
  kernel with each other, so the intra-op case does not apply either. Nothing outside
  `ttnn/cpp/ttnn/operations/normalization/batch_norm/` was modified by this port.
- **The two `ttnn/cpp/ttnn/kernel/` donors are still `CircularBuffer`-native internally** while every caller
  in this op is now DFB-native and Metal-2.0-bound: `cb_fill_helpers.hpp:19` constructs
  `CircularBuffer cb(cb_id)`, and `dest_format_helpers.hpp:14` includes `api/dataflow/circular_buffer.h` and
  builds `CircularBuffer` objects at `:78-80,115-117,152-154`. This is invisible at the `uint32_t cb_id`
  boundary and cost this port nothing, but it is a tidy-up for the kernel-pool owners and it will keep
  showing up as ops in this family migrate. `dest_format_helpers.hpp` has 4 consumers, `cb_fill_helpers.hpp`
  has 1 (this op).
- **`fill_tile_utils.hpp` is broadly shared and must not be modernised opportunistically** — ~35 kernel files
  across `eltwise/binary_ng`, `eltwise/ternary` and this op include it. Every function this op calls takes a
  bare `uint32_t l1_write_ptr`, so it needs no change for a Metal 2.0 port; a future porter tempted to
  "upgrade" it would break every co-borrower. Recorded so the next porter in this family doesn't re-derive it.
- **Sibling-op carry-over:** the *config-selected DFB target* shape (host varies `dfb_spec_name`, kernel is
  unchanged) and the *inert `#else` handle alias* shape (for a conditionally-bound DFB passed as a function
  argument) should both transfer directly to any op with an optional typecast staging buffer. Ops with
  optional input tensors will hit the *CTA-gate → `#define`* promotion for `tensor::` tokens; that is the
  single most mechanical-looking change in this port that is actually load-bearing.
- **Test coverage notes.**
  - The confirmed baseline (`tests/ttnn/unit_tests/operations/fused/test_batch_norm.py` +
    `test_batch_norm_program_cache.py`) does cover the non-default FPU compute path — `test_batch_norm.py:577`
    passes `fp32_dest_acc_en=False`, which with bfloat16 tensors selects `batch_norm_kernel.cpp` /
    `running_statistics_kernel.cpp`. Worth knowing that this is the *only* handle on those two sources, and
    that it is one `parametrize` entry: if it were ever dropped, both FPU kernels would go untested.
  - **No C++ gtest exists for this op** (`--gtest_filter='*BatchNorm*'` matches nothing in any
    `unit_tests_ttnn*` binary), so verification is pytest-only. A small gtest would give the next porter of a
    normalization op a fast pre-flight signal; not added here (out of port scope).
  - The `output`-typecast path (compute packs FP32, typecasts down to the output dtype) is exercised by the
    mixed-dtype parametrizations, but I did not find a test that pins *only* that path, so a regression in
    the `c_9` / `c_12` / `c_13` staging bindings would surface as a general numerics failure rather than a
    targeted one.
- **Doc-evolution suggestion beyond the Gap entries:** the three friction items above (config-selected DFB
  target, inert `#else` alias, idle-node named-RTA restructure) are all instances of one missing theme — *what
  happens to a legacy factory's host-side "which resource" ternaries under named bindings*. A single short
  catalog entry covering that theme would probably subsume all three.
