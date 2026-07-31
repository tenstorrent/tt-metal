# Port Report — `experimental/unary_backward/gelu_backward`

## Outcome

**`PORTED`** — the op's single factory (`GeluBackwardProgramFactory`, its only `program_factory_t`
variant) is converted to `ProgramSpecFactoryConcept`, together with all four kernel entry points it
binds: both runtime-selectable compute sources converted in place, and the two borrowed dataflow
kernels converted as new `_metal2` forks. No factories are left on the legacy concept.

**Verification caveat — stated plainly:** at the invoker's explicit request this port was **not built
and its tests were not run** by the porter; no compile or hardware evidence backs the change yet. The
recipe's `PORTED` definition includes "and its tests pass", which is therefore **not yet
established**. The anti-pattern self-audit checklist was run in full and passes (see below), and
every API shape used was checked against the declaring header rather than a precedent, but that is
static review, not a green build. Commands to exercise the impacted code are at the end of this
report. If either the build or a previously-passing test fails, this outcome should be revisited
before the PR is accepted.

## Provenance

`git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` run
from this checkout root **prints nothing** — the recipe docs are not tracked in the port checkout, so
the version cannot be pinned from here. The docs actually followed live in a sibling checkout
(`/localdev/edwinlee/Port_Recipe`), where the same command yields the line below; it matches the
brief's recorded line exactly, so audit and port ran against the same doc revision.

- **Recipe docs (this port):** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit chose. `GeluBackwardProgramFactory::create_descriptor`
became `create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`
(`device/gelu_backward_program_factory.hpp:13`, `device/gelu_backward_program_factory.cpp:18`).
`op_owned_tensors` is left defaulted-empty; the factory allocates no device tensors of its own.
No disagreement with the audit arose at any point.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — the op never had one, so it was already on the
  default reflection-based hash.
- **Pybind entry points removed:** none. `gelu_backward_nanobind.cpp` exposes only the user-facing
  `ttnn.experimental.gelu_bw` op, never `create_descriptor`, so no pybind line referenced a symbol
  the port removed and the public Python surface is unchanged.
- **Other:** two now-dead `#include <tt-metalium/program_descriptors.hpp>` lines were removed —
  `device/gelu_backward_program_factory.hpp:8` (forced: the factory's return type changed) and
  `device/gelu_backward_device_operation.hpp:11` (the include existed only to declare the legacy
  factory's `ProgramDescriptor` return type). The latter is a one-line include removal in the
  device-op class file, taken under kernel-side whitelist rule 1's "sweep both sides … unused
  `#include`s" instruction rather than as a freelance edit; nothing else in that file changed, and
  it declares no other user of that header.

### Open items

- **Relaxation candidates: none applied, and none identified.** `grep -rn 'ArgConfig::Runtime'` over
  the op directory and both borrowed dataflow kernels returns zero hits, so the migration guide's
  `eltwise`-family heads-up (which predicts `dynamic_tensor_shape = true` for this family) does **not**
  apply here — the legacy op declared no relaxation, and strict matching is the faithful port. Worth
  recording that the heads-up's prediction missed for this op, so the next `eltwise` porter runs the
  grep rather than assuming.
- No capability outside `ProgramSpecFactoryConcept` was wanted: single program, no op-owned tensors,
  no op-owned `GlobalSemaphore`s. The concept fit this op with zero friction.

## Handoff points

### 1. `unpack_modes` for a Float32 `grad_output` with a narrower output — a behavior change the port deliberately did not paper over

*Owner: the op owner (dtype contract) plus the Metal 2.0 compute-config owners.*

- **Site:** `device/gelu_backward_program_factory.cpp:159-171` (the ported `unpack_modes`
  derivation); legacy source at `device/gelu_backward_program_factory.cpp:114-116` pre-port.
- **What legacy did:** requested `UnpackToDestFp32` for CBs `c_0` (grad_output) and `c_1` (input)
  **unconditionally**, regardless of dtype. The legacy lowering honours the request only when the
  buffer's format is `Float32` (`tt_metal/jit_build/data_format.cpp:213-214` gates it on
  `src_format == DataFormat::Float32`); for every narrower format the entry is inert.
- **What the port does:** emits `UnpackMode::UnpackToDest` only for a DFB whose data format is
  `Float32`, which reproduces the legacy unpack path byte-for-byte in every case.
- **The residual case:** when `grad_output` is `Float32` while `input`/`output` is a ≤16-bit dtype,
  `enable_32_bit_dest` is false (it is derived from the *output* format) yet `GRAD_OUTPUT_DFB` is
  `Float32`, so the Metal 2.0 validator rejects the program
  (`tt_metal/impl/metal2_host_api/program_spec.cpp:1024-1031`: "A 32-bit datum cannot be unpacked
  into a 16-bit Dest register"). Legacy silently accepted this and configured exactly that
  32-bit-into-16-bit unpack — i.e. the op was genuinely misconfigured for that combination.
- **Why nothing was invented:** forcing `enable_32_bit_dest = true`, or suppressing the entry, would
  each change the op's numerics/perf, which is out of scope for a port. Per the recipe's treatment of
  the analogous legacy-tolerated DM misconfiguration, this is reported rather than worked around.
- **Reachability:** not reached by any test or documented dtype. The device op requires
  `output_dtype == input.dtype()` (`device/gelu_backward_device_operation.cpp:29-33`) but places **no**
  constraint on `grad_output`'s dtype relative to `input`'s; the nanobind docstring documents
  BFLOAT16 only, and every test in the confirmed set is BFLOAT16 for both operands. So the change is
  latent, not a regression the suite can see.
- **Ask:** decide whether `grad_output.dtype() == input.dtype()` should be validated outright (which
  would make the case unreachable and the port's residual moot), or whether the op is meant to
  support the mix — in which case the legacy unpack configuration for it needs a deliberate fix.

### 2. No other handoffs

No boundary-rule assumption violation (no out-of-op call site needed a `sem::` or `tensor::`
handle — the op has no semaphores, and both `tensor::` tokens are consumed inside the forked kernels
themselves). No kernel-lib gap: every out-of-op callee the ported kernels reach
(`unary_op_init_common`, `copy_tile`, `pack_tile`, `gelu_derivative_tile`, `noc.async_read` /
`async_write`) either takes a `uint32_t` CB id that `dfb::name` converts to implicitly, or is a
framework primitive the porter consumes directly. No framework gap bit. No removed pybind surface.
No capitulation.

## Successes

- **[Caution: Porting a shared kernel](../shared/port_patterns.md#caution-porting-a-shared-kernel) —
  the "sunset list, not authorization" warning fired correctly, twice.** The brief enumerates the
  writer's ~34 binding factories and the reader's three. Read as a to-do list, that enumeration
  invites converting both files in place and "just fixing up" the binders; the catalog's rung 3
  explicitly names that reading as the failure the rung exists to prevent, and the brief repeats the
  phrase per kernel. Both files were forked instead
  (`.../eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id_metal2.cpp`,
  `.../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`), with the
  originals receiving only the prescribed pointer comment. The rung-1 check was run **locationally**
  (`ls` of each original's directory) as the catalog insists, not as a tree-wide grep — which matters
  because a tree-wide grep for `*_metal2*` returns hits from the out-of-bounds `experimental/quasar/`
  tree that are not siblings of anything.
- **"Name the bindings for the kernel, not for your op."** Left to instinct the forks would have
  carried this factory's vocabulary — `dfb::cb_grad_out`, `tensor::src0_buffer` — which the next
  consumer of a *binary reader* or a *unary writer* could not reuse and is not permitted to rename.
  The instruction redirected the names to the kernels' own vocabulary (`dfb::in0` / `dfb::in1` /
  `dfb::out`, `tensor::src0` / `tensor::src1` / `tensor::dst`), taken from the legacy kernels' own
  locals. This is the single highest-leverage line in the brief for a port whose substance is two
  forks.
- **[Compiler options](../port/metal2_port.md#compiler-options) rule 2 caught a silent perf loss that
  looked like a no-op.** `grep -n opt_level device/gelu_backward_program_factory.cpp` returns nothing,
  which reads as "the op expressed no preference, so there is nothing to carry over." The recipe's
  insistence that an absent `KernelDescriptor::opt_level` still *resolves* — `O3` for a
  `ComputeConfigDescriptor`, confirmed at `tt_metal/impl/program/program.cpp:456` — is what put the
  explicit `KernelBuildOptLevel::O3` on the compute `KernelSpec`
  (`device/gelu_backward_program_factory.cpp:196`). Without it the port would have silently dropped
  the compute kernel a level, with nothing in any test to catch it.
- **[Pattern: Unity-build hygiene](../shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols).**
  The natural place for `READER` / `WRITER` / `INPUT` / `OUTPUT` typed-name constants is an anonymous
  namespace at file scope, per the migration guide's typed-constants examples. TTNN enables unity
  builds (`ttnn/CMakeLists.txt:96,197,286`) and these identifiers are maximally generic, so that
  would have been a latent duplicate-symbol collision with any other ported factory in the same
  target. They are declared function-local instead
  (`device/gelu_backward_program_factory.cpp:32-44`), which sidesteps the hazard entirely.
- **[CB→DFB API whitelist](../shared/cb_dfb_api_whitelist.md) §A/§B answered both metadata rewrites
  without guesswork.** `get_tile_size(cb_id_in0/in1)` → `dfb0.get_tile_size()` / `dfb1.get_tile_size()`
  (§A) and `get_local_cb_interface(cb_id_out).fifo_page_size` → `dfb.get_entry_size()` (§B). The
  second is the one a porter would plausibly improvise (`get_page_size()` is the tempting guess, and
  does not exist); having the table meant no invented spelling.
- **Endpoint re-derivation instruction.** Re-running the kernel-touch census rather than transcribing
  the brief's table cost a few minutes and confirmed three plain 1P+1C DFBs with no ambiguity — and,
  usefully, confirmed the *absence* of the hidden-co-filler shapes (no `get_write_ptr()` /
  `fifo_wr_ptr` write, no cursor mutation, no semaphore) that would have forced a flag. Cheap
  verification of a claim that would have been expensive to get wrong.

## Friction

### Gaps

- **The `unpack_modes` translation table is stated unconditionally, but legacy semantics are
  Float32-gated — and the literal reading produces a spec the Gen1 validator rejects.** This was the
  single largest time sink in the port and the only genuinely non-mechanical decision in it.
  [Hardware configuration → Compute kernels](../port/metal2_port.md#compute-kernels) says: "derive
  its value from the legacy vector (`Default` → `UnpackToSrc`, `UnpackToDestFp32` → `UnpackToDest`);
  do not guess", and the migration guide's
  [DataflowBufferSpec](../shared/migration_guide.md#dataflowbufferspec) and troubleshooting table say
  the same, with no format condition anywhere. Applied literally to this op — whose legacy vector
  marks `c_0` and `c_1` `UnpackToDestFp32` regardless of dtype — that yields
  `unpack_modes = {{GRAD_OUTPUT_DFB, UnpackToDest}, {INPUT_DFB, UnpackToDest}}` always, which for the
  op's *only shipped and only tested* dtype (BFLOAT16, so `enable_32_bit_dest = false` and both DFBs
  ≤16-bit) is rejected outright by
  `tt_metal/impl/metal2_host_api/program_spec.cpp:1032-1039`. The op would not run at all.
  The resolution required reading the legacy lowering
  (`tt_metal/jit_build/data_format.cpp:199-221`) to discover that `unpack_to_dest_mode[i]` is
  consulted only inside `if (src_format == DataFormat::Float32 && …)`, making the entry inert for
  every narrower format — so the behavior-preserving translation is to emit the entry **only where
  the legacy request was live.** *Suggested doc fix:* state the gate in the recipe's translation
  table — "translate `UnpackToDestFp32` → `UnpackToDest` **only for a DFB whose data format is
  `Float32`**; for any narrower format the legacy entry was inert, so omit it (`UnpackToSrc`)" — and
  note that a legacy vector setting `UnpackToDestFp32` unconditionally across dtypes is common and is
  *not* a signal to transcribe it unconditionally. Without this, the natural failure is loud (a
  validator `TT_FATAL`), but the natural *fix* a porter would then reach for is to flip
  `enable_32_bit_dest` or drop the entry blindly — both silent behavior changes.
- **`TensorParameter`'s relaxation field is misnamed in both docs.** The recipe and the migration
  guide refer to `TensorParameter::advanced_options` holding `dynamic_tensor_shape` /
  `match_padded_shape_only` ([migration guide — TensorParameter](../shared/migration_guide.md#tensorparameter),
  [ttnn_factory — Tensor-arg matching](../shared/ttnn_factory.md#tensor-arg-matching--keep-strict)).
  The actual field is `TensorParameter::relaxations`, of type `TensorSpecRelaxations`
  (`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp:45`, defined in
  `tensor_spec_relaxations.hpp`). Not load-bearing for this port (no relaxation applied), but a
  porter who *does* need one and follows the doc writes a field name that does not compile. Both
  bool names are correct; only the containing field is wrong.
- **The `hw_config` helpers' namespace is never stated.** [Hardware configuration → Data movement
  kernels](../port/metal2_port.md#data-movement-kernels) shows
  `.hw_config = create_reader_datamovement_config(device->arch())` bare. The helpers are in namespace
  **`ttnn`** (`ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp`),
  not `tt::tt_metal::experimental` like everything else in the spec block — easy to mis-assume when
  the surrounding factory body has `using namespace tt::tt_metal::experimental;`. One qualifier in
  the example would fix it. Same for `to_compute_hardware_config`.
- **`KernelSpec::source` and a non-literal path.** The header notes "(A string literal binds directly
  to the path variant alternative.)" but says nothing about a `std::string` or
  `std::filesystem::path` lvalue, and the variant's other alternative (`SourceCode`) is an aggregate
  with a single `std::string` member — which raises a real question about whether the variant's
  converting constructor is ambiguous for a `std::string` argument. It is not (the converting
  constructor's array-copy-init test excludes `SourceCode`, which has no converting constructor), but
  establishing that required reading the C++20 variant rules rather than the docs. The port sidesteps
  the question by holding the runtime-selected compute path as a `const char*`
  (`device/gelu_backward_program_factory.cpp:179`). *Suggested doc fix:* one line saying which
  spellings are safe for a computed source path.
- **The Provenance command assumes the recipe docs are in the port checkout.** They were not — the
  docs live in a sibling clone, so the prescribed command printed nothing and the recipe's fallback
  ("record that fact instead") loses the version. Since a matching hash *was* obtainable from the doc
  checkout, the fallback could usefully say: if the docs are in a separate checkout, run the command
  there and say so.

### Confusion

- **Rule 1's stale-CB-comment sweep collides with rule 8's preserve-comments rule, and the recipe
  doesn't say which wins — least clearly for a *new fork* file.** The forked reader inherits a
  provenance header reading "…to demonstrate the new ability to keep the `CircularBufferConfigs`
  continuous during dispatching. See the use of `CBIndex::c_2` below." Rule 1 says a post-port grep
  for `CircularBuffer` should return zero hits in code and names "stale comments referencing CB" for
  the sweep; rule 8 says preserve comments, deletion is not sanctioned, err toward preserving.
  Compounding it, the `CBIndex::c_2` pointer was **already** dangling pre-port (the file uses only
  `c_0` and `c_1`), so the comment was stale before the port touched it. Resolution taken: keep both
  live facts (that the file is a temporary copy from `datamovement/binary/device/`, and that it is
  expected to be deleted or refactored when broadcasting is properly supported) and drop only the
  clause naming constructs the fork no longer contains — rule 8's sanctioned "slight tweak to align
  an existing comment with the line you're forced to change." The original file's copy of the comment
  is untouched. *Suggested doc fix:* one sentence — when a comment's *referent* is a construct the
  port removes, trim the referent and keep the information; rule 1 wins over rule 8 only for the
  dangling clause, never for the surrounding explanation.
- **`TT_KERNEL` exists and the recipe never mentions it.** `experimental/kernel_args.h` — the one
  header the whitelist tells you to add — documents a `TT_KERNEL` macro marking "the named-arg entry
  point; the JIT generates `kernel_main()` from its signature", which reads as *the* Metal 2.0 entry
  style and made the hand-written `void kernel_main()` in every recipe example look like a legacy
  leftover. Resolving it meant reading `tt_metal/jit_build/genfiles.cpp:331-373`, which confirms a
  hand-written `kernel_main()` is "fully backward compatible" and that `TT_KERNEL` is an optional
  alternative. A porter who guessed the other way would have restructured four kernel entry points
  for no reason. *Suggested doc fix:* one line in the kernel-side whitelist saying `kernel_main()`
  stays as-is and `TT_KERNEL` is out of scope for a port.

## Open items for downstream

- **Shared kernel touches (the coordination signal / fork sunset checklist).** Two forks created;
  both are **rung 2 (created the fork)**, and the pointer comment landed in each legacy original.
  - **(a)** `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
    → **(b)** fork created at `…/reader_binary_interleaved_start_id_metal2.cpp`; pointer comment added
    to the original (its only change). **(c)** Remaining unmigrated consumers:
    `ttnn/cpp/ttnn/operations/eltwise/unary_backward/gelu_bw`,
    `ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw`, and the non-factory consumer
    `tests/ttnn/unit_tests/gtests/test_generic_op.cpp` (which references the path as a string; it is
    a filename-grep hit that is *not* a program factory, noted so the next porter doesn't miscount it
    as one). Fork bindings: `dfb::in0`, `dfb::in1`, `tensor::src0`, `tensor::src1`; CTA
    `block_or_width_sharded`; RTAs `num_tiles`, `start_id`, `block_height`, `block_width`,
    `num_cores_y`; gated on the inherited `IN0_SHARDED` / `IN1_SHARDED` defines (this factory defines
    neither).
  - **(a)** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
    → **(b)** fork created at `…/writer_unary_interleaved_start_id_metal2.cpp`; pointer comment added
    to the original (its only change). **(c)** Remaining unmigrated consumers: **all ~34 non-quasar
    factories** enumerated in `METAL2_PREPORT_AUDIT.md` → Heads-ups (tilize,
    tilize_with_val_padding, reduction/generic, reduction/prod, transpose, slice, concat, copy,
    permute, reshape_on_device, bcast, typecast, embedding, examples, attn_matmul, nlp_concat_heads,
    kv_cache, `gelu_bw`, `tanh_bw`, …) — this fork's sunset is far off, and it is the fork most
    likely to be hit at rung 1 by the next several ports. Fork bindings: `dfb::out`, `tensor::dst`;
    no CTAs; RTAs `num_pages`, `start_id`; gated on the inherited `OUT_SHARDED` / `BACKWARDS`
    defines (this factory defines neither). Note for rung-1 reusers: because this fork declares no
    CTAs at all, a consumer needing one cannot add it — see the catalog's "a fork that already has a
    consumer is read-only to you."
  - No build-system change was made or needed for either fork: `eltwise/binary/CMakeLists.txt:16` and
    `eltwise/unary/CMakeLists.txt:16` both `file(GLOB_RECURSE kernels device/kernels/*.cpp)`, which
    already cover the new files.
- **`num_cores_y` is an RTA that is really a CRTA.** The reader's `num_cores_y` takes the *same value
  on every node* (`device.compute_with_storage_grid_size().y`), so it belongs in
  `common_runtime_arg_values`. Deliberately **not** converted here: RTA→CRTA changes dispatch
  semantics, which is outside a syntax-swap port. Pick it up in the later name-first / CRTA cleanup
  pass — and note it must be done in the *fork*, so it is a breaking change for every consumer of
  the fork at that time, not a local edit.
- **The forked reader carries dead plumbing this port had to preserve.** `block_height`,
  `block_width` and `num_cores_y` feed only the `block_or_width_sharded` branch, which this factory
  hard-codes off (CTA `0`), so all three are dead for every current consumer while still being read
  unconditionally at the top of `kernel_main`. Dropping them is a shared-kernel change, out of scope.
  Related: the legacy reader's own header comment states it is a temporary copy expected to be
  deleted or refactored once broadcasting is properly supported — the fork inherits that debt, so
  sunset the pair together rather than migrating the fork's remaining consumers onto a file that is
  itself slated for removal.
- **Test-coverage gaps the verification step surfaced but did not act on.** The confirmed test set
  exercises **BFLOAT16 only**, and never a `grad_output` dtype differing from `input`'s — which is
  exactly the combination the Handoff-points item 1 residual affects, so no test can see it either
  way. Two additions would close it: a same-dtype FP32 case (which exercises the
  `enable_32_bit_dest = true` + `UnpackToDest` + required-entry path that BFLOAT16 never reaches), and
  a mixed `grad_output`/`input` dtype case to pin down whether that combination is meant to be
  supported at all. Also worth noting: no test covers the sharded or ROW_MAJOR paths, because the
  device op rejects them outright — so the forks' `IN0_SHARDED` / `IN1_SHARDED` / `OUT_SHARDED` /
  `BACKWARDS` branches are carried unexercised by *this* op and will first be exercised by whichever
  op reuses a fork at rung 1.
- **Per-op carry-over.** `eltwise/unary_backward/gelu_bw` and `eltwise/unary_backward/tanh_bw` are the
  two closest siblings: both bind the same reader, both bind the same writer, and both are
  structurally the same three-kernel interleaved shape. Whoever ports them can go straight to rung 1
  on both forks and should reuse this op's spec shape almost verbatim — including the `unpack_modes`
  format gate above, since `gelu_bw` builds its `unpack_to_dest_mode` vector the same way.

---

## Commands to exercise the impacted code (N150)

Impacted surface is this op only: the two forks are new files with a single consumer, and the two
legacy originals changed by one comment block each, so no other op's behavior is touched.

```bash
# Build (Metal + all TTNN test binaries)
./build_metal.sh --build-tests

# C++ gtests — fastest signal on a broken spec
./build/test/ttnn/unit_tests_ttnn --gtest_filter='*GeluBw*'

# Primary pytest — ULP/accuracy, both approximate="none" (poly) and "tanh" compute paths
pytest tests/ttnn/unit_tests/operations/eltwise/test_gelu_bw_ulp.py -x -v

# Nightly pytest — PCC across shapes incl. non-tile-aligned, both compute paths,
# and the preallocated-output (input_grad=) path; also exercises the program cache
# across its parametrized runs
pytest tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_gelu_fused.py -x -v
```

Excluded as *different* ops despite matching on name: `tests/ttnn/unit_tests/operations/eltwise/test_gelu_bw_main_ulp.py`,
`tests/ttnn/unit_tests/gtests/test_gelu_bw_main_ulp.cpp` and
`tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_gelu.py` all call
`ttnn.gelu_bw` / `ttnn::gelu_bw` — the legacy `eltwise/unary_backward/gelu_bw` op, not
`ttnn.experimental.gelu_bw`. They are still worth running as a **no-regression check on the two
untouched legacy kernels**, since that op binds both originals.
