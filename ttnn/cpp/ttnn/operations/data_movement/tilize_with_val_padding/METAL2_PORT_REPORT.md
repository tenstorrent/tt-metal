# Metal 2.0 Port Report — `data_movement/tilize_with_val_padding`

## Outcome

**`PORTED`** — three of the op's four factories are converted to `ProgramSpecFactoryConcept`
(`create_program_artifacts`): `TilizeWithValPaddingSingleCoreFactory`,
`TilizeWithValPaddingMultiCoreDefaultFactory`, `TilizeWithValPaddingMultiCoreShardedFactory`.

`TilizeWithValPaddingMultiCoreBlockInterleavedFactory` is **left on the legacy
`ProgramDescriptorFactoryConcept`** at the invoker's explicit instruction: it cleared the audit, but
the op owner reports it is subtly broken and must not be ported until that is resolved. The
`program_factory_t` variant is therefore mixed-concept, which is legal — `AllFactoriesValid` requires
each alternative to satisfy *exactly one* concept, not the same one, and the framework adapter
dispatches per factory. The block-interleaved factory and its three kernels
(`reader_unary_pad_multicore_both_dims.cpp`, `writer_unary_interleaved_start_id_wh.cpp`,
`tilize_wh.cpp`) are untouched.

**Verification status — read this before merging.** The port was **not compiled or run**: the
`clang-20` toolchain the build requires is not present in this shell (`/usr/bin/clang++-20` does not
exist here; a sibling checkout's `build.ninja` references it, so builds in this workspace happen inside
a container). `./build_metal.sh --build-tests` fails at CMake configure with *"CMAKE_C_COMPILER:
clang-20 ... not found in the PATH"* before touching any source. The invoker stated they would run the
build and tests. In place of a build, every new API use was checked against the declaring headers and
the spec validator (`tt_metal/impl/metal2_host_api/program_spec.cpp`); the specific invariants verified
are listed under [Successes](#successes). Pytest commands are in
[Test set](#test-set--not-yet-run-by-the-porter).

## Provenance

- **Recipe docs (this port):** `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
  printed **nothing** — the `metal_2.0` doc tree is not present in this checkout, and the recipe arrived
  as a standalone file (`/localdev/edwinlee/metal2_port.md`). The version cannot be pinned.
  Repo HEAD at port time: `1f944828f1d 2026-07-31 Add markdown files`.
- **Audit docs (inherited):** provenance not pinnable (same reason). Repo HEAD at audit:
  `033960ede6d 2026-07-23`.
- The four shared reference docs the recipe links (`port_patterns.md`, `migration_guide.md`,
  `ttnn_factory.md`, `cb_dfb_api_whitelist.md`) are **absent from this checkout**. Where the recipe
  deferred to them, the declaring headers under
  `tt_metal/api/tt-metalium/experimental/metal2_host_api/` and the validator in
  `tt_metal/impl/metal2_host_api/program_spec.cpp` were used as ground truth instead (which the recipe
  names as the preferred source anyway). See [Friction](#friction).

## TTNN ProgramFactory

- **Concept realized:** `ProgramSpecFactoryConcept` on all three ported factories — matching the
  audit's `MetalV2FactoryConcept` decision. No `override_runtime_arguments`, so the base spec concept
  applies and program-cache hits refresh only tensor bindings.
- **Custom `compute_program_hash` deletion:** none — the op already used the default reflection-based
  hash. No device-op-class edit was forced.
- **Pybind entry points removed:** none. `tilize_with_val_padding_nanobind.cpp` binds only the two
  top-level host functions; no `create_descriptor` was exposed, so there is no user-visible surface
  change.
- **Device-operation class edits:** **none.** `tilize_with_val_padding_device_operation.{hpp,cpp}` is
  byte-identical to pre-port. The three factory *headers* swapped
  `#include <tt-metalium/program_descriptors.hpp>` for `"ttnn/metal_v2_artifacts.hpp"` and changed
  their one declaration; the block-interleaved header is unchanged.
- **Open items:**
  - Both sharded DFBs use `DataflowBufferSpec::borrowed_from`, which has **no other user in TTNN**
    outside `experimental/quasar/` (grep-verified). This port is the first non-quasar exercise of
    borrowed-memory DFBs; treat the sharded factory as the shakedown for that path.
  - The sharded factory's `TensorParameter`s carry no kernel `TensorBinding` — they exist only to back
    the borrowed DFBs. `program_spec.cpp:540-541` explicitly registers a `borrowed_from` parameter as
    "used (no kernel user)", so this is supported, but it is an unusual shape worth a reviewer's eye.

## Handoff points

1. **Per-node runtime-vararg counts require a `[[deprecated]]` API — and there is no alternative.**
   *Owner: Metal 2.0 host-API team.*
   `MultiCoreDefaultFactory`'s reader consumes a variable-length block-representation stream whose
   length **differs per core** (`5 × <distinct consecutive BlockRep groups>`). The only mechanism for
   that is `KernelAdvancedOptions::num_runtime_varargs_per_node`
   (`advanced_options.hpp:86-87`), which is marked
   `[[deprecated("Per-node-vararg-count feature is deprecated and will be removed.")]]` with the note
   *"will be removed from the API once existing uses are refactored to avoid it."* The scalar
   `num_runtime_varargs` cannot express it: `program_run_args.cpp:186-196` validates each node's
   supplied vararg count against the schema *exactly*, so a single count would reject every node whose
   stream is a different length. Use site:
   `tilize_with_val_padding_multi_core_default_program_factory.cpp:267`.
   **This op is one of the "existing uses" blocking that deprecation's removal.** Removing the field
   requires either a per-node-length vararg contract or an op-side refactor that pads every core's
   stream to a common length (a functional change, out of scope for a port). Flagging so the API team
   can count this op before scheduling removal. (The build does not warn — the root
   `CMakeLists.txt:207` sets `-Wno-deprecated-declarations` globally.)

2. **Three shared kernels had to be forked because their only existing Metal 2.0 versions are in
   `experimental/quasar/`.** *Owner: the shared-kernel owners (eltwise/unary, data_movement/sharded,
   `ttnn/cpp/ttnn/kernel/` pool).*
   `writer_unary_interleaved_start_id_metal2.cpp`, `writer_unary_sharded_metal2.cpp` and
   `tilize_metal2.cpp` are new forks created beside their originals (details under
   [Open items for downstream](#open-items-for-downstream)). In each case a converted copy already
   exists under `experimental/quasar/`, but that tree is off-limits to a production port, so the work
   was redone from the recipe. Every subsequent porter of any of the ~30 co-borrowing ops will now
   *reuse* these forks rather than fork again — but the duplication cost of the quasar tree is real and
   worth quantifying for whoever plans the sunset.

3. **No compile/test verification possible in this environment.** *Owner: invoker.*
   See the verification note under [Outcome](#outcome). `clang-20` is absent from this shell, so
   `./build_metal.sh --build-tests` fails at configure. The invoker owns the build and test run.

## Successes

- **Re-deriving the CB-endpoint census instead of transcribing the brief paid off — and confirmed it.**
  The recipe's instruction to *verify, not transcribe* dispositions
  ([Read this first](../../../../../../metal2_port.md)) led to reading every FIFO call in all three
  readers. The census agreed with the brief on all eight ported DFBs: two sharded self-loops
  (`src_shard`, `pad` — reader-only touchers), and 1P+1C everywhere else. Crucially it also confirmed
  that MultiCoreDefault's *two* compute KernelSpecs binding `in`/`out` is **not** a multi-binding: the
  validator (`program_spec.cpp:1247-1306`, `1355-1391`) counts endpoints **per node**, and the two
  compute groups have disjoint node sets, identical `access_pattern` and identical `num_threads`. No
  `allow_instance_multi_binding` anywhere in the port.

- **The "self-loop pair" shape is documented at the field and is exactly what the sharded reader
  needs.** `program_spec.cpp:290-343` states that reusing one `accessor_name` across a PRODUCER and a
  CONSUMER binding of the *same* DFB is the sanctioned self-loop form, and
  `MakeDataflowBufferBindingHandles` (`:2466-2481`) `emplace`s into an `unordered_map`, so the pair
  emits a single `dfb::in0` token — no duplicate definition in the generated header. Reading the
  declaring code rather than guessing settled this in one pass.
  (`tilize_with_val_padding_multi_core_sharded_program_factory.cpp:132-157`.)

- **The "compute config Style A vs Style B" warning fired correctly.** This op sets a Metal
  `ComputeConfigDescriptor` directly (no TTNN `ComputeKernelConfig` anywhere), so the recipe's Style B
  path applies: build `ComputeGen1Config` by hand. Had I reflexively reached for
  `to_compute_hardware_config`, its high-performance defaults would have silently flipped
  `sfpu_precision_mode` (`Precise` → the helper's choice) and `fpu_math_fidelity` on every dtype this
  op supports, with no build or test signal. Verified field-by-field that `ComputeGen1Config`'s
  defaults coincide with the legacy `ComputeConfigDescriptor` defaults for every field the op leaves
  unset.

- **`WorkUnitSpec`s may not overlap — caught before writing the wrong shape.** My first instinct was
  one work unit for `{reader, writer}` on `all_cores` plus one per compute group, mirroring the legacy
  per-kernel `core_ranges` literally. `program_spec.cpp:1693-1706` rejects overlapping work units, so
  the correct shape is a *partition*: reader and writer ride along in each compute group's work unit
  (their union is exactly `all_cores`). Documented inline at
  `tilize_with_val_padding_multi_core_default_program_factory.cpp:279-282` so the next reader of that
  file doesn't retry the overlapping version.

- **`dfb::name → uint32_t` crossed cleanly into the kernel-lib's *template* parameters.**
  `compute_kernel_lib::tilize` and `is_fp32_input_format` take `uint32_t` non-type template
  parameters; `DFBBindingToken::operator uint32_t()` is `constexpr` (`dataflow_buffer.h:55`), so
  `dfb::in` / `dfb::out` substitute directly with no `.id` extraction and no temporary wrapper. The
  recipe's promise about the decoupling shim holds for template arguments as well as function
  arguments — worth stating explicitly in the patterns catalog, since "call site" reads as
  function-call-only.

## Friction

### Gaps

- **The four shared reference docs are absent from this checkout**, so `port_patterns.md`
  (endpoint-assignment procedure, aliased DFBs, shared-kernel Caution rungs), `migration_guide.md`,
  `ttnn_factory.md` (the factory-concept contract and the device-op-class edit procedures) and
  `cb_dfb_api_whitelist.md` (the authoritative CB→DFB method mapping) could not be consulted. Every
  reference to them in the recipe became a header/validator dig. This mostly worked — the headers are
  genuinely good — but two places needed reasoning the whitelist presumably answers outright:
  - `get_local_cb_interface(cb_id).fifo_page_size` → `DataflowBuffer::get_entry_size()`. Establishing
    that these are *identical* required reading `dataflow_buffer.inl:35-40` (`get_entry_size()` returns
    `address_units_to_bytes(fifo_page_size)`) plus `circular_buffer_interface.h:144-149`
    (`cb_addr_shift == 0` outside `COMPILE_FOR_TRISC`). The writer is a DM kernel, so the shift is
    zero and the swap is exact — but on a compute kernel the same swap would *not* be a no-op. A
    whitelist row would have said so in one line.
    (`writer_unary_interleaved_start_id_metal2.cpp:34`.)
  - Whether an unread-but-emitted positional CTA should be carried across or dropped (see the
    "Deliberate non-changes" item below). Neither the recipe's *Dropped Plumbing* categories nor any
    header covers it.

- **`num_runtime_varargs_per_node` is undocumented in the recipe.** The recipe's vararg guidance
  (whitelist rule 4, *Caution: Avoid varargs*) covers *whether* something is a vararg but not the
  per-node-count case, which is the only shape this op's reader can use. I found the field by reading
  `advanced_options.hpp` and confirmed the semantics from `program_spec.cpp:3149-3184` and
  `program_run_args.cpp:175-210`. A short recipe note — "if the vararg count varies per node, the only
  mechanism is the deprecated per-node override; record it as a handoff" — would save the next porter
  the archaeology.

### Confusion

- **Style B compute config + the `unpack_modes` legality table is a near-miss worth documenting.**
  The validator rejects `UnpackToDest` on a *consumed* ≤16-bit-format DFB when
  `enable_32_bit_dest == false` (`program_spec.cpp:1032-1039`), and this op reaches exactly that shape
  on paper: with `bfloat16` input and `bfloat8_b` output, legacy sets
  `unpack_to_dest_mode[c_0] = UnpackToDestFp32` on a 16-bit CB. It is saved only because the *same*
  `fp32_llk_acc` flag also drives `fp32_dest_acc_en` → `enable_32_bit_dest`, so the
  `enable_32_bit_dest` escape at `:1011-1012` always applies wherever the entry exists. That coupling
  is load-bearing and invisible at the call site; it took a careful read of both the legacy factory and
  the validator to be confident this was a faithful translation and not a capitulation. Recorded in
  `METAL2_PORT_PLAN.md` under *Deferred / Flagged* so a future refactor of `fp32_llk_acc` does not
  quietly break it.

- **"Preserve multiplicity" and "work units may not overlap" pull in opposite directions, and the
  resolution isn't stated.** Preserving two compute KernelSpecs is mandatory; work units must
  partition the nodes; therefore the *reader and writer* — which legacy placed on the union — have to
  be replicated into each group's work unit. That consequence is derivable but non-obvious, and it is
  the single most likely place for a porter of a cliff-split op to write an invalid spec. Candidate
  patterns-catalog entry: *work-split cliff → one WorkUnitSpec per group, DM kernels in every group*.

- **Deliberate non-changes worth a reviewer's confirmation.** Both are recorded in the plan's *Flags*:
  - Two positional CTAs are emitted but never read by their kernel — single-core reader slot 1
    (`unpadded_row_size_bytes`) and multicore reader slot 5 (`aligned_page_size`, which the audit's
    *Misc anomalies* already routed to the ops team). In legacy they are load-bearing *positionally*
    (they shift the `TensorAccessorArgs<N>` offset); after the port that role is gone and they are pure
    dead weight. **I carried them across as named CTAs** rather than dropping them, so the diff stays a
    pure syntax swap and the cleanup stays with the owner who already owns the finding. Dropping them
    would be provably behaviour-neutral if the owner prefers that.
  - The sharded factory's RTAs are **node-invariant** (all seven reader RTAs and the single writer RTA
    take the same value on every core), i.e. they are really CRTAs. The recipe explicitly forbids
    converting RTA→CRTA during a port (it changes dispatch semantics), so they stay RTAs. Real
    dispatch-efficiency win available in a follow-up pass.

## Open items for downstream

### Shared kernel touches

All three are the *create the fork* rung: a new `_metal2` file beside the original, plus a pointer
comment in the original. **In every case a converted copy already exists under
`experimental/quasar/`, which a production port may not use**, so the fork was written from the recipe.

| # | kernel | fork created | pointer comment landed | remaining unmigrated consumers |
|---|---|---|---|---|
| 1 | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `…/writer_unary_interleaved_start_id_metal2.cpp` | yes | ~24 op dirs: `copy/typecast`, `data_movement/{bcast,concat,copy,permute,reshape_on_device,slice,tilize,transpose}`, `eltwise/unary_backward/{gelu_bw,tanh_bw}`, `embedding`, `examples/example`, `experimental/matmul/attn_matmul`, `experimental/transformer/{nlp_concat_heads,nlp_concat_heads_boltz}`, `experimental/unary_backward/gelu_backward`, `kv_cache`, `matmul`, `reduction/{generic,prod}` — **plus this op's own block-interleaved factory**, which binds the `_wh` sibling |
| 2 | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | `…/writer_unary_sharded_metal2.cpp` | yes | `sharded/interleaved_to_sharded`, `sharded_partial/interleaved_to_sharded_partial`, `data_movement/{tilize,transpose,untilize}`, `experimental/padded_slice`, `experimental/transformer/nlp_kv_cache_load_slice`, `reduction/generic` |
| 3 | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` (shared pool, not an op dir) | `ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp` | yes | all three `data_movement/tilize` factories (`tilize_{single_core,multi_core_default,multi_core_sharded}_program_factory.cpp`) |

Binding vocabulary each fork now fixes for future reusers (documented in each fork's header comment):
- fork 1: `dfb::out` (CONSUMER), `tensor::dst`, `args::num_pages`, `args::start_id`; optional defines
  `OUT_SHARDED`, `BACKWARDS` preserved with their legacy meaning.
- fork 2: `dfb::out` (CONSUMER), `args::num_units`.
- fork 3: `dfb::in` (CONSUMER), `dfb::out` (PRODUCER), `args::per_core_block_cnt`,
  `args::per_core_block_tile_cnt`.

**Intra-op note for the next porter of this op:** the block-interleaved factory shares *no* kernel
source with the three ported factories (it binds `writer_unary_interleaved_start_id_wh.cpp` and
`tilize_wh.cpp`, both distinct files), so nothing in this change constrains it. When it is eventually
ported it will need its own two forks in `eltwise/unary/` and `data_movement/tilize/`.

### Per-op carry-over

- **`data_movement/tilize` is the obvious next port.** It is the same op minus the padding, all four of
  its factories are `descriptor`-concept, and it binds fork 3 (`tilize_metal2.cpp`) plus fork 1 — both
  of which now exist. Its single-core / multi-core-default / sharded factories map almost line-for-line
  onto the three ported here.
- Any port that binds `writer_unary_interleaved_start_id.cpp` or `writer_unary_sharded.cpp` should
  **reuse** forks 1 and 2 rather than creating another.

### Other

- **Pre-existing unreferenced file, left as found:**
  `device/factories/tilize_with_val_padding_shared_variables.hpp` declares
  `struct shared_variables_interleaved`, which nothing in the repo includes (leftover from the
  pre-`ProgramDescriptor` era). Not audited, not converted, not deleted — a cleanup candidate for the
  op owner.
- **Comment/dtype drift preserved verbatim**, per the audit's *Misc anomalies*: the "Assuming bfloat16
  dataformat" comments (e.g. `…single_core_program_factory.cpp:60-61`) remain, although the op supports
  fp32/int32/uint32/uint16/fp8. `element_size()` is used correctly throughout; the comments are
  cosmetically stale.
- **Test coverage note:** there are **no C++ gtests** for this op (`grep` over `tests/` for
  `TilizeWithValPadding` is empty), so the pytest set below is the entire no-regression baseline. The
  sharded factory's only direct coverage is a *single* test —
  `test_sharded.py::test_sharded_tilize_with_val_padding` — which is thin for the factory carrying the
  port's most novel construct (borrowed-memory DFBs). Worth widening in a follow-up.

## Test set — not yet run by the porter

Discovered by sweeping `tests/` for the op name and filtering to this op (`tilize` /
`untilize_with_unpadding` / `tilize_hpadding_matmul` hits that do *not* reach
`tilize_with_val_padding` were excluded). **Please confirm this set is complete before treating a green
run as a no-regression signal.**

Primary (the no-regression baseline):

```bash
pytest tests/ttnn/unit_tests/operations/data_movement/test_tilize_with_val_padding.py \
       tests/ttnn/unit_tests/operations/data_movement/test_tilize_with_zero_padding.py \
       tests/ttnn/unit_tests/base_functionality/test_tilize_pad_cb.py \
       tests/ttnn/unit_tests/base_functionality/test_tilize_untilize_2D.py -x -v
```

Sharded factory (the only direct coverage of the borrowed-DFB path — run this one specifically):

```bash
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py \
       -k test_sharded_tilize_with_val_padding -x -v
```

Indirect callers (`to_layout` routes row-major→tile through this op):

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_to_layout.py -x -v
pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_tilize_hpadding_matmul.py -x -v
pytest tests/ttnn/docs_examples/test_data_movement_examples.py -x -v
```

Sweeps (optional, slow):

```bash
# tests/sweep_framework/sweeps/tilize_with_val_padding.py
# tests/sweep_framework/sweeps/model_traced/tilize_with_val_padding_model_traced.py
# tests/sweep_framework/sweeps/model_traced/tilize_with_zero_padding_model_traced.py
```

Note: the block-interleaved factory is still legacy, so tests that select it exercise the *unported*
path. It is reached when `enough_space_height == false`, or when the wide/tall heuristic in
`select_program_factory` (`…device_operation.cpp:71-80`) prefers it — no `-k` exclusion is needed,
since both concepts coexist and dispatch correctly.
