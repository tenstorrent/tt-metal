# Metal 2.0 Port Report — `reduction/generic`

## Outcome

**`PORTED`** — all **four** factories converted in one change: `ReduceSingleCoreHwProgramFactory`,
`ReduceMultiCoreHProgramFactory`, `ReduceMultiCoreWProgramFactory` (on `ReduceDeviceOperation`) and
`WelfordReduceProgramFactory` (on `WelfordReduceDeviceOperation`). 14 own kernels converted in place;
the two borrowed writers forked (rung 2). Nothing left on the legacy concept, nothing capitulated.

**Why all four at once** rather than the recipe's default of one factory at a time: six kernel sources
are shared *between* these factories (table in `METAL2_PORT_PLAN.md`), so a one-at-a-time port would
have had to create an intra-op `_metal2` fork of nearly every kernel in the directory. Converting the
whole unit means **zero intra-op forks** — smaller diff, smaller risk, and it matches the brief's
framing of the four factories as one porting unit.

**Verification:** `./build_metal.sh --build-tests` clean; all four factories' `create_program_artifacts`
symbols confirmed present in `build_Release/lib/_ttnncpp.so` and no `create_descriptor` symbol survives
(the unity-build masking trap checked explicitly, not inferred from the exit code). Test results
against the pre-port baseline are in [Test results](#test-results).

## Provenance

- **Recipe docs (this port):** `7e91046b794 2026-07-31 docs(metal_2.0): add the op-porting recipe set`
- **Audit docs (inherited):** `7e91046b794 2026-07-31 docs(metal_2.0): add the op-porting recipe set`

## Test results

Baseline captured on the same commit with the port stashed, so the two runs are directly comparable.

| | gtests (`unit_tests_ttnn --gtest_filter='*Sum*:*MinMax*'`) | pytests (9 files under `tests/ttnn/unit_tests/operations/reduce/`) |
|---|---|---|
| **pre-port baseline** | 19 passed, 0 failed | 1787 passed, 348 skipped, 0 failed |
| **post-port** | 19 passed, 0 failed | 1787 passed, 348 skipped, 0 failed |

**No regressions — the two runs are identical.** The confirmed test set (agreed with the invoker) is:
`tests/ttnn/unit_tests/gtests/test_reduction.cpp` via
`./build/test/ttnn/unit_tests_ttnn --gtest_filter='*Sum*:*MinMax*'`, plus
`test_reduction.py`, `test_reduction_h_interleaved.py`, `test_reduction_mean.py`,
`test_reduction_min.py`, `test_reduction_on_batch.py`, `test_reduction_program_cache.py`,
`test_row_major_reduce.py`, `test_sum.py`, `test_max.py`.

`test_reduction_program_cache.py` passing matters specifically: it exercises the cache-hit path, where
only tensor bindings are refreshed — the failure mode a surviving custom `compute_program_hash` would
have produced (this op never had one).

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` on all four factories, exactly as the audit chose. Each
`create_descriptor` became `create_program_artifacts` returning
`ttnn::device_operation::ProgramArtifacts`; both `program_factory_t` variants keep their existing
membership, and both variants are now wholly on the new concept (no mixed-concept variant). No
op-owned tensors. Strict tensor-arg matching kept — no relaxation declared anywhere.

### Device-op-class edits

- Custom `compute_program_hash` deleted: **none** — the op never had one.
- Pybind entry points removed: **none** — `std_var_reductions_nanobind.cpp` binds only the
  user-facing `ttnn.std` / `ttnn.var`, never `create_descriptor`, so no pybind surface changed.
- The only device-op *header* edits are the two forced by the concept flip: the factory method
  signatures, and swapping `<tt-metalium/program_descriptors.hpp>` for
  `"ttnn/metal_v2_artifacts.hpp"` (`reduce_op_device_operation.hpp:14,25,32,39`;
  `welford_reduce_device_operation.hpp:13,24`). `validate_on_program_cache_miss`,
  `compute_output_specs`, `create_output_tensors`, `select_program_factory` untouched.

### Open items

- **Relaxation candidates:** none identified. The op's kernels bake `Ht` / `Wt` / `W_logical` /
  `H_logical` in as compile-time args on nearly every path, so they are *not* shape-agnostic and would
  not tolerate `dynamic_tensor_shape`. Keeping strict matching is correct here, not merely
  conservative.
- The width-sharded H config would benefit from a documented statement of how
  `DataflowBufferSpec::borrowed_from` interacts with the "every `TensorParameter` needs a binding"
  rule — see Friction 8.
- **The width-sharded H config sits one step away from a known framework bug — deliberately, and
  worth flagging for whoever touches it next.** There is a standing defect where a borrowed-memory
  DFB's device-side base address is corrupted in specs with **more than one `WorkUnitSpec`** (garbage
  write pointer, silently wrong output, no reliable workaround). This config has *two* borrowed DFBs
  (`SRC1_DFB` from the input, `OUT_DFB` from the output) but only **one** work unit, because
  width-sharding forces `core_group_2 = CoreRangeSet()` and therefore
  `has_core_group_2 == false` — so the bug is not reachable here. It becomes reachable the moment
  anyone gives this config a second core group (e.g. a work split across shard grids), so that change
  should not be made until the borrowed-DFB defect is fixed. Nothing to do now; recorded because the
  safety margin is one boolean wide and is not obvious from the code.

## Handoff points

**No capitulation, no boundary-rule violation, no kernel-lib gap.** Specifically:

- No call site needed a `sem::name` or `tensor::name` handle to cross out of the op directory — the
  boundary assumption held. Every out-of-op callee this op's kernels invoke
  (`dataflow_kernel_lib::prepare_reduce_scaler<>`, `compute_kernel_lib::reduce<>`,
  `compute_kernel_lib::tilize<>`, `compute_kernel_lib::Accumulate::at`, and the LLKs `reduce_init` /
  `reduce_tile` / `copy_tile` / `pack_tile` / `pack_block` / `transpose_tile` / `init_sfpu` /
  `compute_kernel_hw_startup` / `llk_pack_reduce_mask_config`) takes a `uint32_t` buffer id, and
  `dfb::name`'s `constexpr operator uint32_t()` bridged all of them — in **both** call-argument and
  non-type-template-parameter position. No `.id`, no temporary wrapper, no shim.
- No kernel-lib or framework file was modified. The op's own
  `reduce_rm_dataflow_common.hpp:110-113` helper did change its parameter type
  (`experimental::CB&` → `DataflowBuffer&`), which the brief already sanctioned: the header is the
  op's own and has no external consumer.
- No `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`, no GlobalCircularBuffer, no
  `address_offset`, no GlobalSemaphore, no CTA varargs — matching the audit.

**One shared-kernel item that will need a peer-team owner eventually** (recorded under Open items for
downstream, not a blocker): the two `_metal2` forks this port created live in `eltwise/unary` and
`data_movement/sharded`. Their binding vocabulary is now the interface every later port inherits.

## Successes

- **[Caution: Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)
  — "name the bindings for the kernel, not for your op" fired correctly.** My first instinct on
  `writer_unary_interleaved_start_id_metal2.cpp` was to name its buffer `dfb::out_tiles` and its RTAs
  after reduce's `num_cols_per_core` / `num_cols_read`. The caution stopped that; the fork instead
  uses the kernel's own vocabulary (`dfb::out` from `cb_id_out`, `tensor::dst` from `dst_addr`,
  `num_pages` / `start_id` from its own locals). With ~34 factories eventually binding this fork, a
  reduce-flavoured name would have been a lasting tax on every one of them — and per the same
  caution, not something a later porter is allowed to rename.

- **[Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  — the "classify per instantiation, not per CB" rule earned its keep.** `c_0` is a genuine
  reader→compute FIFO on the tiled and width-sharded paths but a compute-private tilize scratchpad on
  the dense-RM path (`reduce_op_multi_core_h_program_factory.cpp` `IN_DFB`, bound `"in0"`
  CONSUMER-only vs `"tile_in"` PRODUCER+CONSUMER). One verdict per buffer index would have mis-bound
  one of the two.

- **[Two-toucher DFB → assign 1P+1C, "re-derive, don't transcribe"](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split).**
  Re-running the census confirmed the brief on every `(buffer, config)` pair, and it surfaced one
  thing worth stating explicitly because it inverts the naive reading: Welford HW's `c_22`
  (`COMBINED_DFB`) is **produced by the writer and consumed by compute**. Assigning roles from kernel
  *names* rather than FIFO calls would have inverted it, and the validator's per-node census would
  then have reported 2 producers / 0 consumers on that buffer.

- **[Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options)
  — the `O3` trap.** `grep -n opt_level` over the op returned nothing, which under the legacy
  descriptor API means `O2` for the DM kernels and **`O3`** for the compute ones. Without that
  section I would have left `compiler_options` defaulted and silently dropped every compute kernel to
  `O2`. All four factories now set it explicitly (4 statements, covering all 9 compute `KernelSpec`s
  — each factory's `make_compute` lambda serves both core groups).

- **[Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols).**
  All four factory `.cpp`s are unity-built into one TU and all four wanted the same constant names
  (`IN_DFB`, `SCALER_DFB`, `OUT_DFB`, `READER`, …). Declaring them **function-local** rather than in an
  anonymous namespace avoided the duplicate-symbol collision outright.

- **Named args dissolved a legacy positional-slot workaround.** `reader_unary_reduce_rm.cpp` and
  `writer_reduce_rm_scalar.cpp` each carried `get_compile_time_arg_val((DIM == REDUCE_COL) ? N : 0)`
  selectors written *specifically* so a discarded `if constexpr` branch wouldn't instantiate an
  out-of-range positional slot — plus a comment block explaining the trick. Named CTAs make the whole
  construct unnecessary: the H-only values are simply emitted on both paths (free for a CTA) and read
  by name. The CT-arg builders also lost their `ReduceOpDim` parameter as a result
  (`common.hpp` / `common.cpp:93-134`).

## Friction

### Gaps

1. **A conditionally-bound resource whose gate is *not* a host bool the kernel also sees — the
   recipe's Pattern doesn't cover this shape, and it is a silent-miss.**
   `reduce_h_neg.cpp` / `reduce_w_neg.cpp` each contain an SFPU path and an FPU path in one function;
   only the FPU path touches the `acc` / `ineg` scratch buffers, and the legacy factory allocates
   those CBs only when `use_fpu_negate` (`= negate && !is_sfpu_reduce`). The kernel selects between
   the paths with `if constexpr (is_sfpu_reduce_path<REDUCE_OP, REDUCE_DIM, reduce_format>())` — a
   predicate derived from the buffer's *data format*, not from any CTA. So:
   - the FPU code is not even in a discarded branch (the SFPU branch `return`s and the FPU code
     follows at function scope), so it is unconditionally parsed **and** compiled; and
   - `dfb::acc` / `dfb::ineg` must therefore exist as names in a build where the host deliberately
     did not bind them.

   [Pattern: Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
   prescribes exactly the right remedy (host-emitted define + `#ifdef`-gated references) but frames
   the condition as a host-side flag mirrored by a define, and its "promote a CTA gate to a define"
   note assumes the gate is a CTA. Here the gate is a format-derived `if constexpr` and the block to
   wrap is ~130 lines rather than a `constexpr` alias. Resolution:
   `compute_defines_map["REDUCE_FPU_NEGATE"] = "1"` when `use_fpu_negate`
   (`reduce_op_multi_core_h_program_factory.cpp:559-564`,
   `reduce_op_multi_core_w_program_factory.cpp:356-361`) and `#ifdef REDUCE_FPU_NEGATE` around the
   FPU section (`compute/reduce_h_neg.cpp:121-255`, `compute/reduce_w_neg.cpp:117-200`).

   **Worth calling out as a near-miss:** I wrote the kernel-side `#ifdef` first and only caught the
   missing host-side `defines` emission during the anti-pattern self-audit, *after* a green build.
   Nothing in the build flags it — the kernel compiles fine with the define absent, it just does
   nothing at runtime (`ttnn.min` on bf16 would have returned garbage). A checklist item of the form
   "for every `#ifdef` a port adds to a kernel, confirm a matching `compiler_options.defines` entry
   emits it" would have caught it immediately; the existing checklist's conditional-DFB item asks
   about bindings, not about define *emission*.

2. **Conditionally-declared named *runtime* args have the same name-lookup problem, and the recipe is
   silent on them.** `reduce_rm.cpp` is shared by the H and W dense-RM configs; its H branch reads a
   per-core count as an RTA, and the W factory declares no compute RTAs at all. `if constexpr` does
   not gate the `args::num_output_tiles_local` lookup, so the W build would fail to compile.
   Resolution: a `REDUCE_RM_H_PATH` define plus an `#ifdef`/`#else` around the single read
   (`compute/reduce_rm.cpp:117-127`, emitted at
   `reduce_op_multi_core_h_program_factory.cpp:555-558`).

   The **useful generalisation** the recipe could state: a conditionally-needed **CTA** should just be
   emitted on *both* paths (a compile-time arg costs nothing on the path that ignores it, and the name
   then always resolves), whereas a conditionally-needed **RTA** needs the `#ifdef` treatment (it
   costs a per-node dispatch word and is semantically absent on the other path). This port uses both
   halves of that rule — `H_logical`, `Wt`, `W_logical`, `wt_tiles_per_chunk` are now emitted
   unconditionally by `build_rm_reader_ct_args` / `build_rm_writer_ct_args` (`common.cpp:93-121`),
   while the RTA is gated.

3. **The brief's prescription for `constexpr` metadata reads does not compile.** The brief says to
   bind the `constexpr DataFormat` sites "through a `constexpr` `DataflowBuffer`", citing that
   `DataflowBuffer`'s getters are `constexpr`. The getters are, but **the constructors are not**
   (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:72,75`; the tt-1xx definition binds a reference to
   the runtime `get_local_cb_interface(id)` —
   `tt_metal/hw/inc/internal/tt-1xx/dataflow_buffer.inl:31`), so no `DataflowBuffer` object is usable
   in a constant expression and its `constexpr` getters are unreachable in one. Since all five
   affected sites feed **template arguments**, there is no object-based spelling available at all.

   Resolution: keep the legacy retrieval form but index it with the **DFB handle** —
   `get_dataformat(dfb::in0)` in the two DM readers
   (`reader_unary_transpose_wh_universal_input_cols_partitioned.cpp:37`,
   `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp:39`) and
   `unpack_src_format[dfb::in0]` in the three compute kernels (`compute/reduce.cpp:47`,
   `compute/reduce_h_neg.cpp:39`, `compute/reduce_w_neg.cpp:41`). The magic CB index still
   disappears, which is [whitelist rule 7](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist)'s
   substance; only the *retrieval* form stays legacy. The free-function spelling has in-tree Metal 2.0
   precedent (`data_movement/scatter/device/kernels/dataflow/reader_bf16_reduction_scatter.cpp:137`).
   **Suggested fix:** either make the `DataflowBuffer(DFBBindingToken)` constructor `constexpr` for
   metadata-only use, or have rule 7 say explicitly that a `constexpr`-required site keeps the
   free-function / descriptor-array form indexed by `dfb::name`. Runtime metadata sites *did* move onto
   the object as rule 7 intends (`get_tile_size(id)` → `dfb.get_tile_size()` ×7,
   `get_local_cb_interface(id).fifo_page_size` → `dfb.get_entry_size()` ×2 — byte-identical on DM,
   where `cb_addr_shift == 0`).

4. **The compute-config Style A / Style B dichotomy has no slot for this op's shape, and following
   Style A literally would have silently changed two settings.** All four factories resolve a TTNN
   `ComputeKernelConfig` (Style A's signal) but then forward only a *subset* onto
   `ComputeConfigDescriptor`, deliberately leaving the rest at the **Metal** defaults. Concretely:
   - no factory sets `ComputeConfigDescriptor::math_approx_mode`, whose default is `false`, so the
     legacy resolved SFPU precision is **always `Precise`** — while `to_compute_hardware_config` maps
     the *caller's* `math_approx_mode` into `sfpu_precision_mode`, and the `ComputeKernelConfig`
     struct's own default for that field is `true`. Any caller-supplied config with
     `math_approx_mode = true` would have flipped to `Approximate`.
   - three of the four factories never forward `dst_full_sync_en` (the audit's anomaly 3), so their
     legacy `double_buffer_dest` is unconditionally `true`, while the helper would have used
     `!dst_full_sync_en`.

   Both are pure perf/precision settings with no build or test signal — exactly the hazard class the
   [Hardware configuration](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#hardware-configuration)
   section warns about, arriving through a shape the section's two styles don't name. Resolution: use
   the helper (for its generation selection) and then explicitly restate the legacy values on the
   `ComputeGen1Config` alternative, each with an inline comment naming the legacy field it reproduces —
   `sfpu_precision_mode = Precise` in all four factories, plus `double_buffer_dest = true` in
   `MultiCoreW` / `SingleCoreHw` / `Welford`. **Suggested fix:** add a Style A′ — "the op resolves a
   TTNN config but forwards only some fields onto the Metal descriptor" — with the instruction to diff
   *every* field of the resolved config against the descriptor the legacy op actually built, not
   against the TTNN config it started from.

   Preserving the `dst_full_sync_en` drop is deliberate: fixing it is a behaviour change and belongs
   in its own PR (see Open items).

5. **The Float32 `unpack_modes` required-entry rule reaches further than the recipe's framing
   suggests.** The recipe presents it as "derive its value from the legacy vector," which reads as
   *reindex the entries legacy had*. But the rule fires for **every** Float32 buffer a compute kernel
   *consumes* under `enable_32_bit_dest` — and this op's own default config sets
   `fp32_dest_acc_en = true`, so with Float32 io that is the input, the scaler (whose format is matched
   to the input), and on the negate / dense-RM paths the `acc` / `ineg` / `rm` scratch buffers too.
   Legacy had `UnpackToDestMode::Default` for all of those, i.e. no entry at all. Resolution: a small
   `require_explicit_unpack_mode(name, format)` lambda in each factory that `emplace`s
   `UnpackMode::UnpackToSrc` (the legacy `Default`) for each consumed Float32 buffer, run *after* the
   genuine `UnpackToDest` entries so `Table::emplace`'s insert-if-absent semantics leave those
   untouched.

6. **`TensorParameter`'s relaxation field is named `relaxations`, not `advanced_options`.** The recipe
   (§Dropped Plumbing, §Tensor-arg matching) and the migration guide both say
   `TensorParameter::advanced_options`; the header field is `TensorSpecRelaxations relaxations`
   (`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp:45`). Nothing turned on
   it here (no relaxation used), but a porter following the doc name gets a compile error.

### Confusion

7. **"Local DFB invariant: producer and consumer must share *identical* `WorkUnitSpec` membership" is
   not what the validator enforces, and taken literally it forbids the standard two-core-group
   shape.** The migration guide's troubleshooting list states that invariant. Every factory here uses
   the ordinary `split_work_to_cores` shape: reader and writer belong to **both** `wu_g1` and `wu_g2`,
   while `COMPUTE_G1` belongs only to `wu_g1` and `COMPUTE_G2` only to `wu_g2` — so a DFB's producer
   and consumer emphatically do *not* have identical WU membership. I spent a while believing I had to
   restructure into per-group reader/writer specs before reading the validator, which turns out to do a
   **per-node census** (exactly one producer instance and one consumer instance *per node*) rather than
   any membership comparison — `tt_metal/impl/metal2_host_api/program_spec.cpp:1396-1452`, whose own
   comment says it "subsumes the old within-role disjointness check … and reports the offending node in
   node terms rather than WorkUnitSpec terms". The guide's wording appears to predate that rewrite.
   **Suggested fix:** restate it as the per-node rule, and use the reader/writer-in-both-WUs +
   compute-per-group shape as the worked example, since it is what nearly every ported op looks like.

8. **How `DataflowBufferSpec::borrowed_from` interacts with "every `TensorParameter` needs ≥1
   `TensorBinding`" is undocumented.** On the width-sharded H config the input and output tensors reach
   the kernels *only* through borrowed-memory buffers — no kernel builds a `TensorAccessor` on either,
   so neither has a `TensorBinding`. The two rules are documented independently and their interaction is
   not, leaving two plausible readings (add a dummy `TensorBinding`, which would also inject the
   accessor's layout CTAs; or rely on `borrowed_from`). Reading the validator settled it:
   `borrowed_from` registers the parameter as used — `program_spec.cpp:533-552` ("register as used (no
   kernel user)"). No dummy binding needed. **Suggested fix:** one sentence in the migration guide's
   borrowed-memory-DFB paragraph.

9. **Capturing a comparable pre-port baseline needs a step the recipe doesn't mention.** Kernels are
   JIT-compiled from the working tree at *test* time, not by `build_metal.sh`, so as soon as the first
   kernel is converted the baseline can no longer be taken — the legacy host code would launch
   converted kernels. I recovered by `git stash push -u` over just the port's paths, running the
   baseline, then popping. Worth a sentence in
   [Locate and confirm the op's tests](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#locate-and-confirm-the-ops-tests):
   *run the baseline before touching any kernel source.*

## Open items for downstream

### Shared kernel touches

Both are **rung 2 — created the fork**; no `_metal2` sibling existed for either.

| kernel | fork created | pointer comment landed | remaining unmigrated consumers |
|---|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `…/writer_unary_interleaved_start_id_metal2.cpp` | yes (top of the legacy file) | ~34 factories tree-wide still bind the legacy copy — everything except this op's `ReduceMultiCoreH` (interleaved-tiled), `ReduceMultiCoreW` (tiled), `ReduceSingleCoreHw`, and `WelfordReduce` (W, H) |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | `…/writer_unary_sharded_metal2.cpp` | yes | ~11 factories tree-wide still bind the legacy copy — everything except this op's `ReduceMultiCoreH` width-sharded config |

The forks' binding vocabulary — now the interface later ports inherit — is:

- `writer_unary_interleaved_start_id_metal2.cpp`: `dfb::out` (CONSUMER), `tensor::dst`, RTAs
  `num_pages` / `start_id`, **no** compile-time args; honours the `OUT_SHARDED` and `BACKWARDS`
  defines exactly as the legacy copy does (reduce sets neither).
- `writer_unary_sharded_metal2.cpp`: `dfb::out` (CONSUMER), RTA `num_units`, no CTAs, no tensor
  binding (the output is written in place by the producer).

No **lent** kernel: every source in this op's `device/kernels/` is bound only by these four factories,
so all 14 converted in place with no fork. No intra-op forks either — see Outcome.

### Findings routed to the op owners (deliberately **not** fixed in this diff)

Carried from the audit and confirmed during the port; each is a behaviour change that belongs in its
own PR.

1. **`dst_full_sync_en` from the user's compute-kernel config is silently dropped by three of the four
   factories** (audit anomaly 3). Only `ReduceMultiCoreH` forwarded it. The port **preserves the drop**
   (`double_buffer_dest = true` on the other three) so numerics and perf are unchanged. Whoever fixes
   it should fix all three together, and note the related latent trap the audit flagged: the Welford
   factory passes `DST_SYNC_FULL` as a *define* to reader and compute while the compute kernel's
   `DEST_AUTO_LIMIT` resolves from the JIT-generated `DST_SYNC_MODE`.
2. **Dead compute RTA on the H dense-RM path** (audit anomaly 4). `reduce_rm.cpp` reads only the first
   of its two RTAs and its own comment says the second is unused. Both are carried forward as named
   RTAs (`num_output_tiles_local`, `output_tiles_seen`) so behaviour is unchanged; the name now makes
   the dead one obvious at the declaration site, which should make the cleanup trivial. *Correction to
   the audit's framing:* the dead slot exists only on the H path — the W factory declares no compute
   RTAs at all.
3. **Welford's `c_2` scalar buffer is filled by the reader and read by no compute kernel** (audit
   anomaly 2, and the audit's open question for the op owner). Carried forward faithfully as a reader
   self-loop. If it is confirmed vestigial, dropping it removes one tile of L1 plus a zero-fill and a
   push per core per launch — and the port has made it cheap to spot, since the buffer now visibly has
   no consumer kernel and needs a self-loop binding to satisfy the validator.
4. **Stale comment deleted, not preserved.** `reduce_op_multi_core_w_program_factory.cpp:365` read
   *"Use raw addresses (not Buffer\*) so mesh program-cache fast paths re-apply per-core args"* and
   described a shape the code had already stopped having (audit anomaly 5). It annotated the exact
   `emplace_runtime_args` call the port rewrites, so it went with that line rather than being carried
   onto code it no longer describes.
5. **`MULTI_CORE_HW` never selects a distinct factory** (audit anomaly 1) — untouched, still true.
6. **`math_approx_mode` and `packer_l1_acc` are destructured and never used in all four factories**
   (audit anomaly 6). Still true, and now slightly more consequential: `math_approx_mode`'s non-use is
   what forces the explicit `sfpu_precision_mode = Precise` restatement (Friction 4), so a cleanup here
   should decide whether the op *means* to ignore the caller's request or to honour it.

### Test coverage notes

- The confirmed baseline set exercises the interleaved-tiled, dense-RM and Welford (W/H/HW) configs,
  plus the fused-negate path via `test_reduction_min.py`. It does **not** appear to cover the
  `ReduceMultiCoreH` **width-sharded** config — the one config whose port is structurally distinctive
  (two borrowed-memory DFBs, a DM self-loop on a borrowed buffer, the sharded writer fork, and the only
  config with no `TensorAccessor` at all). That config is therefore the least test-protected part of
  this diff. A sharded-input/sharded-output `ttnn.sum(dim=-2)` case would close the gap; worth adding
  before the fork's binding names harden through reuse.
- `tests/sweep_framework/sweeps/reduction/{std,var,mean,sum.py}` and
  `tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_reduce.py` were found during test
  discovery but excluded from the confirmed baseline; they are additional coverage a reviewer may want
  to run.

### Per-op carry-over

- **Any later port of a reduce-shaped op will hit Friction 3 and Friction 5 immediately**, because
  `is_sfpu_reduce_path<…, reduce_format, …>()` and a format-matched scaler buffer are common to the
  whole family (`moreh_mean`, `moreh_norm`, the `experimental/reduction/*` ops). The
  `require_explicit_unpack_mode` lambda shape used here is a reasonable thing to lift into the patterns
  catalog.
- **The `dfb::name`-as-non-type-template-parameter path is now well exercised** — this port passes
  handles into `prepare_reduce_scaler<dfb::scaler, …>`,
  `reduce<…, dfb::tile_in, dfb::scaler, dfb::out, …>` and
  `tilize<wt_tiles_per_chunk, dfb::rm, dfb::tile_in>`. Worth citing in the patterns-catalog entry,
  which currently gives only one NTTP example.
