# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/data_movement/pad`

## Outcome

**`PORTED`** — all seven `PadDeviceOperation` factories converted to
`ProgramSpecFactoryConcept`, together with all twelve of the op's kernel sources.

| Factory | Ported | Verified |
|---|---|---|
| `PadRmReaderWriterProgramFactory` | ✅ | on device (test set) |
| `PadRmReaderWriterMultiCoreProgramFactory` | ✅ | **build only — unreachable**, see Handoff points |
| `PadRmReaderWriterMultiCoreDefaultProgramFactory` | ✅ | on device (test set), both conditional-DFB branches |
| `PadRmShardedHeightOnlyProgramFactory` | ✅ | on device (test set) |
| `PadRmShardedWidthOnlyProgramFactory` | ✅ | on device (test set) |
| `PadTileCoreProgramFactory` | ✅ | on device (ad-hoc, **not** in the test set — see Open items) |
| `PadTileMulticoreProgramFactory` | ✅ | on device (test set) |

**Tests: 863 passed, 63 skipped — bit-identical to the pre-port baseline** (863 / 63), over the
invoker-confirmed set:
`tests/ttnn/unit_tests/operations/data_movement/test_pad.py`,
`test_pad_subcoregrids.py`,
`tests/ttnn/nightly/unit_tests/operations/data_movement/test_pad_universal_input.py`,
`tests/tt_eager/python_api_testing/unit_testing/misc/test_padding_test.py`,
`tests/tt_eager/python_api_testing/sweep_tests/pytests/tensor/test_pad.py`, `test_pad_to_tile.py`.
The baseline was captured **before the first kernel edit**. The post-port run carries **2674
`METAL2_CHECKS_FORCED` markers** from both forced translation units
(`program_spec.cpp:2847`, `program_run_args.cpp:502`), so the green was measured with the Metal 2.0
legality checks on. The forcing scaffolding was reverted before commit; `git diff` against the merge
base touches no `tt_metal/` file.

## Provenance

- **Recipe docs (this port):** `64668f470e4 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `64668f470e4 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` on **all seven** factories — six as the audit chose, and one deviation
that was **raised with the invoker before construction and decided by them**:

- **`PadRmShardedHeightOnlyProgramFactory`: `CustomProgramSpecFactoryConcept` → base concept.** The
  audit routed it to the custom concept because it carries an `override_runtime_arguments`, and in
  the same breath raised an open question about whether that was necessary. Reading the method
  settled it: its whole body (`pad_rm_sharded_height_only_program_factory.cpp:412-424`, pre-port)
  re-points exactly two CB base addresses — the input CB and the output CB — and does nothing else.
  Both become `borrowed_from` DFBs in the port, and Metal 2.0 resolves a borrowed DFB's backing L1
  address from its `TensorArgument`; on the base concept the framework refreshes every
  `TensorArgument` on a cache hit. The refreshed set is therefore identical statement-for-statement,
  with no non-tensor work to lose. `PadRmShardedWidthOnlyProgramFactory` carries the same two
  borrowed CBs and never had an override, which is independent evidence for the same conclusion.
  The invoker chose to drop the override; the method and its `.hpp:22-27` declaration are gone.
- The two `MeshWorkloadFactoryConcept` (`create_workload_descriptor`) factories collapse onto the
  single-program concept. Each built **one** `ProgramDescriptor` above the loop and pushed the *same*
  object once per range in `tensor_coords`; nothing was per-mesh-coordinate. The `WorkloadDescriptor`
  wrapper existed only to carry the op-owned pad-value tensor, which
  `ProgramArtifacts::op_owned_tensors` carries natively. **This port is the first production
  exercise of the op-owned-tensor path we are aware of** — see Friction for what it cost.

### Device-op-class edits

- **Pybind entry points removed: none.** `pad_nanobind.cpp` binds only the two public `ttnn::pad`
  overloads — no `create_descriptor` / `create_workload_descriptor` was ever exposed, so the port
  removes no user-visible API.
- **Custom `compute_program_hash`: none**, and no backdoor `attribute_values` / `to_hash`. Nothing
  to leave intact, nothing touched.
- **`pad_device_operation.{hpp,cpp}` is byte-identical.** No device-op-class edit was forced: every
  factory already lived in a factory struct inside the `program_factory_t` variant, so
  `ttnn_factory.md` exception 3 (direct-descriptor) did not apply.

### Open items

- **Relaxation candidates: none applied, none obviously available.** All seven readiness rows read
  `relaxation = none`, and the port kept strict `TensorSpec` matching everywhere.
- The two sharded factories would benefit from a way to say "allocate this DFB on a node set wider
  than its bindings" — or, more likely, from someone confirming that they never needed it. See
  Friction 2.

## Handoff points

1. **`PadRmReaderWriterMultiCoreProgramFactory` is ported but unverifiable.** It is declared in the
   `program_factory_t` variant (`pad_device_operation.hpp:33`) and **never returned by
   `select_program_factory`** — the RM multicore path returns
   `PadRmReaderWriterMultiCoreDefaultProgramFactory` instead (`pad_device_operation.cpp:99-101`).
   The audit asked whether it should be deleted rather than ported; the invoker chose to port it. It
   compiles, satisfies `ProgramSpecFactoryConcept`, and its `create_program_artifacts` symbol is in
   `_ttnncpp.so`, but **no test can select it, so its correctness is unverified by construction** —
   including its op-owned-tensor allocation and its hardcoded resnet core split
   (`split_across_cores`, `:27-159`). *Owner: the pad op owners.* Either wire it into
   `select_program_factory` behind something a test can reach, or delete it; leaving it is 433 lines
   of code no CI signal covers.
2. **`get_vararg()` is read-only, and two kernels need to write back into their vararg block.**
   `reader_pad_tiled.cpp` / `writer_pad_tiled.cpp` walk four `num_dims`-long RTA blocks, and
   `advance_tensor_index` (`device/kernels/dataflow/common.hpp:12-19`) **mutates two of them in
   place** — the legacy kernels used the RTA buffer as scratch. Metal 2.0's vararg API exposes
   `get_vararg(i)` (a value getter, `genfiles.cpp:354`) and no address form, so there is no way to
   write a vararg back. The port copies the two mutable blocks into local `uint32_t[num_dims]`
   arrays at kernel entry (`num_dims` is a CTA, so these are fixed-size) and leaves the two
   read-only blocks as direct `get_vararg` reads; `common.hpp`'s signature drops its
   `volatile tt_l1_ptr` qualifiers accordingly. That is the only place in this port where a value
   changes *where it lives* rather than *how it is named*, and it is forced by the API.
   *Owner: the Metal 2.0 API team.* A `get_vararg_addr(i)` — or the planned `std::array` typed
   arguments, which would make the whole question moot — would remove the need.
3. **A kernel's vararg count is per-`KernelSpec`, not per-node, unless you use a deprecated field.**
   `reader_pad_dims_rm_sharded.cpp`'s gather plan is genuinely different lengths on different cores
   (it grows with the number of source cores and coalesced chunks that core reads from), and
   `SetProgramRunArgs` enforces `args.size() == expected_varargs` exactly
   (`program_run_args.cpp:189-195`). The only exact-fit API is
   `KernelAdvancedOptions::num_runtime_varargs_per_node`, which is `[[deprecated]]` and documented
   as "truly bizarre… will be removed". The port instead declares the **maximum** block length and
   zero-fills shorter cores (`pad_rm_sharded_height_only_program_factory.cpp`, the
   `num_reader_varargs` computation): the kernel walks the block using counts it reads out of the
   block itself, so the zero tail is never read. Behaviour is identical; the cost is a few extra
   zero words of dispatch traffic on cores whose gather plan is shorter than the longest.
   *Owner: the Metal 2.0 API team* — a non-deprecated way to express a ragged per-node vararg block
   would let this be exact rather than padded.

## Successes

- **[CB endpoints → self-loop / 1P+1C](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
  fired correctly on `out_shard` (legacy `c_16`) in `PadRmShardedHeightOnlyProgramFactory`.** The
  writer's only touch of that buffer is a bare `get_write_ptr()` at
  `device/kernels/dataflow/writer_pad_dims_rm_sharded.cpp:90` — no FIFO op, invisible to a
  producer/consumer trace, and easy to read as "the writer doesn't use this buffer". The brief and
  the catalog both insisted the writer must still be *bound*; the independent census agreed, and the
  1P+1C assignment (reader PRODUCER, writer CONSUMER) validated first time. Had the writer been left
  unbound, the failure would have been a build-time unbound-DFB reference rather than anything
  numerical.
- **[Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  caught the `if constexpr` trap before it cost a build.** `pad_align` (legacy `c_2`) in
  `PadRmReaderWriterMultiCoreDefaultProgramFactory` is host-allocated only under
  `stick_size_padded_front != 0 || unaligned`, but the legacy kernel constructed the buffer and
  called `get_read_ptr()` **unconditionally** (`reader_pad_dims_rm_interleaved_v2.cpp:93,99`
  pre-port), gating only the *uses* behind `if constexpr`. The pattern's "`if constexpr` does not
  suppress `dfb::` name lookup" warning is exactly right, and it applies to the discarded `if
  constexpr` **branches** too, not just the construction — so the `#ifdef PAD_ALIGN_DFB` gate had to
  wrap the whole `if constexpr` chain (splicing in `} else` before `#endif` so the plain path stays
  a single block). Both preprocessor branches were then confirmed to have actually compiled and run:
  the JIT cache holds **49 variants of that reader with `dfb::pad_align` bound and 323 without**.
- **[Porting a shared kernel, rung 1](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)
  and the locational fork test both paid off.** `PadTileCoreProgramFactory` borrows
  `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`, which already has a
  `_metal2` fork beside it. The brief's trap warning was live: a *second*, differently-named fork of
  the same kernel sits at
  `copy/typecast/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp` with
  vocabulary `tensor::input`. The locational test (same stem + `_metal2`, **same directory as the
  original**) selects the `eltwise/unary` one, whose vocabulary is `tensor::src` — so the factory was
  written to `tensor::src` / `dfb::in` / `args::num_pages` / `args::start_id`. No new fork, no edit
  to either the original or the fork.
- **The [`constexpr` metadata rule](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md)
  resolved cleanly by reading the legacy declaration.** `writer_unary_pad_dims_interleaved.cpp:28`
  reads `get_tile_size(cb_id_out0)` into a **non-`constexpr`** `const uint32_t`, so it becomes the
  member getter `dfb_out0.get_tile_size()` — no token form needed, no `constexpr` demoted.
- **[Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
  was load-bearing here, not theoretical.** Seven factory `.cpp`s land in one
  `ttnn_op_data_movement` unity TU; without the per-factory prefixes (`RM_SC_`, `RM_MC_`, `RM_DEF_`,
  `SH_H_`, `SH_W_`, `TILE_SC_`, `TILE_MC_`) all seven `READER` / `WRITER` / `IN0` / `PAD` /
  `INPUT` / `OUTPUT` constants would have collided in the merged anonymous namespace.
- **The recipe's "run the baseline before the first kernel edit" rule was the right sequencing.**
  Because host-side edits are safe during a test run but kernel edits are not, the port ran as:
  build → baseline (kernels frozen, host-side factories rewritten in parallel) → kernels → build →
  verify. Nothing was blocked, and the baseline measured the true pre-port tree.

## Friction

### Gaps

1. **The recipe has no verdict for "the legacy CB was allocated on nodes no kernel runs on."** Both
   sharded factories give their `CBDescriptor`s `core_ranges = total_cores` — the *entire* compute
   grid — while their kernels run on `all_cores_padded` only
   (`pad_rm_sharded_height_only_program_factory.cpp:288,303,318` and
   `pad_rm_sharded_width_only_program_factory.cpp:69,85,100`, pre-port). Metal 2.0 **derives** DFB
   placement from kernel bindings and `DataflowBufferSpec` has no `target_nodes` by design, so the
   ported `pad` DFB is allocated only where a kernel binds it. That is a real (if benign, and
   strictly smaller) L1-footprint change that the port cannot avoid or express otherwise. The
   [migration guide's](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#dataflowbufferspec)
   "placement is derived, not specified" note explains the mechanism but doesn't say what to do when
   the legacy `core_ranges` was *wider* than the bindings — which reads, to a porter mid-diff, like
   a behavior change they are not entitled to make. A sentence saying "a legacy CB allocated beyond
   its kernels' nodes narrows to the binding set; this is expected and is a footprint reduction"
   would have saved a re-read of the whole DFB section.
2. **Nothing in the recipe or the catalog says what to do about a kernel helper that constructs its
   own `DataflowBuffer` from a CB id passed as a parameter.** Three kernels have one —
   `fill_pad_cb_with_val` (`reader_pad_dims_rm_interleaved_v2.cpp:16`), `fill_pad_cb_with_val` /
   `fill_pad_cb_with_zero` (`writer_pad_dims_rm_sharded.cpp:13,40`), and
   `fill_cb_with_padding_value` (`writer_pad_dims_rm_sharded_stickwise.cpp:19`) — each taking
   `const uint32_t cb_id` and building a **second** `DataflowBuffer` for a buffer the caller already
   holds an object for. `dfb::name` converts implicitly to `uint32_t`, so the *laziest* port
   compiles and runs: pass the token, let the helper build its own object. That is exactly what
   [Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
   forbids ("don't construct two `DataflowBuffer` objects from the same `DFBAccessor`… device-side
   debug tooling depends on the object↔DFB identity"), but that entry is filed under *aliasing*, so
   a porter looking for guidance on a *helper signature* will not find it. The port changed each
   helper's parameter to `DataflowBuffer&`. Suggest a line in
   [kernel-side whitelist rule 1](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist):
   *a file-local helper taking a `uint32_t cb_id` and constructing a CB from it takes the caller's
   `DataflowBuffer&` instead — passing `dfb::name` into it is the shim used where it does not belong.*
3. **The dead-argument rule is stated for CBs and CTAs but not for RTAs, and this op needed it for
   both.** The recipe covers a **dead CB** (drop it, plus the dead CTA carrying its index) and the
   brief instructs dropping the v1 RM kernels' unread RTA slots ("give each kernel only the args it
   actually reads"), but neither states the general rule, so each of the eight sites had to be
   argued from first principles. What the port settled on, consistently: **an argument no kernel
   reads gets no name and is not emitted** — a CTA nothing reads is a compile-time constant with no
   effect, and an unread RTA slot is a word nothing loads, so both drops are zero-functional-change.
   Full list under Open items. A one-line statement of that rule beside the dead-CB paragraph would
   make it a lookup instead of a derivation.

### Confusion

4. **The brief's dead-RTA list for the v1 RM reader is one slot short.** It names reader slots 3
   (`num_total_W`), 7 (`num_total_Y`), 18 (`start_src_stick_wi`), 23 (`full_unpadded_X_nbytes`).
   The census adds **slot 6, `num_unpadded_Y`** — declared at
   `device/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp:46` (pre-port) and never referenced;
   the padding test at `:87` uses `num_local_unpadded_Y` (slot 22), whose name is one word longer.
   Following the brief rather than a per-variable census would have left a named arg for a value
   nothing reads. Recorded per the recipe's "re-derive, don't transcribe" instruction — which is
   written about *endpoint dispositions* but turns out to apply just as well to the brief's argument
   inventories.
5. **Both `writer_rt_args = reader_rt_args` factories hide how much of the arg list is dead.** With
   the two kernels handed identical 27-slot lists, "which args does this kernel read?" cannot be
   answered from the host side at all; it takes a per-variable grep of each kernel. The brief warned
   about this and was right to. Post-port the reader takes 12 named args and the writer 6 — 18
   between them instead of 54 slots pushed twice.

## Open items for downstream

### Findings the port deliberately preserved (bugs and oddities, not fixed)

1. **`PadTileCoreProgramFactory` packs a FLOAT32 pad value as two bfloat16s.**
   `device/pad_tile_program_factory.cpp` has cases for `INT32`/`UINT32` and `UINT16` and falls
   through to `pack_two_bfloat16_into_uint32` for everything else — including `FLOAT32`, which
   `pad_device_operation.cpp:157-161` explicitly admits. Measured on device: padding a FLOAT32 TILE
   tensor with `value=7.0` fills the pad region with **7.0079193115234375** (= `0x40E040E0`, the
   bf16 pair read back as a float32), and with `value=-3.0` gives **-3.0117**. Data is copied
   correctly; only the pad value is wrong, and only for FLOAT32. **This is pre-existing** — the
   packing block is byte-identical before and after the port (verified against the merge base) —
   and the port preserved it per the scope discipline. The sibling
   `pad_tile_multicore_program_factory.cpp` has the correct case
   (`case DataType::FLOAT32: packed_pad_value = std::bit_cast<uint32_t>(pad_value);`), so the fix is
   to copy that one line across. **Filed as #54223.** No test caught it because no test reached this
   factory; the regression test added under item 5 now does, with the FLOAT32 non-zero-pad case
   `xfail`ed against that issue.
2. **`reader_pad_dims_rm_interleaved.cpp:52` hardcodes `pad_value_const_buffer_nbytes = 64`** with a
   *"fails on BH when > 64"* comment (issue #21978), while the host computed and passed the real
   value at RTA slot 14. The hardcode is preserved verbatim; the arg it shadowed simply gets no name
   now, so the host no longer computes a value nothing reads.
3. **`pad_tile_multicore_program_factory.cpp:194-197` divides both trailing dims by `TILE_HEIGHT`,**
   never `TILE_WIDTH`. Benign while both are 32, latent otherwise. Untouched.
4. **`split_across_cores` has ~30 lines of unreachable code after a `TT_THROW`**
   (`pad_rm_reader_writer_multi_core_program_factory.cpp:97-131`), including a second `TT_THROW` and
   commented-out assignments. Untouched — it belongs to the unreachable factory (Handoff 1).

### Test coverage

5. **`PadTileCoreProgramFactory` (TILE, `use_multicore=False`) is not reached by any test in the
   confirmed set** — and was not before the port either. `test_pad_op`
   (`tests/ttnn/unit_tests/operations/data_movement/test_pad.py:603-612`) *does* parametrize
   `TILE_LAYOUT × use_multicore=False`, but its shapes (`[1,1,18,13]` → `[1,1,32,32]`) are already
   tile-aligned, so `invoke_tile` (`pad.cpp:478-480`) takes the `ttnn::experimental::view` fast path
   and never reaches `prim::pad`. Confirmed empirically: after the full test run the JIT cache held
   **zero** variants of `writer_unary_pad_dims_interleaved`, while every other pad kernel had
   post-port variants. The port verified this factory with an ad-hoc script instead — TILE input
   with a real tile-dim change and `use_multicore=False`, across bfloat16 / float32 / uint32, plus
   three program-cache-hit iterations; everything passed except the pre-existing FLOAT32 pad-value
   bug in item 1.
   **Closed:** `test_pad_tile_single_core` in
   `tests/ttnn/unit_tests/operations/data_movement/test_pad.py` now pads a TILE tensor across a tile
   boundary with `use_multicore=False` over six dtypes, two memory configs and three shapes, and
   repeats each case three times so the second and third dispatches land on a program-cache hit.
   60 passed / 6 skipped / 6 xfailed; the JIT cache confirms it reaches the factory (variants of
   `writer_unary_pad_dims_interleaved` went from zero to ten). Added in response to the Copilot
   review on the port PR.
6. **The op-owned-tensor path is exercised only by the single-core RM factory here.** Its sibling
   (`PadRmReaderWriterMultiCoreProgramFactory`) carries the same allocation but is unreachable, so
   the two factories' op-owned handling has one live test subject between them.
7. **CI caught a borrowed-DFB sizing hole the confirmed set misses: an input smaller than its own
   shard.** `test_paged_cache_mask.py::test_update_cache[1x2_grid]` (ttnn misc ops group,
   wh_n300_civ2) drives `from_torch(..., TILE, height-sharded (32, 128))` on a per-device
   `(1, 16, 1, 128)` tensor; the mesh construction path builds the row-major input on device
   already carrying the requested tile-aligned shard spec, so `prim::pad` receives 16 sticks under
   a 32-stick shard spec. Both sharded factories sized the borrowed `in_shard` DFB as one full
   shard (`shard_height × stick_bytes` = 8192 B), which trips `ValidateProgramSpec`'s spec-time
   bound `dfb_bytes <= compute_packed_buffer_size_bytes()` (= 4096 B, `program_spec.cpp:1567`).
   The legacy CB never validated this: `set_globally_allocated_address` is checked per-bank at
   attach time, and the per-bank allocation is a full shard (8192 ≤ 8192). The confirmed set never
   produces such a tensor — its sharded inputs always fill their shard specs.
   **Fixed:** both sharded factories clamp `in_shard.num_entries` to
   `min(shard_height, total input sticks)`. The count is inert on device — both readers only
   raw-peek the DFB's base pointer, no FIFO ops — so the clamp affects validation only, and in the
   normal (tensor ≥ one shard) case it reduces to the previous value. Reproduced and verified on
   n150 with the exact per-device shapes (direct `ttnn.pad` on a 16-sticks-under-a-32-stick-shard
   input, and the `from_torch` mesh path end-to-end); the confirmed set re-ran green afterward
   (923 passed / 69 skipped / 6 xfailed — the 863/63 baseline plus item 5's added test).

### Shared kernel touches

7. **Borrowed, rung 1 — reused an existing fork; no new file created.**
   - Kernel path:
     `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
   - Rung taken: **reused the existing `_metal2` fork**,
     `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp`
     (same directory as the original). No new file; neither the original nor the fork was edited, and
     no pointer comment was added (the original already has one, and rung 1 forbids touching it).
   - Adopted vocabulary: `dfb::in`, `tensor::src`, `args::num_pages`, `args::start_id`. `pad` defines
     no `BACKWARDS`, the fork's only `#ifdef`.
   - **Remaining unmigrated consumers of the legacy original** (the sunset checklist — the legacy copy
     can be deleted when the last of these migrates; `data_movement/pad` is now off the list):
     `data_movement/untilize_with_unpadding` (`untilize_with_unpadding_multi_core_interleaved_program_factory.cpp`,
     `untilize_with_unpadding_single_core_program_factory.cpp`) · `examples/example`
     (`multi_core_program_factory.cpp`, `single_core_program_factory.cpp`) ·
     `examples/example_multiple_return` (`single_core_program_factory.cpp`) ·
     `experimental/transformer/nlp_create_qkv_heads_falcon7b`
     (`nlp_create_qkv_heads_falcon7b_program_factory.cpp`) · `reduction/topk`
     (`topk_route_prep_program_factory.cpp`) · plus
     `tests/ttnn/unit_tests/gtests/test_generic_op.cpp` and
     `tests/ttnn/unit_tests/operations/debug/test_generic_op.py`.
8. **Intra-op, converted in place — no fork needed.**
   `device/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp` and
   `writer_pad_dims_rm_interleaved.cpp` are bound by **two** of this op's factories
   (`PadRmReaderWriterProgramFactory` and `PadRmReaderWriterMultiCoreProgramFactory`), and **both
   converted in this change**, so the shared-kernel Caution resolved without a fork. Their named-arg
   set is now a shared contract between those two factories: reader takes
   `num_unpadded_W, num_unpadded_Z, num_total_Z, unpadded_X_nbytes, padded_X_nbytes,
   padded_X_diff_nbytes, pad_value_packed, start_src_stick_id, start_src_stick_offset, num_local_Y,
   num_local_unpadded_Y, num_local_W`; writer takes `num_total_Z, padded_X_nbytes,
   start_dst_stick_id, num_local_Y, dst_stick_offset, num_local_W`. A future change to either
   factory must keep both in step.
   The census found **no *lent* kernel**: every other source under `device/kernels/dataflow/` is
   bound by exactly one pad factory, and `device/kernels/dataflow/common.hpp` is included relatively
   by the two tiled kernels only.

### Dead arguments dropped (each a zero-functional-change drop; recorded per the recipe)

9. Nothing below is read by any kernel, so none was given a name or emitted. `file:line` refers to
   the **pre-port** revision.
   - **v1 RM reader RTAs** (`pad_rm_reader_writer_program_factory.cpp:141-169`,
     `..._multi_core_program_factory.cpp:346-374`): slot 3 `num_total_W`, slot 6 `num_unpadded_Y`,
     slot 7 `num_total_Y`, slot 14 `pad_value_const_buffer_nbytes`, slot 18 `start_src_stick_wi`,
     slot 23 `full_unpadded_X_nbytes`.
   - **v1 RM writer RTAs**: slots 3, 7, 9 `num_total_X`, 19 `start_dst_stick_wi` (and its dead local
     `dst_stick_wi`), 22 `num_local_unpadded_Y`, 24 `full_padded_X_nbytes` — plus every reader-only
     slot, since `writer_rt_args = reader_rt_args` (`:172` / `:385`) no longer holds.
   - **v1 RM reader + writer CTAs** (`:81-85` / `:245-249`): slots 0 and 1
     (`unpadded_row_size_nbytes`, `padded_row_size_nbytes`). Neither kernel contains a single
     `get_compile_time_arg_val` call; the slots existed only to offset the `TensorAccessorArgs<2>()`
     chain, and the writer's copy of the pad-value tensor's accessor args was never instantiated.
   - **RM default reader CTAs** (`..._default_program_factory.cpp:141-163`): slot 9
     `stick_size_padded_end`, 10 `num_zero_pad_sticks_read`, 11 `last_zero_stick_size` (all three
     declared in the kernel and never used), and 14-17 (`row_major_min_bytes` and its three
     quotients — never even declared kernel-side).
   - **Sharded width-only** (`..._width_only_program_factory.cpp:134-152`): reader CTA 1
     `padded_stick_bytes`, reader CTA 3 `padded_shard_height`, writer CTA 6 `padded_stick_step`.
   - **Tile multicore** (`pad_tile_multicore_program_factory.cpp:125`): writer CTA 1
     `output_cb_index` — the dead CTA that accompanies the dead CB below.
10. **Dead CB dropped:** legacy `c_1` in `PadTileMulticoreProgramFactory`, allocated at
    `pad_tile_multicore_program_factory.cpp:70-78` (`page_size * multi_buffering_size` bytes per
    core) with **zero touchers in every config**; its index reached
    `device/kernels/dataflow/writer_pad_tiled.cpp:23` as `output_cb_id` and was never read again.
    Both the allocation and the CTA are gone. Net effect: `2 × page_size` bytes of L1 recovered per
    core, and nothing else.

### Other

11. **A pre-existing `.md` citation survives in a ported kernel.**
    `device/kernels/dataflow/reader_pad_dims_rm_interleaved_v2.cpp:99` cites
    `WormholeB0/TensixTile/BabyRISCV/MemoryOrdering.md`. The self-audit's "no ephemeral doc cited
    from code" sweep flags it because the sweep is scoped by the diff and the port touched that
    file. It is **not** a port artifact (present verbatim at the merge base, line 105) and **not** a
    tt-metal repo-relative path — it names a document in the tt-isa-documentation repo, so it does
    not dangle the way an in-repo path would. Left alone as out of scope; recorded so the next
    porter of this file does not re-adjudicate it.
12. **Per-op carry-over.** `data_movement/untilize_with_unpadding` binds the same `eltwise/unary`
    reader and will reach rung 1 on the same fork; its porter inherits the `dfb::in` / `tensor::src`
    vocabulary recorded in item 7.
