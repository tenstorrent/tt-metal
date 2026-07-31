# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/full`

## Outcome

**`PORTED`** — all three factories of `FullDeviceOperation` converted to `ProgramSpecFactoryConcept` in one
change: `FullInterleavedProgramFactory`, `FullShardedProgramFactory`, `FullNDShardedProgramFactory`, together
with all three kernel sources and the op's own kernel header. Nothing left for a later pass.

No-regression baseline: **1540 passed pre-port, 1540 passed post-port, zero failures either side**
(`tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_full.py` +
`tests/ttnn/unit_tests/operations/data_movement/test_full_like.py`). `test_graph_capture.py`: 19 passed.
Plus 52 targeted zero-fill checks (see Open items 8) covering a path the suite does not reach.

## Provenance

- **Recipe docs (this port):** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` on all three factories, as the audit decided. No disagreement, nothing
re-derived. Each `static ProgramDescriptor create_descriptor(...)` became
`static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)` with the same three
parameters; each returns `ProgramArtifacts{.spec, .run_params}` with `op_owned_tensors` left defaulted
(the factories allocate no device tensors beyond the op's output).

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — the op never had one.
- **Pybind entry points removed:** none. [full_nanobind.cpp](full_nanobind.cpp) binds only the `moreh_full`
  free function, so no pybind line referenced `create_descriptor`. No pybind-hook-only factory parameter
  either.

The device-operation class, `full.cpp`, `full_nanobind.cpp`, `sources.cmake`, and `CMakeLists.txt` are
untouched. The diff is confined to the three factory `.cpp`/`.hpp` pairs, the shared factory helper header,
and the four kernel files.

### Open items

- **Relaxation candidates: none, and this is a structural "no" rather than a conservative one.** Every kernel
  bakes its geometry into compile-time args (`elems_per_page`, `page_size`, `tensor_width_in_pages`,
  `num_shards`, `num_cores`), so a tensor whose spec differed would need a differently-compiled program
  regardless of what the `TensorParameter` tolerated. Strict `TensorSpec` matching on the single output
  parameter therefore costs nothing here. `grep -rn 'ArgConfig::Runtime'` over the op: zero hits.
- **The concept fit was clean.** The op has no input tensors, one output tensor, no semaphores, no compute
  kernel, and no op-owned tensors — the narrowest possible use of the concept. No capability gap surfaced.

## Handoff points

**None.** No capitulation, no boundary-rule assumption violation (no call site outside the op directory
needed a `sem::` or `tensor::` handle), no kernel-lib or LLK gap, no framework gap, no removed pybind
surface, and no edit anywhere outside the op's own directory. Every kernel `#include` in the op resolves to
`tt_metal/hw/inc/api/*` or to the op's own `full_kernel_common.hpp`.

One item worth naming even though it is *not* a handoff, because a reader may expect it to be: the sweep
`tests/sweep_framework/sweeps/model_traced/moreh_full_model_traced.py` fails, and it fails for a reason
unrelated to this port. Line 98 references a bare `input_a_tensor_placement` that is not a parameter of
`run()` and is never bound (line 92 correctly reads the same value via `kwargs.get(...)`), so the sweep
raises a Python `NameError` — *after* `ttnn.moreh_full` has already returned successfully and its output has
been converted to torch. It is a pre-existing bug in Python test code, last touched by an unrelated commit
(`f644d0f2c5a`), and a C++ port cannot cause it. Driving the sweep's own `model_traced_sample` configuration
directly through the ported factory gives PCC `1.0`. Routing this to the sweep owners is a Python-side
matter, not a Metal 2.0 one.

## Successes

- **[Two-toucher DFB → assign 1P+1C](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split),
  step 3 ("re-derive, don't transcribe"), fired exactly as intended.** The interleaved factory has the
  textbook dual-instance work-split *shape*: one `kernel_source` pushed into two `KernelDescriptor`s that
  differ only by Writer/Reader config, both over the same `all_cores`
  ([full_program_factory_interleaved.cpp:108-137](device/full_program_factory_interleaved.cpp#L108-L137) in
  the ported form). Pattern-matching on that shape leads straight to a 1P+1C assignment or, worse, to
  `allow_instance_multi_binding`. Running the census on the kernel bodies instead showed each instance gets
  its *own* buffer index from its own `CBDescriptor`, so no buffer in this op has more than one toucher and
  all five `(buffer, config)` pairs are plain one-toucher self-loops. The catalog's instruction to count
  rather than recognize is what kept a Gen2-forbidden flag out of this diff.
- **[Same-FIFO aliasing](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
  caught a near-miss on the `zero_buffer` helper.** The obvious mechanical conversion of
  `zero_buffer(uint32_t cb_id, uint32_t bytes)` keeps a handle parameter and constructs a `DataflowBuffer`
  inside — which that entry forbids ("alias the handle, keep *one* object"), because the caller is already
  holding an object for that same FIFO. Redirected to passing the caller's buffer:
  `zero_buffer(const DataflowBuffer&, uint32_t)`
  ([full_kernel_common.hpp:19-23](device/kernels/full_kernel_common.hpp#L19-L23)). Same two NoC calls, same
  barrier, one object per FIFO.
- **The brief's `defines` warning earned its space.** `OUTPUT_DTYPE_*` selects the fill loop via `#ifdef`, so
  a define that fails to reach the kernel compiles the loop out entirely and the buffer ships whatever was
  already in SRAM — wrong data, no build error, no validator complaint. Carried onto
  `compiler_options.defines` on every `KernelSpec`, with the consequence stated in a comment at each of the
  three sites rather than left implicit.
- **[Compiler options](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options)
  told me to grep rather than read the config, which is the check that actually answers the question.**
  `grep -n opt_level` over the whole op directory returns zero hits, so no kernel sets one; all four kernels
  are data movement, whose legacy default (`O2`) equals Metal 2.0's `CompilerOptions` default, so nothing
  needed setting and nothing silently dropped a level. Rule 2 (explicit `O3` on compute) does not apply —
  the op has no compute kernel.
- **[Hardware configuration](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#hardware-configuration)'s
  "match on values, not role names" resolved a genuinely confusing case.** Both instances of
  `writer_full.cpp` are named "writer" in the legacy code, but one carries `ReaderConfigDescriptor{}`. The
  resolved triples are the reader default and the writer default respectively, so they take
  `create_reader_datamovement_config` / `create_writer_datamovement_config` accordingly — which also keeps
  them on distinct RISCs and distinct NOCs, as the Gen1 node invariant requires.

## Friction

### Gaps

1. **The no-regression baseline has to be captured before the first kernel edit, and nothing says so.**
   Kernel sources are JIT-compiled from the working tree at test time, so editing a kernel invalidates any
   "pre-port baseline" collected afterwards *even though the host library is still the pre-port build*. I
   lost a baseline run to this: it reported a `JitBuildState::compile_one` failure that reads exactly like a
   port bug, but was just the pre-port host emitting positional CTAs to an already-converted kernel.
   Recovered by `git stash push -- <op dir>`, running the baseline (1540 passed), then `git stash pop`.
   [Verification → Run tests](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#run-tests)
   says "All tests passing pre-conversion should continue to pass post-conversion" but never says *when* to
   establish that set, and the workflow order (Build → Run tests, both after Construct) reads as though the
   baseline comes afterwards. One sentence would close it: capture the baseline before touching any kernel
   source, or from a stashed tree. This belongs next to the existing warning about a selected-but-unconverted
   kernel path crashing the pytest session — the two are the same hazard seen from opposite sides, and
   grouping them would make the JIT-reads-the-worktree fact explicit, which is the part a porter is missing.

2. **No verdict for an op-local kernel helper that takes a `uint32_t cb_id`, and the brief's suggestion
   conflicts with the catalog.** `zero_buffer` (pre-port
   [full_kernel_common.hpp:15](device/kernels/full_kernel_common.hpp#L15)) is an op-authored free function
   taking a CB index and constructing a `CircularBuffer` from it internally. Whitelist rule 1 makes the
   CB→DFB transition total across the op directory, so it must change — but nothing says *how*, and three
   shapes are defensible: keep `uint32_t` (what the brief suggests: "a `dfb::` handle passes through
   unchanged via the constexpr cast and no signature change is forced"), take a `DFBBindingToken`, or take
   the caller's `const DataflowBuffer&`. The first two conflict with
   [Same-FIFO aliasing](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names),
   which says one FIFO gets one `DataflowBuffer` object: at all three call sites the caller already holds
   one, so a helper that constructs its own makes two. I took the third shape. This is the audit's Recipe
   note 5 seen from the port side — the auditor asked for the *classification* rule ("is this a Device 2.0
   holdover?"); the porter needs the *conversion* rule. A line under whitelist rule 1 saying that an
   op-local helper which builds a buffer from a passed CB index should instead take the caller's
   `DataflowBuffer&` would settle it, and would also stop the brief's "no signature change is forced" from
   reading as a recommendation.

3. **Dead non-CB-index CTA: still no rule, and the brief's phrasing supports both answers.** Both sharded
   factories emit `aligned_page_size` at legacy slot 3, which no kernel reads. The brief says "do not treat
   cleaning it up as port work, but do not let it silently misalign your named `compile_time_args` schema
   either." I read that as "carry it forward," so it survives as a named CTA on both sharded `KernelSpec`s
   ([sharded](device/full_program_factory_sharded.cpp#L91),
   [nd_sharded](device/full_program_factory_nd_sharded.cpp#L91)) — but the same sentence reads just as well
   as "drop it, just don't renumber wrongly," and the recipe's dead-CB rule (drop the allocation *and* any
   dead CTA carrying its index) points the other way for the CB-index case. The audit's own Recipe note 6
   flags this hole and asks for an explicit "carry it forward unread, do not clean it up." I'd second that,
   and add the reason it is safe, which is the genuinely reassuring part: named CTAs carry no positions, so
   the slot-3 gap that made this a live renumbering trap in the legacy code cannot misalign anything
   post-port. Worth stating the cost too, since it is the only argument for dropping: the generated
   `kernel_args_generated.h` gains an `args::aligned_page_size` no kernel references.

### Confusion

4. **"Extract `MeshTensor` and work with it throughout" does not fit a factory whose geometry comes off
   `Buffer*`.** [ttnn_factory.md → Extracting the tensor](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md#extracting-the-tensor)
   and the migration guide's "Tensor types in a ProgramFactory" both prescribe extracting `const MeshTensor&`
   at factory entry and using it for the rest of the body, reasoning that a ProgramFactory should hold a
   Metalium memory object. All of this op's geometry comes off `Buffer*` — `output.buffer()->num_pages()`,
   `->page_size()`, `->aligned_page_size()`, `->shard_spec().tensor2d_shape_in_pages`,
   `->buffer_distribution_spec()` — and `MeshTensor` exposes none of them (it has `mesh_buffer()`, a
   different type). Following the guidance literally means rewriting every geometry query, which is exactly
   the out-of-scope churn scope discipline forbids; following it partially means holding two handles to the
   same tensor for no gain. I kept the `ttnn::Tensor` queries verbatim and reached for `MeshTensor` only
   where the API demands it (`output.mesh_tensor()` for the `TensorArgument`). The one Metal 2.0 port on
   `main` outside the out-of-bounds quasar tree (`experimental/reduction/integral_image`) does the same. A
   caveat on the recommendation — it governs tensor *identity and spec*, not `Buffer*` geometry queries,
   which stay on `ttnn::Tensor` — would save the next porter the detour.

5. **Near-miss on the one in-tree reference port: its CTA-retrieval style is not the documented API, and
   nothing flags that.** Looking for shape guidance, the only Metal 2.0 port on `main` outside the quasar
   tree is `experimental/reduction/integral_image`, whose kernels read compile-time args as
   `constexpr auto ctas = get_ctas(); ... ctas.block_depth` rather than the `get_arg(args::name)` the recipe
   and `kernel_args.h` document. That looks like a competing framework API. It is not: `get_ctas()` is a
   hand-written op-local helper in that op's own kernel header
   (`.../integral_image/device/kernels/common_dataflow.hpp:60`) that bundles nine `get_arg(args::…)` calls
   into a struct. I resolved it by reading `tt_metal/jit_build/genfiles.cpp` (which emits `namespace args`
   with `RtaArg` / `CrtaArg` / `CtaVal`) rather than by trusting either source. The recipe's "go to the
   headers first; treat ported code as skeptically-held reference" advice is what made this cheap, and it is
   the entry I'd most want kept as-is — but the episode is a concrete argument for that advice, since the
   ported code here was *locally* fine and still misleading about what the API is.

## Open items for downstream

1. **Shared kernel touches: none — nothing to coordinate, nothing to sunset.** No borrowed kernel (all three
   `kernel_source` paths are inside this op), no lent kernel (`grep -rl writer_full ttnn/cpp/ttnn/operations/`
   hits only this op), and no intra-op fork case (each factory owns its kernel; the interleaved factory's two
   instances of `writer_full.cpp` both convert here). No `_metal2` fork was reused or created.

2. **Dead `aligned_page_size` CTA still emitted** (audit Misc anomalies 1 and 2), now as a named CTA on both
   sharded `KernelSpec`s. Post-port it is harmless in a *new* way that makes it a cheaper cleanup than
   before: named CTAs carry no positions, so it can no longer misalign a `TensorAccessorArgs` base index, and
   deleting it is now a two-line change with no renumbering. Good candidate for the ops team whenever they
   pick up those anomalies.

3. **`elems_per_page` computed two different ways across factories** (audit Misc anomaly 3): interleaved uses
   `page_size / output.element_size()`, both sharded use `page_size / datum_size(data_format)`. Preserved
   verbatim in the port. They agree for the three dtypes `validate_inputs` allows and would diverge for a
   block-float dtype, so it stays a hazard if the allowlist widens.

4. **The buffer is sized `page_size` but the kernels transfer `get_aligned_page_size()` bytes out of it**
   (audit Misc anomaly 4). Preserved exactly: `entry_size = page_size`, `num_entries = 1`, and the NoC write
   still passes `get_aligned_page_size()`. Flagging it here because the port makes it *easier to act on* —
   the size is now one `DataflowBufferSpec::entry_size` field instead of a `CBDescriptor::total_size` /
   `CBFormatDescriptor::page_size` pair that had to agree — but changing it is a functional change and stays
   out of this diff.

5. **`write_addr` computed on a path that does not use it** (audit Misc anomaly 6): `dfb.get_write_ptr()` is
   called unconditionally but only read in the non-zero fill branch; the zero branch's `zero_buffer` derives
   the pointer itself. Preserved.

6. **The reader instance can be launched with zero work** (audit Misc anomaly 5): a core in the smaller work
   group holding one page gets a reader instance with `num_pages_per_core == 0`, which still reserves, fills,
   and pushes a page before writing nothing. Preserved — it is correct, just wasted work.

7. **`defines_from_map` deleted** ([full_program_factory_common.hpp](device/full_program_factory_common.hpp)).
   It existed only to flatten a `std::map` into the legacy `KernelDescriptor::Defines` vector.
   `KernelSpec::CompilerOptions::Defines` is a `Table<std::string, std::string>` with a range constructor, so
   `get_writer_defines(dtype)` now converts directly. `get_writer_defines` itself is unchanged.

8. **Test coverage gap the verification step surfaced — `fill_value == 0` is never exercised on a sharded
   output.** The `val.u == 0` branch calls `zero_buffer`, whose signature this port changes, so its call
   sites in `writer_full_sharded.cpp` and `writer_full_nd_sharded.cpp` matter. Across both files in the
   confirmed set, every sharded and ND-sharded test parameterizes `fill_value` as `[3, -1]` or `[3]`; only
   the two interleaved tests `test_full_callback` (`[3, 0]`) and `test_big_full` (`[0.0, 1.0]`) reach the
   zero path at all, and `test_full_like.py` never does. I verified the uncovered paths out-of-band with a
   throwaway script (52 checks: ND-sharded and legacy-sharded across three shapes × both orientations ×
   TILE/ROW_MAJOR × int32/fp32/bf16, plus interleaved with and without the reader instance) — all 52 match
   `torch.zeros` exactly. That script is deliberately **not** committed; the durable fix is adding `0` to the
   `fill_value` parameterization of the sharded and ND-sharded tests in
   `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_full.py` and
   `tests/ttnn/unit_tests/operations/data_movement/test_full_like.py`, which is a test change outside this
   port's scope.

9. **Per-op carry-over for sibling ports.** Two things here generalize. First, an op whose factories each
   allocate a private single-page staging buffer touched by exactly one kernel is the cleanest possible
   self-loop case — five of them here, no endpoint judgment needed once the census is run. Second, the
   dual-instance-work-split-that-isn't is worth recognizing by name: same source, Reader- + Writer-config,
   one core range, but each instance handed its own buffer index. `full` is a compact example for anyone
   writing that up.

10. **RTA→CRTA tidy-up, noted and deliberately not done.** `fill_value` has the same value on every node in
    all three factories, so it is really a common runtime arg. The recipe is explicit that RTA→CRTA changes
    dispatch semantics and belong to a later pass, so it stays an RTA here. Three sites:
    [interleaved](device/full_program_factory_interleaved.cpp#L99),
    [sharded](device/full_program_factory_sharded.cpp#L95),
    [nd_sharded](device/full_program_factory_nd_sharded.cpp#L94).

11. **No varargs retained** anywhere in the port — every runtime and compile-time arg in all three kernels is
    a distinct field read once, so all are named. Nothing to report under the varargs caution.
