# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/copy/typecast`

## Outcome

**`PORTED`** — all four factories of `TypecastDeviceOperation` (`TypecastProgramFactory`,
`TypecastSubgridProgramFactory`, `TypecastShardedProgramFactory`,
`TypecastRowMajorChunkedProgramFactory`) converted to `MetalV2FactoryConcept`
(`create_program_artifacts`), together with the six kernel entry points they bind. They had to move
as one unit: every factory binds the same compute source (`kernels/compute/eltwise_typecast.cpp`).

*Build:* see the *Build and test status* section below. *Device tests:* run by the invoker; commands
listed there.

## Provenance

```
git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/
```

printed nothing in this checkout, so the recipe-doc version **cannot be pinned**. The recipe was
supplied standalone at `/localdev/edwinlee/metal2_port.md`, outside its doc tree.

- **Recipe docs (this port):** unpinnable — standalone `metal2_port.md`, no tracked `metal_2.0/` doc tree.
- **Audit docs (inherited):** unpinnable — "`git log` on the metal_2.0 docs path was empty; recipe
  supplied standalone at `/localdev/edwinlee/metal2_audit.md`" (copied from `METAL2_PORT_BRIEF.md`).

## TTNN ProgramFactory

- **Concept realized:** `MetalV2FactoryConcept` on all four factories. Each
  `static ProgramDescriptor create_descriptor(...)` became
  `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`; the headers swapped
  `<tt-metalium/program_descriptors.hpp>` for `"ttnn/metal_v2_artifacts.hpp"`.
- **Custom `compute_program_hash` deletion:** none — the op never had one (audit confirmed;
  `typecast_device_op.{hpp,cpp}` untouched).
- **Pybind entry points removed:** none — `ttnn-nanobind/operations/copy.cpp` binds `typecast` as a
  plain function; no `create_descriptor` was ever exposed.
- **Device-operation-class edits forced:** **zero.** `TypecastDeviceOperation` is byte-identical:
  `program_factory_t` keeps all four alternatives and, because all four flipped concept together,
  `AllFactoriesValid` stays satisfied. This is the clean case the recipe describes.
- **Open items (concept fit):**
  - No `TensorParameter` relaxation was needed, so the op never exercises
    `advanced_options.dynamic_tensor_shape`; the sharded factory's specs are re-derived per cache
    miss as before.
  - The adapter's cache-hit path (`MetalV2MeshWorkloadFactoryAdapter::apply_descriptor`) rebuilds a
    fresh `TensorArgument` table each dispatch. The legacy `BufferBinding` pointer-patching path this
    op used was cheaper. Worth watching if typecast shows up in host-bound profiles — noted, not acted on.

## Handoff points

1. **Cross-op donor kernels: forked, not converted (eltwise/unary family owners).**
   Three dataflow kernels this op instantiated by file path live in the eltwise/unary family:
   `eltwise/unary/device/kernels/dataflow/{reader_unary_interleaved_start_id,writer_unary_interleaved_start_id,reader_unary_sharded}.cpp`.
   A grep over `ttnn/` finds **~70 factories** (data_movement tilize/untilize/transpose/slice/pad/concat/permute/…,
   reduction, matmul, embedding, kv_cache, transformer, examples, and the eltwise/unary ops themselves)
   instantiating those paths, **all still on legacy positional CTAs + address RTAs**. An in-place
   named-arg / `dfb::` rewrite breaks every one of them at JIT time, so the port forked instead:
   `copy/typecast/device/kernels/dataflow/{reader_unary_interleaved_start_id,writer_unary_interleaved_start_id,reader_unary_sharded}_metal2.cpp`.
   The forks are byte-for-byte equivalent dataflow with only the binding plumbing changed (see the
   header comment in each). **Ask:** when the eltwise/unary family ports, adopt the same rewrite in
   the donor files and delete these three forks. Until then this is duplicated source, deliberately.
2. **`preserve_fp32_precision` without `fp32_dest_acc_en` would now be rejected (typecast op owners).**
   Legacy set `unpack_to_dest_mode[c_0] = UnpackToDestFp32` whenever `preserve_fp32_precision`, and
   the legacy JIT *ignores* that entry unless the DFB format is `Float32`
   (`tt_metal/jit_build/data_format.cpp:213`). The Metal 2.0 validator does not: `UnpackToDest` on a
   consumed ≤16-bit DFB with `enable_32_bit_dest == false` is a hard `TT_FATAL` on Gen1
   (`tt_metal/impl/metal2_host_api/program_spec.cpp:1064`). The port is safe **only because**
   `typecast.cpp:38` derives `fp32_dest_acc_en = preserve_fp32_precision or …`, and
   `ttnn::prim::typecast` has exactly one caller. If those two flags are ever decoupled — or a second
   caller appears — a `preserve_fp32_precision && !fp32_dest_acc_en` call on a bf16/uint8 input starts
   failing where it used to silently no-op. Worth an explicit `TT_FATAL` or a comment at the
   `TypecastParams` definition; **not changed here** (device-op-class code is out of the port's scope).
3. **Stale comment referencing a method that does not exist (typecast op owners).**
   `ttnn/core/tensor/py_to_tt_tensor.cpp:348` says "See explicit assertion in the
   `TypecastShardedProgramFactory::create` method implementation." There is no `create` method (it was
   `create_descriptor`, now `create_program_artifacts`). Outside the op directory and unrelated to the
   port, so left alone.
4. **Audit anomaly carried through unchanged (typecast op owners).**
   `device/kernels/compute/eltwise_typecast.cpp:31` re-runs `TYPECAST_LLK_INIT()` inside the innermost
   per-tile loop. The audit flagged it as probably-redundant per-tile overhead; the port carries it
   through verbatim (the legacy kernel is the source of truth for behavior).
5. **Recipe / doc tree missing from the porter's checkout (Metal 2.0 doc owners).**
   None of `../shared/port_patterns.md`, `../shared/migration_guide.md`, `../shared/ttnn_factory.md`,
   `../shared/cb_dfb_api_whitelist.md` — nor the `docs/source/tt-metalium/.../metal_2.0/` tree the
   provenance command targets — exist in this checkout. Every structural decision that the recipe
   delegates to those documents (endpoint-assignment procedure, fork-vs-in-place decision, the CB→DFB
   method table) had to be re-derived from the recipe's inline summaries plus the framework headers
   (`kernel_spec.hpp`, `dataflow_buffer_spec.hpp`, `program_spec.cpp`, `dataflow_buffer.h`). It worked,
   but it is luck-dependent: the headers happen to be well-commented. Shipping the doc set with the
   recipe, or inlining the two decision procedures, would remove the dependency.

## Successes

- **[Anti-pattern: Demoting per-group CTA to RTA] fired correctly.** The interleaved and RM-chunked
  factories each emit two compute `KernelSpec`s that differ *only* in `per_core_block_cnt`
  (`typecast_program_factory.cpp:188-196`, `typecast_rm_chunked_program_factory.cpp:255-269`). With a
  `KernelRunArgs` mechanism right there, collapsing them into one spec with a per-node RTA is the
  obvious "simplification" — the recipe's explicit non-negotiable stopped it, and the two-spec shape
  also turned out to be exactly what the DFB endpoint rules want (disjoint node sets, one role each).
- **[§Construct: prefer designated initializers, not `ProducerOf`/`ConsumerOf`] paid off immediately.**
  `kernel_spec.hpp:244-284` offers those factories and they are shorter; writing every binding in full
  designated-initializer form instead (e.g. `typecast_sharded_program_factory.cpp:200-203`) made the
  sharded self-loop *legible* — the PRODUCER and CONSUMER lines sit adjacent with the same
  `accessor_name`, which is precisely the thing a reviewer needs to see.
- **[Hardware configuration → Compute kernels, Style B] caught a silent-precision trap.** The legacy
  op sets a Metal `ComputeConfigDescriptor` directly, so the recipe's instruction to build
  `ComputeGen1Config` by hand rather than routing through `to_compute_hardware_config` avoided
  flipping `sfpu_precision_mode` / `bfp_pack_precision_mode` to the helper's high-performance
  defaults. The `math_approx_mode=false → Precision::Precise` and `bfp8_pack_precise → Precise/Approximate`
  mappings are both stated in the recipe table and both applied
  (`typecast_program_factory.cpp:72-82`).
- **[Kernel-side whitelist rule 7: DFB metadata via the object] was load-bearing.** The donor readers
  read their page size via `get_local_cb_interface(cb_id).fifo_page_size`, which the *audit* explicitly
  blessed as a "sanctioned free function" for Device 2.0 purposes. Without rule 7 I would have left it
  as-is; rule 7 says query the object, so it became `dfb.get_entry_size()`
  (`reader_unary_interleaved_start_id_metal2.cpp:32`). Given the cb-id is gone in Metal 2.0, leaving
  the free function would have been a latent stale-metadata read.
- **[Stop signal: don't add `pop_front` to balance a DFB]** applied verbatim to the sharded output DFB:
  compute `push_back`s into the borrowed output buffer and nothing drains it. The temptation is to add
  a `wait_front`/`pop_front` pair so the FIFO "looks balanced"; the fix is the host-side self-loop
  binding, and the kernel is untouched.

## Friction

### Gaps

- **Fork placement is unspecified.** The recipe's *Open items for downstream* mentions recording "the
  `_metal2`-suffixed new file's path" for a fork but never says *where* a fork should live — beside the
  legacy original (inside the donor op's directory, violating the scope boundary) or inside the porting
  op's own directory. I chose the porting op's directory, which keeps the writeable surface inside
  `copy/typecast/` and matches the pre-existing `experimental/quasar/*/kernels/dataflow/*_metal2.cpp`
  files. One sentence in the fork Caution would settle it for every future porter.
- **The self-loop × `unpack_modes` interaction is undocumented.** The recipe's `unpack_modes` rules
  ("a *consumed* Float32 DFB with `enable_32_bit_dest = true` requires an explicit entry") and its
  self-loop pattern are described in separate sections, and nothing connects them. They do interact: a
  self-loop binds the compute kernel as its output DFB's **CONSUMER**, so the output DFB falls under
  the required-entry rule too (`program_spec.cpp:1073-1090` — "A compute self-loop DFB binds both
  roles; the consumer rules govern it"). Reachable here: input `UINT32` → output `FLOAT32` on the
  sharded path gives `enable_32_bit_dest = true` with a Float32 *output* DFB. Handled at
  `typecast_sharded_program_factory.cpp:180-185`; found by reading the validator, not the docs.
- **No build path on the bench the recipe assumes.** `./build_metal.sh --build-tests` cannot configure
  on this host: the toolchain file pins `/usr/bin/clang-20` and only clang-14 is installed. The real
  build environment is the `ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64` container.
  Worse, the failed host-side configure **overwrote `build_Release/CMakeCache.txt`** before erroring,
  which would have cost a reconfigure for anyone who had a warm tree. `workspace_setup.md` should say
  "run the build inside the dev container, as your own uid" (`docker exec -u "$(id -u):$(id -g)"`), so
  artifacts don't end up root-owned.
- **`Table` has no `emplace_back`, and the recipe's warning stops one step short.** The recipe warns
  that `Table` is a map with no `push_back` — true and useful. What it doesn't say is that the legacy
  code being replaced (`defines.emplace_back(name, value)` loops over a `std::map`, present in all four
  factories) is exactly the shape that trips it. Naming that legacy idiom in the warning would have
  saved a compile cycle.

### Confusion

- **Contradictory guidance on omitting `KernelRunArgs`.** `program_run_args.hpp:90` states "A
  KernelRunArgs must be specified for ALL kernels in the ProgramSpec", while the recipe says "If the
  kernel has no RTAs, the run-args entry may be omitted entirely." The validator sides with the recipe
  (`program_run_args.cpp:1053`: "A registered kernel may be omitted from kernel_run_args only if it has
  nothing regular to supply"). All four factories rely on the omission for their CTA-only compute
  kernels. The header comment is stale and should be fixed — it is the more authoritative-looking source.
- **"The port adds exactly two headers" vs. the existing precedent.** The recipe says a ported kernel
  adds `experimental/kernel_args.h` + `api/dataflow/dataflow_buffer.h` and drops the now-unused
  `circular_buffer.h`. The already-ported `*_metal2.cpp` kernels under `experimental/quasar/` keep
  `api/dataflow/circular_buffer.h` and never include `dataflow_buffer.h`, which reads as a
  contradiction until you take the recipe's "already-ported ops are not best practice" clause
  seriously. I followed the recipe. (The two donors were already on Device 2.0 and already included
  `dataflow_buffer.h`, so for them the include set only gained `kernel_args.h`.)

## Open items for downstream

- **Cross-op kernel touches** (per the scope boundary):
  | donor kernel (unchanged) | path taken | fork path | remaining unmigrated consumers |
  |---|---|---|---|
  | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | **fork** | `copy/typecast/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp` | ~70 factories across `data_movement/*`, `reduction/*`, `matmul`, `embedding`, `kv_cache`, `experimental/*`, `examples/*`, and `eltwise/unary*` |
  | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **fork** | `…/writer_unary_interleaved_start_id_metal2.cpp` | same set |
  | `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | **fork** | `…/reader_unary_sharded_metal2.cpp` | same set |

  No donor file was modified — the legacy copies are bit-identical, so every co-borrower is unaffected.
  Sunset checklist: delete each fork once its donor carries the same named-binding rewrite.
- **Sibling-op carry-over:** `eltwise/unary` is the natural next port — it owns all three donors, and
  porting it is what lets these forks be deleted. The three donors' rewrites are already written here
  and can be lifted verbatim.
- **RTA → CRTA candidate (later cleanup pass, not port work):** the sharded reader's
  `num_tiles_per_core` is the *same value on every node* (`typecast_sharded_program_factory.cpp:217-219`),
  so it is really a common runtime arg. Not converted here — RTA→CRTA changes dispatch semantics, which
  is out of scope for a syntax-swap port.
- **Name-first RTA restructure (later cleanup pass):** all four factories keep their legacy node-first
  per-core loops and bridge with `AddRuntimeArgsForNode`, as the recipe prescribes. A native name-first
  rewrite is a separate tidy-up.
- **Test-coverage notes surfaced but not acted on:**
  - There is **no C++ gtest coverage** for typecast (`unit_tests_ttnn --gtest_filter='*Typecast*'`
    matches nothing), so the port's first-line verification is necessarily pytest-only.
  - No typecast program-cache test exists (the eltwise family has `test_unary_program_cache.py` /
    `test_binary_ng_program_cache.py` but there is no typecast analog). The Metal 2.0 cache-hit path
    (`UpdateTensorArgs`) is only exercised where an existing test happens to invoke the op twice with
    the same shapes — a program-cache test per factory would pin the new fast path down properly.
  - `TypecastProgramFactory` doubles as the non-optimized-sharded fallback; that path is covered by
    `test_eltwise_typecast.py:521-522` only. Its `TensorParameter` now carries a *sharded* TensorSpec
    through the binding channel where legacy passed a `Buffer*`, so it is the least-exercised new path
    in this port and worth a closer look if anything misbehaves.

## Build and test status

- **Build:** `./build_metal.sh --build-tests`, run inside the dev container
  (`docker exec -u "$(id -u):$(id -g)" … wh-15-special-edwinlee-for-reservation-149084`) — **SUCCESS**
  (see the *Gaps* entry above for why the host-side invocation cannot work).
- **Anti-pattern self-audit:** clean. Zero hits across the op directory for
  `buffer()->address()`, `TensorAccessorArgs` (outside fork-header prose), `.id` extraction on a
  `dfb::` handle, `allow_instance_multi_binding`, `get_vararg`, `emplace_runtime_args`, and any
  positional `compile_time_args`; every CTA/RTA is named; `CircularBuffer` / `CBDescriptor` /
  `CBIndex` survive only in "legacy `CBIndex::c_0`" mapping comments. `hw_config` values were diffed
  field-by-field against the legacy `ComputeConfigDescriptor` / `ReaderConfigDescriptor` /
  `WriterConfigDescriptor` (see *Successes*).
- **Device tests:** run by the invoker. Commands, grouped by the factory each exercises:

  ```bash
  # interleaved / tiled  (+ the non-optimized-sharded fallback)
  pytest tests/ttnn/unit_tests/operations/eltwise/test_eltwise_typecast.py -x -v

  # sub_core_grids factory
  pytest tests/ttnn/unit_tests/operations/eltwise/test_typecast_int.py -x -v

  # L1-sharded optimized factory (borrowed-memory DFBs + the output self-loop)
  pytest tests/ttnn/unit_tests/operations/eltwise/test_typecast_sharded.py -x -v

  # ROW_MAJOR chunked factory
  pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_copy_ops.py -x -v -k typecast

  # older harness, broad dtype matrix
  pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_typecast.py -x -v
  ```

  No pre-port baseline run was taken by the porter (the invoker runs the device tests), so the
  comparison to make is against the suites' known-good state on this branch before the port. The port
  is a syntax swap with no intended behavior change, so any newly-failing test is a port defect.

  Two run notes:
  - Activate the venv first (`source python_env/bin/activate`, `export PYTHONPATH=$(pwd)`); on this
    bench the tests must run where the built `_ttnn.so` is importable.
  - Every kernel path this op can select is converted, so there is no not-yet-converted path to
    exclude with `-k` (the recipe's exit-139 pytest-session-crash hazard does not apply here).
