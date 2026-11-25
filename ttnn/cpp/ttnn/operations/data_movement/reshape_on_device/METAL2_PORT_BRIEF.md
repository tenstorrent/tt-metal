# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/reshape_on_device`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** ``547b694ae6f8 2026-11-23 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`` *(carry this line into the port report's Provenance section)*

## Before you start

The audit is GREEN on this tree. Three things behind that verdict are not in the repository's committed state, so check them first.

- **The relaxation analysis doc is on `akertesz/op-porting-recipe`, not on this branch.** It is `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/analyses/relaxations/data_movement_reshape_on_device.md` there, blob `5c6b33c12e64`. The audit validated blob `a437b248cfad`. The two differ on line 13 only, where two links became relative. The port needs the doc present and unchanged. Read it with `git show origin/akertesz/op-porting-recipe:<path>`. If the blob differs, re-run its six validity checks before you transcribe §2 of it.
- **The sheet cell still reads `Padded Shape + Alignment`** on both rows. The audit's vocabulary is `none` / `dynamic (see analysis)` / `legality (see analysis)`. The relaxation conjunct was cleared on the user's direction to read the cell as `legality (see analysis)`. Nothing in the port depends on how the cell is spelled.
- **The local volatile L1 byte copies** in `reader_unary_reshape_interleaved.cpp` (`:40`, `:87-91`) are Device 2.0 compliant by the user's ruling of 2026-10-22. Leave them as they are; the kernel-side whitelist does not admit a respelling.

No rebase is pending: the base is `origin/main` `e39e42718ca`. If one of these open PRs lands first, re-read the file it touches and re-run the six checks: #65944 (`device/reshape_rm_program_factory.cpp`, `Buffer*` → `MeshTensor`, six lines); #63418 (`device/reshape_op.cpp`, `reshape.cpp`; swaps the layout helper behind `compute_output_specs`, on a base that predates #51656); #63553 (`device/reshape_op.cpp`, an older form of the key); #66313 and #66844 (`reshape_nanobind.cpp` docstring); #65284 (`reshape.cpp`).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor`. `ReshapeRMProgramFactory::create_descriptor` (`device/reshape_rm_program_factory.cpp:21`) and `ReshapeTileProgramFactory::create_descriptor` (`device/reshape_tile_program_factory.cpp:21`), both taking `(const ReshapeOnDeviceParams&, const ReshapeOnDeviceInputs&, Tensor& output_tensor)`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (the sheet's `Porting Target`) for both factories. There is no `override_runtime_arguments` to translate, so a cache hit refreshes the tensor bindings and nothing else. One spec across the mesh (`SPMD`).
- **Custom hash:** `compute_program_hash`, `device/reshape_op.cpp:111-127`. Leave it exactly as it is. It is not looser than the declared relaxation (the two are equal by construction), and it is what makes that declaration legal.
- **Device-op-class edits:** none forced. `program_factory_t` already exists (`device/reshape_op.hpp:20`); there is no pybound `create_descriptor` (`reshape_nanobind.cpp` binds only the user-facing function); no factory parameter exists for a pybind hook.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): `get_dynamic_runtime_args` (deprecated hook). The relaxation cell is not spelled as a `see analysis` pointer on the sheet; it clears through the user's reading (see Before you start). A custom hash is present, which is not a gate. `override_runtime_arguments` and a pybound `create_descriptor` are absent.

## Construct — to do

**Tensor bindings** (per binding). All four are **Case 1** (via `TensorAccessor`); none is Case 2 or a borrowed-memory DFB:

| Factory | Tensor | Bound on | The kernel builds today |
|---|---|---|---|
| RM | input | RM reader | `TensorAccessor(src_args, src_addr)` (`reader_unary_reshape_stick_layout_interleaved_multi_core.cpp:23`) |
| RM | output | RM writer | `TensorAccessor(dst_args, dst_addr)` (`writer_unary_reshape_stick_layout_interleaved_multi_core.cpp:21`) |
| Tile | input | tile reader | `TensorAccessor(src0_args, src0_addr)` (`reader_unary_reshape_interleaved.cpp:42`) |
| Tile | output | writer fork, as `tensor::dst` | the fork already builds `TensorAccessor(tensor::dst)` |

Express each as a `TensorParameter` / `TensorBinding`; the kernels build `TensorAccessor(tensor::<name>)`. Drop the slot-0 `Buffer*` runtime args (and the literal `0u` idle-core variants in the RM factory), the `TensorAccessorArgs(*buffer).append_to(...)` compile-time plumbing, and the kernels' `TensorAccessorArgs<N>()`.

**TensorParameter relaxation:** `legality (see analysis)`. The per-slot declaration, transcribed from the doc (do not re-derive it and do not generalise it):

- **Input** `TensorParameter`, in **both** factories: `.relaxations = {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true}`
- **Output** `TensorParameter`, in both: no relaxation (leave the default).

Do **not**:
- leave the input at `{}`. It throws on the second call within a cache entry.
- set `match_padded_shape_only` alone. The key does not pin Alignment, so the flag alone would throw on #63798's own test pair.
- set `dynamic_tensor_shape`. Both factories bake runtime args derived from padded shapes, and a base-concept cache hit never refreshes them. A different padded shape would be silent corruption, not a throw.
- copy the flags onto the output. Its spec is a function of the key, so strict is correct.

There is no conditional: the declaration does not depend on the factory, layout, dtype or buffer type. In any new code you write, take sizes from the padded shape or physical volume, never from the logical shape. Run the doc's six checks first; they pass at audit time.

**TensorAccessor 3rd arg:** none. No accessor in the op passes one.

**CB endpoints:**

| DFB (working name) | Factory | Entry size / entry count | Bindings | Nodes |
|---|---|---|---|---|
| `src0` | RM | `new_stick_size` / `num_pages * max_page_size / new_stick_size` (an exact integer: `reshape_rm_program_factory.cpp:47-52` enforces `max % min == 0`) | RM reader PRODUCER, RM writer CONSUMER (legal 1P+1C) | the full compute grid, idle nodes included |
| `in0` | Tile | `single_tile_size` / 2 | tile reader PRODUCER, writer fork CONSUMER as `dfb::out` (legal 1P+1C) | core (0,0) |
| `scratch` | Tile | `alignment` / 1 | tile reader PRODUCER **and** CONSUMER (self-loop: one toucher, a raw peek) | core (0,0); **conditional** |

The `scratch` DFB exists only when `alignment > FACE_WIDTH * element_size` (`device/reshape_tile_program_factory.cpp:71`), which is Blackhole with BFLOAT16 input in DRAM and no other configuration. Make its spec and its binding conditional on that same condition, and emit the matching define to the tile reader, the only kernel that references it. The kernel side is under Watch for.

**Named arguments.** The names are the kernels' own locals; the legacy positions are in brackets.

| Kernel | RTAs | CTAs |
|---|---|---|
| RM reader | `num_sticks_per_core_read` [1], `num_read_per_barrier` [2], `num_sticks_per_cb_push` [3], `start_id` [4] | `old_stick_size` [0] |
| RM writer | the same four names, [1]-[4] | `new_stick_size` [1] |
| tile reader | `input_Wt` [1], `output_N` [2], `output_C` [3], `output_Ht` [4], `output_Wt` [5] | `ALIGNMENT` [0], `ELEMENT_SIZE` [1] |
| writer fork | `num_pages` (= `num_tiles`, `reshape_tile_program_factory.cpp:39`), `start_id` (= 0, `:116`) | none |

Slot 0 of every legacy RTA list (the address) becomes a binding. The RM writer's CTA [0] is its CB index, which becomes a binding too. The RTAs stay per node: the legacy RM loop sets them on every core of the grid (`reshape_rm_program_factory.cpp:116-216`).

## Watch for

- **CB endpoints (multi-binding):** none. There is no hidden second writer, no multi-reader shape and no semaphore in the op.
- **Cross-op / shared kernels:** the tile writer is `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`. Its **`_metal2` fork already exists** beside it, `writer_unary_interleaved_start_id_metal2.cpp`, so you bind it and do not re-fork. Leave the legacy file alone; it already carries the pointer comment. The fork is read-only to you. Fit check:

  | The fork needs | Reshape supplies |
  |---|---|
  | `dfb::out` bound as CONSUMER, one page per DFB entry (default path, `page_bytes = dfb.get_entry_size()`, fork `:38`) | the `in0` DFB, one tile per entry |
  | `tensor::dst` (fork `:48`) | the output `TensorParameter` |
  | named RTAs `num_pages`, `start_id` (fork `:23-24`) | `num_tiles` and `0` |
  | optional defines `ENTRIES_PER_PAGE` / `PAGE_BYTES`, `OUT_SHARDED`, `BACKWARDS` | none needed: the legacy factory sets no defines |

  Other binders of the legacy copy are a **sunset list, not authorization to convert the kernel in place**: `eltwise/unary_backward/device/unary_backward_program_factory.cpp:343`, `examples/example/device/multi_core_program_factory.cpp:81`, `examples/example/device/single_core_program_factory.cpp:83`, plus tests and a model test (tracking issue #52228). The op's own three kernels have no other binder anywhere in the repository, so convert them in place; no fork is needed.
- **RTA varargs:** none. Every kernel reads each runtime arg at a fixed index as a distinct field; name them all.
- **Anything else the porter needs:**
  - **Hard-coded buffer numbers become tokens.** `reader_unary_reshape_interleaved.cpp:34` (`dfb_id_in0 = 0`) and `:37` (`dfb_id_intermed = 1`); `reader_unary_reshape_stick_layout_interleaved_multi_core.cpp:21` (`tt::CBIndex::c_0`); the RM writer's CB index arrives as CTA 0 (`writer_unary_reshape_stick_layout_interleaved_multi_core.cpp:17`, fed by `reshape_rm_program_factory.cpp:92`). Keep no number. The op has no compute kernel, so no LLK table indexes a CB number.
  - **The scratch handle is unguarded in the kernel.** `reader_unary_reshape_interleaved.cpp:37-39` constructs `DataflowBuffer dfb_intermed(...)` and reads it in a ternary in every config, but the factory allocates the scratch CB in only one family of configs. The token `dfb::<scratch>` does not exist when the DFB is not bound, and a ternary or `if constexpr` still looks the name up. So gate lines `:37-39` at the preprocessor, and in the unbound case keep `intermed_l1_scratch = 0`. `MISALIGNED` (`:36`) stays a constexpr built from the `ALIGNMENT` and `ELEMENT_SIZE` CTAs. The `if (MISALIGNED)` at `:74` touches only locals and stays as it is. Pattern: `port_patterns.md`, *Conditional / optional resource bindings*, including *Promote a CTA gate to a define*. The define and the binding share one condition.
  - **The Blackhole-only path cannot run on this machine.** The scratch DFB and the volatile byte-copy loop (`reader_unary_reshape_interleaved.cpp:74-92`) execute only for Blackhole with BFLOAT16 input in DRAM. The n150 builds neither, so the green 32-test baseline on this tree does not cover them. Cover them with a Blackhole run if one is available, or with a temporary scratch-only switch that forces the variant on Wormhole (the audit did not try whether the forced variant is functionally valid on Wormhole). A compile of the gated variant is the minimum.
  - **RM idle cores.** Cores with no work get a literal `0u` address and zero counts today, and the loops never touch the address. After the port the tensor binding gives every node the real base; the zero-trip loops still never use it. Keep the per-node arg coverage and the full-grid node sets for the kernels and the DFB.
  - **The kernels are already part-modernized.** They use `Noc`, `DataflowBuffer`, `TensorAccessor` and, in one, `CoreLocalMem`. The port is a binding-layer change, not an idiom rewrite.
  - **Prove the relaxation live.** The hit-path tensor-spec check is off by default: `validate_program_args` is `false` (`ttnn/api/ttnn/config.hpp:31`) and gates only the hit-path re-checks (`mesh_device_operation_adapter.hpp:54-57`), while a miss always validates. Run `tests/ttnn/unit_tests/base_functionality/test_reshape.py::test_reshape_on_device_tile_same_padded_shape_different_alignment` with `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'`. With the knob off the hit-path check is skipped, so that test passing says nothing about whether the input declares the relaxation.
  - **Verification scope.** Run every test that reaches these kernels, nightly files and sweeps included; do not sample and do not defer sweeps to CI. Baseline green on this tree (Wormhole n150, fresh `TT_METAL_CACHE`): `tests/ttnn/unit_tests/base_functionality/test_reshape.py -k "reshape_on_device or on_device"` (17), `tests/ttnn/unit_tests/operations/conv/data_movement/test_fold_op.py -k reshape_on_device` (2), `tests/tt_eager/python_api_testing/unit_testing/misc/test_reshape.py` (13). Run the files as separate pytest invocations: two of them share the basename `test_reshape.py` and one `@file` list does not resolve. The other callers and tests (fold and split families, nightly files, sweeps) are listed in the audit under Team-only, *Who reaches the kernels*. Enumerate the full set by a fresh-cache kernel build.
  - **No build-file edit is needed.** `data_movement/CMakeLists.txt:43` globs `reshape_on_device/device/kernels/*.cpp` recursively, and the fork sits under `eltwise/unary`'s own kernel glob.
  - **Stay out of `experimental/quasar/`.** That tree holds hacky pre-port copies and is not a source for names, idioms or precedent here.
