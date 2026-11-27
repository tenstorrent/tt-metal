# Metal 2.0 Port Report — `experimental/ccl/all_to_all_async`

## Outcome

**PORTED, unverified on device.** `AllToAllAsyncProgram` is converted to `MeshWorkloadSpecFactoryConcept`, and its four kernels to Metal 2.0 bindings. Evidence:

- **Compiled:** the host side, via `./build_metal.sh --build-tests` on a WH build node.
- **Not compiled:** the four kernels. The JIT builds them only when a program is constructed, and no available configuration constructs one.
- **Not tested:** every applicable test case skips on the only hardware available, an N300 (28 of 28).
  - The op requires a `Ring` of 3 or more devices: it demotes `Ring` to `Linear` on 2 devices, then its factory `TT_FATAL`s on `Linear`.
  - The tests skip before calling it, on device count or "needs a mesh axis that wraps into a ring".
- **Blocked:** mock-device runs of the 1×8 / 8-device tests fail in test setup, before the op is called (see Open items).

None of the recipe's outcome values fits "converted, compiles, but nothing exercises it" (see Friction).

## Provenance

- **Recipe docs (this port):** `6d8d49be280 2026-11-26 docs(metal_2.0): state that a port is performance- and functionality-neutral`. The command was run in the `akertesz/op-porting-recipe` worktree: the port branch is off `main`, which has no `metal_2.0/ai/` tree, so it prints nothing there.
- **Audit docs (inherited):** `6d8d49be280 2026-11-26 docs(metal_2.0): state that a port is performance- and functionality-neutral`

## TTNN ProgramFactory

### Concept realized

`MeshWorkloadSpecFactoryConcept`, as the audit chose. The struct keeps its name `AllToAllAsyncProgram`, so `program_factory_t` is unchanged.
- `create_mesh_workload_artifacts` emits one `programs` entry per coordinate in `tensor_coords` (`range = MeshCoordinateRange(coord)`), one per program the per-coordinate `create_descriptor` built.
- The translated `override_runtime_arguments(attrs, args, ret, range)` returns a `TensorArgument` for every io-tensor `TensorParameter` (`input`, `intermediate`, `output`) and the `chunk_ready` `GlobalSemaphore`. That is exactly the set the ported-from override patched: every address slot of every kernel, plus the semaphore. It ignores the coordinate, as the ported-from override did.

### Device-op-class edits

- Pybind entry points removed: none. The op never pybound `create_descriptor`.
- Custom `compute_program_hash`: left intact at `device/all_to_all_async_device_operation.cpp:246`. All device-op files are byte-identical to `main`.

### Open items

- No relaxation candidates: the hash pins `tensor_args` whole.

## Handoff points

None. The port did not capitulate, and nothing outside the op directory needed changing.

## Successes

- **`ccl_port.md` — One sender, or a FabricConnectionManager.** The present-flag → two one-target specs → `build_from_optional_bindings` mapping fit this op exactly (`device/all_to_all_async_program_factory.cpp`, `add_fabric_side`; `kernels/interleaved_all_to_all_writer.cpp:61-63`). I checked its `fabric_target` helper against legacy `append_fabric_connection_rt_args` (`tt_metal/fabric/fabric.cpp:185-203`); the N/S/E/W/Z first-neighbour loop matches (read).
- **`ccl_port.md` — Kernel side.**
  - "One construction per semaphore, where the legacy kernel read the argument, keeping the legacy variable's name" made the kernel diff a pure substitution (`global_semaphore_addr` in both kernels).
  - "Change nothing downstream" settled the `safe_get_noc_addr` / `noc_semaphore_*` / `perform_atomic_fabric_write` uses without thought.
- **`ccl_port.md` / `ttnn_factory.md` — the override owns the bindings.** Clear enough that the override was a one-liner sharing the miss path's resource run args (`make_resource_run_args`), so the miss and the hit can't drift.
- **`port_patterns.md` — Unity-build hygiene.** `experimental/ccl` is a unity target (`CMakeLists.txt:7`), so the per-factory spec-name constants are prefixed `A2A_`.
- **Anti-pattern self-audit — "search separately for `emplace_runtime_args` / a bare `Buffer*`".** This op delivers every tensor as a `Buffer*`, so a `->address()` grep alone would have passed it vacuously.
- **Audit — Device 2.0 sanctions for CCL kernels.** The `legacy primitive retained` markers and the raw-`GlobalSemaphore`-primitive sanction cleared the receiver reader with no judgment call.

## Friction

### Gaps

- **No path to verify a CCL op on 2 chips, and no outcome value for that** (`metal2_port.md` § Verification, § Outcome). Every test of a Ring CCL op needs a ring of ≥3 devices.
  - On an N300, all 28 applicable cases skip, so the kernels are never JIT-compiled and the spec is never validated. Bindings, DFB roles, fabric-target resolution and named-arg reconciliation are all untested.
  - The recipe reads "all tests passing" as the success signal; 28 skips would pass that check vacuously.
  - The recipe should require at least one test that *invoked* the op (non-zero denominator, as the self-audit already asks for greps).
  - It should also say what a porter does when the op can't run on the hardware they have.
- **Watcher on a fabric-enabled device open overflows the idle ERISC firmware** on WH (`idle_erisc.elf … overflows region:0 limit of 0x58c0 bytes`), so the device never opens. The recipe says Watcher on for every run, with no carve-out.
  - I ran with `TT_METAL_WATCHER_DISABLE_ETH=1`: Watcher stays on for the Tensix cores where the op's kernels run, and only the ethernet cores lose it.
  - The failed open also left the card's ethernet stuck: the next runs failed until `tt-smi -r`.
  - This belongs in `ccl_port.md` or `workspace_setup.md`.
- **`TT_FATAL` census can pass a dropped guard** (`metal2_port.md` § Anti-pattern self-audit). It compares per-file *counts*.
  - This port dropped `TT_FATAL(mesh_dispatch_coordinate.has_value())`: the coordinate is no longer optional, so the subject is deleted. It added ccl_port's `fabric_target` `TT_THROW`. Counts matched and the census printed nothing.
  - Diffing the guard *list* (condition text) instead of the count would catch it.
- **A CTA and an RTA with the same name** (`metal2_port.md` rule 4). Both receiver kernels read a file-scope CTA `num_pages_per_packet` and a local RTA of the same name that shadows it. Arg names share one `args::` namespace (`kernel_spec.hpp:225-226`), so "names that match the variables" collides.
  - I named the dead CTA `num_pages_per_packet_ct` and left the kernel variables untouched.
  - The recipe could say: the arg read by the live variable keeps the name; the other gets a suffix.
- **Dead args** (`metal2_port.md` § Dropped Plumbing, rule 4). The recipe has a rule for dead CBs but none for dead RTAs/CTAs; this op has 23 (audit, Misc anomalies).
  - I kept every one as a named arg (a 1:1 translation), including the unused `output` binding on the sender writer.
  - I dropped only the writer's `arg_for_fab`, a snapshot of the positional cursor the port removes.
- **`build_metal.sh` failed on a stale Tracy WASM sub-build** (unrelated to the port), and the recipe forbids hand-targeted cmake.
  - The node's `build_Release/profiler/build_wasm` predated a Tracy bump. Moving it aside wasn't enough: the ExternalProject's stamp directory also had to go before `build_metal.sh` passed.
  - Environment, not recipe, but a cold porter has no route past it without breaking the "only `build_metal.sh`" rule. It belongs in the `tt-metal-build-test` skill or `workspace_setup.md`.
- **`Table` has no `at()`.** The recipe's `Table`s note lists `push_back` and iterator-pair construction as absent; `at()` is a third that the first build hit (`find(key)->second` works).

### Confusion

- **[CCL] `metal2_port.md` rule 3 boundary note contradicts `ccl_port.md` § Kernel side.**
  - Rule 3 says the recipe assumes no out-of-op call site takes "a `GlobalSemaphore`'s address", and that meeting one is an assumption violation (stop).
  - `ccl_port.md` says helpers "in or outside the op directory" stay unchanged.
  - This op's writer passes `safe_get_noc_addr(…, global_semaphore_addr, 0)` into `perform_atomic_fabric_write` (`ccl/common/kernels/minimal_ccl_common.hpp:45`), as most CCL writers built on that header do.
  - I followed `ccl_port.md`: the address is a plain integer the kernel holds, so § Read this first's definition of a violation ("structurally unavailable") doesn't apply. A literal porter would stop here.
- **[CCL] `ccl_port.md` puts `fabric_target` in "the factory file's anonymous namespace".** `ccl/` and `experimental/ccl/` are unity builds, which merge anonymous namespaces. The second CCL op in the same unity batch to follow the instruction gets a redefinition error.
  - I first followed it literally (this is the first such port), then moved `fabric_target` into `namespace CMAKE_UNIQUE_NAMESPACE` per the recipe fix that followed this run. That compiles, and the generated unity source redefines the macro per source.
- **[CCL] `FABRIC_2D` rule scope** (`ccl_port.md` § Fabric connections). "A kernel with `#if defined(FABRIC_2D)` code" doesn't say whether included headers count. The fabric headers every CCL writer includes have such branches, including the binding word layout (`edm_fabric_worker_adapters.hpp:32`).
  - I read it as "branches the kernel executes" and checked by hand: no switch on executed paths.
- **The audit's donor shape table vs `ccl_port.md`.** The table marks `uint64_t NOC-encoded sem` "✗ not OK", which reads as covering a `GlobalSemaphore` address too. Recorded in the audit's Recipe notes; I followed `ccl_port.md`.
- **"Confirm the test set with the invoker"** (`metal2_port.md` § Locate and confirm the op's tests). This run's invoker had already fixed the policy ("existing tests that fit an N300"), so I took that as the confirmation. A delegated porter needs a way to treat the invoker's instructions as the confirmation.
- **"Extract `MeshTensor` at the top and work with it throughout" vs "don't refactor adjacent code"** (`ttnn_factory.md` § Extracting the tensor; `metal2_port.md` § Host-side). The legacy body reads metadata from the `ttnn::Tensor`s in about 20 places. I extracted `MeshTensor`s only for the bindings and left the metadata reads alone (minimal diff).
- **`port_patterns.md` — Self-loop DFB binding example** names its kernel local `cb_acc` and uses a `BindDFB` helper. Both are against the recipe's own `cb_*`→`dfb_*` rule and its designated-initializer rule.
- **Readiness fetch** (`ttnn_op_porting_readiness.md` step 2): the harness saved the CSV directly, not base64 JSON. Recorded in the audit.

## Open items for downstream

- **Verification still owed.** Run `tests/nightly/t3000/ccl/test_all_to_all.py` and `test_ccl_topology_labels.py::test_all_to_all_async_whole_mesh[1x8]` on a T3K with `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'` and `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`. Until then this port is compiled-host-only.
- **Mock devices can't stand in.**
  - With a mocked T3K, `test_all_to_all.py` segfaults in `mesh_device.load_sub_device_manager` (`:157`), and the labels test fails in `ttnn.from_torch`'s `tilize`: `WorkUnitSpec 'tilize_full' targets node (0,7) … grid … 8x7` (`work_unit.cpp:46`).
  - With a mocked 8×P150, the BH box test segfaults the same way at `:162`.
  - The `tilize` failure is a Metal 2.0 spec-validation failure in another ported op under the WH mock; route to its owner.
- **Op bugs (not fixed; the port preserves them):**
  - The op cannot run on 2 devices (Ring demoted to Linear at `device/all_to_all_async_device_operation.cpp:295-298`, then rejected at `device/all_to_all_async_program_factory.cpp` `create_program_at`).
  - The receiver reader hardcodes a 2-page packet (`kernels/interleaved_all_to_all_receiver_reader.cpp`, `first_id = … 2 * N_DRAM_BANKS …`) where the sender writer uses `contig_pages_advanced`.
  - 23 dead args (audit, Misc anomalies).
- **Shared kernel touches:** none. All four kernels are this op's own and bound only by this factory.
- **CRTA candidates (not converted; RTA→CRTA changes dispatch):** on both receiver kernels, `in_row_tiles`, `in_col_tiles`, `out_row_tiles`, `out_col_tiles` and `num_pages_per_packet` are set to the same value on every receiver core.
