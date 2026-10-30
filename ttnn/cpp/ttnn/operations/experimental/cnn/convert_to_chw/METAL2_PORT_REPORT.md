# Port Report — convert_to_chw

Op: `ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw`, factory `ConvertToCHWProgramFactory` (the op's only
factory). Shorthand: `.cpp` / `.hpp` = `device/convert_to_chw_program_factory.{cpp,hpp}`; `reader` / `writer` /
`compute` = `device/kernels/{reader_convert_to_chw,writer_convert_to_chw,convert_to_chw}.cpp`. Line numbers are
post-port unless marked "pre-port" (pre-port = `1308c6a3fc8`).

## Outcome

**PORTED.** `ConvertToCHWProgramFactory` is on `CustomProgramSpecFactoryConcept`. All 111 cases of the confirmed
baseline pass before and after the port on Blackhole, with the Metal 2.0 legality checks forced on and Watcher on,
and 220/220 captured outputs are bit-exact pre vs post. **Wormhole: unverified** (no WH hardware on this bench).

## Provenance

- **Recipe docs (this port):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`
- **Audit docs (inherited):** ``c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`` (from `METAL2_PORT_BRIEF.md`)
- Code: branch `iwrosz/convert_to_chw_2.0`; the port sits on `1308c6a3fc8` (recipe branch `c41f7f4ef0a` + `origin/main`
  `d70d725e238`). Reference port (invoker-supplied, read for spec shape only): `6d661daa4d1` (transpose).

## Verification

Bench: Blackhole p150b, chip 0 only (`TT_VISIBLE_DEVICES=0` under the shared device lock), `TT_METAL_WATCHER=10` on
every run, a fresh `TT_METAL_CACHE` per run. Legality checks: `skip_validation = false` forced at all nine sites under
`tt_metal/impl/metal2_host_api/` (five in `program_run_args.cpp`, four in `program_spec/program_spec.cpp`; working
tree only, not committed). Both of the recipe's one-shot marker warnings (one in each forced file) appear in every
post-port log, once per program build.

| run | build | result |
|---|---|---|
| `test_convert_to_chw.py::test_convert_to_chw` (70) | pre-port (cold build of `1308c6a3fc8`) | **70 passed** |
| `test_convert_to_chw_padded` (40) + `test_convert_to_chw_with_program_cache` (1) | pre-port | **41 passed** |
| `test_convert_to_chw` (70) | post-port | **70 passed**; markers 70 (`program_spec.cpp`) + 70 (`program_run_args.cpp`) |
| `test_convert_to_chw_padded` + `…_with_program_cache` (41) | post-port | **41 passed**; markers 44 + 44 = 40 padded misses + the 4 cache entries the cache test asserts; its other 20 dispatches are cache hits through the (forced) `UpdateProgramRunArgs` |
| bit-exact capture: all 110 test configs, each dispatched twice | pre vs post | **220/220** outputs identical (SHA-256 of the raw bf16 bytes) |

The capture (scratch script, kept outside the repo) dispatches every configuration of both parametrized tests twice:
once as a program-cache miss, then — after allocating a dummy L1 tensor to shift addresses — with a freshly
allocated input of the same data, as a cache hit that runs the translated `override_runtime_arguments`. Pre and post
also agree on program-cache entry counts (70/70, 40/40), on hit output == miss output for every configuration, on
every `bfloat16` input matching torch exactly, and on the `bfloat8_b` PCC (min 0.99997). Watcher log and run logs: no
errors. No skips (every configuration needs ≤ 64 cores). C++ gtests: none exist for this op.

## TTNN ProgramFactory

### Concept realized
`CustomProgramSpecFactoryConcept`, as the audit chose. `create_program_artifacts` replaces `create_descriptor`, and
`override_runtime_arguments` drops its `Program&` parameter and returns `ProgramRunArgs` (`.hpp`). The override
(`.cpp:231-252`) returns a `TensorArgument` for **both** io-tensor `TensorParameter`s, `INPUT` and `OUTPUT` — the
same two addresses the ported-from override re-pointed (the borrowed `c_0` / `c_1`, pre-port `.cpp:163-166`) — and
no `kernel_run_args`, because the ported-from override refreshed no runtime args. Its two `TT_FATAL` guards are
carried across verbatim (`.cpp:242-243`). `coord` is unused, as `mesh_dispatch_coordinate` was. No op-owned tensors.

### Device-op-class edits
- Pybind entry points removed: none (`convert_to_chw_nanobind.cpp` binds only the free function).
- Custom `compute_program_hash`: none — default reflection hash. The device-operation files are byte-identical.

### Open items
- Relaxation candidates: none (both parameters strict, relaxation `none`).
- Capabilities not yet on the concept that this op would want: none.
- Friction with the concept fit or the entry-point wiring: none — a method swap inside the existing struct.

## Handoff points

1. **Device 2.0 — open question carried, not a capitulation.** The writer still feeds raw `get_read_ptr()` /
   `get_write_ptr()` peeks from its `DataflowBuffer`s into the pool donor helpers `experimental::set_read_state` /
   `read_with_state` (`writer:29, 41, 56`), which build their NoC source endpoint with the hand-built loopback
   `experimental::local_addr()` (`ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp:37-39`).
   The audit cleared Device 2.0 on a judgment call about that shape (audit, Questions 2); the Device 2.0 team has not
   ruled, and the user said proceed. The pool header is untouched, as the brief asked. If the team rules the shape
   non-compliant, the fix belongs in that header (45 includers outside the quasar tree), not in this op.
   **Owner: Device 2.0 team.**

No kernel-lib gaps, no boundary-rule violations (no `sem::` / `tensor::` handle crosses out of the op), no removed
pybind surface, no framework gaps hit.

## Successes

- **"Guards move with it"** ([Translating `override_runtime_arguments`](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#translating-override_runtime_arguments)).
  The override's two null-buffer `TT_FATAL`s guard `Buffer*`s that no longer feed anything, which reads exactly
  like the census's one legitimate loss ("a guard on a raw `Buffer*` that a `TensorBinding` replaced"). The explicit
  "an `override_runtime_arguments` is never a subject-deleted case" kept them (`.cpp:242-243`); the `TT_FATAL` census
  is a clean no-diff (factory 6 → 6).
- **`constexpr` metadata keeps the free function** (whitelist rule 7 / CB→DFB whitelist §A). `writer:20`,
  `constexpr uint32_t in_transpose_tile_size = get_tile_size(dfb::in_transpose);` — the declaration-is-the-test rule
  made it a lookup instead of a judgment call, so the `constexpr` was not demoted to fit the member getter.
- **Compiler options.** The "absent line" warning and the mechanical `grep opt_level` pairing put the explicit `O3` on
  the compute spec (`.cpp:183`) on the first pass. Nothing else would have caught a silent `O2`.
- **The brief's traps fired correctly.** "`experimental::CB` is an alias for `CircularBuffer`" (a `CircularBuffer`
  grep never hits `reader` / `writer`), and "the helper's local `cb_out` is `c_2`, not `c_1`". Both would have cost
  a debugging round.
- **Forced checks with per-file markers.** The markers counted exactly one per cache miss in every log, which is the
  positive proof that both rebuilt translation units were live (see the first Friction entry for why that mattered).

## Friction

### Gaps
1. **The legality-forcing grep misses the spec-side choke point.** The recipe's
   `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp` is one directory deep. On this tree it returns
   only the five sites in `program_run_args.cpp` and misses all four in `program_spec/program_spec.cpp` — including
   `BuildProgramFromSpec`, the choke point the next paragraph names. The recipe says to trust the grep over a list, so
   a porter following it exactly forces the run-args side only and leaves `ValidateProgramSpec` at whatever TTNN
   passes; the two-marker proof would show one marker, but only if the porter thinks to put a marker in a file the
   grep never named. Used `grep -rn` over the directory: nine sites. Fix: make the recipe's command recursive.
2. **Hardware-config API spellings are stale against this tree's headers.** The recipe writes
   `create_reader_datamovement_config(device->arch())`; the TTNN helper here is
   `create_reader_datamovement_config(bool disable_dfb_implicit_sync_for_all = false)` (no arch parameter — passing
   an arch would not even compile). It names `DataMovementGen1Config{…}`, `ComputeGen1Config` and
   `std::get<ComputeGen1Config>(compute_hw)`; none exist here — `ComputeHardwareConfig` is a flat struct
   (`fpu_math_fidelity`, `sfpu_precision_mode`, `enable_32_bit_dest`, `double_buffer_dest`, `unpack_modes`) with
   optional per-generation `config_1xx` / `config_12x` / `config_2xx`, and `DataMovementHardwareConfig` carries an
   optional `DataMovement1XXConfig config_1xx`. The value tables still map 1:1 onto the real fields (confirmed against
   `MakeGen1ComputeConfig`, `tt_metal/impl/metal2_host_api/program_spec/construction/kernel_lowering.cpp:154-182`);
   only the spellings are stale.
3. **No entry for kernel-local helpers that take CB ids and build their own wrappers.** `compute:12-36`
   (pre-port `transpose<BATCH_SIZE>(uint32_t cb_in, uint32_t cb_out)` building `CircularBuffer`s inside). The brief
   left the shape to the porter. Options weighed: keep `uint32_t` parameters and build with the low-level
   `DataflowBuffer(uint16_t)` constructor (breaks whitelist rule 3, "construct from the named token"); pass
   `DataflowBuffer&` and call `get_id()` at the LLK sites (the temp-wrapper anti-pattern's shape); pass the tokens as
   `DFBBindingToken<P, C>` parameters. Took the last — the legacy shape with the token as the handle, the same form as
   the ported `moreh_layer_norm_backward` readers — at the cost of four access-pattern template parameters. A pattern
   entry would settle it for the next porter.
4. **`DataflowBuffer` FIFO counts are `uint16_t`; `CircularBuffer`'s were `int32_t`.** The whitelist maps
   `reserve_back` / `push_back` / `wait_front` / `pop_front` "1:1, names unchanged" without mentioning the narrowing.
   Harmless here: `reader:12` pushes `total_tiles`, but `in` is borrowed from the input shard, so the count is bounded
   by one core's L1, far below 65536 tiles. A port pushing a larger count would truncate silently. Worth a note in the
   whitelist.
5. **The scaffolding self-audit fails by construction on a recipe-branch checkout.** The recipe's
   "no forced-legality scaffolding in the diff" grep over `git diff "$BASE"` (the marker string and the
   do-not-commit tag) also matches six lines of the recipe docs themselves, which the port branch carries on top of
   `main` — and would match a port report that quotes either string, which is why this one paraphrases them. Ran it
   scoped to the port's files instead (`git diff "$BASE" HEAD -- ':!docs'`): no hits. Suggest adding the pathspec to
   the recipe's command.
6. **Test discovery: the only sweep can't serve as a baseline.**
   `tests/sweep_framework/sweeps/model_traced/convert_to_chw_model_traced.py`: its `model_traced` suite needs
   `model_tracer/traced_operations/ttnn_operations_master.json`, which is not in the checkout, and its
   `model_traced_sample` suite feeds a DRAM-interleaved input that the op rejects before the device op
   (`TT_FATAL(input.is_sharded(), …)`, `convert_to_chw_device_operation.cpp:65`).

### Confusion
7. **"Adds exactly two headers" vs a donor include that only supplied the CB alias.** The reader got `CircularBuffer`
   only through the pool donor header's `using CB = CircularBuffer`; after the swap that include is unused. Dropped it
   as part of rule 1's sweep ("unused `#include`s"), with the invoker's agreement; the writer keeps it because it still
   calls the pool NoC helpers. The recipe doesn't say whether such a donor include counts as "the `CircularBuffer`
   include".
8. **One DFB, two legacy names.** The compute kernel called the intermediate `cb_transpose_in`; the writer and the host
   called it `cb_in_transpose`. Applying the rename rule literally (`cb_X` → `dfb::X`, no other renames) gives two
   accessor names for one DFB — `transpose_in` on the compute binding, `in_transpose` on the writer's. Legal
   (accessor names are per-kernel, as the migration guide's `out_dfb` / `in_dfb` example shows), but the recipe
   doesn't say whether to keep each kernel's vocabulary or unify. Kept each kernel's.

## Open items for downstream

- **Shared kernel touches:** none. All three kernel sources are private to this factory (filename census: no other
  binder; no `_metal2` sibling), so they were converted in place. Sunset list: empty.
- **`cnn_cb_descriptor.hpp` (`make_cnn_circular_buffer`)** now has a single consumer, `convert_to_hwc`
  (`convert_to_hwc/device/convert_to_hwc_program_factory.cpp:313-355`). The helper itself is untouched; when
  `convert_to_hwc` ports, it can go. Side effect of no longer calling it: its three debug-level
  `"Creating CB at index …"` `LogOp` lines are no longer emitted for this op.
- **Preserved bugs** (from the audit's Misc anomalies; unchanged by the port, still worth an owner's look):
  - `writer:10-12` hard-codes a 2-byte output element (`STICK_SIZE = 64`) while the output dtype is unrestricted and
    defaults to the input's (`convert_to_chw_device_operation.cpp:88`); a 4-byte output dtype would be written at
    the wrong stride.
  - The writer's channel stride assumes the minimal shard height (`total_tiles × 64` bytes, `writer:31`), while
    validation still admits taller input shards (`convert_to_chw_device_operation.cpp:32-34`, `.cpp:83-85`).
- **Quasar-uplift debt introduced or carried by this port:** the DM self-loop on `out` (writer bound PRODUCER +
  CONSUMER, `.cpp:160-171`); the Gen1-only token conversion in `writer:20`
  (`constexpr … get_tile_size(dfb::in_transpose)`); the compute helper's token parameters (Gen1 `DataflowBuffer` is
  not a template, so they deduce cleanly here); the pool helpers' loopback endpoint (Handoff 1).
- **CRTA candidate:** `total_tiles` has the same value on every node for all three kernels (`.cpp:44-57`). Left an
  RTA: an RTA→CRTA move changes dispatch semantics and is a separate cleanup.
- **Test coverage:** the confirmed baseline is the unit test file only. Real-model callers exist —
  `models/experimental/functional_unet/tests/test_unet_output_layer.py`, `test_unet_model.py` (UNet shallow, random
  weights), vanilla UNet (Wormhole-only test), SDXL VAE decoder (needs HF weights) — and were left out of this run by
  the invoker. Wormhole is unverified.
- **Working-tree scaffolding:** the nine forced `skip_validation` sites and two markers remain uncommitted in the
  worktree (`tt_metal/impl/metal2_host_api/program_run_args.cpp`, `…/program_spec/program_spec.cpp`); never part of
  the commit.
