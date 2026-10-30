# Port Report — deepseek_moe_gate

Op: `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate`, factory `DeepseekMoeGateProgramFactory`
(the op's only factory). Shorthand: `factory.{hpp,cpp}` = `device/deepseek_moe_gate_program_factory.{hpp,cpp}`;
`builder.{hpp,cpp}` = `device/deepseek_moe_gate_program_descriptor_builder.{hpp,cpp}`; `kernel` =
`device/kernels/deepseek_moe_gate_kernel.cpp`; `gate.hpp` / `utils.hpp` =
`device/unified_kernels/{deepseek_moe_gate,kernel_utils}.hpp`; `hw_startup.h` =
`device/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h`. Line numbers are
post-port unless marked "pre-port" (pre-port = `b5ec776b097`).

## Outcome

**PORTED.** `DeepseekMoeGateProgramFactory` is on `CustomProgramSpecFactoryConcept`. On Blackhole, with the Metal 2.0
legality checks forced on and Watcher on, the unit test passes before and after. The direct op capture is bit-exact
against the pre-port baseline — 328/328 full output tiles when device history is matched, and the answer region
328/328 in every run — and the `TTMoEGate` capture is 18/18 identical (details below). **Wormhole: unverified**
(no WH hardware on this bench).

Two invoker-approved choices are worth a reviewer's eye: the DFBs are declared in legacy CB-index order because an
untouched op-local LLK helper hard-codes slot 0 (Handoff 1), and one guard of the legacy override is recorded as
subject-deleted, which the recipe says an override guard never is (TTNN ProgramFactory; Friction 5).

## Provenance

- **Recipe docs (this port):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`
- **Audit docs (inherited):** ``c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`` (from `METAL2_PORT_BRIEF.md`)
- Code: branch `iwrosz/deepseek_moe_gate_2.0`; the port sits on `b5ec776b097` (recipe branch `c41f7f4ef0a` +
  `origin/main` `012c19beaae` + the audit commit). Reference port (invoker-supplied, read for spec shape only):
  `eb6bb8ffffc` (convert_to_chw).

## Verification

Bench: Blackhole p150b, chip 0 only (`TT_VISIBLE_DEVICES=0` under the shared device lock), compute grid 11×10,
`TT_METAL_WATCHER=10` on every run, a fresh `TT_METAL_CACHE` per run. Legality checks: `skip_validation = false`
forced at all nine sites under `tt_metal/impl/metal2_host_api/` (five in `program_run_args.cpp`, four in
`program_spec/program_spec.cpp`) with one marker warning in each file; working tree only, reverted before the commit.

Confirmed test set (invoker-approved): (A) the op's unit test; (B) a scratch bit-exact capture of the op itself;
(C) a scratch capture of `TTMoEGate` on a 1×1 mesh. Not run, because they need a 4×8 Galaxy or a T3K+/TG mesh with
HF weights: `models/common/tests/moe/test_tt_moe_gate.py` (its `n_group=8` configs) and the deepseek_v3 model tests
(`test_moe_gate.py`, and transitively `test_moe.py`, `test_decoder_block.py`, `test_mtp.py`,
`fused_op_unit_tests/moe/test_ds_moe.py`). No C++ gtests or sweeps exist for this op.

| run | pre-port (cold build of `b5ec776b097`) | post-port, checks forced | post-port, final build (no scaffolding) |
|---|---|---|---|
| (A) `tests/ttnn/unit_tests/operations/experimental/deepseek/test_deepseek_moe_gate_program_cache.py` | 1 passed | 1 passed; markers 1 + 1 (one miss; the hit runs through the forced `UpdateProgramRunArgs`) | 1 passed; 0 markers |
| (B) capture, two back-to-back runs | 41/41 configs ran; run 1 vs run 2 identical (328/328 full-tile hashes) | 41/41; markers 82 per run (2 per miss) | 41/41; 0 markers |
| (B) pre vs post, run 2 (matched device history) | — | **328/328 full-tile, 328/328 answer-region** | **328/328 full-tile, 328/328 answer-region** |
| (B) pre vs post, run 1 | — | 328/328 answer-region; 318/328 full-tile (see below) | same |
| (C) `TTMoEGate`, `n_group=8` configs `deepseek_v3`, `deepseek_v3_single_glx`, `ling_1t`; forward(x1), forward(x2), forward(x1) | identical run to run (18/18); 3 op calls per config | **18/18 identical to pre-port in both runs**; same cache-entry deltas (10 / 0 / 2) | — |

(B) in detail: 41 configs — core counts {1, 2, 14, 32, 110} × `enable_sigmoid` × (eps, scale) ∈ {(1e-20, 2.5),
(1e-6, 1.0)} × output layout {TILE, ROW_MAJOR (the deepseek_v3 decode path's)}, plus one column-major-sharded config.
Each is dispatched four times: cache miss; all five tensors re-allocated at new addresses (cache hit); input only
changed (`TTMoEGate`'s loop pattern); dispatch 1's data replayed at new addresses. Recorded per dispatch: SHA-256 of
the full output tiles and of the answer region (row 0, columns 0..7 of scores and indices), the buffer addresses,
and the program-cache delta. Both builds: exactly 1 cache entry per config (41 total); the replay's answer equals
the miss's in 41/41 configs; addresses distinct across dispatches in 41/41.

**The full-tile hash depends on device history.** This kernel leaves data from earlier runs in the lanes it does not
write (the tt-llk test source documents it: `tt_metal/tt-llk/tests/sources/deepseek_moe_gate_test.cpp`, the
"Determinism sanitize" comment). So the first dispatch on a core inherits whatever the previous process ran there.
The 10 run-1 differences are exactly the cache misses of configs 0, 8, 16, 24 and 32 — the first touch of each new
core range — and the legacy op shows the same 318/328 against itself across processes. With matched history
(run 2 after an identical run 1 on each build) the full tiles are bit-exact, residue included.

Informational: the capture's golden check (the unit test's, applied to every row) passes 83 and fails 81 of 164
dispatches **identically on both builds** — 32/32 at 1 core, 28/32 at 2, 19/36 at 14, 4/32 at 32, 0/32 at 110 —
see Open items. Watcher logs: no errors in any run.

## TTNN ProgramFactory

### Concept realized
`CustomProgramSpecFactoryConcept`, as the audit chose. `create_program_artifacts` replaces `create_descriptor`, and
`override_runtime_arguments` drops its `Program&` parameter and returns `ProgramRunArgs` (`factory.hpp:18-28`).
The override (`factory.cpp:32-44`) returns a `TensorArgument` for **all five** io-tensor `TensorParameter`s
(input, bias, output, input_indices, output_indices — the legacy refresh order), no `kernel_run_args` (the op has
no RTAs or CRTAs) and no `dfb_run_overrides` (the legacy override set no size). `coord` is unused, as before.
No op-owned tensors.

Guards of the legacy override (pre-port `factory.cpp:21-30`, run once per tensor):
- `TT_FATAL(buffer != nullptr, …)` — **carried across** (`factory.cpp:21`). Its message named the CB index
  (`"… is null for CB {}"`); there is no CB index any more, so it names the tensor parameter instead.
- `TT_FATAL(cb_it != cbs.end(), "Missing globally allocated circular buffer {}", …)` — **recorded as
  subject-deleted** (invoker-approved). It searched `program.circular_buffers()`, and the Metal 2.0 override is not
  given the `Program`. This deviates from the recipe's "an `override_runtime_arguments` is never a
  subject-deleted case"; see Friction 5. The invariant it protected (each tensor has its tensor-backed buffer in the
  program) is now structural: the spec validator requires every `borrowed_from` to name a declared
  `TensorParameter`, and the run-args update rejects a `TensorArgument` with no parameter.

`TT_FATAL` census (vs `012c19beaae`): `device_operation.cpp` 41 → 41, `builder.cpp` 19 → 19 (all verbatim),
`factory.cpp` 2 → 1 (the guard above).

### Device-op-class edits
- Pybind entry points removed: none (`deepseek_moe_gate_nanobind.cpp` binds only the free function).
- Custom `compute_program_hash`: none — default reflection hash. The device-operation files, the op's top-level
  `.cpp/.hpp`, the nanobind file and the CMake files are byte-identical.

### Open items
- Relaxation candidates: none (all five parameters strict, relaxation `none`).
- The translated override now does exactly what `ProgramSpecFactoryConcept`'s automatic refresh does (all five
  tensor bindings, nothing else). Moving to the base concept would be a pure simplification — a TTNN / ops call,
  not taken here (the audit makes the same observation).

## Handoff points

1. **Literal CB index in an op-local LLK-API copy — Owner: op owners (deepseek MoE).** `hw_startup.h:20`
   (`deepseek_compute_kernel_init()`) calls `compute_kernel_hw_startup(0, 0, 0)`: slot 0 as srcA, srcB and pack
   operand, meaning the input buffer. The brief scoped `kernel_includes/**` out of the port, so the literal stays,
   and the port keeps its meaning by declaring the DFBs in legacy CB-index order (`builder.cpp:133-143`, with a
   comment saying why). DFB slots are assigned lowest-free in declaration order
   (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2182-2202`; `…/construction/resource/dfb.cpp:253-254`), so
   the five DFBs get slots 0-4 — the legacy CB indices — which also makes the compiled kernels' buffer ids identical
   to legacy. This leans on implementation behavior, not a documented contract. Suggested fix: give
   `deepseek_compute_kernel_init` an operand parameter and pass `dfb::moe_gate_input` from `kernel:55`; or, on the
   framework side, document the declaration-order slot assignment.

No kernel-lib gaps, no boundary-rule violations (no `sem::` / `tensor::` handle crosses out of the op), no removed
pybind surface, no framework gaps hit.

## Successes

- **"Force them and prove it"** (recipe, *Ensure the Metal 2.0 host-side legality checks are enabled*). Here TTNN
  derives `skip_validation` from `validate_program_args`, which is `false` in the default config
  (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1032`), so without the force every spec check in this port
  would have been skipped. The markers counted exactly two per cache miss in every forced run.
- **The brief's literal-slot heads-up** (audit Recipe note 7). Without it the first green could have been an
  accident of slot numbering. Choosing declaration order also kept every device buffer id equal to the legacy CB
  index, which is what made a full-tile bit-exact comparison meaningful.
- **Compiler options** (recipe, *Compiler options* and the absent-line self-audit item). The compute spec states
  `O3` (`builder.cpp:198`); `grep opt_level` pairs it with the one compute `KernelSpec`.
- **"The override owns the tensor bindings too"** (`ttnn_factory.md`). All five `TensorArgument`s are returned; the
  capture's re-allocated, input-only and replay dispatches exercise exactly this path, with addresses verified
  distinct.
- **Endpoint census.** Re-deriving it from the kernel matched the brief exactly (five plain 1P+1C DFBs).
- **Headers first.** The question of how to carry a token through a template (Friction 4) was answered by a
  framework header (`tt_metal/hw/inc/api/llk_operand_from_tokens.h:34, 58`), not by a precedent hunt.

## Friction

### Gaps
1. **The legality-forcing grep is one directory deep.** `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp`
   finds 5 of the 9 sites and misses all four in `program_spec/program_spec.cpp`, including `BuildProgramFromSpec`
   (the convert_to_chw port reported the same). Used `grep -rn`. Fix: make the recipe's command recursive.
2. **"If TTNN is behaving, you will find the checks already on"** does not hold on this tree: the default
   `validate_program_args = false` turns them off for every TTNN Metal 2.0 factory (see Successes). Worth saying
   plainly in the recipe, since it makes the force mandatory rather than precautionary.
3. **Hardware-config spellings are stale.** The recipe's `create_reader_datamovement_config(device->arch())` takes no
   arch here (`create_reader_datamovement_config(bool disable_dfb_implicit_sync_for_all = false)`); there are no
   `DataMovementGen1Config` / `ComputeGen1Config` types (`DataMovementHardwareConfig::DataMovement1XXConfig`, a
   flat `ComputeHardwareConfig` plus `config_1xx` / `config_12x`). The brief named the Metal-layer
   `CreateReaderDataMovementConfig()`; the recipe prefers the TTNN wrapper, which was used (same 1xx triple, plus
   the 2xx branch).
4. **No catalog entry for a unified kernel's CTArgs templates that carry CB ids as `uint32_t` template parameters**
   (pre-port `gate.hpp:53-79`). Passing `dfb::x` there compiles (the token converts to `uint32_t`), but the
   `DataflowBuffer` is then built through the raw-id constructor, against whitelist rule 3. A token cannot be a value
   template argument (private members make it non-structural); a reference to the namespace-scope `constexpr` token
   can. Used `const auto&` template parameters (`gate.hpp:53, 62-66`) with `static constexpr auto` copies, so
   `Op::impl()` builds each `DataflowBuffer` from its token (`gate.hpp:104-105, 119-123`) and the LLKs receive the
   token through its `uint32_t` conversion. `setup_sharded_buffer` takes a `DFBBindingToken<Pap, Cap>` instead
   (`utils.hpp:66-71`), as convert_to_chw did for its helper. A pattern entry would settle both shapes; the
   `generalized_moe_gate` op has the same structure.
5. **Deviation: an override guard with no Metal 2.0 subject.** The recipe says an `override_runtime_arguments` "is
   never a subject-deleted case". The legacy override's CB-lookup guard (pre-port `factory.cpp:28`) inspects the
   `Program`, which the Metal 2.0 override does not receive, so it cannot be carried across. Recorded as
   subject-deleted with the invoker's approval (TTNN ProgramFactory above). Suggested recipe wording: "… unless the
   guard inspects the `Program` object the Metal 2.0 override no longer receives."
6. **The self-audit's `cb` grep cannot reach zero here.** 35 hits are in the never-called helpers of `utils.hpp`
   (lines 85, 121-237), which the audit explicitly ruled out of scope; 12 more are false positives in the untouched
   `kernel_includes/` LLK copies (`SRCB_VLD`, `SRCB_NO_BCAST`, `srcb_dummy_valid`: the pattern's first alternative,
   `[Cc][Bb]_`, has no leading word boundary). Every other file is clean. Suggest `\b[Cc][Bb]_` and a note on
   audit-ruled dead code.
7. **The scaffolding self-audit grep matches the recipe itself** on a recipe-branch checkout: `git diff "$BASE"`
   shows 6 hits, all in `docs/…/metal_2.0/`. Scoped to `':!docs'` it is clean (convert_to_chw reported the same).
8. **Bit-exact verification needs a protocol the recipe doesn't describe.** Run-history residue in unwritten lanes
   (see Verification) makes naive full-tile hashing flag 10 of 328 outputs that are not port differences. Matching
   device history between the pre and post runs, or hashing the documented answer region, both work; a short
   paragraph in the recipe's verification section would save the next porter the investigation.
9. **Test discovery: almost all real coverage is mesh- or weights-bound.** The only single-device test covers one
   core, sigmoid on, two dispatches. The broader configs come from the scratch capture, not from in-tree tests.

### Confusion
10. **Rule 2's named-CTA note vs the self-audit's no-`cb` rule.** Rule 2 says `get_named_compile_time_arg_val("cb_in")`
    becomes `dfb::cb_in`; the self-audit then requires no `cb` in a DFB name. The legacy names carry a `_cb` type
    suffix (`moe_gate_input_cb`). Took `dfb::moe_gate_input` — the `dfb::` namespace now carries the type — keeping
    the `moe_gate_` prefix that the surviving named CTAs (`moe_gate_eps`, …) still use.
11. **Renames the API change forces, outside the documented `cb_*` → `dfb_*`.** `build_moe_gate_program_descriptor`
    → `build_moe_gate_program_artifacts` (it returns the new type), and the `kXxxCb` CB-index constants →
    `kXxxTensor` `TensorParamName`s (their successors, as the brief anticipated). File names unchanged, so
    `CMakeLists.txt` (co-owned by infra) is untouched — `builder.cpp` keeps its descriptor-era name.
12. **`DataflowBuffer` FIFO counts are `uint16_t`** where `CircularBuffer`'s were `int32_t`; every count here is 1.

## Open items for downstream

- **Shared kernel touches:** none. `kernel`, `gate.hpp` and `utils.hpp` are bound only by this factory (no other
  includer in the repo; `generalized_moe_gate` carries its own copies), so they were converted in place. No `_metal2`
  fork created; sunset list empty. `device/kernel_includes/**` is untouched, so the tt-llk test that includes six of
  its headers (`tt_metal/tt-llk/tests/sources/deepseek_moe_gate_test.cpp:141-147`) is unaffected.
- **Sibling op:** `experimental/deepseek/moe/generalized_moe_gate` has the same unified-kernel shape
  (`CTArgs` templates on CB ids, a `setup_sharded_buffer` copy); Friction 4's approach should carry over.
- **Golden mismatches on multi-row batches, legacy behavior** (unchanged by the port): applying the unit test's
  golden check per row fails for most dispatches above 2 cores on both builds, cache miss included (Verification).
  The pattern fits bf16 near-ties in the grouped selection (`test_tt_moe_gate.py` describes the same thing for the
  model path), and the unit test only exercises 1 core. Worth an owner's look: either the check's tolerance or a
  test-coverage gap.
- **Preserved, from the audit's Misc anomalies** (unchanged by the port): `moe_gate_is_active_core` is always 1, so
  the inactive-core branches are dead; the never-called helpers in `utils.hpp`; output shards outside the input grid
  keep their prior contents; the builder's subset re-validation (`builder.cpp:80-115`).
- **Quasar-uplift debt:** the Gen1-only `uint32_t` token conversion at every compute LLK call site
  (`gate.hpp:115-156`). The `const auto&` CTArgs carry the tokens themselves, so
  Quasar's `DataflowBuffer` still deduces their access patterns. No DM self-loop, no multi-binding flag.
- **Not run here:** the 4×8 Galaxy `TTMoEGate` test and the deepseek_v3 mesh model tests; Wormhole.
- **Working-tree scaffolding:** reverted before the commit (`git checkout --` on both files); the branch diff has no
  `tt_metal/` file.
