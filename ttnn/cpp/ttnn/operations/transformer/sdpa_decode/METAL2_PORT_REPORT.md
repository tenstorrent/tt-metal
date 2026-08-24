# Metal 2.0 Port Report — sdpa_decode

## Outcome
**PORTED** — the single `SdpaDecodeProgramFactory` (paged / MLA / sharded / sliding-window / attention-sink /
geometry-override branches all in one factory) is fully converted to Metal 2.0 (`ProgramSpecFactoryConcept`).
All host build targets compile; the op's unit + nightly tests pass. Legality checks were forced and proven
live (`METAL2_CHECKS_FORCED` present on both translation units in every test run); the forcing scaffolding was
reverted before commit (tt_metal diff empty).

No-regression test results (Wormhole, `run_safe_pytest.sh`, checks forced):
- `unit_tests/operations/sdpa/test_sdpa_decode.py` — 11 passed, 1 skipped
- `unit_tests/operations/sdpa/test_paged_sdpa_decode_flexible_geometry.py` — 10 passed
- `unit_tests/operations/sdpa/test_bounded_sliding_kv_cache.py` — 7 passed
- `unit_tests/operations/sdpa/test_mla_decode.py` — 2 passed (MLA borrowed-Q / reuse_k path)
- `nightly/.../sdpa/test_sdpa_decode_sink.py` — 15 passed, 3 skipped
- `nightly/.../sdpa/test_sdpa_decode_cache.py` — 10 passed
- `nightly/.../sdpa/test_sdpa_decode.py` — 74 passed, 14 skipped

## Provenance
- **Recipe docs (this port):** `c9ef66ee339 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `c9ef66ee339 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory
### Concept realized
`ProgramSpecFactoryConcept` (base) — as the audit chose. No `override_runtime_arguments` (the framework
refreshes tensor bindings on cache hit). The op was a **direct-descriptor** op (`create_descriptor` static
member, no `program_factory_t`), so **exception 3** was applied: the factory body moved into a nested
`SdpaDecodeProgramFactory` struct with `using program_factory_t = std::variant<SdpaDecodeProgramFactory>;`
and a trivial `select_program_factory` returning it. `validate_on_program_cache_miss` /
`compute_output_specs` / `create_output_tensors` on the device-op are untouched.

### Device-op-class edits
- Direct-descriptor → conventional factory (exception 3): nested struct + variant + `select_program_factory`
  added to `sdpa_decode_device_operation.hpp`; `create_descriptor` declaration removed.
- Pybind entry points removed: **none** — `sdpa_decode_nanobind.cpp` binds only the four user functions;
  no `create_descriptor` was exposed.
- Custom `compute_program_hash`: **none** (was removed historically). Nothing touched.

### Open items
- No `TensorParameter` relaxations applied (audit said none; strict matching kept). No relaxation candidate
  spotted worth flagging.
- The op would benefit from typed / `std::array` kernel arguments once available — the tree-reduction
  `children_per_round[6]` block is currently six individually-named RTAs (see Friction).

## Handoff points
- **`c_11` (`cb_col_identity`) dead code (ops team).** The writer unconditionally fills c_11 via
  `generate_bcast_col_scalar` but sdpa_decode's compute never consumes it (it reduces via `reduce_c` on
  c_5). It is consumed by sdpa **prefill**'s `matmul_reduce`, so this is carried-over dead code from a
  matmul-based reduce path. Preserved as a single-toucher self-loop DFB (zero functional change per the
  audit); removing the host allocation + writer fill is a separate ops-team cleanup, out of port scope.

## Successes
- **Direct-descriptor exception 3** (`ttnn_factory.md`): the recognition signal (no `program_factory_t`,
  `create_descriptor` on the struct) matched exactly; the nested-struct + variant procedure landed the op on
  the same shape as other ported ops with no concept error.
- **Multi-binding `c_16`** (`port_patterns.md` — the tree-reduction bidirectional-reuse case the audit
  flagged): binding writer P+C and compute P+C with `allow_instance_multi_binding=true` validated and ran
  correctly, including the GQA / tree-reduction configs (`test_sdpa_decode` nkv=8, `test_sdpa_decode_sharded`).
- **`DataflowBuffer(uint16_t)` low-level ctor**: let the shared `dataflow_common.hpp` helpers keep their
  `uint32_t cb`-id NTTP signatures — callers pass `dfb::name` (constexpr→uint32_t) and the helper constructs
  `DataflowBuffer dfb(dfb_id)`. Minimal-diff win; no helper-signature churn.
- **Legality forcing proven**: `METAL2_CHECKS_FORCED` appeared on both `BuildProgramFromSpec` and
  `SetProgramRunArgs` in every test run, so every green is a validated green (the spec passed
  `ValidateProgramSpec` with the multi-binding flag, borrowed DFBs, and 30+ DFBs).

## Friction
### Gaps
- **`sem::name` genfiles reality vs. recipe.** The recipe's boundary note says `sem::name` "has no implicit
  conversion to `uint32_t` today." In fact genfiles (`tt_metal/jit_build/genfiles.cpp`) emits each
  `sem::<name>` as a plain `constexpr std::uint32_t <name> = <id>u;`. So it *is* a `uint32_t` id and flows
  freely into `Semaphore<>(sem::x)`, `get_semaphore(sem::x)`, and through a plain `uint32_t` struct field
  (`KMcastParams.mcast_sem_id`) with no wrapper or bridge. This made the reducer/output/k_mcast semaphore
  ports trivial (no restructuring). Recommend the recipe note be softened for `sem::` (still true for
  `tensor::`, which is a real `TensorBindingToken`). This was the single biggest "expected-hard, actually-easy"
  surprise.
- **Shared donor `read_page_table_for_batch`** (`../sdpa/.../dataflow/dataflow_common.hpp`, prefill-shared)
  constructs `TensorAccessor(args, addr, page_size)` internally — a 3-arg form that cannot take a binding
  token, and the donor is out of port scope. The recipe's shared-kernel Caution would have me fork the whole
  (huge) donor. Instead the ~6-line page-table read was **inlined** into sdpa_decode's own reader using
  `TensorAccessor(tensor::page_table)`; the donor is untouched (still serves prefill). The recipe could add a
  note that a tiny donor helper whose only incompatibility is internal accessor construction is cheaper to
  inline in-op than to fork.
- **`if constexpr` name-lookup on `tensor::`/`dfb::` in a header shared by the writer TU.** `read_q` lives in
  sdpa_decode's own `dataflow_common.hpp`, which the writer also includes. Two-phase lookup means a
  non-dependent `tensor::q` in `read_q`'s body would be looked up even in the writer TU (which does not bind
  `tensor::q`) → compile error. Resolved by keeping the k/v/mask pattern (accessor passed as a dependent
  template param) for Q too, and moving the Q-locally-available reserve/push out of `read_q` into the reader
  behind `#ifdef Q_LOCALLY_AVAILABLE`. Worth a recipe note: shared in-op headers must take resource handles
  as dependent params, never reference `tensor::`/`dfb::` directly.

- **`cb_`→`dfb_` rename can silently create a name collision (cost me a long debug).** The self-audit's
  "rename `cb_*`→`dfb_*`" step, done as a blanket `cb_`→`dfb_` replace, collided with a local `DataflowBuffer`
  object I had *pre-emptively* named with a `dfb_` prefix: `DataflowBuffer dfb_col_identity(cb_col_identity)`
  became `DataflowBuffer dfb_col_identity(dfb_col_identity)` — the object now constructs from **itself**
  (an uninitialized garbage DFB id), which the compiler accepts with only a `-Wunused-but-set-variable`
  warning on the shadowed alias. It corrupted L1 on **every** config (the writer's `generate_bcast_col_scalar`
  runs unconditionally) and manifested as a dispatch-level device **hang** (watcher `k_ids: 0`), not a
  compile error. It cost significant debug time because the failure appeared *after* a fully-green run, so it
  read as device/cache corruption. Lesson for the recipe/self-audit: after the `cb_`→`dfb_` sweep, grep for
  `DataflowBuffer (\w+)\((\1)\)` self-construction and treat every new `-Wunused-but-set-variable` on a DFB
  alias as a collision, not noise. (The `--dev` watcher build separately over-ran the L1 kernel-config buffer
  for the `nh=64` config — a watcher-size artifact, not a port issue; non-dev fits and passes.)

### Confusion
- The volume of config-gated conditional bindings (7 `#define`s: `USE_ATTENTION_SINK`, `USE_CUR_POS_TENSOR`,
  `IS_PAGED_ATTENTION`, `SLIDING_WINDOW`, `HAS_BLOCK_PADDING`, `HAS_INTERMED_OUT`, `TILIZE_Q`, plus
  `Q_LOCALLY_AVAILABLE`, `REUSE_K`, and the mutually-exclusive `IS_CAUSAL`/`USE_ATTENTION_MASK` for the c_3
  producer flip) is high for one op. The Conditional-DFB pattern scaled but the bookkeeping (which flag gates
  which DFB reference, in which kernel) was the bulk of the kernel work. A per-op "conditional-resource
  matrix" template in the plan would have helped; I built one ad hoc in `METAL2_PORT_PLAN.md`.

## Open items for downstream
- **Placement change (behavior-preserving, flagged for reviewer attention).** Legacy placed all three
  kernels on the full `core_grid` and had idle cores early-return via the address RTA that becomes `0` for
  idle cores (`q_addr==0` reader, `out_addr==0` writer, `arg(0)==65` compute). Those address RTAs became
  `TensorBinding`s (auto-injected, never 0), so the idle signal vanished. The port instead places the single
  `WorkUnitSpec {reader, writer, compute}` on the **active core set** (`core_group`, `num_active_cores`
  cores) and drops the idle-args host loop and the three kernel idle early-returns. This is behavior-
  preserving: idle cores did no work, and no cross-core operation (K-multicast, tree reduction, sharded-output
  gather) ever targets an idle core — all targets are active reducer / output / reduction-group cores. It
  also satisfies "do not create kernels on unused cores." Reviewers should confirm no downstream depends on
  idle cores having the program loaded (none found).
- **`children_per_round[6]` as six named RTAs.** Per the audit ("fixed 6-count, per-round distinct field →
  nameable"), the tree-reduction children array is six named RTAs (`children_per_round_0..5`) read into a
  local array in the writer and compute kernels, rather than a vararg. A future `std::array` typed-arg would
  collapse these back to one argument.
- **Retained varargs.** Genuine indexed-collection coordinate arrays kept as varargs: reader
  `all_output_noc_x/y`; writer `reduction_group_core_xs/ys`, `all_reducer_noc_x/y`, `all_output_noc_x/y`.
  Data-indexed (by `cur_batch` / `reduce_core_index` / `parent_core_in_group`), so genuinely un-nameable.
- **Shared-kernel touches (coordination signal).**
  - `../sdpa/.../dataflow/dataflow_common.hpp` and `.../compute/compute_common.hpp` (prefill-shared donors):
    reused unchanged via the `dfb::name`→`uint32_t` / raw-pointer bridge (their helpers take `uint32_t cb` and
    raw ptrs). NOT forked. Remaining consumer: sdpa prefill (unchanged). The page-table 3rd-arg drop was made
    by inlining, not by editing the donor.
  - `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp`: an existing `_metal2` fork (rung 1
    reuse) — the writer binds it (`generate_bcast_col_scalar(DataflowBuffer&, uint32_t)`). No new fork created.
  - `ttnn/cpp/ttnn/kernel_lib/{tilize_helpers,untilize_helpers,reduce_helpers_dataflow,l1_helpers}.hpp`:
    lib-owned, bridged cleanly; not forked.
  sdpa_decode's own `dataflow_common.hpp` / `rt_args_common.hpp` are private to this op (only its reader/writer
  include them), so they converted in place with the port; `rt_args_common.hpp` is pure math and was unchanged.
- **`unpack_modes` for Float32-format compute-consumed DFBs.** When `fp32_dest_acc_en` is set, the port adds
  explicit `UnpackMode::UnpackToSrc` entries (legacy default) for each Float32-format DFB the compute kernel
  consumes (q/k/v/mask/scale/zero/q_rm/sliding/block-pad, per config). This mirrors legacy behavior (legacy
  set no `unpack_to_dest_mode`, i.e. all `Default`); it is required by the Metal 2.0 validator, not a change.
