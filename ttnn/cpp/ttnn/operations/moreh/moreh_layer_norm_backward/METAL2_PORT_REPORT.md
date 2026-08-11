# Port Report — `moreh_layer_norm_backward`

## Outcome

**`PORTED`** — both device-operations in this directory converted to
`ProgramSpecFactoryConcept` and their tests pass:

- `MorehLayerNormBackwardGammaBetaGradOperation::MorehLayerNormBackwardGammaBetaGradFactory` (3 kernel sources)
- `MorehLayerNormBackwardInputGradOperation::MorehLayerNormBackwardInputGradFactory` (5 kernel sources —
  writer plus both runtime-selected reader/compute pairs)

No factories left on the legacy concept; nothing in the directory still builds a `ProgramDescriptor`.

Brought up to date with `main` after the port was written; see
[Catching up with main](#catching-up-with-main) for the two compute-API renames that required a
follow-up commit. Tests re-run green afterwards.

## Provenance

- **Recipe docs (this port):** `925d3c36ce9 2026-08-07 docs(metal_2.0): permit repairing what a pass falsified, and only that`
- **Audit docs (inherited):** `a38e7b405db 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

*(First line from `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
at the start of the port; second copied verbatim from `METAL2_PORT_BRIEF.md`.)*

**The docs moved between the audit and the port** — 20 commits, `a38e7b405db..925d3c36ce9`. Two of them
changed what this port had to do, and both were followed: `3eba8861e37` added the dropped-`TT_FATAL`
check to the anti-pattern self-audit (ran it; no delta) and `258d1ddc0ce` added the leftover-`cb`-name
sweep (ran it; see the *Grep-check caveat* under Open items). The rest are audit-side or Quasar-side.

**Not run: the `ai/post_port/` passes.** Four post-port procedures landed in that window
(`pass_procedure.md`, `semantic/gen2_hardware_configs.md`, `style/sync_free_dfbs.md`,
`style/dm_self_loop_dfbs.md`). They are a separate workflow that `metal2_port.md` does not invoke and
the invoker did not ask for, so this port did not run them. Flagging it because the op is now a
candidate for at least the style passes — it has 13–14 self-looped compute DFBs, though **no** DM
self-loop and no sync-free (raw-pointer) DFB, so the two style passes may well find nothing.

## Catching up with main

The branch was brought up to date with `main` after the port was written, and the port needed one
adaptation commit on top. Recording it here because it is the kind of churn the next porter of a moreh
op will hit.

The branch was rebased onto `main`, so the port sits directly on it and the PR diff is this op
directory and nothing else. (An earlier round integrated `main` with a merge commit instead, to avoid
rewriting the published branch; that merge is gone now that the branch was rebuilt.)

**What main changed under this op:** nothing structural — two mechanical renames in the compute-kernel
API, applied to the legacy kernels on main while this port was in flight. Exactly three files
conflicted (the three compute kernels), and every conflict hunk had the same shape: main had renamed
the helper, the port had renamed the operands.

| legacy name (what the port was written against) | name on main |
|---|---|
| `binary_op_init_common(icb0, icb1, ocb)` | `compute_kernel_hw_startup(icb0, icb1, ocb)` |
| `sub_bcast_cols_init_short_with_dt` | `sub_bcast_cols_init_with_dt` |
| `sub_tiles_bcast_scalar_init_short_with_dt` | `sub_bcast_scalar_init_with_dt` |
| `mul_bcast_cols_init_short_with_dt` | `mul_bcast_cols_init_with_dt` |
| `mul_tiles_bcast_scalar_init_short_with_dt` | `mul_bcast_scalar_init_with_dt` |
| `mul_bcast_rows_init_short_with_dt` | `mul_bcast_rows_init_with_dt` |

Note the two `*_bcast_scalar_*` rows also drop the `_tiles` infix, so they are not a uniform
`_init_short_with_dt` → `_init_with_dt` substitution. `binary_op_init_common` is still present on main
but `[[deprecated]]` (removal announced for after 2026-09-15); main's own kernels had already moved off
it, so the port follows main rather than leaning on the deprecated spelling.

**How it was resolved.** The conflicts were resolved to the *ported* version — keeping the Metal 2.0
structure and temporarily keeping the pre-rename helper names — and the renames were then applied as a
separate follow-up commit. That keeps the port
commit readable as "the Metal 2.0 transformation" and isolates "what main's churn forced" into its own
diff, rather than blending the two.

**Nothing else in the port was affected.** The factories, both readers, both writers, the device-op
headers and the op-local helper header replayed without conflict. `metal_v2_artifacts.hpp`,
`kernel_spec.hpp`, `program_spec.hpp`, `program_run_args.hpp`, `dataflow_buffer.h`,
`experimental/kernel_args.h`, `reduce_helpers_compute.hpp` and the *dataflow* `moreh_common.hpp` are all
unchanged across that range. The one `metal2_host_api` change in it
(`advanced_options.hpp`) is comments plus a compile-time-vararg field this op does not set.

**The recipe docs are not in this PR.** The port ran against the `metal_2.0` recipe tree, which lives on
its own doc branch and is deliberately never merged to `main`; the Provenance hashes above name commits
there, not commits in this PR. An earlier round of this branch carried those ~97 doc commits along, which
would have landed the whole recipe tree on `main`; they have been dropped.

**Equivalence check on the resolution.** Because the port is a syntax-only swap, the *ordered sequence*
of LLK / helper calls in each ported compute kernel should be identical to main's legacy version of the
same file. It is, for all three: 143 calls (gamma_beta_grad), 219 (input_grad small), 300 (input_grad
large), matching main position for position. That is the check that would have caught a conflict
resolution which silently dropped or reordered an init call.

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit decided, for both device-operations. No re-decision, no friction
with the concept fit: each op is single-program, has no op-owned device tensors, and needs no tensor-arg
relaxation.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — neither device-operation defined one (no
  `compute_program_hash`, no `to_hash`, no `attribute_values`). They were already on the default
  reflection-based hash.
- **Pybind entry points removed:** none. `moreh_layer_norm_backward_nanobind.cpp` binds only the composite
  op; nothing pybinds `create_descriptor`.
- The only header edits are the two the port forces, in each of the two device-op headers: the factory
  method signature (`static tt::tt_metal::ProgramDescriptor create_descriptor(...)` →
  `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`) and the include swap
  (`<tt-metalium/program_descriptors.hpp>` → `"ttnn/metal_v2_artifacts.hpp"`). Everything else in the
  device-operation classes is byte-identical.

### Open items

- **Tensor-arg relaxation candidates: none applied, one worth a look.** The InputGrad reader and writer are
  written shape-agnostically (they iterate page-by-page off runtime `num_inner` / `Wt` counts) and the op's
  kernels carry no `ArgConfig::Runtime*` uses, so strict matching was kept everywhere. The op's cache
  equivalence is therefore per-shape, which is correct but narrow. Not a port-time call.
- **`is_groupnorm` is dead weight in the cache key and the CTA list.** Both factories hardwire
  `const bool is_groupnorm = false` and pass it to the compute kernels, which carry live groupnorm branches
  that can never be taken. See *Open items for downstream* below.

## Handoff points

**None.** The port stayed entirely inside the op's own directory:

- No shared kernel was touched, forked, or reused — all eight sources live in this directory and are bound
  only by these two factories.
- No out-of-op call site required a `sem::` or `tensor::` handle. The three donor headers the kernels call
  into all take operands the converted code can supply:
  - `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` and `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
    take `DataflowBuffer` objects already (`fill_cb_with_value`, `generate_mask_h`, `generate_mask_h_w`, and
    the whole `*_init_with_dt` / `pack_tile_with_dt` family) — the ported kernels pass the objects they
    construct, unchanged.
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` — `compute_kernel_lib::reduce` takes its three
    buffer ids as `uint32_t` **non-type template parameters**; `DFBBindingToken::operator uint32_t()` is
    `constexpr`, so `dfb::name` (and a `constexpr auto` alias of one) works in that position with no shim.
- No kernel-lib gap, no framework gap, no `GlobalCircularBuffer`, no Case 2 binding, no missing DFB API.
- No pybind surface removed.

## Successes

- **[Same-FIFO aliasing (one DFB, multiple kernel-side names)](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
  fired exactly as written, on the hardest part of this port.** The two InputGrad compute kernels reuse
  three scratch buffers under nine different phase names (`xmm`, `dyadd`, `ydy`, `ydyadd`, `ndy`,
  `ndymdysum`, `yydysum`, plus `recip_nrstd` and `tmp4` in the large kernel), and the legacy code
  constructed a *fresh* `DataflowBuffer` object for each name. The obvious-looking move — one
  `DataflowBufferSpec` per name, mutually `alias_with`-ed — is exactly the bug the entry's comparison table
  warns about: it would have produced independent FIFOs at one address and silently lost the pointer
  coherence the kernel depends on (it pushes through one name and pops through another —
  e.g. `moreh_layer_norm_backward_input_grad_large_kernel.cpp:252`/`:267` push `dfb_dyadd_obj` (tmp1) and
  the reduce at `:330` consumes it through the `dfb_dyadd` alias). The catalog's prescription — one
  spec, one binding, `constexpr auto` handle
  aliases, one object — is what the port implemented, with the per-phase objects becoming
  `DataflowBuffer&` references to the one real object (e.g.
  `moreh_layer_norm_backward_input_grad_small_kernel.cpp:109-110`).
- **The `#ifdef`-over-`if constexpr` rule caught a compile error before it happened, in a place the brief
  had already flagged.** `moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` FIFO-touches a mask-w
  buffer the factory has never allocated (`is_groupnorm` is hardwired false). Under legacy that was merely
  dead code; under Metal 2.0 `dfb::mask_w` simply does not exist, and
  [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  is explicit that a `constexpr bool do_mask_w = false` would not stop the name lookup. The path is now
  gated on a `DO_MASK_W` define the factory never emits (`…_gamma_beta_grad_kernel.cpp:31-36` and the four
  `#ifdef DO_MASK_W` use sites at `:79`, `:110`, `:181`, `:298`) — zero functional change, and the
  groupnorm scaffolding stays intact per decision D1.
- **The recipe's insistence on an explicit compute `opt_level` is a real save.** Nothing in this directory
  sets `opt_level`, so `grep -n opt_level` over the legacy source returns nothing and the field reads as
  "not a thing this op cares about". It resolves to `O3` for a `ComputeConfigDescriptor` and Metal 2.0
  defaults to `O2`, so all four compute `KernelSpec`s would have quietly dropped a level. Nothing in the
  build or the tests would have said so.

## Friction

### Gaps

- **The brief's `unpack_modes` "inputs carry the io dtype, not Float32" claim needs a stated premise.**
  Brief item 1 asserts that `c_0`–`c_7` never need an entry. That is true, but only because the op's
  `check_tensor` calls take the **default** `data_types = {DataType::BFLOAT16}`
  (`ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.hpp:153`, and `:162` for the
  optional-tensor overload), so the factories' `dfb_data_format` cannot be Float32. Nothing in the brief
  says where that guarantee comes from, and the porter has to go find it —
  the alternative being to hand-list entries for the io buffers too, "just in case", which the same
  section forbids as guessing. **Suggested doc fix:** where the audit derives an `unpack_modes` count, have
  it record the dtype premise (`io dtype is X because <validation site>`), so the count is checkable rather
  than trusted. Recorded in `METAL2_PORT_PLAN.md` so a future dtype widening knows to revisit it.
- **The recipe says nothing about how to consolidate objects when a same-FIFO alias is a *local object*,
  not just a name.** [Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
  says "alias the *handle*, keep *one* object", and shows the handle alias — but its example has one
  `DataflowBuffer` at file scope already. Here the legacy code had **nine** phase-local
  `DataflowBuffer x_obj(cb_x);` declarations to collapse, inside loop bodies, each carrying the comment
  that documents what the buffer holds during that phase. Rewriting them as
  `DataflowBuffer& dfb_xmm_obj = dfb_tmp2_obj;` keeps the diff to one line per site and preserves both the
  call shape and the comment, but the porter has to invent that. **Suggested doc fix:** add the
  reference-alias line to the entry's *Correct port* block, beside the `constexpr auto` handle alias.

### Confusion

- **Brief item 9 is wrong about which factory re-resolves its compute config.** It states that InputGrad
  re-resolves via `init_device_compute_kernel_config` inside the factory and that "GammaBetaGrad does not do
  this — don't 'harmonize' them." Both do:
  the call now at `moreh_layer_norm_backward_gamma_beta_grad_program_factory.cpp:86-87` is the same one as at
  `moreh_layer_norm_backward_input_grad_program_factory.cpp:98-99`, and both were there pre-port
  (legacy `:34-35` and `:33-34` respectively). Followed the source, not the brief:
  each factory translates the value it actually resolved. The instruction not to harmonize them was
  followed anyway, since there was nothing to harmonize. Cheap to catch, but it is exactly the kind of
  claim a porter is told to inherit rather than re-derive, so it is worth the audit's attention.
- **`opt_level` is a `KernelDescriptor` field, and `ComputeConfigDescriptor` has none — the recipe warns
  about this and it still nearly reads the wrong way.** The recipe's *Compiler options* section says to
  `grep -n opt_level` rather than read `config`, and explains that an absent field resolves per kernel
  *type*. Having read that, the natural next thought on seeing a bare `ComputeConfigDescriptor{...}` with
  four fields is still "the op didn't ask for anything special here." The point that survives is the
  table — **legacy compute default O3, Metal 2.0 default O2** — not the grep. **Suggested doc fix:** lead
  that section with the table rather than closing with it.

## Open items for downstream

- **Shared kernel touches: none.** Every one of the eight kernel sources lives in this op's directory and is
  bound only by these two factories (verified with `grep -rl <filename> ttnn/cpp/ttnn/operations/`). No
  `_metal2` fork was reused or created, no pointer comment was added anywhere, and no peer op's directory
  was written to. There is no remaining-consumer list to track and no fork to sunset.
- **A likely bug in the pre-existing small InputGrad compute kernel, left as-is.** At the very end of
  `moreh_layer_norm_backward_input_grad_small_kernel.cpp` the mask buffer is released with a second
  **`wait_front(2)`** where the large kernel does **`pop_front(2)`**
  (`…_small_kernel.cpp:482` vs `…_large_kernel.cpp:632`). It is harmless today — per-execution DFB
  state is reinitialized at each enqueue, and the kernel is about to return — but it reads as a
  copy-paste slip, and it is the sort of thing that stops being harmless if the surrounding loop structure
  ever changes. **Preserved verbatim** (the legacy kernel is the source of truth for what the op does);
  flagging for the op owner.
- **`is_groupnorm` is hardwired `false` in both factories while all three compute kernels carry live
  `is_groupnorm` branches** (`…gamma_beta_grad_program_factory.cpp:102`,
  `…input_grad_program_factory.cpp:115`). Roughly a third of each compute kernel is unreachable, and the
  flag still occupies a compile-time argument, so it also widens the JIT cache key for no benefit. Either
  the groupnorm path should be wired up or the scaffolding retired; settled as out-of-scope for this port
  by invoker decision D1, so it needs a decision from the op owner.

  **This surfaced in review.** A reviewer asked whether the `#ifdef DO_MASK_W` blocks in
  `moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp` — which the port's own comment describes as
  dead — could simply be deleted. They can, in the sense that nothing reaches them; but mask_w is dead
  *only* because `is_groupnorm` is false, and that same one reason kills 11 other `is_groupnorm`
  branches across the three compute kernels (6 / 2 / 3) plus the `is_groupnorm` compile-time argument
  and the `do_mask_w` derivation. Deleting mask_w alone would leave the scaffolding half torn down.
  Declined for this PR and the kernel comment was expanded to say so, so the next reader does not have
  to re-derive it. Whoever takes the groupnorm decision should treat all of that as one change.
- **`packer_l1_acc` is destructured from `get_compute_kernel_config_args` and dropped** in both factories
  (`…gamma_beta_grad_program_factory.cpp:139`, `…input_grad_program_factory.cpp:163`). Not a port artifact —
  Metal 2.0's `ComputeGen1Config` has no field for it either, so there was nothing to carry across. If a
  user sets `packer_l1_acc` on this op's `compute_kernel_config` today it is silently ignored, exactly as
  before.
- **Two `log_info(tt::LogTest, …)` calls fire on every program-cache miss at the wrong severity**
  (`…input_grad_program_factory.cpp:203`, `:208`). `LogTest` is not the right channel for a production
  factory and `log_debug` is the right level. **Deliberately not touched:** with no committed test reaching
  the large algorithm, the `_large` line is the only signal that the path was exercised at all, and this
  port used it as exactly that (see the test-coverage note below). Worth tidying once the large path has
  real coverage — the two changes belong together.
- **`ttnn::to_compute_hardware_config` cannot reach `unpack_modes` or `bfp_pack_precision_mode`,
  so every caller re-derives the same `std::get<ComputeGen1Config>` + `TT_FATAL` guard.** This port copied
  `gen1_compute_config` from
  `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_distributed_metal2_helpers.hpp:73`
  into its own op-local header
  because the peer op's header is out of scope to include — the second op to need it, and the helper is
  neither op-specific nor a judgment call. It reads like it belongs beside `to_compute_hardware_config` in
  `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp`. (`unpack_via_src` should
  **not** move there — per the invoker's policy the per-DFB listing has to stay visible in each factory's
  diff — but the Gen1 resolution is pure boilerplate.)
- **Grep-check caveat for the next porter of a moreh op.** The recipe's "no `cb` survives" sweep
  (`grep -rnE '[Cc][Bb]_|_[Cc][Bb]\b|\b[Cc][Bb]\b|\bCB[A-Z]'`) does not come back clean here, and cannot:
  the shared dataflow helper the readers call is named **`fill_cb_with_value`**
  (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:98`). It is out-of-directory framework-adjacent code the
  port may not rename, so the three call sites are expected hits, not leftovers. Every other hit was
  renamed. If `moreh_common.hpp` is ever swept for Metal 2.0 naming, these call sites come with it.

### Test coverage notes

All runs on **Wormhole n300**, from a `./build_metal.sh --build-tests` build of this branch. Every
number below was re-confirmed **after** merging main in; the pre-merge run gave the same counts.

- **The gate, pre- and post-port.** All six backward tests, per the brief's test gate:
  `pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py -k backward -v`
  - pre-port: **28 passed, 26 skipped, 45 deselected**
  - post-port: **28 passed, 26 skipped, 45 deselected** — identical, no test changed status.

  The 26 skips are the file's `bfloat8_b` parametrizations, which the tests skip themselves
  (`pytest.skip("bfloat8_b is not supported in the kernel")`); they skip identically before and after.
- **Full-file regression (forward + backward):** `pytest …/test_moreh_layer_norm.py -v` →
  **50 passed, 49 skipped**, post-port and again after merging main in. No forward test regressed.
- **`_large` algorithm — verified locally, deliberately not committed** (invoker decision D2b).
  No committed parametrization reaches `use_large_algorithm`
  (`dfb_usage >= available_L1`, `…input_grad_program_factory.cpp:200`) — the file's widest normalized
  region is 64 elements, and the threshold needs a few hundred tiles.
  - **Shape used:** `([2, 16384], normalized_dims=1)` → `num_inner = 512` tiles, so the small
    algorithm's intermediate footprint `(2 * num_inner + 6)` tiles alone is ~2.1 MB against ~1.4 MB of
    usable L1 per core. Run for both `elementwise_affine=False` and `True`, so both the
    `GAMMA_HAS_VALUE` and the no-gamma build of the large kernels were exercised, and the
    GammaBetaGrad factory came along on the affine run.
  - **Large path confirmed selected:** the log line
    `"Large moreh_layer_norm_backward_input_grad algorithm is selected."` appears **twice** (once per
    parametrization) and `"Small …"` **zero** times.
  - **Result: 2 passed**, both before and after merging main in. Both ported `_large` sources produce
    correct numerics against the torch reference at the same `rtol=0.1 / atol=0.5` the committed tests
    use.
  - The scaffold file was deleted after the run and is not in the diff.
- **Gap worth closing: the large InputGrad path has no committed coverage at all.** Both of its sources
  (`reader_…_input_grad_large.cpp`, `…_input_grad_large_kernel.cpp`) are now Metal 2.0 and would break
  silently on any future change — the local run above is the only thing that has ever exercised them
  here. A single wide-shape parametrization on `test_moreh_layer_norm_backward` (the `[2, 16384]` /
  `nd=1` case above works and costs ~3 s) would cover it. Adding one is a test change, not port work,
  so it is left here rather than bundled.
