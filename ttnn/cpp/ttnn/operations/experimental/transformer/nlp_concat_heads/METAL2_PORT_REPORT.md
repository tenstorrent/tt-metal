# Port Report — nlp_concat_heads

Metal 2.0 port attempt for `experimental/transformer/nlp_concat_heads`.

**Result: grounded stop (successful capitulation). The factory stays entirely on legacy `create_descriptor`.
No code changed.** Tests not run — no code change to validate.

The op has a **single factory with runtime kernel-source selection** (`if (in_sharded)`). The recipe's
atomic-unit rule requires that a single runtime-source-selecting factory + **every** kernel source it can
select convert *together* — there is no half-Metal-2.0 factory that can dispatch to an unconverted kernel.
The sharded path uses only op-owned kernels and is cleanly portable on its own; the **interleaved path binds a
cross-op shared writer** the porter may neither edit nor fork. So the whole factory is blocked, and I left it
entirely on legacy rather than produce a half-converted factory that could select an unconverted kernel.

## TTNN ProgramFactory

### Concept realized
**None.** The factory remains on `ProgramDescriptorFactoryConcept` (`create_descriptor`).
`program_factory_t = std::variant<NLPConcatHeadsProgramFactory>` unchanged. `validate_on_program_cache_miss`,
`compute_output_specs`, `create_output_tensors` untouched.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op uses the default reflection hash — confirmed by grep).
- Pybind entry points removed: none (`nlp_concat_heads_nanobind.cpp` exposes only the user-facing op; no
  `create_descriptor` was pybound).

### Open items
- The port is fully blocked on the cross-op writer (Handoff points below). Once that writer is prepared for
  Metal 2.0 by its owner, this factory is a straightforward port: the sharded path mirrors the reference
  `nlp_concat_heads_decode` (borrowed-memory DFBs + fake-CB self-loops, two `KernelSpec`s of one source); the
  interleaved reader is a clean Case-1 `TensorBinding`.

## Handoff points

- **Cross-op shared writer needs Metal-2.0 prep by its owner (the blocker).**
  File: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`.
  Used by the interleaved path of this factory (`nlp_concat_heads_program_factory.cpp:130`). The kernel reads:
  - `constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);` — a **positional** CTA.
  - `constexpr auto dst_args = TensorAccessorArgs<1>();` — **`TensorAccessorArgs` baked into CTAs**.
  - `const uint32_t dst_addr = get_arg_val<uint32_t>(0);` — a **buffer-address RTA**.

  A Metal 2.0 factory emits no positional CTAs, cannot bake `TensorAccessorArgs` into CTAs, and does not thread
  a buffer address through an RTA — those are precisely the legacy plumbing patterns `TensorBinding` replaces.
  The implicit `dfb::name → uint32_t` conversion would rescue **only** the `cb_id_out` CTA; it does nothing for
  the `TensorAccessorArgs` / address-RTA dependence. So the kernel **cannot be used unmodified** by a Metal 2.0
  factory. Driving it would require a named-CTA / `TensorBinding` / `TensorAccessor(ta::out)` rewrite of the
  kernel.

  **Why this isn't a porter touch:** the kernel is cross-op shared — **31 co-borrower factories** depend on it
  (typecast, bcast, concat, copy, permute, reshape_on_device, slice, tilize / tilize_with_val_padding,
  transpose, embeddings, attn_matmul, matmul, reduce (h/w/hw/welford), prod, kv_cache, examples,
  gelu_bw/tanh_bw/gelu_backward, nlp_concat_heads_boltz, …). Editing in place would break every legacy
  co-borrower; the orchestrator's cross-op rule also forbids forking it for this port. The fix is upstream:
  the writer's owner prepares a Metal-2.0 form (named CTAs + `TensorBinding`/`ta::out`), at which point all
  borrowers can migrate. Tagged: **API: shared kernel requires Metal-2.0 prep.**
  - Sibling op `nlp_concat_heads_boltz` borrows the same writer on its interleaved path → same blocker; both
    unblock together.

## Successes

- **Atomic-unit rule fired correctly (`port_op_to_metal2_recipe.md` → "The atomic unit of a port is one
  ProgramFactory" / runtime kernel-source selection).** The temptation was to port the clean sharded path and
  "leave the interleaved path for later." The recipe is explicit that a single runtime-source-selecting factory
  + all its selectable sources convert together — a half-converted factory can select an unconverted kernel at
  runtime and crash. The rule turned a tempting partial port into the correct grounded stop. The reference op
  (`nlp_concat_heads_decode`) was a near-perfect template for the *sharded* path's intended shape (borrowed
  output DFB, fake-CB self-loop, two `KernelSpec`s of one source), which is what made the cross-op writer the
  only real obstacle.

- **Cross-op writer caution (`metal2_port_patterns.md` → Modifying a shared dataflow kernel) named the exact
  kernel.** The catalog lists `writer_unary_interleaved_start_id.cpp` as the canonical cross-op shared kernel;
  the `grep -rl` confirmed 31 co-borrowers, validating the "do not edit / do not fork" decision.

## Friction

- **Gap — guidance for "clean path + blocked path inside ONE runtime-source-selecting factory."** The recipe
  covers (a) multi-*factory* ops (port factories one at a time) and (b) single factories that select sources at
  runtime (convert all sources together). This op is the awkward intersection: a *single* factory whose
  runtime-selected paths split into one cleanly-portable (op-owned kernels) and one blocked (cross-op writer).
  The correct call — leave the *entire* factory on legacy rather than half-convert — follows from combining the
  atomic-unit note with the cross-op caution, but it would help to spell out this specific shape ("if any
  selectable source of a single factory is blocked, the whole factory stays legacy") as its own catalog/recipe
  sub-note. The orchestrator's brief stated this explicitly; the docs imply it.

- **Confusion (near-miss, resolved) — does `dfb::name → uint32_t` rescue the cross-op writer?** The
  orchestrator flagged the precise test: the implicit conversion lets `dfb::name` flow into a `uint32_t` CB-id
  CTA slot. For a moment that suggested the writer might work unmodified. It does not: the writer's `uint32_t`
  CB id arrives via `get_compile_time_arg_val(0)` (a *positional* CTA the kernel reads itself), not as a
  host-supplied DFB-id argument — and the kernel *additionally* depends on `TensorAccessorArgs<1>()` in CTAs +
  an address RTA, which the implicit conversion cannot touch. The implicit-conversion bridge applies at *call
  sites where the porter passes `dfb::name`*, not to a kernel that reads its own positional CTAs. Worth a
  one-line clarification in the patterns "Pass DFB handles directly" entry.

## Open items for downstream

- **Cross-op kernel touches:** none made. The blocker is `writer_unary_interleaved_start_id.cpp` (path taken:
  **neither** — not modified, not forked; left for upstream owner per the orchestrator's cross-op rule). When
  it gains a Metal-2.0 form, this factory and `nlp_concat_heads_boltz` (and the other 29 borrowers) can migrate.

- **Fake-CB self-loop bindings (interim hack — for the eventual port, flag prominently).** When this factory is
  eventually ported, the two borrowed-memory CBs become fake-CB self-loop DFBs (validator-satisfying devices,
  not real FIFOs): `src0` (c_0, `.buffer = in0_buffer` → `borrowed_from = INPUT`) and `out`
  (c_16, `.buffer = out_buffer` → `borrowed_from = OUTPUT`, sharded only). Both are **tensor-local-view**
  (base-pointer read/write, no FIFO) and should be replaced by the forthcoming local-`TensorAccessor`
  migration. Recorded now so the eventual port carries the flag forward; **no such binding exists in the tree
  today** since no code changed.

- **Test coverage:** `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads.py` exercises
  the op (sharded and interleaved layouts). Not run — there is no code change to validate. When the port lands,
  this suite (both `in_sharded` and interleaved paths) is the validation gate; note the interleaved path cannot
  be validated until the cross-op writer is ported, so even a future sharded-only port leaves part of the suite
  on the legacy path.

- **Sibling op:** `experimental/transformer/nlp_concat_heads_boltz` shares the same cross-op writer and the same
  runtime-source-selection shape — port it in the same wave once the writer is ready.
