# Port Report — `experimental/transformer/nlp_create_qkv_heads_decode`

## Outcome

**PORTED — all three factories** (Interleaved, Sharded, ShardedSubcoregrid) on `MetalV2FactoryConcept`.
Builds clean; the op's full pytest suite passes **205 / 39 skipped / 0 failed**.

A Metal 2.0 **framework bug** (borrowed-DFB device-side base corrupted in multi-work-unit programs) was
discovered during the port and would have blocked the Sharded + ShardedSubcoregrid factories (their
`!overlap_qk_coregrid` layout requires two work units). It is **worked around** — see below. The
Interleaved factory is single-work-unit and uses the recipe-prescribed borrowed DFBs unchanged.

> **Draft-PR / repo caveat:** partway through, this checkout was externally switched off the
> `…_decode` branch and my interleaved-port commit + the four `METAL2_*.md` artifacts were lost from the
> tree. This branch reconstructs them from the session. The interleaved port and all four artifacts here
> are reconstructions; the Sharded/Subcoregrid workaround survived on disk and is the tested artifact.
> The branch is currently stacked on the unmerged prefill port (`12b020127fe`) — base needs
> reconciliation before un-drafting.

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for all three factories (`create_program_artifacts`). The device-op
`program_factory_t` variant holds three Metal 2.0 factories; the framework dispatches per-factory.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op never had one).
- Pybind entry points removed: none (`nlp_create_qkv_heads_decode_nanobind.cpp` binds via
  `ttnn::bind_function`, not a `create_descriptor` binding).
- The device-operation class (`validate_*`, `compute_output_specs`, `create_output_tensors`,
  `select_program_factory`) was **not** modified.

## Handoff points

- **FRAMEWORK BUG — borrowed-DFB device-side base address corrupted in multi-work-unit programs.**
  A borrowed-memory DFB comes back with a **garbage device-side write pointer** when the program has ≥2
  `WorkUnitSpec`s and the borrowed DFB lives on a work unit's cores that another DFB does not.
  Manifestations isolated by DPRINT (garbage sentinel `4244701459` vs a valid `1472512`): (1) a per-node
  interior DFB-id hole (fixable by reordering `dataflow_buffers`); (2) a borrowed DFB alone on a
  second-work-unit multi-core grid (`k_out` across 8 k-cores), **not** fixable by ordering. The
  `!overlap_qk_coregrid` sharded/subcoregrid layout needs two work units (disjoint q/k grids), so it hit
  this.
  - **Owner action (framework/runtime team):** fix per-node borrowed-DFB base-address dispatch/CRTA
    generation so a borrowed DFB attaches correctly (a) with non-contiguous per-node DFB ids and (b) as
    the sole/second-work-unit DFB spanning multiple cores. Minimal repro: any `MetalV2FactoryConcept` op
    with ≥2 `WorkUnitSpec`s where a borrowed DFB lives only on the second work unit's cores.
  - **Porter workaround applied (Sharded + Subcoregrid):** bind the q/k/v **outputs as `TensorParameter`s**
    written via `TensorAccessor(tensor::q_out).get_bank_base_address()` + offset (the sanctioned Case-2
    raw-base bridge) instead of borrowed DFBs — the same mechanism the input already uses correctly in
    the two-WU spec. No output DFBs remain there; only the (non-borrowed, self-loop) batch-offset scratch.
    **Caveats:** off the recipe's prescribed borrowed-DFB disposition; Gen2/Quasar story for
    `get_bank_base_address`-on-outputs is unresolved; models the same outputs differently from the
    Interleaved factory. Full audit of this workaround under **Successes / Open items** below.

- **Sharded factory batch-offset writer-CB wiring (audit Question #1) — one-line fix applied.** The
  legacy sharded factory allocated a distinct writer batch-offset CB `c_14` but never switched the
  writer kernel's CTA to it (present in the subcoregrid factory at the shifted index; omitted here), so
  reader and writer both produced into `c_15` and `c_14` was dead. The port applies the missing switch
  (writer binds the writer batch-offset DFB), making each a clean single-toucher self-loop.
  Behavior-preserving (both read page 0 of the same `batch_offset` tensor). **Owner action:** land the
  same one-line fix on the legacy path independently.

## Successes

- **Case-2 output binding sidesteps the framework bug (Sharded/Subcoregrid).** The output access is
  genuinely a Case-2 shape (raw base + explicit offset, no FIFO ops), so `get_bank_base_address()` is a
  sanctioned fit; it is **not** a smuggled buffer-address RTA. `get_bank_base_address()` on a
  height-sharded L1 output returns the uniform per-bank shard base = byte-for-byte the legacy borrowed CB's
  `get_write_ptr()`. Verified across the full matrix incl. the worst case (`test_create_heads_with_slice`,
  8 passed: `!overlap` two-WU + self-loop scratch DFB + output TensorParameters + real batch_offset).
- **Two-toucher → 1P+1C (Interleaved).** Reader/writer dual-instance raw-writes each borrowed output DFB
  → reader PRODUCER, writer CONSUMER; no multi-binding flag.
- **Conditional-binding `#ifdef` promotion.** `USE_ALIGNED_PATH` (interleaved scratch), and
  `USE_BATCH_OFFSET` / `PROCESS_QV` / `PROCESS_K` (sharded/subcoregrid) — CTA gates promoted to defines,
  gating both the token alias and its uses.
- **Case-2 raw-pointer input via `get_bank_base_address`** (sharded/subcoregrid): base pulled off the
  TensorAccessor, hand-rolled `UnicastEndpoint` reads left unchanged.

## Friction

- **DFB declaration order and multi-WU borrowed DFBs are silently correctness-load-bearing** — nothing in
  the recipe/guide/headers warns of it; only DPRINT-diffing the write pointers surfaced the framework bug.
  This consumed the bulk of the port's debugging.
- **`-Werror,-Wunused-variable`** on host factory code (inlined a legacy intermediate → unused).
- **`TT_METAL_HOME` pointed at a different worktree** than the checkout; all build/test commands had to
  override it (and `PYTHONPATH`) or they'd target the wrong tree.

## Open items for downstream

- **Revert Sharded/Subcoregrid outputs to borrowed DFBs once the framework bug is fixed** — that's the
  recipe-faithful disposition and would make all three factories model outputs consistently.
- **Gen2/Quasar:** validate `get_bank_base_address`-on-outputs, or plan to move to the fixed borrowed-DFB
  path, before this op targets Quasar.
- **RTA→CRTA cleanup** for the noc-coord varargs (identical on every node).
- **Cross-op kernel touches:** none (`tt_memmove` in the interleaved kernel is unmodified).

## Verification

- **Build:** `./build_metal.sh --build-tests` clean (host `-Werror`).
- **Tests:** `pytest .../test_nlp_create_qkv_heads_decode.py` → **205 passed, 39 skipped, 0 failed**.
  No regression vs legacy (verified by git-stashing the port and confirming legacy passes the same
  `!overlap` cases).
- **Anti-pattern self-audit:** clean — no `buffer()->address()` / magic CB indices in CTAs /
  `TensorAccessorArgs`/`get_arg_val`/`get_compile_time_arg_val` in kernels / `.id` extraction /
  `allow_instance_multi_binding`; all CTAs named; conditional DFB/tensor bindings `#ifdef`-gated.
