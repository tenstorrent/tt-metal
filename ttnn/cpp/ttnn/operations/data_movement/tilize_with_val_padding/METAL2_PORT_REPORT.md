# Port Report — `tilize_with_val_padding` · `TilizeWithValPaddingMultiCoreBlockInterleavedFactory`

## Outcome

**PORTED** — the `TilizeWithValPaddingMultiCoreBlockInterleavedFactory` converted to
`ProgramSpecFactoryConcept` (`create_program_artifacts`). This was the last of the op's four
factories still on the `descriptor` concept (the other three were already ported); the op is now
fully on Metal 2.0. Verification: see the Verification section below.

## Provenance

- **Recipe docs (this port):** `d51708326b5 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `d51708326b5 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept` (base), exactly as the audit chose. The factory declares no
`override_runtime_arguments`; the framework refreshes the two io tensor bindings on a cache hit. The
port was a method swap `create_descriptor` → `create_program_artifacts` inside the existing
`TilizeWithValPaddingMultiCoreBlockInterleavedFactory` struct.

### Device-op-class edits
- Pybind entry points removed: **none** (the block factory had no pybound `create_descriptor`).
- Custom `compute_program_hash`: **none** (nothing to leave intact).
- `program_factory_t` variant: **untouched** — the block factory struct's method changed; the variant
  already declared it. With the block factory now ported, all four factories are on
  `ProgramSpecFactoryConcept` (the variant was already a working mixed set of 3 ported + 1 descriptor
  before this change, which confirms per-factory dispatch on this very op).

### Open items
- **Relaxation candidates:** none applied; strict `TensorSpec` match kept for both io tensors, as the
  audit declared (`relaxation = none`).
- No concept-fit friction.

## Handoff points

- **Shared-kernel forks created (3).** Each source was co-bound by the still-legacy **tilize** block
  factory (`data_movement/tilize/device/tilize_multi_core_block_program_factory.cpp`), so each was
  forked beside its original (rung 2) rather than converted in place, and a pointer comment added to
  each original. No usable `_metal2` fork pre-existed (the quasar `_metal2` copies are out of bounds;
  the eltwise `writer_unary_interleaved_start_id_metal2.cpp` is the non-`_wh` variant). This is a
  coordination/sunset signal, not a request to port tilize's block factory.

  | forked kernel | new fork path | remaining legacy consumer |
  |---|---|---|
  | `reader_unary_pad_multicore_both_dims.cpp` (this op's dir, *lent*) | `…/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims_metal2.cpp` | tilize block factory |
  | `writer_unary_interleaved_start_id_wh.cpp` (eltwise/unary, *borrowed*) | `…/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh_metal2.cpp` | tilize block factory |
  | `tilize_wh.cpp` (data_movement/tilize, *borrowed*) | `…/data_movement/tilize/device/kernels/compute/tilize_wh_metal2.cpp` | tilize block factory |

  Sunset: when tilize's block factory ports (and reuses these forks at rung 1), each legacy original
  can be deleted and the fork can take its name. The `writer_..._wh_metal2.cpp` fork preserves the
  `BACKWARDS` `#ifdef` for any consumer that uses the untilize direction.
- **Boundary-rule assumption violations:** none — no out-of-op call site required a `sem::`/`tensor::`
  handle; all kernel-lib helpers (`tt_memmove`, `fill_l1_range`, `compute_kernel_lib::tilize`,
  `is_fp32_input_format`, `compute_kernel_hw_startup`) accept `uint32_t`/NTTP and take `dfb::` via the
  implicit conversion.
- **Framework gaps:** none.

## Successes

- **[`cb_dfb_api_whitelist.md` §A — `constexpr` vs member getter]** steered the writer's
  `get_tile_size` correctly: the legacy `const uint32_t tile_bytes = get_tile_size(cb_id_out)` is not
  `constexpr`, so the member getter form `dfb.get_tile_size()` was used (after constructing the DFB),
  and it compiles on the DM path.
- **[Hardware configuration — `unpack_modes`]** the recipe's warning that `UnpackToDest` is the
  dangerous field paid off. The legacy `unpack_to_dest_mode[input_index]=UnpackToDestFp32` (set only
  when `fp32_dest_acc`) ports to `unpack_modes.insert({<bound input DFB>, UnpackToDest})` gated on
  `fp32_llk_acc`, keyed to *only* the DFB each compute `KernelSpec` binds (the legacy shared vector's
  entry for the other set's input CB is dropped — the validator rejects an entry for an unbound DFB).
  Confirmed validator-safe by reading `program_spec.cpp:1515-1516`: `UnpackToDest` with
  `enable_32_bit_dest=true` is always permitted, and `enable_32_bit_dest` is true in exactly the cases
  that set `UnpackToDest` here — so the reachable ≤16-bit-input configs (bfloat16 in / bfloat8_b out,
  fp8 in) never hit the `enable_32_bit_dest=false` ≤16-bit rejection.
- **[Compiler options]** each compute `KernelSpec` explicitly sets `opt_level = O3` (legacy compute
  default; Metal 2.0 defaults to O2). The reader/writer DM specs leave it default (O2, matching legacy).
- **[Preserved multiplicity]** the 4 compute `KernelSpec`s of one source over disjoint core ranges,
  and the per-set reader/writer listed in each region's `WorkUnitSpec` with disjoint `target_nodes`,
  reproduce the legacy work-split 1:1. No CTA→RTA demotion, no multi-binding flag.

## Friction

- **Gaps:** none blocking.
- **Confusion:** none material. The untilize op's already-ported
  `untilize_multi_core_block_program_factory.cpp` (same `make_block_plan` helper, same #51305 fix, same
  4-region preserved-multiplicity shape, opposite direction) was an unusually close structural
  reference for the host-side assembly; the only structural additions for the tilize direction were the
  per-set self-loop **staging** DFB and the reader's `pad_value` RTA.

## Open items for downstream

- **Shared kernel touches** (coordination for the next port of a sharing op):
  - `reader_unary_pad_multicore_both_dims.cpp` — **created fork** `..._metal2.cpp` beside it, pointer
    comment landed. Remaining unmigrated consumer: **tilize** block factory
    (`data_movement/tilize/device/tilize_multi_core_block_program_factory.cpp`).
  - `writer_unary_interleaved_start_id_wh.cpp` — **created fork** `..._wh_metal2.cpp`, pointer comment
    landed. Remaining consumer: tilize block factory. (Distinct from the non-`_wh`
    `writer_unary_interleaved_start_id_metal2.cpp` fork that already existed.)
  - `tilize_wh.cpp` — **created fork** `tilize_wh_metal2.cpp`, pointer comment landed. Remaining
    consumer: tilize block factory.
  - Fork binding vocabulary (the interface the next consumer inherits): reader → `dfb::in`,
    `dfb::stage`, `tensor::src` + named args {total_num_rows, third_dim, tile_height, element_size,
    unpadded_X_size, dram_alignment (CTA); pad_value, width_size, start_row_id, start_column_id,
    single_block_size_row_arg, single_block_size_col_arg, sub_block_width_size,
    single_sub_block_size_row_arg (RTA)}; writer → `dfb::out`, `tensor::dst` + {num_tiles_per_2d,
    third_dim, total_tiles_per_row (CTA); start_id, single_block_size_row_arg,
    single_block_size_col_arg (RTA)}; compute → `dfb::in`, `dfb::out` + {block_size_col, block_size_row,
    third_dim (CTA)}.
- **Misc anomalies observed but not changed** (per scope discipline — reported, not fixed):
  - Stale comment `// Assuming bfloat16 dataformat` on `unpadded_row_size_bytes` /
    `padded_row_size_bytes` in the factory: the values use `a.element_size()` and are correct for any
    dtype. Preserved verbatim.
  - The reader re-reads its per-`third_dim`-iteration RTAs (start_row_id … single_sub_block_size_row_arg)
    each loop iteration at constant indices although the values are constant across iterations —
    harmless, left as-is (named args, read in the loop as the legacy did).
- **Sibling op carry-over:** the tilize (plain) block factory
  (`tilize_multi_core_block_program_factory.cpp`) is the direct next candidate — it binds all three of
  the forks created here, so its port is rung-1 reuse of these forks (adopt the vocabulary above).
- **Concurrent-port collision (coordination):** a parallel effort ported the tilize (plain) block
  factory on a separate branch and created its own `_metal2` forks of these same three kernels. As of
  this port those forks are **not on `origin/main`** and not in this checkout, so the rung-1 locational
  check correctly found none and this port created them (rung 2) — the sanctioned "concurrent ports
  collide by design" case. When both land, expect a git add/add conflict on the three fork files;
  resolve by keeping one (the reviewed one) and repointing the other factory's `KernelSpec::source` +
  binding names, per the shared-kernel Caution. The fork vocabulary this port uses is listed above so a
  resolver can align the two.

## Update: rebased onto the tilize block port + kernel sunset

After both ports existed, this branch was rebased onto `gchoudhary/tilize-block-metal2-port` (which
ports the plain `tilize` block factory, `TilizeMultiCoreBlockProgramFactory`). The concurrent-fork
collision above was resolved and the shared kernels were **sunset** — the two block factories were the
only two consumers of all three kernels, so no legacy consumer remained once both were ported.

- **Collision resolution:** both branches created same-named `_metal2` forks. Kept the target branch's
  fork content for all three; the only real vocabulary difference was the reader's staging DFB
  (`dfb::staging` on the target vs `dfb::stage` here) — this factory's staging `DFBBinding.accessor_name`
  was changed to `staging` to match. Writer and compute fork vocabularies were already identical.
- **Sunset (repo-wide census cleared it):** the three `_metal2` forks were renamed onto their original
  canonical names (the legacy content deleted, the Metal 2.0 content taking the original name), and
  **both** block factories (`tilize_with_val_padding` and `tilize`) now bind the canonical names. No
  `_metal2` kernel files remain. Census confirmed no other consumer anywhere in the repo (the only
  non-fork references were the two block factories, the two explicit `CMakeLists.txt` entries — which
  reference the original names and stay valid — and a comment in `tilize_device_operation.cpp` naming the
  reader, whose name is unchanged). The forks' stale "this is a fork of X" headers were trimmed to
  canonical-kernel headers keeping the binding-interface note.
- **Final shared-kernel state:** `reader_unary_pad_multicore_both_dims.cpp` (this op's dir),
  `eltwise/unary/.../writer_unary_interleaved_start_id_wh.cpp`, and
  `data_movement/tilize/.../compute/tilize_wh.cpp` are now single canonical Metal 2.0 kernels bound by
  both block factories. Reader vocab is `dfb::in` / `dfb::staging` / `tensor::src`.
