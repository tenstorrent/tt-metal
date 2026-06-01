# Eltwise helper API simplification — progress / continuation note

Branch: `astancov/eltwise_vx_migration`
Proposal (full design + status): `ttnn/cpp/ttnn/kernel_lib/docs/eltwise_api_simplification_proposal.html`
Core files: `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.{hpp,inl}`, `eltwise_convenience.hpp`, `eltwise_optional.hpp`, `eltwise_rand.hpp`

## DONE + committed + device-tested (~21 commits)

- **Change 1** — deleted phantom `eltwise_chain_with_init` docs (→ rejected-pattern note) + the unused `PackTileBlock` struct/decl.
- **Change 2** — documented why `CopyTileReconfig::None` is load-bearing (first CB-reader emits an unconditional single-arg reconfig with `Input`; `None` skips it when boot-init already programmed the format).
- **Change 5** — dispatch clamps runtime `block_size` to `chain_max_block_v<Chain>` (`DEST_AUTO_LIMIT / chain_lane_width`) so it can't overflow DEST.
- **Change 8** — removed the dead 1D dispatch path; renamed survivors `_2d`→unsuffixed (`window_2d→window`, `idx_2d→idx`, `exec_2d→exec`, `apply_*_phase_2d`, `wait/reserve/pop/push_*_2d`, `elem_*_2d`). Internal only.
- **Change 7** — `UnaryBcast`: dropped `CbOut`; `init()` now does only the bcast datacopy MOP (icb-only). Reconfig is **fold-driven**: `reconfig_srca_cb = reconfig_srcb_cb = (Reconfig==Input) ? Cb : NO_PREV_CB` — **both** sides because the COL/ROW/SCALAR bcast datacopy drives the FPU **SrcB** lane (`ELWADD + SRCB_BCAST_*`); a srca-only reconfig left `ALU_FORMAT_SPEC_REG1_SrcB` stale (root-caused via subagent for a layernorm-welford numeric regression). Removed the mid-kernel `unary_bcast_init` stopgap from `layernorm_large_tensor{,_welford}.cpp` (pack reconfig → downstream `PackTile` `Output`).
- **Change 4** — `TileBase` (3 types) → `enum class TileOffset : bool { Unset, Set }`. Elements take a `TileOffset Offset` param + plain `uint32_t tile_base` field (ctor when `Set`); read via `tile_base_value<Offset>(stored)` (0 folded when Unset). `BinaryFpu` → `OffsetA`/`OffsetB`. Migrated ~78 sites.
- **Change 6** — removed `PackTile`'s `OperandKind IndexMode`. Derived `static constexpr bool walk = (Policy == OutBulk)`; `out_idx = walk ? base+i_flat : base`; OutBulk reserves/pushes `Ht*Wt`, else pinned. Behavior-identical (all sites Scalar; OutBulk only at onetile). Dropped dead `index_mode` member + `elem_has_index_mode` trait. Migrated 326 sites.
- **Change 3a** — lifecycle constants → static members of `InputLifecycle`/`OutputLifecycle`, written type-qualified (`InputLifecycle::Bulk`, `OutputLifecycle::Streaming`, **`Out` prefix dropped**). Pattern: in-class `static const X a,b,...;` decl + out-of-line `inline constexpr X X::a{...};` def (needs C++20, which the project uses). Order unchanged. Migrated 82 call sites + 141 internal refs.
- **bcast_w.cpp** — added missing `#include "api/dataflow/circular_buffer.h"` (pre-existing breakage, not mine).

## REMAINING — Change 3b (reorder) + Change 9 (infer), merged

**3b reorder**: reorder each element's template params variance-first + axis-major pairing, defaults trailing (proposal §3). Hard part: this **moves positional template args** across 82 multi-line call sites — a regex can't reorder positional args reliably, so it needs careful per-element handling. PackTile is the tractable, highest-value one: move `DstSlot` (currently param 2, always `D0`) to trailing → call sites drop the `Dst::D0` token (4-arg sites delete it; the few `TileOffset::Set` sites move it to position 4).

**Change 9 (merged in)**: PackTile `DstSlot` trailing-default `Dst::Infer` (add `Infer = 0xFFFFFFFF` to the `Dst` enum). Resolve `Infer` from the nearest preceding DEST-writer's output slot (writers expose `dst_slot`; SFPU expose `dst_idx`/`out`) via a compile-time fold + **element-pack rebind** (transform `PackTile<…Infer…>` instances → `PackTile<…resolved…>` preserving `tile_base`), injected **before** `lane_width`/`chain_lane_width` are computed (or the `Infer` sentinel poisons them). Validation static_asserts: `Infer` PackTile-only (other elements `static_assert(Slot != Infer)`); single-pack-only (≥2 PackTiles ⇒ require explicit slots); slot range. block_size: resolve base slot only, `slot_offset` rides on top.
NOTE: inference's correctness benefit is currently **unexercised** (all 346 packs are D0); the verbosity win is the reorder. A safe fallback is trailing-default `Dst::D0` (no rebind metaprogramming).

**Open question (undecided)**: unify the 4 reconfig enums into one shared two-state type.

## Workflow / gotchas
- Build: kernels JIT at test time; no host rebuild for kernel changes.
- Test from repo root after `source python_env/bin/activate`: `scripts/run_safe_pytest.sh [--run-all] <test>`. Regression set used: `test_eltwise_typecast.py`, `test_unary.py` (-k mish/exp), `test_softmax.py` (OutBulk walk), `test_layer_norm.py` (incl welford w=4096), `test_group_norm.py`, `test_moreh_adam.py`, `test_bcast_to.py`, `test_binary_bcast.py`, `test_rotary_embedding.py`, `test_moreh_softmax.py`.
- **Pre-commit hook auto-fixes whitespace** → first `git commit` silently aborts (files become `MM`); re-`git add` the same files + re-`commit`.
- **Mass-rename gotcha**: lifecycle/word renames match unrelated files (sdpa, batch_norm, deepseek, "dram_streaming…"). Filter to files containing `compute_kernel_lib` or including `eltwise_chain.hpp`; revert false positives.
- Commit message footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Commit after each big change (user preference).
