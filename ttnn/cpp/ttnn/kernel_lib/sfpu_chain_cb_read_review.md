# SFPU chain / CB-reading ops review — handoff

## Scope reviewed

All operations in the kernel lib that pull data from a CB inside a chain/pipeline/postop context:

1. `Load<CB, Slot, Policy>` — `sfpu_chain.hpp:258`. User-facing CB→DEST copy.
2. `CompactLoad<CB, DoWait, DoPop, Slots...>` — `sfpu_chain.hpp:283`. Internal compaction product.
3. `DestReuseOp<CB, OpType, ReuseType, Slot, Policy, Reconfig>` — `binary_op_helpers.hpp:409`. Binary-op PostOp that reads a CB tile straight into an SRC register.

## What was changed this session

### `DestReuseOp` wait path cleaned up and constrained
- `binary_op_helpers.inl:215-227` — `operator()` now splits the wait into two explicit `if constexpr` branches instead of threading `do_wait` + a mixed `cb_wait_front(CB, cb_tile_idx + 1)` expression:
  ```
  if constexpr (Policy == LoadPolicy::WaitAndPop) {
      ASSERT(cb_tile_idx == 0);
      cb_wait_front(CB, 1);
  } else if constexpr (Policy == LoadPolicy::WaitNoPop) {
      cb_wait_front(CB, cb_tile_idx + 1);
  }
  ```
- `binary_op_helpers.hpp:364-371` — docstring rewritten; the "same semantics as Load's LoadPolicy" claim is gone. Per-policy behaviour with the `cb_tile_idx` constraint is now explicit.

Why: `cb_wait_front(CB, cb_tile_idx + 1) + cb_pop_front(CB, 1)` under `WaitAndPop` was incoherent. If the producer pushes tile-by-tile and the consumer only pops one, indexing into a batch doesn't match the streaming lifecycle — the math only works for a contrived "producer locksteps with consumer" pattern that no kernel in the tree uses. `cb_tile_idx != 0` only makes sense when the batch is already waited upfront (`WaitNoPop`) or the caller owns the CB (`NoWaitNoPop`). Asserting it in the `WaitAndPop` branch makes the invariant load-bearing.

No callers are affected: grep shows zero users of `cb_tile_idx != 0` in the tree.

Build verified (`./build_metal.sh`) after both edits.

## Identified inconsistencies — status

1. **`SfpuInputPolicy` is dead code.** Declared and threaded through `sfpu_pipeline`, `sfpu_op`, and every `sfpu_*.hpp` alias, but the pipeline body never references it. Wait/pop is owned entirely by per-`Load` `LoadPolicy`. **User confirmed removal**, **NOT YET EXECUTED** — scope (see "next steps").

2. **Three overlapping policy taxonomies.** `LoadPolicy` (3 values, used by Load and DestReuseOp), `SfpuInputPolicy` (3 values, dead), `BinaryInputPolicy` (6 values, binary_op). Partially resolved by #1 removing `SfpuInputPolicy`. Still open: `LoadPolicy` cannot express `WaitUpfrontPopAtEnd` / `WaitAndPopPerChunk` / `NoWaitPopAtEnd`, so a CB feeding a `DestReuseOp` PostOp inside a `binary_op` has strictly fewer lifecycle options than the parent's A/B operands. **Not yet decided.**

3. **`DestReuseOp` "same semantics as Load" doc lie + weird wait count.** Fixed this session (see above).

4. **`LoadTag` doc claim "pipeline handles these specially" is misleading.** The pipeline is Load-agnostic at runtime; special handling is in `sfpu_chain()` (compaction) and `FirstLoadCB` (one-shot format reconfig before the loop). **Not yet fixed — doc-only.**

5. **Mixed-CB chains silently miscompile when CBs have different data formats.** `sfpu_pipeline` calls `reconfig_data_format_srca(FirstLoadCB)` and `copy_tile_to_dst_init_short(FirstLoadCB)` exactly once; `CompactLoad::init` is a no-op. A chain like `Load<cb_fp32, D0>, SomeOp, Load<cb_bf16, D1>` runs the bf16 `copy_tile` under the fp32 unpacker config. Nothing in `sfpu_chain()` or the pipeline detects this. **Not yet fixed.** Options: (a) static_assert that all Load CBs in a chain resolve to the same format (requires format info at compile time — not available), (b) document the constraint loudly, (c) add per-CompactLoad re-init when the CB changes.

6. **`SfpuDataFormatReconfig` has no per-SRC variants — retracted.** My initial claim was wrong; `copy_tile` only uses SRCA in pure SFPU chains, so INPUT/OUTPUT/INPUT_AND_OUTPUT/NONE is sufficient. Per-SRC reconfig belongs only to binary_op / DestReuseOp, which already handle it.

7. **`DestReuseReconfig::CbSide` → `Input` already done** (commit `c358e0f068a`) — rename noted for context.

## Next steps (recommended order)

### A. Execute `SfpuInputPolicy` removal (#1)
Pure mechanical sweep — all existing uses pass the default (`WaitAndPopPerTile`), so no behaviour changes. Files touched:
- `sfpu_chain.hpp` — delete enum (line 92), drop template param from `sfpu_pipeline` (line 571) and `sfpu_op` (line 584)
- `sfpu_chain.inl` — drop param from both function signatures (lines 73, 159)
- Alias headers (drop `SfpuInputPolicy P` template param):
  - `sfpu_math.hpp` (8 sites)
  - `sfpu_trig.hpp` (12 sites)
  - `sfpu_activations.hpp` (5 sites)
  - `sfpu_rounding.hpp` (4 sites)
  - `sfpu_predicates.hpp` (10 sites)
  - `sfpu_special.hpp` (6 sites)
- Matching `.inl` files for the above
- `sfpu_helpers.inl` (many sites — convenience wrappers)
- External call sites to strip `SfpuInputPolicy::WaitAndPopPerTile` args:
  - `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w_neg.cpp:52`
  - `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_hw_neg.cpp:47`
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/hardswish_kernel_sfpu.cpp:28`
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/tanhshrink_sfpu_kernel.cpp:28`
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/mish_kernel.cpp:47,60`

Grep command to find everything: `grep -rn "SfpuInputPolicy" ttnn/`.

Approach: sed-style replace across headers, then manually remove the arg at the 5 kernel call sites, then `./build_metal.sh`.

### B. Decide `BinaryInputPolicy` vs `LoadPolicy` convergence (#2)
Open design question. Two paths:
- **Align down**: keep `LoadPolicy` as the CB-read vocabulary everywhere, trim `BinaryInputPolicy` to the same 3 values. Loses `WaitUpfrontPopAtEnd` / `WaitAndPopPerChunk` / `NoWaitPopAtEnd` — check whether any kernel actually uses those before dropping.
- **Align up**: expand `LoadPolicy` to match `BinaryInputPolicy`'s 6 values. Adds complexity to the Load compaction in `sfpu_chain()` because the OR-merge rules over 6 policies are messier than over `{wait, pop}` bits.

Recommend **align down** — the 6-value binary enum looks like speculative generality; gap analysis should confirm.

### C. Fix `LoadTag` doc and add mixed-format chain guard (#4, #5)
Low-risk:
- Rewrite the `LoadTag` comment to say "used by `sfpu_chain()` compaction and `FirstLoadCB` detection" instead of "pipeline handles these specially".
- Either add a prominent warning in the `sfpu_chain()` docstring ("all Load CBs must share a data format"), or — better — make `CompactLoad::init()` call `copy_tile_to_dst_init_short(CB)` for its own CB when there's more than one distinct CB in the chain. Requires a compile-time check `num_distinct_cbs > 1` in the chain, then conditionally emit the init. Adds one instruction per CompactLoad in mixed-CB chains; free in single-CB chains.

### D. Consider dropping `cb_tile_idx` entirely from `DestReuseOp`
Alternative to this session's fix: just remove the runtime field and require the caller to `cb_pop_front` the unwanted prefix before calling the op. Simpler surface, one fewer footgun. `cb_tile_idx` has zero real users right now, so the cost is nil. Worth considering before documenting the current shape as stable.

## Files to read for context
- `ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp` — chain + Load + policy enums
- `ttnn/cpp/ttnn/kernel_lib/sfpu_chain.inl` — pipeline implementation (note: `input_policy` threaded through but unused)
- `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp` — binary_op + DestReuseOp surface
- `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.inl:205-235` — DestReuseOp implementation (freshly edited)
- `ttnn/cpp/ttnn/kernel_lib/binary_sfpu_gap_analysis.md` — existing gap analysis doc (may need updating after #1)
- `ttnn/cpp/ttnn/kernel_lib/binary_migration_plan.md` — migration plan doc
