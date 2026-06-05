# Revisit audit — moreh optimizer kernels (eltwise_chain missed migrations)

Scope: find every remaining raw LLK eltwise stage left unmigrated to
`compute_kernel_lib::eltwise_chain` (and convenience wrappers) in the four assigned
moreh optimizer kernels. Read-only.

Principle applied: "Every CB operation that can be part of the helper should be part of
the helper." `tile_regs_*`, subblock/DEST loops, bulk-vs-streaming lifecycle, broadcasts,
held-operand patterns, and multi-op fused chains are NOT blockers.

## Key finding on the `*_to_cb` macros (moreh_common.hpp)

All remaining raw stages in `moreh_adam` and `moreh_adamw` are calls to the moreh_common
helper macros `mul_tiles_to_cb` / `add_tiles_to_cb` / `sub_tiles_to_cb` / `copy_tile_to_cb`.
Each macro body is a **self-contained per-call DEST window** — `cb_reserve_back` +
`cb_wait_front` + `tile_regs_acquire` + `*_tiles_init_with_dt` + `*_tiles`/`copy_tile` +
`tile_regs_commit/wait` + `pack_tile_with_dt` + conditional `cb_pop_front` +
`cb_push_back` (moreh_common.hpp:141-173 mul, 502-534 add, 657-689 sub, 455-475 copy).

There is NO in-DEST cross-iteration accumulation here: every macro acquires, computes,
packs, and releases DEST within a single call. The task's "in-DEST accumulation" blocker
hypothesis does NOT apply. These are textbook `BinaryFpu(D0)->PackTile(D0)` and
`CopyTile(D0)->PackTile(D0)` chains.

=> Every one of these is a **MISSED migration (MIGRATABLE)**.

Lifecycle mapping rules used below (derived from the macro bodies):
- `wait_front(itile+1)` + `pop=1`, `itile==0`  -> InputLifecycle::Streaming
- `wait_front(itile+1)` + `pop=1`, `itile!=0`   -> InputLifecycle::BulkDrain + TileOffset::Set{itile}
- `wait_front(itile+1)` + `pop=0`, `itile==0`  -> InputLifecycle::HeldStream
- `wait_front(itile+1)` + `pop=0`, `itile!=0`   -> InputLifecycle::HeldBulk + TileOffset::Set{itile}
- All operands here are single-tile -> OperandKind::Scalar.
- `*_tiles_init_with_dt` reconfigs both srca+srcb -> BinaryDataFormatReconfig::Input;
  `copy_tile_init_with_dt` -> CopyTileReconfig::Input; `pack_tile_with_dt` ->
  PackTileReconfig::Output. (Same `_with_dt` calls the already-migrated chains in these
  files use.)
- Same-CB in/out (e.g. `cb_tmp1,...,cb_tmp1`) is handled by the chain exactly as the
  existing migrated recip/sqrt stages in these files already do.

Note: some held operands here are NOT externally pre-waited (the macro itself does the
`wait_front`), so the correct held cell is `HeldStream`/`HeldBulk` (chain owns the wait,
not the pop) — NOT `CallerManaged`. Per MEMORY (`feedback_callermanaged_needs_external_wait`),
`CallerManaged` requires an external wait; these macros wait internally, so the chain must
own the wait edge.

---

## moreh_adam.cpp

Already migrated (NOT counted): the 8 `eltwise_chain` / convenience stages at lines
146-156, 165-174, 184-190, 197-199, 211-220, 223-232, 242-258, 264-274, 281-290 (Power,
Recip, BinaryMax binary_sfpu, copy, Sqrt mul-chains, add+Recip).

Remaining raw `*_to_cb` stages — all MIGRATABLE:

1. **param*weight_decay** — moreh_adam.cpp:98
   `mul_tiles_to_cb(cb_param_in, cb_scalar_args, cb_tmp1, first_tile, weight_decay_tile, 0, 0)`
   Verdict: MIGRATABLE.
   Chain: `mul<cb_param_in, cb_scalar_args, cb_tmp1, BroadcastDim::None,
   InputLifecycle::HeldStream, InputLifecycle::HeldBulk(+TileOffset::Set{weight_decay_tile}),
   OutputLifecycle::Streaming>` — i.e. `BinaryFpu<cb_param_in, cb_scalar_args, Mul, None,
   HeldStream, HeldBulk, Input, D0, Scalar, Scalar, Unset, Set>{0u, weight_decay_tile}` ->
   `PackTile<cb_tmp1>`. (both operands pop0=pop1=0 -> held; param popped later at loop end,
   scalar_args popped at MAIN end.)

2. **grad + tmp1** — moreh_adam.cpp:101
   `add_tiles_to_cb(cb_grad_in, cb_tmp1, tmp_cb_grad, first_tile, first_tile, 0, 1)`
   Verdict: MIGRATABLE.
   Chain: `add<cb_grad_in, cb_tmp1, tmp_cb_grad>` with A=HeldStream (pop0=0, grad held to
   loop end), B=Streaming (pop1=1). `BinaryFpu<cb_grad_in, cb_tmp1, Add, None, HeldStream,
   Streaming>{}` -> `PackTile<tmp_cb_grad>`.

3. **(1 - beta1)** — moreh_adam.cpp:106
   `sub_tiles_to_cb(cb_one, cb_scalar_args, cb_tmp1, first_tile, beta1_tile, 0, 0)`
   Verdict: MIGRATABLE.
   Chain: `BinaryFpu<cb_one, cb_scalar_args, Sub, None, HeldStream, HeldBulk, Input, D0,
   Scalar, Scalar, Unset, Set>{0u, beta1_tile}` -> `PackTile<cb_tmp1>`.

4. **grad*(1-beta1)** — moreh_adam.cpp:107
   `mul_tiles_to_cb(tmp_cb_grad, cb_tmp1, cb_tmp1, first_tile, first_tile, 0, 1)`
   Verdict: MIGRATABLE. Same-CB in/out on cb_tmp1.
   Chain: `BinaryFpu<tmp_cb_grad, cb_tmp1, Mul, None, HeldStream, Streaming>{}` ->
   `PackTile<cb_tmp1>` (tmp_cb_grad held — popped after the exp_avg_sq section uses it).

5. **exp_avg*beta1** — moreh_adam.cpp:110
   `mul_tiles_to_cb(cb_exp_avg_in, cb_scalar_args, tmp_cb_exp_avg, first_tile, beta1_tile, 0, 0)`
   Verdict: MIGRATABLE.
   Chain: `BinaryFpu<cb_exp_avg_in, cb_scalar_args, Mul, None, HeldStream, HeldBulk, Input,
   D0, Scalar, Scalar, Unset, Set>{0u, beta1_tile}` -> `PackTile<tmp_cb_exp_avg>`.

6. **exp_avg += tmp1** — moreh_adam.cpp:113
   `add_tiles_to_cb(tmp_cb_exp_avg, cb_tmp1, tmp_cb_exp_avg, first_tile, first_tile, 1, 1)`
   Verdict: MIGRATABLE. Same-CB in/out on tmp_cb_exp_avg.
   Chain: `add<tmp_cb_exp_avg, cb_tmp1, tmp_cb_exp_avg>` (both Streaming).

7. **exp_avg_out copy** — moreh_adam.cpp:116
   `copy_tile_to_cb(tmp_cb_exp_avg, cb_exp_avg_out, first_tile, 0)`
   Verdict: MIGRATABLE.
   Chain: `copy<tmp_cb_exp_avg, cb_exp_avg_out, InputLifecycle::HeldStream,
   OutputLifecycle::Streaming, CopyTileReconfig::Input, PackTileReconfig::Output>` (pop=0 ->
   held; tmp_cb_exp_avg reused later at line 296).

8. **(1 - beta2)** — moreh_adam.cpp:121
   `sub_tiles_to_cb(cb_one, cb_scalar_args, cb_tmp1, first_tile, beta2_tile, 0, 0)`
   Verdict: MIGRATABLE.
   Chain: `BinaryFpu<cb_one, cb_scalar_args, Sub, None, HeldStream, HeldBulk, Input, D0,
   Scalar, Scalar, Unset, Set>{0u, beta2_tile}` -> `PackTile<cb_tmp1>`.

9. **grad*grad** — moreh_adam.cpp:124
   `mul_tiles_to_cb(tmp_cb_grad, tmp_cb_grad, cb_tmp2, first_tile, first_tile, 1, 0)`
   Verdict: MIGRATABLE. Same-buffer-both-operands -> use `square`.
   Chain: `square<tmp_cb_grad, cb_tmp2, InputLifecycle::Streaming>` (the macro's pop1=0 is
   the same-buffer single-pop dedup; chain's same-buffer path waits/pops once -> Streaming).

10. **tmp1 * tmp2** — moreh_adam.cpp:127
    `mul_tiles_to_cb(cb_tmp1, cb_tmp2, cb_tmp1, first_tile, first_tile, 1, 1)`
    Verdict: MIGRATABLE. Same-CB in/out on cb_tmp1.
    Chain: `mul<cb_tmp1, cb_tmp2, cb_tmp1>` (both Streaming).

11. **exp_avg_sq*beta2** — moreh_adam.cpp:130
    `mul_tiles_to_cb(cb_exp_avg_sq_in, cb_scalar_args, tmp_cb_exp_avg_sq, first_tile, beta2_tile, 0, 0)`
    Verdict: MIGRATABLE.
    Chain: `BinaryFpu<cb_exp_avg_sq_in, cb_scalar_args, Mul, None, HeldStream, HeldBulk,
    Input, D0, Scalar, Scalar, Unset, Set>{0u, beta2_tile}` -> `PackTile<tmp_cb_exp_avg_sq>`.

12. **exp_avg_sq += tmp1** — moreh_adam.cpp:133
    `add_tiles_to_cb(tmp_cb_exp_avg_sq, cb_tmp1, tmp_cb_exp_avg_sq, first_tile, first_tile, 1, 1)`
    Verdict: MIGRATABLE. Same-CB in/out.
    Chain: `add<tmp_cb_exp_avg_sq, cb_tmp1, tmp_cb_exp_avg_sq>` (both Streaming).

13. **exp_avg_sq_out copy** — moreh_adam.cpp:136
    `copy_tile_to_cb(tmp_cb_exp_avg_sq, cb_exp_avg_sq_out, first_tile, 0)`
    Verdict: MIGRATABLE.
    Chain: `copy<tmp_cb_exp_avg_sq, cb_exp_avg_sq_out, InputLifecycle::HeldStream, ...,
    CopyTileReconfig::Input, PackTileReconfig::Output>` (pop=0; tmp_cb_exp_avg_sq reused at
    lines 184-234).

14. **lr * tmp2** — moreh_adam.cpp:293
    `mul_tiles_to_cb(cb_scalar_args, cb_tmp2, cb_tmp2, lr_tile, first_tile, 0, 1)`
    Verdict: MIGRATABLE. Same-CB in/out on cb_tmp2; lr at offset lr_tile (==0).
    Chain: `BinaryFpu<cb_scalar_args, cb_tmp2, Mul, None, HeldStream, Streaming>{}` ->
    `PackTile<cb_tmp2>` (lr_tile==0 so no TileOffset needed; scalar_args held, pop0=0).

15. **tmp2 * exp_avg** — moreh_adam.cpp:296
    `mul_tiles_to_cb(cb_tmp2, tmp_cb_exp_avg, cb_tmp2, first_tile, first_tile, 1, 1)`
    Verdict: MIGRATABLE. Same-CB in/out on cb_tmp2.
    Chain: `mul<cb_tmp2, tmp_cb_exp_avg, cb_tmp2>` (both Streaming).

16. **tmp1 * tmp2** — moreh_adam.cpp:299
    `mul_tiles_to_cb(cb_tmp1, cb_tmp2, cb_tmp1, first_tile, first_tile, 1, 1)`
    Verdict: MIGRATABLE. Same-CB in/out on cb_tmp1.
    Chain: `mul<cb_tmp1, cb_tmp2, cb_tmp1>` (both Streaming).

17. **param - tmp1** — moreh_adam.cpp:302
    `sub_tiles_to_cb(cb_param_in, cb_tmp1, cb_param_out, first_tile, first_tile, 0, 1)`
    Verdict: MIGRATABLE.
    Chain: `sub<cb_param_in, cb_tmp1, cb_param_out, BroadcastDim::None,
    InputLifecycle::HeldStream, InputLifecycle::Streaming>` (param pop0=0 -> popped at loop
    end via the explicit `cb_param_in_obj.pop_front` at line 304; B Streaming).

moreh_adam.cpp count: **17 MIGRATABLE**, 0 BLOCKED, 0 OUT-OF-SCOPE.

---

## moreh_adamw.cpp

Already migrated (NOT counted): eltwise_chain / convenience stages at 125-140, 169-179,
186-192, 201-222, 231-247, 256-266.

Remaining raw `*_to_cb` stages — all MIGRATABLE:

1. **weight_decay * param** — moreh_adamw.cpp:93
   `mul_tiles_to_cb(cb_scalar_args, cb_param_in, cb_tmp1, weight_decay_tile, first_tile, 0, 0)`
   Verdict: MIGRATABLE.
   Chain: `BinaryFpu<cb_scalar_args, cb_param_in, Mul, None, HeldBulk, HeldStream, Input,
   D0, Scalar, Scalar, Set, Unset>{weight_decay_tile, 0u}` -> `PackTile<cb_tmp1>`.

2. **lr * tmp1** — moreh_adamw.cpp:96
   `mul_tiles_to_cb(cb_scalar_args, cb_tmp1, cb_tmp1, lr_tile, first_tile, 0, 1)`
   Verdict: MIGRATABLE. Same-CB in/out on cb_tmp1; lr_tile==0.
   Chain: `BinaryFpu<cb_scalar_args, cb_tmp1, Mul, None, HeldStream, Streaming>{}` ->
   `PackTile<cb_tmp1>`.

3. **param - tmp1** — moreh_adamw.cpp:99
   `sub_tiles_to_cb(cb_param_in, cb_tmp1, tmp_cb_param, first_tile, first_tile, 0, 1)`
   Verdict: MIGRATABLE.
   Chain: `sub<cb_param_in, cb_tmp1, tmp_cb_param, None, HeldStream, Streaming>` (param
   held, popped at loop end line 280).

4. **(1 - beta1)** — moreh_adamw.cpp:104
   `sub_tiles_to_cb(cb_one, cb_scalar_args, cb_tmp1, first_tile, beta1_tile, 0, 0)`
   Verdict: MIGRATABLE.
   Chain: `BinaryFpu<cb_one, cb_scalar_args, Sub, None, HeldStream, HeldBulk, Input, D0,
   Scalar, Scalar, Unset, Set>{0u, beta1_tile}` -> `PackTile<cb_tmp1>`.

5. **grad*(1-beta1)** — moreh_adamw.cpp:107
   `mul_tiles_to_cb(cb_grad_in, cb_tmp1, cb_tmp1, first_tile, first_tile, 0, 1)`
   Verdict: MIGRATABLE. Same-CB in/out on cb_tmp1; grad held (pop0=0, reused at 143).
   Chain: `BinaryFpu<cb_grad_in, cb_tmp1, Mul, None, HeldStream, Streaming>{}` ->
   `PackTile<cb_tmp1>`.

6. **exp_avg*beta1** — moreh_adamw.cpp:110
   `mul_tiles_to_cb(cb_exp_avg_in, cb_scalar_args, tmp_cb_exp_avg, first_tile, beta1_tile, 0, 0)`
   Verdict: MIGRATABLE.
   Chain: `BinaryFpu<cb_exp_avg_in, cb_scalar_args, Mul, None, HeldStream, HeldBulk, Input,
   D0, Scalar, Scalar, Unset, Set>{0u, beta1_tile}` -> `PackTile<tmp_cb_exp_avg>`.

7. **exp_avg += tmp1** — moreh_adamw.cpp:113
   `add_tiles_to_cb(tmp_cb_exp_avg, cb_tmp1, tmp_cb_exp_avg, first_tile, first_tile)` (pop defaults =1,1)
   Verdict: MIGRATABLE. Same-CB in/out.
   Chain: `add<tmp_cb_exp_avg, cb_tmp1, tmp_cb_exp_avg>` (both Streaming).

8. **exp_avg_out copy** — moreh_adamw.cpp:116
   `copy_tile_to_cb(tmp_cb_exp_avg, cb_exp_avg_out, first_tile, 0)`
   Verdict: MIGRATABLE.
   Chain: `copy<tmp_cb_exp_avg, cb_exp_avg_out, InputLifecycle::HeldStream, ...,
   CopyTileReconfig::Input, PackTileReconfig::Output>` (pop=0; reused at 272).

9. **grad*grad** — moreh_adamw.cpp:143
   `mul_tiles_to_cb(cb_grad_in, cb_grad_in, cb_tmp2, first_tile, first_tile, 0, 0)`
   Verdict: MIGRATABLE. Same-buffer both operands -> `square`.
   Chain: `square<cb_grad_in, cb_tmp2, InputLifecycle::HeldStream>` (pop0=pop1=0: grad held
   to loop end; same-buffer path waits once, no pop -> HeldStream).

10. **tmp1 * tmp2** — moreh_adamw.cpp:146
    `mul_tiles_to_cb(cb_tmp1, cb_tmp2, cb_tmp1, first_tile, first_tile)` (pop=1,1)
    Verdict: MIGRATABLE. Same-CB in/out on cb_tmp1.
    Chain: `mul<cb_tmp1, cb_tmp2, cb_tmp1>` (both Streaming).

11. **exp_avg_sq*beta2** — moreh_adamw.cpp:149-150
    `mul_tiles_to_cb(cb_exp_avg_sq_in, cb_scalar_args, tmp_cb_exp_avg_sq, first_tile, beta2_tile, 0, 0)`
    Verdict: MIGRATABLE.
    Chain: `BinaryFpu<cb_exp_avg_sq_in, cb_scalar_args, Mul, None, HeldStream, HeldBulk,
    Input, D0, Scalar, Scalar, Unset, Set>{0u, beta2_tile}` -> `PackTile<tmp_cb_exp_avg_sq>`.

12. **exp_avg_sq += tmp1** — moreh_adamw.cpp:153
    `add_tiles_to_cb(tmp_cb_exp_avg_sq, cb_tmp1, tmp_cb_exp_avg_sq, first_tile, first_tile)` (pop=1,1)
    Verdict: MIGRATABLE. Same-CB in/out.
    Chain: `add<tmp_cb_exp_avg_sq, cb_tmp1, tmp_cb_exp_avg_sq>` (both Streaming).

13. **exp_avg_sq_out copy** — moreh_adamw.cpp:156
    `copy_tile_to_cb(tmp_cb_exp_avg_sq, cb_exp_avg_sq_out, first_tile, 0)`
    Verdict: MIGRATABLE.
    Chain: `copy<tmp_cb_exp_avg_sq, cb_exp_avg_sq_out, InputLifecycle::HeldStream, ...,
    CopyTileReconfig::Input, PackTileReconfig::Output>` (pop=0; reused at 186-224).

14. **max_exp_avg_sq_out copy (AMSGRAD)** — moreh_adamw.cpp:195
    `copy_tile_to_cb(tmp_cb_max_exp_avg_sq, cb_max_exp_avg_sq_out, first_tile, 0)`
    Verdict: MIGRATABLE.
    Chain: `copy<tmp_cb_max_exp_avg_sq, cb_max_exp_avg_sq_out, InputLifecycle::HeldStream,
    ..., CopyTileReconfig::Input, PackTileReconfig::Output>` (pop=0; reused at 201-211).
    NOTE: moreh_adam.cpp:197 already migrated this exact stage with `copy<...
    CallerManaged>` after an explicit `wait_front` — moreh_adamw's version has no explicit
    external wait, so HeldStream (chain owns the wait) is the correct cell here.

15. **lr * tmp2** — moreh_adamw.cpp:269
    `mul_tiles_to_cb(cb_scalar_args, cb_tmp2, cb_tmp2, lr_tile, first_tile, 0, 1)`
    Verdict: MIGRATABLE. Same-CB in/out on cb_tmp2; lr_tile==0.
    Chain: `BinaryFpu<cb_scalar_args, cb_tmp2, Mul, None, HeldStream, Streaming>{}` ->
    `PackTile<cb_tmp2>`.

16. **tmp2 * exp_avg** — moreh_adamw.cpp:272
    `mul_tiles_to_cb(cb_tmp2, tmp_cb_exp_avg, cb_tmp2, first_tile, first_tile)` (pop=1,1)
    Verdict: MIGRATABLE. Same-CB in/out on cb_tmp2; tmp_cb_exp_avg now popped (last use).
    Chain: `mul<cb_tmp2, tmp_cb_exp_avg, cb_tmp2>` (both Streaming).

17. **tmp1 * tmp2** — moreh_adamw.cpp:275
    `mul_tiles_to_cb(cb_tmp1, cb_tmp2, cb_tmp1, first_tile, first_tile)` (pop=1,1)
    Verdict: MIGRATABLE. Same-CB in/out on cb_tmp1.
    Chain: `mul<cb_tmp1, cb_tmp2, cb_tmp1>` (both Streaming).

18. **param = tmp_param - tmp1** — moreh_adamw.cpp:278
    `sub_tiles_to_cb(tmp_cb_param, cb_tmp1, cb_param_out, first_tile, first_tile)` (pop=1,1)
    Verdict: MIGRATABLE.
    Chain: `sub<tmp_cb_param, cb_tmp1, cb_param_out>` (both Streaming).

moreh_adamw.cpp count: **18 MIGRATABLE**, 0 BLOCKED, 0 OUT-OF-SCOPE.

---

## moreh_clip_grad_norm_step1_kernel.cpp

Fully migrated. Every eltwise stage (abs+mask 4-branch dispatch, power/log/exp chains,
correct_xpow mul, xpowadd seed copy + accumulate add) uses eltwise_chain / convenience
wrappers. The trailing `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(...)`
(step1:211) is a reduction -> OUT-OF-SCOPE:reduce, and already uses the reduce helper.

No raw eltwise LLK calls remain. **0 MIGRATABLE, 0 BLOCKED.**

(The 4-branch `if (mh && mw) / mh / mw / else` is a runtime data-conditioned masking
dispatch, but each branch is already a fully-formed eltwise_chain — not a missed migration.)

## moreh_clip_grad_norm_step2_kernel.cpp

Fully migrated. Seed copy + accumulate add (step2:44-64), power/log/exp chains
(step2:78-115), final `mul<cb_xpow, cb_exp_lxmd, cb_y>` (step2:118). All eltwise_chain /
convenience. No reduce here.

No raw eltwise LLK calls remain. **0 MIGRATABLE, 0 BLOCKED.**

---

## Totals

| Kernel | MIGRATABLE | BLOCKED | OUT-OF-SCOPE |
|---|---|---|---|
| moreh_adam.cpp | 17 | 0 | 0 |
| moreh_adamw.cpp | 18 | 0 | 0 |
| moreh_clip_grad_norm_step1 | 0 | 0 | 1 (reduce, already helper) |
| moreh_clip_grad_norm_step2 | 0 | 0 | 0 |
| **Total** | **35** | **0** | **1** |

## Verdict on the task's "full but ~17-18 raw calls" flag

CONFIRMED MISSED MIGRATIONS, not genuine blockers. moreh_adam (17) and moreh_adamw (18)
were marked "full" but still route every plain CB-to-CB eltwise op through the legacy
`mul_tiles_to_cb` / `add_tiles_to_cb` / `sub_tiles_to_cb` / `copy_tile_to_cb` moreh_common
macros. The "in-DEST accumulation" exemption does NOT apply: each macro is a self-contained
per-call DEST window (acquire->compute->pack->release) with full CB lifecycle baked in —
exactly the `BinaryFpu(D0)->PackTile(D0)` / `CopyTile(D0)->PackTile(D0)` chain shapes the
helper expresses. The only non-trivial wrinkles are (a) held operands via pop0/pop1==0 ->
Held{Stream,Bulk}, and (b) scalar-args tile offsets via TileOffset::Set — both fully
supported. Every one is migratable.
