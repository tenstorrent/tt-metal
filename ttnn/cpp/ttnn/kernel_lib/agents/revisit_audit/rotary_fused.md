# Rotary (fused) compute-kernel migration audit — eltwise_chain

Scope: find every remaining raw-LLK eltwise stage, classify it, and **verify the
"runtime-selected CB" PARTIAL claim** documented on the fused_qk kernels.

Verdict legend: `MIGRATABLE` | `BLOCKED:<reason>` | `OUT-OF-SCOPE:<kind>`.
Read-only audit — no files edited.

---

## Verification of the "runtime CB" claim (central question)

The fused_qk kernels resolve their input/output CB ids at **runtime**:

```cpp
const bool is_q = get_arg_val<uint32_t>(0);   // RUNTIME arg
constexpr uint32_t q_in_cb  = get_compile_time_arg_val(0);
constexpr uint32_t q_out_cb = get_compile_time_arg_val(1);
constexpr uint32_t k_in_cb  = get_compile_time_arg_val(3);
constexpr uint32_t k_out_cb = get_compile_time_arg_val(4);
uint32_t in_cb  = q_in_cb;     // <-- plain uint32_t, NOT constexpr
uint32_t out_cb = q_out_cb;    // <-- plain uint32_t, NOT constexpr
uint32_t Ht     = q_Ht;
if (!is_q) { in_cb = k_in_cb; out_cb = k_out_cb; Ht = k_Ht; }
```

`in_cb` and `out_cb` are plain (non-`constexpr`) `uint32_t` locals whose value is
chosen by the runtime `is_q` flag. The chain element templates
(`CopyTile`, `BinaryFpu`, `PackTile`, …) take the CB id as a **`uint32_t`
non-type template parameter** — a constant expression is mandatory. A runtime
`uint32_t` cannot be a template argument.

**CONCLUSION: the "runtime-selected CB" claim HOLDS** for both fused_qk kernels
on every stage that touches `in_cb` or `out_cb`.

- `rotary_embedding_llama_sharded.cpp` (fused_qk): cos stage reads `in_cb`
  (runtime) → BLOCKED. add stage writes `out_cb` (runtime) → BLOCKED.
- `rotary_embedding_llama_sharded_row_major.cpp` (fused_qk): same — cos reads
  `in_cb`, add writes `out_cb` → both BLOCKED.

The accompanying "TRISC2 code-size budget" justification is **secondary and is
NOT itself a blocker** per audit policy. The genuine blocker is the non-constexpr
CB template-argument constraint. (Note: the documented workaround — duplicate the
chain in both q/k constexpr branches — IS structurally migratable but the kernel
author declined it for code size. Recorded below as a MIGRATABLE-via-duplication
note, not a clean MIGRATABLE.)

The non-fused kernels (`rotary_embedding/.../rotary_embedding.cpp` and
`rotary_embedding_llama/.../rotary_embedding_llama_sharded.cpp`) have **no runtime
CB** — every CB is `constexpr` (compile-time args). Their eltwise stages are
already fully migrated.

---

## 1. rotary_embedding/device/kernels/compute/rotary_embedding.cpp

All CBs are `constexpr` compile-time args (lines 96-117). Already FULLY MIGRATED;
audited only to confirm no raw eltwise stage remains.

### Stage R1 — UNTILIZE (DECODE_MODE only) — file:70-78,119-120
- LLK: `untilize_*` (via `compute_kernel_lib::untilize` wrapper).
- Verdict: **OUT-OF-SCOPE:untilize** (already on untilize_helpers, not eltwise).

### Stage R2 — TILIZE rows (DECODE_MODE only) — file:80-91,123-124
- LLK: `tilize_*` (via `compute_kernel_lib::tilize` wrapper).
- Verdict: **OUT-OF-SCOPE:tilize**.

### Stage R3 — rotated_in * scalar(-1), bcast scalar — file:147-153
- Already migrated: `compute_kernel_lib::mul<…, BroadcastDim::Scalar, Streaming, CallerManaged>`.
- Verdict: **(already migrated)**.

### Stage R4 — rotated_in * sin — file:28-67 (`mul_tiles_chain`), called 157/162
- Already migrated: `mul_tiles_chain` wraps `eltwise_chain` (DECODE_MODE branch:
  `BinaryFpu` Mul, BroadcastDim::Row, CallerManaged, TileOffset::Set; non-decode:
  `mul<…Streaming>`). The surrounding `cb_wait_front/cb_reserve_back/cb_pop_front/
  cb_push_back` (lines 32-52) are the CallerManaged + TileOffset::Set contract
  (Set requires a Bulk-family/CallerManaged lifecycle; the wait/pop are emitted
  externally by design).
- Verdict: **(already migrated)**.

### Stage R5 — in * cos — file:166 (`mul_tiles_chain<in_cb, updated_cos_cb, cos_interm_cb>`)
- Already migrated (same wrapper). All three CBs constexpr.
- Verdict: **(already migrated)**.

### Stage R6 — cos_interm + sin_interm -> out — file:174
- Already migrated: `compute_kernel_lib::add<cos_interm_cb, sin_interm_cb, out_cb>`.
- Verdict: **(already migrated)**.

**Kernel total: 0 unmigrated eltwise stages. 2 OUT-OF-SCOPE (tilize/untilize).**

---

## 2. rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama_sharded.cpp

Non-fused. All CBs `constexpr` (lines 19-29). Already FULLY MIGRATED for eltwise.

### Stage L1 — rotated = x @ trans_mat — file:61-68
- LLK: `mm_init_short` + `matmul_tiles` + `pack_tile` in a Wt loop.
- Verdict: **OUT-OF-SCOPE:matmul**.

### Stage L2 — sin_interm = rotated * sin (bcast ROW) — file:74-84
- Already migrated: `mul<… BroadcastDim::Row, Bulk, HeldBulk, Bulk, …, OperandKind::Block>`
  over `EltwiseShape::tiles(Wt, block_size=Wt)`.
- Verdict: **(already migrated)**.

### Stage L3 — cos_interm = x * cos (bcast ROW) — file:87-97
- Already migrated: same shape as L2.
- Verdict: **(already migrated)**.

### Stage L4 — out = cos_interm + sin_interm — file:100-110
- Already migrated: `add<… Bulk/Bulk/Bulk, OperandKind::Block>`.
- Verdict: **(already migrated)**.

**Kernel total: 0 unmigrated eltwise stages. 1 OUT-OF-SCOPE (matmul).**

---

## 3. rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded.cpp

FUSED. `in_cb` / `out_cb` / `Ht` are RUNTIME (non-constexpr), selected via `is_q`
(lines 23-41). Only the sin stage (constexpr-only CBs) is migrated.

### Stage F1 — rotated = x @ trans_mat — file:86-93
- LLK: `mm_init_short` + `matmul_tiles(in_cb, trans_mat_cb, …)` + `pack_tile`, Wt loop.
- Reads runtime `in_cb` AND is matmul.
- Verdict: **OUT-OF-SCOPE:matmul** (also would be BLOCKED:runtime-selected CB id `in_cb`).

### Stage F2 — sin_interm = rotated_in_interm * sin (bcast ROW) — file:108-118
- Already migrated: `mul<… BroadcastDim::Row, CallerManaged×3, …, OperandKind::Block>`
  over `EltwiseShape::tiles(Wt, block_size=Wt)`. Inputs `rotated_in_interm_cb`,
  `sin_cb` and output `sin_interm_cb` are all constexpr. Outer
  cb_push_back/cb_pop_front (119-120) are the CallerManaged contract.
- Verdict: **(already migrated)**.

### Stage F3 — cos_interm = x * cos (bcast ROW) — file:122-130
- LLK: `mul_tiles_bcast<BroadcastType::ROW>(in_cb, cos_cb, j,j,j)` + `pack_tile`, Wt loop,
  inside `ACQ()/REL()`.
- **Reads runtime `in_cb`** (non-constexpr). `cos_cb`/`cos_interm_cb` constexpr.
- Chain shape it *would* take (if in_cb were constexpr):
  `mul<in_cb, cos_cb, cos_interm_cb, BroadcastDim::Row, CallerManaged, CallerManaged,
   CallerManaged, …, OperandKind::Block>(EltwiseShape::tiles(Wt, Wt))`
  with outer push/pop kept (mirrors F2).
- Verdict: **BLOCKED:runtime-selected CB id (`in_cb`)**. ACQ/REL, the Wt subblock
  loop, and bcast are NOT blockers — the only blocker is the non-constexpr `in_cb`.
  (Migratable-via-q/k-branch-duplication; author declined for TRISC2 size.)

### Stage F4 — out = cos_interm + sin_interm — file:132-144
- LLK: `add_tiles_init` + `add_tiles(cos_interm_cb, sin_interm_cb, j,j,j)` +
  `pack_tile(j, out_cb, j)`, Wt loop, inside ACQ/REL.
- Inputs `cos_interm_cb`/`sin_interm_cb` constexpr; **output `out_cb` is runtime**.
- Chain shape it *would* take:
  `add<cos_interm_cb, sin_interm_cb, out_cb, BroadcastDim::None, CallerManaged,
   CallerManaged, CallerManaged, …, OperandKind::Block>(EltwiseShape::tiles(Wt, Wt))`
  (or Bulk lifecycles like sibling kernel #2's L4) with outer wait/push/pop kept.
- Verdict: **BLOCKED:runtime-selected CB id (`out_cb`)**.
  (Migratable-via-duplication; declined for size.)

**Kernel total: 2 unmigrated eltwise stages, both BLOCKED:runtime CB
(F3 in_cb, F4 out_cb). 1 OUT-OF-SCOPE (matmul, F1). 1 already migrated (F2).**

---

## 4. rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded_row_major.cpp

FUSED, row-major. Single-tile compute per ht (no inner Wt loop). Same runtime
`in_cb`/`out_cb` selection as #3 (lines 23-41).

### Stage F1 — rotated = x @ trans_mat — file:70-77
- LLK: `mm_init_short` + `matmul_tiles(in_cb,…)` + `pack_tile` (single tile).
- Verdict: **OUT-OF-SCOPE:matmul** (input also runtime `in_cb`).

### Stage F2 — sin_interm = rotated * sin (single tile) — file:87-96
- Already migrated: `mul<… BroadcastDim::None, CallerManaged×3, …>(1)`.
  All CBs constexpr. Outer push/pop (97-98) kept.
- Verdict: **(already migrated)**.

### Stage F3 — cos_interm = x * cos (single tile) — file:100-107
- LLK: `mul_tiles_init(in_cb, cos_cb)` + `mul_tiles(in_cb, cos_cb, 0,0,0)` +
  `pack_tile(0, cos_interm_cb, 0)`, inside ACQ/REL.
- **Reads runtime `in_cb`.** `cos_cb`/`cos_interm_cb` constexpr.
- Chain shape it *would* take:
  `mul<in_cb, cos_cb, cos_interm_cb, BroadcastDim::None, CallerManaged,
   CallerManaged, CallerManaged, BinaryDataFormatReconfig::Input,
   PackTileReconfig::None>(1)` with outer push/pop kept (mirrors F2).
- Verdict: **BLOCKED:runtime-selected CB id (`in_cb`)**.

### Stage F4 — out = cos_interm + sin_interm (single tile) — file:109-119
- LLK: `add_tiles_init` + `add_tiles(cos_interm_cb, sin_interm_cb, 0,0,0)` +
  `pack_tile(0, out_cb, 0)`, inside ACQ/REL.
- Inputs constexpr; **output `out_cb` runtime**.
- Chain shape it *would* take:
  `add<cos_interm_cb, sin_interm_cb, out_cb, BroadcastDim::None, CallerManaged,
   CallerManaged, CallerManaged, …>(1)` with outer wait/push/pop kept.
- Verdict: **BLOCKED:runtime-selected CB id (`out_cb`)**.

**Kernel total: 2 unmigrated eltwise stages, both BLOCKED:runtime CB
(F3 in_cb, F4 out_cb). 1 OUT-OF-SCOPE (matmul, F1). 1 already migrated (F2).**

---

## Summary

| Kernel | Already migrated | Unmigrated eltwise | OUT-OF-SCOPE | "runtime CB" claim |
|---|---|---|---|---|
| rotary_embedding.cpp | R3,R4,R5,R6 | 0 | tilize, untilize, (no matmul) | N/A — all CBs constexpr |
| rotary_embedding_llama_sharded.cpp (non-fused) | L2,L3,L4 | 0 | matmul (L1) | N/A — all CBs constexpr |
| fused_qk sharded.cpp | F2 | **2** (F3,F4) | matmul (F1) | **HOLDS** — in_cb/out_cb are non-constexpr |
| fused_qk sharded_row_major.cpp | F2 | **2** (F3,F4) | matmul (F1) | **HOLDS** — in_cb/out_cb are non-constexpr |

- **No clean-MIGRATABLE stages found.** Every remaining raw eltwise stage is
  BLOCKED on a genuine non-constexpr CB template-argument constraint.
- The two non-fused kernels are fully migrated; their only leftovers are
  matmul/tilize/untilize (out of scope).
- The fused_qk PARTIAL documentation is **accurate on the substance** (runtime CB)
  but **mis-cites the cause**: the real blocker is the non-constexpr CB NTTP, not
  "TRISC2 code-size budget" (code size is policy-NOT-a-blocker). Both stages are
  structurally migratable if the chain were duplicated under the two constexpr
  q/k branches; the author declined that for size.
