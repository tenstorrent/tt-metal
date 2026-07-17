# Quasar ↔ Wormhole parity gaps — experimental-quasar `binary_ng` (no-broadcast)

Scope: the `SubtileBroadcastType::NONE`, tensor-tensor, TILE binary op in
`ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/`. "Gap" = works on Wormhole but does
**not** (yet) work / is not validated on **Quasar**. Branch `dchen/no_bcast_quasar`.

This is the **op-author's view** (gate / structural / arch / test-coverage). For **which LLK primitive
is available on Quasar**, the authoritative source is the WH-baseline matrix
`qualification/QUASAR_LLK_GAPS.md` — this doc **references** it
rather than duplicating per-op LLK status (that duplication is what previously drifted). The runbook
`qualification/QUASAR_BINARY_QUALIFICATION.md` documents the localize / verify-on-bump process, and
`qualification/qualify_quasar_binary.py --coverage` / `--supports OP` track the LLK side live.
(`METAL2_PORT_REPORT.md` remains a local, unstaged working note.)

**Keep current** as Quasar / LLK / craq-sim development closes or opens gaps.

## Framing (why the gate is the boundary)

The descriptor / fallback path (`ProgramFactory`, the non-v2 factory) **does not run on Quasar at
all**: it emits `DataMovementKernel` / `ComputeKernel`, whose constructors hard-throw on Quasar
("… is not supported on Quasar", `tt_metal/impl/kernels/kernel.hpp`). So `matches_metal_v2_slice`
(the DFB/v2 gate in `binary_ng_device_operation.cpp`) **is** the Quasar capability boundary: anything
it rejects has no working Quasar path (Wormhole runs it via the descriptor). All Quasar DFB runs are
also **slow-dispatch-only** (`tt_metal/impl/program/dispatch.cpp`).
(Line-number citations were dropped on purpose — they rot on rebase; grep the symbol.)

## Three ways a config can be a "gap" (read this first)

Separating these tells you **who** must act — and, per the request, surfaces what **Quasar LLK can
already do that the op has simply not exercised.**

- **(A) LLK-incapable** — the primitive isn't on Quasar (matrix cell `kernel` / `bridge`). Needs an LLK
  port. *Not* fixable in this op.
- **(B) LLK-capable, but the gate blocks it** — Quasar LLK could run it (matrix cell `✓`), but
  `matches_metal_v2_slice` rejects the config. **Op-side work only** (relax the gate + factory/kernel
  wiring); no LLK port needed. → §1 rows marked *capable*.
- **(C) LLK-capable, gate admits, but no Quasar op test** — it should work today; it just isn't
  validated. **Just add a test.** → §5 (the coverage headroom).

(B) and (C) are the "LLK is capable but the op didn't test/enable it" set — the cheap wins.

## What works on Quasar today (the validated slice)

bf16 · TILE · no-broadcast · tensor-tensor · matching lhs/rhs dtype · all-interleaved (DRAM/L1) **or**
all-L1-height/block-sharded with identical specs · add/sub (FPU) + multiply/divide (SFPU) ·
**lhs/rhs + post activations — `relu`/`silu`/`tanh`/`square`/`sigmoid` sim-certified; `gelu` is `broken` (§5)** ·
**fp32 multiply and divide** (see §3).

## 1. Structural — gate rejects → no Quasar path via this op  [STRUCTURAL]

Each is a `return false` in `matches_metal_v2_slice`; WH handles it via the descriptor. The **LLK on
Quasar** column is the key per-request signal: *capable* = Quasar LLK could do it, so closing the gap is
**op-side work only** (class B); otherwise it also needs an LLK port (A) or is an arch wall.

| Config | LLK on Quasar | Class |
|---|---|---|
| **Subtile broadcast** (scalar/row/col) | **capable** — `unary_bcast` ported + sim-certified (§6) | **B** — op-side wiring; the single largest WH-vs-Quasar surface, next milestone |
| Tensor-scalar (`add(t, 5.0)`) | **capable** — scalar-fill + FPU/SFPU binop exist | **B** — gate requires tensor-tensor |
| where / select (ternary) | **capable** — `where` bridge is `✓` in the matrix | **B** — gate rejects ternary |
| Row-major (non-TILE) in/out | needs op dataflow work | op (gate + RM kernels), not a pure LLK gap |
| Non-32×32 tile | op work | op |
| Mixed sharded/interleaved across a/b/out | op work | op (gate + factory) |
| Sharding ≠ L1 height/block (width-, DRAM-sharded) | op work | op |
| Mismatched a/b/out shard specs (grid/shape/orientation) | op work | op |
| Mixed lhs/rhs dtype (e.g. bf16 + fp32) | **arch** — Quasar reconfig is a no-op (§4) | A/arch, not just gate |
| Quantization (quant/dequant/requant) | **unported** (matrix `kernel`) | A — gate rejects **and** LLK absent |
| Zero-volume (degenerate) | — | moot; `skip_launch` no-ops volume==0 |

## 2. Admitted by the gate but hard-errors on Quasar  [ARCH] / [PORT]

These pass the gate → route to the DFB factory → fail deep on Quasar while WH runs them. The failure is
**loud** (host throw or JIT compile error), not silent-wrong. Per-op LLK status is in the **matrix**
(Table 1) — not re-listed here; this section is about op *behavior*.

| Class of config | Failure on Quasar | Matrix cell |
|---|---|---|
| bf8_b / bf4_b (block-float), uint32 / uint16 | host format validator throws — not in Quasar's format set (MX/microscaling) | `format` |
| Integer / float SFPU ops that are unported (bitwise, shift, sub_int, int-div, remainder/fmod, gcd/lcm, xlogy, atan2, isclose, power, quant, float compares) | JIT compile fails — LLK header `#ifndef ARCH_QUASAR` | `kernel` |

int32 add/mul, SFPU compare-to-zero, and max/min **do** compile (matrix `✓`). **Gate-hygiene idea:** a
Quasar-aware `validate` could reject a Quasar-unsupported format/op with a clear "unsupported on Quasar"
message instead of a deep validator/compile error.

## 3. fp32 (Float32)

fp32 routes the SFPU kernel for **every** op (`is_binary_sfpu_op` is true for any fp32 op, incl. add);
the factory sets `fp32_dest_acc_en=true` + `unpack_to_dest_mode=UnpackToDestFp32`.

- fp32 **multiply / divide / add / sub** → **WORK** (PCC 1.0); the op test covers all four.
  (bf16 add/sub are unaffected — they use the FPU kernel.)

## 4. Arch / HW constraints  [ARCH]

| Constraint | Effect |
|---|---|
| Quasar worker grid **8×4 (32 cores)** vs WH **8×8 (64)** (`soc_descriptors/quasar_32_arch.yaml`) | caps shard/core configs needing >4 rows (tall height-shards, block grids taller than y=3) |
| Format family: bf8_b/bf4_b/uint16/uint32 unsupported (`is_supported_quasar`) | see §2 |
| `copy_tile_to_dst_init_short_with_dt` is a **no-op** on Quasar; `pack_reconfig_data_format` is **gasket-only** | forced 2 of this session's 3 op fixes (operand switch via `copy_tile_to_dst_init_short`, pack retarget via `pack_init`); is the root of the mixed-dtype block (§1) |

## 5. LLK-capable but op-untested — coverage headroom  [COVERAGE]

The matrix says Quasar LLK supports these (`✓`); `matches_metal_v2_slice` admits them; **the op has no
Quasar test.** These should pass today — the work is *a test case*, not a port (class C). This is the
direct "what Quasar LLK can do that the op didn't exercise" list.

**Fusable activations** — as of 2026-07-07 the op also tests `tanh`/`square`/`sigmoid` fusion
(`test_no_bcast_activation_supported`, post + lhs, bf16), all now **sim-certified `✓`** on Quasar. That
sweep confirmed the matrix's "supported" claim for those three — but caught that it did **NOT** hold for
`gelu`:
- **`gelu` / `bias_gelu` — `broken`, not headroom.** The Quasar gelu LLK bridge fails to JIT-compile
  (`gelu_init` calls `_sfpu_load_config32_` unqualified; it lives in `ckernel::math`, `cmath_common.h:101`).
  This is an LLK bug — a one-line qualifier fix — not a coverage gap (QUASAR_LLK_GAPS.md Table 2), tracked
  in tenstorrent/tt-metal#49314. The op test **skips `gelu` on Quasar** (it still runs on WH). Cautionary
  case: static 3-layer presence ≠ compiles.
- Still `✓` but not yet exercised: `exp` · `sqrt` · `rsqrt` · `reciprocal` · the six compare-to-zero
  (`eqz/nez/gtz/ltz/gez/lez`).

**Binary ops** (bf16, gate-admitted, matrix `✓`) — op tests only add/sub/mul/div:
- `maximum` / `minimum` (float + int32) · `add_int` / `mul_int` (int32) · `where` is capable but
  gate-blocked → §1, class B.
- Derived (FPU + supported activation): `squared_difference`, `hypot`, `logical_and`/`or`/`xor` — their
  activation pieces are `✓` (matrix Table 1). (`bias_gelu` is `broken` — it needs `gelu`, see above.)

**Config coverage** (within the validated slice): rhs pre-activation (only lhs is tested) ·
block-sharded **+** activation · divide **+** activation · larger / nD interleaved shapes (group-2 / nD
stride path) · interleaved program-cache-hit. (Uneven height/block shards ARE covered by the resnet canary.)

## 6. Subtile-broadcast foundation (`unary_bcast`) — Quasar-ready

Subtile broadcast is the next milestone (§1). Its LLK + compute-API foundation on Quasar was audited and
**sim-certified**, so the remaining work is **op-side wiring** (class B), not a foundation port.

- **Ported across all 3 layers**, on both our tt-llk pin and `origin/main`: compute API `bcast.h`
  (`unary_bcast`/`_init`/`_uninit` carry `#ifdef ARCH_QUASAR` branches — landed via #41329) · metal
  wrappers `hw/ckernels/quasar/…/{llk_unpack_A_api.h, llk_math_unary_datacopy_api.h}` · core LLK
  `tt_llk_quasar/llk_lib/{llk_unpack_unary_broadcast_operands.h, llk_math_unary_broadcast.h}` (real
  per-type SCALAR/ROW/COL bodies, no `#ifndef ARCH_QUASAR` no-op).
- **Sim-certified GREEN** (`release_qsr` libttsim.so, via `run_test.sh`): `test_unary_broadcast_quasar.py`
  bf16 **scalar / column / row** all PASS.
- **Caveats — design around these:** fp32 not wired (Quasar branch forces `unpack_to_dest=false`; start
  bf16, same theme as §3) · `reconfigure_unary_bcast` is a no-op on Quasar (init per bcast type, don't
  rely on mid-program reconfigure) · A2D/movA2D variant is unsupported (static_assert) — `unary_bcast`
  uses the B2D/SrcB path, so unaffected; do not route A2D.
- **Op-side work to enable:** relax `matches_metal_v2_slice` (rejects `SubtileBroadcastType != NONE`) and
  have the factory/kernel emit the `unary_bcast` pre-broadcast of the smaller operand.

## Priorities

1. **`gelu` bridge fix** (§5) — LLK one-liner: qualify `ckernel::math::_sfpu_load_config32_` in
   `ckernel_sfpu_gelu.h::gelu_init`. Unblocks `gelu` + `bias_gelu` (both `broken`). Highest value/effort.
2. **Remaining class-(C) coverage** (§5) — `tanh`/`square`/`sigmoid` now done; add tests for
   `maximum`/`minimum`/int-add-mul/derived ops (all matrix-`✓`). No LLK work.
3. **Subtile broadcast** (§1, §6) — the next major porting milestone; foundation is ready, so op-side
   wiring (gate + factory/kernel).
4. **Gate hygiene** (§2) — optionally reject Quasar-unsupported formats/ops with a clear message.

## Systemic lesson

Every Quasar "gap" audited this session was an **op/port/coverage matter**, not a sim/LLK/arch wall — the
CB→DFB mirror assumed WH/BH had fully-ported Quasar primitives. On Quasar the descriptor path can't run,
`*_with_dt` reconfig is a no-op, `pack_reconfig` is gasket-only, and some SFPU ops (int families) are
unported — but many primitives the op never exercised (§5) already work. Treat "it's a
sim/LLK/arch gap" as a hypothesis to disprove with an instruction-level trace and a matrix lookup, not a
conclusion.
