# Quasar ↔ Wormhole parity gaps — experimental-quasar `binary_ng` (no-broadcast)

Scope: the `SubtileBroadcastType::NONE`, tensor-tensor, TILE binary op in
`ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/`. "Gap" = works on Wormhole but does
**not** (yet) work / is not validated on **Quasar**. Branch `dchen/no_bcast_quasar`.

This doc is **committed and maintained** — keep it current as Quasar / LLK / craq-sim development
closes or opens gaps (re-run `qualify_quasar_binary.py --coverage` after a foundation bump and
reconcile §2/§5 here). (`METAL2_PORT_REPORT.md` remains a local, unstaged working note.)

The qualification tooling tracks these gaps live (`tests/ttnn/unit_tests/operations/experimental/quasar/`):
`qualify_quasar_binary.py --coverage` enumerates the §2/§5 headroom straight from the LLK suite,
`--supports OP` answers "is op X in LLK yet", and the runbook `QUASAR_BINARY_QUALIFICATION.md` documents
the localize/verify-on-bump process.

## Framing (why the gate is the boundary)

The descriptor / fallback path (`ProgramFactory`, the non-v2 factory) **does not run on Quasar at
all**: it emits `DataMovementKernel` / `ComputeKernel`, whose constructors hard-throw on Quasar
(`tt_metal/impl/kernels/kernel.hpp:339,476` — "DataMovementKernel is not supported on Quasar").
So `matches_metal_v2_slice` (the DFB/v2 gate, `binary_ng_device_operation.cpp:534`) **is** the
Quasar capability boundary: anything it rejects has no working Quasar path (Wormhole runs it via
the descriptor). Additionally, **all Quasar DFB runs are slow-dispatch-only**
(`tt_metal/impl/program/dispatch.cpp:2038`).

Classifications: **[STRUCTURAL]** = needs the descriptor path (no Quasar path); **[ARCH]** = genuine
Quasar HW / host-validator / LLK wall; **[PORT]** = op/LLK port gap, op-fixable (like the bugs fixed
this session); **[UNTESTED]** = plausibly works, not validated; **[COVERAGE]** = works, not tested.

## What works on Quasar today (the validated slice)

bf16 · TILE · no-broadcast · tensor-tensor · matching lhs/rhs dtype · all-interleaved (DRAM/L1) **or**
all-L1-height/block-sharded with identical specs · add/sub (FPU) + multiply/divide (SFPU) ·
lhs/rhs + post activations. **fp32 multiply and divide also work** (see §3).

## 1. Structural — gate rejects → no Quasar path  [STRUCTURAL]

Each is a `return false` in `matches_metal_v2_slice`; WH handles it via the descriptor path.

| Config | Gate |
|---|---|
| Row-major (non-TILE) in/out | requires TILE |
| Non-32×32 tile | requires 32×32 |
| Tensor-scalar (`add(t, 5.0)`) | requires tensor-tensor |
| where / select (ternary) | rejected |
| Quantization (quant/dequant/requant) | rejected (its SFPU LLK is also unported on Quasar → [ARCH] at the kernel layer) |
| Mixed lhs/rhs dtype (e.g. bf16 + fp32) | requires `a.dtype()==b.dtype()`; rooted in the Quasar reconfig no-op (§4) |
| Mixed sharded/interleaved across a/b/out | requires uniform layout |
| Sharding ≠ L1 height/block (width-sharded, DRAM-sharded) | rejected |
| Mismatched a/b/out shard specs (grid/shape/orientation) | requires identical |
| Zero-volume (degenerate) | rejected; moot — `skip_launch` no-ops volume==0 |
| **Subtile broadcast** (scalar/row/col) | *out of no-broadcast scope, but the single largest WH-vs-Quasar surface — the next porting milestone* |

## 2. Admitted by the gate but hard-errors on Quasar  [ARCH] / [PORT]

These pass the gate (matching dtype, TILE, NONE) → route to the DFB factory → fail deep on Quasar
while WH runs them. No routing makes them work on Quasar (the descriptor would also throw). The
failure is **loud** (host throw or JIT compile error), not silent-wrong.

| Config | Failure on Quasar | Class |
|---|---|---|
| bf8_b / bf4_b (block-float) | host format validator throws — Quasar uses MX/microscaling, not block-float | [ARCH] |
| uint32 / uint16 | host validator throws — not in Quasar's format set | [ARCH] |
| Integer SFPU ops: bitwise (and/or/xor), shift, sub_int, int-div, remainder/fmod, gcd/lcm, xlogy, atan2, isclose | JIT compile fails — LLK headers are `#ifndef ARCH_QUASAR` (impls absent on Quasar) | [PORT] (op-fixable: port the LLK op) |

int32 add/mul and SFPU compare/max/min **do** compile on Quasar. If graceful behavior is wanted,
the gate (or a Quasar `validate`) could reject Quasar-unsupported formats/ops with a clear message
instead of a deep validator/compile error.

## 3. fp32 (Float32) — re-audited 2026-06-26

fp32 routes the SFPU kernel for every op (`is_binary_sfpu_op` is true for any fp32 op, incl. add).
The factory sets `fp32_dest_acc_en=true` (32-bit DEST) + `unpack_to_dest_mode=UnpackToDestFp32`.

| Op | Quasar result | Class |
|---|---|---|
| fp32 **multiply** | **WORKS** (PCC 1.0) | — |
| fp32 **divide** | **WORKS** (PCC 1.0) | — |
| fp32 **add** (and **sub**) | **JIT compile fail** — `add_binary_tile` undeclared | **[PORT]** (op-fixable, verified) |

Root cause: SFPU **MUL/DIV are ported to Quasar, ADD/SUB are not** — a 3-layer gap:
1. Compute API `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` — `add_binary_tile` /
   `add_binary_tile_init` live in an `#ifndef ARCH_QUASAR` block (mul/div have `#ifdef ARCH_QUASAR`
   branches; add does not).
2. Quasar LLK API `tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`
   — has `_mul`/`_div` wrappers, no `_add`.
3. Quasar ckernel `.../ckernel_sfpu_binary.h` — `static_assert(BINOP == MUL || DIV)`, no ADD branch.

NOT an fp32-geometry issue (the `dst_tile_size_sfpi=32` / `i*2` stride is shared with mul/div, which
pass) and NOT an arch limit (SFPU add `in0+in1` is simpler than mul/div; the Quasar `BinaryOp` enum
already has ADD; Blackhole's ckernel shows the canonical `if constexpr (BINOP==ADD) result=in0+in1`).
**Verified fix (probe):** mirror MUL/DIV in all 3 layers → fp32 add PCC 1.0 (mul/div unchanged).
The fix touches **shared Quasar LLK/compute-API files** (outside the experimental op dir), so it
needs code-review + a WH/BH SFPU-add regression check; also wire `sub_binary_tile` for full parity.

Practical impact: bf16 add/sub use the **FPU** kernel (work). Only **SFPU-routed add/sub** — i.e.
fp32 add/sub — hit this. The test skips **fp32 add/sub** on Quasar (fp32 mul/div run and pass; the op
test now covers fp32 divide too); once SFPU add/sub are ported, fp32 add/sub can be un-skipped.

## 4. Arch / HW constraints  [ARCH]

| Constraint | Effect |
|---|---|
| Quasar worker grid **8×4 (32 cores)** vs WH **8×8 (64)** (`tt_metal/soc_descriptors/quasar_32_arch.yaml`) | caps shard/core configs needing >4 rows (tall height-shards, block grids taller than y=3) |
| Format family: bf8_b/bf4_b/uint16/uint32 unsupported (`tt_backend_api_types.cpp` `is_supported_quasar`) | see §2 |
| `copy_tile_to_dst_init_short_with_dt` is a **no-op** on Quasar; `pack_reconfig_data_format` is **gasket-only** on Quasar | forced 2 of this session's 3 fixes (operand switch via `copy_tile_to_dst_init_short`, pack retarget via `pack_init`); blocks mixed-dtype (§1) |

## 5. Coverage-only — within the validated bf16 slice, expected to work, no Quasar test  [COVERAGE]

rhs pre-activation (only lhs is tested) · block-sharded **+** activation · divide **+** activation ·
larger / nD interleaved shapes (group-2 / nD-stride path) · isclose/max/min · int32 add ·
interleaved program-cache-hit. (Uneven height/block shards ARE covered by the resnet canary.)

## Priorities

1. **SFPU add/sub port** (§3) — un-blocks fp32 add/sub; shared-file change + WH/BH regression check.
2. **Gate hygiene** (§2) — optionally reject Quasar-unsupported formats/int-SFPU ops with a clear
   "unsupported on Quasar" message rather than a deep throw/compile error.
3. **Coverage** (§5) — add Quasar test cases to lock down the rest of the bf16 slice.
4. **Subtile broadcast** (§1) — the next major Quasar porting milestone beyond no-broadcast.

## Systemic lesson

Every Quasar "gap" audited this session was an **op/port bug**, not a sim/LLK/arch wall — the CB→DFB
mirror assumed WH/BH had fully-ported Quasar primitives. On Quasar: the descriptor path can't run,
`*_with_dt` reconfig is a no-op, `pack_reconfig_data_format` is gasket-only, and several SFPU ops
(int families + float add/sub) are unported. Treat "it's a sim/LLK/arch gap" as a hypothesis to
disprove with an instruction-level trace, not a conclusion.
