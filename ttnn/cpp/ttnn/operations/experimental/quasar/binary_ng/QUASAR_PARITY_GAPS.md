# Quasar ↔ Wormhole parity gaps — experimental-quasar `binary_ng`

Scope: the TILE binary op in `ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/` — tensor-tensor
`SubtileBroadcastType::NONE`, the single-operand subtile broadcast types (`SCALAR_A/B`, `ROW_A/B`,
`COL_A/B`), and tensor-scalar (§7). "Gap" = works on Wormhole but does **not** (yet) work / is not
validated on **Quasar**. Branch `dchen/next_bcast_quasar`.

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

**No broadcast:** bf16 · TILE · tensor-tensor · matching lhs/rhs dtype · all-interleaved (DRAM/L1) **or**
all-L1-height/block-sharded with identical specs · add/sub (FPU) + multiply/divide (SFPU) ·
**lhs/rhs + post activations — `relu`/`silu`/`tanh`/`square`/`sigmoid`/`gelu` all sim-certified** ·
**fp32 add/subtract/multiply/divide** (see §3).

**Subtile broadcast (single-operand):** all 6 types — `SCALAR_A/B`, `ROW_A/B`, `COL_A/B` — bf16 · TILE ·
tensor-tensor · add/subtract (FPU) + multiply/divide/maximum (SFPU) · interleaved **and** a sharded
broadcast operand (NoC-read path) **and** mixed a/b/out layouts, i.e. full per-operand layout
independence · lhs/rhs/post activation fusion with `relu`/`gelu`/`tanh`/`sigmoid` (see §6).

**Tensor-scalar:** bf16 and fp32 · TILE · add/subtract (FPU for bf16, SFPU for fp32) + multiply/divide
(SFPU) · a writer-fills-`in1`-once mechanism (the writer produces the RHS input DFB and fills it with the
packed scalar via a coherent, non-cacheable-L1-alias store; the compute waits on it once and reuses tile
index 0) · interleaved **and** sharded LHS (NoC-read path — a scalar never triggers the borrow path) ·
LHS activation fusion (see §7).

## 1. Structural — gate rejects → no Quasar path via this op  [STRUCTURAL]

Each is a `return false` in `matches_metal_v2_slice`; WH handles it via the descriptor. The **LLK on
Quasar** column is the key per-request signal: *capable* = Quasar LLK could do it, so closing the gap is
**op-side work only** (class B); otherwise it also needs an LLK port (A) or is an arch wall.

| Config | LLK on Quasar | Class |
|---|---|---|
| **Mixed subtile broadcast** (`ROW_A_COL_B` / `ROW_B_COL_A`) | not ported | op — gate rejects; WH via descriptor (§6) |
| Tensor-scalar, int32 (`add(t_int32, 5)`) | **capable** in isolation (int32 add/mul compile, matrix `✓`) but the DFB *compute* path returns wrong results for int32 — see §7 | **bug** — the `is_scalar` branch deliberately excludes every int32 op, not a plain LLK/gate gap |
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

int32 add/mul, SFPU compare-to-zero, and max/min **do** compile (matrix `✓!`/`✓`). int32 add/mul are the
dangerous ones: run on the DFB compute path they are **silent-wrong** (garbage / all-zero output; suspected
locus: the factory's `set_unpack_mode`, which emits an SFPU unpack mode only for `Float32`, never `Int32` —
a compute-path bug, not a gate gap; no no-bcast test exercised int32, so this shipped un-caught). **Both
gate branches now exclude int32** — the tensor-tensor no-broadcast branch as well as the tensor-scalar
`is_scalar` branch (§7) — so `int32 == int32` routes to the descriptor and throws a clean "unsupported on
Quasar" instead of returning garbage. That is a GUARD, not a fix: the compute bug is unfixed and int32 stays
off the DFB until `set_unpack_mode` handles `Int32`. **Gate-hygiene idea:** a Quasar-aware `validate` could reject a Quasar-unsupported
format/op with a clear "unsupported on Quasar" message instead of a deep validator/compile error — or, per
this finding, a silent-wrong result.

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

**Fusable activations** — the op tests `tanh`/`square`/`sigmoid`/`gelu` fusion
(`test_no_bcast_activation_supported`, post + lhs, bf16), all **sim-certified `✓`** on Quasar (`gelu`:
interleaved/height × post/lhs). `bias_gelu` (`ADD` + post `GELU`) works too.
- Still `✓` but not yet exercised: `exp` · `sqrt` · `rsqrt` · `reciprocal` · the six compare-to-zero
  (`eqz/nez/gtz/ltz/gez/lez`).

**Binary ops** (bf16, gate-admitted, matrix `✓`) — op tests only add/sub/mul/div:
- `maximum` / `minimum` (float + int32) · `add_int` / `mul_int` (int32) · `where` is capable but
  gate-blocked → §1, class B.
- Derived (FPU + supported activation): `squared_difference`, `hypot`, `logical_and`/`or`/`xor` — their
  activation pieces are `✓` (matrix Table 1).

**Config coverage** (within the validated slice): rhs pre-activation (only lhs is tested) ·
block-sharded **+** activation · divide **+** activation · larger / nD interleaved shapes (group-2 / nD
stride path) · interleaved program-cache-hit. (Uneven height/block shards ARE covered by the resnet canary.)

## 6. Subtile broadcast (`unary_bcast`) — single-operand, validated on Quasar

Single-operand subtile broadcast (`SCALAR_A/B`, `ROW_A/B`, `COL_A/B`) is wired through the DFB factory and
sim-certified **through the op itself** (`test_binary_ng_bcast.py`), not just the standalone LLK test —
see `qualification/QUASAR_LLK_GAPS.md` Table 3 for the primitive-level (`unary_bcast`) status.

- **Mechanism:** `unary_bcast<BroadcastType::{ROW,COL,SCALAR}>` all lower to the MOVB2D srcB→dest
  datacopy, differentiated only by broadcast constants (`dst_lo`/`bcast0`/`srcb_col_inc`); there is no
  `ELWADD` on the `unary_bcast` path.
- **Design constraint:** `reconfigure_unary_bcast` (mid-program bcast-type/format switch) has no Quasar
  branch (`#ifndef ARCH_QUASAR`-only) — each broadcast type is brought up via its own `unary_bcast_init`,
  not a runtime reconfigure.
- **Ops:** add/subtract (FPU) + multiply/divide/maximum (SFPU).
- **Layouts:** interleaved **and** a sharded broadcast operand (via the NoC-read sharding-aware
  `TensorAccessor` path, not borrowing) **and** mixed a/b/out layouts — full per-operand layout
  independence.
- **Activation fusion:** lhs/rhs/post × `relu`/`gelu`/`tanh`/`sigmoid` compose with broadcast (`relu` uses
  the SFPU post chain; the PACK_RELU fast path is disabled under subtile broadcast).
- **Validation:** 112 broadcast cases green on the QSR sim (`test_binary_ng_bcast.py`); 88 no-bcast
  regression cases green (`test_binary_ng_no_bcast.py`).
- **Two facts that make this work:** the `bcast.h` Quasar `unary_bcast` branch does not reference
  `DataFormat::UInt32` (Quasar has no uint32 device format — its 32-bit formats are `Float32`/`Int32`; the
  enum slot WH/BH use for `UInt32` is `MxFp4_2x_B` on Quasar) · the Quasar bcast compute inserts
  `pack_init` under `#ifdef ARCH_QUASAR` after `pack_reconfig_data_format`, which is gasket-only on Quasar
  (§4).
- **Still deferred:**
  - Mixed subtile types `ROW_A_COL_B` / `ROW_B_COL_A` — not ported; the gate rejects them (no Quasar
    path; WH runs via descriptor) → §1.
  - Tensor-scalar (`add(t, 5.0)`) is a separate, now-supported path (§7) — a scalar operand is always
    `SubtileBroadcastType::NONE`, so it never engages subtile broadcast and has no interaction with this
    section.
  - fp32 / int subtile broadcast — bf16-only: the Quasar `unary_bcast` branch forces
    `unpack_to_dest=false`, and MOVB2D is fp32-fragile (BH #449).
  - Natively-borrowed all-sharded subtile broadcast is currently **unreachable**: `is_native_L1_sharding`
    requires the broadcast operand to be *unsharded*, so `all_borrowed` can't hold for a broadcast pair
    (the `num_tiles_per_cycle == 1` guard on the bcast branch is therefore never exercised for
    broadcast). A borrowed sharded-broadcast path needs `is_native_L1_sharding` to recognize "both
    operands sharded, one a subtile broadcast" — structurally different for the `_A` vs `_B` families. A
    sharded broadcast operand already works today via the NoC-read path (see layouts, above).
  - Gate hygiene (§2) is unresolved for broadcast too: the gate admits bf16 SFPU ops that are unported on
    Quasar (float compares, `xlogy`, `atan2`, `isclose`) regardless of broadcast type — they JIT-fail;
    same open item as the no-bcast slice.
  - WH/BH execution of this broadcast path is unverified — validated on the QSR sim only; the v2 kernels
    are ports of the WH/BH `kernels_ng` reference.
  - Other `DataFormat::UInt32` references under `hw/inc/api/compute/**` should be audited for
    `!ARCH_QUASAR` guarding — `bcast.h` was the only offender in this op's include closure.

## 7. Tensor-scalar (`add(t, 5.0)`) — validated on Quasar

Tensor-scalar (`ttnn.experimental.quasar.<op>(tensor, python_scalar)`) is wired through the DFB factory via
a dedicated `is_scalar` early-return in `matches_metal_v2_slice` (`binary_ng_device_operation.cpp`) — a
different admission path from the tensor-tensor / subtile-broadcast one above, since there is no `b`
tensor and a scalar never subtile-broadcasts (`SubtileBroadcastType::NONE` always).

- **Mechanism:** with no `b` tensor, the writer becomes the producer of the RHS input DFB (`in1`) and
  fills it ONCE with the packed scalar via a coherent store — the non-cacheable L1 alias on Quasar DM
  cores, because the DM core's write-back L1 D$ is incoherent with the TL1 SRAM the compute consumer
  reads, so a plain cacheable fill would leave the consumer reading zeros or corrupt a neighbor DFB (WH/BH
  keep the plain fill; the alias offset is 0 there — the same coherence idiom the mixed subtile-broadcast
  reader uses for its COL-operand fill, `reader_row_col_mixed_bcast_dfb.cpp`, §6). The reader produces
  `in0` only. The compute waits on `in1` once, outside its per-chunk loop,
  and reuses tile index 0 for every LHS tile — fill-once, reuse-many, not a per-tile re-materialize.
- **Ops / dtypes:** add/subtract (FPU for bf16, SFPU for fp32) + multiply/divide (SFPU). All four ops are
  SFPU for fp32 on Quasar (`is_binary_sfpu_op` is dtype-aware), so the derived scalar RHS format equals `a`
  for every fp32 op; bf16 add/subtract stay FPU but still derive a bf16 RHS == `a`. The gate additionally
  requires the derived RHS format to equal `a` (an fp32 FPU-only op would derive a bf16 RHS and correctly
  fall to the descriptor) and 32×32 tiles.
- **Layouts:** interleaved and sharded LHS. A scalar never triggers the borrow path — `borrow_shards`
  requires all three operands sharded, and a scalar has no `b`, so `b_shard_volume`/`b_sharded` is
  unconditionally false; a sharded `a` (and output) is read/written over the NoC via its sharding-aware
  `TensorAccessor`, the same reader/writer code the interleaved case uses.
- **Activation fusion:** LHS (pre) activations compose with the scalar RHS (`PREPROCESS(LHS, ...)` runs
  before the binary op, identical in structure to the no-broadcast kernels' lhs-activation self-loop). The
  scalar RHS itself carries no activation chain (there is no `b` to pre-process).
- **Validation:** 24 scalar cases green on the QSR sim (`test_binary_ng_scalar.py`) — bf16
  add/subtract/multiply/divide interleaved, fp32 all four ops interleaved, sharded LHS (add + multiply),
  LHS `relu` (add + multiply).
- **Still deferred:**
  - **int32 tensor-scalar** — although the gate's format-derivation rule would admit int32 (int add/mul
    are SFPU, so the derived RHS format is int32 == `a`), the int32 SFPU tile ops do not produce correct
    results on the Quasar DFB compute path (empirically: scalar add/multiply return all-zero output tiles).
    The `is_scalar` branch explicitly excludes every int32 op regardless of which binary op, keeping it on
    the descriptor (a clean "unsupported on Quasar" instead of a silent-wrong result) until the int32
    DFB-compute path is fixed — see §2 for the matching tensor-tensor finding (int32 `add`/`mul` on the DFB
    are silent-wrong, so the tensor-tensor no-broadcast branch now also excludes int32 → clean throw).
  - **maximum / minimum tensor-scalar** — NOT a `matches_metal_v2_slice` gate rejection: their
    `ttnn.experimental.quasar` scalar overloads (`binary_composite_op.cpp`) route through
    `ttnn::operations::unary::detail::unary_impl` with `UnaryOpType::MAXIMUM`/`MINIMUM` and never call
    `invoke_binary_ng`, so this factory is never consulted for them. The Quasar unary op has no DFB /
    Metal-2.0 program factory of its own, so it falls through to the descriptor-style
    `DataMovementKernel`/`ComputeKernel` construction, which hard-throws on Quasar ("... is not supported on
    Quasar", `tt_metal/impl/kernels/kernel.hpp`). The fix is to reroute those two scalar overloads to
    `invoke_binary_ng(BinaryOpType::MAXIMUM/MINIMUM)`, not to relax a gate — `is_binary_sfpu_op` already
    returns true for `MAXIMUM`/`MINIMUM` regardless of dtype, so bf16/fp32 tensor-scalar `maximum`/`minimum`
    would be admitted by the existing gate logic unchanged once rerouted; no new LLK work is implied.
  - **row-major-scalar, where-with-scalar, quantization-scalar** — same disposition as their tensor-tensor
    counterparts (§1): the `is_scalar` branch's own row-major / `is_where_op` / `is_quant_op` checks send
    them to the descriptor.

## Priorities

1. **Fix the int32 DFB-compute bug** (§2, §7) — int32 `add`/`mul` are silent-wrong on the DFB compute path;
   both gate branches now exclude int32 (→ descriptor, clean throw) as a GUARD, but the compute bug is
   unfixed — fixing it would re-enable int32 on the DFB. Suspected locus: the factory's `set_unpack_mode`
   (`Float32`-only). Root-cause analysis captured in the int32-DFB issue draft.
2. **Remaining class-(C) coverage** (§5) — add tests for `maximum`/`minimum`/int-add-mul/derived ops
   (all matrix-`✓`). No LLK work.
3. **Reroute `maximum`/`minimum` tensor-scalar** (§7) — their `ttnn.experimental.quasar` overloads call the
   unary clamp path (`UnaryOpType::MAXIMUM`/`MINIMUM`), not `invoke_binary_ng`, so they never reach this
   factory and hard-throw on Quasar (`DataMovementKernel` not supported); reroute to
   `invoke_binary_ng(BinaryOpType::MAXIMUM/MINIMUM)`.
4. **Mixed subtile broadcast** (`ROW_A_COL_B`/`ROW_B_COL_A`, §1, §6) — remaining subtile milestone; not
   yet ported (no Quasar path), WH still runs it via the descriptor.
5. **Gate hygiene** (§2, §6) — optionally reject Quasar-unsupported formats/ops with a clear message;
   applies under broadcast too, not just the no-broadcast slice.

## Systemic lesson

Every Quasar "gap" audited this session was an **op/port/coverage matter**, not a sim/LLK/arch wall — the
CB→DFB mirror assumed WH/BH had fully-ported Quasar primitives. On Quasar the descriptor path can't run,
`*_with_dt` reconfig is a no-op, `pack_reconfig` is gasket-only, and some SFPU ops (int families) are
unported — but many primitives the op never exercised (§5) already work. Treat "it's a
sim/LLK/arch gap" as a hypothesis to disprove with an instruction-level trace and a matrix lookup, not a
conclusion.
