# Quasar Uplift Report — `ttnn/cpp/ttnn/operations/reduction/accumulation`

Recipe: `docs/source/ttnn/ttnn/ai/quasar_porting.md`, with the canonical passes it extends
(`ai/audit/quasar_audit.md`, `ai/audit/cb_dfb_quasar_audit_helper.md`,
`ai/post_port/semantic/gen2_hardware_configs.md`, `ai/post_port/pass_procedure.md` from branch
`akertesz/op-porting-recipe`).

**Leave this file uncommitted; delete it before merge.**

---

## Status: RED

The op directory holds two program factories, and both are blocked on a dependency that lives
outside the op directory. Neither blocker is fixable inside this op, so per the recipe's
definition-of-done both are flagged here for a dedicated PR rather than bundled into this diff.

| Factory | Kernels | Uplift verdict | Blocker |
|---|---|---|---|
| `AccumulationProgramFactory` (cumsum / cumprod) | reader, writer, compute | **RED** | `api/compute/eltwise_unary/fill.h` has no Quasar branch: its unconditional `#include "sfpu/ckernel_sfpu_fill.h"` does not resolve on the Quasar include path. Everything else in this factory audits Quasar-clean. |
| `EmaProgramFactory` | reader, writer, compute | **RED** | The EMA SFPU LLK is not ported to Quasar at all. `ckernel_sfpu_ema.h` / `llk_math_ema_sfpu_entry.h` exist only for `tt_llk_wormhole_b0` and `tt_llk_blackhole`. |

Both are Gen1 Metal 2.0 already (`create_program_artifacts` → `ProgramArtifacts`; kernels use
`dfb::` / `args::` / `tensor::`), so the RED is not the "not Metal 2.0 yet" condition. It is the
"an LLK the op needs is a stub / unported" condition, once for each factory.

The **only** change applied is the canonical `gen2_hardware_configs.md` post-port pass, which is a
pass in its own right and is owed regardless of the uplift verdict. No §7–§8 reactive fix was
applied: nothing has been built or run on any target, so no symptom has fired.

---

## Files changed

| File | Reason |
|---|---|
| `device/accumulation_program_factory.cpp` | `gen2_hardware_configs.md` shape 4 (compute): the hand-written `ComputeGen1Config` gained a `ComputeGen2Config` alternative selected on `device->arch()`, with the prescribed `TODO(#52269)` `unpack_modes` marker. |
| `ema/device/ema_program_factory.cpp` | `gen2_hardware_configs.md` shape 4 (data movement): the two hand-written `DataMovementGen1Config`s were hoisted verbatim to locals and gained a default-constructed `DataMovementGen2Config{}` alternative under one arch branch. Two comments the pass falsified were repaired (see below). |

The op's directory and namespace are unchanged. Nothing was copied into
`ttnn/cpp/ttnn/operations/experimental/quasar/`; no `::qsr` or other Quasar-only namespace was
introduced; that tree was not opened, cited, or used as evidence.

### Repaired because the pass falsified it

- `ema/device/ema_program_factory.cpp`, the comment above the NOC lookups: it claimed *"This factory
  targets Gen1 (Wormhole / Blackhole) only … it pins this whole factory to Gen1: a Gen2 build would
  need placement decisions that cannot be derived from these values."* The pass gives the factory a
  Gen2 data-movement config, and Gen2 has no processor / NOC placement concept, so there are no
  placement decisions to derive. The surviving sentences, which explain why the reader / writer
  cannot use the architecture-agnostic helpers, are unchanged.
- `ema/device/ema_program_factory.cpp`, the comment on the compute kernel's `hw_config`: it claimed
  *"the data movement kernels above are Gen1-only, so the whole factory is."* No longer true.

### `unpack_modes` marker placement

The `TODO(#52269)` marker sits on the accumulation factory's `ComputeGen2Config`, the one place this
pass causes Quasar's `unpack_modes` to take a Gen1-derived value. The EMA factory carries no marker:
its two sites are shape-4 *data movement*, and `DataMovementGen2Config` has no `unpack_modes` field,
while its compute config comes from `to_compute_hardware_config(arch, …)` with nothing set afterwards
(shape 2, no work). That absence is meaningful, not an omission.

---

## Audit findings

### `quasar_audit.md` check 1 — device-side CB / DFB redesign

Every DFB in both factories is at its Quasar end-state already. No `evil_set_*`, no
`get_local_cb_interface`, no `fifo_wr_ptr` / `fifo_rd_ptr` surgery, no `fifo_page_size` read, and no
bare `get_read_ptr` / `get_write_ptr` peek appears anywhere under the op directory.

| DFB | Binding kernels | Class | 2xx (Quasar) status |
|---|---|---|---|
| `src` (accumulation) | reader PRODUCER, compute CONSUMER | 1, linear FIFO | Portable |
| `dst` (accumulation) | compute PRODUCER, writer CONSUMER | 1, linear FIFO | Portable |
| `acc` (accumulation) | compute PRODUCER **and** CONSUMER | 5, compute accumulator | Portable — compute self-loop, the sanctioned Gen2 end-state |
| `src` (EMA) | reader PRODUCER, compute CONSUMER | 1, linear FIFO | Portable |
| `dst` (EMA) | compute PRODUCER, writer CONSUMER | 1, linear FIFO | Portable |
| `prev` (EMA) | compute PRODUCER **and** CONSUMER | 5, packer round trip | Portable — compute self-loop |

`acc` and `prev` are both same-kernel PRODUCER + CONSUMER on a **compute** kernel, with a canonical
PACK → UNPACK tile stream. That is a `SELF-LOOP-CANDIDATE` already realised, and a compute self-loop
is legal on both generations. Neither is a **DM** self-loop, which is the case Gen2 rejects, so no
`Scratchpad` / `LocalTensorAccessor` conversion is due. No `borrowed_from` DFB, so no shard-capacity
check applies.

### `quasar_audit.md` check 2 — non-zero-initialized semaphores

Neither factory creates a semaphore of any kind. Nothing to remove.

### §7–§12 walk

| Item | Verdict |
|---|---|
| `disable_dfb_implicit_sync_for_all` / `disable_dfb_implicit_sync_for` | Not set anywhere, and the pass did not introduce one. Both factories rely on the Gen2 implicit-sync default. |
| `compute_kernel_hw_startup` exactly once at `main()` start | Clean. EMA calls it once at `ema_compute.cpp:88`. Accumulation reaches the same hardware configuration through `unary_op_init_common(dfb::in, dfb::out)` once at `accumulation_compute.cpp:31`; that helper has its own Quasar branch. No `hw_configure` sits mid-kernel in either. |
| Re-init on every DFB-id change | **Unpack side, accumulation: clean** — `copy_tile_to_dst_init_short` is re-run for each operand before every `copy_tile` (`accumulation_compute.cpp:74`, `:78`). **Pack side and EMA unpack side: risk, see "Deferred" below.** |
| Every DFB has a valid `data_format_metadata` | Clean. All six `DataflowBufferSpec`s set it. |
| Kernels read sizes via a whitelisted getter | Clean. All six read sites use the `DataflowBuffer::get_tile_size()` **member** getter, which is section-A whitelisted and arch-safe. The Gen1-only construct is the free-function token form `get_tile_size(dfb::x)`, which this op does not use. No kernel reads `fifo_page_size`. |
| Quasar has Int32, no uint16 / uint32 | **Risk for accumulation, see "Deferred".** |
| Tilize / wide-tilize / DEST wrap | Not applicable. Neither kernel tilizes. |
| NoC / multicast, degenerate-grid mcast clamp, W-dim tail padding | Not applicable. No multicast; both dataflow kernels address whole tiles through a `TensorAccessor` by `page_id`. |
| `MEM_ZEROS_BASE` | Not used. |
| RM shard width 16-byte alignment | Not applicable. Both ops require TILE layout. |
| `evil_set_read_ptr` / `evil_set_write_ptr` ring rewind | Not used, so the Gen1-only DFB rewind API is not a blocker here. |
| 32-bit unpack-to-dest transpose (`tt-llk#1559`) | **Checked and clear for EMA.** Quasar's `transpose_init` / `transpose_tile` assert when the operand's unpack destination format is Float32 or Int32. EMA validates input and output to BFLOAT16, sets no `unpack_modes`, and `get_single_unpack_dst_format` leaves a `Float16_b` source at `Float16_b`, so the assert condition is false. Accumulation does not transpose. |
| `opt_level` | Unchanged by this pass, and correct as it stands: both compute kernels state `O3` explicitly and both dataflow kernels leave it absent, which resolves to the O2 data-movement default. |

---

## Deferred / follow-up items

### 1. `fill.h` has no Quasar branch — blocks cumsum / cumprod on Quasar. Dedicated PR.

**Symptom to expect:** the accumulation compute kernel's MATH TRISC JIT build fails on Quasar with
`fatal error: sfpu/ckernel_sfpu_fill.h: No such file or directory`, from
`tt_metal/hw/inc/api/compute/eltwise_unary/fill.h:9`.

**Why:** that `#include` is inside `#ifdef TRISC_MATH` but carries no arch guard. Quasar's fill SFPU
LLK exists — `tt_metal/tt-llk/tt_llk_quasar/common/inc/experimental/ckernel_sfpu_fill.h` — but it
sits under `experimental/`, not `sfpu/`, so the path does not resolve on the Quasar include list.
The entry points also differ in signature, so the include path is not the whole fix:

| | Gen1 (`sfpu/ckernel_sfpu_fill.h`) | Quasar (`experimental/ckernel_sfpu_fill.h`) |
|---|---|---|
| `_calculate_fill_bitcast_` | `<APPROXIMATION_MODE, ITERATIONS>` | `<ITERATIONS>` |
| `_calculate_fill_` | `<APPROXIMATION_MODE, ITERATIONS>` | `<ITERATIONS>` |
| `_calculate_fill_int_` | `<APPROX, InstrModLoadStore, ITERATIONS>`, supports Int32 / UInt32 / UInt16 | `<DataFormat FMT, ITERATIONS>`, supports Int32 / Int16 / Int8 / UInt8 |

`ckernel::SfpuType::fill` **does** exist on Quasar (`tt_llk_quasar/llk_lib/llk_defs.h`), so
`fill_tile_init()`'s `SFPU_UNARY_INIT(fill)` needs nothing.

**Why it is not fixed here:** the fix belongs in a shared framework header, which the post-port pass
procedure puts out of scope, and the format-set difference in the third row is a mapping decision
rather than a straight copy. The wanted change follows the `#ifdef ARCH_QUASAR` branch that
`api/compute/add_int_sfpu.h` and `api/compute/mul_int_sfpu.h` already carry in the same directory:
switch the include, and give the Quasar branch its own `static_assert` naming the formats Quasar
supports.

**What this op needs from it:** `fill_tile_bitcast` for float and bfloat16 output, and
`fill_tile_int<DataFormat::Int32>` for int32 output. Both are within what the Quasar LLK already
provides, so the branch this op needs is a straight copy; only the wider format set is a decision.

### 2. EMA SFPU LLK is unported on Quasar — blocks EMA. LLK team.

**Symptom to expect:** the EMA compute kernel's MATH TRISC JIT build fails on Quasar with
`fatal error: llk_math_ema_sfpu_entry.h: No such file or directory`, from
`tt_metal/hw/inc/api/compute/ema.h:9`.

**Why:** `llk_math_ema_sfpu_entry.h` exists only under
`tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/`, and the microcode it wraps,
`ckernel_sfpu_ema.h`, exists only under `tt_llk_wormhole_b0` and `tt_llk_blackhole`. There is no
Quasar counterpart of either, and `api/compute/ema.h` has no arch guard because there is nothing to
guard to. All three entry points the kernel calls are affected: `ema_init`,
`ema_clear_previous_output`, `ema_tile`.

This is the recipe's "an LLK the op needs is a stub / unported" RED condition. EMA cannot run on
Quasar until the SFPU is ported; no amount of op-side work changes that. Once it lands, the EMA
factory's Gen2 data-movement configs are already in place from this pass, and items 3 and 4 below
become the next things to check.

### 3. Pack-output DFB alternation without `pack_init` — Quasar semantic risk, op-owner decision.

Both compute kernels switch the packer's output DFB inside their inner loop:

- `accumulation_compute.cpp` alternates `acc` → `out` → `acc` (`:61`, `:92`, `:98`), reconfiguring
  with `pack_reconfig_data_format` (`:60`, `:91`, `:97`) but never calling `pack_init`.
- `ema_compute.cpp` alternates `trp` → `dst` (`:106`, `:119`) with no reconfig or init at all.

On Gen1 that is correct. On Quasar it is not obviously so:
`api/compute/reconfig_data_format.h` states that buffer descriptors are programmed at op init and
that `pack_reconfig_data_format` reprograms only the THCON input data format, not the MOP or the
buffer descriptors, so *"when the pack output operand changes, call `pack_init(new_cb_id)` before
`pack_tile`."* That is §7's "re-init on every DFB-id change" rule on the pack side.

**Not applied, deliberately.** The naive fix is unsafe, and this is exactly the "I would have had to
decide that myself" case. On Quasar, `llk_pack_init` does more than reprogram the descriptor: with
`UnpackToDestEn` and `DstSync::SyncHalf` it calls `_reset_dest_register_offset_()` and resets the
dest section base to bank 0 (`tt_metal/hw/ckernels/quasar/metal/llk_api/llk_pack_tile_api.h:35-40`).
The accumulation factory sets `unpack_modes[acc] = UnpackToDest`, which makes `UnpackToDestEn` true
kernel-wide, and SyncHalf DEST is a two-bank ping-pong that never wraps. Dropping a `pack_init`
between `tile_regs_wait()` and `pack_tile()` would therefore reset the bank pointer mid-ping-pong,
which is the documented recipe for a `0x19`. Choosing between "re-init and lose the bank position"
and "skip the re-init and risk a stale descriptor" is a semantic decision for the op owner together
with the LLK owners, not a mechanical uplift step.

The same question applies to EMA's **unpack** side: `transpose_init(dfb::src)` is called once before
the loop (`ema_compute.cpp:90`), and the loop then transposes from `src` and from `trp` alternately
(`:99`, `:113`). Under the same rule the operand change wants the op init re-run. On Gen1 it works
because `src` and `prev` are specified with the same entry size, format and geometry.

### 4. Non-Int32 integer dtypes on Quasar — expect a JIT `static_assert`, not a graceful error.

`accumulation_program_factory.cpp` derives the accumulator format from the output dtype and templates
the SFPU calls on it, so a UINT32 or UINT16 output produces `add_int_tile<DataFormat::UInt32>` /
`mul_int_tile<DataFormat::UInt32>` / `fill_tile_int<DataFormat::UInt32>` in the kernel. On Quasar
`add_int_tile` and `mul_int_tile` carry
`static_assert(data_format == DataFormat::Int32, "Unsupported data format for … on Quasar")`, so the
JIT build fails rather than the op refusing the dtype at validation time.

Not fixed here: the only clean fix is an arch-conditional dtype check in the device operation's
`validate_on_program_cache_miss`, and the device-operation class is the op's contract with its
callers, which the post-port pass procedure puts out of scope. It is also not urgent for the current
test surface: `test_cumsum.py` and `test_cumprod.py` exercise float32, bfloat16 and int32 only.
Raise it with the op owners as a separate, deliberate decision.

### 5. Observation, not this op's to fix: `SIGN_MAGNITUDE_FORMAT` on Quasar's integer SFPU.

`api/compute/add_int_sfpu.h:44-48` and `api/compute/mul_int_sfpu.h:44-50` both pass
`SIGN_MAGNITUDE_FORMAT = true` on the Quasar branch, directly beneath a comment saying *"Native Int32
tiles use 2's-comp dest and keep SIGN_MAGNITUDE_FORMAT=false."* The accumulation int32 path feeds
native Int32 tiles, not Int8 tiles reconstructed through the FPU, so my read is that the comment
describes the case this op is in and the argument contradicts it. I have not traced the LLK far
enough to call it a defect. Worth a look from the compute-API owners before the int32 cumsum /
cumprod numerics are trusted on Quasar, since a wrong sign convention here would show up as bad PCC
rather than as a failure.

### 6. Observation: `unpack_modes` is per-DFB on Gen1 and kernel-wide on Gen2.

The accumulation factory's comment at `device/accumulation_program_factory.cpp:141-144` explains a
per-DFB choice: `acc` unpacks to Dest, and `src` joins it only when the input is not `Float16_b`. On
Quasar the generated `UnpackToDestEn` is a single kernel-wide flag, true if **any** DFB routes to Dest
(`tt_metal/jit_build/genfiles.cpp:938-945`), so `src` will route to Dest on Quasar even for a
bfloat16 input. Quasar's `llk_unpack_A_init` supports that explicitly and states it routes 16-bit
operands to Dest too when asked, so this should be functionally fine, and Gen2 has no penalty for
unpacking to Dest. It is a real Gen1 / Gen2 semantic difference behind a copied value, which is what
the `TODO(#52269)` marker exists to flag. Left alone; the marker points at it.

---

## Parity claim

**WH and BH keep the original path.** No test was run, so this is argued structurally rather than
measured; the commands to confirm it are below.

- The only changes are the two hardware-config sites. Every Gen1 initializer survives with the same
  fields, the same values and the same order: the accumulation `ComputeGen1Config` block is
  byte-for-byte unchanged, and the two EMA `DataMovementGen1Config` initializers were moved verbatim
  from inside their `KernelSpec` designated initializers to locals, not retyped.
- The Gen2 alternative is reachable only under `device->arch() == tt::ARCH::QUASAR`. On WH and BH the
  variant holds exactly the config it held before, so the `ProgramSpec` those targets build is
  identical.
- No custom config was rerouted through an architecture-agnostic TTNN helper, so no field the
  factories had set has silently flipped to a helper default.
- No kernel source changed at all, so no JIT output changes on any target.
- `clang-format --dry-run --Werror` is clean on both files.

**Nothing has been compiled.** There is no build tree in this checkout, so neither factory has been
through a compiler since the edit. Treat the first build as the check on the pass, not as a
formality.

---

## Commands for the human

Build (from the repository root):

```bash
./build_metal.sh -e --enable-fake-kernels-target
```

Gen1 parity sentinels — must be green on **Blackhole first, then Wormhole**, and must match their
pre-change results:

```bash
source python_env/bin/activate && pytest \
  tests/ttnn/unit_tests/operations/reduce/test_cumsum.py \
  tests/ttnn/unit_tests/operations/reduce/test_cumprod.py \
  tests/ttnn/unit_tests/operations/reduce/test_ema.py \
  2>&1 | tee /tmp/accumulation_gen1.log
```

Quasar emulator, once the two blockers above are resolved. Set `TT_METAL_SIMULATOR` (or
`TT_METAL_SIMULATOR_BASE`), `NNG_SOCKET_ADDR` and `NNG_SOCKET_LOCAL_PORT` as
`tests/scripts/quasar/run_quasar_regression.sh` documents, then force a JIT rebuild because the
kernels are being built for a new target:

```bash
source python_env/bin/activate && TT_METAL_FORCE_JIT_COMPILE=1 pytest \
  tests/ttnn/unit_tests/operations/reduce/test_cumsum.py \
  2>&1 | tee /tmp/accumulation_quasar.log
```

Run it **both** with `TT_METAL_LLK_ASSERTS` set and unset, starting with it set: several Quasar hangs
only assert with it on, and a pass with it unset is not proof. `tt-triage`, `tt-exalens` and
device-side gdb are not available on the emulator, so the tools left are DPRINT (which needs
`TT_METAL_LLK_ASSERTS` unset and the `DPRINT("fmt {}", args)` form), `log_debug()`, WATCHER, the LLK
and lightweight asserts, and host-side gdb.
