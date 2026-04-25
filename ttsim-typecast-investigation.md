# ttsim Test Failure Investigation: `test_typecast_int` with GCC Cost Tuning PR

**PR:** https://github.com/tenstorrent/sfpi-gcc/pull/17
**Date:** 2026-04-24
**Target:** Blackhole (`ARCH_NAME=blackhole`, `-mcpu=tt-bh-tensix`)
**Simulator:** ttsim (interpretive RISC-V + Tensix simulator)

## Failing Test

```
pytest "tests/ttnn/unit_tests/operations/eltwise/test_typecast_int.py::test_typecast_subcore_grid[
  shape=torch.Size([1, 2, 32, 960])-sub_core_grid={
    [(x=1,y=0) - (x=3,y=6)], [(x=5,y=0) - (x=6,y=6)]
  }]"
```

Environment variables:
```
TT_METAL_SIMULATOR=<path>/libttsim.so
TT_METAL_SLOW_DISPATCH_MODE=1
TT_METAL_DISABLE_SFPLOADMACRO=1
ARCH_NAME=blackhole
TT_METAL_FORCE_JIT_COMPILE=1
```

The test performs an element-wise typecast on integer tensors across a sub-core grid
spanning multiple Tensix tiles. It passes with the baseline GCC (`7.43.0[505]`) and
fails with the PR's GCC (`7.43.0-div-25192[521]`).

---

## 1. What the PR Changes

The PR tunes GCC's cost model for integer multiply/divide operations on the Tensix
`trisc` cores. This causes the compiler to prefer `mul` instructions (RISC-V M-extension)
over shift-subtract sequences for multiplying by constants. The change affects all code
compiled by this GCC fork, including firmware and kernel binaries.

---

## 2. Firmware Binary Differences

Only the `trisc` firmware ELFs change between before and after. The kernel ELFs are
compiled separately and are unaffected. The firmware differences were analyzed via
`objdump -d` diffs stored in the `after/` directory.

### trisc1 (the critical core)

| Property | Before | After |
|---|---|---|
| `.text` size | 0x248 (584 bytes) | 0x24c (588 bytes) |
| `.comment` | `7.43.0[505]` | `7.43.0-div-25192[521]` |
| Entry point | 0x6EE0 | 0x6EE0 |
| `__fw_export_text_end` | 0x7130 | 0x7130 |
| `MEM_TRISC1_FIRMWARE_BASE` | 0x6EE0 | 0x6EE0 |

Key instruction changes in the dispatch loop:

**Before (shift-subtract sequence, 3 instructions):**
```asm
70b8: slli  a5, a4, 0x3      # a5 = X * 8
70bc: sub   a5, a5, a4       # a5 = X * 7
70c0: slli  a5, a5, 0x4      # a5 = X * 112
```

**After (mul instruction, 2 instructions):**
```asm
70c0: mul   a4, a4, s3       # a4 = X * 112  (s3 = 112, set once at init)
70c4: mv    a5, a4           # a5 = X * 112
```

Additional differences:
- Added `sw s8, 8(sp)` (callee-save for s8)
- `li s8, 128` replaces `li s7, 128` (mailbox comparison register)
- `li s3, 112` added (constant for mul)
- Various register re-allocations (a0/a1 swapped in some uses, a6/a7 swapped)
- Exit loop moves from 0x7124 to 0x7128 (4-byte shift)

### trisc0 and trisc2

These became 4 bytes **shorter** (replaced 3-instruction shift-subtract with 1-instruction `mul`).

---

## 3. Isolation Experiments

### 3.1 Firmware Swap (All Cores)

Copied all "before" firmware ELFs (`brisc`, `ncrisc`, `trisc0`, `trisc1`, `trisc2`)
into the "after" build's JIT cache at `~/.cache/tt-metal-cache/<hash>/firmware/`.

**Result:** Test **passes**. Proves the firmware binary is the cause, not the kernel code.

### 3.2 Per-Core Isolation

Starting from all "before" firmware (passing), swapped individual firmware files to
their "after" versions one at a time:

| Swapped Core | Result |
|---|---|
| brisc only | Pass |
| ncrisc only | Pass |
| trisc0 only | Pass |
| **trisc1 only** | **FAIL** |
| trisc2 only | Pass |

**Conclusion:** Only the `trisc1` firmware change triggers the failure.

### 3.3 mul Instruction Verification

A standalone test was written to verify the RISC-V `mul` instruction works correctly
on ttsim. The test compiled and ran a compute kernel that performs `mul` operations
and validates results.

**Result:** `mul` instruction executes correctly on ttsim. The bug is not in `mul`
instruction decoding/execution.

---

## 4. Firmware Functional Equivalence Proof

Despite the instruction and register allocation differences, the before and after
firmware are **functionally identical**. This was proven through:

### 4.1 Dispatch Parameter Tracing

ttsim was instrumented to log register values at every `jalr` dispatch point (kernel
call) across all 190 kernel dispatches on all tiles.

**Semantic values at dispatch (normalized across register allocations):**

| Parameter | Before | After | Match? |
|---|---|---|---|
| Kernel target address | e.g. 0xa920 | e.g. 0xa920 | Yes, all 190 |
| `rta_l1_base` | 0x9e00 | 0x9e00 | Yes, all 190 |
| `crta_l1_base` | 0x9e00 | 0x9e00 | Yes, all 190 |
| `my_relative_x_` | varies per tile | same | Yes, all 190 |
| `my_relative_y_` | varies per tile | same | Yes, all 190 |

### 4.2 mul Operand Tracing

Every `mul` execution at pc=0x70c4 (the firmware dispatch multiply) was logged:

```
src0=0x0, src1=0x70, result=0x0   (for ALL tiles, ALL dispatches)
```

The dispatch value X is always 0, so `0 * 112 = 0`. Both shift-subtract and `mul`
produce the same result.

### 4.3 s3 Register Integrity

The callee-saved register `s3` (x19) holds the constant 112 (0x70) for the `mul`
instruction. It was verified to remain 0x70 across all kernel calls — no kernel
clobbers it.

---

## 5. Tensix Instruction Analysis

### 5.1 TTI Push Sequences

The Tensix Instruction (TTI) push sequences from TRISC1 were compared between the
before and after runs. These are the instructions that TRISC1's kernel pushes to
the math pipe (pipe=1) of the Tensix engine.

**Result:** Identical instruction sequences for tile=1. Same 383 TTI pushes, same
instructions, same order.

### 5.2 Tensix Execution (TEXEC) Sequences

The actual Tensix instruction execution was traced across all three pipes:

| Pipe | Function | Before Count | After Count | Instruction Sequence |
|---|---|---|---|---|
| 0 (unpack) | TRISC0 | 208 | 208 | **Identical** |
| 1 (math) | TRISC1 | 539 | 539 | **Identical** |
| 2 (pack) | TRISC2 | 290 | 290 | **Identical** |

Each pipe executes the **exact same instructions in the exact same order.**

### 5.3 Cross-Pipe Timing Difference

The ONLY difference is in the **relative timing** of when instructions from different
pipes execute within the same simulator clock cycle. The "after" TRISC1 firmware
reaches its kernel dispatch 1-2 clock cycles earlier than "before":

| Dispatch | Before Clock | After Clock | Delta |
|---|---|---|---|
| 1st (tile=1) | 9329 | 9327 | -2 |
| 2nd (tile=1) | 10280 | 10279 | -1 |
| 3rd (tile=1) | 11174 | 11173 | -1 |

This timing shift propagates through the kernel, causing pipe=1 instructions to
be pushed to the Tensix FIFO 1-2 clocks earlier relative to pipe=0 and pipe=2.

---

## 6. Root Cause: ttsim Multi-Pipe Execution Ordering Bug

### 6.1 The Simulator's Execution Model

ttsim executes Tensix pipe instructions **sequentially within each clock cycle**,
iterating from the lowest-numbered pipe to the highest:

```cpp
// libttsim.cpp, lines 328-349
for (uint32_t pipe_mask = p_tensix->inst_pipes_active; pipe_mask; pipe_mask &= pipe_mask-1) {
    uint32_t pipe = __builtin_ctz(pipe_mask);  // processes pipe 0, then 1, then 2
    uint32_t inst_rd_ptr = p_tensix->inst_rd_ptr[pipe];
    if (inst_rd_ptr != p_tensix->inst_wr_ptr[pipe]) {
        uint32_t inst = p_tensix->inst[pipe][inst_rd_ptr];
        if (tensix_decode_and_execute(p_tensix, pipe, inst)) {
            // advance read pointer
            inst_rd_ptr = (inst_rd_ptr + 1) % TENSIX_INST_FIFO_SIZE;
            p_tensix->inst_rd_ptr[pipe] = inst_rd_ptr;
        }
    }
}
```

When multiple pipes have instructions ready in the same clock cycle, they execute
in order: **pipe=0 (unpack) → pipe=1 (math) → pipe=2 (pack)**.

### 6.2 The Write-Before-Read Hazard

This serial execution creates a **write-before-read hazard** on shared Tensix state:

- **pipe=1 (math)** writes to DST registers (computation results)
- **pipe=2 (pack)** reads from DST registers (to store results to L1)

When both execute in the same clock cycle, pipe=1 executes **first** and writes new
values to DST. Then pipe=2 executes and reads the **just-written** values.

**In real hardware:** These pipes operate concurrently. pipe=2 would read the
**pre-math** DST values (from the previous cycle), not the values being written
by pipe=1 in the current cycle.

### 6.3 How the Timing Shift Triggers the Bug

The 1-clock timing shift from the "after" firmware changes which pipe=1 instruction
is co-scheduled with each pipe=2 instruction at certain clock boundaries.

**Concrete example at clock 11307 on tile=1:**

Before (passing):
```
clock=11307: pipe=1 inst=0xb20f0000 (math op A), then pipe=2 inst=0x45000038 (STOREIND)
```

After (failing):
```
clock=11307: pipe=1 inst=0xb21f0000 (math op B), then pipe=2 inst=0x45000038 (STOREIND)
```

The same STOREIND (pack) instruction reads DST after a **different** math instruction
has written to it. Math op A and math op B write different intermediate values to DST.
The STOREIND picks up whichever value was just written, producing different output data.

### 6.4 Affected Clock Cycles

The TEXEC diff shows instruction reorderings at multiple points in the third dispatch:

```
Lines 673-677:  pipe=2 STOREIND shifted relative to pipe=1 math ops
Lines 682-687:  pipe=0 and pipe=2 reordered around pipe=1
Lines 696-703:  pipe=0 operations shifted relative to pipe=1
```

These all stem from the same root cause: the 1-clock shift in pipe=1 instruction
timing changes the cross-pipe scheduling within individual clock cycles.

---

## 7. Conclusion

### The GCC PR is Correct

The firmware produced by the PR is functionally identical to the baseline:
- Same dispatch parameters on all 190 kernel calls across all tiles
- Same Tensix instruction sequences on all three pipes
- `mul` instruction executes correctly

### The Bug is in ttsim

ttsim has a pre-existing simulation fidelity bug: it does not properly model
concurrent multi-pipe Tensix execution. Instructions from different pipes that
execute in the same clock cycle see each other's side effects, violating the
hardware's concurrent execution semantics.

This bug is **latent** and triggered by any firmware or kernel change that shifts
the relative timing of TRISC cores by even 1 clock cycle. The GCC cost tuning
happens to change the TRISC1 firmware instruction count, shifting pipe=1 timing
and exposing the bug.

### Recommended Fix for ttsim

The Tensix execution loop should **snapshot shared state** (DST registers, src_a,
src_b, valid flags) at the beginning of each clock cycle, then have all pipes read
from the snapshot while writing to the live state. This ensures all pipes within a
cycle see the same pre-cycle state, matching real hardware behavior:

```cpp
// Pseudocode for correct multi-pipe execution
tensix_snapshot_state(p_tensix);  // capture DST, src_a, src_b at cycle start
for each active pipe:
    tensix_decode_and_execute(p_tensix, pipe, inst);  // reads from snapshot, writes to live state
```

Alternatively, process all pipe reads before any pipe writes within each cycle
(split decode/execute into read and write phases).

---

## 8. Reproduction Steps

### Reproduce the failure (after firmware)

```bash
cd /localdev/shengxiangji/div-25192/tt-metal
deactivate 2>/dev/null
export PYTHONPATH=$PWD
source python_env/bin/activate
rm -rf ~/.cache/tt-metal-cache
env TT_METAL_SIMULATOR=/localdev/shengxiangji/div-25192/ttsim-private/src/_out/debug_bh/libttsim.so \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    TT_METAL_DISABLE_SFPLOADMACRO=1 \
    ARCH_NAME=blackhole \
    TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest "tests/ttnn/unit_tests/operations/eltwise/test_typecast_int.py::test_typecast_subcore_grid[shape=torch.Size([1, 2, 32, 960])-sub_core_grid={[(x=1,y=0) - (x=3,y=6)], [(x=5,y=0) - (x=6,y=6)]}]"
```

### Reproduce the pass (swap to before firmware)

After the above failing run, swap firmware without recompilation:

```bash
CACHE_FW=$(find ~/.cache/tt-metal-cache -type d -name firmware | head -1)
BEFORE=/localdev/shengxiangji/div-25192/tt-metal-base/before/13650482829056336578/firmware
for core in brisc ncrisc trisc0 trisc1 trisc2; do
    cp "$BEFORE/$core/$core.elf" "$CACHE_FW/$core/$core.elf"
done
# Re-run without TT_METAL_FORCE_JIT_COMPILE → passes
```

### Isolate to trisc1 only

```bash
# Start from all-before (passing), swap only trisc1 to after:
AFTER=/localdev/shengxiangji/div-25192/tt-metal/after/13650482829056336578/firmware
cp "$AFTER/trisc1/trisc1.elf" "$CACHE_FW/trisc1/trisc1.elf"
# Re-run → fails
```

---

## 9. Key File Locations

| File | Description |
|---|---|
| `tt-metal-base/before/` | Passing run logs and ELF dumps (baseline GCC) |
| `tt-metal/after/` | Failing run logs and ELF dumps (PR GCC) |
| `after/.../trisc1/trisc1.elf.diff` | Disassembly diff of the critical firmware |
| `ttsim-private/src/riscv_impl.h` | RISC-V instruction execution (with added traces) |
| `ttsim-private/src/libttsim.cpp` | Main simulation loop (Tensix pipe execution) |
| `ttsim-private/src/tensix.cpp` | Tensix instruction decode/execute |
| `/tmp/ttsim_before3.log` | ttsim stderr trace for passing (before firmware) run |
| `/tmp/ttsim_trisc1only.log` | ttsim stderr trace for failing (trisc1-only after) run |
