# SFPI 7.93.0: ICE in `rvtt_synth_renumber` (gimple-rvtt-synth.cc:470) on a `TTI_REPLAY` whose synthesized instruction word reaches the call through a PHI

Found while rebasing the bring-up branch onto llk_helper_library 107f838f379 (SFPI 7.89.0 -> 7.93.0). Filed as a
tt-metal issue (tenstorrent/sfpi has issues disabled).

## Summary

`riscv-tt-elf-g++ -O3 -mcpu=tt-bh-tensix` crashes compiling a Blackhole TRISC1 (math) compute kernel:

```
during GIMPLE pass: rvtt_synth_renumber
sparse_sdpa_compute_trisc1_H32.ii.cpp: In function 'void kernel_main()':
sparse_sdpa_compute_trisc1_H32.ii.cpp:40757:6: internal compiler error: in build, at config/riscv/tt/gimple-rvtt-synth.cc:470
40757 | void kernel_main() {
      |      ^~~~~~~~~~~
0x7b17cb029d8f __libc_start_call_main
	../sysdeps/nptl/libc_start_call_main.h:58
0x7b17cb029e3f __libc_start_main_impl
	../csu/libc-start.c:392
gcc (tenstorrent/sfpi:7.93.0[1087]) 15.1.0
```

In tt-metal it shows up as a kernel build failure at JIT time: `TT_THROW program.cpp:433`, with the ICE reported from
the LTO link (`lto1`, `In function '_start'`). Compiling with `-fno-lto` gives the same ICE in `kernel_main`.

## Versions

Each release's x86_64 debian `.txz` was checked against the sha256 in that release's `tt_metal/sfpi-version`. Each one
was tested twice: (a) built from the tt-metal source with that release's own `include/` (the headers changed between
releases), and (b) on the preprocessed reproducer below.

| SFPI                  | tt-metal bump              | (a) from source, H=32 | (b) reproducer `.ii`                         |
|-----------------------|----------------------------|-----------------------|----------------------------------------------|
| 7.89.0 [1050]         | (old base)                 | n/a¹                  | n/a¹                                          |
| 7.90.0 [1068]         | a02a3fea40c (#62327)       | OK                    | OK                                            |
| **7.91.0 [1077]**     | 43dd65d79a6 (#62693)       | **ICE** (gimple-rvtt-synth.cc:**468**) | **ICE** (:468)               |
| 7.92.0 [1080]         | 4323a08c6bb (#62951)       | OK                    | n/a² (header mismatch)                        |
| 7.93.0 [1087]         | 012c19beaae (#63458)       | **ICE** (:470)        | **ICE** (:470)                                |

- The first version that ICEs is **7.91.0**. 7.92.0 builds the same source, but its `lltt.h` encodes the replay operand
  differently: `__builtin_rvtt_ttreplay(__instrn_buffer, (start << replay_start_shift_) | length, ...)`, against
  separate `START`/`LENGTH` operands in 7.90/7.91/7.93. 7.93.0 goes back to separate operands, and the ICE is back.
  So the crash seems tied to the separate-operand form of `ttreplay` when `LENGTH` is not a constant.
- ¹ 7.89.0 cannot compile the current tt-metal LLK headers (e.g. no `__builtin_rvtt_ttinsn`). It did build the same
  kernel from the older tt-metal base.
- ² The `.ii` was preprocessed with the 7.93.0 headers, so 7.92.0 rejects the `ttreplay` redeclaration.

## Reproducer (self-contained: one file, one command)

File: `sparse_sdpa_compute_trisc1_H32.ii.cpp.xz` next to this report (183 KB; `xz -d` gives the preprocessed TU, 41k
lines, 1.8 MB, sha256 `e95b09a02d3a76977461ad9779b3e501fb572dbe7a9d179bffb3e13b9392d42e`). It needs no includes or
defines.

```
riscv-tt-elf-g++ -O3 -mcpu=tt-bh-tensix -std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval -ftt-no-dyninit \
    -fno-exceptions -fno-rtti -c sparse_sdpa_compute_trisc1_H32.ii.cpp -o /dev/null
```

Expected result: the ICE above, about 1 s on x86_64.

The source is tt-metal's `ttnn.bringup` sparse SDPA compute kernel (`sparse_sdpa_compute.cpp`, ROW_BFP4 path), built
for TRISC1 (`UCK_CHLKC_MATH`) with 32 heads (`KERNEL_COMPILE_TIME_ARGS=32,16,16,4,...`). With 64, 96, 128, 160 or 256
heads the kernel compiles. With 352 heads it ICEs. Both ICE cases have one query tile-row per DST pass (`qsb == 1`),
which makes the matmul's `rt_dim`/`ct_dim` loops disappear.

A hand-written reduction was not reached in the time available: small kernels with the same shape (runtime replay
length from a 3-way ternary, two replays per tile, unrolled loops) all compiled. A line-level delta reduction was
started and stopped early.

## What triggers it (from `-fdump-tree-all`)

The no-mop matmul (`tt-llk .../experimental/llk_math_matmul_custom_no_mop.h`, `_llk_math_matmul_no_mop_`) replays a
recorded MVMUL sequence with `lltt::replay(replay_buf_offset /*16*/, replay_buf_len)`. Here `replay_buf_len` is
`matmul_no_mop_replay_len(...)` (4, 8 or 16), which depends on the in1 operand's tile geometry. The geometry is read
from the `unpack_*` tables indexed by a CB id chosen at runtime (`cb_k = chunk > 0 ? cb_k4 : cb_k_in`), so the length
is not a compile-time constant and the replay instruction word has to be synthesized.

Just before the crashing pass (`279t.dce9`), `kernel_main` contains the following. Jump threading and tail-merge or
PRE turned the two possible lengths into a PHI, and the synthesized words `li + (len << 4)` became a PHI of two
`synth_opcode`-derived sums:

```
li_1691 = __builtin_rvtt_synth_opcode (2, 0);
sum_1692 = li_1691 + 128;                       // length 8
__builtin_rvtt_ttreplay (0B, 8, 0, 0, 16, 0, 0);
...
<bb 49>:
_1742 = li_1691 + 256;                          // length 16
...
<bb 50>:
# _1836 = PHI <8(48), 16(49)>
# prephitmp_1732 = PHI <sum_1692(48), _1742(49)>
__builtin_rvtt_ttreplay (&__instrn_buffer, _1836, prephitmp_1732, 2, 16, 0, 0);
__builtin_rvtt_ttreplay (&__instrn_buffer, _1836, prephitmp_1732, 2, 16, 0, 0);
```

The partial `280t.rvtt_synth_renumber` dump shows the pass failing to trace the instruction word back through the PHI:

```
2:Following sum_1692 = li_1691 + 128;
2:  Unfollowable prephitmp_1732 = PHI <sum_1692(48), _1742(49)>
2:Following _1742 = li_1691 + 256;

EMERGENCY DUMP:
```

The expected behaviour is either to handle a PHI of synth words that come from the same `synth_opcode`, or to stop
the earlier passes from merging synth words into a PHI.

Compiler flags (compiling the reproducer with `-O3`):

| added flag                                                                                   | result |
|----------------------------------------------------------------------------------------------|--------|
| (none)                                                                                       | ICE    |
| `-fno-tree-pre`, `-fno-tree-tail-merge`, `-fno-thread-jumps`, `-fno-tree-dominator-opts`, `-fno-tree-fre` | OK (each one alone) |
| `-fno-ssa-phiopt`, `-fno-tree-sink`, `-fno-split-paths`, `-fno-code-hoisting`, `-fno-tree-vrp`, `-fno-tree-loop-im` | still ICE |
| `-O2` instead of `-O3` (this compile only)                                                   | OK     |

In tt-metal's real JIT build, a kernel-level `-O2` or `-O1` does not avoid it: the LTO link still ICEs. `-Os` avoids
the ICE but breaks the build because TTI immediates are no longer constant.

## Workaround (tt-metal side, not applied)

The four H=32 fork tests are waived (DeepSeek-V4.1 spec, tests.fork_waivers) until a fixed SFPI arrives.

Two workarounds were checked offline:

- **Source change in the kernel.** Call the QK matmul with a CB id that is a compile-time constant in each branch:
  `if (bfp4_chunk) qk(cb_k4); else qk(cb_k_in);`, where `qk` is the existing `blocked_matmul_and_pack` loop wrapped in a
  lambda. The replay length is then constant and no synth word is needed. Behaviour is the same (the same CB id
  reaches every call), and H=32 builds. The cost is code size: H=64's TRISC1 `.text` grows from 12532 to 13096 bytes,
  because the matmul loop is duplicated.
- **Compiler flag.** Add `-fno-tree-tail-merge` (or `-fno-tree-pre`) for the kernel. This also changes the generated
  code for configurations that compile fine today.

Making a runtime value opaque (`asm("" : "+r"(cb_k))`) does **not** avoid the crash: the 3-way ternary inside
`matmul_no_mop_replay_len` still produces the PHI.

## Offline harness

A harness that repeats the JIT's TRISC1 compile and LTO link for any SFPI release and H exists outside the repo;
ask in the issue if it helps.
