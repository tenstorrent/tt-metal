# The Metal 2.0 gate

Throwaway probes for `unified_metal2_spec.md` §7: *can a Metal 2.0 ProgramSpec compile one
source for all five projections and bind one DFB with a DM producer and a compute consumer?*

Run on a Wormhole n150, against `origin/main` at `8bb48ab0f1d` + the one-line adaptor change
noted below. Build:

    g++ -std=c++20 -O1 -DSPDLOG_FMT_EXTERNAL -DFMT_HEADER_ONLY \
        -I build/include -I build/include/tt-metalium \
        unified_gate/gate_host.cpp -o gate_host \
        -L build/lib -ltt_metal -ltt_stl -Wl,-rpath,$PWD/build/lib

    TT_METAL_HOME=$PWD ./gate_host unified_gate/gate_a.cpp

| file | what it is | result |
|---|---|---|
| `gate_a.cpp` + `gate_host.cpp` | one source, three KernelSpecs, five projections, two DFBs whose endpoints straddle DM and compute; compute squares each tile | **PASS**, 0/4096 values wrong |
| `gate_a_tokens.cpp` | the same kernel with `dfb::in` / `dfb::out` binding tokens instead of compile-time slot values | **compile error, as predicted** — `'out' is not a member of 'dfb'` on the reader build |
| `gate_b.cpp` + `gate_host_b.cpp` | the unified library itself (`#include <tt/unified/core>`), the shape of `unified_kernels/unary.cpp`, under a ProgramSpec with named CTAs and tensor binding tokens | **PASS**, 0/16384 values wrong |
| `GATE_OMIT_PRODUCER=1 GATE_BUILD_ONLY=1 ./gate_host_b` | drops compute's producer binding of `out` | **rejected at build**: `DFB 'out' has no producer` (program_spec.cpp:393) |

The one library change Gate B needed is in `tt/unified/adaptor_v1.hpp`: the compute-projection
`TensorAccessor` stand-in gained a one-argument constructor, so `TensorAccessor(tensor::in)`
compiles on a TRISC the same way `TensorAccessor(args, addr)` already did. Additive; the v1
suite is unaffected.
