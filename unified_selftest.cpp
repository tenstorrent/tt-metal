// Compile + trace harness for the unified programming model.
//
//   unified_expr.hpp  -- domain-free expression tree + DST register allocator
//   fusion.hpp        -- leaves, ops, fusion kinds, driver strategies
//   unified.hpp       -- core API (Storage / Block / ComputeBlock / noc_*)
//
// Those headers are a design sketch: the CB / NOC / Tensix intrinsics they call
// come from the metal kernel headers in a real build. This file supplies traced
// versions of just those, then runs the example kernels once per thread
// projection and checks that every circular buffer balances.
//
//   for s in "DM0 -DIS_DM_THREAD=1 -DTT_DM_THREAD_ID=0" \
//            "DM1 -DIS_DM_THREAD=1 -DTT_DM_THREAD_ID=1" \
//            "COMPUTE -DIS_COMPUTE_THREAD=1"; do
//     set -- $s; l=$1; shift
//     clang++-20 -std=c++17 -Wall -Wextra "$@" -DTT_LABEL="\"$l\"" \
//       unified_selftest.cpp -o /tmp/u_$l && /tmp/u_$l
//   done

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

static std::vector<std::string> trace;
static void T(const std::string& s) { trace.push_back(s); }
static std::string n(int v) { return std::to_string(v); }

// ---- CB protocol -----------------------------------------------------------
inline void cb_reserve(int cb) { T("cb_reserve(cb" + n(cb) + ")"); }
inline void cb_push(int cb) { T("cb_push   (cb" + n(cb) + ")"); }
inline void cb_wait(int cb) { T("cb_wait   (cb" + n(cb) + ")"); }
inline void cb_pop(int cb) { T("cb_pop    (cb" + n(cb) + ")"); }

// ---- NOC -------------------------------------------------------------------
inline void noc_async_read() { T("noc_async_read()"); }
inline void noc_async_write() { T("noc_async_write()"); }

// ---- Tensix compute --------------------------------------------------------
inline void tile_regs_acquire() { T("  tile_regs_acquire"); }
inline void tile_regs_commit() { T("  tile_regs_commit"); }
inline void tile_regs_wait() { T("  tile_regs_wait"); }
inline void tile_regs_release() { T("  tile_regs_release"); }
inline void copy_tile(int cb, int tile, int dst) {
    T("    copy_tile(cb" + n(cb) + ",tile=" + n(tile) + " -> dst" + n(dst) + ")");
}
inline void pack_tile(int dst, int cb) { T("  pack_tile(dst" + n(dst) + " -> cb" + n(cb) + ")"); }
inline void sfpu_add(int a, int b, int o) { T("    sfpu_add (dst" + n(a) + ",dst" + n(b) + " -> dst" + n(o) + ")"); }
inline void sfpu_relu(int a, int o) { T("    sfpu_relu(dst" + n(a) + " -> dst" + n(o) + ")"); }
inline void sfpu_sum(int a, int o) { T("    sfpu_sum (dst" + n(a) + " -> dst" + n(o) + ")"); }
inline void relu_from_pack(int base, int count) {
    T("  relu_from_pack(dst" + n(base) + "..dst" + n(base + count - 1) + ")  [replaces tile_regs_wait]");
}
inline void matmul_block(int in0, int in1, int h, int w, int kt) {
    T("    matmul_block(cb" + n(in0) + ",cb" + n(in1) + " h=" + n(h) + " w=" + n(w) + " kt=" + n(kt) + " -> dst0..dst" +
      n(h * w - 1) + ")");
}
inline void pack_block(int dst, int cb, int count) {
    T("  pack_block(dst" + n(dst) + ".." + n(dst + count - 1) + " -> cb" + n(cb) + ")");
}

#include "unified.hpp"

namespace tt {
namespace unified {

// Tensor is only forward-declared by the core header; the DM bodies never
// dereference it, so an empty definition is enough to run the examples.
class Tensor {};

// ===========================================================================
//  Example kernels
//
//  Each is written as if single-threaded. The same source is compiled once per
//  baby RISC-V thread, and each statement lowers to that thread's half of the
//  circular-buffer protocol.
// ===========================================================================

// INPUT + INTERMED + OUTPUT: two DRAM loads, an SFPU add into an intermediate,
// a second add, then a DRAM store.
void example_eltwise() {
    Tensor t0, t1, t2;
    Storage lhs_storage(0, 2);
    Storage rhs_storage(1, 2);
    Storage tmp_storage(2, 2);
    Storage out_storage(3, 2);

    for (int i = 0; i < 1; ++i) {
        ComputeBlock lhs = noc_load<0>(lhs_storage, t0, i);
        ComputeBlock rhs = noc_load<1>(rhs_storage, t1, i);

        ComputeBlock tmp = tmp_storage.store(lhs + rhs);

        Block result = out_storage.store(tmp + lhs);
        noc_store<0>(std::move(result), t2, i);
    }
}

// Two-stage reduction with a core-to-core hop in the middle.
// NOTE: the hop is not yet correct -- see the noc_write TODO in unified.hpp.
void example_reduce() {
    Tensor t0, t2;
    Coord coord_0x0{0, 0};
    Storage stage0_storage(0, 8);
    Storage stage1_storage(1, 8);
    Storage tmp_storage(2, 2);
    Storage out1_storage(3, 8);

    for (int i = 0; i < 1; ++i) {
        ComputeBlock s0 = noc_load<0>(stage0_storage, t0, i);

        Block tmp = tmp_storage.store(s0.sum());

        ComputeBlock s1 = noc_write<0>(stage1_storage, std::move(tmp), coord_0x0, /*offset=*/0);

        noc_store<0>(out1_storage.store(s1.sum()), t2, i);
    }
}

// The FPU path: matmul with a fused relu epilogue, then the SFPU path consuming
// its result out of an intermediate Storage.
void example_matmul_relu() {
    Tensor t0, t1, t2;
    Storage a_storage(0, 1);
    Storage b_storage(1, 1);
    Storage mm_storage(2, 1);
    Storage out_storage(3, 1);

    // out_subblock 2x2 = 4 DST tiles, k-dim 2, 2 inner blocks
    using Geom = MatmulGeometry</*h=*/2, /*w=*/2, /*in0_block_w=*/2, /*num_blocks=*/2>;

    ComputeBlock a = noc_load<0>(a_storage, t0, 0);
    ComputeBlock b = noc_load<1>(b_storage, t1, 0);

    // relu folds into the matmul's pack-side epilogue rather than wrapping it
    ComputeBlock mm = mm_storage.store(relu(matmul<Geom>(a, b)));

    // ... and the SFPU path picks it up from there
    noc_store<0>(out_storage.store(mm + a), t2, 0);
}

}  // namespace unified
}  // namespace tt

static bool report(const char* title) {
    printf("\n===== %s :: %s =====\n", TT_LABEL, title);
    if (trace.empty()) {
        printf("  <nothing on this thread>\n");
    }
    for (auto& s : trace) {
        printf("  %s\n", s.c_str());
    }
    bool bad = false;
    for (int cb = 0; cb <= 4; ++cb) {
        int res = 0, push = 0, wait = 0, pop = 0;
        std::string tag = "cb" + n(cb) + ")";
        for (auto& s : trace) {
            if (s.find(tag) == std::string::npos) {
                continue;
            }
            if (s.rfind("cb_reserve", 0) == 0) {
                res++;
            } else if (s.rfind("cb_push", 0) == 0) {
                push++;
            } else if (s.rfind("cb_wait", 0) == 0) {
                wait++;
            } else if (s.rfind("cb_pop", 0) == 0) {
                pop++;
            }
        }
        if (res || push || wait || pop) {
            bool ok = (res == push) && (wait == pop);
            bad |= !ok;
            printf(
                "  [cb%d] reserve=%d push=%d | wait=%d pop=%d -> %s\n",
                cb,
                res,
                push,
                wait,
                pop,
                ok ? "balanced" : "*** IMBALANCED ***");
        }
    }
    printf("  RESULT: %s\n", bad ? "*** PROTOCOL IMBALANCE ***" : "protocol balanced");
    trace.clear();
    return !bad;
}

int main() {
    bool ok = true;
    tt::unified::example_eltwise();
    ok &= report("eltwise");
    tt::unified::example_reduce();
    ok &= report("reduce");
    tt::unified::example_matmul_relu();
    ok &= report("matmul_relu");
    printf("\n%s: %s\n", TT_LABEL, ok ? "ALL BALANCED" : "FAILURES PRESENT");
    return ok ? 0 : 1;
}
