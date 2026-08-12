// Compile + trace harness for unified.hpp / unified_expr.hpp.
//
// unified.hpp is a design sketch: the CB / NOC / Tensix intrinsics it calls come
// from the metal kernel headers in a real build. This file supplies traced
// versions of just those, so the *structure* of each thread projection can be
// checked on the host.
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
inline void sfpu_sum(int a, int o) { T("    sfpu_sum (dst" + n(a) + " -> dst" + n(o) + ")"); }

// ---- operands referenced by test() / reduce(), declared before the include
// because those functions live inside the header ----------------------------
namespace tt {
namespace unified {
class Tensor {};
struct Coord;
extern Tensor t0;
extern Tensor t1;
extern Tensor t2;
extern Coord coord_0x0;
extern int offset;
}  // namespace unified
}  // namespace tt

#include "unified.hpp"

namespace tt {
namespace unified {
Tensor t0, t1, t2;
Coord coord_0x0{0, 0};
int offset = 0;
}  // namespace unified
}  // namespace tt

static void report(const char* title) {
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
}

int main() {
    tt::unified::test();
    report("test");
    tt::unified::reduce();
    report("reduce");
    tt::unified::matmul_relu();
    report("matmul_relu");
    return 0;
}
