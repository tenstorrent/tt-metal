// Compile + trace harness for the unified programming model.
//
//   tt/unified_expr.hpp    -- domain-free expression tree + DST allocator
//   tt/unified_math.hpp    -- leaves, ops, fusion kinds, driver strategies
//   tt/unified_api.h       -- core API (Storage / Block / ComputeBlock / noc_*)
//   tt/unified_impl_v1.hpp -- its definitions
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

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

static std::vector<std::string> trace;
static void T(const std::string& s) { trace.push_back(s); }
static std::string n(int v) { return std::to_string(v); }
static std::string n(uint32_t v) { return std::to_string(v); }
static void T2(const std::string& s) { trace.push_back(s); }

// ---- CB protocol -----------------------------------------------------------
inline void cb_reserve_back(uint32_t cb, uint32_t p) { T("cb_reserve_back(cb" + n(cb) + "," + n(p) + ")"); }
inline void cb_push_back(uint32_t cb, uint32_t p) { T("cb_push_back   (cb" + n(cb) + "," + n(p) + ")"); }
inline void cb_wait_front(uint32_t cb, uint32_t p) { T("cb_wait_front  (cb" + n(cb) + "," + n(p) + ")"); }
inline void cb_pop_front(uint32_t cb, uint32_t p) { T("cb_pop_front   (cb" + n(cb) + "," + n(p) + ")"); }

// ---- NOC -------------------------------------------------------------------
inline void noc_async_read() { T("noc_async_read()"); }
inline void noc_async_write() { T("noc_async_write()"); }

// ---- Tensix compute --------------------------------------------------------
// The model calls these as ckernel::* straight from fusion.hpp, so the harness
// fakes metal's namespace rather than a shim of its own.
namespace ckernel {
inline void tile_regs_acquire() { T("  tile_regs_acquire"); }
inline void tile_regs_commit() { T("  tile_regs_commit"); }
inline void tile_regs_wait() { T("  tile_regs_wait"); }
inline void tile_regs_release() { T("  tile_regs_release"); }
inline void copy_tile(uint32_t cb, uint32_t tile, uint32_t dst) {
    T("    copy_tile(cb" + n(cb) + ",tile=" + n(tile) + " -> dst" + n(dst) + ")");
}
inline void pack_tile(uint32_t dst, uint32_t cb) { T("  pack_tile(dst" + n(dst) + " -> cb" + n(cb) + ")"); }
inline void add_binary_tile_init() {}
inline void add_binary_tile(uint32_t a, uint32_t b, uint32_t o) {
    T("    add_binary_tile(dst" + n(a) + ",dst" + n(b) + " -> dst" + n(o) + ")");
}
inline void exp_tile_init() {}
inline void exp_tile(uint32_t o) { T("    exp_tile (dst" + n(o) + ")"); }
inline void relu_tile_init() {}
inline void relu_tile(uint32_t o) { T("    relu_tile(dst" + n(o) + ")"); }
}  // namespace ckernel

inline void compute_init(uint32_t, uint32_t) {}
inline uint32_t get_write_ptr(uint32_t) { return 0; }
inline uint32_t get_read_ptr(uint32_t) { return 0; }
inline uint32_t cb_page_bytes(uint32_t) { return 2048; }

// Stand-ins for TensorAccessorArgs / TensorAccessor, under metal's own names so
// the harness presents the same surface the device binding does.
struct FakeArgs {
    uint32_t id;
};
struct TensorAccessor {
    uint32_t id;
    constexpr TensorAccessor(FakeArgs a, uint32_t) : id(a.id) {}
    // encode (tensor, page) into the fake address so traces stay readable
    uint64_t get_noc_addr(uint32_t page) const { return (uint64_t(id) << 32) | page; }
};
inline void noc_async_read(uint64_t src, uint32_t, uint32_t) {
    T2("noc_async_read (t" + n(uint32_t(src >> 32)) + ",page=" + n(uint32_t(src & 0xffffffffu)) + ")");
}
inline void noc_async_write(uint32_t, uint64_t dst, uint32_t) {
    T2("noc_async_write(t" + n(uint32_t(dst >> 32)) + ",page=" + n(uint32_t(dst & 0xffffffffu)) + ")");
}
inline uint64_t get_noc_addr(uint32_t x, uint32_t y, uint32_t addr) {
    T2("get_noc_addr(" + n(x) + "," + n(y) + ")");
    return addr;
}
inline void noc_async_read_barrier() { T2("noc_async_read_barrier()"); }
inline void noc_async_writes_flushed() { T2("noc_async_writes_flushed()"); }
inline void noc_async_write_barrier() { T2("noc_async_write_barrier()"); }
inline void relu_from_pack(uint32_t base, uint32_t count) {
    T("  relu_from_pack(dst" + n(base) + "..dst" + n(base + count - 1) + ")  [replaces tile_regs_wait]");
}
namespace ckernel {
inline void matmul_block(uint32_t in0, uint32_t in1, uint32_t h, uint32_t w, uint32_t kt) {
    T("    matmul_block(cb" + n(in0) + ",cb" + n(in1) + " h=" + n(h) + " w=" + n(w) + " kt=" + n(kt) + " -> dst0..dst" +
      n(h * w - 1) + ")");
}
inline void pack_block(uint32_t dst, uint32_t cb, uint32_t count) {
    T("  pack_block(dst" + n(dst) + ".." + n(dst + count - 1) + " -> cb" + n(cb) + ")");
}
}  // namespace ckernel

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
#include <cassert>
#define ASSERT(x) assert(x)
#endif

#define TT_UNIFIED_CUSTOM_BINDING 1
#include <tt/unified>

namespace tt {
namespace unified {

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
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t1 = TensorAccessor(FakeArgs{1}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    Storage lhs_storage(0, 2);
    Storage rhs_storage(1, 2);
    Storage tmp_storage(2, 2);
    Storage out_storage(3, 2);

    for (int i = 0; i < 1; ++i) {
        ComputeBlock lhs = noc_load<0>(lhs_storage, t0, i).wait();
        ComputeBlock rhs = noc_load<1>(rhs_storage, t1, i).wait();

        ComputeBlock tmp = tmp_storage.store(lhs + rhs);

        Block result = out_storage.store(tmp + lhs);
        noc_store<0>(std::move(result), t2, i);
    }
}

// A unary chain: out = exp(in). Exercises Un<> and the in-place SFPU path.
void example_unary() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    Storage in_storage(0, 2);
    Storage out_storage(3, 2);

    ComputeBlock x = noc_load<1>(in_storage, t0, 0).wait();
    noc_store<0>(out_storage.store(x.exp()), t2, 0);
}

// Core-to-core hop, exercising NocAsyncCopyTx. The peer handshake is still not
// right (see the TODO in unified.hpp) -- this only checks that the local half of
// the protocol balances and that the handle's two halves fire.
void example_peer_hop() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    Coord peer{0, 0};
    Storage in_storage(0, 2);
    Storage hop_storage(1, 2);
    Storage out_storage(3, 2);

    ComputeBlock x = noc_load<1>(in_storage, t0, 0).wait();
    Block staged = hop_storage.store(x.exp());
    ComputeBlock y = noc_write<0>(out_storage, std::move(staged), peer, 0).wait();
    noc_store<0>(out_storage.store(y.exp()), t2, 0);
}

// The FPU path: matmul with a fused relu epilogue, then the SFPU path consuming
// its result out of an intermediate Storage.
void example_matmul_relu() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t1 = TensorAccessor(FakeArgs{1}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    Storage a_storage(0, 1);
    Storage b_storage(1, 1);
    Storage mm_storage(2, 1);
    Storage out_storage(3, 1);

    // out_subblock 2x2 = 4 DST tiles, k-dim 2, 2 inner blocks
    using Geom = MatmulGeometry</*h=*/2, /*w=*/2, /*in0_block_w=*/2, /*num_blocks=*/2>;

    ComputeBlock a = noc_load<0>(a_storage, t0, 0).wait();
    ComputeBlock b = noc_load<1>(b_storage, t1, 0).wait();

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
            if (s.rfind("cb_reserve_back", 0) == 0) {
                res++;
            } else if (s.rfind("cb_push_back", 0) == 0) {
                push++;
            } else if (s.rfind("cb_wait_front", 0) == 0) {
                wait++;
            } else if (s.rfind("cb_pop_front", 0) == 0) {
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
    tt::unified::example_unary();
    ok &= report("unary");
    tt::unified::example_matmul_relu();
    ok &= report("matmul_relu");
    tt::unified::example_peer_hop();
    ok &= report("peer_hop");
    printf("\n%s: %s\n", TT_LABEL, ok ? "ALL BALANCED" : "FAILURES PRESENT");
    return ok ? 0 : 1;
}
