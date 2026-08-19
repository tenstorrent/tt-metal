// Compile + trace harness for the unified programming model.
//
//   tt/unified/expr.hpp    -- op-agnostic tree, DST allocator, method syntax
//   tt/unified/math.hpp    -- leaves, ops, fusion kinds, driver strategies
//   tt/unified/api.h       -- core API (Storage / Block / ComputeBlock / noc_*)
//   tt/unified/impl_v1.hpp -- its definitions
//
// Those headers are a design sketch: the CB / NOC / Tensix intrinsics they call
// come from the metal kernel headers in a real build. This file supplies traced
// versions of just those, then runs the example kernels once per thread
// projection, checks that every circular buffer balances, and checks that the
// method and free-function spellings of the ops emit the same instructions.
//
// -Werror is deliberate. This is the only build that compiles the headers with
// -Wextra: the JIT uses -Wall -Werror without it, so -Wunused-parameter and friends
// are simply off there. A warning only this build can see should fail it rather
// than scroll past.
//
//   for s in "DM0 -DIS_DM_THREAD=1 -DTT_DM_THREAD_ID=0" \
//            "DM1 -DIS_DM_THREAD=1 -DTT_DM_THREAD_ID=1" \
//            "COMPUTE -DIS_COMPUTE_THREAD=1"; do
//     set -- $s; l=$1; shift
//     clang++-20 -std=c++17 -Wall -Wextra -Werror -I. "$@" -DTT_LABEL="\"$l\"" \
//       unified_selftest.cpp -o /tmp/u_$l && /tmp/u_$l
//   done

#include <sys/mman.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

// One tile per page, and room for a handful of them.
static constexpr uint32_t kPageBytes = 2048;
static constexpr uint32_t kFakeL1Bytes = 64 * 1024;

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

enum class PoolType { SUM, AVG, MAX };
enum class ReduceDim { REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR };

inline void reconfig_data_format(uint32_t srca, uint32_t srcb) {
    T("    reconfig_data_format(cb" + n(srca) + ",cb" + n(srcb) + ")");
}
template <PoolType pool, ReduceDim dim>
inline void reduce_init(uint32_t icb, uint32_t iscaler, uint32_t ocb) {
    T("  reduce_init(cb" + n(icb) + ",scaler=cb" + n(iscaler) + " -> cb" + n(ocb) + ")");
}
template <PoolType pool, ReduceDim dim>
inline void reduce_tile(uint32_t icb, uint32_t iscaler, uint32_t itile, uint32_t, uint32_t idst) {
    T("    reduce_tile(cb" + n(icb) + ",tile=" + n(itile) + ",scaler=cb" + n(iscaler) + " -> dst" + n(idst) + ")");
}
inline void reduce_uninit(uint32_t icb = 0) { T("  reduce_uninit(cb" + n(icb) + ")"); }

inline void add_bcast_rows_init_short(uint32_t icb0, uint32_t icb1) {
    T("  add_bcast_rows_init(cb" + n(icb0) + ",cb" + n(icb1) + ")");
}
inline void add_tiles_bcast_rows(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    T("    add_bcast_rows(cb" + n(icb0) + ",tile=" + n(itile0) + " + cb" + n(icb1) + ",tile=" + n(itile1) + " -> dst" +
      n(idst) + ")");
}
inline void pack_tile(uint32_t dst, uint32_t cb) { T("  pack_tile(dst" + n(dst) + " -> cb" + n(cb) + ")"); }
inline void add_binary_tile_init() {}
inline void add_binary_tile(uint32_t a, uint32_t b, uint32_t o) {
    T("    add_binary_tile(dst" + n(a) + ",dst" + n(b) + " -> dst" + n(o) + ")");
}
inline void sub_binary_tile_init() {}
inline void sub_binary_tile(uint32_t a, uint32_t b, uint32_t o) {
    T("      sub_binary_tile(dst" + n(a) + ",dst" + n(b) + " -> dst" + n(o) + ")");
}
inline void mul_binary_tile_init() {}
inline void mul_binary_tile(uint32_t a, uint32_t b, uint32_t o) {
    T("      mul_binary_tile(dst" + n(a) + ",dst" + n(b) + " -> dst" + n(o) + ")");
}
inline void div_binary_tile_init() {}
inline void div_binary_tile(uint32_t a, uint32_t b, uint32_t o) {
    T("      div_binary_tile(dst" + n(a) + ",dst" + n(b) + " -> dst" + n(o) + ")");
}
inline void exp_tile_init() {}
inline void exp_tile(uint32_t o) { T("    exp_tile (dst" + n(o) + ")"); }
inline void relu_tile_init() {}
inline void relu_tile(uint32_t o) { T("    relu_tile(dst" + n(o) + ")"); }
inline void recip_tile_init() {}
inline void recip_tile(uint32_t o) { T("    recip_tile(dst" + n(o) + ")"); }
inline void sqrt_tile_init() {}
inline void sqrt_tile(uint32_t o) { T("    sqrt_tile(dst" + n(o) + ")"); }
inline void rsqrt_tile_init() {}
inline void rsqrt_tile(uint32_t o) { T("    rsqrt_tile(dst" + n(o) + ")"); }
inline void copy_tile_to_dst_init_short_with_dt(uint32_t old_cb, uint32_t new_cb) {
    T("  copy_tile_to_dst_init_short_with_dt(cb" + n(old_cb) + " -> cb" + n(new_cb) + ")");
}
inline void copy_block(uint32_t cb, uint32_t start_tile, uint32_t start_dst, uint32_t n_tiles) {
    T("  copy_block(cb" + n(cb) + "[" + n(start_tile) + "] -> dst" + n(start_dst) + ".." + n(start_dst + n_tiles - 1) +
      ")  [reload]");
}
inline void reconfig_data_format_srca(uint32_t old_cb, uint32_t new_cb) {
    T("  reconfig_data_format_srca(cb" + n(old_cb) + " -> cb" + n(new_cb) + ")");
}
inline void init_sfpu(uint32_t icb, uint32_t ocb) { T("  init_sfpu(cb" + n(icb) + " -> cb" + n(ocb) + ")"); }
}  // namespace ckernel

// A stand-in for L1. Almost every stub only hands an address to another stub, so a
// fake value would do -- but fill_reduce_scaler DEREFERENCES the write pointer to lay
// the scaler pattern down, and the model's addresses are uint32_t, so the memory has
// to live below 4GB.
//
// The address is FIXED, not merely sub-4GB, and that is the load-bearing part. L1
// addresses reach the trace: noc_core_write folds one into a NOC address, which the
// noc_async_write stub prints. A kernel-chosen address would make the trace differ
// run to run under ASLR, and comparing traces is how every refactor here is checked.
// Verified by running one binary twice and diffing.
//
// MAP_FIXED_NOREPLACE fails rather than clobbering an existing mapping, so a clash
// is a loud abort instead of silent corruption.
inline uint32_t* l1_base() {
    static uint32_t* base = [] {
        void* want = reinterpret_cast<void*>(uintptr_t{0x30000000});
        void* m =
            mmap(want, kFakeL1Bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (m != want) {
            fprintf(stderr, "selftest: could not map the L1 stand-in at a fixed sub-4GB address\n");
            abort();
        }
        return static_cast<uint32_t*>(m);
    }();
    return base;
}

inline uint32_t get_write_ptr(uint32_t) { return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(l1_base())); }
inline uint32_t get_read_ptr(uint32_t) { return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(l1_base())); }
inline uint32_t cb_page_bytes(uint32_t) { return kPageBytes; }

// An L1 pointer attribute on device (risc_attribs.h); nothing on the host.
#define tt_l1_ptr

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
// --- semaphores / coordinates -----------------------------------------------
enum class ProgrammableCoreType { TENSIX };
enum class NocOptions : uint32_t { DEFAULT = 0, MCAST_INCL_SRC = 1 };
inline uint32_t noc_index = 0;
inline uint32_t my_x[2] = {1, 1};
inline uint32_t my_y[2] = {2, 2};
inline uint32_t worker_logical_row_to_virtual_row[8] = {2, 3, 4, 5, 6, 7, 8, 9};
inline uint32_t worker_logical_col_to_virtual_col[8] = {1, 2, 3, 4, 5, 6, 7, 8};
inline uint32_t get_relative_logical_x() { return 0; }
inline uint32_t get_relative_logical_y() { return 0; }
inline uint64_t get_noc_multicast_addr(uint32_t xs, uint32_t ys, uint32_t xe, uint32_t ye, uint32_t addr) {
    T2("get_noc_multicast_addr(" + n(xs) + "," + n(ys) + ".." + n(xe) + "," + n(ye) + ")");
    return addr;
}
inline void noc_async_write_multicast(uint32_t, uint64_t, uint32_t, uint32_t num_dests) {
    T2("noc_async_write_multicast(dests=" + n(num_dests) + ")");
}
template <ProgrammableCoreType = ProgrammableCoreType::TENSIX>
inline uintptr_t get_semaphore(uint32_t id) {
    return 0x9000 + id * 16;
}
class Noc {
public:
    Noc() = default;
    uint8_t get_noc_id() const { return 0; }
};
template <ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX>
class Semaphore {
public:
    explicit Semaphore(uint32_t id) : id_(id) {}
    void wait(uint32_t v) { T2("sem" + n(id_) + ".wait(" + n(v) + ")"); }
    void wait_min(uint32_t v) { T2("sem" + n(id_) + ".wait_min(" + n(v) + ")"); }
    void set(uint32_t v) { T2("sem" + n(id_) + ".set(" + n(v) + ")"); }
    void up(const Noc&, uint32_t, uint32_t, uint32_t v, uint8_t = 0) { T2("sem" + n(id_) + ".up(" + n(v) + ")"); }
    template <NocOptions = NocOptions::DEFAULT>
    void set_multicast(const Noc&, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t d, bool = false) {
        T2("sem" + n(id_) + ".set_multicast(dests=" + n(d) + ")");
    }
    void inc_multicast(const Noc&, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t v, uint32_t d) {
        T2("sem" + n(id_) + ".inc_multicast(" + n(v) + ",dests=" + n(d) + ")");
    }

private:
    uint32_t id_;
};

inline void noc_async_read_barrier() { T2("noc_async_read_barrier()"); }
inline void noc_async_writes_flushed() { T2("noc_async_writes_flushed()"); }
inline void noc_async_write_barrier() { T2("noc_async_write_barrier()"); }
inline void noc_async_atomic_barrier() { T2("noc_async_atomic_barrier()"); }
inline void noc_async_write_multicast_loopback_src(uint32_t l1, std::uint64_t, uint32_t bytes, uint32_t dests) {
    T2("noc_async_write_multicast_loopback_src(l1=" + n(l1) + "," + n(bytes) + "B, dests=" + n(dests) + ")");
}
namespace ckernel {
enum class SrcOrder { Regular, Reverse };
template <SrcOrder order = SrcOrder::Regular>
inline void compute_kernel_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    T("  hw_startup(cb" + n(icb0) + ",cb" + n(icb1) + " -> cb" + n(ocb) +
      (order == SrcOrder::Reverse ? ", SrcOrder::Reverse)" : ")"));
}
inline void matmul_block_init(uint32_t in0, uint32_t in1, uint32_t transpose, uint32_t ct, uint32_t rt, uint32_t kt) {
    (void)transpose;
    T("  matmul_block_init(cb" + n(in0) + ",cb" + n(in1) + " ct=" + n(ct) + " rt=" + n(rt) + " kt=" + n(kt) + ")");
}
inline void matmul_block(
    uint32_t in0,
    uint32_t in1,
    uint32_t in0_tile,
    uint32_t in1_tile,
    uint32_t idst,
    uint32_t transpose,
    uint32_t ct,
    uint32_t rt,
    uint32_t kt) {
    (void)transpose;
    (void)kt;
    T("    matmul_block(cb" + n(in0) + "[" + n(in0_tile) + "],cb" + n(in1) + "[" + n(in1_tile) + "] -> dst" + n(idst) +
      ".." + n(idst + rt * ct - 1) + ")");
}
inline void pack_reconfig_l1_acc(uint32_t en) { T("  pack_reconfig_l1_acc(" + n(en) + ")"); }
inline void pack_block(uint32_t dst, uint32_t cb, uint32_t count) {
    T("  pack_block(dst" + n(dst) + ".." + n(dst + count - 1) + " -> cb" + n(cb) + ")");
}
}  // namespace ckernel

#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
#include <cassert>
#define ASSERT(x) assert(x)
#endif

#define TT_UNIFIED_CUSTOM_BINDING 1
#include <tt/unified/core>

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
    using Row2 = Shape<1, 2>;
    Storage<Row2> lhs_storage(0);
    Storage<Row2> rhs_storage(1);
    Storage<Row2> tmp_storage(2);
    Storage<Row2> out_storage(3);

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
    using Row2 = Shape<1, 2>;
    Storage<Row2> in_storage(0);
    Storage<Row2> out_storage(3);

    ComputeBlock x = noc_load<1>(in_storage, t0, 0).wait();
    noc_store<0>(out_storage.store(x.exp()), t2, 0);
}

// A reduction, which is the only path that both fills a scaler and drives
// Strategy<ReduceFusion>. fill_reduce_scaler is the reason the harness needs a real
// L1 stand-in: it is the one call in the model that writes through the pointer rather
// than handing it to an intrinsic. Stage 1 of the shape refactor broke it and NO
// selftest example caught it -- the device build did -- so it earns a permanent one.
//
// The scaler is deliberately NOT popped here: it is held as a kernel-scope
// ComputeBlock whose destructor pops it, which is what report() checks balances.
void example_reduce() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    using In = Shape<2, 2>;
    using Out = reduce_shape<In, ReduceAxis::Rows>;
    Storage<In> in_storage(0);
    Storage<Shape<1, 1>> scaler_storage(3);
    Storage<Out> out_storage(4);

    ComputeBlock scaler = fill_reduce_scaler<1>(scaler_storage);
    ComputeBlock a = noc_load<1>(in_storage, t0, 0).wait();
    noc_store<0>(out_storage.store(reduce_sum<ReduceGeometry<2, 2>, ReduceAxis::Rows>(a, scaler)), t2, 0);
}

// Core-to-core hop, exercising NocAsyncCopyTx. The peer handshake is still not
// right (see the TODO in unified.hpp) -- this only checks that the local half of
// the protocol balances and that the handle's two halves fire.
void example_peer_hop() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    LogicalCoord peer{0, 0};
    using Row2 = Shape<1, 2>;
    Storage<Row2> in_storage(0);
    Storage<Row2> hop_storage(1);
    Storage<Row2> out_storage(3);

    ComputeBlock x = noc_load<1>(in_storage, t0, 0).wait();
    Block staged = hop_storage.store(x.exp());
    ComputeBlock y =
        noc_core_write<0>(out_storage, std::move(staged), peer, /*write_predicate=*/true).wait(/*num_writers=*/1);
    noc_store<0>(out_storage.store(y.exp()), t2, 0);
}

// The two spellings of every unary op, side by side. The method form delegates to
// the free function, so these must emit the same instructions; report_same() is
// what keeps that true rather than merely intended. Keep the pair in lockstep --
// a new op belongs in both.
//
// Only the COMPUTE projection can actually catch a divergence: the unaries emit
// nothing on a data-movement thread, so DM0/DM1 compare identical traces however
// wrong the mixin is. Verified by making Fluent::relu delegate to exp_ -- COMPUTE
// reported it and exited non-zero, DM0 passed regardless.
void example_syntax_free() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    // One buffer set per SHAPE the probe needs, now that Storage::store checks that
    // the destination is exactly what the expression produces. The old single set was
    // silently inconsistent -- a MatmulGeometry<2,2,2> result stored into a
    // Storage<Shape<1,2>> -- which is what the new assert caught.
    using Row2 = Shape<1, 2>;  // eltwise operands and result
    using Sq2 = Shape<2, 2>;   // matmul operands and output block
    using Col2 = Shape<2, 1>;  // reduce input
    using One = Shape<1, 1>;   // scaler, and a rows-collapse result
    Storage<Row2> in0(0), in1(1), out(2);
    Storage<Sq2> mm_a(5), mm_b(6), mm_out(7);
    Storage<Col2> red_in(8);
    Storage<One> scaler(3), red_out(4);
    using RG = ReduceGeometry</*ht=*/2, /*wt=*/1>;

    {  // Bin -- EVERY binary. Left-associated and non-commutative on purpose:
       // a swapped operand order shows up as swapped dst indices in the trace.
        ComputeBlock a = noc_load<1>(in0, t0, 0).wait();
        ComputeBlock b = noc_load<1>(in1, t0, 0).wait();
        noc_store<0>(out.store(relu(((a + b) - a) * b / a)), t2, 0);
    }
    {  // ComputeBlock
        ComputeBlock a = noc_load<1>(in0, t0, 0).wait();
        noc_store<0>(out.store(exp_(a)), t2, 0);
    }
    {  // Un, chained -- EVERY unary, in one chain
        ComputeBlock a = noc_load<1>(in0, t0, 0).wait();
        noc_store<0>(out.store(rsqrt(sqrt_(recip(exp_(relu(a)))))), t2, 0);
    }
    {  // MatmulNode -- appends to the epilogue chain
        ComputeBlock a = noc_load<1>(mm_a, t0, 0).wait();
        ComputeBlock b = noc_load<1>(mm_b, t0, 0).wait();
        noc_store<0>(mm_out.store(rsqrt(sqrt_(recip(exp_(relu(matmul(a, b))))))), t2, 0);
    }
    {  // ReduceNode -- likewise
        ComputeBlock sc = noc_load<1>(scaler, t0, 0).wait();  // resident operand
        ComputeBlock a = noc_load<1>(red_in, t0, 0).wait();
        noc_store<0>(red_out.store(rsqrt(sqrt_(recip(exp_(relu(reduce_sum<RG, ReduceAxis::Rows>(a, sc))))))), t2, 0);
    }
}

void example_syntax_method() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    // One buffer set per SHAPE the probe needs, now that Storage::store checks that
    // the destination is exactly what the expression produces. The old single set was
    // silently inconsistent -- a MatmulGeometry<2,2,2> result stored into a
    // Storage<Shape<1,2>> -- which is what the new assert caught.
    using Row2 = Shape<1, 2>;  // eltwise operands and result
    using Sq2 = Shape<2, 2>;   // matmul operands and output block
    using Col2 = Shape<2, 1>;  // reduce input
    using One = Shape<1, 1>;   // scaler, and a rows-collapse result
    Storage<Row2> in0(0), in1(1), out(2);
    Storage<Sq2> mm_a(5), mm_b(6), mm_out(7);
    Storage<Col2> red_in(8);
    Storage<One> scaler(3), red_out(4);
    using RG = ReduceGeometry</*ht=*/2, /*wt=*/1>;

    {  // Bin -- EVERY binary. Left-associated and non-commutative on purpose:
       // a swapped operand order shows up as swapped dst indices in the trace.
        ComputeBlock a = noc_load<1>(in0, t0, 0).wait();
        ComputeBlock b = noc_load<1>(in1, t0, 0).wait();
        noc_store<0>(out.store((((a + b) - a) * b / a).relu()), t2, 0);
    }
    {  // ComputeBlock
        ComputeBlock a = noc_load<1>(in0, t0, 0).wait();
        noc_store<0>(out.store(a.exp()), t2, 0);
    }
    {  // Un, chained -- EVERY unary, in one chain
        ComputeBlock a = noc_load<1>(in0, t0, 0).wait();
        noc_store<0>(out.store(a.relu().exp().recip().sqrt().rsqrt()), t2, 0);
    }
    {  // MatmulNode -- appends to the epilogue chain
        ComputeBlock a = noc_load<1>(mm_a, t0, 0).wait();
        ComputeBlock b = noc_load<1>(mm_b, t0, 0).wait();
        noc_store<0>(mm_out.store(matmul(a, b).relu().exp().recip().sqrt().rsqrt()), t2, 0);
    }
    {  // ReduceNode -- likewise
        ComputeBlock sc = noc_load<1>(scaler, t0, 0).wait();  // resident operand
        ComputeBlock a = noc_load<1>(red_in, t0, 0).wait();
        noc_store<0>(red_out.store(reduce_sum<RG, ReduceAxis::Rows>(a, sc).relu().exp().recip().sqrt().rsqrt()), t2, 0);
    }
}

// Single-shot matmul: one k-block straight through Storage::store(), no
// accumulation buffer.
void example_matmul_single() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t1 = TensorAccessor(FakeArgs{1}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);

    using Sq2 = Shape<2, 2>;
    Storage<Sq2> a_storage(0);
    Storage<Sq2> b_storage(1);
    Storage<Sq2> out_storage(3);

    ComputeBlock a = noc_load<1>(a_storage, t0, 0).wait();
    ComputeBlock b = noc_load<1>(b_storage, t1, 0).wait();
    noc_store<0>(out_storage.store(matmul(a, b)), t2, 0);
}

// The FPU path: a two-k-block matmul accumulated through a separate buffer,
// with a DST-side relu epilogue on the final block.
void example_matmul_acc() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t1 = TensorAccessor(FakeArgs{1}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);

    constexpr uint32_t kBlocks = 2;  // a kernel loop bound, not a geometry

    using Sq2 = Shape<2, 2>;
    Storage<Sq2> a_storage(0);
    Storage<Sq2> b_storage(1);
    Storage<Sq2> acc_storage(24);  // running total -- a different CB from out
    Storage<Sq2> out_storage(3);

    Accumulator<Sq2, AccumulatorMode::Dst> acc(acc_storage, out_storage);
    acc.clear();

    for (uint32_t k = 0; k < kBlocks; ++k) {
        const bool finish = (k == kBlocks - 1);
        ComputeBlock a = noc_load<1>(a_storage, t0, k).wait();
        ComputeBlock b = noc_load<1>(b_storage, t1, k).wait();
        Block result = acc.accumulate(matmul(a, b), finish, [](auto mm) { return relu(mm); });
        if (finish) {
            noc_store<0>(std::move(result), t2, 0);
        }
    }
}

}  // namespace unified
}  // namespace tt

// Runs two examples that should be indistinguishable and diffs their traces. A
// different question from report()'s: not "is the protocol balanced" but "did
// these two spellings emit the same thing".
static bool report_same(const char* title, void (*lhs)(), void (*rhs)()) {
    lhs();
    std::vector<std::string> a = trace;
    trace.clear();
    rhs();
    std::vector<std::string> b = trace;
    trace.clear();

    printf("\n===== %s :: %s =====\n", TT_LABEL, title);
    if (a == b) {
        printf("  %d instructions, identical\n", (int)a.size());
        printf("  RESULT: spellings agree\n");
        return true;
    }
    size_t max = a.size() > b.size() ? a.size() : b.size();
    for (size_t i = 0; i < max; ++i) {
        const char* l = i < a.size() ? a[i].c_str() : "<end>";
        const char* r = i < b.size() ? b[i].c_str() : "<end>";
        if (i >= a.size() || i >= b.size() || a[i] != b[i]) {
            printf("    [%d] free  : %s\n", (int)i, l);
            printf("    [%d] method: %s\n", (int)i, r);
        }
    }
    printf("  RESULT: *** SPELLINGS DIVERGE ***\n");
    return false;
}

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
    tt::unified::example_matmul_single();
    ok &= report("matmul_single");
    tt::unified::example_matmul_acc();
    ok &= report("matmul_acc");
    tt::unified::example_reduce();
    ok &= report("reduce");
    tt::unified::example_peer_hop();
    ok &= report("peer_hop");
    ok &= report_same("syntax: free vs method", tt::unified::example_syntax_free, tt::unified::example_syntax_method);
    printf("\n%s: %s\n", TT_LABEL, ok ? "ALL BALANCED" : "FAILURES PRESENT");
    return ok ? 0 : 1;
}
