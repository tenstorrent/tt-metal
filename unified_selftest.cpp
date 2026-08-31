// Compile + trace harness for the unified programming model.
//
//   tt/unified/expr.hpp    -- op-agnostic tree, DST allocator, method syntax
//   tt/unified/math.hpp    -- leaves, ops, fusion kinds, driver strategies
//   tt/unified/api.h       -- core API (Storage / Block / ComputeBlock / noc_*)
//   tt/unified/impl.hpp    -- its definitions
//
// Those headers are a design sketch: the DFB / NOC / Tensix intrinsics they call
// come from the metal kernel headers in a real build. This file supplies traced
// versions of just those, then runs the example kernels once per thread
// projection, checks that every dataflow buffer balances, and checks that the
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
static constexpr uint32_t kEntryBytes = 2048;
static constexpr uint32_t kFakeL1Bytes = 64 * 1024;

static std::vector<std::string> trace;
static void T(const std::string& s) { trace.push_back(s); }
static std::string n(int v) { return std::to_string(v); }
static std::string n(uint32_t v) { return std::to_string(v); }
static void T2(const std::string& s) { trace.push_back(s); }

// ---- DFB protocol -----------------------------------------------------------
inline void cb_reserve_back(uint32_t dfb, uint32_t p) { T("cb_reserve_back(dfb" + n(dfb) + "," + n(p) + ")"); }
inline void cb_push_back(uint32_t dfb, uint32_t p) { T("cb_push_back   (dfb" + n(dfb) + "," + n(p) + ")"); }
inline void cb_wait_front(uint32_t dfb, uint32_t p) { T("cb_wait_front  (dfb" + n(dfb) + "," + n(p) + ")"); }
inline void cb_pop_front(uint32_t dfb, uint32_t p) { T("cb_pop_front   (dfb" + n(dfb) + "," + n(p) + ")"); }

// ---- NOC -------------------------------------------------------------------
inline void noc_async_read() { T("noc_async_read()"); }
inline void noc_async_write() { T("noc_async_write()"); }

// ---- Tensix compute --------------------------------------------------------
// The model calls these as ckernel::* straight from fusion.hpp, so the harness
// fakes metal's namespace rather than a shim of its own.

// Metal generates APPROX into chlkc_descriptors.h from the ComputeConfigDescriptor's
// math_approx_mode, and declares it on the math TRISC only. TRISC_MATH is how the
// library asks whether the name is in scope, and this projection is the one that
// stands in for that thread -- it is the trace that carries the math ops. Overridable
// so the trace can be taken down both paths: -DAPPROX=true is what a
// math_approx_mode=true program compiles as.
#ifndef APPROX
#define APPROX false
#endif
#ifndef TRISC_MATH
#define TRISC_MATH 1
#endif

namespace ckernel {
inline void tile_regs_acquire() { T("  tile_regs_acquire"); }
inline void tile_regs_commit() { T("  tile_regs_commit"); }
inline void tile_regs_wait() { T("  tile_regs_wait"); }
inline void tile_regs_release() { T("  tile_regs_release"); }
inline void copy_tile(uint32_t dfb, uint32_t tile, uint32_t dst) {
    T("    copy_tile(dfb" + n(dfb) + ",tile=" + n(tile) + " -> dst" + n(dst) + ")");
}

enum class PoolType { SUM, AVG, MAX };
enum class ReduceDim { REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR };

inline void reconfig_data_format(uint32_t srca, uint32_t srcb) {
    T("    reconfig_data_format(dfb" + n(srca) + ",dfb" + n(srcb) + ")");
}
template <PoolType pool, ReduceDim dim>
inline void reduce_init(uint32_t icb, uint32_t iscaler, uint32_t ocb) {
    T("  reduce_init(dfb" + n(icb) + ",scaler=dfb" + n(iscaler) + " -> dfb" + n(ocb) + ")");
}
template <PoolType pool, ReduceDim dim>
inline void reduce_tile(uint32_t icb, uint32_t iscaler, uint32_t itile, uint32_t, uint32_t idst) {
    T("    reduce_tile(dfb" + n(icb) + ",tile=" + n(itile) + ",scaler=dfb" + n(iscaler) + " -> dst" + n(idst) + ")");
}
inline void reduce_uninit(uint32_t icb = 0) { T("  reduce_uninit(dfb" + n(icb) + ")"); }

inline void add_bcast_rows_init_short(uint32_t icb0, uint32_t icb1) {
    T("  add_bcast_rows_init(dfb" + n(icb0) + ",dfb" + n(icb1) + ")");
}
// metal's own enums, under its names, since the FPU binary calls are templated on them.
enum class EltwiseBinaryType { ELWMUL, ELWDIV, ELWADD, ELWSUB };
enum class EltwiseBinaryReuseDestType { NONE = 0, DEST_TO_SRCA = 1, DEST_TO_SRCB = 2 };

inline std::string op_name(EltwiseBinaryType t) {
    return t == EltwiseBinaryType::ELWADD   ? "add"
           : t == EltwiseBinaryType::ELWSUB ? "sub"
           : t == EltwiseBinaryType::ELWMUL ? "mul"
                                            : "div";
}

// The reuse forms: one operand from a dataflow buffer, the other already in DST. The
// trace spells out which side DST is on, because a chain that got that backwards would
// compute buffer - dst where it meant dst - buffer and still look plausible.
template <EltwiseBinaryType Type, EltwiseBinaryReuseDestType Dir>
inline void binary_dest_reuse_tiles_init(uint32_t dfb) {
    T("      " + op_name(Type) + "_reuse_init(dfb" + n(dfb) +
      (Dir == EltwiseBinaryReuseDestType::DEST_TO_SRCA ? ",dst=lhs)" : ",dst=rhs)"));
}

template <EltwiseBinaryType Type, EltwiseBinaryReuseDestType Dir>
inline void binary_dest_reuse_tiles(uint32_t dfb, uint32_t tile, uint32_t dst) {
    const bool dst_lhs = Dir == EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    T("    " + op_name(Type) + "_reuse(" + (dst_lhs ? "dst" + n(dst) + " " : "dfb" + n(dfb) + "[" + n(tile) + "] ") +
      (dst_lhs ? "dfb" + n(dfb) + "[" + n(tile) + "]" : "dst" + n(dst)) + " -> dst" + n(dst) + ")");
}

// FPU elementwise binaries: two dataflow buffers in, DST out, no copy_tile in sight.
// The trace shows the DFBs rather than DST slots for the operands, which is the whole
// difference from the SFPU forms above.
inline void add_tiles_init(uint32_t dfb0, uint32_t dfb1) {
    T("      add_tiles_init(dfb" + n(dfb0) + ",dfb" + n(dfb1) + ")");
}
inline void sub_tiles_init(uint32_t dfb0, uint32_t dfb1) {
    T("      sub_tiles_init(dfb" + n(dfb0) + ",dfb" + n(dfb1) + ")");
}
inline void mul_tiles_init(uint32_t dfb0, uint32_t dfb1) {
    T("      mul_tiles_init(dfb" + n(dfb0) + ",dfb" + n(dfb1) + ")");
}
inline void add_tiles(uint32_t dfb0, uint32_t dfb1, uint32_t t0, uint32_t t1, uint32_t d) {
    T("    add_tiles(dfb" + n(dfb0) + "[" + n(t0) + "],dfb" + n(dfb1) + "[" + n(t1) + "] -> dst" + n(d) + ")");
}
inline void sub_tiles(uint32_t dfb0, uint32_t dfb1, uint32_t t0, uint32_t t1, uint32_t d) {
    T("    sub_tiles(dfb" + n(dfb0) + "[" + n(t0) + "],dfb" + n(dfb1) + "[" + n(t1) + "] -> dst" + n(d) + ")");
}
inline void mul_tiles(uint32_t dfb0, uint32_t dfb1, uint32_t t0, uint32_t t1, uint32_t d) {
    T("    mul_tiles(dfb" + n(dfb0) + "[" + n(t0) + "],dfb" + n(dfb1) + "[" + n(t1) + "] -> dst" + n(d) + ")");
}

inline void add_tiles_bcast_rows(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    T("    add_bcast_rows(dfb" + n(icb0) + ",tile=" + n(itile0) + " + dfb" + n(icb1) + ",tile=" + n(itile1) +
      " -> dst" + n(idst) + ")");
}
// The nine (op, axis) broadcast pairs. The init_short names are metal's own and are
// NOT uniform -- add's scalar form omits `tiles_` where sub's and mul's include it.
inline void add_bcast_cols_init_short(uint32_t b, uint32_t v) {
    T("    add_bcast_cols_init(dfb" + n(b) + ",dfb" + n(v) + ")");
}
inline void add_tiles_bcast_cols(uint32_t b, uint32_t v, uint32_t bt, uint32_t vt, uint32_t d) {
    T("      add_bcast_cols(dfb" + n(b) + "[" + n(bt) + "],dfb" + n(v) + "[" + n(vt) + "] -> dst" + n(d) + ")");
}
inline void add_bcast_scalar_init_short(uint32_t b, uint32_t v) {
    T("    add_bcast_scalar_init(dfb" + n(b) + ",dfb" + n(v) + ")");
}
inline void add_tiles_bcast_scalar(uint32_t b, uint32_t v, uint32_t bt, uint32_t vt, uint32_t d) {
    T("      add_bcast_scalar(dfb" + n(b) + "[" + n(bt) + "],dfb" + n(v) + "[" + n(vt) + "] -> dst" + n(d) + ")");
}
inline void sub_bcast_rows_init_short(uint32_t b, uint32_t v) {
    T("    sub_bcast_rows_init(dfb" + n(b) + ",dfb" + n(v) + ")");
}
inline void sub_tiles_bcast_rows(uint32_t b, uint32_t v, uint32_t bt, uint32_t vt, uint32_t d) {
    T("      sub_bcast_rows(dfb" + n(b) + "[" + n(bt) + "],dfb" + n(v) + "[" + n(vt) + "] -> dst" + n(d) + ")");
}
inline void sub_bcast_cols_init_short(uint32_t b, uint32_t v) {
    T("    sub_bcast_cols_init(dfb" + n(b) + ",dfb" + n(v) + ")");
}
inline void sub_tiles_bcast_cols(uint32_t b, uint32_t v, uint32_t bt, uint32_t vt, uint32_t d) {
    T("      sub_bcast_cols(dfb" + n(b) + "[" + n(bt) + "],dfb" + n(v) + "[" + n(vt) + "] -> dst" + n(d) + ")");
}
inline void sub_tiles_bcast_scalar_init_short(uint32_t b, uint32_t v) {
    T("    sub_bcast_scalar_init(dfb" + n(b) + ",dfb" + n(v) + ")");
}
inline void sub_tiles_bcast_scalar(uint32_t b, uint32_t v, uint32_t bt, uint32_t vt, uint32_t d) {
    T("      sub_bcast_scalar(dfb" + n(b) + "[" + n(bt) + "],dfb" + n(v) + "[" + n(vt) + "] -> dst" + n(d) + ")");
}
inline void mul_bcast_rows_init_short(uint32_t b, uint32_t v) {
    T("    mul_bcast_rows_init(dfb" + n(b) + ",dfb" + n(v) + ")");
}
inline void mul_tiles_bcast_rows(uint32_t b, uint32_t v, uint32_t bt, uint32_t vt, uint32_t d) {
    T("      mul_bcast_rows(dfb" + n(b) + "[" + n(bt) + "],dfb" + n(v) + "[" + n(vt) + "] -> dst" + n(d) + ")");
}
inline void mul_bcast_cols_init_short(uint32_t b, uint32_t v) {
    T("    mul_bcast_cols_init(dfb" + n(b) + ",dfb" + n(v) + ")");
}
inline void mul_tiles_bcast_cols(uint32_t b, uint32_t v, uint32_t bt, uint32_t vt, uint32_t d) {
    T("      mul_bcast_cols(dfb" + n(b) + "[" + n(bt) + "],dfb" + n(v) + "[" + n(vt) + "] -> dst" + n(d) + ")");
}
inline void mul_tiles_bcast_scalar_init_short(uint32_t b, uint32_t v) {
    T("    mul_bcast_scalar_init(dfb" + n(b) + ",dfb" + n(v) + ")");
}
inline void mul_tiles_bcast_scalar(uint32_t b, uint32_t v, uint32_t bt, uint32_t vt, uint32_t d) {
    T("      mul_bcast_scalar(dfb" + n(b) + "[" + n(bt) + "],dfb" + n(v) + "[" + n(vt) + "] -> dst" + n(d) + ")");
}
inline void pack_tile(uint32_t dst, uint32_t dfb) { T("  pack_tile(dst" + n(dst) + " -> dfb" + n(dfb) + ")"); }
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
inline void binary_max_tile_init() {}
inline void binary_max_tile(uint32_t a, uint32_t b, uint32_t o) {
    T("      binary_max_tile(dst" + n(a) + ",dst" + n(b) + " -> dst" + n(o) + ")");
}
inline void div_binary_tile_init() {}
inline void div_binary_tile(uint32_t a, uint32_t b, uint32_t o) {
    T("      div_binary_tile(dst" + n(a) + ",dst" + n(b) + " -> dst" + n(o) + ")");
}
template <bool Approx = false>
inline void exp_tile_init() {}
// The approximation flag is traced, not dropped: it is a template argument the caller
// has to thread from APPROX, and a trace that omitted it could not tell the cheap
// path from the expensive one.
template <bool Approx = false>
inline void exp_tile(uint32_t o) {
    T("    exp_tile (dst" + n(o) + ", approx=" + std::string(Approx ? "1" : "0") + ")");
}
inline void relu_tile_init() {}
inline void relu_tile(uint32_t o) { T("    relu_tile(dst" + n(o) + ")"); }
inline void silu_tile_init() {}
inline void silu_tile(uint32_t o) { T("    silu_tile(dst" + n(o) + ")"); }
inline void recip_tile_init() {}
inline void recip_tile(uint32_t o) { T("    recip_tile(dst" + n(o) + ")"); }
inline void sqrt_tile_init() {}
inline void sqrt_tile(uint32_t o) { T("    sqrt_tile(dst" + n(o) + ")"); }
inline void rsqrt_tile_init() {}
inline void rsqrt_tile(uint32_t o) { T("    rsqrt_tile(dst" + n(o) + ")"); }
inline void copy_tile_to_dst_init_short_with_dt(uint32_t old_dfb, uint32_t new_dfb) {
    T("  copy_tile_to_dst_init_short_with_dt(dfb" + n(old_dfb) + " -> dfb" + n(new_dfb) + ")");
}
inline void copy_block(uint32_t dfb, uint32_t start_tile, uint32_t start_dst, uint32_t n_tiles) {
    T("  copy_block(dfb" + n(dfb) + "[" + n(start_tile) + "] -> dst" + n(start_dst) + ".." +
      n(start_dst + n_tiles - 1) + ")  [reload]");
}
inline void reconfig_data_format_srca(uint32_t old_dfb, uint32_t new_dfb) {
    T("  reconfig_data_format_srca(dfb" + n(old_dfb) + " -> dfb" + n(new_dfb) + ")");
}
// One-argument form: unconditional, needs no previous operand. What an SFPU leaf uses.
inline void reconfig_data_format_srcb(uint32_t new_dfb) { T("        reconfig_srcb(dfb" + n(new_dfb) + ")"); }
inline void reconfig_data_format_srca(uint32_t new_dfb) { T("      reconfig_srca(dfb" + n(new_dfb) + ")"); }
inline void copy_tile_to_dst_init_short(uint32_t dfb) { T("      copy_init(dfb" + n(dfb) + ")"); }
inline void init_sfpu(uint32_t icb, uint32_t ocb) { T("  init_sfpu(dfb" + n(icb) + " -> dfb" + n(ocb) + ")"); }
// pack_to's two forms: unconditional on the first pass, old -> new after that.
inline void pack_reconfig_data_format(uint32_t new_dfb) { T("  pack_reconfig(dfb" + n(new_dfb) + ")"); }
inline void pack_reconfig_data_format(uint32_t old_dfb, uint32_t new_dfb) {
    T("  pack_reconfig(dfb" + n(old_dfb) + " -> dfb" + n(new_dfb) + ")");
}
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
inline uint32_t dfb_entry_bytes(uint32_t) { return kEntryBytes; }

// Metal 2.0's buffer handle. Every method traces EXACTLY what the free function it
// replaced traced, which is the point: the balance checker below parses those strings,
// and a trace recorded before the port has to keep comparing equal to one recorded
// after. The port is supposed to be a change of spelling, and this is where that claim
// is actually tested.
class DataflowBuffer {
public:
    explicit DataflowBuffer(uint16_t dfb) : dfb_(dfb) {}
    void reserve_back(uint32_t p) const { cb_reserve_back(dfb_, p); }
    void push_back(uint32_t p) const { cb_push_back(dfb_, p); }
    void wait_front(uint32_t p) const { cb_wait_front(dfb_, p); }
    void pop_front(uint32_t p) const { cb_pop_front(dfb_, p); }
    uint32_t get_write_ptr() const { return ::get_write_ptr(dfb_); }
    uint32_t get_read_ptr() const { return ::get_read_ptr(dfb_); }
    uint32_t get_entry_size() const { return kEntryBytes; }
    // What the HOST configured. This harness has no host, so the number is a stand-in
    // large enough never to be the binding constraint -- which does make Storage's
    // capacity assert (hazard A1/A2) VACUOUS here. That check belongs to the device
    // suites, which have a real host to disagree with.
    uint32_t get_total_num_entries() const { return 1u << 16; }

private:
    uint16_t dfb_;
};
inline DataflowBuffer buffer(uint32_t dfb) { return DataflowBuffer(static_cast<uint16_t>(dfb)); }
inline uint32_t dfb_num_entries(uint32_t dfb) { return buffer(dfb).get_total_num_entries(); }

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
    // D19's check compares this against dfb_entry_bytes; agreeing here keeps it quiet,
    // which is right -- the mismatch it catches is a HOST configuration error.
    uint32_t get_aligned_page_size() const { return kEntryBytes; }
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
    explicit Noc(uint8_t noc_id) : noc_id_(noc_id) {}
    uint8_t get_noc_id() const { return noc_id_; }
    // Traced as the free functions were, for the same reason DataflowBuffer is: the
    // handle carries which NOC it used, but on this harness there is one fake NOC, so
    // printing the id would make every pre-port trace differ for no real difference.
    void async_read_barrier() const { T2("noc_async_read_barrier()"); }
    void async_writes_flushed() const { T2("noc_async_writes_flushed()"); }
    void async_write_barrier() const { T2("noc_async_write_barrier()"); }
    void async_atomic_barrier() const { T2("noc_async_atomic_barrier()"); }

private:
    uint8_t noc_id_ = 0;
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
    T("  hw_startup(dfb" + n(icb0) + ",dfb" + n(icb1) + " -> dfb" + n(ocb) +
      (order == SrcOrder::Reverse ? ", SrcOrder::Reverse)" : ")"));
}
inline void matmul_block_init(uint32_t in0, uint32_t in1, uint32_t transpose, uint32_t ct, uint32_t rt, uint32_t kt) {
    (void)transpose;
    T("  matmul_block_init(dfb" + n(in0) + ",dfb" + n(in1) + " ct=" + n(ct) + " rt=" + n(rt) + " kt=" + n(kt) + ")");
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
    T("    matmul_block(dfb" + n(in0) + "[" + n(in0_tile) + "],dfb" + n(in1) + "[" + n(in1_tile) + "] -> dst" +
      n(idst) + ".." + n(idst + rt * ct - 1) + ")");
}
inline void pack_reconfig_l1_acc(uint32_t en) { T("  pack_reconfig_l1_acc(" + n(en) + ")"); }
inline void pack_block(uint32_t dst, uint32_t dfb, uint32_t count) {
    T("  pack_block(dst" + n(dst) + ".." + n(dst + count - 1) + " -> dfb" + n(dfb) + ")");
}
}  // namespace ckernel

// Metal's assert.h defines ASSERT unconditionally -- a no-op when asserts are off --
// so code may use it without guarding. Match that, or every unguarded ASSERT in the
// library is an undeclared identifier here and only here.
#if defined(ASSERT_ENABLED) && ASSERT_ENABLED
#include <cassert>
#define ASSERT(x, ...) assert(x)
#else
#define ASSERT(x, ...) ((void)sizeof(!(x)))
#endif

// Device attributes and build defines the metal headers would supply.
#define FORCE_INLINE inline
// The NOC mode metal compiles into every kernel. Dedicated is the default everywhere
// and the only mode with device coverage; impl.hpp reads it to decide whether a write
// release owes a full barrier (hazard 30). Tracing the dedicated projection is the
// right choice here -- it is what every shipping kernel builds as -- and it means this
// harness does NOT exercise the dynamic-NOC branch.
enum : uint8_t { DM_DEDICATED_NOC = 0, DM_DYNAMIC_NOC = 1 };
#define NOC_MODE DM_DEDICATED_NOC

#define TT_UNIFIED_CUSTOM_BINDING 1
#include <tt/unified/core>

namespace tt {
namespace unified {

// ===========================================================================
//  Example kernels
//
//  Each is written as if single-threaded. The same source is compiled once per
//  baby RISC-V thread, and each statement lowers to that thread's half of the
//  dataflow-buffer protocol.
// ===========================================================================

// INPUT + INTERMED + OUTPUT: two DRAM loads, an SFPU add into an intermediate,
// a second add, then a DRAM store.
// The two coordinate orderings must actually disagree, or naming them is decoration.
// Compile-time facts, so static_asserts are the whole test.
namespace coord_checks {
static_assert(LogicalCoord::xy(1, 2).x == 1 && LogicalCoord::xy(1, 2).y == 2, "xy takes x first");
static_assert(LogicalCoord::yx(1, 2).y == 1 && LogicalCoord::yx(1, 2).x == 2, "yx takes y first");
static_assert(LogicalCoord::xy(1, 2) != LogicalCoord::yx(1, 2), "the orderings differ");
static_assert(LogicalCoord::xy(1, 2) == LogicalCoord::yx(2, 1), "and mirror each other");
static_assert(PhysicalCoord::xy(3, 4).x == 3 && PhysicalCoord::yx(3, 4).y == 3, "same for physical");

// Extent carries the same hazard -- h before w here, w before h in metal's grid sizes --
// so it gets the same treatment and the same check.
static_assert(Extent::hw(1, 8).h == 1 && Extent::hw(1, 8).w == 8, "hw takes h first");
static_assert(Extent::wh(1, 8).w == 1 && Extent::wh(1, 8).h == 8, "wh takes w first");
static_assert(Extent::hw(1, 8) != Extent::wh(1, 8), "the orderings differ");
static_assert(Extent::hw(1, 8) == Extent::wh(8, 1), "and mirror each other");
}  // namespace coord_checks

// Shape::dim, both directions. These are compile-time facts, so a static_assert is the
// whole test -- nothing runs, and a wrong answer never reaches a device.
namespace shape_checks {
using R3 = Shape<2, 3, 4>;
using R1 = Shape<7>;

// Forward indexing, unchanged.
static_assert(R3::dim(0) == 2 && R3::dim(1) == 3 && R3::dim(2) == 4, "forward indices");

// Backward, which is the point: the last dimension is dim(-1) whatever the rank, so code
// that means "columns" can say so without knowing how many leading axes there are.
static_assert(R3::dim(-1) == 4 && R3::dim(-2) == 3 && R3::dim(-3) == 2, "negative indices");
static_assert(R1::dim(-1) == 7 && R1::dim(0) == 7, "rank 1 has one dimension either way");

// And that the two namings agree, which is what would break if the wrap arithmetic were
// off by one in either direction.
static_assert(R3::dim(-1) == R3::cols && R3::dim(-2) == R3::rows, "dim agrees with cols/rows");
static_assert(R1::dim(-1) == R1::cols, "rank 1 columns");
static_assert(R3::dim(R3::rank - 1) == R3::dim(-1), "the two ends meet");
}  // namespace shape_checks

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

// Every branch of the FPU eltwise predicate, with one shape per rule so a failure says
// which rule broke. The trace is the verification: each lands on different calls.
//
//   a + b - c        seed, then a chain link with DST on the LEFT
//   a - (b + c)      seed, then a chain link with DST on the RIGHT -- the direction
//                    that decides whether a subtraction comes out backwards
//   (a - b).exp()    FPU seed with an SFPU unary applied in place on top
//   max_(a, b)       no FPU form for max: the OP rule sends the tree to the SFPU
//   (a + b) - (c+a)  two non-leaf children: the SHAPE rule does, even though every op
//                    here has an FPU form
//   a * b            FPU like the rest, and the one op where that is a real trade:
//                    -DTT_UNIFIED_SFPU_MUL takes the more accurate form back
//
// The chains are built from add and sub on purpose: those two are unconditional, so the
// coverage does not move if the multiply's default is ever taken back.
void example_fpu_eltwise() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t3 = TensorAccessor(FakeArgs{3}, 0);
    using Row2 = Shape<1, 2>;
    Storage<Row2> a_storage(0);
    Storage<Row2> b_storage(1);
    Storage<Row2> c_storage(2);
    Storage<Row2> out_storage(3);
    Storage<Row2> scratch(4);
    // The store-out buffer is its own, not scratch: a block handed to noc_store is
    // consumed by the writer thread, so on this projection it is pushed and never
    // waited on. Mixing that with blocks this thread does read would leave the buffer
    // with more pushes than waits, which is exactly what the balance check objects to.
    Storage<Row2> sink(5);

    ComputeBlock a = noc_load<1>(a_storage, t0, 0).wait();
    ComputeBlock b = noc_load<1>(b_storage, t0, 1).wait();
    ComputeBlock c = noc_load<1>(c_storage, t0, 2).wait();

    T("-- a + b - c: seed, then reuse with dst on the left");
    ComputeBlock chain = scratch.store(a + b - c);

    T("-- a - (b + c): seed, then reuse with dst on the RIGHT");
    ComputeBlock flipped = out_storage.store(a - (b + c));

    T("-- (a - b).exp(): FPU seed, SFPU unary on top");
    ComputeBlock fused = scratch.store((a - b).exp());

    T("-- max_(a, b): no FPU max, so this falls back to the SFPU tree");
    ComputeBlock fell_back = out_storage.store(max_(a, b));

    T("-- (a + b) - (c + a): two non-leaf children, so the shape rule falls back");
    ComputeBlock both_sides = scratch.store((a + b) - (c + a));

    T("-- a * b: FPU by default; TT_UNIFIED_SFPU_MUL takes the accurate form back");
    Block both = sink.store(a * b);
    noc_store<0>(std::move(both), t3, 0);
    (void)sizeof(chain);
    (void)sizeof(flipped);
    (void)sizeof(fused);
    (void)sizeof(fell_back);
    (void)sizeof(both_sides);
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
    noc_store<0>(out_storage.store(reduce_sum<ReduceAxis::Rows>(a, scaler)), t2, 0);
}

// Broadcast, one example per axis, and the softmax shape of the thing: a reduction
// feeding the broadcast that undoes it. The axis appears in both and means the same in
// both, so the shapes agree by construction -- writing Rows in one place would make the
// vector Shape<1, wt> and the reduction's result no longer fit its buffer.
void example_bcast() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    using In = Shape<2, 3>;                    // non-square on purpose
    using Col = reduce_shape<In, Axis::Cols>;  // Shape<2, 1>
    using Row = reduce_shape<In, Axis::Rows>;  // Shape<1, 3>

    Storage<In> x_storage(0), e_storage(2), out_storage(4);
    Storage<Col> m_storage(5);
    Storage<Row> r_storage(6);
    Storage<Shape<1, 1>> one_storage(3);

    ComputeBlock one = fill_reduce_scaler<1>(one_storage);
    ComputeBlock x = noc_load<1>(x_storage, t0, 0).wait();

    {  // Cols: a reduction and the broadcast that expands it again, with a fused exp
        ComputeBlock m = m_storage.store(reduce_max<Axis::Cols>(x, one));
        noc_store<0>(e_storage.store((x - bcast<Axis::Cols>(m)).exp()), t2, 0);
    }
    {  // Rows
        ComputeBlock r = noc_load<1>(r_storage, t0, 0).wait();
        noc_store<0>(out_storage.store(x * bcast<Axis::Rows>(r)), t2, 0);
    }
    {  // Both -- a scalar
        ComputeBlock sc = noc_load<1>(one_storage, t0, 0).wait();
        noc_store<0>(out_storage.store(x + bcast<Axis::Both>(sc)), t2, 0);
    }
}

// State carried across a loop, which is what RetainedBlock is for: iteration 0 writes a
// running value and iteration 1 reads it. The slot is what makes the lifetime visible --
// taking the Block as a ComputeBlock instead would pop the state at the end of iteration 0,
// and the push/wait rule in report() is what catches that.
void example_retained_state() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    using Row2 = Shape<1, 2>;
    Storage<Row2> in(0), state(1), out(3);

    RetainedBlock<Row2> carried;
    for (uint32_t j = 0; j < 2; ++j) {
        ComputeBlock x = noc_load<1>(in, t0, j).wait();
        if (j == 0) {
            carried = state.store(x.exp());
        } else {
            ComputeBlock<Row2> prev = carried.release();
            noc_store<0>(out.store(prev + x), t2, j);
        }
    }
}

// Core-to-core hop, exercising NocAsyncCopyTx. The peer handshake is still not
// right (see the TODO in unified.hpp) -- this only checks that the local half of
// the protocol balances and that the handle's two halves fire.
void example_peer_hop() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    LogicalCoord peer = LogicalCoord::yx(0, 0);
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
    // One buffer set per SHAPE the probe needs, now that Storage::store checks that the
    // destination is exactly what the expression produces. The old single set was
    // silently inconsistent -- a 2x2 matmul's Shape<2,2> output stored into a
    // Storage<Shape<1,2>> -- which is what that assert caught.
    using Row2 = Shape<1, 2>;  // eltwise operands and result
    using Sq2 = Shape<2, 2>;   // matmul operands and output block
    using Col2 = Shape<2, 1>;  // reduce input
    using One = Shape<1, 1>;   // scaler, and a rows-collapse result
    Storage<Row2> in0(0), in1(1), out(2);
    Storage<Sq2> mm_a(5), mm_b(6), mm_out(7);
    Storage<Col2> red_in(8);
    Storage<One> scaler(3), red_out(4);

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
        noc_store<0>(red_out.store(rsqrt(sqrt_(recip(exp_(relu(reduce_sum<ReduceAxis::Rows>(a, sc))))))), t2, 0);
    }
}

void example_syntax_method() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);
    // One buffer set per SHAPE the probe needs, now that Storage::store checks that the
    // destination is exactly what the expression produces. The old single set was
    // silently inconsistent -- a 2x2 matmul's Shape<2,2> output stored into a
    // Storage<Shape<1,2>> -- which is what that assert caught.
    using Row2 = Shape<1, 2>;  // eltwise operands and result
    using Sq2 = Shape<2, 2>;   // matmul operands and output block
    using Col2 = Shape<2, 1>;  // reduce input
    using One = Shape<1, 1>;   // scaler, and a rows-collapse result
    Storage<Row2> in0(0), in1(1), out(2);
    Storage<Sq2> mm_a(5), mm_b(6), mm_out(7);
    Storage<Col2> red_in(8);
    Storage<One> scaler(3), red_out(4);

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
        noc_store<0>(red_out.store(reduce_sum<ReduceAxis::Rows>(a, sc).relu().exp().recip().sqrt().rsqrt()), t2, 0);
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

// matmul(a, b).add(m): a whole block added to the product while it is still in DST.
//
// The point is the trace, and specifically what is NOT in it. There is no second pass --
// no pack of the product, no wait, no re-read of it as an operand -- just an add_reuse per
// output tile between the matmul and the pack, taking the addend from its buffer and DST
// as the other operand. Then matmul_block_init again, because the reuse op reprogrammed
// the math unit and a later band would otherwise run a matmul against eltwise state.
void example_matmul_add() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t1 = TensorAccessor(FakeArgs{1}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);

    using Sq2 = Shape<2, 2>;
    Storage<Sq2> a_storage(0);
    Storage<Sq2> b_storage(1);
    Storage<Sq2> m_storage(2);
    Storage<Sq2> out_storage(3);

    ComputeBlock a = noc_load<1>(a_storage, t0, 0).wait();
    ComputeBlock b = noc_load<1>(b_storage, t1, 0).wait();
    ComputeBlock m = noc_load<1>(m_storage, t1, 1).wait();
    noc_store<0>(out_storage.store(matmul(a, b).add(m)), t2, 0);

    // .add(m).relu(): the chained form, and the reason it is here. Every unary builder
    // rebuilds the node with one more link, and each has to carry the addend across --
    // an omission is not a compile error, it silently drops the fused add and yields
    // relu(A@B) instead of relu(A@B + m). Five builders got that wrong until this case
    // existed, so what the trace has to show is add_reuse BEFORE relu, not relu alone.
    Storage<Sq2> chained_storage(4);
    Block chained = chained_storage.store(relu(matmul(a, b).add(m)));
    noc_store<0>(std::move(chained), t2, 1);
}

// A matmul whose output block does not fit one acquire, so the strategy walks it in row
// bands. Shape<4,2> @ Shape<2,8> gives a 4x8 output: 32 tiles against a budget of 8, so
// four bands of one row each.
//
// The trace is the verification, and there are two things in it that a plain "it ran"
// would not catch. Operand A's tile index has to advance by kt_dim per BAND -- band r
// starts at r*kt_dim -- while operand B restarts at 0 every band, because the unpacker
// walks A as row*kt_dim + k and B as k*ct_dim + col. And the four packs have to be four
// separate pack_blocks inside ONE reserve/push pair, since pack_block advances the write
// pointer and only cb_push_back rewinds it: a push between bands would stack all four on
// top of each other.
//
// Both were wrong in the first draft and both were caught on device by the error going to
// 2.5 and 3.5 -- but on device they are one number, and here they are visible.
void example_matmul_banded() {
    auto t0 = TensorAccessor(FakeArgs{0}, 0);
    auto t1 = TensorAccessor(FakeArgs{1}, 0);
    auto t2 = TensorAccessor(FakeArgs{2}, 0);

    using A = Shape<4, 2>;
    using B = Shape<2, 8>;
    Storage<A> a_storage(0);
    Storage<B> b_storage(1);
    Storage<Shape<4, 8>> out_storage(3);

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
    Storage<Sq2> acc_storage(24);  // running total -- a different DFB from out
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
// Both spellings have to START from the same memoized state or they cannot emit the
// same trace, however identical the code is. pack_to() remembers which buffer the
// packer is programmed for -- per RISC, and deliberately so (see math.hpp) -- and that
// memory outlives one example, so whichever spelling runs second sees the other's
// leftovers and prints a different transition. It is the only such state in the
// library; `static` in math.hpp, impl.hpp and expr.hpp finds exactly this one.
//
// Driving it to a fixed buffer before each side, untraced, makes the comparison about
// the two spellings rather than about what ran before them.
static constexpr uint32_t kPackProbeReset = 31;
static void reset_memoized_state() {
    tt::unified::pack_to(kPackProbeReset);
    trace.clear();
}

static bool report_same(const char* title, void (*lhs)(), void (*rhs)()) {
    reset_memoized_state();
    lhs();
    std::vector<std::string> a = trace;
    trace.clear();
    reset_memoized_state();
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
    // Every dataflow buffer the model uses, not 0..4: the output DFB is 16 and the
    // accumulator's is 24. And the tag needs the COMMA, because every trace line writes
    // "dfb3,2" -- the entry count follows the id. Matching "dfb3)" matched nothing at all,
    // which made this whole check vacuous for as long as it has existed.
    for (int dfb = 0; dfb < 32; ++dfb) {
        int res = 0, push = 0, wait = 0, pop = 0;
        std::string tag = "(dfb" + n(dfb) + ",";
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

            // And for a buffer this thread both PRODUCES and CONSUMES, every pushed block
            // must be waited on exactly once. reserve==push with wait==pop says nothing about
            // whether the two sides agree: a state buffer written once and read twice
            // balances on both counts and still waits for pages that are gone. Only checked
            // when both halves are on this thread -- for a cross-thread buffer each thread
            // legitimately sees one half.
            const bool intra_thread = res > 0 && wait > 0;
            const bool matched = !intra_thread || push == wait;
            ok = ok && matched;

            bad |= !ok;
            printf(
                "  [dfb%d] reserve=%d push=%d | wait=%d pop=%d -> %s\n",
                dfb,
                res,
                push,
                wait,
                pop,
                ok ? "balanced" : (matched ? "*** IMBALANCED ***" : "*** PUSH/WAIT MISMATCH ***"));
        }
    }
    printf("  RESULT: %s\n", bad ? "*** PROTOCOL IMBALANCE ***" : "protocol balanced");
    trace.clear();
    return !bad;
}

// Guarded so a scratch file can #include this whole harness -- all the ckernel and
// dataflow stubs, the trace, report() -- and supply its own main. See tmp.cpp.
#ifndef TT_UNIFIED_NO_MAIN
int main() {
    bool ok = true;
    tt::unified::example_eltwise();
    ok &= report("eltwise");
    tt::unified::example_fpu_eltwise();
    ok &= report("fpu_eltwise");
    tt::unified::example_unary();
    ok &= report("unary");
    tt::unified::example_matmul_single();
    ok &= report("matmul_single");
    tt::unified::example_matmul_add();
    ok &= report("matmul_add");
    tt::unified::example_matmul_banded();
    ok &= report("matmul_banded");
    tt::unified::example_matmul_acc();
    ok &= report("matmul_acc");
    tt::unified::example_reduce();
    ok &= report("reduce");
    tt::unified::example_bcast();
    ok &= report("bcast");
    tt::unified::example_retained_state();
    ok &= report("retained state");
    tt::unified::example_peer_hop();
    ok &= report("peer_hop");
    ok &= report_same("syntax: free vs method", tt::unified::example_syntax_free, tt::unified::example_syntax_method);
    printf("\n%s: %s\n", TT_LABEL, ok ? "ALL BALANCED" : "FAILURES PRESENT");
    return ok ? 0 : 1;
}
#endif  // TT_UNIFIED_NO_MAIN
