#include <algorithm>
#include <tuple>
#include <utility>

// This header proposes a unified/single-threaded programming model
// built on top of the existing metal programming model.  Metal
// typically has 2 DM threads and 1 compute thread (which then is
// split into 3, but this header does not concern itself with compute
// thread splitting and treats compute as the abstraction that metal
// provides, this is just an extension).

namespace tt {
namespace unified {

class Tensor;
struct Block;

template <template <typename...> class Derived, typename... Fusions>
class FusionBase;

template <typename... Fusions>
struct NaryFusion;

struct Coord {
    int y;
    int x;
};

struct Shape {
    int h;
    int w;
};

struct Mcast {
    Coord coord;
    Shape shape;
};

struct Storage {
    Storage(int cb_id, int num_tiles) : cb_id(cb_id), num_tiles(num_tiles) {}

    Storage(Storage&&) = delete;
    Storage(const Storage&) = delete;
    Storage& operator=(Storage&&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Defined out-of-line below: needs Block and Fusion complete.
    template <template <typename...> class D, typename... Fusions>
    Block store(const FusionBase<D, Fusions...>& fusion);

    int cb_id;
    int num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

// CRTP-style base: `Derived` is a template-template parameter so that
// append()/concat() rebuild the *derived* fusion type rather than decaying to
// the base. That is what keeps `x + y + z` chainable.
template <template <typename...> class Derived, typename... Fusions>
class FusionBase {
public:
    explicit FusionBase(int dst_index, Fusions... fusions) : fusions_(std::move(fusions)...), dst_index_(dst_index) {}

    // Runs every compute fusion, in order.
    void run() { invoke(std::index_sequence_for<Fusions...>{}); }

    // DST slot holding this fusion's result.
    int dst() const { return dst_index_; }

    // Append one op, with `new_dst` becoming the result slot.
    template <typename Lambda>
    Derived<Fusions..., Lambda> append(Lambda lambda, int new_dst) const {
        return appendLambda(std::index_sequence_for<Fusions...>{}, std::move(lambda), new_dst);
    }

    // Concatenate another fusion's ops after this one's.
    template <template <typename...> class D2, typename... Others>
    Derived<Fusions..., Others...> concat(const FusionBase<D2, Others...>& other, int new_dst) const {
        return concatImpl(std::index_sequence_for<Fusions...>{}, std::index_sequence_for<Others...>{}, other, new_dst);
    }

    template <template <typename...> class D2, typename... Others>
    int next_dst_idx(const FusionBase<D2, Others...>& other) const {
        return std::max(dst_index_, other.dst()) + 1;
    }

protected:
    template <template <typename...> class, typename...>
    friend class FusionBase;

    std::tuple<Fusions...> fusions_;
    int dst_index_;

    template <std::size_t... Is>
    void invoke(std::index_sequence<Is...>) {
        // Comma-operator fold guarantees left-to-right evaluation order.
        // std::get<Is> is a compile-time index, so each call below binds
        // directly to that fusion's concrete type -- static dispatch.
        (std::get<Is>(fusions_)(), ...);
    }

    template <std::size_t... Is, typename Lambda>
    Derived<Fusions..., Lambda> appendLambda(std::index_sequence<Is...>, Lambda lambda, int d) const {
        return Derived<Fusions..., Lambda>(d, std::get<Is>(fusions_)..., std::move(lambda));
    }

    template <std::size_t... Is, std::size_t... Js, template <typename...> class D2, typename... Others>
    Derived<Fusions..., Others...> concatImpl(
        std::index_sequence<Is...>, std::index_sequence<Js...>, const FusionBase<D2, Others...>& o, int d) const {
        return Derived<Fusions..., Others...>(d, std::get<Is>(fusions_)..., std::get<Js>(o.fusions_)...);
    }
};

template <typename... Fusions>
struct NaryFusion : FusionBase<NaryFusion, Fusions...> {
    using FusionBase<NaryFusion, Fusions...>::FusionBase;

    template <template <typename...> class D2, typename... Fs2>
    auto add(const FusionBase<D2, Fs2...>& other, int dst_out) const {
        const int a = this->dst(), b = other.dst();
        return this->concat(other, dst_out).append([a, b, dst_out]() { sfpu_add(a, b, dst_out); }, dst_out);
    }

    template <template <typename...> class D2, typename... Fs2>
    auto operator+(const FusionBase<D2, Fs2...>& other) const {
        return add(other, this->next_dst_idx(other));
    }
};

// C++17 does not consider inherited constructors for CTAD, so the guides are
// spelled out rather than relying on `using FusionBase::FusionBase`.
template <typename... Fusions>
NaryFusion(int, Fusions...) -> NaryFusion<Fusions...>;

template <typename... Fusions>
struct MatmulFusion : FusionBase<MatmulFusion, Fusions...> {
    using FusionBase<MatmulFusion, Fusions...>::FusionBase;
};

template <typename... Fusions>
MatmulFusion(int, Fusions...) -> MatmulFusion<Fusions...>;

struct Block {
    explicit Block(const Storage& storage) : cb_id(storage.cb_id), num_tiles(storage.num_tiles) {}
    Block(int cb_id, int num_tiles) : cb_id(cb_id), num_tiles(num_tiles) {}

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    Block(Block&& o) {
        cb_id = o.cb_id;
        num_tiles = o.num_tiles;
    }

    Block& operator=(Block&& o) {
        cb_id = o.cb_id;
        num_tiles = o.num_tiles;
        return *this;
    }

    int cb_id;
    int num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

template <template <typename...> class D, typename... Fusions>
Block Storage::store(const FusionBase<D, Fusions...>& fusion) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
    cb_reserve(cb_id);
    for (int i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        fusion.invoke(i);  // TODO: invoke is private and takes no tile index
        tile_regs_commit();
        tile_regs_wait();
        pack_tile();
        tile_regs_release();
    }
    cb_push(cb_id);
#endif
    return Block(cb_id, num_tiles);
}

class ComputeBlock {
public:
    ComputeBlock(Block block) : cb_id(block.cb_id), num_tiles(block.num_tiles) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        cb_wait(cb_id);
#endif
    }

    ~ComputeBlock(){
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        cb_pop(cb_id);
#endif
    }

    ComputeBlock(const ComputeBlock&) = delete;
    ComputeBlock& operator=(const ComputeBlock&) = delete;
    ComputeBlock(ComputeBlock&&) = delete;
    ComputeBlock& operator=(ComputeBlock&&) = delete;

    template <typename... Fusions>
    NaryFusion<Fusions...> ld(int dst_idx) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        return NaryFusion(dst_idx, [=]() { copy_tile(this->cb_id, i, dst_idx); });
#endif
    }

    NaryFusion add(const ComputeBlock& other) { return ld(0) + other; }

    NaryFusion operator+(const ComputeBlock& other) { return add(other); }

    MatmulFusion matmul(const ComputeBlock& other) { return ...; }

private:
    int cb_id;
    int num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

template <int thread>
Block noc_load(const Storage& storage, const Tensor& t, int idx) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_reserve(storage.cb_id);
        noc_read();
        cb_push(storage.cb_id);
    }
#endif
    return Block(storage);
}

template <int thread>
Block noc_load_mcast(const Storage& storage, Mcast mcast, const Tensor& t, int idx);

template <int thread>
void noc_store(Block block, const Tensor& t, int idx) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait(block.cb_id);
        noc_write();
        cb_pop(block.cb_id);
    }
#endif
}

template <int thread>
Block noc_read(const Storage& storage, Block block, Coord coord, int offset) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait(block.cb_id);
        cb_reserve(storage.cb_id);
        noc_read();
        cb_push(storage.cb_id);
        cb_pop(block.cb_id);
    }
#endif
    return Block(storage);
}

template <int thread>
Block noc_write(const Storage& storage, Block block, Coord coord, int offset) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait(block.cb_id);
        cb_reserve(storage.cb_id);
        noc_write();
        cb_push(storage.cb_id);
        cb_pop(block.cb_id);
    }
#endif
    return Block(storage);
}

//
// INPUT
//        DM    Compute
//   reserve <- *
//     write
//      push ->    wait
//                 read
//         * <-     pop
//
// OUTPUT
//        DM    Compute
//         * -> reserve
//                write
//      wait <-    push
//      read
//       pop -> *
//
// INTERMED
//        DM    Compute
//              reserve
//                write
//                 push
//                 wait
//                 read
//                  pop
//
void test() {
    Storage lhs_storage(0, 8);
    Storage rhs_storage(1, 8);
    Storage tmp_storage(2, 8);
    Storage out_storage(3, 8);

    for (int i = 0; i < 10; ++i) {
        ComputeBlock lhs = noc_load<0>(lhs_storage, t0, i);
        ComputeBlock rhs = noc_load<1>(rhs_storage, t1, i);

        ComputeBlock tmp = tmp_storage.store(lhs + rhs);

        Block result = out_storage.store(tmp + lhs);
        noc_store<0>(std::move(result), t2, i);
    }
}

void reduce() {
    Storage stage0_storage(0, 8);
    Storage stage1_storage(1, 8);
    Storage tmp_storage(2, 8);
    Storage out1_storage(3, 8);

    for (int i = 0; i < 10; ++i) {
        ComputeBlock s0 = noc_load<0>(stage0_storage, t0, i);

        Block tmp = tmp_storage.store(s0.sum());

        ComputeBlock s1 = noc_write<0>(stage1_storage, std::move(tmp), coord_0x0, offset);

        noc_store<0>(out1_storage.store(s1.sum()), t2, i);
    }
}

}  // namespace unified
}  // namespace tt
