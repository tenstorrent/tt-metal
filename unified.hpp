#include <tuple>

// This header proposes a unified/single-threaded programming model
// built on top of the existing metal programming model.  Metal
// typically has 2 DM threads and 1 compute thread (which then is
// split into 3, but this header does not concern itself with compute
// thread splitting and treats compute as the abstraction that metal
// provides, this is just an extension).

namespace tt {
namespace unified {

class Tensor;

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

    Block store(const Fusion& fusion) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        cb_reserve for (int i = 0; i < num_tiles; ++i) {
            tile_regs_acquire();
            fusion.invoke(std::index_sequence_for<Blocks...>{}, i);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile();
            tile_regs_release();
        }
        cb_push
#endif
            return Block(cb_id, num_tiles);
    }

    int cb_id;
    int num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

template <typename... Fusions>
class Fusion {
public:
    explicit Fusion(int dst_index, Fusions... fusions) : dst_index(dst_index), fusions_(std::move(fusions)...) {}

    // Runs every compute fusion, in order.
    void run() { invoke(std::index_sequence_for<Fusions...>{}); }

    // Returns a *new* Fusion with `lambda` appended after this one's
    // fusions. The tuple size is fixed at compile time, so appending can't
    // mutate `*this` -- it produces a Fusion<Fusions..., Lambda>.
    template <typename Lambda>
    Fusion<Fusions..., Lambda> append(Lambda lambda) const {
        return appendImpl(std::index_sequence_for<Fusions...>{}, std::move(lambda));
    }

    // Returns a new Fusion with every fusion from `other` appended
    // after this one's, i.e. combines two Fusions into one.
    template <typename... OtherFusions>
    Fusion<Fusions..., OtherFusions...> append(const Fusion<OtherFusions...>& other) const {
        return appendOther(std::index_sequence_for<Fusions...>{}, std::index_sequence_for<OtherFusions...>{}, other);
    }

private:
    template <typename...>
    friend class Fusion;

    std::tuple<Fusions...> fusions_;
    int dst_index;

    template <std::size_t... Is>
    void invoke(std::index_sequence<Is...>) {
        // Comma-operator fold guarantees left-to-right evaluation order.
        // std::get<Is> is a compile-time index, so each call below binds
        // directly to that fusion's concrete type -- static dispatch.
        (std::get<Is>(fusions_)(), ...);
    }

    template <std::size_t... Is, typename Lambda>
    Fusion<Fusions..., Lambda> appendImpl(std::index_sequence<Is...>, Lambda lambda) const {
        return Fusion<Fusions..., Lambda>(std::get<Is>(fusions_)..., std::move(lambda));
    }

    template <std::size_t... Is, std::size_t... Js, typename... OtherFusions>
    Fusion<Fusions..., OtherFusions...> appendOther(
        std::index_sequence<Is...>, std::index_sequence<Js...>, const Fusion<OtherFusions...>& other) const {
        return Fusion<Fusions..., OtherFusions...>(std::get<Is>(fusions_)..., std::get<Js>(other.fusions_)...);
    }

protected:
    template <typename... FusionsA>
    int next_dst_idx(const Fusion<FusionsA...>& other) {
        return std::max(dst_index, other.dst_index) + 1;
    }
};

template <typename... Fusions>
struct NaryFusion : Fusion<Fusions...> {
    using Fusion<Fusions...>;

    template <typename... FusionsA, typename... FusionsB>
    NaryFusion<FusionsA...> add(const NaryFusion<FusionsB...>& a, int dst_idx) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        return NaryFusion(dst_idx, [=]() { sfpu_add(dst_idx_a, dst_idx_b, dst_idx_out); });
#endif
    }

    template <typename... FusionsA, typename... FusionsB>
    NaryFusion<FusionsA...> add(const NaryFusion<FusionsB...>& a) {
        return add(a, next_dst_idx(a));
    }

    template <typename... FusionsA, typename... FusionsB>
    NaryFusion<FusionsA...> operator+(const Fusion<FusionsA...>& other) {
        return add(other);
    }

    template <typename... FusionsA>
    NaryFusion<FusionsA...> operator+(const Block& other) {
        return add(other);
    }
};

// Deduction guide: lets callers write NaryFusion fb(lambda1, lambda2, ...)
// and have each lambda's exact closure type deduced automatically.
template <typename... Fusions>
NaryFusion(Fusions...) -> NaryFusion<Fusions...>;

struct MatmulFusion : Fusion<Fusions...> {
    using Fusion<Fusions...>;
};

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

class ComputeBlock {
public:
    ComputeBlock(Block block) : cb_id(block.cb_id), num_tiles(block.num_tiles) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        cb_wait
#endif
    }

    ~ComputeBlock(){
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        cb_pop
#endif
    }

    ComputeBlock(const ComputeBlock&) = delete;
    ComputeBlock& operator=(const ComputeBlock&) = delete;
    ComputeBlock(const ComputeBlock&&) = delete;
    ComputeBlock& operator=(const ComputeBlock&&) = delete;

    template <typename... Fusions>
    NaryFusion<Fusions...> ld(int dst_idx) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        return NaryFusion(dst_idx, [= this]() { copy_tile(this->cb_id, i, dst_idx); });
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
        noc_read;
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
        noc_write;
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
        noc_read;
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
        noc_write;
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
