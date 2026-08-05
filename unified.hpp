
namespace tt {
namespace unified {

class Tensor;

class Block {
public:
    Block(Block&&) = delete;
    Block(const Block&) = delete;
    Block& operator=(Block&&) = delete;
    Block& operator=(const Block&) = delete;

private:
    friend class Tensor;  // whichever type owns read()

    template <int thread>
    explicit Block(int) {
        if constexpr (thread == TT_DM_THREAD_ID) {
            reserve;
            copy;
            push;
        } else if constexpr (TT_COMPUTE_THREAD_ID) {
            cb_wait
        }
    }

    ~Block() {
        if constexpr (TT_COMPUTE_THREAD_ID) {
            cb_pop
        }
    }
};

struct Tensor {
    // BlockSpec

    template <int thread>
    Block read(int index) {
        return Block<thread>(index);
    }

    template <int thread>
    void write(Block block, int index) {}
};

void test() {
    for (int i = 0; i < 10; ++i) {
        Block lhs = t0.read<0>(i);
        Block rhs = t1.read<1>(i);

        Block out = lhs + rhs;

        t2.write<0>(out, i);
    }
}

}  // namespace unified
}  // namespace tt
