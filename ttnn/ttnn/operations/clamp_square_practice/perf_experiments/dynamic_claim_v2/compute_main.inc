// ---- dynamic_claim_v2 compute main (appended to the op's compute kernel body by descriptor.py) ----
// Static share: one eltwise_chain over static_tiles (as the op). Pool blocks: the count is unknown at
// launch, so every TRISC (unpack, math, pack) polls the reader's SEM_PUB word = (count << 1) | done and runs
// one chain call per announced block (each TRISC independently; the per-block sequence is identical).
#include "hostdev/dev_msgs.h"
#include "internal/tt-1xx/cache.h"

constexpr uint32_t DC2_SEM_PUB = 2;

inline uint32_t dc2_local_sem_addr(uint32_t id) {
    tt_l1_ptr mailboxes_t* const mb = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
    const launch_msg_t* lm = &(mb->launch[mb->launch_msg_rd_ptr]);
    return lm->kernel_config.kernel_config_base[0] + lm->kernel_config.sem_offset[0] + id * L1_ALIGNMENT;
}

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    static_assert(
        block_tiles >= 1 && block_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT, "FullBlock block must fit DEST");

    const uint32_t static_tiles = get_arg_val<uint32_t>(0);
    const uint32_t min_bits = get_arg_val<uint32_t>(1);
    const uint32_t max_bits = get_arg_val<uint32_t>(2);

    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer output_tiles(cb_output_tiles);
    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);

    using namespace compute_kernel_lib;
    auto run = [&](uint32_t tiles) {
        eltwise_chain(
            IterationShape::tiles(tiles).block_size(block_tiles, BlockTailSync::FullBlock),
            CopyTile<
                input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
                Dst::D0>{input_tiles},
            ClampSquareFused{min_bits, max_bits},
            PackTile<output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{output_tiles});
    };
    MaybeDeviceZoneScope("compute_chain");
    if (static_tiles > 0) {
        run(static_tiles);
    }
    volatile tt_l1_ptr uint32_t* pub = (volatile tt_l1_ptr uint32_t*)dc2_local_sem_addr(DC2_SEM_PUB);
    uint32_t processed = 0;
    for (;;) {
        uint32_t w;
        do {
            invalidate_l1_cache();
            w = *pub;
        } while ((w >> 1) == processed && (w & 1u) == 0);
        const uint32_t n = w >> 1;
        if (n == processed) {
            break;  // done bit set and nothing new
        }
        for (; processed < n; ++processed) {
            run(block_tiles);
        }
    }
}
