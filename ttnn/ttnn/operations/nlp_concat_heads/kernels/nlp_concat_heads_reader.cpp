// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// nlp_concat_heads reader (regime `tile_relay`, D % 32 == 0): load_block.
//
// This core owns the contiguous output pages [start_page, start_page + core_tiles). Output page p is input
// tile (b, h, r, c) with
//     c = p mod Dt,  h = (p / Dt) mod H,  row = p / (Dt * H) = b * St + r
//     src = ((b * H + h) * St + r) * Dt + c
// The counters (c, h, r, b) are seeded by division once, at this hart's first page, and then advanced by
// increments: walking p upward advances c, wrapping into h, then r, then b.
//
// Ring protocol: implicit sync. One tagged NoC read per tile lands at the ring's cursor; the DM0 interrupt
// handler posts the credits, so there is no reserve / barrier / push here. Raw Noc API: no dataflow helper
// wraps an implicit-sync page read into a ring (design, Block Operation Realization).
//
// Thread plan: R reader harts on a STRIDED ring; hart t reads this core's pages t, t + R, ... The page index
// is carried across blocks, so the dealing is the same whatever block_tiles is.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {

// Position of one output page in the input, kept as counters so a step costs no division.
struct HeadWalk {
    uint32_t Dt, H, St;  // head-dim tiles, heads, token tile rows per image
    uint32_t b, h, r, c;

    void seed(uint32_t page) {
        c = page % Dt;
        const uint32_t q = page / Dt;
        h = q % H;
        const uint32_t row = q / H;
        r = row % St;
        b = row / St;
    }
    void step() {
        if (++c == Dt) {
            c = 0;
            if (++h == H) {
                h = 0;
                if (++r == St) {
                    r = 0;
                    ++b;
                }
            }
        }
    }
    uint32_t src_page() const { return ((b * H + h) * St + r) * Dt + c; }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t R = get_arg(args::reader_threads);
    const uint32_t core_tiles = get_arg(args::core_tiles);
    const uint32_t start_page = get_arg(args::start_page);
    const uint32_t block_tiles = get_arg(args::block_tiles);

    DataflowBuffer cb_relay_tiles(dfb::relay);
    Noc noc;
    const auto acc_in = TensorAccessor(tensor::input);

    HeadWalk walk{get_arg(args::head_dim_tiles), get_arg(args::heads), get_arg(args::token_rows), 0, 0, 0, 0};

    uint32_t i = get_my_thread_id();  // this hart's next page, as an offset into the core's range
    if (i < core_tiles) {
        walk.seed(start_page + i);
    }
    for (uint32_t block_start = 0; block_start < core_tiles; block_start += block_tiles) {
        // load_block: the pages of this block that fall to this hart.
        const uint32_t block_end = block_start + block_tiles < core_tiles ? block_start + block_tiles : core_tiles;
        for (; i < block_end; i += R) {
            noc.async_read<NocOptions::TXN_ID>(acc_in, cb_relay_tiles, {.page_id = walk.src_page()}, {});
            for (uint32_t s = 0; s < R; ++s) {
                walk.step();
            }
        }
    }
    // No barrier: firmware credits the last partial batch when the kernel returns. Hand the hart's sticky
    // tag register back.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
