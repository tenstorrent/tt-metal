// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// nlp_create_qkv_heads writer, block operation `relocate_block` (op_design.md, regime tile_relocate).
//
// The ring `cb_input_tiles` carries this cluster's input pages in page order (hart w sees entries w, w+W, ...).
// Input page p = ((b * St + r) * sections + head) * hd_t + c is output page
//     ((b * heads[sec] + h) * St + r) * hd_t + c   of tensor sec (q, k or v),   head = first_head(sec) + h,
// written whole by one tagged NoC write straight from the ring slot (implicit sync: the DM0 interrupt handler
// acks the entry once sent; firmware waits for the writes to land at kernel exit). The walk position
// (sec, h, b, r, c) of the cluster's first page comes from the host; the device only advances it by carries.
//
// No helper: the kernel library has no dataflow helper for a tagged ring-to-accessor write (op_design.md,
// "Helpers considered and rejected").

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t W = get_arg(args::writer_threads);
constexpr uint32_t blocks_per_core = get_arg(args::blocks_per_core);
constexpr uint32_t row_tiles = get_arg(args::row_tiles);            // St
constexpr uint32_t head_dim_tiles = get_arg(args::head_dim_tiles);  // hd_t
constexpr uint32_t q_heads = get_arg(args::q_heads);
constexpr uint32_t kv_heads = get_arg(args::kv_heads);
constexpr uint32_t num_sections = 3;  // q, k, v

struct Walk {
    uint32_t sec, h, b, r, c;

    uint32_t heads() const { return sec == 0 ? q_heads : kv_heads; }

    // Output page of the current input page, in tensor `sec`.
    uint32_t dst_page() const { return ((b * heads() + h) * row_tiles + r) * head_dim_tiles + c; }

    // Next input page: c -> h -> sec -> r -> b carries.
    void advance() {
        if (++c < head_dim_tiles) {
            return;
        }
        c = 0;
        if (++h < heads()) {
            return;
        }
        h = 0;
        if (++sec < num_sections) {
            return;
        }
        sec = 0;
        if (++r < row_tiles) {
            return;
        }
        r = 0;
        ++b;
    }

    void advance(uint32_t n) {
        for (uint32_t k = 0; k < n; ++k) {
            advance();
        }
    }
};

}  // namespace

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    Walk walk{
        get_arg(args::start_sec),
        get_arg(args::start_h),
        get_arg(args::start_b),
        get_arg(args::start_r),
        get_arg(args::start_c)};

    DataflowBuffer cb_input_tiles(dfb::cb_input_tiles);
    Noc noc;
    const auto acc_q = TensorAccessor(tensor::q);
    const auto acc_k = TensorAccessor(tensor::k);
    const auto acc_v = TensorAccessor(tensor::v);

    const uint32_t block_pages = (num_pages + blocks_per_core - 1) / blocks_per_core;
    uint32_t i = get_my_thread_id();  // this hart's next page, local to the cluster's range
    walk.advance(i);
    for (uint32_t block_end = block_pages; i < num_pages; block_end += block_pages) {
        // relocate_block: this hart's share of pages [block_end - block_pages, block_end)
        const uint32_t end = block_end < num_pages ? block_end : num_pages;
        for (; i < end; i += W) {
            const uint32_t dst = walk.dst_page();
            if (walk.sec == 0) {
                noc.async_write<NocOptions::TXN_ID>(cb_input_tiles, acc_q, {}, {.page_id = dst});
            } else if (walk.sec == 1) {
                noc.async_write<NocOptions::TXN_ID>(cb_input_tiles, acc_k, {}, {.page_id = dst});
            } else {
                noc.async_write<NocOptions::TXN_ID>(cb_input_tiles, acc_v, {}, {.page_id = dst});
            }
            walk.advance(W);
        }
    }
    // Firmware waits for the tagged writes when the kernel returns. Hand the hart's sticky tag back.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
