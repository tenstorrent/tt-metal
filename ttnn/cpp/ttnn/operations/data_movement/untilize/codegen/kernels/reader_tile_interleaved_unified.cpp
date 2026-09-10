// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilize-local interleaved tile reader (identity / sequential pages only).
// Full unified reader with all sequencers lives in
// data_movement/common/kernels/codegen/ (see PR #52806).
//
// CT args:
//   Named:      seq_id (must be SEQ_IDENTITY), cb_id, batch, src_page_pitch
//   Positional: TensorAccessorArgs (starts at index 0)
//
// RT args: src_addr, num_pages, start_id
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "sequencers.h"

struct ArgsBase {
    uint32_t src_addr;
    uint32_t num_pages;
    uint32_t start_id;
};

template <typename Accessor, typename State, typename NextFn>
FORCE_INLINE void read_pages(
    uint32_t cb_id,
    uint32_t BATCH,
    uint32_t cb_page_size,
    uint32_t source_read_size,
    const Accessor& accessor,
    uint32_t num_pages,
    State& state,
    NextFn next_fn) {
    Noc noc;
    CircularBuffer cb(cb_id);

    uint32_t pages_left = num_pages;
    while (pages_left > 0) {
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        cb.reserve_back(batch);
        uint32_t l1_offset = 0;
        for (uint32_t t = 0; t < batch; t++) {
            const uint32_t source_page = next_fn(state);
            noc.async_read(
                accessor,
                cb,
                source_read_size,
                {.page_id = source_page, .offset_bytes = 0},
                {.offset_bytes = l1_offset});
            l1_offset += cb_page_size;
        }
        noc.async_read_barrier();
        cb.push_back(batch);
        pages_left -= batch;
    }
}

void kernel_main() {
    constexpr uint32_t SEQ_ID = get_named_compile_time_arg_val("seq_id");
    static_assert(SEQ_ID == SEQ_IDENTITY, "untilize reader supports SEQ_IDENTITY only");

    constexpr uint32_t cb_id = get_named_compile_time_arg_val("cb_id");
    constexpr uint32_t BATCH = get_named_compile_time_arg_val("batch");

    constexpr auto src_args = TensorAccessorArgs<0>();

    const auto* base = reinterpret_cast<const ArgsBase*>(get_arg_addr(0));
    constexpr uint32_t source_page_size_override = get_named_compile_time_arg_val("src_page_pitch");
    const uint32_t source_page_size =
        source_page_size_override != 0 ? source_page_size_override : src_args.get_aligned_page_size();
    const uint32_t cb_page_size = get_local_cb_interface(cb_id).fifo_page_size << cb_addr_shift;
    const uint32_t source_read_size = source_page_size < cb_page_size ? source_page_size : cb_page_size;
    const auto s = TensorAccessor(src_args, base->src_addr, source_page_size);

    auto st = seq_identity_init(base->start_id);
    read_pages(cb_id, BATCH, cb_page_size, source_read_size, s, base->num_pages, st, seq_identity_next);
}
