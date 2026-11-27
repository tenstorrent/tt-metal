// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

namespace kda_relay {

// The call's chronology as the relay's readers derive it. Compile-time args from `ct`: cb_chron and cb_ctl, sp_size,
// local_rows, whether there is a valid end, the slab's args (slab_kernel_args, then its entry's tiles and the word of
// cb_chron's second DRAM-aligned slot its index lands at), then actual_start's, the slab index's and the end's
// accessor args. Common args from `common`: those three buffers, then sp_rank.
template <uint32_t ct, uint32_t common>
struct Chronology {
    static constexpr uint32_t cb_chron = get_compile_time_arg_val(ct);
    static constexpr uint32_t cb_ctl = get_compile_time_arg_val(ct + 1);
    static constexpr uint32_t sp_size = get_compile_time_arg_val(ct + 2);
    static constexpr uint32_t local_rows = get_compile_time_arg_val(ct + 3);
    static constexpr bool has_actual_end = get_compile_time_arg_val(ct + 4);
    static constexpr bool slab = get_compile_time_arg_val(ct + 5);
    static constexpr uint32_t slab_index_limit = get_compile_time_arg_val(ct + 6);
    static constexpr uint32_t slab_scale = get_compile_time_arg_val(ct + 7);
    static constexpr uint32_t slab_offset = get_compile_time_arg_val(ct + 8);
    static constexpr uint32_t entry_tiles = get_compile_time_arg_val(ct + 9);
    static constexpr uint32_t slab_index_word = get_compile_time_arg_val(ct + 10);
    static constexpr auto start_args = TensorAccessorArgs<ct + 11>();
    static constexpr auto slab_index_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();
    static constexpr auto end_args = TensorAccessorArgs<slab_index_args.next_compile_time_args_offset()>();
    static constexpr uint32_t next_ct = end_args.next_compile_time_args_offset();
    static constexpr uint32_t next_common = common + 4;

    kda_chronology::Topology topology;
    kda_chronology::Topology interval;
    uint32_t sp_rank;
    uint32_t position;  // this rank's place in the relay, 0 on the chronologically first rank (the holder)
    uint32_t index;     // the slab index
    bool zero_entry;    // the chunk starts from zeros (a request's head, or no slab entry)

    // Reserves cb_chron and cb_ctl, reads actual_start (word 0 of cb_chron), the slab index and the end into them,
    // and derives the chronology. The caller fills and pushes both.
    Chronology() {
        cb_reserve_back(cb_chron, 1);
        cb_reserve_back(cb_ctl, 1);
        const uint32_t chron = get_write_ptr(cb_chron);
        const auto start = TensorAccessor(start_args, get_common_arg_val<uint32_t>(common));
        noc_async_read(start.get_noc_addr(0), chron, sizeof(uint32_t));
        if constexpr (slab) {
            const auto slab_index = TensorAccessor(slab_index_args, get_common_arg_val<uint32_t>(common + 1));
            noc_async_read(slab_index.get_noc_addr(0), chron + slab_index_word * sizeof(uint32_t), sizeof(uint32_t));
        }
        if constexpr (has_actual_end) {
            const auto end = TensorAccessor(end_args, get_common_arg_val<uint32_t>(common + 2));
            noc_async_read(end.get_noc_addr(0), get_write_ptr(cb_ctl), sizeof(uint32_t));
        }
        noc_async_read_barrier();
        const auto* chron_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chron);
        sp_rank = get_common_arg_val<uint32_t>(common + 3);
        const uint32_t actual_start = chron_words[0];
        index = slab ? chron_words[slab_index_word] : 0;
        topology = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
        interval = has_actual_end ? kda_chronology::derive_interval(
                                        actual_start,
                                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl))[0],
                                        sp_rank,
                                        sp_size,
                                        local_rows)
                                  : topology;
        position = (topology.rank + sp_size - topology.first_rank) % sp_size;
        zero_entry = slab && (actual_start == 0 || index >= slab_index_limit);
    }

    bool holder() const { return position == 0; }
    // The slab entry's first tile (0 without a slab).
    uint32_t entry_base() const { return slab ? (index * slab_scale + slab_offset) * entry_tiles : 0; }
};

}  // namespace kda_relay
