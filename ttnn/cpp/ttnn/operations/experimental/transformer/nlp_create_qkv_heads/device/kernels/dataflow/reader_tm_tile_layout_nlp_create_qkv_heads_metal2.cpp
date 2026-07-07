// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_tm_tile_layout_nlp_create_qkv_heads.cpp. The legacy reader is still
// bound by sibling ops on the ProgramDescriptor path (nlp_create_qkv_heads_segformer / _vit), so
// this op's Metal 2.0 Interleaved factory binds a forked copy with named args, DFB handles, and
// typed tensor bindings.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    // READER RUNTIME ARGS
    uint32_t num_blocks = get_arg(args::num_blocks);
    uint32_t in0_tensor_tile_id = get_arg(args::in0_tensor_tile_id);
    uint32_t in1_tensor_tile_id = get_arg(args::in1_tensor_tile_id);

    // COMPILE TIME ARGS
    // READER COMPILE TIME ARGS
    constexpr uint32_t q_num_tiles = get_arg(args::q_num_tiles);
    constexpr uint32_t kv_num_tiles = get_arg(args::kv_num_tiles);

    constexpr auto cb_id_qv = dfb::qv;  // cb for Q, V heads
#ifdef TRANSPOSE_K_HEADS
    constexpr auto cb_id_k = dfb::in_k;  // cb for K heads (used by compute)
#else
    constexpr auto cb_id_k = dfb::qv;  // cb for K heads (directly to writer)
#endif

    constexpr uint32_t onetile = 1;
    const auto s0 = TensorAccessor(ta::input_q);

#ifdef READ_FROM_INPUT_TENSOR_KV
    const auto s1 = TensorAccessor(ta::input_kv);
#endif

    DataflowBuffer cb_qv(cb_id_qv);
    DataflowBuffer cb_k(cb_id_k);

    // get_entry_size() is the arch-portable per-entry byte size (== single tile size here, since
    // the factory sets the DFB entry_size to single_tile_size). DataflowBuffer::get_tile_size() is
    // #ifndef ARCH_QUASAR-gated, so it does not exist on Gen2/Quasar; get_entry_size() does.
    const uint32_t tile_bytes_qv = cb_qv.get_entry_size();
    const uint32_t tile_bytes_k = cb_k.get_entry_size();

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Q
        for (uint32_t i = 0; i < q_num_tiles; i++) {
            cb_qv.reserve_back(onetile);
            uint32_t l1_write_addr = cb_qv.get_write_ptr();
            noc.async_read(
                s0, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes_qv, {.page_id = in0_tensor_tile_id}, {});
            noc.async_read_barrier();
            cb_qv.push_back(onetile);
            in0_tensor_tile_id++;
        }

        // K
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            cb_k.reserve_back(onetile);
            uint32_t l1_write_addr = cb_k.get_write_ptr();
#ifdef READ_FROM_INPUT_TENSOR_KV
            noc.async_read(
                s1, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes_k, {.page_id = in1_tensor_tile_id}, {});
            in1_tensor_tile_id++;
#else
            noc.async_read(
                s0, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes_k, {.page_id = in0_tensor_tile_id}, {});
            in0_tensor_tile_id++;
#endif
            noc.async_read_barrier();
            cb_k.push_back(onetile);
        }

        // V
        for (uint32_t i = 0; i < kv_num_tiles; i++) {
            cb_qv.reserve_back(onetile);
            uint32_t l1_write_addr = cb_qv.get_write_ptr();
#ifdef READ_FROM_INPUT_TENSOR_KV
            noc.async_read(
                s1, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes_qv, {.page_id = in1_tensor_tile_id}, {});
            in1_tensor_tile_id++;
#else
            noc.async_read(
                s0, CoreLocalMem<uint32_t>(l1_write_addr), tile_bytes_qv, {.page_id = in0_tensor_tile_id}, {});
            in0_tensor_tile_id++;
#endif
            noc.async_read_barrier();
            cb_qv.push_back(onetile);
        }
    }
}
