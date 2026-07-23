// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "internal/risc_attribs.h"
#include "experimental/kernel_args.h"
#include <tt-metalium/constants.hpp>

using namespace tt::constants;
void kernel_main() {
    Noc noc;

    // Case 2 bindings (input remote-read base + q/k/v output local-write bases). The outputs use
    // TensorParameters rather than borrowed-memory DFBs to avoid a Metal 2.0 framework bug that corrupts
    // a borrowed DFB's device-side base in multi-work-unit programs (needed by the !overlap layout); the
    // TensorParameter base-address path is unaffected. Raw arithmetic below is unchanged from legacy.
    auto qkv_input = TensorAccessor(tensor::input);
    uint32_t q_start_addr = qkv_input.get_bank_base_address();
    uint32_t index_in_cores = get_arg(args::index_in_cores);

    constexpr auto ELEMENT_SIZE = get_arg(args::element_size);
    constexpr auto SUBTILE_LINE_BYTES = get_arg(args::sub_tile_line_bytes);
    constexpr auto head_size = get_arg(args::head_size);
    constexpr auto num_q_heads = get_arg(args::num_q_heads);
    constexpr auto num_kv_heads = get_arg(args::num_kv_heads);
    constexpr auto head_size_num_tiles = get_arg(args::head_size_num_tiles);
    constexpr auto PHASES_TO_READ =
        get_arg(args::phases_to_read);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase
    constexpr auto in_num_cores = get_arg(args::in_num_cores);
    // PROCESS_QV / PROCESS_K are host-emitted defines gating which output tensors this kernel writes.
#ifdef USE_BATCH_OFFSET
    constexpr auto index_stick_size = get_arg(args::index_stick_size);
#endif

    // The input-core NoC coordinate arrays are a variable-count block indexed at runtime
    // (in0_mcast_noc_x[cur_core_idx]); they are runtime varargs — noc_x occupies varargs
    // [0, in_num_cores), noc_y occupies [in_num_cores, 2 * in_num_cores).

    uint32_t device_batch_offset = 0;

#ifdef USE_BATCH_OFFSET
    {
        const auto addrg = TensorAccessor(tensor::batch_offset);
        DataflowBuffer cb_batch_offset(dfb::batch_offset);
        cb_batch_offset.reserve_back(1);
        uint32_t index_cb_wr_ptr = cb_batch_offset.get_write_ptr();
        // Read the batch offset 1 page to read
        noc.async_read(addrg, CoreLocalMem<uint32_t>(index_cb_wr_ptr), index_stick_size, {.page_id = 0}, {});
        noc.async_read_barrier();
        cb_batch_offset.push_back(1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);
        // Always pick 1st value in tensor as batch offset
        device_batch_offset = index_ptr[0];
    }
#endif
    device_batch_offset += index_in_cores;
    uint32_t in_tile_offset_by_batch = device_batch_offset < 16
                                           ? device_batch_offset * SUBTILE_LINE_BYTES
                                           : (device_batch_offset - 16) * SUBTILE_LINE_BYTES + 512 * ELEMENT_SIZE;

    UnicastEndpoint src_ep;

    // Output shard base pointers (resident q/k/v output tensors).
#ifdef PROCESS_QV
    uint32_t q_out_base = TensorAccessor(tensor::q_out).get_bank_base_address();
    uint32_t v_out_base = TensorAccessor(tensor::v_out).get_bank_base_address();
#endif
#ifdef PROCESS_K
    uint32_t k_out_base = TensorAccessor(tensor::k_out).get_bank_base_address();
#endif

    // Q
    uint32_t cur_core_idx = 0;
    uint32_t num_tiles_per_core = head_size_num_tiles * (num_q_heads + 2 * num_kv_heads) / in_num_cores;
    uint32_t num_q_cores = (num_q_heads * head_size_num_tiles) / num_tiles_per_core;
    uint32_t num_kv_cores = (num_kv_heads * head_size_num_tiles) / num_tiles_per_core;

    uint32_t qkv_noc_x = get_vararg(cur_core_idx);
    uint32_t qkv_noc_y = get_vararg(in_num_cores + cur_core_idx);
    uint32_t qkv_read_addr = q_start_addr + in_tile_offset_by_batch;
    uint32_t num_tiles_read_cur_core = 0;
    uint32_t q_write_addr = 0;
    constexpr uint32_t tile_size = head_size / head_size_num_tiles;
    constexpr uint32_t HALF_TILE_ELEMENTS = FACE_HEIGHT * TILE_WIDTH;
    constexpr uint32_t SUBTILE_ROWS = FACE_HEIGHT;

    // Skip Q section if PROCESS_QV is False
#ifdef PROCESS_QV
    {
        for (uint32_t q = 0; q < num_q_heads; ++q) {
            uint32_t tile_row_index = q / TILE_HEIGHT;
            uint32_t row_in_tile = q % TILE_HEIGHT;
            uint32_t offset_in_tile = row_in_tile < SUBTILE_ROWS ? row_in_tile * SUBTILE_LINE_BYTES
                                                                 : (row_in_tile - SUBTILE_ROWS) * SUBTILE_LINE_BYTES +
                                                                       HALF_TILE_ELEMENTS * ELEMENT_SIZE;
            uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
            uint32_t q_write_addr = q_out_base + wptr_offset;
            for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
                // Read first phase
                if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                    noc.async_read(
                        src_ep,
                        CoreLocalMem<uint32_t>(q_write_addr),
                        SUBTILE_LINE_BYTES,
                        {.noc_x = qkv_noc_x, .noc_y = qkv_noc_y, .addr = qkv_read_addr},
                        {});
                }
                // Read second phase
                if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                    noc.async_read(
                        src_ep,
                        CoreLocalMem<uint32_t>(q_write_addr + FACE_HW * ELEMENT_SIZE),
                        SUBTILE_LINE_BYTES,
                        {.noc_x = qkv_noc_x, .noc_y = qkv_noc_y, .addr = qkv_read_addr + FACE_HW * ELEMENT_SIZE},
                        {});
                }

                qkv_read_addr += tile_size;
                q_write_addr += tile_size;
                num_tiles_read_cur_core++;

                if (num_tiles_read_cur_core == num_tiles_per_core) {
                    cur_core_idx++;
                    qkv_noc_x = get_vararg(cur_core_idx);
                    qkv_noc_y = get_vararg(in_num_cores + cur_core_idx);
                    qkv_read_addr = q_start_addr + in_tile_offset_by_batch;
                    num_tiles_read_cur_core = 0;
                }
            }
        }
    }
#else
    {
        cur_core_idx += num_q_cores;
        qkv_noc_x = get_vararg(cur_core_idx);
        qkv_noc_y = get_vararg(in_num_cores + cur_core_idx);
        qkv_read_addr = q_start_addr + in_tile_offset_by_batch;
    }
#endif

#ifdef PROCESS_K
    {
        // K
        uint32_t k_write_addr = 0;

        // Read 2 phases per tile, where there are num_q_heads * q_num_tiles tiles
        for (uint32_t k = 0; k < num_kv_heads; ++k) {
            uint32_t tile_row_index = k / TILE_HEIGHT;
            uint32_t row_in_tile = k % TILE_HEIGHT;
            uint32_t offset_in_tile = row_in_tile < SUBTILE_ROWS ? row_in_tile * SUBTILE_LINE_BYTES
                                                                 : (row_in_tile - SUBTILE_ROWS) * SUBTILE_LINE_BYTES +
                                                                       HALF_TILE_ELEMENTS * ELEMENT_SIZE;
            uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
            uint32_t k_write_addr = k_out_base + wptr_offset;
            for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
                // Read first phase
                if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                    noc.async_read(
                        src_ep,
                        CoreLocalMem<uint32_t>(k_write_addr),
                        SUBTILE_LINE_BYTES,
                        {.noc_x = qkv_noc_x, .noc_y = qkv_noc_y, .addr = qkv_read_addr},
                        {});
                }
                // Read second phase
                if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                    noc.async_read(
                        src_ep,
                        CoreLocalMem<uint32_t>(k_write_addr + FACE_HW * ELEMENT_SIZE),
                        SUBTILE_LINE_BYTES,
                        {.noc_x = qkv_noc_x, .noc_y = qkv_noc_y, .addr = qkv_read_addr + FACE_HW * ELEMENT_SIZE},
                        {});
                }

                qkv_read_addr += tile_size;
                k_write_addr += tile_size;
                num_tiles_read_cur_core++;

                if (num_tiles_read_cur_core == num_tiles_per_core) {
                    cur_core_idx++;
                    qkv_noc_x = get_vararg(cur_core_idx);
                    qkv_noc_y = get_vararg(in_num_cores + cur_core_idx);
                    qkv_read_addr = q_start_addr + in_tile_offset_by_batch;
                    num_tiles_read_cur_core = 0;
                }
            }
        }
    }
#else
    {
        cur_core_idx += num_kv_cores;
        qkv_noc_x = get_vararg(cur_core_idx);
        qkv_noc_y = get_vararg(in_num_cores + cur_core_idx);
        qkv_read_addr = q_start_addr + in_tile_offset_by_batch;
    }
#endif

#ifdef PROCESS_QV
    {
        // v
        uint32_t v_write_addr = 0;

        // Read 2 phases per tile, where there are num_q_heads * q_num_tiles tiles
        for (uint32_t v = 0; v < num_kv_heads; ++v) {
            uint32_t tile_row_index = v / TILE_HEIGHT;
            uint32_t row_in_tile = v % TILE_HEIGHT;
            uint32_t offset_in_tile = row_in_tile < SUBTILE_ROWS ? row_in_tile * SUBTILE_LINE_BYTES
                                                                 : (row_in_tile - SUBTILE_ROWS) * SUBTILE_LINE_BYTES +
                                                                       HALF_TILE_ELEMENTS * ELEMENT_SIZE;
            uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
            uint32_t v_write_addr = v_out_base + wptr_offset;
            for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
                // Read first phase
                if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                    noc.async_read(
                        src_ep,
                        CoreLocalMem<uint32_t>(v_write_addr),
                        SUBTILE_LINE_BYTES,
                        {.noc_x = qkv_noc_x, .noc_y = qkv_noc_y, .addr = qkv_read_addr},
                        {});
                }
                // Read second phase
                if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                    noc.async_read(
                        src_ep,
                        CoreLocalMem<uint32_t>(v_write_addr + FACE_HW * ELEMENT_SIZE),
                        SUBTILE_LINE_BYTES,
                        {.noc_x = qkv_noc_x, .noc_y = qkv_noc_y, .addr = qkv_read_addr + FACE_HW * ELEMENT_SIZE},
                        {});
                }

                qkv_read_addr += tile_size;
                v_write_addr += tile_size;
                num_tiles_read_cur_core++;

                if (num_tiles_read_cur_core == num_tiles_per_core) {
                    cur_core_idx++;
                    qkv_noc_x = get_vararg(cur_core_idx);
                    qkv_noc_y = get_vararg(in_num_cores + cur_core_idx);
                    qkv_read_addr = q_start_addr + in_tile_offset_by_batch;
                    num_tiles_read_cur_core = 0;
                }
            }
        }
    }
#endif

    noc.async_read_barrier();
}
