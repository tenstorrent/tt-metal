// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t in_tile_offset_by_head = get_arg(args::in_tile_offset_by_head);

    // Input base address arrives via the tensor binding (was a Buffer* runtime arg). This is a Case 2
    // (raw-pointer) binding: the kernel keeps its hand-rolled cross-core NoC gather and only pulls the
    // base pointer through the TensorAccessor bridge.
    TensorAccessor input(tensor::input);
    uint32_t q_start_addr = input.get_bank_base_address();

    constexpr uint32_t ELEMENT_SIZE = get_arg(args::element_size);
    constexpr uint32_t SUBTILE_LINE_BYTES = get_arg(args::sub_tile_line_bytes);
    constexpr uint32_t head_size = get_arg(args::head_size);
    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t head_size_num_tiles = get_arg(args::head_size_num_tiles);
    constexpr uint32_t PHASES_TO_READ =
        get_arg(args::phases_to_read);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase

    constexpr uint32_t in_num_cores = get_arg(args::in_num_cores);
    constexpr uint32_t face_h = get_arg(args::face_h);
    constexpr uint32_t face_hw = get_arg(args::face_hw);

    // The input NoC coordinates arrive as runtime varargs, laid out [x-coords (in_num_cores),
    // y-coords (in_num_cores)]; read positionally via get_vararg (formerly two L1 arrays reached
    // through get_arg_addr).

    DataflowBuffer cb_q_out(dfb::q_out);
    UnicastEndpoint src_ep;

    // Q
    uint32_t cur_core_idx = 0;
    uint32_t total_input_cores = in_num_cores;
    uint32_t num_tiles_per_core = (head_size_num_tiles * batch) / total_input_cores;

    uint32_t qkv_noc_x = get_vararg(cur_core_idx);
    uint32_t qkv_noc_y = get_vararg(in_num_cores + cur_core_idx);
    uint32_t qkv_read_addr = q_start_addr + in_tile_offset_by_head;
    uint32_t num_tiles_read_cur_core = 0;
    uint32_t q_write_addr = 0;
    uint32_t tile_size = head_size / head_size_num_tiles;
    const uint32_t cb_write_ptr_base = cb_q_out.get_write_ptr();

    for (uint32_t q = 0; q < batch; ++q) {
        uint32_t wptr_offset = q < face_h ? q * SUBTILE_LINE_BYTES : (q + face_h) * SUBTILE_LINE_BYTES;
        uint32_t q_write_addr = cb_write_ptr_base + wptr_offset;
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
                    CoreLocalMem<uint32_t>(q_write_addr + face_hw * ELEMENT_SIZE),
                    SUBTILE_LINE_BYTES,
                    {.noc_x = qkv_noc_x, .noc_y = qkv_noc_y, .addr = qkv_read_addr + face_hw * ELEMENT_SIZE},
                    {});
            }

            qkv_read_addr += tile_size;
            q_write_addr += tile_size;
            num_tiles_read_cur_core++;

            if (num_tiles_read_cur_core == num_tiles_per_core) {
                cur_core_idx++;
                qkv_noc_x = get_vararg(cur_core_idx);
                qkv_noc_y = get_vararg(in_num_cores + cur_core_idx);
                qkv_read_addr = q_start_addr + in_tile_offset_by_head;
                num_tiles_read_cur_core = 0;
            }
        }
    }

    noc.async_read_barrier();
}
