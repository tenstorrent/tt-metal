// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <cstdint>

/*
To improve performance of both reader and writer kernels the work has been split so that they both prepare input and
save output data.

Reader:
    * Reads input value data from DRAM and writes it to L1 circular buffer.
    * Write processed index data from L1 to DRAM.

Writer:
    * Generates index input data and writes it to L1 circular buffer.
    * Write output values from L1 to DRAM.
*/
void kernel_main() {
    // Runtime args
    const uint32_t core_loop_count = get_arg(args::core_loop_count);

    // Compile time args
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t total_number_of_cores = get_arg(args::total_number_of_cores);
    constexpr uint32_t compute_with_storage_grid_size_x = get_arg(args::compute_with_storage_grid_size_x);
    constexpr uint32_t compute_with_storage_grid_size_y = get_arg(args::compute_with_storage_grid_size_y);
    constexpr uint32_t W_value_bytes = get_arg(args::W_value_bytes);
    constexpr uint32_t W_index_bytes = get_arg(args::W_index_bytes);

    // Input tensor config
    constexpr uint32_t one_tile = 1;

    // TensorAccessors handle both interleaved and sharded buffers natively.
    // For TILE layout: one "page" in the accessor = one tile.
    // For ROW_MAJOR layout: one "page" in the accessor = one row of W elements.
    const auto input_accessor = TensorAccessor(tensor::input_tensor);
    const auto index_accessor = TensorAccessor(tensor::index_tensor);

    Noc noc;
#ifndef IS_ROW_MAJOR
    DataflowBuffer input_tensor_dfb(dfb::input_tensor);
    DataflowBuffer index_output_dfb(dfb::index_tensor_output);
    const uint32_t input_tensor_tile_size = input_tensor_dfb.get_tile_size();
    const uint32_t index_tensor_tile_size = index_output_dfb.get_tile_size();

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

#ifdef IS_UINT16_FP32_MODE
        // UINT16 input mode: the hardware unpack cannot numerically convert
        // UInt16 → Float32 (ISA only allows UInt16 → UInt16 destination), so
        // the reader does a software conversion on the RISC-V core.  DMA the
        // raw UInt16 tile from DRAM into a staging DFB (UInt16 format), then
        // loop over the 1024 elements and write float(uint16_val) into
        // input_tensor (Float32).  The compute kernel then sees correct
        // Float32 values and sorts them exactly.
        //
        // Kept behind IS_UINT16_FP32_MODE so bf16/fp32/uint32 sorts don't pay
        // the extra dfb::uint16_input_stage allocation or the per-element
        // conversion loop — the factory only binds the staging DFB when the
        // define is set.
        DataflowBuffer uint16_input_stage_dfb(dfb::uint16_input_stage);
        const uint32_t uint16_stage_tile_size = uint16_input_stage_dfb.get_tile_size();
        constexpr uint32_t ELEMENTS_PER_TILE = 1024;  // 32 × 32

        for (uint32_t w = 0; w < Wt; w++) {
            uint16_input_stage_dfb.reserve_back(one_tile);
            noc.async_read(
                input_accessor,
                uint16_input_stage_dfb,
                uint16_stage_tile_size,
                {.page_id = h * Wt + w, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            uint16_input_stage_dfb.push_back(one_tile);

            uint16_input_stage_dfb.wait_front(one_tile);
            input_tensor_dfb.reserve_back(one_tile);

            volatile tt_l1_ptr uint16_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(uint16_input_stage_dfb.get_read_ptr());
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_tensor_dfb.get_write_ptr());

            for (uint32_t i = 0; i < ELEMENTS_PER_TILE; i++) {
                float fval = static_cast<float>(static_cast<uint32_t>(src[i]));
                uint32_t bits;
                __builtin_memcpy(&bits, &fval, sizeof(bits));
                dst[i] = bits;
            }

            // RISC-V stores retire before the write lands in L1 and the
            // RISC-V core / NoC are independent L1 clients with no
            // program-order guarantee (see WormholeB0/TensixTile/BabyRISCV/
            // MemoryOrdering.md).  __sync_synchronize() drains the fill so
            // the compute kernel cannot source a stale input tile.
            __sync_synchronize();

            uint16_input_stage_dfb.pop_front(one_tile);
            input_tensor_dfb.push_back(one_tile);
        }
#else
        // Read input tiles from DRAM → tile input DFB
        for (uint32_t w = 0; w < Wt; w++) {
            input_tensor_dfb.reserve_back(one_tile);
            noc.async_read(
                input_accessor,
                input_tensor_dfb,
                input_tensor_tile_size,
                {.page_id = h * Wt + w, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            input_tensor_dfb.push_back(one_tile);
        }
#endif

        // Write sorted index tiles from index output DFB → DRAM
        for (uint32_t w = 0; w < Wt; w++) {
            index_output_dfb.wait_front(one_tile);
            noc.async_write(
                index_output_dfb,
                index_accessor,
                index_tensor_tile_size,
                {.offset_bytes = 0},
                {.page_id = h * Wt + w, .offset_bytes = 0});
            noc.async_write_barrier();
            index_output_dfb.pop_front(one_tile);
        }
    }
#else
    // ------------------------------------------------------------------
    // ROW_MAJOR path
    //
    // The input accessor's page size = W_value_bytes (one RM row).
    // The index accessor's page size = W_index_bytes (one RM index row).
    //
    // For each tile-row (TILE_HEIGHT = 32 consecutive logical rows):
    //   Input:  read 32 pages via noc.async_read → rm_input_dfb
    //           so the compute kernel can tilize them.
    //   Output: drain 32 untilized index pages from rm_index_output_dfb
    //           → write via noc.async_write → index DRAM buffer.
    //
    // With IS_UINT16_FP32_MODE the DRAM buffer is UInt16 but rm_input is
    // Float32, so the reader stages the row in dfb::rm_uint16_input_stage
    // (UInt16) and then software-converts element-by-element into rm_input.
    // ------------------------------------------------------------------
    DataflowBuffer rm_input_dfb(dfb::rm_input);
    DataflowBuffer rm_index_output_dfb(dfb::rm_index_output);
#ifdef IS_UINT16_FP32_MODE
    DataflowBuffer rm_uint16_input_stage_dfb(dfb::rm_uint16_input_stage);
#endif

    constexpr uint32_t TILE_H = 32;  // TILE_HEIGHT

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Base page index for this tile-row group in the RM buffer
        const uint32_t row_base = h * TILE_H;

        // --- Read TILE_H input rows into rm_input_dfb ---
        for (uint32_t row = 0; row < TILE_H; row++) {
#ifdef IS_UINT16_FP32_MODE
            // Stage one raw UInt16 row, then software-convert to Float32.
            // W_value_bytes is sized for the raw UInt16 accessor (2 × W).
            rm_uint16_input_stage_dfb.reserve_back(one_tile);
            noc.async_read(
                input_accessor,
                rm_uint16_input_stage_dfb,
                W_value_bytes,
                {.page_id = row_base + row, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            rm_uint16_input_stage_dfb.push_back(one_tile);

            rm_uint16_input_stage_dfb.wait_front(one_tile);
            rm_input_dfb.reserve_back(one_tile);

            const uint32_t elements_per_row = W_value_bytes / sizeof(uint16_t);
            volatile tt_l1_ptr uint16_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rm_uint16_input_stage_dfb.get_read_ptr());
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rm_input_dfb.get_write_ptr());
            for (uint32_t i = 0; i < elements_per_row; i++) {
                float fval = static_cast<float>(static_cast<uint32_t>(src[i]));
                uint32_t bits;
                __builtin_memcpy(&bits, &fval, sizeof(bits));
                dst[i] = bits;
            }
            __sync_synchronize();

            rm_uint16_input_stage_dfb.pop_front(one_tile);
            rm_input_dfb.push_back(one_tile);
#else
            rm_input_dfb.reserve_back(one_tile);
            noc.async_read(
                input_accessor,
                rm_input_dfb,
                W_value_bytes,
                {.page_id = row_base + row, .offset_bytes = 0},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            rm_input_dfb.push_back(one_tile);
#endif
        }

        // --- Drain TILE_H untilized index rows from rm_index_output_dfb → DRAM ---
        //
        // Compute kernel pack_untilize'd Wt sorted index tiles into
        // TILE_HEIGHT contiguous RM pages in rm_index_output_dfb.
        // pack_untilize_block writes uint16/uint32 elements in the natural
        // little-endian layout that the host expects, so no byte swap is
        // required here regardless of the index dtype.
        for (uint32_t row = 0; row < TILE_H; row++) {
            rm_index_output_dfb.wait_front(one_tile);
            noc.async_write(
                rm_index_output_dfb,
                index_accessor,
                W_index_bytes,
                {.offset_bytes = 0},
                {.page_id = row_base + row, .offset_bytes = 0});
            noc.async_write_barrier();
            rm_index_output_dfb.pop_front(one_tile);
        }
    }
#endif
}
