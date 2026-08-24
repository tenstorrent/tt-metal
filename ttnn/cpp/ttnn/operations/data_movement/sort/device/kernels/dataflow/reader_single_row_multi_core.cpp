// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t coordinator_core_physical_coord_x = get_arg(args::coordinator_core_physical_coord_x);
    const uint32_t coordinator_core_physical_coord_y = get_arg(args::coordinator_core_physical_coord_y);

    // Compile time args
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t compute_with_storage_grid_size_x = get_arg(args::compute_with_storage_grid_size_x);
    constexpr uint32_t number_of_available_cores = get_arg(args::number_of_available_cores);
    constexpr uint32_t W_tile_bytes = get_arg(args::W_tile_bytes);
    constexpr uint32_t W_index_bytes = get_arg(args::W_index_bytes);

    constexpr uint32_t one_tile = 1;
    constexpr uint32_t TILE_H = 32;

    // The coordinator has already staged the input into the value-output buffer and generated the
    // index tensor beside it, so a worker reads both of its operands from the output tensors.
    const auto input_tensor_addr_gen = TensorAccessor(tensor::input_tensor);
    const auto index_tensor_addr_gen = TensorAccessor(tensor::index_tensor);

    Noc noc;
#ifdef IS_ROW_MAJOR
    DataflowBuffer rm_input_value_dfb(dfb::rm_input_value);
    DataflowBuffer rm_input_index_dfb(dfb::rm_input_index);
#ifdef IS_UINT16_FP32_MODE
    // Hardware unpack cannot numerically convert UInt16 to Float32 (the ISA only allows a UInt16
    // destination), so the conversion runs on the RISC-V core. The staging buffer holds the raw
    // UInt16 row and is only bound by the factory when the define is set, so other dtypes pay
    // neither the allocation nor the per-element loop.
    DataflowBuffer uint16_input_stage_dfb(dfb::uint16_input_stage);
#endif
#else
    DataflowBuffer input_tensor_dfb(dfb::input_tensor);
    DataflowBuffer index_tensor_dfb(dfb::index_tensor);
    const uint32_t input_tensor_tile_size = input_tensor_dfb.get_tile_size();
    const uint32_t index_tensor_tile_size = index_tensor_dfb.get_tile_size();
#ifdef IS_UINT16_FP32_MODE
    // See the ROW_MAJOR branch above: the UInt16 to Float32 conversion is done in software, staged
    // through a raw UInt16 buffer that only exists in this mode.
    DataflowBuffer uint16_input_stage_dfb(dfb::uint16_input_stage);
    const uint32_t uint16_stage_tile_size = uint16_input_stage_dfb.get_tile_size();
    constexpr uint32_t ELEMENTS_PER_TILE = 1024;  // 32 x 32
#endif
#endif

    // Semaphore setup
    Semaphore<> coordinator_to_cores_sem(sem::coordinator_to_cores);
    Semaphore<> cores_to_coordinator_ready_sem(sem::cores_to_coordinator_ready);
    coordinator_to_cores_sem.set(VALID);  // Reset the semaphore (Valid - we wait for 0)

    for (uint32_t h = 0; h < Ht; h++) {
        // Get core start value
        const uint32_t core_start =
            get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Indicate to the coordinator that the core is ready
        cores_to_coordinator_ready_sem.up(noc, coordinator_core_physical_coord_x, coordinator_core_physical_coord_y, 1);
        noc.async_atomic_barrier();
        coordinator_to_cores_sem.wait(0);     // Wait for coordinator to signal to start
        coordinator_to_cores_sem.set(VALID);  // Reset the semaphore

        // Processing each row
        uint32_t stages = 0;
        for (uint32_t temp = Wt; temp > 1; temp >>= 1) {
            stages++;
        }

        for (uint32_t stage = 1; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);

                // Wait for coordinator
                coordinator_to_cores_sem.wait(0);
                coordinator_to_cores_sem.set(VALID);  // Reset the semaphore

                uint16_t pair_id = 0;
                uint32_t processing_pair_id = core_start;
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;

                    if (j > i) {
                        if (pair_id == processing_pair_id) {
                            // Get indexes of tiles to compare
                            const uint32_t left_tile_id = i;
                            const uint32_t right_tile_id = j;
#ifdef IS_ROW_MAJOR
                            const uint32_t row_base = h * TILE_H;

                            // Construct TILE_H pair-rows: each entry holds one row of both tiles'
                            // data concatenated (left half, then right half). That is the layout
                            // tilize_block(buffer, 2) expects, namely TILE_H rows of 2*TILE_W
                            // elements each. Every row takes two half reads into the same reserved
                            // entry: the left tile's half at offset 0, the right tile's half at
                            // offset W_tile_bytes (the raw dtype's half width).
#ifdef IS_UINT16_FP32_MODE
                            // The raw reads land in a UInt16 staging entry sized for the pair, then
                            // both halves are converted to Float32 into the pair-row entry.
                            {
                                constexpr uint32_t W_elements_pair = 2 * W_tile_bytes / sizeof(uint16_t);
                                for (uint32_t row = 0; row < TILE_H; row++) {
                                    uint16_input_stage_dfb.reserve_back(one_tile);
                                    noc.async_read(
                                        input_tensor_addr_gen,
                                        uint16_input_stage_dfb,
                                        W_tile_bytes,
                                        {.page_id = row_base + row,
                                         .offset_bytes = static_cast<uint32_t>(left_tile_id * W_tile_bytes)},
                                        {.offset_bytes = 0});
                                    noc.async_read(
                                        input_tensor_addr_gen,
                                        uint16_input_stage_dfb,
                                        W_tile_bytes,
                                        {.page_id = row_base + row,
                                         .offset_bytes = static_cast<uint32_t>(right_tile_id * W_tile_bytes)},
                                        {.offset_bytes = W_tile_bytes});
                                    noc.async_read_barrier();
                                    uint16_input_stage_dfb.push_back(one_tile);

                                    uint16_input_stage_dfb.wait_front(one_tile);
                                    rm_input_value_dfb.reserve_back(one_tile);

                                    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                                        uint16_input_stage_dfb.get_read_ptr());
                                    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                        rm_input_value_dfb.get_write_ptr());

                                    for (uint32_t i = 0; i < W_elements_pair; i++) {
                                        float fval = static_cast<float>(static_cast<uint32_t>(src[i]));
                                        uint32_t bits;
                                        __builtin_memcpy(&bits, &fval, sizeof(bits));
                                        dst[i] = bits;
                                    }
                                    // RISC-V stores retire before the write lands in SRAM, and the
                                    // RISC-V core and NoC are independent SRAM clients with no
                                    // program-order guarantee between them (WormholeB0/TensixTile/
                                    // BabyRISCV/MemoryOrdering.md). Drain the fill so the compute
                                    // kernel cannot source a stale pair-row.
                                    __sync_synchronize();

                                    uint16_input_stage_dfb.pop_front(one_tile);
                                    rm_input_value_dfb.push_back(one_tile);
                                }
                            }
#else
                            // Stage all TILE_H pair-rows under one reserve and drain them with a
                            // single barrier. A barrier per row serialises a full DRAM round-trip
                            // for every pair-row entry; batching lets the 2 * TILE_H half-row reads
                            // overlap instead.
                            //
                            // This costs no pipelining: the buffer holds exactly TILE_H entries and
                            // the compute kernel consumes them as one wait_front(TILE_H) /
                            // pop_front(TILE_H), so it could never start on a partially filled run
                            // anyway. Entries form a contiguous run from the write pointer because
                            // the entry count is a multiple of TILE_H (the pointer resets only on
                            // exact equality with the limit, so a block never straddles the end);
                            // consecutive entries are get_stride_size() apart, which is the entry
                            // size on tt-1xx but not necessarily under a strided multi-producer
                            // layout, hence the query.
                            ASSERT(rm_input_value_dfb.get_total_num_entries() % TILE_H == 0);
                            const uint32_t value_stride = rm_input_value_dfb.get_stride_size();
                            rm_input_value_dfb.reserve_back(TILE_H);
                            for (uint32_t row = 0; row < TILE_H; row++) {
                                noc.async_read(
                                    input_tensor_addr_gen,
                                    rm_input_value_dfb,
                                    W_tile_bytes,
                                    {.page_id = row_base + row,
                                     .offset_bytes = static_cast<uint32_t>(left_tile_id * W_tile_bytes)},
                                    {.offset_bytes = row * value_stride});
                                noc.async_read(
                                    input_tensor_addr_gen,
                                    rm_input_value_dfb,
                                    W_tile_bytes,
                                    {.page_id = row_base + row,
                                     .offset_bytes = static_cast<uint32_t>(right_tile_id * W_tile_bytes)},
                                    {.offset_bytes = row * value_stride + W_tile_bytes});
                            }
                            noc.async_read_barrier();
                            rm_input_value_dfb.push_back(TILE_H);
#endif
                            // Same batching as the value pair-rows above.
                            ASSERT(rm_input_index_dfb.get_total_num_entries() % TILE_H == 0);
                            const uint32_t index_stride = rm_input_index_dfb.get_stride_size();
                            rm_input_index_dfb.reserve_back(TILE_H);
                            for (uint32_t row = 0; row < TILE_H; row++) {
                                noc.async_read(
                                    index_tensor_addr_gen,
                                    rm_input_index_dfb,
                                    W_index_bytes,
                                    {.page_id = row_base + row,
                                     .offset_bytes = static_cast<uint32_t>(left_tile_id * W_index_bytes)},
                                    {.offset_bytes = row * index_stride});
                                noc.async_read(
                                    index_tensor_addr_gen,
                                    rm_input_index_dfb,
                                    W_index_bytes,
                                    {.page_id = row_base + row,
                                     .offset_bytes = static_cast<uint32_t>(right_tile_id * W_index_bytes)},
                                    {.offset_bytes = row * index_stride + W_index_bytes});
                            }
                            noc.async_read_barrier();
                            rm_input_index_dfb.push_back(TILE_H);
#else
#ifdef IS_UINT16_FP32_MODE
                            // Stage each raw UInt16 tile, then software-convert it to Float32 into
                            // the input buffer, which carries Float32 entries in this mode.
                            for (const uint32_t tile_id : {left_tile_id, right_tile_id}) {
                                uint16_input_stage_dfb.reserve_back(one_tile);
                                noc.async_read(
                                    input_tensor_addr_gen,
                                    uint16_input_stage_dfb,
                                    uint16_stage_tile_size,
                                    {.page_id = h * Wt + tile_id, .offset_bytes = 0},
                                    {.offset_bytes = 0});
                                noc.async_read_barrier();
                                uint16_input_stage_dfb.push_back(one_tile);

                                uint16_input_stage_dfb.wait_front(one_tile);
                                input_tensor_dfb.reserve_back(one_tile);

                                volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                                    uint16_input_stage_dfb.get_read_ptr());
                                volatile tt_l1_ptr uint32_t* dst =
                                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_tensor_dfb.get_write_ptr());

                                for (uint32_t i = 0; i < ELEMENTS_PER_TILE; i++) {
                                    float fval = static_cast<float>(static_cast<uint32_t>(src[i]));
                                    uint32_t bits;
                                    __builtin_memcpy(&bits, &fval, sizeof(bits));
                                    dst[i] = bits;
                                }
                                // Drain the store buffer before the compute kernel reads this tile.
                                __sync_synchronize();

                                uint16_input_stage_dfb.pop_front(one_tile);
                                input_tensor_dfb.push_back(one_tile);
                            }
#else
                            input_tensor_dfb.reserve_back(one_tile);
                            noc.async_read(
                                input_tensor_addr_gen,
                                input_tensor_dfb,
                                input_tensor_tile_size,
                                {.page_id = h * Wt + left_tile_id, .offset_bytes = 0},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            input_tensor_dfb.push_back(one_tile);

                            input_tensor_dfb.reserve_back(one_tile);
                            noc.async_read(
                                input_tensor_addr_gen,
                                input_tensor_dfb,
                                input_tensor_tile_size,
                                {.page_id = h * Wt + right_tile_id, .offset_bytes = 0},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            input_tensor_dfb.push_back(one_tile);
#endif

                            index_tensor_dfb.reserve_back(one_tile);
                            noc.async_read(
                                index_tensor_addr_gen,
                                index_tensor_dfb,
                                index_tensor_tile_size,
                                {.page_id = h * Wt + left_tile_id, .offset_bytes = 0},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            index_tensor_dfb.push_back(one_tile);

                            index_tensor_dfb.reserve_back(one_tile);
                            noc.async_read(
                                index_tensor_addr_gen,
                                index_tensor_dfb,
                                index_tensor_tile_size,
                                {.page_id = h * Wt + right_tile_id, .offset_bytes = 0},
                                {.offset_bytes = 0});
                            noc.async_read_barrier();
                            index_tensor_dfb.push_back(one_tile);
#endif

                            processing_pair_id += number_of_available_cores;
                        }  // if pair_id == processing_pair_id
                        pair_id++;
                    }  // if j > i
                }  // i loop
            }  // sub loop
        }  // stage loop
    }  // h loop
}
