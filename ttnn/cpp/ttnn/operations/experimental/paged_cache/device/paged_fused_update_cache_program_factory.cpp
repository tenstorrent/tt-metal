// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "paged_cache_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/experimental/paged_cache/device/paged_fused_update_cache_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::paged_cache::detail {

using namespace tt::constants;
using namespace tt;

bool enable_fp32_dest_acc(
    const tt_metal::IDevice* device, const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    return fp32_dest_acc_en;
}

// Helper struct to group related parameters
struct UpdateCacheParams {
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_compile_time_args;
    std::vector<uint32_t> compute_kernel_args;
    CoreRangeSet cores;
    uint32_t num_cores;
    Buffer* dst_buffer;
    uint32_t src_cb_indices[2];  // Stores pairs of src indices
};

struct UpdateCacheKernels {
    uint32_t unary_reader_kernel_id;
    uint32_t unary_writer_kernel_id;
    uint32_t compute_kernel_id;
    std::vector<CoreCoord> cores;
};

// Helper function to create and set kernel arguments
UpdateCacheKernels setup_kernels_for_update_cache(
    Program& program,
    const UpdateCacheParams& params,
    const IDevice* device,
    bool use_index_tensor,
    bool is_paged_cache,
    bool share_cache,
    uint32_t cache_batch_num_tiles,
    uint32_t Wt,
    uint32_t Wbytes,
    const std::vector<uint32_t>& update_idxs,
    bool row_major,
    uint64_t index_buffer_addr,
    std::optional<const Tensor>& page_table,
    bool fp32_dest_acc_en,
    int input_index) {
    // Create reader kernel
    auto unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "reader_update_cache_interleaved_start_id.cpp",
        params.cores,
        tt_metal::ReaderDataMovementConfig(params.reader_compile_time_args));

    // Create writer kernel
    auto unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "writer_update_cache_interleaved_start_id.cpp",
        params.cores,
        tt_metal::WriterDataMovementConfig(params.writer_compile_time_args));

    // Create compute kernel
    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/update_cache.cpp",
        params.cores,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = params.compute_kernel_args});

    const auto& cores = corerange_to_cores(params.cores, params.num_cores, row_major);

    for (uint32_t i = 0; i < params.num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t update_idx = use_index_tensor ? 0 : update_idxs.at(i);

        // Cache tile info
        const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
        uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

        // Calculate synchronization parameters
        bool wait_to_start = share_cache ? (i != 0) : false;
        bool send_signal = share_cache ? (i != params.num_cores - 1) : false;
        uint32_t send_core_x = 0, send_core_y = 0;

        if (share_cache && send_signal) {
            auto next_core = cores.at(i + 1);
            auto next_core_physical = device->worker_core_from_logical_core(next_core);
            send_core_x = next_core_physical.x;
            send_core_y = next_core_physical.y;
        }

        // Set runtime args for reader
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                params.dst_buffer->address(),
                use_index_tensor ? 0 : cache_start_id,
                index_buffer_addr,
                i,
                is_paged_cache ? page_table.value().buffer()->address() : 0,
                wait_to_start,
            });

        // Set runtime args for writer
        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                params.dst_buffer->address(),
                use_index_tensor ? 0 : cache_start_id,
                use_index_tensor ? 0 : tile_update_offset_B,
                i,
                send_signal,
                send_core_x,
                send_core_y,
            });
    }
    return {unary_reader_kernel_id, unary_writer_kernel_id, compute_kernel_id, cores};
}

operation::ProgramWithCallbacks paged_fused_update_cache_multi_core(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    std::optional<const Tensor> update_idxs_tensor,
    std::optional<const Tensor> page_table,
    const std::vector<uint32_t>& update_idxs,
    const uint32_t batch_offset,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const bool share_cache) {
    Program program{};

    uint32_t num_caches = 2;
    tt_metal::IDevice* device = input_tensor1.device();

    tt::DataFormat cache_cb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor1.get_dtype());
    uint32_t cache_single_tile_size = tt_metal::detail::TileSize(cache_cb_data_format);

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor1.get_dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    bool fp32_dest_acc_en = enable_fp32_dest_acc(device, compute_kernel_config);

    tt::DataFormat interm_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt_metal::detail::TileSize(interm_cb_data_format);

    // Index tensor-specific parameters
    bool use_index_tensor = update_idxs_tensor.has_value();
    uint32_t index_tensor_tile_size = 0;
    uint32_t index_buffer_addr = 0;
    uint32_t log2_page_size = 0;
    uint32_t index_stick_size = 0;
    tt::DataFormat index_data_format = tt::DataFormat::Int32;
    bool index_is_dram = true;
    if (use_index_tensor) {
        index_buffer_addr = use_index_tensor ? update_idxs_tensor.value().buffer()->address() : 0;
        index_data_format = tt_metal::datatype_to_dataformat_converter(update_idxs_tensor.value().get_dtype());
        index_tensor_tile_size = tt_metal::detail::TileSize(index_data_format);
        index_is_dram = update_idxs_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM;
        index_stick_size = update_idxs_tensor.value().buffer()->aligned_page_size();
    }

    // Pagetable-specific parameters
    bool is_paged_cache = page_table.has_value();
    uint32_t batch_size = 0;
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    uint32_t log2_page_table_stick_size = 0;
    tt::DataFormat page_table_data_format = tt::DataFormat::Int32;
    bool page_table_is_dram = true;
    if (is_paged_cache) {
        const auto& page_table_tensor = page_table.value();

        batch_size = page_table_tensor.get_padded_shape()[0];
        block_size = cache_tensor1.get_padded_shape()[2];
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.get_padded_shape()[1];
        page_table_stick_size = page_table_tensor.get_padded_shape()[-1] * page_table_tensor.element_size();

        page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.get_dtype());

        page_table_is_dram = page_table_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    }

    uint32_t Wt = cache_tensor1.get_padded_shape()[-1] / TILE_WIDTH;
    uint32_t St = cache_tensor1.get_padded_shape()[-2] / TILE_HEIGHT;
    uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor1.get_padded_shape()[-1] * sizeof(float)
                                       : cache_tensor1.get_padded_shape()[-1] * 2;  // 2 bytes for bfloat16
    uint32_t cache_total_num_tiles = cache_tensor1.volume() / TILE_HW;
    uint32_t cache_batch_num_tiles =
        share_cache ? 0
                    : cache_total_num_tiles /
                          cache_tensor1.get_padded_shape()[0];  // if share cache, we can set cache batch num tiles to 0
                                                                // so batch offset would be 0 in future calculations
    uint32_t num_tiles = input_tensor1.volume() / TILE_HW;
    uint32_t B = input_tensor1.get_padded_shape()[1];
    uint32_t num_heads = cache_tensor1.get_padded_shape()[1];

    log_debug("cache_cb_data_format: {}", cache_cb_data_format);
    log_debug("input_cb_data_format: {}", input_cb_data_format);
    log_debug("interm_cb_data_format: {}", interm_cb_data_format);
    log_debug("Wbytes: {}", Wbytes);
    log_debug("Wt: {}", Wt);
    log_debug("St: {}", St);

    std::optional<ShardSpec> input1_shard_spec = input_tensor1.shard_spec();
    std::optional<ShardSpec> input2_shard_spec = input_tensor2.shard_spec();
    bool row_major = input1_shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    CoreRangeSet input1_cores = input1_shard_spec.value().grid;
    CoreRangeSet input2_cores = input2_shard_spec.value().grid;
    CoreRangeSet all_cores = input1_cores.merge(input2_cores);
    uint32_t input1_num_cores = input1_cores.num_cores();
    uint32_t input2_num_cores = input2_cores.num_cores();

    uint32_t num_input_tiles = input1_shard_spec.value().shape[0] * input1_shard_spec.value().shape[1] / TILE_HW;

    auto in1_buffer_address = input1_shard_spec.has_value() ? input_tensor1.buffer() : nullptr;

    auto in2_buffer_address = input2_shard_spec.has_value() ? input_tensor2.buffer() : nullptr;

    uint32_t num_cache_tiles = 2 * Wt;   // double buffered
    uint32_t num_interm_tiles = 2 * Wt;  // double buffered
    uint32_t num_output_tiles = B * Wt;

    const tt::CBIndex cache_cb_index = CBIndex::c_0;
    const tt::CBIndex src1_cb_index = CBIndex::c_1;
    const tt::CBIndex src2_cb_index = CBIndex::c_2;
    const tt::CBIndex cb_index_id = CBIndex::c_3;
    const tt::CBIndex cb_pagetable_id = CBIndex::c_4;
    const tt::CBIndex intermed0_cb_index = CBIndex::c_24;
    const tt::CBIndex intermed1_cb_index = CBIndex::c_25;
    const tt::CBIndex intermed2_cb_index = CBIndex::c_26;
    const tt::CBIndex output_cb_index = CBIndex::c_16;

    create_cb(cache_cb_index, program, all_cores, cache_single_tile_size, num_cache_tiles, cache_cb_data_format);
    auto [_1, cb_src1] = create_cb(
        src1_cb_index,
        program,
        input1_cores,
        input_single_tile_size,
        num_input_tiles,
        input_cb_data_format,
        in1_buffer_address);
    auto [_2, cb_src3] = create_cb(
        src2_cb_index,
        program,
        input2_cores,
        input_single_tile_size,
        num_input_tiles,
        input_cb_data_format,
        in2_buffer_address);
    create_cb(
        {intermed0_cb_index, intermed1_cb_index},
        program,
        all_cores,
        interm_single_tile_size,
        num_interm_tiles,
        interm_cb_data_format);
    create_cb(intermed2_cb_index, program, all_cores, interm_single_tile_size, num_interm_tiles, interm_cb_data_format);
    create_cb(output_cb_index, program, all_cores, cache_single_tile_size, num_output_tiles, cache_cb_data_format);

    auto in0_sequential_mode_semaphore_id = tt_metal::CreateSemaphore(
        program, all_cores, 0);  // used for share cache for signaling when the cache is ready to be read

    if (use_index_tensor) {
        create_cb(cb_index_id, program, all_cores, index_tensor_tile_size, 1, index_data_format);
    }

    if (is_paged_cache) {
        create_cb(cb_pagetable_id, program, all_cores, page_table_stick_size, 1, page_table_data_format);
    }

    auto src1_buffer = input_tensor1.buffer();
    auto dst1_buffer = cache_tensor1.buffer();

    auto src2_buffer = input_tensor2.buffer();
    auto dst2_buffer = cache_tensor2.buffer();

    bool src_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader1_compile_time_args = {
        (std::uint32_t)dst_is_dram,
        (std::uint32_t)cache_cb_index,
        (std::uint32_t)src1_cb_index,
        // Index tensor args
        (std::uint32_t)use_index_tensor,
        (std::uint32_t)index_is_dram,
        cb_index_id,
        cache_batch_num_tiles,
        Wt,
        log2_page_size,
        index_stick_size,
        // page_table args
        (std::uint32_t)is_paged_cache,
        (std::uint32_t)num_heads,
        (std::uint32_t)block_size,
        (std::uint32_t)block_size_t,
        (std::uint32_t)max_blocks_per_seq,
        log2_page_table_stick_size,
        page_table_stick_size,
        (std::uint32_t)page_table_is_dram,
        cb_pagetable_id,
        St,
        in0_sequential_mode_semaphore_id,
    };
    std::vector<uint32_t> reader2_compile_time_args = reader1_compile_time_args;
    reader2_compile_time_args[2] = src2_cb_index;

    std::vector<uint32_t> writer1_compile_time_args = {
        (std::uint32_t)dst_is_dram,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)intermed0_cb_index,
        (std::uint32_t)intermed1_cb_index,
        (std::uint32_t)intermed2_cb_index,
        // Index tensor args
        (std::uint32_t)use_index_tensor,
        cb_index_id,
        cache_batch_num_tiles,
        Wt,
        Wbytes,
        // page_table args
        (std::uint32_t)is_paged_cache,
        (std::uint32_t)num_heads,
        (std::uint32_t)block_size,
        (std::uint32_t)block_size_t,
        (std::uint32_t)max_blocks_per_seq,
        cb_pagetable_id,
        St,
        in0_sequential_mode_semaphore_id,
    };
    std::vector<uint32_t> writer2_compile_time_args = writer1_compile_time_args;

    std::vector<uint32_t> compute1_kernel_args = {
        cache_cb_index,
        src1_cb_index,
        intermed0_cb_index,
        intermed1_cb_index,
        intermed2_cb_index,
        output_cb_index,
        Wt,
        num_heads,
    };
    std::vector<uint32_t> compute2_kernel_args = compute1_kernel_args;
    compute2_kernel_args[1] = src2_cb_index;

    // Create array of input parameters
    UpdateCacheParams update_cache_inputs[2] = {// Input 1 parameters
                                                {.reader_compile_time_args = reader1_compile_time_args,
                                                 .writer_compile_time_args = writer1_compile_time_args,
                                                 .compute_kernel_args = compute1_kernel_args,
                                                 .cores = input1_cores,
                                                 .num_cores = input1_num_cores,
                                                 .dst_buffer = dst1_buffer,
                                                 .src_cb_indices = {cache_cb_index, src1_cb_index}},
                                                // Input 2 parameters
                                                {.reader_compile_time_args = reader2_compile_time_args,
                                                 .writer_compile_time_args = writer2_compile_time_args,
                                                 .compute_kernel_args = compute2_kernel_args,
                                                 .cores = input2_cores,
                                                 .num_cores = input2_num_cores,
                                                 .dst_buffer = dst2_buffer,
                                                 .src_cb_indices = {cache_cb_index, src2_cb_index}}};

    // setup kernels both inputs
    UpdateCacheKernels update_cache_kernels[2];
    for (int i = 0; i < num_caches; i++) {
        update_cache_kernels[i] = setup_kernels_for_update_cache(
            program,
            update_cache_inputs[i],
            device,
            use_index_tensor,
            is_paged_cache,
            share_cache,
            cache_batch_num_tiles,
            Wt,
            Wbytes,
            update_idxs,
            row_major,
            index_buffer_addr,
            page_table,
            fp32_dest_acc_en,
            i);
    }

    auto override_runtime_arguments_callback =
        [num_caches,
         update_cache_kernels,
         Wbytes,
         Wt,
         cb_src1,
         cb_src3,
         cache_batch_num_tiles,
         use_index_tensor,
         is_paged_cache](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const std::vector<uint32_t> update_idxs =
                static_cast<const PagedUpdateCacheDeviceOperation*>(operation)->update_idxs;

            auto src1_buffer = input_tensors.at(1).buffer();

            auto src2_buffer = input_tensors.at(3).buffer();

            std::vector dst_buffers = {input_tensors.at(0).buffer(), input_tensors.at(2).buffer()};

            auto index_tensor_addr = use_index_tensor ? optional_input_tensors.at(0).value().buffer()->address() : 0;
            auto page_table_tensor_addr = is_paged_cache ? optional_input_tensors.at(1).value().buffer()->address() : 0;

            if (input_tensors.at(1).is_sharded()) {
                UpdateDynamicCircularBufferAddress(program, cb_src1, *src1_buffer);
            }
            if (input_tensors.at(3).is_sharded()) {
                UpdateDynamicCircularBufferAddress(program, cb_src3, *src2_buffer);
            }

            for (int cache_idx = 0; cache_idx < num_caches; cache_idx++) {
                auto& reader_args_by_core =
                    GetRuntimeArgs(program, update_cache_kernels[cache_idx].unary_reader_kernel_id);
                auto& writer_args_by_core =
                    GetRuntimeArgs(program, update_cache_kernels[cache_idx].unary_writer_kernel_id);
                const auto& cores = update_cache_kernels[cache_idx].cores;

                for (uint32_t i = 0, num_tiles_read = 0; i < cores.size(); ++i) {
                    const uint32_t update_idx = use_index_tensor ? 0 : update_idxs.at(i);
                    // Cache tile info
                    const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
                    const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
                    // Offset to write into untilized cache
                    uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

                    const CoreCoord& core = cores.at(i);

                    {
                        auto& runtime_args = reader_args_by_core.at(core.x).at(core.y);
                        runtime_args[0] = dst_buffers[cache_idx]->address();
                        runtime_args[1] = cache_start_id;
                        runtime_args[2] = index_tensor_addr;
                        runtime_args[4] = page_table_tensor_addr;
                    }

                    {
                        auto& runtime_args = writer_args_by_core.at(core.x).at(core.y);
                        runtime_args[0] = dst_buffers[cache_idx]->address();
                        runtime_args[1] = cache_start_id;
                        runtime_args[2] = tile_update_offset_B;
                    }
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::paged_cache::detail
