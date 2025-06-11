// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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

operation::ProgramWithCallbacks paged_fused_update_cache_multi_core(
    const Tensor& cache_tensor1,
    const Tensor& input_tensor1,
    const Tensor& cache_tensor2,
    const Tensor& input_tensor2,
    const std::optional<const Tensor>& update_idxs_tensor,
    const std::optional<const Tensor>& page_table,
    const std::vector<uint32_t>& update_idxs,
    const uint32_t batch_offset,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const bool share_cache) {
    // if input 1, input 2 are tiled call tiled program factory
    if (input_tensor1.layout() == Layout::TILE && input_tensor2.layout() == Layout::TILE) {
        return paged_tiled_fused_update_cache_multi_core(
            cache_tensor1,
            input_tensor1,
            cache_tensor2,
            input_tensor2,
            update_idxs_tensor,
            page_table,
            update_idxs,
            batch_offset,
            compute_kernel_config,
            share_cache);
    } else if (input_tensor1.layout() == Layout::ROW_MAJOR && input_tensor2.layout() == Layout::ROW_MAJOR) {
        return paged_row_major_fused_update_cache_multi_core(
            cache_tensor1,
            input_tensor1,
            cache_tensor2,
            input_tensor2,
            update_idxs_tensor,
            page_table,
            update_idxs,
            batch_offset,
            compute_kernel_config,
            share_cache);
    } else {
        TT_FATAL(false, "Error: input tensor1 and input tensor2 must be either both tiled or both row-major");
    }
}

operation::ProgramWithCallbacks paged_tiled_fused_update_cache_multi_core(
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

    tt::DataFormat cache_cb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor1.dtype());
    uint32_t cache_single_tile_size = tt_metal::detail::TileSize(cache_cb_data_format);

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor1.dtype());
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
        index_data_format = tt_metal::datatype_to_dataformat_converter(update_idxs_tensor.value().dtype());
        index_tensor_tile_size = tt_metal::detail::TileSize(index_data_format);
        index_is_dram = update_idxs_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM;
        index_stick_size = update_idxs_tensor.value().buffer()->aligned_page_size();
    }

    // Pagetable-specific parameters
    bool is_paged_cache = page_table.has_value();
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    uint32_t log2_page_table_stick_size = 0;
    tt::DataFormat page_table_data_format = tt::DataFormat::Int32;
    bool page_table_is_dram = true;
    if (is_paged_cache) {
        const auto& page_table_tensor = page_table.value();

        block_size = cache_tensor1.padded_shape()[2];
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        page_table_stick_size = page_table_tensor.padded_shape()[-1] * page_table_tensor.element_size();

        page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.dtype());

        page_table_is_dram = page_table_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    }

    uint32_t Wt = cache_tensor1.padded_shape()[-1] / TILE_WIDTH;
    uint32_t St = cache_tensor1.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor1.padded_shape()[-1] * sizeof(float)
                                       : cache_tensor1.padded_shape()[-1] * 2;  // 2 bytes for bfloat16
    uint32_t cache_total_num_tiles = cache_tensor1.physical_volume() / TILE_HW;
    uint32_t cache_batch_num_tiles =
        share_cache ? 0
                    : cache_total_num_tiles /
                          cache_tensor1.padded_shape()[0];  // if share cache, we can set cache batch num tiles to 0
                                                            // so batch offset would be 0 in future calculations
    uint32_t num_tiles = input_tensor1.physical_volume() / TILE_HW;
    uint32_t B = input_tensor1.padded_shape()[1];
    uint32_t num_heads = cache_tensor1.padded_shape()[1];

    log_debug(tt::LogOp, "cache_cb_data_format: {}", cache_cb_data_format);
    log_debug(tt::LogOp, "input_cb_data_format: {}", input_cb_data_format);
    log_debug(tt::LogOp, "interm_cb_data_format: {}", interm_cb_data_format);
    log_debug(tt::LogOp, "Wbytes: {}", Wbytes);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "St: {}", St);

    std::optional<ShardSpec> input1_shard_spec = input_tensor1.shard_spec();
    std::optional<ShardSpec> input2_shard_spec = input_tensor2.shard_spec();
    bool row_major = input1_shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    CoreRangeSet input1_cores = input1_shard_spec.value().grid;
    CoreRangeSet input2_cores = input2_shard_spec.value().grid;
    CoreRangeSet all_cores = input1_cores.merge(input2_cores);
    CoreRangeSet all_cores_bb = all_cores.bounding_box();
    CoreRangeSet unused_cores = all_cores_bb.subtract(all_cores);
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

    create_cb(cache_cb_index, program, all_cores_bb, cache_single_tile_size, num_cache_tiles, cache_cb_data_format);
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
        all_cores_bb,
        interm_single_tile_size,
        num_interm_tiles,
        interm_cb_data_format);
    create_cb(
        intermed2_cb_index, program, all_cores_bb, interm_single_tile_size, num_interm_tiles, interm_cb_data_format);
    create_cb(output_cb_index, program, all_cores_bb, cache_single_tile_size, num_output_tiles, cache_cb_data_format);

    auto in0_sequential_mode_semaphore_id = tt_metal::CreateSemaphore(
        program, all_cores_bb, 0);  // used for share cache for signaling when the cache is ready to be read

    if (use_index_tensor) {
        create_cb(cb_index_id, program, all_cores_bb, index_tensor_tile_size, 1, index_data_format);
    }

    if (is_paged_cache) {
        create_cb(cb_pagetable_id, program, all_cores_bb, page_table_stick_size, 1, page_table_data_format);
    }

    auto src1_buffer = input_tensor1.buffer();
    auto dst1_buffer = cache_tensor1.buffer();

    auto src2_buffer = input_tensor2.buffer();
    auto dst2_buffer = cache_tensor2.buffer();

    bool src_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst1_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
        (std::uint32_t)dst_is_dram,
        (std::uint32_t)cache_cb_index,
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

    std::vector<uint32_t> writer_compile_time_args = {
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

    std::vector<uint32_t> compute_kernel_args = {
        src1_cb_index,
        src2_cb_index,
        cache_cb_index,
        intermed0_cb_index,
        intermed1_cb_index,
        intermed2_cb_index,
        output_cb_index,
        Wt,
        num_heads,
    };

    // Create reader kernel
    auto unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "reader_paged_fused_update_cache_interleaved_start_id.cpp",
        all_cores_bb,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Create writer kernel
    auto unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "writer_paged_fused_update_cache_interleaved_start_id.cpp",
        all_cores_bb,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Create compute kernel
    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/paged_fused_update_cache.cpp",
        all_cores_bb,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});

    constexpr bool has_work = true;
    constexpr bool is_input1 = true;

    const auto& cores1 = corerange_to_cores(input1_cores, input1_cores.num_cores(), row_major);
    const auto& cores2 = corerange_to_cores(input2_cores, input2_cores.num_cores(), row_major);

    for (uint32_t i = 0; i < cores1.size(); ++i) {
        const CoreCoord& core1 = cores1.at(i);
        const CoreCoord& core2 = cores2.at(i);

        const uint32_t update_idx = use_index_tensor ? 0 : update_idxs.at(i);

        // Cache tile info
        const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
        uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

        // Calculate synchronization parameters
        bool wait_to_start = share_cache and (i != 0);
        bool send_signal = share_cache and (i != cores1.size() - 1);
        uint32_t send_core1_x = 0, send_core1_y = 0;
        uint32_t send_core2_x = 0, send_core2_y = 0;

        if (send_signal) {
            auto next_core = cores1.at(i + 1);
            auto next_core_physical = device->worker_core_from_logical_core(next_core);
            send_core1_x = next_core_physical.x;
            send_core1_y = next_core_physical.y;

            next_core = cores2.at(i + 1);
            next_core_physical = device->worker_core_from_logical_core(next_core);
            send_core2_x = next_core_physical.x;
            send_core2_y = next_core_physical.y;
        }

        // Input1 args
        // Set runtime args for reader
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core1,
            {
                has_work,
                is_input1,
                dst1_buffer->address(),
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
            core1,
            {
                has_work,
                dst1_buffer->address(),
                use_index_tensor ? 0 : cache_start_id,
                use_index_tensor ? 0 : tile_update_offset_B,
                i,
                send_signal,
                send_core1_x,
                send_core1_y,
            });

        // Set runtime args for compute
        SetRuntimeArgs(
            program,
            compute_kernel_id,
            core1,
            {
                has_work,
                is_input1,
            });

        // Input2 args
        // Set runtime args for reader
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core2,
            {
                has_work,
                !is_input1,
                dst2_buffer->address(),
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
            core2,
            {
                has_work,
                dst2_buffer->address(),
                use_index_tensor ? 0 : cache_start_id,
                use_index_tensor ? 0 : tile_update_offset_B,
                i,
                send_signal,
                send_core2_x,
                send_core2_y,
            });

        // Set runtime args for compute
        SetRuntimeArgs(
            program,
            compute_kernel_id,
            core2,
            {
                has_work,
                !is_input1,
            });
    }

    // Set runtime args for unused cores
    SetRuntimeArgs(program, unary_reader_kernel_id, unused_cores, {!has_work});
    SetRuntimeArgs(program, unary_writer_kernel_id, unused_cores, {!has_work});
    SetRuntimeArgs(program, compute_kernel_id, unused_cores, {!has_work});

    auto override_runtime_arguments_callback =
        [unary_reader_kernel_id,
         unary_writer_kernel_id,
         cores1,
         cores2,
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

            auto dst1_buffer = input_tensors.at(0).buffer();
            auto dst2_buffer = input_tensors.at(2).buffer();

            auto index_tensor_addr = use_index_tensor ? optional_input_tensors.at(0).value().buffer()->address() : 0;
            auto page_table_tensor_addr = is_paged_cache ? optional_input_tensors.at(1).value().buffer()->address() : 0;

            if (input_tensors.at(1).is_sharded()) {
                UpdateDynamicCircularBufferAddress(program, cb_src1, *src1_buffer);
            }
            if (input_tensors.at(3).is_sharded()) {
                UpdateDynamicCircularBufferAddress(program, cb_src3, *src2_buffer);
            }

            auto& reader_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
            auto& writer_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);

            for (uint32_t i = 0, num_tiles_read = 0; i < cores1.size(); ++i) {
                const uint32_t update_idx = use_index_tensor ? 0 : update_idxs.at(i);
                // Cache tile info
                const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
                const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
                // Offset to write into untilized cache
                uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

                const CoreCoord& core1 = cores1.at(i);
                const CoreCoord& core2 = cores2.at(i);

                // Input1 args
                {
                    auto& runtime_args = reader_args_by_core.at(core1.x).at(core1.y);
                    runtime_args[2] = dst1_buffer->address();
                    runtime_args[3] = cache_start_id;
                    runtime_args[4] = index_tensor_addr;
                    runtime_args[6] = page_table_tensor_addr;
                }

                {
                    auto& runtime_args = writer_args_by_core.at(core1.x).at(core1.y);
                    runtime_args[1] = dst1_buffer->address();
                    runtime_args[2] = cache_start_id;
                    runtime_args[3] = tile_update_offset_B;
                }

                // Input2 args
                {
                    auto& runtime_args = reader_args_by_core.at(core2.x).at(core2.y);
                    runtime_args[2] = dst2_buffer->address();
                    runtime_args[3] = cache_start_id;
                    runtime_args[4] = index_tensor_addr;
                    runtime_args[6] = page_table_tensor_addr;
                }

                {
                    auto& runtime_args = writer_args_by_core.at(core2.x).at(core2.y);
                    runtime_args[1] = dst2_buffer->address();
                    runtime_args[2] = cache_start_id;
                    runtime_args[3] = tile_update_offset_B;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks paged_row_major_fused_update_cache_multi_core(
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

    const int32_t num_caches = 2;
    tt_metal::IDevice* device = input_tensor1.device();

    const tt::DataFormat cache_cb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor1.dtype());
    const uint32_t cache_single_tile_size = tt_metal::detail::TileSize(cache_cb_data_format);

    const tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor1.dtype());
    const uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    const bool fp32_dest_acc_en = enable_fp32_dest_acc(device, compute_kernel_config);

    const tt::DataFormat interm_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t interm_single_tile_size = tt_metal::detail::TileSize(interm_cb_data_format);

    // Index tensor-specific parameters
    const bool use_index_tensor = update_idxs_tensor.has_value();
    uint32_t index_tensor_tile_size = 0;
    uint32_t index_buffer_addr = 0;
    const uint32_t log2_page_size = 0;
    uint32_t index_stick_size = 0;
    tt::DataFormat index_data_format = tt::DataFormat::Int32;
    bool index_is_dram = true;
    if (use_index_tensor) {
        index_buffer_addr = use_index_tensor ? update_idxs_tensor.value().buffer()->address() : 0;
        index_data_format = tt_metal::datatype_to_dataformat_converter(update_idxs_tensor.value().dtype());
        index_tensor_tile_size = tt_metal::detail::TileSize(index_data_format);
        index_is_dram = update_idxs_tensor.value().buffer()->buffer_type() == tt_metal::BufferType::DRAM;
        index_stick_size = update_idxs_tensor.value().buffer()->aligned_page_size();
    }

    // Pagetable-specific parameters
    const bool is_paged_cache = page_table.has_value();
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    const uint32_t log2_page_table_stick_size = 0;
    tt::DataFormat page_table_data_format = tt::DataFormat::Int32;
    bool page_table_is_dram = true;
    if (is_paged_cache) {
        const auto& page_table_tensor = page_table.value();

        block_size = cache_tensor1.padded_shape()[2];
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        page_table_stick_size = page_table_tensor.padded_shape()[-1] * page_table_tensor.element_size();

        page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.dtype());

        page_table_is_dram = page_table_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    }

    const uint32_t Wt = cache_tensor1.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t St = cache_tensor1.padded_shape()[-2] / TILE_HEIGHT;
    const uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor1.padded_shape()[-1] * sizeof(float)
                                             : cache_tensor1.padded_shape()[-1] * 2;  // 2 bytes for bfloat16
    const uint32_t cache_total_num_tiles = cache_tensor1.physical_volume() / TILE_HW;
    const uint32_t cache_batch_num_tiles =
        share_cache ? 0
                    : cache_total_num_tiles /
                          cache_tensor1.padded_shape()[0];  // if share cache, we can set cache batch num tiles to 0
                                                            // so batch offset would be 0 in future calculations
    const uint32_t num_tiles = input_tensor1.physical_volume() / TILE_HW;
    const uint32_t B = input_tensor1.padded_shape()[1];
    const uint32_t num_heads = cache_tensor1.padded_shape()[1];

    log_debug(tt::LogOp, "cache_cb_data_format: {}", cache_cb_data_format);
    log_debug(tt::LogOp, "input_cb_data_format: {}", input_cb_data_format);
    log_debug(tt::LogOp, "interm_cb_data_format: {}", interm_cb_data_format);
    log_debug(tt::LogOp, "Wbytes: {}", Wbytes);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "St: {}", St);

    const auto input1_shard_spec_opt = input_tensor1.shard_spec();
    const auto input2_shard_spec_opt = input_tensor2.shard_spec();

    TT_FATAL(input1_shard_spec_opt.has_value(), "input1_shard_spec is not available");
    TT_FATAL(input2_shard_spec_opt.has_value(), "input2_shard_spec is not available");

    const auto& input1_shard_spec = input1_shard_spec_opt.value();
    const auto& input2_shard_spec = input2_shard_spec_opt.value();

    bool row_major = input1_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const CoreRangeSet input1_cores = input1_shard_spec.grid;
    const CoreRangeSet input2_cores = input2_shard_spec.grid;
    const CoreRangeSet all_cores = input1_cores.merge(input2_cores);
    const CoreRangeSet all_cores_bb = all_cores.bounding_box();
    const CoreRangeSet unused_cores = all_cores_bb.subtract(all_cores);
    const uint32_t input1_num_cores = input1_cores.num_cores();
    const uint32_t input2_num_cores = input2_cores.num_cores();

    const uint32_t num_input_tiles = input1_shard_spec.shape[0] * input1_shard_spec.shape[1] / TILE_HW;

    const auto in1_buffer_address = input_tensor1.buffer();
    const auto in2_buffer_address = input_tensor2.buffer();

    const uint32_t num_cache_tiles = 2 * Wt;   // double buffered
    const uint32_t num_interm_tiles = 2 * Wt;  // double buffered
    const uint32_t num_output_tiles = B * Wt;

    const tt::CBIndex cache_cb_index = CBIndex::c_0;
    const tt::CBIndex src1_cb_index = CBIndex::c_1;
    const tt::CBIndex src2_cb_index = CBIndex::c_2;
    const tt::CBIndex cb_index_id = CBIndex::c_3;
    const tt::CBIndex cb_pagetable_id = CBIndex::c_4;
    const tt::CBIndex intermed0_cb_index = CBIndex::c_5;
    const tt::CBIndex intermed1_cb_index = CBIndex::c_6;
    const tt::CBIndex output_cb_index = CBIndex::c_7;

    create_cb(cache_cb_index, program, all_cores_bb, cache_single_tile_size, num_cache_tiles, cache_cb_data_format);
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
        all_cores_bb,
        interm_single_tile_size,
        num_interm_tiles,
        interm_cb_data_format);

    create_cb(output_cb_index, program, all_cores_bb, cache_single_tile_size, num_output_tiles, cache_cb_data_format);

    const auto in0_sequential_mode_semaphore_id = tt_metal::CreateSemaphore(
        program, all_cores_bb, 0);  // used for share cache for signaling when the cache is ready to be read

    if (use_index_tensor) {
        create_cb(cb_index_id, program, all_cores_bb, index_tensor_tile_size, 1, index_data_format);
    }

    if (is_paged_cache) {
        create_cb(cb_pagetable_id, program, all_cores_bb, page_table_stick_size, 1, page_table_data_format);
    }

    const auto src1_buffer = input_tensor1.buffer();
    const auto dst1_buffer = cache_tensor1.buffer();

    const auto src2_buffer = input_tensor2.buffer();
    const auto dst2_buffer = cache_tensor2.buffer();

    const bool src_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    const bool dst_is_dram = dst1_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    std::vector<uint32_t> reader_compile_time_args = {
        src1_cb_index,
        src2_cb_index,
        dst_is_dram,
        cache_cb_index,
        // Index tensor args
        use_index_tensor,
        index_is_dram,
        cb_index_id,
        cache_batch_num_tiles,
        Wt,
        log2_page_size,
        index_stick_size,
        // page_table args
        is_paged_cache,
        num_heads,
        block_size,
        block_size_t,
        max_blocks_per_seq,
        log2_page_table_stick_size,
        page_table_stick_size,
        page_table_is_dram,
        cb_pagetable_id,
        St,
        in0_sequential_mode_semaphore_id,
    };

    std::vector<uint32_t> writer_compile_time_args = {
        dst_is_dram,
        output_cb_index,
        intermed0_cb_index,
        intermed1_cb_index,
        src1_cb_index,
        src2_cb_index,
        // Index tensor args
        use_index_tensor,
        cb_index_id,
        cache_batch_num_tiles,
        Wt,
        Wbytes,
        // page_table args
        is_paged_cache,
        num_heads,
        block_size,
        block_size_t,
        max_blocks_per_seq,
        cb_pagetable_id,
        St,
        in0_sequential_mode_semaphore_id,
    };

    std::vector<uint32_t> compute_kernel_args = {
        src1_cb_index,
        src2_cb_index,
        cache_cb_index,
        intermed0_cb_index,
        intermed1_cb_index,
        output_cb_index,
        Wt,
        num_heads,
    };

    // Create reader kernel
    const auto unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp",
        all_cores_bb,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Create writer kernel
    const auto unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp",
        all_cores_bb,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Create compute kernel
    const auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/"
        "paged_row_major_fused_update_cache.cpp",
        all_cores_bb,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});

    constexpr bool has_work = true;
    constexpr bool is_input1 = true;

    const auto& cores1 = corerange_to_cores(input1_cores, input1_cores.num_cores(), row_major);
    const auto& cores2 = corerange_to_cores(input2_cores, input2_cores.num_cores(), row_major);

    for (uint32_t i = 0; i < cores1.size(); ++i) {
        const CoreCoord& core1 = cores1.at(i);
        const CoreCoord& core2 = cores2.at(i);

        const uint32_t update_idx = use_index_tensor ? 0 : update_idxs.at(i);

        // Cache tile info
        const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
        const uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

        // Calculate synchronization parameters
        const bool wait_to_start = share_cache and (i != 0);
        const bool send_signal = share_cache and (i != cores1.size() - 1);
        uint32_t send_core1_x = 0, send_core1_y = 0;
        uint32_t send_core2_x = 0, send_core2_y = 0;

        if (send_signal) {
            auto next_core = cores1.at(i + 1);
            auto next_core_physical = device->worker_core_from_logical_core(next_core);
            send_core1_x = next_core_physical.x;
            send_core1_y = next_core_physical.y;

            next_core = cores2.at(i + 1);
            next_core_physical = device->worker_core_from_logical_core(next_core);
            send_core2_x = next_core_physical.x;
            send_core2_y = next_core_physical.y;
        }

        // Input1 args
        // Set runtime args for reader
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core1,
            {
                has_work,
                is_input1,
                dst1_buffer->address(),
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
            core1,
            {
                has_work,
                dst1_buffer->address(),
                use_index_tensor ? 0 : cache_start_id,
                use_index_tensor ? 0 : tile_update_offset_B,
                i,
                send_signal,
                send_core1_x,
                send_core1_y,
                is_input1,
            });

        // Set runtime args for compute
        SetRuntimeArgs(
            program,
            compute_kernel_id,
            core1,
            {
                has_work,
                is_input1,
            });

        // Input2 args
        // Set runtime args for reader
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core2,
            {
                has_work,
                !is_input1,
                dst2_buffer->address(),
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
            core2,
            {
                has_work,
                dst2_buffer->address(),
                use_index_tensor ? 0 : cache_start_id,
                use_index_tensor ? 0 : tile_update_offset_B,
                i,
                send_signal,
                send_core2_x,
                send_core2_y,
                !is_input1,
            });

        // Set runtime args for compute
        SetRuntimeArgs(
            program,
            compute_kernel_id,
            core2,
            {
                has_work,
                !is_input1,
            });
    }

    // Set runtime args for unused cores
    SetRuntimeArgs(program, unary_reader_kernel_id, unused_cores, {!has_work});
    SetRuntimeArgs(program, unary_writer_kernel_id, unused_cores, {!has_work});
    SetRuntimeArgs(program, compute_kernel_id, unused_cores, {!has_work});

    auto override_runtime_arguments_callback =
        [unary_reader_kernel_id,
         unary_writer_kernel_id,
         cores1,
         cores2,
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

            const auto src1_buffer = input_tensors.at(1).buffer();
            const auto src2_buffer = input_tensors.at(3).buffer();

            const auto dst1_buffer = input_tensors.at(0).buffer();
            const auto dst2_buffer = input_tensors.at(2).buffer();

            const auto index_tensor_addr =
                use_index_tensor ? optional_input_tensors.at(0).value().buffer()->address() : 0;
            const auto page_table_tensor_addr =
                is_paged_cache ? optional_input_tensors.at(1).value().buffer()->address() : 0;

            if (input_tensors.at(1).is_sharded()) {
                UpdateDynamicCircularBufferAddress(program, cb_src1, *src1_buffer);
            }
            if (input_tensors.at(3).is_sharded()) {
                UpdateDynamicCircularBufferAddress(program, cb_src3, *src2_buffer);
            }

            auto& reader_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
            auto& writer_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);

            for (uint32_t i = 0, num_tiles_read = 0; i < cores1.size(); ++i) {
                const uint32_t update_idx = use_index_tensor ? 0 : update_idxs.at(i);
                // Cache tile info
                const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
                const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
                // Offset to write into untilized cache
                const uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

                const CoreCoord& core1 = cores1.at(i);
                const CoreCoord& core2 = cores2.at(i);

                // Input1 args
                {
                    auto& runtime_args = reader_args_by_core.at(core1.x).at(core1.y);
                    runtime_args[2] = dst1_buffer->address();
                    runtime_args[3] = cache_start_id;
                    runtime_args[4] = index_tensor_addr;
                    runtime_args[6] = page_table_tensor_addr;
                }

                {
                    auto& runtime_args = writer_args_by_core.at(core1.x).at(core1.y);
                    runtime_args[1] = dst1_buffer->address();
                    runtime_args[2] = cache_start_id;
                    runtime_args[3] = tile_update_offset_B;
                }

                // Input2 args
                {
                    auto& runtime_args = reader_args_by_core.at(core2.x).at(core2.y);
                    runtime_args[2] = dst2_buffer->address();
                    runtime_args[3] = cache_start_id;
                    runtime_args[4] = index_tensor_addr;
                    runtime_args[6] = page_table_tensor_addr;
                }

                {
                    auto& runtime_args = writer_args_by_core.at(core2.x).at(core2.y);
                    runtime_args[1] = dst2_buffer->address();
                    runtime_args[2] = cache_start_id;
                    runtime_args[3] = tile_update_offset_B;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::paged_cache::detail
