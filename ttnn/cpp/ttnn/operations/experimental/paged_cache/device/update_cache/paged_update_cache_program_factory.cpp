// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "paged_update_cache_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include "paged_update_cache_program_factory.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/mesh_device_operation_utils.hpp"
#include <unordered_map>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;

static bool enable_fp32_dest(
    const tt_metal::IDevice* device, const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    return fp32_dest_acc_en;
}

PagedUpdateCacheProgramFactory::cached_program_t PagedUpdateCacheProgramFactory::create(
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    Program program{};

    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    tt_metal::IDevice* device = input_tensor.device();

    tt::DataFormat cache_cb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor.dtype());
    uint32_t cache_single_tile_size = tt::tile_size(cache_cb_data_format);

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    bool fp32_dest_acc_en = enable_fp32_dest(device, operation_attributes.compute_kernel_config);

    tt::DataFormat interm_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt::tile_size(interm_cb_data_format);

    // Index tensor-specific parameters
    bool use_index_tensor = update_idxs_tensor.has_value();
    uint32_t index_tensor_tile_size = 0;
    uint32_t index_buffer_addr = 0;
    uint32_t log2_page_size = 0;
    uint32_t index_stick_size = 0;
    tt::DataFormat index_data_format = tt::DataFormat::Int32;
    if (use_index_tensor) {
        index_buffer_addr = update_idxs_tensor.value().buffer()->address();
        index_data_format = tt_metal::datatype_to_dataformat_converter(update_idxs_tensor.value().dtype());
        index_tensor_tile_size = tt::tile_size(index_data_format);
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
    if (is_paged_cache) {
        const auto& page_table_tensor = page_table.value();

        block_size = cache_tensor.padded_shape()[2];
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        page_table_stick_size = page_table_tensor.padded_shape()[-1] * page_table_tensor.element_size();

        page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.dtype());
    }

    uint32_t Wt = cache_tensor.padded_shape()[-1] / TILE_WIDTH;
    uint32_t St = cache_tensor.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor.padded_shape()[-1] * sizeof(float)
                                       : cache_tensor.padded_shape()[-1] * 2;  // 2 bytes for bfloat16
    uint32_t cache_total_num_tiles = cache_tensor.physical_volume() / TILE_HW;
    uint32_t cache_batch_num_tiles =
        operation_attributes.share_cache
            ? 0
            : cache_total_num_tiles /
                  cache_tensor.padded_shape()[0];  // if share cache, we can set cache batch num tiles to 0
                                                   // so batch offset would be 0 in future calculations
    uint32_t B = input_tensor.padded_shape()[1];
    uint32_t num_heads = cache_tensor.padded_shape()[1];

    log_debug(tt::LogOp, "cache_cb_data_format: {}", cache_cb_data_format);
    log_debug(tt::LogOp, "input_cb_data_format: {}", input_cb_data_format);
    log_debug(tt::LogOp, "interm_cb_data_format: {}", interm_cb_data_format);
    log_debug(tt::LogOp, "Wbytes: {}", Wbytes);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "St: {}", St);

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();
    bool row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    CoreRangeSet all_cores = shard_spec.value().grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t num_input_tiles = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
    auto* in1_buffer_address = shard_spec.has_value() ? input_tensor.buffer() : nullptr;

    uint32_t num_cache_tiles = 2 * Wt;   // double buffered
    uint32_t num_interm_tiles = 2 * Wt;  // double buffered
    uint32_t num_output_tiles = B * Wt;

    const tt::CBIndex src0_cb_index = CBIndex::c_0;
    const tt::CBIndex src1_cb_index = CBIndex::c_1;
    const tt::CBIndex cb_index_id = CBIndex::c_2;
    const tt::CBIndex cb_pagetable_id = CBIndex::c_3;
    const tt::CBIndex intermed0_cb_index = CBIndex::c_24;
    const tt::CBIndex intermed1_cb_index = CBIndex::c_25;
    const tt::CBIndex intermed2_cb_index = CBIndex::c_26;
    const tt::CBIndex output_cb_index = CBIndex::c_16;

    create_cb(src0_cb_index, program, all_cores, cache_single_tile_size, num_cache_tiles, cache_cb_data_format);
    auto [_, cb_src1] = create_cb(
        src1_cb_index,
        program,
        all_cores,
        input_single_tile_size,
        num_input_tiles,
        input_cb_data_format,
        in1_buffer_address);
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

    auto* dst_buffer = cache_tensor.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        // Index tensor args
        (std::uint32_t)use_index_tensor,
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
        cb_pagetable_id,
        St,
        in0_sequential_mode_semaphore_id,
    };
    TensorAccessorArgs(dst_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(update_idxs_tensor.has_value() ? update_idxs_tensor->buffer() : nullptr)
        .append_to(reader_compile_time_args);
    TensorAccessorArgs(page_table.has_value() ? page_table->buffer() : nullptr).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
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
    TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_kernel_args = {
        src0_cb_index,
        src1_cb_index,
        intermed0_cb_index,
        intermed1_cb_index,
        intermed2_cb_index,
        output_cb_index,
        Wt,
        num_heads,
    };

    auto unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "reader_update_cache_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/"
        "writer_update_cache_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/compute/update_cache.cpp",
        all_cores,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});

    const auto& cores = corerange_to_cores(all_cores, num_cores, row_major);
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t update_idx = use_index_tensor ? 0 : operation_attributes.update_idxs.at(i);
        // Cache tile info
        const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + ((update_idx / TILE_HEIGHT) * Wt);
        // Offset to write into untilized cache
        uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

        bool wait_to_start, send_signal;
        uint32_t send_core_x, send_core_y;
        if (operation_attributes.share_cache) {
            // Share cache
            wait_to_start = i != 0;
            send_signal = i != num_cores - 1;
            auto next_core = i == num_cores - 1 ? core : cores.at(i + 1);
            auto next_core_physical = device->worker_core_from_logical_core(next_core);
            send_core_x = next_core_physical.x;
            send_core_y = next_core_physical.y;
        } else {
            wait_to_start = false;
            send_signal = false;
            send_core_x = 0;
            send_core_y = 0;
        }

        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                dst_buffer->address(),
                use_index_tensor ? 0 : cache_start_id,
                index_buffer_addr,
                i,
                is_paged_cache ? page_table.value().buffer()->address() : 0,
                wait_to_start,
            });

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                use_index_tensor ? 0 : cache_start_id,
                use_index_tensor ? 0 : tile_update_offset_B,
                i,
                send_signal,
                send_core_x,
                send_core_y,
            });
    }

    // Store shared variables
    shared_variables_t shared_variables{
        .unary_reader_kernel_id = unary_reader_kernel_id,
        .unary_writer_kernel_id = unary_writer_kernel_id,
        .cores = cores,
        .Wbytes = Wbytes,
        .Wt = Wt,
        .cb_src1 = cb_src1,
        .cache_batch_num_tiles = cache_batch_num_tiles,
        .use_index_tensor = use_index_tensor,
        .is_paged_cache = is_paged_cache};

    return cached_program_t(std::move(program), std::move(shared_variables));
}

PagedUpdateCacheMeshWorkloadFactory::cached_mesh_workload_t PagedUpdateCacheMeshWorkloadFactory::create_mesh_workload(
    const PagedUpdateCacheParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& tensor_return_value) {
    log_debug(tt::LogOp, "PagedUpdateCacheMeshWorkloadFactory::create_mesh_workload called");
    log_debug(tt::LogOp, "tensor_coords has {} ranges", tensor_coords.ranges().size());

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // Filter coordinates based on mesh_coords if provided
    const std::optional<std::set<ttnn::MeshCoordinate>>& mesh_coords_opt = operation_attributes.mesh_coords;

    if (mesh_coords_opt.has_value()) {
        log_debug(tt::LogOp, "mesh_coords provided with {} coordinates", mesh_coords_opt.value().size());
    } else {
        log_debug(tt::LogOp, "mesh_coords not provided, using all tensor_coords");
    }

    // Create programs for each coordinate in tensor_coords (filtered by mesh_coords if provided)
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            // Skip this coordinate if mesh_coords is provided and this coordinate is not in the set
            if (mesh_coords_opt.has_value()) {
                const auto& mesh_coords_set = mesh_coords_opt.value();
                if (!mesh_coords_set.contains(mesh_coord)) {
                    log_debug(
                        tt::LogOp, "Skipping coordinate ({}, {}) - not in mesh_coords", mesh_coord[0], mesh_coord[1]);
                    continue;  // Skip this coordinate
                }
            }

            // Create a program for this specific coordinate using the base factory
            log_debug(tt::LogOp, "Creating program for coordinate ({}, {})", mesh_coord[0], mesh_coord[1]);
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto cached_program =
                PagedUpdateCacheProgramFactory::create(operation_attributes, tensor_args, tensor_return_value);
            shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
            mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        }
    }

    log_debug(tt::LogOp, "Created mesh workload with {} programs", mesh_workload.get_programs().size());
    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void PagedUpdateCacheProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    auto* src_buffer = tensor_args.input_tensor.buffer();
    auto* dst_buffer = tensor_args.cache_tensor.buffer();

    auto index_tensor_addr =
        shared_vars.use_index_tensor ? tensor_args.update_idxs_tensor.value().buffer()->address() : 0;
    auto page_table_tensor_addr = shared_vars.is_paged_cache ? tensor_args.page_table.value().buffer()->address() : 0;

    if (tensor_args.input_tensor.is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_src1, *src_buffer);
    }

    auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.unary_reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.unary_writer_kernel_id);

    for (uint32_t i = 0; i < shared_vars.cores.size(); ++i) {
        const uint32_t update_idx = shared_vars.use_index_tensor ? 0 : operation_attributes.update_idxs.at(i);
        // Cache tile info
        const uint32_t cache_batch_tile_offset = i * shared_vars.cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + ((update_idx / TILE_HEIGHT) * shared_vars.Wt);
        // Offset to write into untilized cache
        uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * shared_vars.Wbytes;

        const CoreCoord& core = shared_vars.cores.at(i);

        {
            auto& runtime_args = reader_args_by_core.at(core.x).at(core.y);
            runtime_args[0] = dst_buffer->address();
            runtime_args[1] = cache_start_id;
            runtime_args[2] = index_tensor_addr;
            runtime_args[4] = page_table_tensor_addr;
        }

        {
            auto& runtime_args = writer_args_by_core.at(core.x).at(core.y);
            runtime_args[0] = dst_buffer->address();
            runtime_args[1] = cache_start_id;
            runtime_args[2] = tile_update_offset_B;
        }
    }
}

void PagedUpdateCacheMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const PagedUpdateCacheParams& operation_attributes,
    const PagedUpdateCacheInputs& tensor_args,
    Tensor& tensor_return_value) {
    PagedUpdateCacheProgramFactory program_factory;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_variables = cached_workload.shared_variables.at(coordinate_range);

        ttnn::device_operation::mesh_device_operation_utils::apply_override_runtime_arguments(
            program_factory,
            program,
            shared_variables,
            operation_attributes,
            *(coordinate_range.begin()),
            tensor_args,
            tensor_return_value);
    }
}

}  // namespace ttnn::experimental::prim
