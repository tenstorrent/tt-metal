// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "paged_fused_update_cache_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include "paged_tiled_fused_update_cache_program_factory.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/mesh_device_operation_utils.hpp"
#include <unordered_map>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::paged_cache::fused_update::program::tiled {

using namespace tt::constants;
using namespace tt;

namespace CMAKE_UNIQUE_NAMESPACE {

bool enable_fp32_dest_acc(
    const tt_metal::IDevice* device, const ttnn::DeviceComputeKernelConfig& compute_kernel_config) {
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    return fp32_dest_acc_en;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

PagedTiledFusedUpdateCacheProgramFactory::cached_program_t PagedTiledFusedUpdateCacheProgramFactory::create(
    const FusedUpdateParams& operation_attributes,
    const FusedUpdateInputs& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    Program program{};

    const auto& cache_tensor1 = tensor_args.cache_tensor1;
    const auto& input_tensor1 = tensor_args.input_tensor1;
    const auto& cache_tensor2 = tensor_args.cache_tensor2;
    const auto& input_tensor2 = tensor_args.input_tensor2;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    tt_metal::IDevice* device = input_tensor1.device();

    tt::DataFormat cache_cb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor1.dtype());
    uint32_t cache_single_tile_size = tt::tile_size(cache_cb_data_format);

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor1.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    bool fp32_dest_acc_en =
        CMAKE_UNIQUE_NAMESPACE::enable_fp32_dest_acc(device, operation_attributes.compute_kernel_config);

    tt::DataFormat interm_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt::tile_size(interm_cb_data_format);

    const uint32_t B = input_tensor1.padded_shape()[1];
    const uint32_t num_heads = cache_tensor1.padded_shape()[1];

    // Index tensor-specific parameters
    bool use_index_tensor = update_idxs_tensor.has_value();
    uint32_t index_buffer_addr = 0;
    uint32_t log2_page_size = 0;
    uint32_t index_stick_size = 0;
    tt::DataFormat index_data_format = tt::DataFormat::Int32;
    bool index_is_dram = true;
    Buffer* index_buffer_ptr = nullptr;
    if (use_index_tensor) {
        index_buffer_ptr = update_idxs_tensor.value().is_sharded() ? update_idxs_tensor.value().buffer() : nullptr;
        index_buffer_addr = use_index_tensor ? update_idxs_tensor.value().buffer()->address() : 0;
        index_data_format = tt_metal::datatype_to_dataformat_converter(update_idxs_tensor.value().dtype());
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
    uint32_t num_pages_page_table = 1;
    tt::DataFormat page_table_data_format = tt::DataFormat::Int32;
    bool page_table_is_dram = true;
    Buffer* page_table_buffer_ptr = nullptr;
    if (is_paged_cache) {
        const auto& page_table_tensor = page_table.value();
        page_table_buffer_ptr = page_table.value().is_sharded() ? page_table_tensor.buffer() : nullptr;
        num_pages_page_table = page_table.value().is_sharded() ? B : 1;
        block_size = cache_tensor1.padded_shape()[2];
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        page_table_stick_size = page_table.value().buffer()->aligned_page_size();
        page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.dtype());
        page_table_is_dram = page_table_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    }

    uint32_t Wt = cache_tensor1.padded_shape()[-1] / TILE_WIDTH;
    uint32_t St = cache_tensor1.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor1.padded_shape()[-1] * sizeof(float)
                                       : cache_tensor1.padded_shape()[-1] * 2;  // 2 bytes for bfloat16
    uint32_t cache_total_num_tiles = cache_tensor1.physical_volume() / TILE_HW;
    uint32_t cache_batch_num_tiles =
        operation_attributes.share_cache
            ? 0
            : cache_total_num_tiles /
                  cache_tensor1.padded_shape()[0];  // if share cache, we can set cache batch num tiles to 0
                                                    // so batch offset would be 0 in future calculations

    log_debug(tt::LogOp, "cache_cb_data_format: {}", cache_cb_data_format);
    log_debug(tt::LogOp, "input_cb_data_format: {}", input_cb_data_format);
    log_debug(tt::LogOp, "interm_cb_data_format: {}", interm_cb_data_format);
    log_debug(tt::LogOp, "Wbytes: {}", Wbytes);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "St: {}", St);

    const std::optional<ShardSpec>& input1_shard_spec = input_tensor1.shard_spec();
    const std::optional<ShardSpec>& input2_shard_spec = input_tensor2.shard_spec();
    bool row_major = input1_shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    CoreRangeSet input1_cores = input1_shard_spec.value().grid;
    CoreRangeSet input2_cores = input2_shard_spec.value().grid;
    CoreRangeSet all_cores = input1_cores.merge(input2_cores);
    CoreRangeSet all_cores_bb = all_cores.bounding_box();
    CoreRangeSet unused_cores = all_cores_bb.subtract(all_cores);

    uint32_t num_input_tiles = input1_shard_spec.value().shape[0] * input1_shard_spec.value().shape[1] / TILE_HW;

    auto* in1_buffer_address = input1_shard_spec.has_value() ? input_tensor1.buffer() : nullptr;

    auto* in2_buffer_address = input2_shard_spec.has_value() ? input_tensor2.buffer() : nullptr;

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

    CBHandle cb_cur_pos_id = 0;
    if (use_index_tensor) {
        auto [_3, cb_src5] =
            create_cb(cb_index_id, program, all_cores_bb, index_stick_size, 1, index_data_format, index_buffer_ptr);
        cb_cur_pos_id = cb_src5;
    }

    CBHandle cb_page_table_id = 0;
    if (is_paged_cache) {
        auto [_4, cb_src7] = create_cb(
            cb_pagetable_id,
            program,
            all_cores_bb,
            page_table_stick_size,
            num_pages_page_table,
            page_table_data_format,
            page_table_buffer_ptr);
        cb_page_table_id = cb_src7;
    }

    auto* dst1_buffer = cache_tensor1.buffer();

    auto* dst2_buffer = cache_tensor2.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
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
        B};
    TensorAccessorArgs(dst1_buffer).append_to(reader_compile_time_args);
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
        B,
        page_table_stick_size,
        page_table_is_dram};
    TensorAccessorArgs(dst1_buffer).append_to(writer_compile_time_args);

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

        const uint32_t update_idx = use_index_tensor ? 0 : operation_attributes.update_idxs.at(i);

        // Cache tile info
        const uint32_t cache_batch_tile_offset = i * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + ((update_idx / TILE_HEIGHT) * Wt);
        uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * Wbytes;

        // Calculate synchronization parameters
        bool wait_to_start = operation_attributes.share_cache and (i != 0);
        bool send_signal = operation_attributes.share_cache and (i != cores1.size() - 1);
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

    // Store shared variables
    shared_variables_t shared_vars{
        .unary_reader_kernel_id = unary_reader_kernel_id,
        .unary_writer_kernel_id = unary_writer_kernel_id,
        .cores1 = cores1,
        .cores2 = cores2,
        .Wbytes = Wbytes,
        .Wt = Wt,
        .cb_src1 = cb_src1,
        .cb_src3 = cb_src3,
        .cb_cur_pos_id = cb_cur_pos_id,
        .cb_page_table_id = cb_page_table_id,
        .cache_batch_num_tiles = cache_batch_num_tiles,
        .use_index_tensor = use_index_tensor,
        .is_paged_cache = is_paged_cache};

    return cached_program_t{std::move(program), std::move(shared_vars)};
}

void PagedTiledFusedUpdateCacheProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FusedUpdateParams& operation_attributes,
    const FusedUpdateInputs& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    const auto& input_tensor1 = tensor_args.input_tensor1;
    const auto& input_tensor2 = tensor_args.input_tensor2;
    const auto& update_idxs_tensor = tensor_args.update_idxs_tensor;
    const auto& page_table = tensor_args.page_table;

    auto* src1_buffer = input_tensor1.buffer();
    auto* src2_buffer = input_tensor2.buffer();

    auto* dst1_buffer = tensor_args.cache_tensor1.buffer();
    auto* dst2_buffer = tensor_args.cache_tensor2.buffer();

    auto* index_tensor_buffer = shared_vars.use_index_tensor ? update_idxs_tensor.value().buffer() : nullptr;
    auto* page_table_buffer = shared_vars.is_paged_cache ? page_table.value().buffer() : nullptr;
    auto index_tensor_addr = shared_vars.use_index_tensor ? update_idxs_tensor.value().buffer()->address() : 0;
    auto page_table_tensor_addr = shared_vars.is_paged_cache ? page_table.value().buffer()->address() : 0;

    if (input_tensor1.is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_src1, *src1_buffer);
    }
    if (input_tensor2.is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_src3, *src2_buffer);
    }
    auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.unary_reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.unary_writer_kernel_id);

    for (uint32_t i = 0; i < shared_vars.cores1.size(); ++i) {
        const uint32_t update_idx = shared_vars.use_index_tensor ? 0 : operation_attributes.update_idxs.at(i);
        // Cache tile info
        const uint32_t cache_batch_tile_offset = i * shared_vars.cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + ((update_idx / TILE_HEIGHT) * shared_vars.Wt);
        // Offset to write into untilized cache
        uint32_t tile_update_offset_B = update_idx % TILE_HEIGHT * shared_vars.Wbytes;

        const CoreCoord& core1 = shared_vars.cores1.at(i);
        const CoreCoord& core2 = shared_vars.cores2.at(i);

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
    if (shared_vars.use_index_tensor and update_idxs_tensor.has_value() and update_idxs_tensor.value().is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_cur_pos_id, *index_tensor_buffer);
    }
    if (shared_vars.is_paged_cache and page_table.has_value() and page_table.value().is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_page_table_id, *page_table_buffer);
    }
}

PagedTiledFusedUpdateCacheMeshWorkloadFactory::cached_mesh_workload_t
PagedTiledFusedUpdateCacheMeshWorkloadFactory::create_mesh_workload(
    const FusedUpdateParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const FusedUpdateInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    log_debug(tt::LogOp, "PagedTiledFusedUpdateCacheMeshWorkloadFactory::create_mesh_workload called");
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
            auto cached_program = PagedTiledFusedUpdateCacheProgramFactory::create(
                operation_attributes, tensor_args, tensor_return_value);
            shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
            mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        }
    }

    log_debug(tt::LogOp, "Created mesh workload with {} programs", mesh_workload.get_programs().size());
    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void PagedTiledFusedUpdateCacheMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const FusedUpdateParams& operation_attributes,
    const FusedUpdateInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    PagedTiledFusedUpdateCacheProgramFactory program_factory;

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

}  // namespace ttnn::operations::experimental::paged_cache::fused_update::program::tiled
