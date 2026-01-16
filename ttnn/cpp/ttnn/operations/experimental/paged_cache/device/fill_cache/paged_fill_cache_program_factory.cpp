// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "paged_fill_cache_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include "paged_fill_cache_program_factory.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/mesh_device_operation_utils.hpp"
#include <unordered_map>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::paged_cache::fill::program {

using namespace tt::constants;
using namespace tt;

PagedFillCacheProgramFactory::cached_program_t PagedFillCacheProgramFactory::create(
    const FillParams& operation_attributes, const FillInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    Program program{};

    const auto& cache_tensor = tensor_args.cache_tensor;
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& page_table_tensor = tensor_args.page_table;
    const auto& batch_idx_tensor = tensor_args.batch_idx_tensor_opt;

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    // input_tensor: [1, num_heads, input_seq_len, head_dim]
    // cache_tensor: [max_num_blocks, 1, block_size, head_dim]
    // page_table_tensor: [b, max_num_blocks_per_seq]
    const uint32_t num_heads = input_tensor.padded_shape()[1];
    const uint32_t input_seq_len = input_tensor.padded_shape()[2];

    const uint32_t block_size = cache_tensor.padded_shape()[2];
    const uint32_t head_dim = cache_tensor.padded_shape()[3];

    const uint32_t input_seq_len_t = input_seq_len / TILE_HEIGHT;
    const uint32_t Wt = head_dim / TILE_WIDTH;
    const uint32_t block_size_t = block_size / TILE_HEIGHT;

    uint32_t num_blocks_of_work = num_heads * input_seq_len_t;
    uint32_t num_blocks_of_work_per_head = input_seq_len_t;

    // Pagetable-specific parameters
    uint32_t page_table_stick_size_B = page_table_tensor.buffer()->aligned_page_size();
    TT_FATAL(
        page_table_stick_size_B % 32 == 0,
        "page table page size in bytes must be a multiple of 32 due to address alignment");
    uint32_t log2_page_table_stick_size_B = std::log2(page_table_stick_size_B);
    tt::DataFormat page_table_data_format = tt_metal::datatype_to_dataformat_converter(page_table_tensor.dtype());

    // batch_idx_tensor specific parameters
    bool use_batch_idx_tensor = batch_idx_tensor.has_value();
    uint32_t batch_idx_buffer_addr = 0;
    tt::DataFormat batch_idx_data_format = tt::DataFormat::UInt32;  // Assuming batch_idx is uint32
    uint32_t batch_idx_stick_size_B = 4;                            // Assuming scalar uint32

    if (use_batch_idx_tensor) {
        const auto& tensor = batch_idx_tensor.value();
        batch_idx_buffer_addr = tensor.buffer()->address();
        batch_idx_data_format = tt_metal::datatype_to_dataformat_converter(tensor.dtype());
        batch_idx_stick_size_B = tensor.element_size();
        TT_FATAL(tensor.physical_volume() == 1, "batch_idx_tensor must contain a single element.");
    }

    tt_metal::IDevice* device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    row_major = true;
    std::tie(
        num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2) =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_of_work, row_major);
    uint32_t num_input_tiles = Wt * 2;  // double buffered

    tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
    tt::CBIndex page_table_cb_index = tt::CBIndex::c_1;
    tt::CBIndex cb_batch_idx_id = tt::CBIndex::c_2;  // New CB for batch_idx_tensor

    create_cb(src0_cb_index, program, all_cores, single_tile_size, num_input_tiles, cb_data_format);
    create_cb(page_table_cb_index, program, all_cores, page_table_stick_size_B, 1, page_table_data_format);
    if (use_batch_idx_tensor) {
        create_cb(cb_batch_idx_id, program, all_cores, batch_idx_stick_size_B, 1, batch_idx_data_format);
    }

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = cache_tensor.buffer();
    auto* page_table_buffer = page_table_tensor.buffer();

    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index, Wt};
    TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)src0_cb_index,
        (uint32_t)page_table_cb_index,
        num_heads,
        num_blocks_of_work_per_head,
        block_size_t,
        Wt,
        log2_page_table_stick_size_B,
        page_table_stick_size_B,
        // New compile-time args for batch_idx_tensor
        (uint32_t)use_batch_idx_tensor,
        cb_batch_idx_id,        // Meaningful only if use_batch_idx_tensor is true
        batch_idx_stick_size_B  // Meaningful only if use_batch_idx_tensor is true
    };
    TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(page_table_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(batch_idx_tensor.has_value() ? batch_idx_tensor->buffer() : nullptr)
        .append_to(writer_compile_time_args);

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_fill_cache_interleaved.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_fill_cache_interleaved.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_blocks_per_core = 0;
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (i < g1_numcores + g2_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            num_blocks_per_core = 0;
        }

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src_buffer->address(),
                num_blocks_written * Wt,              // start_tile_id
                num_blocks_per_core,                  // num_rows
                (uint32_t)operation_attributes.noop,  // noop flag
            });

        uint32_t writer_batch_arg =
            use_batch_idx_tensor ? batch_idx_buffer_addr : operation_attributes.batch_idx_fallback;

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                page_table_buffer->address(),
                num_blocks_written,                   // start_row_num
                num_blocks_per_core,                  // num_rows
                writer_batch_arg,                     // batch_idx_tensor_addr or batch_idx_fallback
                (uint32_t)operation_attributes.noop,  // noop flag
            });
        num_blocks_written += num_blocks_per_core;
    }

    // Store shared variables
    shared_variables_t shared_variables{
        .unary_reader_kernel_id = unary_reader_kernel_id,
        .unary_writer_kernel_id = unary_writer_kernel_id,
        .cores = cores,
        .g1_numcores = g1_numcores,
        .g2_numcores = g2_numcores,
        .num_blocks_per_core_group_1 = num_blocks_per_core_group_1,
        .num_blocks_per_core_group_2 = num_blocks_per_core_group_2,
        .Wt = Wt,
        .use_batch_idx_tensor = use_batch_idx_tensor};

    return cached_program_t(std::move(program), std::move(shared_variables));
}

void PagedFillCacheProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FillParams& operation_attributes,
    const FillInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    auto dst_addr = tensor_args.cache_tensor.buffer()->address();
    auto src_addr = tensor_args.input_tensor.buffer()->address();
    auto page_table_addr = tensor_args.page_table.buffer()->address();

    uint32_t current_kernel_batch_arg;
    if (shared_vars.use_batch_idx_tensor) {
        TT_FATAL(
            tensor_args.batch_idx_tensor_opt.has_value(),
            "batch_idx_tensor_opt is expected but not provided when use_batch_idx_tensor is true.");
        current_kernel_batch_arg = tensor_args.batch_idx_tensor_opt.value().buffer()->address();
    } else {
        current_kernel_batch_arg = operation_attributes.batch_idx_fallback;
    }

    auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.unary_reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.unary_writer_kernel_id);

    for (uint32_t i = 0, num_blocks_written = 0; i < shared_vars.cores.size(); i++) {
        const CoreCoord& core = shared_vars.cores.at(i);
        uint32_t num_blocks_per_core = 0;
        if (i < shared_vars.g1_numcores) {
            num_blocks_per_core = shared_vars.num_blocks_per_core_group_1;
        } else if (i < shared_vars.g1_numcores + shared_vars.g2_numcores) {
            num_blocks_per_core = shared_vars.num_blocks_per_core_group_2;
        } else {
            num_blocks_per_core = 0;
        }

        auto& reader_args = reader_args_by_core.at(core.x).at(core.y);
        reader_args[0] = src_addr;
        reader_args[1] = num_blocks_written * shared_vars.Wt;  // start_tile_id
        reader_args[2] = num_blocks_per_core;                  // num_rows
        reader_args[3] = (uint32_t)operation_attributes.noop;  // noop flag

        auto& writer_args = writer_args_by_core.at(core.x).at(core.y);
        writer_args[0] = dst_addr;
        writer_args[1] = page_table_addr;
        writer_args[2] = num_blocks_written;        // start_row_num
        writer_args[3] = num_blocks_per_core;       // num_rows
        writer_args[4] = current_kernel_batch_arg;  // batch_idx_tensor_addr or batch_idx_fallback
        writer_args[5] = (uint32_t)operation_attributes.noop;  // noop flag

        num_blocks_written += num_blocks_per_core;
    }
}

PagedFillCacheMeshWorkloadFactory::cached_mesh_workload_t PagedFillCacheMeshWorkloadFactory::create_mesh_workload(
    const FillParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const FillInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // Filter coordinates based on mesh_coords if provided
    const std::optional<std::set<ttnn::MeshCoordinate>>& mesh_coords_opt = operation_attributes.mesh_coords;

    if (mesh_coords_opt.has_value()) {
        // Validate that all mesh_coords are present in tensor_coords BEFORE creating programs
        const auto& mesh_coords_set = mesh_coords_opt.value();
        const auto tensor_coords_vector = tensor_coords.coords();
        std::set<ttnn::MeshCoordinate> tensor_coords_set(tensor_coords_vector.begin(), tensor_coords_vector.end());

        for (const auto& mesh_coord : mesh_coords_set) {
            TT_FATAL(
                tensor_coords_set.contains(mesh_coord),
                "Mesh coordinate ({}, {}) is in mesh_coords but not found in tensor_coords. "
                "mesh_coords size: {}, tensor_coords size: {}",
                mesh_coord[0],
                mesh_coord[1],
                mesh_coords_set.size(),
                tensor_coords_set.size());
        }
        for (const auto& mesh_coord : mesh_coords_set) {
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto cached_program =
                PagedFillCacheProgramFactory::create(operation_attributes, tensor_args, tensor_return_value);
            shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
            mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        }

        // Create dummy programs for excluded coordinates
        std::vector<ttnn::MeshCoordinate> dummy_coords;
        for (const auto& coord : tensor_coords_set) {
            if (!mesh_coords_set.contains(coord)) {
                dummy_coords.push_back(coord);
            }
        }

        if (!dummy_coords.empty()) {
            for (const auto& mesh_coord : dummy_coords) {
                const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
                // Create operation attributes with noop=true for dummy programs
                FillParams dummy_attrs{
                    .batch_idx_fallback = operation_attributes.batch_idx_fallback,
                    .mesh_coords = operation_attributes.mesh_coords,
                    .noop = true};
                auto cached_program =
                    PagedFillCacheProgramFactory::create(dummy_attrs, tensor_args, tensor_return_value);
                shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
                mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
            }
        }
    } else {
        // When mesh_coords is not provided, iterate over all tensor_coords
        for (const auto& mesh_coord_range : tensor_coords.ranges()) {
            for (const auto& mesh_coord : mesh_coord_range) {
                const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
                auto cached_program =
                    PagedFillCacheProgramFactory::create(operation_attributes, tensor_args, tensor_return_value);
                shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
                mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
            }
        }
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void PagedFillCacheMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const FillParams& operation_attributes,
    const FillInputs& tensor_args,
    Tensor& tensor_return_value) {
    PagedFillCacheProgramFactory program_factory;

    // Determine which coordinates should have noop=true (excluded from mesh_coords)
    std::set<ttnn::MeshCoordinate> mesh_coords_set;
    if (operation_attributes.mesh_coords.has_value()) {
        mesh_coords_set = operation_attributes.mesh_coords.value();
    }

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_variables = cached_workload.shared_variables.at(coordinate_range);
        const ttnn::MeshCoordinate coord = *(coordinate_range.begin());

        // Determine if this coordinate should be a noop (dummy program)
        // If mesh_coords is provided and this coord is not in it, it's a dummy program
        bool is_dummy = operation_attributes.mesh_coords.has_value() && !mesh_coords_set.contains(coord);

        // Create modified operation_attributes with correct noop value for this coordinate
        FillParams coord_attrs{
            .batch_idx_fallback = operation_attributes.batch_idx_fallback,
            .mesh_coords = operation_attributes.mesh_coords,
            .noop = is_dummy};

        ttnn::device_operation::mesh_device_operation_utils::apply_override_runtime_arguments(
            program_factory, program, shared_variables, coord_attrs, coord, tensor_args, tensor_return_value);
    }
}

}  // namespace ttnn::operations::experimental::paged_cache::fill::program
