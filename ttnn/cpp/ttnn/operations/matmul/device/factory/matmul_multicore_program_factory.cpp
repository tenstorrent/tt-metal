// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_program_factory.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::constants;

namespace ttnn::prim {

MatmulMultiCoreProgramFactory::cached_program_t MatmulMultiCoreProgramFactory::create(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    if (!tensor_args.optional_input_tensors.empty()) {
        TT_FATAL(!tensor_args.optional_input_tensors[0].has_value(), "Bias is not supported for matmul multi core");
    }

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    auto& output = tensor_return_value.at(0);

    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    bool bcast_batch = operation_attributes.bcast_batch.value();

    tt_metal::Program program{};

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    tt_metal::Buffer* src0_buffer = a.buffer();
    tt_metal::Buffer* src1_buffer = b.buffer();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();
    const auto& cshape = output.padded_shape();  // C=A*B, N1MK*11KN->N1MN

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t c_batch_size = get_batch_size(cshape);
    auto num_output_tiles_total = c_batch_size * cshape[-2] * cshape[-1] / TILE_HW;
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles_total);

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // C = A*B*...
    // MN = MK*KN
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    uint32_t src0_addr = src0_buffer->address();
    uint32_t src1_addr = src1_buffer->address();
    uint32_t dst_addr = dst_buffer->address();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(num_input_tiles * in0_single_tile_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(num_input_tiles * in1_single_tile_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    uint32_t last_ktile_w = a.logical_shape()[-1] % TILE_WIDTH;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)last_ktile_w};
    tt::tt_metal::TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    auto reader_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args_group_1 = {
        1,                                 // B
        1,                                 // Mt
        Kt,                                // Kt
        num_output_tiles_per_core_group_1  // Nt
    };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set
        // Nt for simplicity

    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .dst_full_sync_en = true, .compile_args = compute_args_group_1});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_group_2 = {
            1,                                 // B
            1,                                 // Mt
            Kt,                                // Kt
            num_output_tiles_per_core_group_2  // Nt
        };  // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only
            // set Nt for simplicity

        tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp",
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity, .dst_full_sync_en = true, .compile_args = compute_args_group_2});
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_output_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        tt_metal::SetRuntimeArgs(
            program,
            reader_id,
            core,
            {src0_addr,
             src1_addr,
             Mt,
             Kt,
             Nt,
             MtKt,
             KtNt,
             B,
             uint32_t(bcast_batch),
             num_tiles_written,
             num_output_tiles_per_core,
             MtNt});
        tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, num_output_tiles_per_core, num_tiles_written});
        num_tiles_written += num_output_tiles_per_core;
    }

    return {std::move(program), {reader_id, writer_id, num_cores, num_cores_y}};
}

void MatmulMultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::MatmulParams& /*operation_attributes*/,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto reader_kernel_id = shared_variables.reader_kernel_id;
    auto writer_kernel_id = shared_variables.writer_kernel_id;
    auto num_cores = shared_variables.num_cores;
    auto num_cores_y = shared_variables.num_cores_y;

    auto* src_dram_buffer_a = tensor_args.input_tensors.at(0).buffer();
    auto* src_dram_buffer_b = tensor_args.input_tensors.at(1).buffer();
    auto* dst_dram_buffer = tensor_return_value.at(0).buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer_a->address();
            runtime_args[1] = src_dram_buffer_b->address();
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
        }
    }
}

////////////////////////////////////////////////////////////////////////////
//                      Mesh Workload Setup
////////////////////////////////////////////////////////////////////////////

MatmulMeshWorkloadMultiCoreFactory::cached_mesh_workload_t MatmulMeshWorkloadMultiCoreFactory::create_mesh_workload(
    const ttnn::prim::MatmulParams& attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& output) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange mesh_coord_range{mesh_coord, mesh_coord};
            auto single_device_program = MatmulMultiCoreProgramFactory::create(attributes, tensor_args, output);
            shared_variables[mesh_coord_range] = single_device_program.shared_variables;
            workload.add_program(mesh_coord_range, std::move(single_device_program.program));
        }
    }
    return {std::move(workload), std::move(shared_variables)};
}

void MatmulMeshWorkloadMultiCoreFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ttnn::prim::MatmulParams& attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    for (auto& [mesh_coord_range, program] : cached_workload.workload.get_programs()) {
        auto cached_program_proxy = MatmulMultiCoreProgramFactory::cached_program_t::proxy(
            program, cached_workload.shared_variables.at(mesh_coord_range));
        MatmulMultiCoreProgramFactory::override_runtime_arguments(
            cached_program_proxy, attributes, tensor_args, tensor_return_value);
    }
}

}  // namespace ttnn::prim
