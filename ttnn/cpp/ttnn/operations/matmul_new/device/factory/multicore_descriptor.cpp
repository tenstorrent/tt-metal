// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multicore_descriptor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::constants;

namespace ttnn::prim::matmul_new_detail {

// ---- MeshWorkloadFactoryConcept methods ----

MultiCoreDescriptorFactory::cached_mesh_workload_t MultiCoreDescriptorFactory::create_mesh_workload(
    const MatmulParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // Extract grid info needed for override_runtime_arguments
    auto* device = tensor_args.input_tensors.at(0).device();
    auto compute_grid = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_grid.y;
    const auto& cshape = tensor_return_value.at(0).padded_shape();
    uint32_t c_batch_size = get_batch_size(cshape);
    auto num_output_tiles_total = c_batch_size * cshape[-2] * cshape[-1] / TILE_HW;
    auto [num_cores, all_cores, cg1, cg2, npt1, npt2] =
        tt::tt_metal::split_work_to_cores(compute_grid, num_output_tiles_total);

    for (const auto& range : tensor_coords.ranges()) {
        auto desc = create_descriptor(operation_attributes, tensor_args, tensor_return_value);
        tt::tt_metal::Program program{desc};
        mesh_workload.add_program(range, std::move(program));
        shared_variables[range] = shared_variables_t{
            .num_cores = static_cast<uint32_t>(num_cores),
            .num_cores_y = num_cores_y,
        };
    }
    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void MultiCoreDescriptorFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const MatmulParams& /*operation_attributes*/,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    // Only buffer addresses change between invocations.
    // Reader kernel (handle 0): runtime_args[0] = src0_addr, [1] = src1_addr
    // Writer kernel (handle 1): runtime_args[0] = dst_addr
    uint32_t src0_addr = tensor_args.input_tensors.at(0).buffer()->address();
    uint32_t src1_addr = tensor_args.input_tensors.at(1).buffer()->address();
    uint32_t dst_addr = tensor_return_value.at(0).buffer()->address();

    for (auto& [coord_range, program] : cached_workload.workload.get_programs()) {
        auto& sv = cached_workload.shared_variables.at(coord_range);
        for (uint32_t i = 0; i < sv.num_cores; i++) {
            CoreCoord core = {i / sv.num_cores_y, i % sv.num_cores_y};
            {
                auto& args = tt::tt_metal::GetRuntimeArgs(program, 0, core);
                args[0] = src0_addr;
                args[1] = src1_addr;
            }
            {
                auto& args = tt::tt_metal::GetRuntimeArgs(program, 1, core);
                args[0] = dst_addr;
            }
        }
    }
}

// ---- ProgramDescriptor construction (used by create_mesh_workload) ----

tt::tt_metal::ProgramDescriptor MultiCoreDescriptorFactory::create_descriptor(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace tt::tt_metal;

    if (!tensor_args.optional_input_tensors.empty()) {
        TT_FATAL(!tensor_args.optional_input_tensors[0].has_value(), "Bias is not supported for matmul multi core");
    }

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    auto& output = tensor_return_value.at(0);

    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    bool bcast_batch = operation_attributes.bcast_batch.value();

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::DataFormat in0_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    tt_metal::Buffer* src0_buffer = a.buffer();
    tt_metal::Buffer* src1_buffer = b.buffer();

    tt::tt_metal::IDevice* device = a.device();
    const auto& cshape = output.padded_shape();

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

    // ---- Build ProgramDescriptor ----

    ProgramDescriptor desc;

    // Circular buffers
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * in0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
        }}},
    });

    uint32_t src1_cb_index = 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * in1_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size,
        }}},
    });

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    // Reader kernel
    uint32_t last_ktile_w = a.logical_shape()[-1] % TILE_WIDTH;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)last_ktile_w};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp";
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    // Compute kernel - group 1
    std::vector<uint32_t> compute_args_group_1 = {
        1,                                 // B
        1,                                 // Mt
        Kt,                                // Kt
        num_output_tiles_per_core_group_1  // Nt
    };

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp";
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = compute_args_group_1;
    compute_desc_1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .dst_full_sync_en = true,
    };

    // Compute kernel - group 2 (may be empty)
    bool has_group_2 = !core_group_2.ranges().empty();
    KernelDescriptor compute_desc_2;
    if (has_group_2) {
        std::vector<uint32_t> compute_args_group_2 = {
            1,                                 // B
            1,                                 // Mt
            Kt,                                // Kt
            num_output_tiles_per_core_group_2  // Nt
        };

        compute_desc_2.kernel_source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm.cpp";
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = compute_args_group_2;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .dst_full_sync_en = true,
        };
    }

    // Runtime args per core
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

        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                src0_addr,
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

        writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{dst_addr, num_output_tiles_per_core, num_tiles_written});

        num_tiles_written += num_output_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::prim::matmul_new_detail
