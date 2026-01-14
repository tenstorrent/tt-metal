// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_single_core_hw_program_factory.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cmath>

using namespace tt::constants;

namespace ttnn::operations::reduction::generic::program {

ReduceSingleCoreHwProgramFactory::cached_program_t ReduceSingleCoreHwProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const auto& a = tensor_args.input_tensor;
    auto& output = tensor_return_value;
    const auto& shape = a.padded_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    float scaler = std::sqrt(operation_attributes.scaler);

    uint32_t num_tensor_tiles = NC * H * W / TILE_HW;

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord selected_core_coord = {0, 0};
    if (operation_attributes.sub_core_grids.has_value() && !operation_attributes.sub_core_grids->ranges().empty()) {
        const auto& r = operation_attributes.sub_core_grids->ranges().front();
        selected_core_coord = r.start_coord;
    }
    CoreRange core(selected_core_coord, selected_core_coord);

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);

    // Scaler datatype is hardcoded bfloat16 due to tile creation in reader
    tt::DataFormat scaler_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::Buffer* src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    tt_metal::CircularBufferConfig cb_scaler_config =
        tt_metal::CircularBufferConfig(
            num_input_tiles * scaler_single_tile_size, {{CBIndex::c_2, scaler_cb_data_format}})
            .set_page_size(CBIndex::c_2, scaler_single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_scaler_config);

    uint32_t output_cb_index = tt::CBIndex::c_3;
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    bfloat16 bfloat_scaler_value = bfloat16::truncate(scaler);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    std::vector<uint32_t> reader_compile_time_args = {packed_scaler_value};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
        "reader_unary_reduce_universal_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args = {
        Ht,  // Ht
        Wt,  // Wt
        NC,  // NC
    };

    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_hw.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args,
            .defines = reduce_op_utils::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::HW)});

    tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, {a.buffer()->address(), num_tensor_tiles, 0});

    uint32_t out_dim_divider = Ht * Wt;

    tt_metal::SetRuntimeArgs(
        program, writer_kernel_id, core, {output.buffer()->address(), num_tensor_tiles / out_dim_divider, 0});

    return {std::move(program), {reader_kernel_id, writer_kernel_id, selected_core_coord}};
}

void ReduceSingleCoreHwProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    auto* src_dram_buffer = tensor_args.input_tensor.buffer();
    auto* dst_dram_buffer = tensor_return_value.buffer();
    CoreCoord core = cached_program.shared_variables.selected_core_coord;

    {
        auto& runtime_args =
            GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id, core);
        runtime_args[0] = src_dram_buffer->address();
    }

    {
        auto& runtime_args =
            GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id, core);
        runtime_args[0] = dst_dram_buffer->address();
    }
}

}  // namespace ttnn::operations::reduction::generic::program
