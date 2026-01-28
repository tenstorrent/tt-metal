// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/where/device/program_factory/element_wise_multi_core_where_program.hpp"
#include "ttnn/operations/experimental/where/device/program_factory/elemwise_factory_common.hpp"
#include "ttnn/operations/experimental/where/device/kernels/dataflow/elemwise_reader_kernel_args.hpp"
#include "ttnn/operations/experimental/where/device/kernels/dataflow/elemwise_writer_kernel_args.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <tt-metalium/work_split.hpp>
namespace ttnn::experimental::prim {

ElementWiseMultiCoreWhereProgram::cached_program_t ElementWiseMultiCoreWhereProgram::create(
    const WhereParams& operation_attributes, const WhereInputs& args, Tensor& output) {
    using namespace ttnn::kernel::eltwise::where_args;
    using namespace tt;
    using namespace tt::tt_metal;

    TT_FATAL(
        args.condition_tensor.dtype() == args.true_value_tensor.dtype(),
        "Mismatched data types: 'condition_tensor' and 'true_values_tensor' must have the same dtype.");

    TT_FATAL(
        args.condition_tensor.dtype() == args.false_value_tensor.dtype(),
        "Mismatched data types: 'condition_tensor' and 'false_values_tensor' must have the same dtype.");

    TT_FATAL(
        args.condition_tensor.dtype() == output.dtype(),
        "Mismatched data types: 'condition_tensor' and 'output' tensor must have the same dtype.");

    TT_FATAL(
        args.condition_tensor.dtype() == DataType::BFLOAT16 || args.condition_tensor.dtype() == DataType::FLOAT32 ||
            args.condition_tensor.dtype() == DataType::BFLOAT8_B,
        "Invalid data type: expected BFLOAT16 or FLOAT32 or BFLOAT8_B for 'condition_tensor'.");

    Program program{};
    const auto& all_device_cores = operation_attributes.worker_grid;
    auto dtype = tt_metal::datatype_to_dataformat_converter(args.condition_tensor.dtype());

    auto createCircularBuffer = [&program, &all_device_cores, dtype = dtype](
                                    tt::CBIndex cb_idx, uint32_t tile_size, uint32_t num_input_tiles = 1) {
        auto cb_config = tt::tt_metal::CircularBufferConfig(num_input_tiles * tile_size, {{cb_idx, dtype}})
                             .set_page_size(cb_idx, tile_size);
        return tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_config);
    };

    uint32_t single_tile_size = tt::tile_size(dtype);
    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    auto condition_cb = tt::CBIndex::c_0;
    auto true_values_cb = tt::CBIndex::c_1;
    auto false_values_cb = tt::CBIndex::c_2;
    auto tmp_cb = tt::CBIndex::c_3;
    auto output_cb_index = tt::CBIndex::c_4;

    CBHandle condition_cb_handle = createCircularBuffer(condition_cb, single_tile_size);
    CBHandle true_values_cb_handle = createCircularBuffer(true_values_cb, single_tile_size);
    CBHandle false_values_cb_handle = createCircularBuffer(false_values_cb, single_tile_size);

    /* Temporary buffers to hold intermediate results */
    createCircularBuffer(tmp_cb, single_tile_size);

    constexpr uint32_t num_output_tiles = 2;
    CBHandle cb_output = createCircularBuffer(output_cb_index, single_tile_size, num_output_tiles);

    CompileTimeReaderKernelArgs reader_compile_time_args = {
        .condition_cb = condition_cb, .true_tensor_cb = true_values_cb, .false_tensor_cb = false_values_cb};

    /* Specify data movement kernels for reading/writing data to/from DRAM */
    std::map<std::string, std::string> reader_defines;
    std::vector<uint32_t> reader_compile_time_vec = ttnn::kernel_utils::to_vector(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(args.condition_tensor.buffer()).append_to(reader_compile_time_vec);
    tt::tt_metal::TensorAccessorArgs(args.true_value_tensor.buffer()).append_to(reader_compile_time_vec);
    tt::tt_metal::TensorAccessorArgs(args.false_value_tensor.buffer()).append_to(reader_compile_time_vec);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/where/device/kernels/dataflow/elemwise_reader_kernel.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_vec, reader_defines));

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    CompileTimeWriterKernelArgs writer_compile_time_args = {.cb_dst = output_cb_index};
    std::map<std::string, std::string> writer_defines;
    std::vector<uint32_t> writer_compile_time_vec = ttnn::kernel_utils::to_vector(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_vec);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/where/device/kernels/dataflow/elemwise_writer_kernel.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_vec, writer_defines));

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/where/device/kernels/compute/elemwise_where_kernel.cpp",
        all_device_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {}});

    set_eltwise_ternary_runtime_args<true>(
        program,
        args.condition_tensor,
        args.true_value_tensor,
        args.false_value_tensor,
        output,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        all_device_cores);
    return {
        std::move(program),
        {reader_kernel_id,
         writer_kernel_id,
         compute_kernel_id,
         condition_cb_handle,
         true_values_cb_handle,
         false_values_cb_handle,
         cb_output,
         all_device_cores,
         single_tile_size,
         single_tile_size,
         single_tile_size,
         single_tile_size}};
}

void ElementWiseMultiCoreWhereProgram::override_runtime_arguments(
    cached_program_t& cached_program,
    const WhereParams& /*operation_attributes*/,
    const WhereInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& sh_var = cached_program.shared_variables;
    set_eltwise_ternary_runtime_args<false>(
        cached_program.program,
        tensor_args.condition_tensor,
        tensor_args.true_value_tensor,
        tensor_args.false_value_tensor,
        tensor_return_value,
        sh_var.reader_kernel_id,
        sh_var.writer_kernel_id,
        sh_var.eltwise_kernel_id,
        sh_var.all_device_cores);
}
}  // namespace ttnn::experimental::prim
