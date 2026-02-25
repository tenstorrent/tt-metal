// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/add/device/dram_optimized/dram_optimized_program.hpp"
#include "ttnn/operations/experimental/add/device/dram_optimized/elemwise_factory_common.hpp"
#include "ttnn/operations/experimental/add/device/dram_optimized/kernels/elemwise_args_kernel.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <tt-metalium/work_split.hpp>
namespace ttnn::experimental::prim {

std::string kernel_prefix = "ttnn/cpp/ttnn/operations/experimental/add/device/dram_optimized/kernels/";

ElementWiseMultiCoreAddProgram::cached_program_t ElementWiseMultiCoreAddProgram::create(
    const AddParams& operation_attributes, const AddInputs& args, Tensor& output) {
    using namespace eltwise_dram_optimized;
    using namespace tt;
    using namespace tt::tt_metal;

    CoreRangeSet worker_grid;
    auto* device = args.a_tensor.device();
    if (operation_attributes.sub_core_grids.has_value()) {
        worker_grid = operation_attributes.sub_core_grids.value();
    } else {
        std::cout << "sub_core_grids not provided, using all cores" << std::endl;
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers =
                device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id);
            worker_grid = worker_grid.merge(sub_device_workers);
        }
    }

    Program program{};
    const auto& all_device_cores = worker_grid;
    auto dtype = tt_metal::datatype_to_dataformat_converter(args.a_tensor.dtype());

    // std::vector<CoreCoord> all_cores_list;
    // for (auto core_range : all_device_cores.ranges()) {
    //     for (auto core : core_range) {
    //         auto virtual_core = args.a_tensor.device()->worker_core_from_logical_core(core);
    //         log_info(tt::LogTest, "logical core: {}, virtual core: {}", core, virtual_core);
    //     }
    // }

    /***************   CIRCULAR BUFFERS ***************/

    auto createCircularBuffer = [&program, &all_device_cores, dtype = dtype](
                                    tt::CBIndex cb_idx, uint32_t tile_size, uint32_t num_input_tiles = 1) {
        auto cb_config = tt::tt_metal::CircularBufferConfig(num_input_tiles * tile_size, {{cb_idx, dtype}})
                             .set_page_size(cb_idx, tile_size);
        return tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_config);
    };

    uint32_t single_tile_size = tt::tile_size(dtype);
    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    auto a_tensor_cb = tt::CBIndex::c_0;
    auto b_tensor_cb = tt::CBIndex::c_1;
    auto output_cb_index = tt::CBIndex::c_2;

    constexpr uint32_t num_tiles_per_batch = 4;
    constexpr uint32_t num_batches = 2;
    constexpr uint32_t num_tiles_per_cb = 2 * num_tiles_per_batch * (num_batches);

    CBHandle a_tensor_cb_handle = createCircularBuffer(a_tensor_cb, single_tile_size, num_tiles_per_cb);
    CBHandle b_tensor_cb_handle = createCircularBuffer(b_tensor_cb, single_tile_size, num_tiles_per_cb);
    CBHandle cb_output = createCircularBuffer(output_cb_index, single_tile_size, num_tiles_per_cb);

    /***************   READER KERNEL ***************/

    EltwiseReaderCTArgs reader_compile_time_args = {
        .a_tensor_cb = a_tensor_cb,
        .b_tensor_cb = b_tensor_cb,
        .num_batches = num_batches,
        .num_tiles_per_batch = num_tiles_per_batch,
    };
    std::vector<uint32_t> reader_compile_time_vec = ttnn::kernel_utils::to_vector(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(args.a_tensor.buffer()).append_to(reader_compile_time_vec);
    tt::tt_metal::TensorAccessorArgs(args.b_tensor.buffer()).append_to(reader_compile_time_vec);

    std::map<std::string, std::string> reader_defines;
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        kernel_prefix + "elemwise_reader_kernel.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_vec, reader_defines));

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /***************   WRITER KERNEL ***************/

    EltwiseWriterCTArgs writer_compile_time_args = {
        .cb_dst = output_cb_index, .num_batches = num_batches, .num_tiles_per_batch = num_tiles_per_batch};

    std::vector<uint32_t> writer_compile_time_vec = ttnn::kernel_utils::to_vector(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_vec);

    std::map<std::string, std::string> writer_defines;
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        kernel_prefix + "elemwise_writer_kernel.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_vec, writer_defines));

    /***************   COMPUTE KERNEL ***************/
    /* Use the add_tiles operation in the compute kernel */
    EltwiseComputeCTArgs compute_compile_time_args = {
        .a_tensor_cb = a_tensor_cb,
        .b_tensor_cb = b_tensor_cb,
        .output_cb = output_cb_index,
        .num_batches = num_batches,
        .num_tiles_per_batch = num_tiles_per_batch};
    std::vector<uint32_t> compute_compile_time_vec = ttnn::kernel_utils::to_vector(compute_compile_time_args);
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        kernel_prefix + "elemwise_compute_kernel.cpp",
        all_device_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_vec});

    if (operation_attributes.sub_core_grids.has_value()) {
        std::cout << "[debug] sub_core_grids provided, using dram cores" << std::endl;
        set_eltwise_binary_runtime_args_for_dram_cores<true>(
            program,
            args.a_tensor,
            args.b_tensor,
            output,
            reader_kernel_id,
            writer_kernel_id,
            compute_kernel_id,
            all_device_cores);
    } else {
        std::cout << "[debug] sub_core_grids not provided, using all cores" << std::endl;
        set_eltwise_binary_runtime_args_across_all_cores<true>(
            program,
            args.a_tensor,
            args.b_tensor,
            output,
            reader_kernel_id,
            writer_kernel_id,
            compute_kernel_id,
            all_device_cores);
    }
    return {
        std::move(program),
        {reader_kernel_id,
         writer_kernel_id,
         compute_kernel_id,
         a_tensor_cb_handle,
         b_tensor_cb_handle,
         cb_output,
         all_device_cores,
         single_tile_size,
         single_tile_size,
         single_tile_size}};
}

void ElementWiseMultiCoreAddProgram::override_runtime_arguments(
    cached_program_t& cached_program,
    const AddParams& operation_attributes,
    const AddInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& sh_var = cached_program.shared_variables;

    if (operation_attributes.sub_core_grids.has_value()) {
        set_eltwise_binary_runtime_args_for_dram_cores<false>(
            cached_program.program,
            tensor_args.a_tensor,
            tensor_args.b_tensor,
            tensor_return_value,
            sh_var.reader_kernel_id,
            sh_var.writer_kernel_id,
            sh_var.eltwise_kernel_id,
            sh_var.all_device_cores);
    } else {
        set_eltwise_binary_runtime_args_across_all_cores<false>(
            cached_program.program,
            tensor_args.a_tensor,
            tensor_args.b_tensor,
            tensor_return_value,
            sh_var.reader_kernel_id,
            sh_var.writer_kernel_id,
            sh_var.eltwise_kernel_id,
            sh_var.all_device_cores);
    }
}
}  // namespace ttnn::experimental::prim
