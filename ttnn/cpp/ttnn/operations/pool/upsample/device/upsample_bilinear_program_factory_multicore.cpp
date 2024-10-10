// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "upsample_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/math.hpp"
//#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"  // for reduce_op_utils

#include "tt_metal/tt_stl/reflection.hpp"
#include "ttnn/operations/numpy/functions.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

using namespace tt::constants;

namespace  ttnn::operations::upsample  {
using namespace tt;
using sliding_window::SlidingWindowConfig;

Tensor HaloTensorCreation(const Tensor &input){
    int batch_size = input.get_legacy_shape()[0];
    int input_height = input.get_legacy_shape()[1];
    int input_width = input.get_legacy_shape()[2];
    int num_cores_nhw = input.shard_spec().value().num_cores();

    ttnn::Tensor input_tensor = input;  // tensor to return
    SlidingWindowConfig sliding_window_config = SlidingWindowConfig(
            batch_size,
            {input_height, input_width},
            {2, 2}, //kernel size
            {1, 1}, // stride
            {1, 0}, //padding
            {1, 1}, //dilation
            num_cores_nhw,
            input_tensor.memory_config().shard_spec.value().grid,
            false, true);

    input_tensor = ttnn::reshape(
        input_tensor,
        Shape(std::array<uint32_t, 4>{
            1,
            1,
            input.get_shape()[0] * input.get_shape()[1] * input.get_shape()[2],
            input.get_shape()[3]}));

    auto halo_output = ttnn::halo(
        DefaultQueueId,
        input_tensor,
        sliding_window_config,
        0,
        false,
        false,
        0,
        input_tensor.memory_config(),
        false);

    return halo_output;
}

operation::ProgramWithCallbacks bilinear_multi_core(const Tensor &input, Tensor& output, const uint32_t scale_factor_h, const uint32_t scale_factor_w, const DeviceComputeKernelConfig compute_kernel_config) {
    Program program = CreateProgram();
    Device *device = input.device();

    auto input_shape = input.get_legacy_shape();
    auto output_shape = output.get_legacy_shape();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    // NOTE: input is assumed to have channels last format: {N, H, W, C}, {N, 1, H * W, C}, {1, 1, N * H * W, C}
    // NOTE: Bfp8_b/TILE is not yet supported
    uint32_t input_stick_nbytes = input.get_legacy_shape()[-1] * input.element_size();
    uint32_t output_stick_nbytes = output.get_legacy_shape()[-1] * output.element_size();
    TT_FATAL(input_stick_nbytes == output_stick_nbytes, "Input and output sticks should have same size");

    uint32_t output_nsticks = output.volume() / output.get_legacy_shape()[-1];
    uint32_t input_nsticks = input.volume() / input.get_legacy_shape()[-1];

    uint32_t in_w = input.get_legacy_shape()[2];
    uint32_t out_w =output.get_legacy_shape()[2];

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);

    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();
    uint32_t ncores_x = device->compute_with_storage_grid_size().x;
    uint32_t ncores_nhw = ncores;

    auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(out_shard_spec.num_cores() == ncores, "Output tensor should have same number of cores {} as input tensor {}", out_shard_spec.num_cores(), ncores);

    uint32_t in_nsticks_per_core = shard_spec.shape[0];
    uint32_t out_nsticks_per_core = in_nsticks_per_core * scale_factor_h * scale_factor_w;

    // extra limitation to avoid post upsample step of resharding
    if (input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        TT_FATAL(in_nsticks_per_core % in_w == 0, "Restriction: Input sticks per core {} should be divisible by input width {}. TODO to remove this restriction", in_nsticks_per_core, in_w);
    } else {
        TT_FATAL(false, "Unsupported sharding layout");
    }

    uint32_t input_nsticks_per_core = div_up(input_nsticks, ncores_nhw);
    uint32_t output_nsticks_per_core = div_up(output_nsticks, ncores_nhw);

    TT_FATAL(in_nsticks_per_core == input_nsticks_per_core, "Input sticks per shard {} should be same as input sticks per core {}", in_nsticks_per_core, input_nsticks_per_core);
    TT_FATAL(out_nsticks_per_core == output_nsticks_per_core, "Output sticks per shard {} should be same as output sticks per core {}", out_nsticks_per_core, output_nsticks_per_core);
    TT_FATAL(input_nsticks_per_core % in_w == 0, "Error");

    //creating halo input tensor
    auto halo_in = HaloTensorCreation(input);
    auto halo_shard_shape = halo_in.shard_spec().value().shape;

    // CBs
    uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs since its sharded

    // input data is in a sharded CB
    uint32_t in_cb_id = CB::c_in0;
    uint32_t aligned_input_stick_nbytes = round_up_to_mul32(input_stick_nbytes);
    uint32_t in_cb_pagesize = aligned_input_stick_nbytes;
    uint32_t in_cb_npages = halo_shard_shape[0] * buffering_factor;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
                                            in_cb_pagesize * in_cb_npages,
                                            {{in_cb_id, input_cb_data_format}})
                                          .set_page_size(in_cb_id, in_cb_pagesize)
                                          .set_globally_allocated_address(*halo_in.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    //intermediate tensor CB
    uint32_t in1_cb_id = CB::c_in1;
    CircularBufferConfig cb_src1_config = CircularBufferConfig(
                                            4 * in_cb_pagesize,  //since 4 pixels per page are needed for intermediate tensor.
                                            {{in1_cb_id, input_cb_data_format}})
                                          .set_page_size(in1_cb_id, in_cb_pagesize);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    //scaler CB
    uint32_t in_scalar_cb_id = CB::c_in4;
    uint32_t in_scalar_cb_pagesize = tile_size(input_cb_data_format);
    uint32_t in_scalar_cb_npages = 1;
    CircularBufferConfig in_scalar_cb_config =
        CircularBufferConfig(in_scalar_cb_npages * in_scalar_cb_pagesize, {{in_scalar_cb_id, input_cb_data_format}})
                    .set_page_size(in_scalar_cb_id, in_scalar_cb_pagesize);


    auto in_scalar_cb = tt_metal::CreateCircularBuffer(program, all_cores, in_scalar_cb_config);

    // output sharded CB with upsampled data
    uint32_t out_cb_id = CB::c_out0;
    uint32_t aligned_output_stick_nbytes = round_up_to_mul32(output_stick_nbytes);
    uint32_t out_cb_pagesize = aligned_output_stick_nbytes;
    uint32_t out_cb_npages = output_nsticks_per_core * buffering_factor;
    CircularBufferConfig out_cb_config = CircularBufferConfig(
                                            out_cb_pagesize * out_cb_npages,
                                            {{out_cb_id, output_cb_data_format}})
                                          .set_page_size(out_cb_id, out_cb_pagesize)
                                          .set_globally_allocated_address(*output.buffer());
    auto out_cb = tt_metal::CreateCircularBuffer(program, all_cores, out_cb_config);

    log_debug(LogOp, "input_cb: {}, npages: {}, pagesize: {}", in_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(LogOp, "output_cb: {}, npages: {}, pagesize: {}", out_cb_id, out_cb_npages, out_cb_pagesize);
    log_debug(LogOp, "input_stick_nbytes: {}, output_stick_nbytes: {}", input_stick_nbytes, output_stick_nbytes);
    log_debug(LogOp, "ncores: {}, ncores_x: {}", ncores, ncores_x);
    log_debug(LogOp, "input_nsticks_per_core: {}, output_nsticks_per_core: {}", input_nsticks_per_core, output_nsticks_per_core);

    // Kernels
    //computation needed for the bilinear kernel. Passing them as an argument.
    float scale_h_inv = 1.0f / (float)scale_factor_h;
    float scale_w_inv = 1.0f / (float)scale_factor_w;
    float y_index = (float)(0.5f) * (float)scale_h_inv + 0.5f;
    float x_index_compute = (float)(0.5f) * (float)scale_w_inv - 0.5f;

    uint32_t scale_h_inv_u32 = *reinterpret_cast<uint32_t*>(&scale_h_inv);
    uint32_t scale_w_inv_u32 = *reinterpret_cast<uint32_t*>(&scale_w_inv);
    uint32_t y_index_u32 = *reinterpret_cast<uint32_t*>(&y_index);
    uint32_t x_index_compute_u32 = *reinterpret_cast<uint32_t*>(&x_index_compute);

    std::vector<uint32_t> reader_compile_time_args = {
        in_cb_id,
        out_cb_id,
        false,
        scale_h_inv_u32,
        scale_w_inv_u32,
        y_index_u32,
        x_index_compute_u32,
    };

    string writer_kernel_fname, reader_kernel_fname, compute_kernel_fname;

    reader_kernel_fname = std::string("ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp");
    compute_kernel_fname = std::string("ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/compute/bilinear.cpp");

    uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / constants::TILE_WIDTH);
    std::vector<uint32_t> compute_compile_time_args = {
        1,
        in_ntiles_c,
        1 * in_ntiles_c,
        4,
        output_shape[1],
        output_shape[2],
        (uint32_t)std::ceil((float)output_shape[2] / constants::TILE_HEIGHT),
        (uint32_t)std::ceil((float)output_shape[3] / constants::TILE_WIDTH),
        output_nsticks_per_core,  // loop count with blocks
        input_shape[3],
    };

    auto reader_kernel =
        CreateKernel(program, reader_kernel_fname, all_cores, ReaderDataMovementConfig(reader_compile_time_args));
    TT_FATAL(fp32_dest_acc_en == false, "fp32_dest_acc_en as true not supported. #12787 issue raised");
    auto reduce_op = ReduceOpMath::SUM;
    auto reduce_dim = ReduceOpDim::H;
    auto compute_config = ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_compile_time_args,
        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};

    auto compute_kernel =
        CreateKernel(program, compute_kernel_fname, all_cores, compute_config);

    // runtime args
    uint32_t reader_nargs = 10;
    vector<uint32_t> reader_rt_args(reader_nargs);
    reader_rt_args[0] = input_stick_nbytes;
    reader_rt_args[1] = input_nsticks_per_core / in_w;
    reader_rt_args[2] = scale_factor_h;
    reader_rt_args[3] = scale_factor_w;
    reader_rt_args[4] = in_w;
    reader_rt_args[5] = out_w;
    reader_rt_args[6] = 0;  // set for each core below

    uint32_t start_input_stick_id = 0;

    if (input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        for (int32_t core = 0; core < ncores_nhw; ++core) {
            CoreCoord core_coord(core % ncores_x, core / ncores_x); // logical
            reader_rt_args[6] = start_input_stick_id;
            reader_rt_args[8] = (core == 0) ? 1 : 0;
            reader_rt_args[9] = (core == ncores_nhw-1) ? 1 : 0;
            SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
            start_input_stick_id += input_nsticks_per_core;
        }
    } else {
        TT_FATAL(false, "Unsupported memory layout");
    }

    auto override_runtime_args_callback = [reader_kernel, cb_src0, out_cb](
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        auto halo_in = HaloTensorCreation(input_tensors.at(0));
        auto src_buffer = halo_in.buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace ttnn::operations::upsample
