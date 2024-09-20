// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "upsample_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/math.hpp"

#include "tt_metal/tt_stl/reflection.hpp"

using namespace tt::constants;

namespace ttnn::operations::upsample {
using namespace tt;

operation::ProgramWithCallbacks upsample_multi_core(const Tensor &input, Tensor& output, const uint32_t scale_factor_h, const uint32_t scale_factor_w) {
    Program program = CreateProgram();
    Device *device = input.device();

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
    uint32_t out_w = output.get_legacy_shape()[2];

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
    } else if (input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        ncores_x = all_cores.ranges().begin()->end_coord.x + 1;
        ncores_nhw = all_cores.ranges().begin()->end_coord.y + 1;
        input_stick_nbytes = input_stick_nbytes / ncores_x;
        output_stick_nbytes = output_stick_nbytes / ncores_x;
    } else {
        TT_THROW("Unsupported sharding layout");
    }

    uint32_t input_nsticks_per_core = div_up(input_nsticks, ncores_nhw);
    uint32_t output_nsticks_per_core = div_up(output_nsticks, ncores_nhw);

    // TODO: Support non-multiple case
    TT_FATAL(in_nsticks_per_core == input_nsticks_per_core, "Input sticks per shard {} should be same as input sticks per core {}", in_nsticks_per_core, input_nsticks_per_core);
    TT_FATAL(out_nsticks_per_core == output_nsticks_per_core, "Output sticks per shard {} should be same as output sticks per core {}", out_nsticks_per_core, output_nsticks_per_core);
    TT_FATAL(input_nsticks_per_core % in_w == 0, "Error");

    // CBs

    uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs since its sharded

    // input data is in a sharded CB
    uint32_t in_cb_id = CB::c_in0;
    uint32_t aligned_input_stick_nbytes = round_up_to_mul32(input_stick_nbytes);
    uint32_t in_cb_pagesize = aligned_input_stick_nbytes;
    uint32_t in_cb_npages = input_nsticks_per_core * buffering_factor;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
                                            in_cb_pagesize * in_cb_npages,
                                            {{in_cb_id, input_cb_data_format}})
                                          .set_page_size(in_cb_id, in_cb_pagesize)
                                          .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

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

    std::vector<uint32_t> writer_compile_time_args = {
        in_cb_id,
        out_cb_id,
        false,
    };
    auto writer_kernel_fname = std::string("ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_multi_core_sharded.cpp");
    auto writer_kernel =
        CreateKernel(program, writer_kernel_fname, all_cores, WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> reader_compile_time_args = {
        in_cb_id,
        out_cb_id,
        true,
    };
    auto reader_kernel_fname = std::string("ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_multi_core_sharded.cpp");
    auto reader_kernel =
        CreateKernel(program, reader_kernel_fname, all_cores, ReaderDataMovementConfig(reader_compile_time_args));

    // no compute kernel

    // runtime args

    uint32_t writer_nargs = 7;
    vector<uint32_t> writer_rt_args(writer_nargs);
    writer_rt_args[0] = input_stick_nbytes;
    writer_rt_args[1] = input_nsticks_per_core / in_w;
    writer_rt_args[2] = scale_factor_h;
    writer_rt_args[3] = scale_factor_w;
    writer_rt_args[4] = in_w;
    writer_rt_args[5] = out_w;
    writer_rt_args[6] = 0;  // set for each core below

    uint32_t start_input_stick_id = 0;
    if (input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        for (int32_t core = 0; core < ncores_nhw; ++core) {
            for (int32_t core_x = 0; core_x < ncores_x; ++core_x) {
                CoreCoord core_coord(core_x, core); // logical
                writer_rt_args[6] = start_input_stick_id;
                SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);
                SetRuntimeArgs(program, reader_kernel, core_coord, writer_rt_args);
            }
            start_input_stick_id += input_nsticks_per_core;
        }
    } else if (input.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        for (int32_t core = 0; core < ncores_nhw; ++core) {
            CoreCoord core_coord(core % ncores_x, core / ncores_x); // logical
            writer_rt_args[6] = start_input_stick_id;
            SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);
            SetRuntimeArgs(program, reader_kernel, core_coord, writer_rt_args);
            start_input_stick_id += input_nsticks_per_core;
        }
    } else {
        TT_THROW("Unsupported memory layout");
    }

    auto override_runtime_args_callback = [writer_kernel, cb_src0, out_cb](
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace ttnn::operations::upsample
