// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sinkhorn_program_factory.hpp"

#include <bit>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace {

constexpr uint32_t kSinkCbCombW = tt::CBIndex::c_0;
constexpr uint32_t kSinkCbCombBias = tt::CBIndex::c_1;
constexpr uint32_t kSinkCbScaler = tt::CBIndex::c_2;
constexpr uint32_t kSinkCbMask = tt::CBIndex::c_3;
constexpr uint32_t kSinkCbComb = tt::CBIndex::c_4;
constexpr uint32_t kSinkCbReduce = tt::CBIndex::c_5;
constexpr uint32_t kSinkCbScratch = tt::CBIndex::c_6;
constexpr uint32_t kSinkCbOut = tt::CBIndex::c_7;

constexpr char kSinkhornReaderKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "reader_sinkhorn.cpp";
constexpr char kSinkhornComputeKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/compute/"
    "sinkhorn.cpp";
constexpr char kSinkhornWriterKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "writer_sinkhorn.cpp";

}  // namespace

SinkhornProgramFactory::cached_program_t SinkhornProgramFactory::create(
    const SinkhornParams& operation_attributes,
    const SinkhornInputs& tensor_args,
    SinkhornTensorReturn& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& comb_w = tensor_args.comb_w;
    const auto& comb_bias = tensor_args.comb_bias;
    auto& comb_out = tensor_return_value;

    Program program = CreateProgram();

    const DataFormat tile_data_format = datatype_to_dataformat_converter(comb_w.dtype());
    const uint32_t tile_size_bytes = tile_size(tile_data_format);

    const CoreRange core_range({0, 0}, {0, 0});
    const CoreRangeSet all_cores{core_range};

    constexpr uint32_t tile_buffering = 2;
    auto make_cb = [&](uint32_t index, uint32_t num_pages) {
        CircularBufferConfig config = CircularBufferConfig(num_pages * tile_size_bytes, {{index, tile_data_format}})
                                          .set_page_size(index, tile_size_bytes);
        CreateCircularBuffer(program, all_cores, config);
    };
    make_cb(kSinkCbCombW, 1);
    make_cb(kSinkCbCombBias, 1);
    make_cb(kSinkCbScaler, 1);
    make_cb(kSinkCbMask, 1);
    make_cb(kSinkCbComb, tile_buffering);
    make_cb(kSinkCbReduce, tile_buffering);
    make_cb(kSinkCbScratch, tile_buffering);
    make_cb(kSinkCbOut, tile_buffering);

    // Scaler tile holds 1.0f (bf16 hi-half) for plain sum/max reductions.
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(1.0f);
    const uint32_t comb_scale_bits = std::bit_cast<uint32_t>(operation_attributes.comb_scale);
    const uint32_t eps_bits = std::bit_cast<uint32_t>(operation_attributes.eps);

    std::vector<uint32_t> reader_compile_time_args = {
        kSinkCbCombW,
        kSinkCbCombBias,
        kSinkCbMask,
        kSinkCbScaler,
        scaler_bits,
        operation_attributes.num_streams,
    };
    TensorAccessorArgs(comb_w.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(comb_bias.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {kSinkCbOut};
    TensorAccessorArgs(comb_out.buffer()).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        kSinkCbCombW,
        kSinkCbCombBias,
        kSinkCbScaler,
        kSinkCbMask,
        kSinkCbComb,
        kSinkCbReduce,
        kSinkCbScratch,
        kSinkCbOut,
        operation_attributes.num_streams,
        operation_attributes.sinkhorn_iters,
        comb_scale_bits,
        eps_bits,
    };

    const KernelHandle reader_kernel_id =
        CreateKernel(program, kSinkhornReaderKernelPath, all_cores, ReaderDataMovementConfig(reader_compile_time_args));

    const KernelHandle writer_kernel_id =
        CreateKernel(program, kSinkhornWriterKernelPath, all_cores, WriterDataMovementConfig(writer_compile_time_args));

    const KernelHandle compute_kernel_id = CreateKernel(
        program,
        kSinkhornComputeKernelPath,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    const CoreCoord core{0, 0};
    SetRuntimeArgs(program, reader_kernel_id, core, {comb_w.buffer()->address(), comb_bias.buffer()->address()});
    SetRuntimeArgs(program, writer_kernel_id, core, {comb_out.buffer()->address()});
    SetRuntimeArgs(program, compute_kernel_id, core, {});

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .compute_kernel_id = compute_kernel_id}};
}

void SinkhornProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const SinkhornParams& /*operation_attributes*/,
    const SinkhornInputs& tensor_args,
    SinkhornTensorReturn& tensor_return_value) {
    auto& program = cached_program.program;
    const CoreCoord core{0, 0};

    {
        auto& reader_args = GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id, core);
        reader_args[0] = tensor_args.comb_w.buffer()->address();
        reader_args[1] = tensor_args.comb_bias.buffer()->address();
    }
    {
        auto& writer_args = GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, core);
        writer_args[0] = tensor_return_value.buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
