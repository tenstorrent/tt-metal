// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_pre_post_program_factory.hpp"

#include <bit>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace {

constexpr uint32_t kCbPreW = tt::CBIndex::c_0;
constexpr uint32_t kCbPostW = tt::CBIndex::c_1;
constexpr uint32_t kCbPreBias = tt::CBIndex::c_2;
constexpr uint32_t kCbPostBias = tt::CBIndex::c_3;
constexpr uint32_t kCbHidden = tt::CBIndex::c_4;
constexpr uint32_t kCbPostOut = tt::CBIndex::c_5;
constexpr uint32_t kCbCollapsed = tt::CBIndex::c_6;
constexpr uint32_t kCbScratch = tt::CBIndex::c_7;
constexpr uint32_t kCbPre = tt::CBIndex::c_8;

constexpr char kReaderKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "reader_fused_pre_post.cpp";
constexpr char kComputeKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/compute/"
    "fused_pre_post.cpp";
constexpr char kWriterKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "writer_fused_pre_post.cpp";

}  // namespace

FusedPrePostProgramFactory::cached_program_t FusedPrePostProgramFactory::create(
    const FusedPrePostParams& operation_attributes,
    const FusedPrePostInputs& tensor_args,
    FusedPrePostTensorReturn& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& pre_w = tensor_args.pre_w;
    const auto& post_w = tensor_args.post_w;
    const auto& pre_bias = tensor_args.pre_bias;
    const auto& post_bias = tensor_args.post_bias;
    const auto& hidden_streams = tensor_args.hidden_streams;
    auto& post_out = tensor_return_value[0];
    auto& collapsed_out = tensor_return_value[1];

    Program program = CreateProgram();

    const DataFormat tile_data_format = datatype_to_dataformat_converter(pre_w.dtype());
    const uint32_t tile_size_bytes = tile_size(tile_data_format);

    // Decode (T == 1): pre_w / post_w / biases are a single tile each; hidden_streams is
    // [1,1,H,D] -> one tile tall, d_tiles wide, and collapsed is [1,1,1,D] -> d_tiles wide.
    const uint32_t d_tiles = hidden_streams.padded_shape()[-1] / constants::TILE_WIDTH;

    const CoreRange core_range({0, 0}, {0, 0});
    const CoreRangeSet all_cores{core_range};

    constexpr uint32_t tile_buffering = 2;
    auto make_cb = [&](uint32_t index, uint32_t num_pages) {
        CircularBufferConfig config = CircularBufferConfig(num_pages * tile_size_bytes, {{index, tile_data_format}})
                                          .set_page_size(index, tile_size_bytes);
        CreateCircularBuffer(program, all_cores, config);
    };
    make_cb(kCbPreW, tile_buffering);
    make_cb(kCbPostW, tile_buffering);
    make_cb(kCbPreBias, tile_buffering);
    make_cb(kCbPostBias, tile_buffering);
    make_cb(kCbHidden, d_tiles);
    make_cb(kCbPostOut, tile_buffering);
    make_cb(kCbCollapsed, d_tiles);
    make_cb(kCbScratch, tile_buffering);
    make_cb(kCbPre, tile_buffering);

    std::vector<uint32_t> reader_compile_time_args = {
        kCbPreW,
        kCbPostW,
        kCbPreBias,
        kCbPostBias,
        kCbHidden,
    };
    TensorAccessorArgs(pre_w.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(post_w.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(pre_bias.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(post_bias.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(hidden_streams.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {kCbPostOut, kCbCollapsed};
    TensorAccessorArgs(post_out.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(collapsed_out.buffer()).append_to(writer_compile_time_args);

    const uint32_t pre_scale_bits = std::bit_cast<uint32_t>(operation_attributes.pre_scale);
    const uint32_t post_scale_bits = std::bit_cast<uint32_t>(operation_attributes.post_scale);
    const uint32_t eps_bits = std::bit_cast<uint32_t>(operation_attributes.eps);
    const uint32_t two_bits = std::bit_cast<uint32_t>(2.0f);

    std::vector<uint32_t> compute_compile_time_args = {
        kCbPreW,
        kCbPostW,
        kCbPreBias,
        kCbPostBias,
        kCbHidden,
        kCbPostOut,
        kCbCollapsed,
        kCbScratch,
        kCbPre,
        pre_scale_bits,
        post_scale_bits,
        eps_bits,
        two_bits,
    };

    const KernelHandle reader_kernel_id =
        CreateKernel(program, kReaderKernelPath, all_cores, ReaderDataMovementConfig(reader_compile_time_args));

    const KernelHandle writer_kernel_id =
        CreateKernel(program, kWriterKernelPath, all_cores, WriterDataMovementConfig(writer_compile_time_args));

    const KernelHandle compute_kernel_id = CreateKernel(
        program,
        kComputeKernelPath,
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args});

    const CoreCoord core{0, 0};
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {pre_w.buffer()->address(),
         post_w.buffer()->address(),
         pre_bias.buffer()->address(),
         post_bias.buffer()->address(),
         hidden_streams.buffer()->address(),
         d_tiles});
    SetRuntimeArgs(
        program, writer_kernel_id, core, {post_out.buffer()->address(), collapsed_out.buffer()->address(), d_tiles});
    SetRuntimeArgs(program, compute_kernel_id, core, {d_tiles});

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .compute_kernel_id = compute_kernel_id}};
}

void FusedPrePostProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FusedPrePostParams& /*operation_attributes*/,
    const FusedPrePostInputs& tensor_args,
    FusedPrePostTensorReturn& tensor_return_value) {
    auto& program = cached_program.program;
    const CoreCoord core{0, 0};

    {
        auto& reader_args = GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id, core);
        reader_args[0] = tensor_args.pre_w.buffer()->address();
        reader_args[1] = tensor_args.post_w.buffer()->address();
        reader_args[2] = tensor_args.pre_bias.buffer()->address();
        reader_args[3] = tensor_args.post_bias.buffer()->address();
        reader_args[4] = tensor_args.hidden_streams.buffer()->address();
    }

    {
        auto& writer_args = GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, core);
        writer_args[0] = tensor_return_value[0].buffer()->address();
        writer_args[1] = tensor_return_value[1].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
