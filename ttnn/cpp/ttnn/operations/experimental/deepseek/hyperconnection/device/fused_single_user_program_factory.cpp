// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_single_user_program_factory.hpp"

#include <algorithm>
#include <bit>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek::hyperconnection {

namespace fused_single_user_detail {

constexpr uint32_t kRoleCollapse = 0;
constexpr uint32_t kRolePost = 1;
constexpr uint32_t kRoleComb = 2;
constexpr uint32_t kCollapseCoreCount = 8;
constexpr uint32_t kTotalCoreCount = 10;

constexpr uint32_t kFsCbFusedW = tt::CBIndex::c_0;
constexpr uint32_t kFsCbPreW = tt::CBIndex::c_1;
constexpr uint32_t kFsCbPreBias = tt::CBIndex::c_2;
constexpr uint32_t kFsCbHidden = tt::CBIndex::c_3;
constexpr uint32_t kFsCbPre = tt::CBIndex::c_4;
constexpr uint32_t kFsCbScratch = tt::CBIndex::c_5;
constexpr uint32_t kFsCbCollapsedOut = tt::CBIndex::c_6;
constexpr uint32_t kFsCbPostW = tt::CBIndex::c_7;
constexpr uint32_t kFsCbPostBias = tt::CBIndex::c_8;
constexpr uint32_t kFsCbPostOut = tt::CBIndex::c_9;
constexpr uint32_t kFsCbPostCol = tt::CBIndex::c_10;
constexpr uint32_t kFsCbCombW = tt::CBIndex::c_11;
constexpr uint32_t kFsCbCombBias = tt::CBIndex::c_12;
constexpr uint32_t kFsCbScaler = tt::CBIndex::c_13;
constexpr uint32_t kFsCbMask = tt::CBIndex::c_14;
constexpr uint32_t kFsCbComb = tt::CBIndex::c_15;
constexpr uint32_t kFsCbReduce = tt::CBIndex::c_16;
constexpr uint32_t kFsCbEpsMask = tt::CBIndex::c_17;
constexpr uint32_t kFsCbCombOut = tt::CBIndex::c_18;

constexpr char kFsReaderKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "reader_fused_single_user.cpp";
constexpr char kFsComputeKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/compute/"
    "compute_fused_single_user.cpp";
constexpr char kFsWriterKernelPath[] =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/device/kernels/dataflow/"
    "writer_fused_single_user.cpp";

CoreRangeSet single_core_set(const CoreCoord& core) { return CoreRangeSet({CoreRange(core, core)}); }

CoreRangeSet contiguous_cores(uint32_t first, uint32_t count) {
    return CoreRangeSet({CoreRange(CoreCoord(first, 0), CoreCoord(first + count - 1, 0))});
}

}  // namespace fused_single_user_detail

using namespace fused_single_user_detail;

FusedSingleUserProgramFactory::cached_program_t FusedSingleUserProgramFactory::create(
    const FusedSingleUserParams& operation_attributes,
    const FusedSingleUserInputs& tensor_args,
    FusedSingleUserTensorReturn& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& fused_w = tensor_args.fused_w;
    const auto& pre_bias = tensor_args.pre_bias;
    const auto& post_bias = tensor_args.post_bias;
    const auto& comb_bias = tensor_args.comb_bias;
    const auto& hidden_streams = tensor_args.hidden_streams;
    auto& post_out = tensor_return_value[0];
    auto& comb_out = tensor_return_value[1];
    auto& collapsed_out = tensor_return_value[2];

    const CoreRangeSet collapse_cores = contiguous_cores(0, kCollapseCoreCount);
    const CoreCoord post_core{8, 0};
    const CoreCoord comb_core{9, 0};
    const CoreRangeSet post_cores = single_core_set(post_core);
    const CoreRangeSet comb_cores = single_core_set(comb_core);
    const CoreRangeSet all_cores = collapse_cores.merge(post_cores).merge(comb_cores);

    const uint32_t tile_size_bytes = tile_size(datatype_to_dataformat_converter(fused_w.dtype()));
    const uint32_t d_tiles = static_cast<uint32_t>(hidden_streams.padded_shape()[-1]) / constants::TILE_WIDTH;
    const uint32_t d_tiles_per_core = d_tiles / kCollapseCoreCount;
    const uint32_t num_streams = operation_attributes.num_streams;
    Program program = CreateProgram();
    const uint32_t data_ready_sem_id = CreateSemaphore(program, all_cores, 0);
    const uint32_t receiver_ready_sem_id = CreateSemaphore(program, all_cores, 0);

    auto* device = fused_w.device();
    const auto sender_noc = device->worker_core_from_logical_core(CoreCoord{0, 0});
    const auto mcast_a = device->worker_core_from_logical_core(CoreCoord{1, 0});
    const auto mcast_b = device->worker_core_from_logical_core(CoreCoord{kTotalCoreCount - 1, 0});
    const uint32_t mcast_start_x = std::min<uint32_t>(mcast_a.x, mcast_b.x);
    const uint32_t mcast_start_y = std::min<uint32_t>(mcast_a.y, mcast_b.y);
    const uint32_t mcast_end_x = std::max<uint32_t>(mcast_a.x, mcast_b.x);
    const uint32_t mcast_end_y = std::max<uint32_t>(mcast_a.y, mcast_b.y);
    const uint32_t num_receivers = kTotalCoreCount - 1;

    auto make_cb = [&](uint32_t index, uint32_t num_pages, const CoreRangeSet& cores) {
        CircularBufferConfig config =
            CircularBufferConfig(
                num_pages * tile_size_bytes, {{index, datatype_to_dataformat_converter(fused_w.dtype())}})
                .set_page_size(index, tile_size_bytes);
        CreateCircularBuffer(program, cores, config);
    };

    // This CB is a private receive buffer on every participating core. Core 0
    // fills its copy from the one-core width-sharded input and multicasts into
    // the same L1 offset on cores 1..9.
    make_cb(kFsCbFusedW, 1, all_cores);

    // Collapse branch: hidden input and collapsed output are both local,
    // globally-addressed width shards.
    CBHandle hidden_cb = 0;
    CBHandle collapsed_output_cb = 0;
    {
        auto config = CircularBufferConfig(
                          d_tiles_per_core * tile_size_bytes,
                          {{kFsCbHidden, datatype_to_dataformat_converter(hidden_streams.dtype())}})
                          .set_page_size(kFsCbHidden, tile_size_bytes)
                          .set_tile_dims(kFsCbHidden, hidden_streams.tensor_spec().tile())
                          .set_globally_allocated_address(*hidden_streams.buffer());
        hidden_cb = CreateCircularBuffer(program, collapse_cores, config);
    }
    {
        auto config = CircularBufferConfig(
                          d_tiles_per_core * tile_size_bytes,
                          {{kFsCbCollapsedOut, datatype_to_dataformat_converter(collapsed_out.dtype())}})
                          .set_page_size(kFsCbCollapsedOut, tile_size_bytes)
                          .set_tile_dims(kFsCbCollapsedOut, collapsed_out.tensor_spec().tile())
                          .set_globally_allocated_address(*collapsed_out.buffer());
        collapsed_output_cb = CreateCircularBuffer(program, collapse_cores, config);
    }
    make_cb(kFsCbPreW, 1, collapse_cores);
    make_cb(kFsCbPreBias, 1, collapse_cores);
    make_cb(kFsCbPre, 1, collapse_cores);
    make_cb(kFsCbScratch, 2, collapse_cores.merge(post_cores));

    // Post branch.
    make_cb(kFsCbPostW, 1, post_cores);
    make_cb(kFsCbPostBias, 1, post_cores);
    make_cb(kFsCbPostOut, 1, post_cores);
    make_cb(kFsCbPostCol, 1, post_cores);

    // Comb/Sinkhorn branch.
    make_cb(kFsCbCombW, 1, comb_cores);
    make_cb(kFsCbCombBias, 1, comb_cores);
    make_cb(kFsCbScaler, 1, comb_cores);
    make_cb(kFsCbMask, 1, comb_cores);
    make_cb(kFsCbComb, 2, comb_cores);
    make_cb(kFsCbReduce, 2, comb_cores);
    make_cb(kFsCbEpsMask, 1, comb_cores);
    make_cb(kFsCbCombOut, 1, comb_cores);

    const uint32_t scaler_bits = std::bit_cast<uint32_t>(1.0f);
    const uint32_t pre_scale_bits = std::bit_cast<uint32_t>(operation_attributes.pre_scale);
    const uint32_t post_scale_bits = std::bit_cast<uint32_t>(operation_attributes.post_scale);
    const uint32_t comb_scale_bits = std::bit_cast<uint32_t>(operation_attributes.comb_scale);
    const uint32_t eps_bits = std::bit_cast<uint32_t>(operation_attributes.eps);
    const uint32_t two_bits = std::bit_cast<uint32_t>(2.0f);

    auto reader_compile_args = [&](uint32_t role) {
        std::vector<uint32_t> args = {
            role,
            kFsCbFusedW,
            data_ready_sem_id,
            kFsCbPreW,
            kFsCbPreBias,
            kFsCbHidden,
            kFsCbPostW,
            kFsCbPostBias,
            kFsCbCombW,
            kFsCbCombBias,
            kFsCbScaler,
            kFsCbMask,
            kFsCbEpsMask,
            d_tiles_per_core,
            num_streams,
            scaler_bits,
            eps_bits,
            mcast_start_x,
            mcast_start_y,
            mcast_end_x,
            mcast_end_y,
            num_receivers,
            receiver_ready_sem_id,
            static_cast<uint32_t>(sender_noc.x),
            static_cast<uint32_t>(sender_noc.y),
        };
        TensorAccessorArgs(fused_w.buffer()).append_to(args);
        TensorAccessorArgs(pre_bias.buffer()).append_to(args);
        TensorAccessorArgs(post_bias.buffer()).append_to(args);
        TensorAccessorArgs(hidden_streams.buffer()).append_to(args);
        TensorAccessorArgs(comb_bias.buffer()).append_to(args);
        return args;
    };

    auto compute_compile_args = [&](uint32_t role) {
        return std::vector<uint32_t>{
            role,
            kFsCbPreW,
            kFsCbPostW,
            kFsCbPreBias,
            kFsCbPostBias,
            kFsCbHidden,
            kFsCbPostOut,
            kFsCbCollapsedOut,
            kFsCbScratch,
            kFsCbPre,
            kFsCbCombW,
            kFsCbCombBias,
            kFsCbScaler,
            kFsCbMask,
            kFsCbComb,
            kFsCbReduce,
            kFsCbEpsMask,
            kFsCbCombOut,
            pre_scale_bits,
            post_scale_bits,
            eps_bits,
            two_bits,
            num_streams,
            operation_attributes.sinkhorn_iters,
            comb_scale_bits,
        };
    };

    auto writer_compile_args = [&](uint32_t role) {
        std::vector<uint32_t> args = {role, kFsCbPostOut, kFsCbPostCol, kFsCbCombOut};
        TensorAccessorArgs(post_out.buffer()).append_to(args);
        TensorAccessorArgs(comb_out.buffer()).append_to(args);
        return args;
    };

    FusedSingleUserSharedVariables shared;
    shared.collapse_cores = corerange_to_cores(collapse_cores, collapse_cores.num_cores(), true);
    shared.post_core = post_core;
    shared.comb_core = comb_core;
    shared.hidden_cb = hidden_cb;
    shared.collapsed_output_cb = collapsed_output_cb;

    shared.collapse_reader_kernel_id = CreateKernel(
        program, kFsReaderKernelPath, collapse_cores, ReaderDataMovementConfig(reader_compile_args(kRoleCollapse)));
    shared.post_reader_kernel_id = CreateKernel(
        program, kFsReaderKernelPath, post_cores, ReaderDataMovementConfig(reader_compile_args(kRolePost)));
    shared.comb_reader_kernel_id = CreateKernel(
        program, kFsReaderKernelPath, comb_cores, ReaderDataMovementConfig(reader_compile_args(kRoleComb)));

    auto compute_config = [&](uint32_t role) {
        return ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_args(role)};
    };
    shared.collapse_compute_kernel_id =
        CreateKernel(program, kFsComputeKernelPath, collapse_cores, compute_config(kRoleCollapse));
    shared.post_compute_kernel_id = CreateKernel(program, kFsComputeKernelPath, post_cores, compute_config(kRolePost));
    shared.comb_compute_kernel_id = CreateKernel(program, kFsComputeKernelPath, comb_cores, compute_config(kRoleComb));

    shared.post_writer_kernel_id = CreateKernel(
        program, kFsWriterKernelPath, post_cores, WriterDataMovementConfig(writer_compile_args(kRolePost)));
    shared.comb_writer_kernel_id = CreateKernel(
        program, kFsWriterKernelPath, comb_cores, WriterDataMovementConfig(writer_compile_args(kRoleComb)));

    for (const auto& core : shared.collapse_cores) {
        SetRuntimeArgs(
            program,
            shared.collapse_reader_kernel_id,
            core,
            {fused_w.buffer()->address(),
             pre_bias.buffer()->address(),
             post_bias.buffer()->address(),
             hidden_streams.buffer()->address(),
             comb_bias.buffer()->address(),
             core == CoreCoord{0, 0} ? 1u : 0u});
        SetRuntimeArgs(program, shared.collapse_compute_kernel_id, core, {d_tiles_per_core});
    }
    SetRuntimeArgs(
        program,
        shared.post_reader_kernel_id,
        post_core,
        {fused_w.buffer()->address(),
         pre_bias.buffer()->address(),
         post_bias.buffer()->address(),
         hidden_streams.buffer()->address(),
         comb_bias.buffer()->address(),
         0});
    SetRuntimeArgs(program, shared.post_compute_kernel_id, post_core, {});
    SetRuntimeArgs(
        program, shared.post_writer_kernel_id, post_core, {post_out.buffer()->address(), comb_out.buffer()->address()});

    SetRuntimeArgs(
        program,
        shared.comb_reader_kernel_id,
        comb_core,
        {fused_w.buffer()->address(),
         pre_bias.buffer()->address(),
         post_bias.buffer()->address(),
         hidden_streams.buffer()->address(),
         comb_bias.buffer()->address(),
         0});
    SetRuntimeArgs(program, shared.comb_compute_kernel_id, comb_core, {});
    SetRuntimeArgs(
        program, shared.comb_writer_kernel_id, comb_core, {post_out.buffer()->address(), comb_out.buffer()->address()});

    return cached_program_t{std::move(program), std::move(shared)};
}

void FusedSingleUserProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FusedSingleUserParams& /*operation_attributes*/,
    const FusedSingleUserInputs& tensor_args,
    FusedSingleUserTensorReturn& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    const uint32_t fused_w_addr = tensor_args.fused_w.buffer()->address();
    const uint32_t pre_bias_addr = tensor_args.pre_bias.buffer()->address();
    const uint32_t post_bias_addr = tensor_args.post_bias.buffer()->address();
    const uint32_t hidden_addr = tensor_args.hidden_streams.buffer()->address();
    const uint32_t comb_bias_addr = tensor_args.comb_bias.buffer()->address();
    const uint32_t post_addr = tensor_return_value[0].buffer()->address();
    const uint32_t comb_addr = tensor_return_value[1].buffer()->address();
    UpdateDynamicCircularBufferAddress(program, shared.hidden_cb, *tensor_args.hidden_streams.buffer());
    UpdateDynamicCircularBufferAddress(program, shared.collapsed_output_cb, *tensor_return_value[2].buffer());

    auto& collapse_reader_args = GetRuntimeArgs(program, shared.collapse_reader_kernel_id);
    for (const auto& core : shared.collapse_cores) {
        auto& args = collapse_reader_args[core.x][core.y];
        args[0] = fused_w_addr;
        args[1] = pre_bias_addr;
        args[2] = post_bias_addr;
        args[3] = hidden_addr;
        args[4] = comb_bias_addr;
    }

    auto& post_reader_args =
        GetRuntimeArgs(program, shared.post_reader_kernel_id)[shared.post_core.x][shared.post_core.y];
    post_reader_args[0] = fused_w_addr;
    post_reader_args[1] = pre_bias_addr;
    post_reader_args[2] = post_bias_addr;
    post_reader_args[3] = hidden_addr;
    post_reader_args[4] = comb_bias_addr;

    auto& comb_reader_args =
        GetRuntimeArgs(program, shared.comb_reader_kernel_id)[shared.comb_core.x][shared.comb_core.y];
    comb_reader_args[0] = fused_w_addr;
    comb_reader_args[1] = pre_bias_addr;
    comb_reader_args[2] = post_bias_addr;
    comb_reader_args[3] = hidden_addr;
    comb_reader_args[4] = comb_bias_addr;

    auto& post_writer_args =
        GetRuntimeArgs(program, shared.post_writer_kernel_id)[shared.post_core.x][shared.post_core.y];
    post_writer_args[0] = post_addr;
    post_writer_args[1] = comb_addr;
    auto& comb_writer_args =
        GetRuntimeArgs(program, shared.comb_writer_kernel_id)[shared.comb_core.x][shared.comb_core.y];
    comb_writer_args[0] = post_addr;
    comb_writer_args[1] = comb_addr;
}

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
