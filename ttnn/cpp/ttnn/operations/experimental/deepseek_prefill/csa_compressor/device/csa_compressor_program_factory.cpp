// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "csa_compressor_device_operation.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {
namespace {

using namespace tt::tt_metal;

constexpr uint32_t kTileBytes = 32 * 32 * sizeof(uint16_t);
constexpr uint32_t kCandidateKvCb = tt::CBIndex::c_0;
constexpr uint32_t kCandidateScoreCb = tt::CBIndex::c_1;
constexpr uint32_t kPooledCb = tt::CBIndex::c_2;
constexpr uint32_t kScratchCb = tt::CBIndex::c_3;
constexpr CoreCoord kCompressionCore{0, 0};
constexpr CoreCoord kStateCore{1, 0};

constexpr auto kStateKernel =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/csa_compressor/device/kernels/"
    "csa_state_update.cpp";
constexpr auto kReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/csa_compressor/device/kernels/"
    "reader_csa_compressor.cpp";
constexpr auto kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/csa_compressor/device/kernels/"
    "compute_csa_compressor.cpp";
constexpr auto kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/csa_compressor/device/kernels/"
    "writer_csa_compressor.cpp";

std::pair<uint32_t, uint32_t> local_runtime(
    const CsaRuntimeParams& params, uint32_t local_seq, const MeshCoordinate& coord) {
    const uint32_t rank = coord[params.cluster_axis];
    const uint32_t local_global_start = rank * local_seq;
    const uint32_t local_valid = params.seq_len_actual > local_global_start
                                     ? std::min(local_seq, params.seq_len_actual - local_global_start)
                                     : 0;
    return {local_valid, params.first_token_position + local_global_start};
}

CBDescriptor cb_descriptor(uint32_t cb, uint32_t pages, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = pages * kTileBytes,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = kTileBytes,
        }}},
    };
}

KernelDescriptor state_kernel_descriptor(
    const CsaStateInputs& args,
    std::array<Tensor, 2>& outputs,
    const CoreCoord& core,
    uint32_t local_valid,
    uint32_t absolute_start) {
    std::vector<uint32_t> compile_args;
    TensorAccessorArgs(args.kv.buffer()).append_to(compile_args);
    TensorAccessorArgs(args.gate.buffer()).append_to(compile_args);
    TensorAccessorArgs(args.position_bias.buffer()).append_to(compile_args);
    TensorAccessorArgs(args.base_kv_state.buffer()).append_to(compile_args);
    TensorAccessorArgs(args.base_score_state.buffer()).append_to(compile_args);
    TensorAccessorArgs(outputs[0].buffer()).append_to(compile_args);
    TensorAccessorArgs(outputs[1].buffer()).append_to(compile_args);

    KernelDescriptor descriptor;
    descriptor.kernel_source = kStateKernel;
    descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
    descriptor.core_ranges = CoreRangeSet(CoreRange(core, core));
    descriptor.compile_time_args = std::move(compile_args);
    descriptor.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
    };
    KernelDescriptor::RTArgList runtime_args;
    runtime_args.reserve(9);
    runtime_args.push_back(args.kv.buffer());
    runtime_args.push_back(args.gate.buffer());
    runtime_args.push_back(args.position_bias.buffer());
    runtime_args.push_back(args.base_kv_state.buffer());
    runtime_args.push_back(args.base_score_state.buffer());
    runtime_args.push_back(outputs[0].buffer());
    runtime_args.push_back(outputs[1].buffer());
    runtime_args.push_back(local_valid);
    runtime_args.push_back(absolute_start);
    descriptor.emplace_runtime_args(core, runtime_args);
    return descriptor;
}

}  // namespace

ProgramDescriptor CsaStatePreparationProgramFactory::create_descriptor(
    const CsaRuntimeParams& params,
    const CsaStateInputs& args,
    std::array<Tensor, 2>& outputs,
    const std::optional<MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "CSA state preparation requires a mesh coordinate");
    const auto [local_valid, absolute_start] =
        local_runtime(params, args.kv.logical_shape()[-2], *mesh_dispatch_coordinate);

    ProgramDescriptor desc;
    const CoreRangeSet state_cores(CoreRange(kCompressionCore, kCompressionCore));
    desc.cbs.push_back(cb_descriptor(kScratchCb, 7, state_cores));
    desc.kernels.push_back(state_kernel_descriptor(args, outputs, kCompressionCore, local_valid, absolute_start));
    return desc;
}

ProgramDescriptor CsaCompressionProgramFactory::create_descriptor(
    const CsaRuntimeParams& params,
    const CsaCompressionInputs& args,
    std::array<Tensor, 3>& outputs,
    const std::optional<MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "CSA compression requires a mesh coordinate");
    const auto grid = args.kv.device()->compute_with_storage_grid_size();
    TT_FATAL(grid.x > 1, "CSA compression requires at least two worker cores");
    const uint32_t local_seq = args.kv.logical_shape()[-2];
    const auto [local_valid, absolute_start] = local_runtime(params, local_seq, *mesh_dispatch_coordinate);
    const uint32_t output_height_tiles = (local_seq / 4 + 31) / 32;
    const uint32_t output_tiles = output_height_tiles * (512 / 32);

    ProgramDescriptor desc;
    const CoreRangeSet compression_cores(CoreRange(kCompressionCore, kCompressionCore));
    const CoreRangeSet state_cores(CoreRange(kStateCore, kStateCore));
    desc.cbs.push_back(cb_descriptor(kCandidateKvCb, 8, compression_cores));
    desc.cbs.push_back(cb_descriptor(kCandidateScoreCb, 8, compression_cores));
    desc.cbs.push_back(cb_descriptor(kPooledCb, 1, compression_cores));
    desc.cbs.push_back(cb_descriptor(kScratchCb, 5, compression_cores));
    for (uint32_t cb = tt::CBIndex::c_4; cb <= tt::CBIndex::c_10; ++cb) {
        desc.cbs.push_back(cb_descriptor(cb, 1, compression_cores));
    }
    desc.cbs.push_back(cb_descriptor(kScratchCb, 7, state_cores));

    std::vector<uint32_t> reader_compile_args = {
        kCandidateKvCb, kCandidateScoreCb, kScratchCb, local_seq / 32, 1024 / 32};
    TensorAccessorArgs(args.kv.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(args.gate.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(args.position_bias.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(args.predecessor_kv_state.buffer()).append_to(reader_compile_args);
    TensorAccessorArgs(args.predecessor_score_state.buffer()).append_to(reader_compile_args);

    KernelDescriptor reader;
    reader.kernel_source = kReaderKernel;
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = compression_cores;
    reader.compile_time_args = std::move(reader_compile_args);
    reader.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};
    KernelDescriptor::RTArgList reader_runtime_args;
    reader_runtime_args.reserve(8);
    reader_runtime_args.push_back(args.kv.buffer());
    reader_runtime_args.push_back(args.gate.buffer());
    reader_runtime_args.push_back(args.position_bias.buffer());
    reader_runtime_args.push_back(args.predecessor_kv_state.buffer());
    reader_runtime_args.push_back(args.predecessor_score_state.buffer());
    reader_runtime_args.push_back(output_tiles);
    reader_runtime_args.push_back(local_valid / 4);
    reader_runtime_args.push_back(absolute_start);
    reader.emplace_runtime_args(kCompressionCore, reader_runtime_args);
    desc.kernels.push_back(std::move(reader));

    KernelDescriptor compute;
    compute.kernel_source = kComputeKernel;
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = compression_cores;
    compute.compile_time_args = {kCandidateKvCb, kCandidateScoreCb, kPooledCb, output_tiles};
    compute.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true};
    desc.kernels.push_back(std::move(compute));

    std::vector<uint32_t> writer_compile_args = {kPooledCb};
    TensorAccessorArgs(outputs[0].buffer()).append_to(writer_compile_args);
    KernelDescriptor writer;
    writer.kernel_source = kWriterKernel;
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = compression_cores;
    writer.compile_time_args = std::move(writer_compile_args);
    writer.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default};
    KernelDescriptor::RTArgList writer_runtime_args;
    writer_runtime_args.reserve(2);
    writer_runtime_args.push_back(outputs[0].buffer());
    writer_runtime_args.push_back(output_tiles);
    writer.emplace_runtime_args(kCompressionCore, writer_runtime_args);
    desc.kernels.push_back(std::move(writer));

    CsaStateInputs state_args{
        args.kv, args.gate, args.position_bias, args.predecessor_kv_state, args.predecessor_score_state};
    std::array<Tensor, 2> state_outputs{outputs[1], outputs[2]};
    desc.kernels.push_back(state_kernel_descriptor(state_args, state_outputs, kStateCore, local_valid, absolute_start));
    return desc;
}

}  // namespace ttnn::experimental::prim
