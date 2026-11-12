// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "compressor_state_select.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {
namespace {

using namespace tt::tt_metal;

constexpr uint32_t kTileBytes = 32 * 32 * sizeof(uint16_t);
constexpr uint32_t kScratchCb = tt::CBIndex::c_0;
constexpr CoreCoord kCore{0, 0};
constexpr auto kKernel =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/compressor_state_exchange/device/kernels/"
    "select_predecessor_state.cpp";
constexpr uint32_t kKernelIdx = 0;
constexpr uint32_t kNumRuntimeArgs = 6;

struct SelectScalars {
    uint32_t output_tiles;
    uint32_t use_initial;
    uint32_t source_offset;
};

SelectScalars select_scalars(
    const CompressorStateSelectParams& params,
    const CompressorStateSelectInputs& args,
    const std::optional<MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "compressor_state_select requires a mesh coordinate");
    const uint32_t rank = (*mesh_dispatch_coordinate)[params.cluster_axis];
    const uint32_t output_tiles = args.initial_state.padded_shape().volume() / (32 * 32);
    const bool use_initial = !params.propagate_trailing && rank == 0;
    const uint32_t source_rank =
        params.propagate_trailing ? std::min(rank, params.last_active_rank) : (rank == 0 ? 0 : rank - 1);
    return {output_tiles, static_cast<uint32_t>(use_initial), source_rank * output_tiles};
}

}  // namespace

ProgramDescriptor CompressorStateSelectProgramFactory::create_descriptor(
    const CompressorStateSelectParams& params,
    const CompressorStateSelectInputs& args,
    Tensor& output,
    const std::optional<MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto scalars = select_scalars(params, args, mesh_dispatch_coordinate);

    std::vector<uint32_t> compile_args;
    TensorAccessorArgs(args.gathered_state.buffer()).append_to(compile_args);
    TensorAccessorArgs(args.initial_state.buffer()).append_to(compile_args);
    TensorAccessorArgs(output.buffer()).append_to(compile_args);

    KernelDescriptor kernel;
    kernel.kernel_source = kKernel;
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = CoreRangeSet(CoreRange(kCore, kCore));
    kernel.compile_time_args = std::move(compile_args);
    kernel.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
    };
    KernelDescriptor::RTArgList runtime_args;
    runtime_args.reserve(kNumRuntimeArgs);
    runtime_args.push_back(args.gathered_state.buffer());
    runtime_args.push_back(args.initial_state.buffer());
    runtime_args.push_back(output.buffer());
    runtime_args.push_back(scalars.output_tiles);
    runtime_args.push_back(scalars.use_initial);
    runtime_args.push_back(scalars.source_offset);
    kernel.emplace_runtime_args(kCore, runtime_args);

    ProgramDescriptor desc;
    const CoreRangeSet cores(CoreRange(kCore, kCore));
    desc.cbs.push_back(CBDescriptor{
        .total_size = kTileBytes,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(kScratchCb),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = kTileBytes,
        }}},
    });
    TT_FATAL(desc.kernels.size() == kKernelIdx, "compressor_state_select kernel must be index {}", kKernelIdx);
    desc.kernels.push_back(std::move(kernel));
    return desc;
}

void CompressorStateSelectProgramFactory::override_runtime_arguments(
    Program& program,
    const CompressorStateSelectParams& params,
    const CompressorStateSelectInputs& args,
    Tensor& output,
    const std::optional<MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto scalars = select_scalars(params, args, mesh_dispatch_coordinate);
    // With this hook present the adapter skips its own address rebinding, so every arg is re-applied.
    auto& runtime_args = GetRuntimeArgs(program, kKernelIdx, kCore);
    TT_FATAL(
        runtime_args.size() == kNumRuntimeArgs,
        "compressor_state_select has {} runtime args, expected {}",
        runtime_args.size(),
        kNumRuntimeArgs);
    runtime_args[0] = args.gathered_state.buffer()->address();
    runtime_args[1] = args.initial_state.buffer()->address();
    runtime_args[2] = output.buffer()->address();
    runtime_args[3] = scalars.output_tiles;
    runtime_args[4] = scalars.use_initial;
    runtime_args[5] = scalars.source_offset;
}

}  // namespace ttnn::experimental::prim
