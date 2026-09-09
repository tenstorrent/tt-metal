// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "csa_compressor_device_operation.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {
namespace {

using namespace tt::tt_metal;

constexpr uint32_t kTileBytes = 32 * 32 * sizeof(uint16_t);
constexpr uint32_t kCandidateKvCb = tt::CBIndex::c_0;
constexpr uint32_t kCandidateScoreCb = tt::CBIndex::c_1;
constexpr uint32_t kPooledCb = tt::CBIndex::c_2;
constexpr uint32_t kScratchCb = tt::CBIndex::c_3;
constexpr uint32_t kMaxCb = tt::CBIndex::c_4;
constexpr uint32_t kCaBiasCb = tt::CBIndex::c_5;
constexpr uint32_t kCbBiasCb = tt::CBIndex::c_6;
constexpr uint32_t kStateScratchTiles = 7;

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

// kv/gate pack the Ca and Cb halves side by side, so the head dimension is half the projection width.
uint32_t factory_head_dim_of(const Tensor& kv) { return kv.logical_shape()[-1] / 2; }

KernelDescriptor state_kernel_descriptor(
    const CsaStateInputs& args,
    std::array<Tensor, 2>& outputs,
    const CoreRangeSet& state_cores,
    const std::vector<CoreCoord>& cores,
    uint32_t state_tiles,
    uint32_t local_valid,
    uint32_t absolute_start) {
    const uint32_t head_dim = factory_head_dim_of(args.kv);
    std::vector<uint32_t> compile_args = {
        2 * head_dim / tt::constants::TILE_WIDTH, head_dim / tt::constants::TILE_WIDTH};
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
    descriptor.core_ranges = state_cores;
    descriptor.compile_time_args = std::move(compile_args);
    descriptor.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
    };

    const uint32_t num_cores = cores.size();
    const uint32_t base_tiles = state_tiles / num_cores;
    const uint32_t extra_tiles = state_tiles % num_cores;
    descriptor.runtime_args.reserve(num_cores);
    uint32_t first_tile = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        KernelDescriptor::RTArgList runtime_args;
        runtime_args.reserve(11);
        runtime_args.push_back(args.kv.buffer());
        runtime_args.push_back(args.gate.buffer());
        runtime_args.push_back(args.position_bias.buffer());
        runtime_args.push_back(args.base_kv_state.buffer());
        runtime_args.push_back(args.base_score_state.buffer());
        runtime_args.push_back(outputs[0].buffer());
        runtime_args.push_back(outputs[1].buffer());
        runtime_args.push_back(local_valid);
        runtime_args.push_back(absolute_start);
        const uint32_t core_tiles = base_tiles + (i < extra_tiles ? 1 : 0);
        runtime_args.push_back(core_tiles);
        runtime_args.push_back(first_tile);
        descriptor.emplace_runtime_args(cores[i], runtime_args);
        first_tile += core_tiles;
    }
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

    // The 64-row slab is two tile rows of state_width_tiles each, and those tiles are independent:
    // each one patches its own eight live rows from its own feature columns.
    const auto grid = args.kv.device()->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const uint32_t state_tiles = 2 * factory_head_dim_of(args.kv) / tt::constants::TILE_WIDTH;
    const uint32_t num_cores = std::min(state_tiles, grid_cores);
    const CoreRangeSet state_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const std::vector<CoreCoord> cores = corerange_to_cores(state_cores, num_cores, /*row_wise=*/true);

    ProgramDescriptor desc;
    desc.cbs.push_back(cb_descriptor(kScratchCb, kStateScratchTiles, state_cores));
    desc.kernels.push_back(
        state_kernel_descriptor(args, outputs, state_cores, cores, state_tiles, local_valid, absolute_start));
    return desc;
}

ProgramDescriptor CsaCompressionProgramFactory::create_descriptor(
    const CsaRuntimeParams& params,
    const CsaCompressionInputs& args,
    std::array<Tensor, 3>& outputs,
    const std::optional<MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "CSA compression requires a mesh coordinate");
    const auto grid = args.kv.device()->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    TT_FATAL(grid_cores > 1, "CSA compression requires at least two worker cores");
    const uint32_t local_seq = args.kv.logical_shape()[-2];
    const auto [local_valid, absolute_start] = local_runtime(params, local_seq, *mesh_dispatch_coordinate);
    const uint32_t head_dim = factory_head_dim_of(args.kv);
    const uint32_t state_width_tiles = head_dim / tt::constants::TILE_WIDTH;
    const uint32_t input_width_tiles = 2 * state_width_tiles;
    const uint32_t output_height_tiles = (local_seq / 4 + 31) / 32;
    const uint32_t output_tiles = output_height_tiles * state_width_tiles;

    // Output tiles are independent -- the softmax runs per feature column and windows never interact --
    // so they split across the grid with no cross-core traffic. The state kernel needs cores of its
    // own because it is a RISCV_0 data movement kernel, which the reader already occupies, so the two
    // blocks are carved row-major with the state block starting where the compression block ends.
    const uint32_t state_tiles = 2 * state_width_tiles;
    const uint32_t state_core_count = std::min(state_tiles, grid_cores / 2);
    const uint32_t num_cores = std::min(output_tiles, grid_cores - state_core_count);
    const CoreRangeSet compression_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const std::vector<CoreCoord> cores = corerange_to_cores(compression_cores, num_cores, /*row_wise=*/true);
    const CoreCoord state_start = grid_to_cores(grid_cores, grid.x, grid.y, /*row_wise=*/true)[num_cores];
    const CoreRangeSet state_cores = num_cores_to_corerangeset(state_start, state_core_count, grid, /*row_wise=*/true);
    const std::vector<CoreCoord> state_core_list = corerange_to_cores(state_cores, state_core_count, /*row_wise=*/true);

    ProgramDescriptor desc;
    desc.cbs.push_back(cb_descriptor(kCandidateKvCb, 8, compression_cores));
    desc.cbs.push_back(cb_descriptor(kCandidateScoreCb, 8, compression_cores));
    desc.cbs.push_back(cb_descriptor(kPooledCb, 1, compression_cores));
    desc.cbs.push_back(cb_descriptor(kScratchCb, 4, compression_cores));
    desc.cbs.push_back(cb_descriptor(kMaxCb, 1, compression_cores));
    desc.cbs.push_back(cb_descriptor(kCaBiasCb, 1, compression_cores));
    desc.cbs.push_back(cb_descriptor(kCbBiasCb, 1, compression_cores));
    desc.cbs.push_back(cb_descriptor(kScratchCb, kStateScratchTiles, state_cores));

    std::vector<uint32_t> reader_compile_args = {
        kCandidateKvCb, kCandidateScoreCb, kScratchCb, input_width_tiles, state_width_tiles, kCaBiasCb, kCbBiasCb};
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

    KernelDescriptor compute;
    compute.kernel_source = kComputeKernel;
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = compression_cores;
    compute.compile_time_args = {kCandidateKvCb, kCandidateScoreCb, kPooledCb, kCaBiasCb, kCbBiasCb};
    compute.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true};

    std::vector<uint32_t> writer_compile_args = {kPooledCb};
    TensorAccessorArgs(outputs[0].buffer()).append_to(writer_compile_args);
    KernelDescriptor writer;
    writer.kernel_source = kWriterKernel;
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = compression_cores;
    writer.compile_time_args = std::move(writer_compile_args);
    writer.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default};

    reader.runtime_args.reserve(num_cores);
    compute.runtime_args.reserve(num_cores);
    writer.runtime_args.reserve(num_cores);
    const uint32_t base_tiles = output_tiles / num_cores;
    const uint32_t extra_tiles = output_tiles % num_cores;
    uint32_t first_tile = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t core_tiles = base_tiles + (i < extra_tiles ? 1 : 0);

        KernelDescriptor::RTArgList reader_runtime_args;
        reader_runtime_args.reserve(9);
        reader_runtime_args.push_back(args.kv.buffer());
        reader_runtime_args.push_back(args.gate.buffer());
        reader_runtime_args.push_back(args.position_bias.buffer());
        reader_runtime_args.push_back(args.predecessor_kv_state.buffer());
        reader_runtime_args.push_back(args.predecessor_score_state.buffer());
        reader_runtime_args.push_back(core_tiles);
        reader_runtime_args.push_back(local_valid / 4);
        reader_runtime_args.push_back(absolute_start);
        reader_runtime_args.push_back(first_tile);
        reader.emplace_runtime_args(core, reader_runtime_args);

        compute.emplace_runtime_args(core, {core_tiles});

        KernelDescriptor::RTArgList writer_runtime_args;
        writer_runtime_args.reserve(3);
        writer_runtime_args.push_back(outputs[0].buffer());
        writer_runtime_args.push_back(core_tiles);
        writer_runtime_args.push_back(first_tile);
        writer.emplace_runtime_args(core, writer_runtime_args);

        first_tile += core_tiles;
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));

    CsaStateInputs state_args{
        args.kv, args.gate, args.position_bias, args.predecessor_kv_state, args.predecessor_score_state};
    std::array<Tensor, 2> state_outputs{outputs[1], outputs[2]};
    desc.kernels.push_back(state_kernel_descriptor(
        state_args, state_outputs, state_cores, state_core_list, state_tiles, local_valid, absolute_start));
    return desc;
}

}  // namespace ttnn::experimental::prim
