// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/shape.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <map>
#include <optional>
#include <vector>

namespace ttnn::operations::experimental::matmul_decode {

using namespace tt;
using namespace tt::tt_metal;

// Batched width-sharded: weights folded along batch and N; block-diagonal, no cross-core reduction.
ProgramDescriptor MatmulDecodeDeviceOperation::BatchedWidthSharded::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    const tt::DataFormat in0_data_format = datatype_to_dataformat_converter(input_tensor_a.dtype());
    const tt::DataFormat in1_data_format = datatype_to_dataformat_converter(input_tensor_b.dtype());
    const tt::DataFormat out_data_format = datatype_to_dataformat_converter(output_tensor.dtype());

    const auto& inputA_tile = input_tensor_a.tensor_spec().tile();
    const auto& inputB_tile = input_tensor_b.tensor_spec().tile();
    const auto& output_tile = output_tensor.tensor_spec().tile();
    const uint32_t in0_tile_size = inputA_tile.get_tile_size(in0_data_format);
    const uint32_t in1_tile_size = inputB_tile.get_tile_size(in1_data_format);
    const uint32_t out_tile_size = output_tile.get_tile_size(out_data_format);

    const TileDescriptor in0_tile_desc{inputA_tile};
    const TileDescriptor in1_tile_desc{inputB_tile};
    const TileDescriptor out_tile_desc{output_tile};

    const uint32_t inputA_tile_height = inputA_tile.get_height();
    const uint32_t inputA_tile_width = inputA_tile.get_width();
    const uint32_t inputB_tile_height = inputB_tile.get_height();
    const uint32_t inputB_tile_width = inputB_tile.get_width();
    const uint32_t output_tile_height = output_tile.get_height();
    const uint32_t output_tile_width = output_tile.get_width();

    TT_FATAL(
        inputA_tile_height == output_tile_height,
        "Input tensor A tile height {} and output tile height {} must be equal",
        inputA_tile_height,
        output_tile_height);
    TT_FATAL(
        inputB_tile_height == tt::constants::TILE_HEIGHT,
        "Input tensor B tile height {} must be 32",
        inputB_tile_height);
    TT_FATAL(
        inputA_tile_width == tt::constants::TILE_WIDTH,
        "Input tensor A tile width {} must be equal to the tile width 32",
        inputA_tile_width);
    TT_FATAL(
        inputB_tile_width == tt::constants::TILE_WIDTH,
        "Input tensor B tile width {} must be equal to the tile width 32",
        inputB_tile_width);
    TT_FATAL(
        output_tile_width == tt::constants::TILE_WIDTH,
        "Output tensor tile width {} must be equal to the tile width 32",
        output_tile_width);

    const uint32_t batch = operation_attributes.batch;
    const uint32_t b_blocks = operation_attributes.b_blocks;
    const uint32_t n_blocks = operation_attributes.n_blocks;
    const uint32_t Bc = batch / b_blocks;                   // batches per core
    const uint32_t Nc = operation_attributes.N / n_blocks;  // N per core

    const uint32_t M_tiles = div_up(operation_attributes.M, inputA_tile_height);
    const uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);
    const uint32_t Nc_tiles = div_up(Nc, tt::constants::TILE_WIDTH);

    TT_FATAL(
        M_tiles <= 8,
        "batched matmul_decode requires out_block_h (= M_tiles) <= 8 so it fits in DST, but got M_tiles={} (M={}, "
        "inputA_tile_height={})",
        M_tiles,
        operation_attributes.M,
        inputA_tile_height);

    const std::array<uint32_t, 2> inputA_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;
    TT_FATAL(
        inputA_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor A shard width {} must be divisible by the tile width {}",
        inputA_shard_shape[1],
        tt::constants::TILE_WIDTH);
    const uint32_t inA_K_tiles_per_core = inputA_shard_shape[1] / tt::constants::TILE_WIDTH;

    const std::array<uint32_t, 2> inputB_shard_shape = input_tensor_b.memory_config().shard_spec().value().shape;
    const uint32_t b_shard_K_tiles = inputB_shard_shape[0] / tt::constants::TILE_HEIGHT;  // = Bc * K_tiles

    const auto inputA_core_range_set = input_tensor_a.memory_config().shard_spec().value().grid;
    const auto inputB_core_range_set = input_tensor_b.memory_config().shard_spec().value().grid;

    const uint32_t num_B_cores = inputB_core_range_set.num_cores();
    TT_FATAL(
        num_B_cores == b_blocks * n_blocks,
        "batched matmul_decode expects B sharded across b_blocks * n_blocks = {} * {} = {} cores, but got {}",
        b_blocks,
        n_blocks,
        b_blocks * n_blocks,
        num_B_cores);

    const auto all_compute_cores = inputA_core_range_set.merge(inputB_core_range_set);
    const auto all_compute_cores_with_bbox = tt::tt_metal::CoreRangeSet(all_compute_cores.bounding_box());

    log_debug(
        tt::LogOp,
        "MatmulDecode(batched): batch={}, b_blocks={}, n_blocks={}, Bc={}, Nc={}, M_tiles={}, K_tiles={}, "
        "Nc_tiles={}, num_B_cores={}",
        batch,
        b_blocks,
        n_blocks,
        Bc,
        Nc,
        M_tiles,
        K_tiles,
        Nc_tiles,
        num_B_cores);

    IDevice* device = input_tensor_a.device();
    const uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);

    ProgramDescriptor desc;

    constexpr uint32_t in0_cb_index = CBIndex::c_0;       // this core's A slice (gather source)
    constexpr uint32_t in1_cb_index = CBIndex::c_1;       // this core's weight block (resident)
    constexpr uint32_t out_cb_index = CBIndex::c_2;       // this core's output block (compute -> writer)
    constexpr uint32_t full_in0_cb_index = CBIndex::c_3;  // gathered full A

    const uint32_t out_block_num_tiles = Bc * M_tiles * Nc_tiles;
    const uint32_t full_in0_num_tiles = Bc * M_tiles * K_tiles;
    const uint32_t a_shard_tiles = batch * M_tiles * inA_K_tiles_per_core;
    const uint32_t block_slice_tiles = Bc * M_tiles * inA_K_tiles_per_core;
    const uint32_t in1_num_tiles = b_shard_K_tiles * Nc_tiles;

    const std::vector<CoreCoord> b_cores = corerange_to_cores(inputB_core_range_set, std::nullopt, true);
    std::vector<CoreRange> b_core_ranges;
    b_core_ranges.reserve(b_cores.size());
    for (const auto& core : b_cores) {
        b_core_ranges.emplace_back(core, core);
    }

    desc.cbs.push_back(CBDescriptor{
        .total_size = a_shard_tiles * in0_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
        .buffer = input_tensor_a.buffer(),
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_num_tiles * in1_tile_size,
        .core_ranges = inputB_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_tile_size,
            .tile = in1_tile_desc,
        }}},
        .buffer = input_tensor_b.buffer(),
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_block_num_tiles * out_tile_size,
        .core_ranges = inputB_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = full_in0_num_tiles * in0_tile_size,
        .core_ranges = inputB_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = full_in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
    });

    const uint32_t num_senders = inputA_core_range_set.num_cores();
    const std::vector<CoreCoord> sender_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, true);
    std::vector<uint32_t> sender_phys_coords;
    sender_phys_coords.reserve(2 * num_senders);
    for (const auto& sender : sender_cores) {
        const CoreCoord phys = device->worker_core_from_logical_core(sender);
        sender_phys_coords.push_back(static_cast<uint32_t>(phys.x));
        sender_phys_coords.push_back(static_cast<uint32_t>(phys.y));
    }

    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
        "reader_batched_width_sharded.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = CoreRangeSet(b_core_ranges);
    reader_kernel_desc.compile_time_args = {
        in0_cb_index,
        full_in0_cb_index,
        block_slice_tiles,
        in0_tile_size,
        num_senders,
        in1_cb_index,
        in1_num_tiles,
    };
    reader_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
    };
    reader_kernel_desc.runtime_args.reserve(b_cores.size());
    for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
        const uint32_t b_idx = idx / n_blocks;
        KernelDescriptor::CoreRuntimeArgs args;
        args.reserve(1 + sender_phys_coords.size());
        args.push_back(b_idx);
        args.insert(args.end(), sender_phys_coords.begin(), sender_phys_coords.end());
        reader_kernel_desc.runtime_args.emplace_back(b_cores[idx], std::move(args));
    }
    desc.kernels.push_back(std::move(reader_kernel_desc));

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
        "writer_batched_width_sharded.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = CoreRangeSet(b_core_ranges);
    writer_kernel_desc.compile_time_args = {
        out_cb_index,
        Bc,
        M_tiles,
        Nc_tiles,
        N_tiles,
    };
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_kernel_desc.compile_time_args);
    writer_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
    };
    for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
        const uint32_t b_idx = idx / n_blocks;
        const uint32_t n_idx = idx % n_blocks;
        writer_kernel_desc.emplace_runtime_args(
            b_cores[idx], {output_tensor.buffer(), static_cast<uint32_t>(b_idx), static_cast<uint32_t>(n_idx)});
    }
    desc.kernels.push_back(std::move(writer_kernel_desc));

    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/compute_batched_width_sharded.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = CoreRangeSet(b_core_ranges);
    compute_kernel_desc.compile_time_args = {
        M_tiles,
        K_tiles,
        Nc_tiles,
        Bc,
        inA_K_tiles_per_core,
    };
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::operations::experimental::matmul_decode
