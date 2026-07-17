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

// Full width-sharded: B/output are width(N)-sharded; reader gathers full A onto every core.
ProgramDescriptor MatmulDecodeDeviceOperation::FullWidthSharded::create_descriptor(
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

    // Tiny tiles can give in0/in1/out different geometries; each CB needs its own tile descriptor.
    const TileDescriptor in0_tile_desc{inputA_tile};
    const TileDescriptor in1_tile_desc{inputB_tile};
    const TileDescriptor out_tile_desc{output_tile};

    log_debug(
        tt::LogOp,
        "MatmulDecode: in0_tile_size: {}, in1_tile_size: {}, out_tile_size: {}",
        in0_tile_size,
        in1_tile_size,
        out_tile_size);

    const uint32_t inputA_tile_height = inputA_tile.get_height();
    const uint32_t inputA_tile_width = inputA_tile.get_width();
    const uint32_t inputB_tile_height = inputB_tile.get_height();
    const uint32_t inputB_tile_width = inputB_tile.get_width();
    const uint32_t output_tile_height = output_tile.get_height();
    const uint32_t output_tile_width = output_tile.get_width();

    TT_FATAL(
        inputA_tile_height == output_tile_height,
        "Input tensor A {} and output tile height {} must be equal",
        inputA_tile_height,
        output_tile_height);

    TT_FATAL(
        inputB_tile_height == tt::constants::TILE_HEIGHT,
        "Input tensor B {} tile height must be 32",
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

    log_debug(tt::LogOp, "MatmulDecode: inputA_tile: {}", inputA_tile);

    uint32_t M_tiles = div_up(operation_attributes.M, inputA_tile_height);
    uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);

    IDevice* device = input_tensor_a.device();
    auto inputA_core_range_set = input_tensor_a.memory_config().shard_spec().value().grid;
    auto inputB_core_range_set = input_tensor_b.memory_config().shard_spec().value().grid;
    auto output_core_range_set = output_tensor.memory_config().shard_spec().value().grid;
    TT_FATAL(
        inputB_core_range_set == output_core_range_set,
        "Input tensor A and output tensor must have the same core range set");

    auto all_compute_cores = inputA_core_range_set.merge(output_core_range_set);
    auto all_compute_cores_with_bbox = tt::tt_metal::CoreRangeSet(all_compute_cores.bounding_box());

    log_debug(tt::LogOp, "MatmulDecode: all_compute_cores: {}", all_compute_cores_with_bbox.str());

    std::array<uint32_t, 2> inputA_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;

    TT_FATAL(
        inputA_shard_shape[0] == (M_tiles * inputA_tile_height),
        "Input tensor A shard shape {} [0] must be equal to M_tiles {} * tile height {}",
        inputA_shard_shape[0],
        M_tiles,
        inputA_tile_height);
    TT_FATAL(
        inputA_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor A must have a width that is divisible by the tile width");
    uint32_t inA_K_tiles_per_core = inputA_shard_shape[1] / tt::constants::TILE_WIDTH;

    std::array<uint32_t, 2> inputB_shard_shape = input_tensor_b.memory_config().shard_spec().value().shape;
    TT_FATAL(
        inputB_shard_shape[0] == (K_tiles * tt::constants::TILE_HEIGHT),
        "Input tensor B shard shape {} [0] must be equal to K_tiles {} * tile height {}",
        inputB_shard_shape[0],
        K_tiles,
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        inputB_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor B must have a width that is divisible by the tile width");
    uint32_t inB_N_tiles_per_core = inputB_shard_shape[1] / tt::constants::TILE_WIDTH;
    ProgramDescriptor desc;

    constexpr uint32_t in0_cb_index = CBIndex::c_0;
    constexpr uint32_t in1_cb_index = CBIndex::c_1;
    constexpr uint32_t out_cb_index = CBIndex::c_2;
    constexpr uint32_t full_in0_cb_index = CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = M_tiles * inA_K_tiles_per_core * in0_tile_size,
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
        .total_size = K_tiles * inB_N_tiles_per_core * in1_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_tile_size,
            .tile = in1_tile_desc,
        }}},
        .buffer = input_tensor_b.buffer(),
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = M_tiles * inB_N_tiles_per_core * out_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
        .buffer = output_tensor.buffer(),
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = M_tiles * K_tiles * in0_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = full_in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
    });
    // stage: senders bump owning hub after slice write; done: each hub bumps all cores after its mcast half.
    const uint32_t num_senders = inputA_core_range_set.num_cores();
    constexpr uint32_t stage_sem_id = 0;
    constexpr uint32_t done_sem_id = 1;
    log_debug(tt::LogOp, "MatmulDecode: num_senders: {}", num_senders);
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = stage_sem_id,
        .core_ranges = all_compute_cores_with_bbox,
        .initial_value = 0,
    });
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = done_sem_id,
        .core_ranges = all_compute_cores_with_bbox,
        .initial_value = 0,
    });

    const CoreRange mcast_bbox = all_compute_cores_with_bbox.bounding_box();
    const CoreCoord hub0_logical = mcast_bbox.start_coord;
    const CoreCoord hub1_logical = mcast_bbox.end_coord;
    const CoreCoord mcast_start_phys = device->worker_core_from_logical_core(hub0_logical);
    const CoreCoord mcast_end_phys = device->worker_core_from_logical_core(hub1_logical);
    const uint32_t num_receivers = all_compute_cores_with_bbox.num_cores();
    const uint32_t shard_num_tiles = M_tiles * inA_K_tiles_per_core;
    const uint32_t split_H = num_senders / 2;

    TT_FATAL(
        num_receivers >= 2 && hub0_logical != hub1_logical,
        "full_width_sharded matmul_decode two-hub broadcast requires a compute rectangle of at least 2 cores");

    const KernelDescriptor::CompileTimeArgs reader_compile_time_args = {
        in0_cb_index,
        full_in0_cb_index,
        shard_num_tiles,
        in0_tile_size,
        num_senders,
        num_receivers,
        static_cast<uint32_t>(mcast_start_phys.x),
        static_cast<uint32_t>(mcast_start_phys.y),
        static_cast<uint32_t>(mcast_end_phys.x),
        static_cast<uint32_t>(mcast_end_phys.y),
        stage_sem_id,
        done_sem_id,
        static_cast<uint32_t>(mcast_start_phys.x),
        static_cast<uint32_t>(mcast_start_phys.y),
        static_cast<uint32_t>(mcast_end_phys.x),
        static_cast<uint32_t>(mcast_end_phys.y),
        split_H,
        in1_cb_index,
        K_tiles * inB_N_tiles_per_core,
    };

    const std::vector<CoreCoord> sender_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, true);
    std::map<CoreCoord, uint32_t> sender_id_by_core;
    for (uint32_t id = 0; id < sender_cores.size(); id++) {
        sender_id_by_core[sender_cores[id]] = id;
    }

    auto role_of = [&](const CoreCoord& core) -> HubRole {
        if (core == hub0_logical) {
            return HubRole::Hub0;
        }
        if (core == hub1_logical) {
            return HubRole::Hub1;
        }
        return HubRole::Plain;
    };

    const std::vector<CoreCoord> all_reader_cores = corerange_to_cores(all_compute_cores_with_bbox, std::nullopt, true);

    auto build_reader_kernel = [&](const std::vector<CoreCoord>& cores, NOC noc) {
        log_debug(tt::LogOp, "MatmulDecode: building reader kernel for cores: {} on noc: {}", cores, noc);
        std::vector<CoreRange> ranges;
        ranges.reserve(cores.size());
        for (const auto& core : cores) {
            ranges.emplace_back(core, core);
        }

        KernelDescriptor reader_kernel_desc;
        reader_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/reader_full_width_sharded.cpp";
        reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_kernel_desc.core_ranges = CoreRangeSet(ranges);
        reader_kernel_desc.compile_time_args = reader_compile_time_args;
        reader_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = noc,
        };

        reader_kernel_desc.runtime_args.reserve(cores.size());
        for (const auto& core : cores) {
            const auto it = sender_id_by_core.find(core);
            const bool is_sender = it != sender_id_by_core.end();
            const uint32_t sender_id = is_sender ? it->second : 0;
            reader_kernel_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    static_cast<uint32_t>(is_sender), sender_id, static_cast<uint32_t>(role_of(core))});
        }
        return reader_kernel_desc;
    };

    // Pick NOC by traffic direction: hub0 mcasts on NOC0, hub1 on NOC1; senders use the NOC toward their hub.
    std::vector<CoreCoord> noc0_cores;
    std::vector<CoreCoord> noc1_cores;
    std::vector<CoreCoord> default_noc_cores;
    for (const auto& core : all_reader_cores) {
        const HubRole role = role_of(core);
        if (role == HubRole::Hub0) {
            noc0_cores.push_back(core);
            continue;
        }
        if (role == HubRole::Hub1) {
            noc1_cores.push_back(core);
            continue;
        }
        const auto it = sender_id_by_core.find(core);
        if (it == sender_id_by_core.end()) {
            default_noc_cores.push_back(core);
        } else if (it->second < split_H) {
            noc1_cores.push_back(core);
        } else {
            noc0_cores.push_back(core);
        }
    }

    if (!noc0_cores.empty()) {
        desc.kernels.push_back(build_reader_kernel(noc0_cores, NOC::NOC_0));
    }
    if (!noc1_cores.empty()) {
        desc.kernels.push_back(build_reader_kernel(noc1_cores, NOC::NOC_1));
    }
    if (!default_noc_cores.empty()) {
        desc.kernels.push_back(build_reader_kernel(default_noc_cores, NOC::RISCV_1_default));
    }

    TT_FATAL(
        M_tiles <= 8,
        "full_width_sharded matmul_decode requires out_block_h (= M_tiles) <= 8 so it fits in DST, but got M_tiles={} "
        "(M={}, inputA_tile_height={})",
        M_tiles,
        operation_attributes.M,
        inputA_tile_height);

    log_debug(
        tt::LogOp,
        "MatmulDecode: M_tiles: {}, K_tiles: {}, inB_N_tiles_per_core: {}, inA_K_tiles_per_core: {}",
        M_tiles,
        K_tiles,
        inB_N_tiles_per_core,
        inA_K_tiles_per_core);
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/compute_full_width_sharded.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = output_core_range_set;
    compute_kernel_desc.compile_time_args = {
        M_tiles,
        K_tiles,
        inB_N_tiles_per_core,
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
