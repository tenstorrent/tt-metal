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

// Partial width-sharded: B is sharded along K and N; K-partials reduce onto base cores.
ProgramDescriptor MatmulDecodeDeviceOperation::PartialWidthSharded::create_descriptor(
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

    log_debug(
        tt::LogOp,
        "MatmulDecode(partial): in0_tile_size: {}, in1_tile_size: {}, out_tile_size: {}",
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

    log_debug(tt::LogOp, "MatmulDecode(partial): inputA_tile: {}", inputA_tile);

    IDevice* device = input_tensor_a.device();

    const uint32_t M_tiles = div_up(operation_attributes.M, inputA_tile_height);
    const uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);
    const uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);

    TT_FATAL(
        M_tiles <= 8,
        "partial_width_sharded matmul_decode requires out_block_h (= M_tiles) <= 8 so it fits in DST, but got "
        "M_tiles={} (M={}, inputA_tile_height={})",
        M_tiles,
        operation_attributes.M,
        inputA_tile_height);

    const std::array<uint32_t, 2> inputA_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;
    TT_FATAL(
        inputA_shard_shape[0] == (M_tiles * inputA_tile_height),
        "Input tensor A shard height {} must equal M_tiles {} * tile height {}",
        inputA_shard_shape[0],
        M_tiles,
        inputA_tile_height);
    TT_FATAL(
        inputA_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor A shard width must be divisible by the tile width");
    const uint32_t inA_K_tiles_per_core = inputA_shard_shape[1] / tt::constants::TILE_WIDTH;

    const std::array<uint32_t, 2> inputB_shard_shape = input_tensor_b.memory_config().shard_spec().value().shape;
    const uint32_t Kc = inputB_shard_shape[0];
    const uint32_t Nc = inputB_shard_shape[1];
    const uint32_t Kc_tiles = Kc / tt::constants::TILE_WIDTH;
    const uint32_t Nc_tiles = Nc / tt::constants::TILE_WIDTH;

    const auto inputA_core_range_set = input_tensor_a.memory_config().shard_spec().value().grid;
    const auto inputB_core_range_set = input_tensor_b.memory_config().shard_spec().value().grid;
    const auto output_core_range_set = output_tensor.memory_config().shard_spec().value().grid;

    const uint32_t num_B_cores = inputB_core_range_set.num_cores();
    const uint32_t num_B_cores_along_N = N_tiles / Nc_tiles;
    TT_FATAL(
        num_B_cores % num_B_cores_along_N == 0,
        "num_B_cores {} must be divisible by num_B_cores_along_N {}",
        num_B_cores,
        num_B_cores_along_N);
    const uint32_t num_B_cores_along_K = num_B_cores / num_B_cores_along_N;
    const uint32_t K_blocks = K_tiles / Kc_tiles;
    TT_FATAL(
        num_B_cores_along_K == K_blocks,
        "num_B_cores_along_K {} must equal K_blocks {}",
        num_B_cores_along_K,
        K_blocks);
    const uint32_t N_blocks = num_B_cores / K_blocks;
    TT_FATAL(
        output_core_range_set.num_cores() == N_blocks,
        "Output must be sharded across N_blocks {} cores, but got {}",
        N_blocks,
        output_core_range_set.num_cores());

    log_debug(
        tt::LogOp,
        "num_B_cores: {}, num_B_cores_along_N: {}, num_B_cores_along_K: {}, K_blocks: {}, N_blocks: {}",
        num_B_cores,
        num_B_cores_along_N,
        num_B_cores_along_K,
        K_blocks,
        N_blocks);
    log_debug(
        tt::LogOp,
        "inputA_num_cores: {}, inputB_num_cores: {}, output_num_cores: {}",
        inputA_core_range_set.num_cores(),
        inputB_core_range_set.num_cores(),
        output_core_range_set.num_cores());
    const auto all_compute_cores = inputA_core_range_set.merge(inputB_core_range_set).merge(output_core_range_set);
    const auto all_compute_cores_with_bbox = tt::tt_metal::CoreRangeSet(all_compute_cores.bounding_box());

    log_debug(
        tt::LogOp,
        "MatmulDecode(partial): M_tiles={}, K_tiles={}, Kc_tiles={}, Nc_tiles={}, K_blocks={}, N_blocks={}",
        M_tiles,
        K_tiles,
        Kc_tiles,
        Nc_tiles,
        K_blocks,
        N_blocks);

    ProgramDescriptor desc;

    constexpr uint32_t in0_cb_index = CBIndex::c_0;       // this core's A slice (gather source)
    constexpr uint32_t in1_cb_index = CBIndex::c_1;       // this core's B block (resident)
    constexpr uint32_t out_cb_index = CBIndex::c_2;       // final output shard (base cores)
    constexpr uint32_t full_in0_cb_index = CBIndex::c_3;  // gathered full A
    constexpr uint32_t partial_cb_index = CBIndex::c_4;   // this core's partial product
    constexpr uint32_t reduce_cb_index = CBIndex::c_5;    // gathered K_blocks partials (base cores)

    const uint32_t block_num_tiles = M_tiles * Nc_tiles;  // tiles in one (partial / output) shard

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
        .total_size = Kc_tiles * Nc_tiles * in1_tile_size,
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
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = output_core_range_set,
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
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = partial_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });
    // reduce_cb is at the same L1 offset on every core (writers use local write ptr as remote dst).
    desc.cbs.push_back(CBDescriptor{
        .total_size = K_blocks * block_num_tiles * out_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = reduce_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });

    // stage/done: two-hub A gather (see reader). reduce_sem: partial count on base core.
    const uint32_t num_senders = inputA_core_range_set.num_cores();
    constexpr uint32_t stage_sem_id = 0;
    constexpr uint32_t done_sem_id = 1;
    constexpr uint32_t reduce_sem_id = 2;
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = stage_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = done_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = reduce_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});

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
        "partial_width_sharded matmul_decode two-hub broadcast requires a compute rectangle of at least 2 cores");

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
        Kc_tiles * Nc_tiles,
    };

    const std::vector<CoreCoord> sender_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, true);
    std::map<CoreCoord, uint32_t> sender_id_by_core;
    for (uint32_t id = 0; id < sender_cores.size(); id++) {
        sender_id_by_core[sender_cores[id]] = id;
    }
    const std::vector<CoreCoord> all_reader_cores = corerange_to_cores(all_compute_cores_with_bbox, std::nullopt, true);

    auto role_of = [&](const CoreCoord& core) -> HubRole {
        if (core == hub0_logical) {
            return HubRole::Hub0;
        }
        if (core == hub1_logical) {
            return HubRole::Hub1;
        }
        return HubRole::Plain;
    };

    auto build_reader_kernel = [&](const std::vector<CoreCoord>& cores, NOC noc) {
        std::vector<CoreRange> ranges;
        ranges.reserve(cores.size());
        for (const auto& core : cores) {
            ranges.emplace_back(core, core);
        }
        KernelDescriptor reader_kernel_desc;
        reader_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
            "reader_partial_width_sharded.cpp";
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

    // Pick NOC by traffic direction (same rules as full-width reader).
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

    // Writer uses the NOC opposite the reader on the same core.
    std::map<CoreCoord, NOC> reader_noc_by_core;
    for (const auto& core : noc0_cores) {
        reader_noc_by_core[core] = NOC::NOC_0;
    }
    for (const auto& core : noc1_cores) {
        reader_noc_by_core[core] = NOC::NOC_1;
    }
    for (const auto& core : default_noc_cores) {
        reader_noc_by_core[core] = NOC::RISCV_1_default;
    }

    const std::vector<CoreCoord> b_cores = corerange_to_cores(inputB_core_range_set, std::nullopt, true);
    log_debug(tt::LogOp, "b_cores: {}", b_cores);
    log_debug(tt::LogOp, "output_core_range_set: {}", output_core_range_set);
    log_debug(tt::LogOp, "inputB_core_range_set: {}", inputB_core_range_set);
    log_debug(tt::LogOp, "inputA_core_range_set: {}", inputA_core_range_set);
    std::vector<CoreRange> b_core_ranges;
    b_core_ranges.reserve(b_cores.size());
    for (const auto& core : b_cores) {
        b_core_ranges.emplace_back(core, core);
    }

    auto build_writer_kernel = [&](const std::vector<uint32_t>& core_indices, NOC noc) {
        std::vector<CoreRange> ranges;
        ranges.reserve(core_indices.size());
        for (const uint32_t idx : core_indices) {
            ranges.emplace_back(b_cores[idx], b_cores[idx]);
        }
        KernelDescriptor writer_kernel_desc;
        writer_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
            "writer_partial_width_sharded.cpp";
        writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_kernel_desc.core_ranges = CoreRangeSet(ranges);
        writer_kernel_desc.compile_time_args = {
            partial_cb_index,
            reduce_cb_index,
            block_num_tiles,
            out_tile_size,
            K_blocks,
            reduce_sem_id,
        };
        writer_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = noc,
        };
        writer_kernel_desc.runtime_args.reserve(core_indices.size());
        for (const uint32_t idx : core_indices) {
            const uint32_t k_idx = idx / N_blocks;
            const uint32_t n_idx = idx % N_blocks;
            const CoreCoord base_logical = b_cores[n_idx];
            const CoreCoord base_phys = device->worker_core_from_logical_core(base_logical);
            const bool is_base = (k_idx == 0);
            log_trace(
                tt::LogOp,
                "Writer core {}, idx {}, k_idx: {}, n_idx: {}, base_logical: {}, base_phys: {}, is_base: {}",
                b_cores[idx],
                idx,
                k_idx,
                n_idx,
                base_logical,
                base_phys,
                is_base);
            writer_kernel_desc.runtime_args.emplace_back(
                b_cores[idx],
                KernelDescriptor::CoreRuntimeArgs{
                    k_idx,
                    static_cast<uint32_t>(base_phys.x),
                    static_cast<uint32_t>(base_phys.y),
                    static_cast<uint32_t>(is_base)});
        }
        return writer_kernel_desc;
    };

    std::vector<uint32_t> writer_noc0_indices;
    std::vector<uint32_t> writer_noc1_indices;
    for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
        NOC reader_noc = NOC::RISCV_1_default;
        const auto it = reader_noc_by_core.find(b_cores[idx]);
        if (it != reader_noc_by_core.end()) {
            reader_noc = it->second;
        }
        log_trace(tt::LogOp, "core {}, idx: {}, reader_noc: {}", b_cores[idx], idx, reader_noc);
        if (reader_noc == NOC::NOC_0) {
            writer_noc1_indices.push_back(idx);
        } else {
            writer_noc0_indices.push_back(idx);
        }
    }
    if (!writer_noc0_indices.empty()) {
        desc.kernels.push_back(build_writer_kernel(writer_noc0_indices, NOC::NOC_0));
    }
    if (!writer_noc1_indices.empty()) {
        desc.kernels.push_back(build_writer_kernel(writer_noc1_indices, NOC::NOC_1));
    }

    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/compute_partial_width_sharded.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = CoreRangeSet(b_core_ranges);
    compute_kernel_desc.compile_time_args = {
        M_tiles,
        K_tiles,
        Kc_tiles,
        Nc_tiles,
        K_blocks,
        inA_K_tiles_per_core,
    };
    log_debug(
        tt::LogOp,
        "M_tiles: {}, K_tiles: {}, Kc_tiles: {}, Nc_tiles: {}, K_blocks: {}",
        M_tiles,
        K_tiles,
        Kc_tiles,
        Nc_tiles,
        K_blocks);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    compute_kernel_desc.runtime_args.reserve(b_cores.size());
    for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
        const uint32_t k_idx = idx / N_blocks;
        const bool is_base = (k_idx == 0);
        if (is_base) {
            TT_FATAL(
                output_core_range_set.contains(b_cores[idx]),
                "Base core {} is not in output core range set",
                b_cores[idx]);
        }
        log_trace(
            tt::LogOp,
            "core {}, idx {}, k_idx: {}, is_base: {}",
            b_cores[idx],
            idx,
            k_idx,
            static_cast<uint32_t>(is_base));
        compute_kernel_desc.runtime_args.emplace_back(
            b_cores[idx], KernelDescriptor::CoreRuntimeArgs{k_idx, static_cast<uint32_t>(is_base)});
    }
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::operations::experimental::matmul_decode
