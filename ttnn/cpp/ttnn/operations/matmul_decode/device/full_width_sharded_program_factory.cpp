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

namespace ttnn::operations::matmul_decode {

using namespace tt;
using namespace tt::tt_metal;

// Full width-sharded matmul: B and the output are width(N)-sharded across the
// core grid, with each core owning a contiguous slice of the N dimension. A is
// width(K)-sharded across a subset of cores ("senders"); since every core needs
// the full A to compute its N slice, the reader gathers A onto all cores.
//
// This sets up the in0 / in1 / out / full_in0 circular buffers, the gather
// semaphores, and the reader kernel. The reader multicasts each sender's A slice
// to every core (assembling full_in0) and publishes B, which is already resident
// in L1. Sender cores are split across both NoCs to balance multicast traffic.
//
// Still TODO to make it functional:
//   1. A compute kernel that does matmul_block over the gathered full A and this
//      core's B slice, accumulating over K, into the output CB.
//   2. A writer kernel (or sharded output CB handoff) to produce the output
//      width shard.
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
    const uint32_t in0_tile_size = tt::tile_size(in0_data_format);
    const uint32_t in1_tile_size = tt::tile_size(in1_data_format);
    const uint32_t out_tile_size = tt::tile_size(out_data_format);

    // Full output width (in tiles) to shard across the core grid.
    uint32_t M_tiles = div_up(operation_attributes.M, tt::constants::TILE_HEIGHT);
    uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);
    // uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);

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
        inputA_shard_shape[0] == (M_tiles * tt::constants::TILE_HEIGHT),
        "Input tensor A shard shape {} [0] must be equal to M_tiles {} * tile height {}",
        inputA_shard_shape,
        M_tiles,
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        inputA_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor A must have a width that is divisible by the tile width");
    uint32_t inA_K_tiles_per_core = inputA_shard_shape[1] / tt::constants::TILE_WIDTH;

    std::array<uint32_t, 2> inputB_shard_shape = input_tensor_b.memory_config().shard_spec().value().shape;
    TT_FATAL(
        inputB_shard_shape[0] == (K_tiles * tt::constants::TILE_HEIGHT),
        "Input tensor A shard shape {} [1] must be equal to K_tiles {} * tile height {}",
        inputA_shard_shape,
        K_tiles,
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        inputB_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor B must have a width that is divisible by the tile width");
    uint32_t inB_N_tiles_per_core = inputB_shard_shape[1] / tt::constants::TILE_WIDTH;
    ProgramDescriptor desc;

    // ---- Circular buffers (allocated on every participating core) ----
    // Input A Real Input
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
        }}},
    });
    // Two semaphores drive the gather:
    //   - `gather`: every sender atomically increments it on the coordinator
    //     (first) core after broadcasting its A slice.  The coordinator waits
    //     for it to reach num_senders.
    //   - `done`: the coordinator multicasts it to all cores once every slice
    //     has arrived, signalling that the full A matrix is available.
    // Both live on the full mcast rectangle so they are addressable on every
    // core that references them (including padding cores inside the box).
    const uint32_t num_senders = inputA_core_range_set.num_cores();
    constexpr uint32_t gather_sem_id = 0;
    constexpr uint32_t done_sem_id = 1;
    log_debug(tt::LogOp, "MatmulDecode: num_senders: {}", num_senders);
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = gather_sem_id,
        .core_ranges = all_compute_cores_with_bbox,
        .initial_value = 0,
    });
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = done_sem_id,
        .core_ranges = all_compute_cores_with_bbox,
        .initial_value = 0,
    });

    // ---- Reader kernel ----
    //
    // Runs on every core in the mcast rectangle.  Sender cores (those that hold
    // a K-slice of A) broadcast their slice into `full_in0_cb` on all cores and
    // increment the coordinator's `gather` semaphore; the coordinator then
    // broadcasts the `done` semaphore so every core knows A is fully gathered.
    const CoreRange mcast_bbox = all_compute_cores_with_bbox.bounding_box();
    // The coordinator (gather/broadcast hub) is the first core of the rectangle.
    const CoreCoord coordinator_logical = mcast_bbox.start_coord;
    const CoreCoord mcast_start_phys = device->worker_core_from_logical_core(coordinator_logical);
    const CoreCoord mcast_end_phys = device->worker_core_from_logical_core(mcast_bbox.end_coord);
    const uint32_t num_receivers = all_compute_cores_with_bbox.num_cores();
    const uint32_t shard_num_tiles = M_tiles * inA_K_tiles_per_core;

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
        gather_sem_id,
        done_sem_id,
        // Coordinator's physical (worker) coords == rectangle start corner.
        static_cast<uint32_t>(mcast_start_phys.x),
        static_cast<uint32_t>(mcast_start_phys.y),
        // in1 (B), already resident in L1.
        in1_cb_index,
        K_tiles * inB_N_tiles_per_core,
    };

    // Map each A-holding core to its K-slice index (== semaphore id).  Cores are
    // walked in row-major order so the slice ordering matches the width-sharded
    // layout of input A across `inputA_core_range_set`.
    const std::vector<CoreCoord> sender_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, true);
    std::map<CoreCoord, uint32_t> sender_id_by_core;
    for (uint32_t id = 0; id < sender_cores.size(); id++) {
        sender_id_by_core[sender_cores[id]] = id;
    }

    // Balance the multicasting sender cores across both NoCs; all other cores
    // stay on the default NoC.
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
            "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/dataflow/reader_full_width_sharded.cpp";
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
            const bool is_coordinator = (core == coordinator_logical);
            reader_kernel_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    static_cast<uint32_t>(is_sender), sender_id, static_cast<uint32_t>(is_coordinator)});
        }
        return reader_kernel_desc;
    };

    // Sender cores: split in half across NOC0 / NOC1.
    const size_t num_noc0_senders = sender_cores.size() / 2;
    const std::vector<CoreCoord> noc0_sender_cores(sender_cores.begin(), sender_cores.begin() + num_noc0_senders);
    const std::vector<CoreCoord> noc1_sender_cores(sender_cores.begin() + num_noc0_senders, sender_cores.end());

    // Every remaining (non-sender) core uses the reader's default NoC.
    std::vector<CoreCoord> default_noc_cores;
    default_noc_cores.reserve(all_reader_cores.size());
    for (const auto& core : all_reader_cores) {
        if (sender_id_by_core.find(core) == sender_id_by_core.end()) {
            default_noc_cores.push_back(core);
        }
    }

    if (!noc0_sender_cores.empty()) {
        desc.kernels.push_back(build_reader_kernel(noc0_sender_cores, NOC::NOC_0));
    }
    if (!noc1_sender_cores.empty()) {
        desc.kernels.push_back(build_reader_kernel(noc1_sender_cores, NOC::NOC_1));
    }
    if (!default_noc_cores.empty()) {
        desc.kernels.push_back(build_reader_kernel(default_noc_cores, NOC::RISCV_1_default));
    }

    // ---- Compute kernel ----
    //
    // Matmul over gathered full A and this core's B slice.
    // Blocking: in0_block_w (K) = inA_K_tiles_per_core, out_block_h (M) = M_tiles,
    // out_block_w (N) = 1. The compute kernel processes the entire M dimension in
    // a single DST block (out_block_h = M_tiles), so M_tiles must fit in DST
    // (<= 8 tiles in half-sync mode). Enforce M < 256 (=> M_tiles <= 8).
    TT_FATAL(
        operation_attributes.M < 256,
        "full_width_sharded matmul_decode requires M < 256 so that out_block_h (= M_tiles) stays < 8 and fits in DST, "
        "but got M={}",
        operation_attributes.M);

    log_debug(
        tt::LogOp,
        "MatmulDecode: M_tiles: {}, K_tiles: {}, inB_N_tiles_per_core: {}, inA_K_tiles_per_core: {}",
        M_tiles,
        K_tiles,
        inB_N_tiles_per_core,
        inA_K_tiles_per_core);
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/compute/compute_full_width_sharded.cpp";
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

}  // namespace ttnn::operations::matmul_decode
