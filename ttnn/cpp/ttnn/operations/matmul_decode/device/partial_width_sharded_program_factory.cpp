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

// Partial width-sharded matmul: C = A @ B, where B is sharded along BOTH K and N.
//
// The caller reshapes/permutes a [K, N] weight into a width-sharded tensor whose
// shard shape is [Kc, Nc] across K_blocks * N_blocks cores (Kc = K / K_blocks,
// Nc = N / N_blocks). Cores are laid out k-major in row-major order, so the core at
// row-major index `c` owns B block (k_idx = c / N_blocks, n_idx = c % N_blocks).
//
// Pipeline (per core):
//   1. Reader (reused reader_full_width_sharded): gather the *entire* A matrix onto
//      every core via multicast, and publish this core's resident B block.
//   2. Compute (phase 1): matmul this core's K-slice of A with its B block to produce
//      a *partial* [M, Nc] product.
//   3. Writer: NoC-write the partial into slot `k_idx` of the reduce CB on the base
//      core (the k_idx == 0 core for this n_idx, which owns the output N-slice), then
//      bump the base core's reduce semaphore.
//   4. Compute (phase 2, base cores only): once all K_blocks partials have arrived,
//      sum them into the output shard.
//
// Base cores (k_idx == 0) coincide with the width(N)-sharded output cores.
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
    const uint32_t in0_tile_size = tt::tile_size(in0_data_format);
    const uint32_t in1_tile_size = tt::tile_size(in1_data_format);
    const uint32_t out_tile_size = tt::tile_size(out_data_format);

    IDevice* device = input_tensor_a.device();

    // ---- Recover the 2D (K x N) block-sharding geometry ----
    const uint32_t M_tiles = div_up(operation_attributes.M, tt::constants::TILE_HEIGHT);
    const uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);

    const std::array<uint32_t, 2> inputA_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;
    TT_FATAL(
        inputA_shard_shape[0] == (M_tiles * tt::constants::TILE_HEIGHT),
        "Input tensor A shard height {} must equal M_tiles {} * tile height {}",
        inputA_shard_shape[0],
        M_tiles,
        tt::constants::TILE_HEIGHT);
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
    const uint32_t K_blocks = K_tiles / Kc_tiles;
    TT_FATAL(
        K_blocks > 0 && num_B_cores % K_blocks == 0,
        "num_B_cores {} must be divisible by K_blocks {}",
        num_B_cores,
        K_blocks);
    const uint32_t N_blocks = num_B_cores / K_blocks;
    TT_FATAL(
        output_core_range_set.num_cores() == N_blocks,
        "Output must be sharded across N_blocks {} cores, but got {}",
        N_blocks,
        output_core_range_set.num_cores());

    // A is multicast onto every B core; senders are the A-holding cores.
    const auto all_compute_cores = inputA_core_range_set.merge(inputB_core_range_set);
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

    // ---- Circular buffers ----
    constexpr uint32_t in0_cb_index = CBIndex::c_0;       // this core's A slice (gather source)
    constexpr uint32_t in1_cb_index = CBIndex::c_1;       // this core's B block (resident)
    constexpr uint32_t out_cb_index = CBIndex::c_2;       // final output shard (base cores)
    constexpr uint32_t full_in0_cb_index = CBIndex::c_3;  // gathered full A
    constexpr uint32_t partial_cb_index = CBIndex::c_4;   // this core's partial product
    constexpr uint32_t reduce_cb_index = CBIndex::c_5;    // gathered K_blocks partials (base cores)

    const uint32_t block_num_tiles = M_tiles * Nc_tiles;  // tiles in one (partial / output) shard

    // in0: this core's resident A slice (buffer-backed).
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
    // in1: this core's resident B block (buffer-backed).
    desc.cbs.push_back(CBDescriptor{
        .total_size = Kc_tiles * Nc_tiles * in1_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_tile_size,
        }}},
        .buffer = input_tensor_b.buffer(),
    });
    // out: final output shard (buffer-backed, base cores only).
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = output_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
        }}},
        .buffer = output_tensor.buffer(),
    });
    // full_in0: gathered full A (multicast destination).
    desc.cbs.push_back(CBDescriptor{
        .total_size = M_tiles * K_tiles * in0_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = full_in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
        }}},
    });
    // partial: this core's matmul partial product (compute -> writer).
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = partial_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
        }}},
    });
    // reduce: gathered K_blocks partials on the base core (writer -> compute). Allocated
    // identically on every core so each sender can use its local write pointer as the
    // (matching) destination L1 address on the base core.
    desc.cbs.push_back(CBDescriptor{
        .total_size = K_blocks * block_num_tiles * out_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = reduce_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
        }}},
    });

    // ---- Semaphores ----
    const uint32_t num_senders = inputA_core_range_set.num_cores();
    constexpr uint32_t gather_sem_id = 0;  // senders -> coordinator (A gather)
    constexpr uint32_t done_sem_id = 1;    // coordinator -> all (A gathered)
    constexpr uint32_t reduce_sem_id = 2;  // partial-producers -> base core (reduction)
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = gather_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = done_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = reduce_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});

    // ---- Reader kernel (A gather) : reuse the full-width-sharded reader ----
    const CoreRange mcast_bbox = all_compute_cores_with_bbox.bounding_box();
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
        static_cast<uint32_t>(mcast_start_phys.x),
        static_cast<uint32_t>(mcast_start_phys.y),
        in1_cb_index,
        Kc_tiles * Nc_tiles,
    };

    const std::vector<CoreCoord> sender_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, true);
    std::map<CoreCoord, uint32_t> sender_id_by_core;
    for (uint32_t id = 0; id < sender_cores.size(); id++) {
        sender_id_by_core[sender_cores[id]] = id;
    }
    const std::vector<CoreCoord> all_reader_cores = corerange_to_cores(all_compute_cores_with_bbox, std::nullopt, true);

    auto build_reader_kernel = [&](const std::vector<CoreCoord>& cores, NOC noc) {
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

    const size_t num_noc0_senders = sender_cores.size() / 2;
    const std::vector<CoreCoord> noc0_sender_cores(sender_cores.begin(), sender_cores.begin() + num_noc0_senders);
    const std::vector<CoreCoord> noc1_sender_cores(sender_cores.begin() + num_noc0_senders, sender_cores.end());
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

    // ---- Writer kernel (cross-core K-reduction) ----
    //
    // Runs on every B core. Each core ships its partial to slot `k_idx` of the base
    // core's reduce CB and bumps that core's reduce semaphore. Base cores additionally
    // wait for all K_blocks partials and publish the reduce CB to the compute kernel.
    const std::vector<CoreCoord> b_cores = corerange_to_cores(inputB_core_range_set, std::nullopt, true);
    std::vector<CoreRange> b_core_ranges;
    b_core_ranges.reserve(b_cores.size());
    for (const auto& core : b_cores) {
        b_core_ranges.emplace_back(core, core);
    }

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/dataflow/writer_partial_width_sharded.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = CoreRangeSet(b_core_ranges);
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
        .noc = NOC::RISCV_0_default,
    };
    writer_kernel_desc.runtime_args.reserve(b_cores.size());
    for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
        const uint32_t k_idx = idx / N_blocks;
        const uint32_t n_idx = idx % N_blocks;
        const CoreCoord base_logical = b_cores[n_idx];  // k_idx == 0 core for this n_idx
        const CoreCoord base_phys = device->worker_core_from_logical_core(base_logical);
        const bool is_base = (k_idx == 0);
        writer_kernel_desc.runtime_args.emplace_back(
            b_cores[idx],
            KernelDescriptor::CoreRuntimeArgs{
                k_idx,
                static_cast<uint32_t>(base_phys.x),
                static_cast<uint32_t>(base_phys.y),
                static_cast<uint32_t>(is_base)});
    }
    desc.kernels.push_back(std::move(writer_kernel_desc));

    // ---- Compute kernel (partial matmul + base-core reduction) ----
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/compute/compute_partial_width_sharded.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = CoreRangeSet(b_core_ranges);
    compute_kernel_desc.compile_time_args = {
        M_tiles,
        K_tiles,
        Kc_tiles,
        Nc_tiles,
        K_blocks,
    };
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    compute_kernel_desc.runtime_args.reserve(b_cores.size());
    for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
        const uint32_t k_idx = idx / N_blocks;
        const bool is_base = (k_idx == 0);
        compute_kernel_desc.runtime_args.emplace_back(
            b_cores[idx], KernelDescriptor::CoreRuntimeArgs{k_idx, static_cast<uint32_t>(is_base)});
    }
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::operations::matmul_decode
