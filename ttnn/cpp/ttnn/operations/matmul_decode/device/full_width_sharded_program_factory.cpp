// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"
#include "matmul_decode_subblock.hpp"
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

    // =========================================================================
    // deep-plan_14 Lever 2: WIDTH-temporal k_stream branch.
    //
    // Gated by operation_attributes.k_stream. Streams A in G_temporal K-slices instead of
    // gathering full A (which busts L1/CBCAP for SigLIP fc2 K=4320 / VLM down K=16384). The
    // mapping pins k_slice_tiles == inA_K_tiles_per_core (one sender's whole shard == one
    // temporal slice), so G_temporal == num_senders. The slice CB (c_3) is SINGLE-buffered
    // (one slice) and a fp32 running accumulator (c_4) + slice-partial scratch (c_5) carry
    // the K-reduction across slices. See reader_full_width_temporal.cpp /
    // compute_full_width_temporal.cpp.
    // =========================================================================
    if (operation_attributes.k_stream) {
        const uint32_t k_slice_tiles = inA_K_tiles_per_core;  // one sender shard == one slice
        const uint32_t num_senders_t = inputA_core_range_set.num_cores();  // == G_temporal
        TT_FATAL(
            k_slice_tiles > 0 && K_tiles % k_slice_tiles == 0,
            "k_stream: k_slice_tiles {} must divide K_tiles {}",
            k_slice_tiles,
            K_tiles);
        const uint32_t G_temporal = K_tiles / k_slice_tiles;
        TT_FATAL(
            G_temporal == num_senders_t,
            "k_stream: G_temporal {} must equal num_senders {} (k_slice == inA_K_tiles_per_core "
            "mapping); the wrapper must shard A so K_tiles / a_cores == k_slice_tiles",
            G_temporal,
            num_senders_t);

        // out_w fat-fill (out_h forced to 1 -- OQ1 de-risk). Must divide N_tiles_per_core and
        // fit the fp32 DST cap (<=4). Honor an explicit out_subblock_w override if it divides.
        auto [_t_mf, _t_approx, _t_fp32, _t_l1acc, _t_dst_sync] = ttnn::get_compute_kernel_config_args(
            input_tensor_a.device()->arch(), operation_attributes.compute_kernel_config);
        uint32_t out_subblock_w_t = 1;
        for (uint32_t w = 4; w >= 1; --w) {
            if (inB_N_tiles_per_core % w == 0) {
                out_subblock_w_t = w;
                break;
            }
        }
        if (operation_attributes.out_subblock_w.has_value()) {
            const uint32_t ovr = *operation_attributes.out_subblock_w;
            if (ovr >= 1 && ovr <= 4 && inB_N_tiles_per_core % ovr == 0) {
                out_subblock_w_t = ovr;
            }
        }
        // The compute kernel holds the full output rectangle in DST across all K-slices
        // (in-DST cross-slice accumulation). DST holds <= 8 fp32 tiles, so M-rows are grouped
        // into rows_per_group chunks (rows*N_tpc <= 8); the reader re-streams the full A once
        // per group. MUST mirror compute_full_width_temporal.cpp DST_CAP/rows_per_group.
        // DST holds 8 fp32 / 16 bf16 tiles. Prefer the larger bf16 cap so the common pi05
        // shapes (fc2 M_tiles=8, VLM-down M_tiles=9, both npc=1) fit a SINGLE output group
        // (no A re-stream). matmul_block still MACs in fp32 internally; only the cross-slice
        // in-DST accumulation precision differs. (fp32 dest acc -> cap 8.)
        const uint32_t DST_CAP = _t_fp32 ? 8u : 16u;
        const uint32_t rows_per_group =
            (inB_N_tiles_per_core >= DST_CAP) ? 1u : (DST_CAP / inB_N_tiles_per_core);
        const uint32_t num_groups = (M_tiles + rows_per_group - 1) / rows_per_group;

        // ---- CBs ----
        constexpr uint32_t in0_cb_index = CBIndex::c_0;
        constexpr uint32_t in1_cb_index = CBIndex::c_1;
        constexpr uint32_t out_cb_index = CBIndex::c_2;
        constexpr uint32_t slice_cb_index = CBIndex::c_3;
        const uint32_t out_num_tiles = M_tiles * inB_N_tiles_per_core;
        const uint32_t slice_num_tiles = M_tiles * k_slice_tiles;

        // c_0: this core's resident A shard (one slice worth, K-sharded across senders).
        desc.cbs.push_back(CBDescriptor{
            .total_size = M_tiles * inA_K_tiles_per_core * in0_tile_size,
            .core_ranges = all_compute_cores_with_bbox,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = in0_cb_index, .data_format = in0_data_format, .page_size = in0_tile_size}}},
            .buffer = input_tensor_a.buffer(),
        });
        // c_1: this core's full-K B shard (resident).
        desc.cbs.push_back(CBDescriptor{
            .total_size = K_tiles * inB_N_tiles_per_core * in1_tile_size,
            .core_ranges = all_compute_cores_with_bbox,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = in1_cb_index, .data_format = in1_data_format, .page_size = in1_tile_size}}},
            .buffer = input_tensor_b.buffer(),
        });
        // c_2: this core's output shard.
        desc.cbs.push_back(CBDescriptor{
            .total_size = out_num_tiles * out_tile_size,
            .core_ranges = all_compute_cores_with_bbox,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = out_cb_index, .data_format = out_data_format, .page_size = out_tile_size}}},
            .buffer = output_tensor.buffer(),
        });
        // c_3: SINGLE-buffered streamed A slice (destination of the per-round multicast). The
        // reader re-streams the full A num_groups times; this holds ONE slice at a time.
        desc.cbs.push_back(CBDescriptor{
            .total_size = slice_num_tiles * in0_tile_size,
            .core_ranges = all_compute_cores_with_bbox,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = slice_cb_index, .data_format = in0_data_format, .page_size = in0_tile_size}}},
        });

        // ---- Semaphores (reuse the two-semaphore gather/done protocol, FIRED PER SLICE) ----
        constexpr uint32_t gather_sem_id = 0;
        constexpr uint32_t done_sem_id = 1;
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = gather_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = done_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});

        // ---- Reader kernel (temporal) ----
        const CoreRange mcast_bbox_t = all_compute_cores_with_bbox.bounding_box();
        const CoreCoord coordinator_logical_t = mcast_bbox_t.start_coord;
        const CoreCoord mcast_start_phys_t = device->worker_core_from_logical_core(coordinator_logical_t);
        const CoreCoord mcast_end_phys_t = device->worker_core_from_logical_core(mcast_bbox_t.end_coord);
        const uint32_t num_receivers_t = all_compute_cores_with_bbox.num_cores();

        const KernelDescriptor::CompileTimeArgs reader_cta_t = {
            in0_cb_index,
            slice_cb_index,
            slice_num_tiles,  // shard_num_tiles == one slice (M_tiles * k_slice)
            in0_tile_size,
            num_senders_t,
            num_receivers_t,
            static_cast<uint32_t>(mcast_start_phys_t.x),
            static_cast<uint32_t>(mcast_start_phys_t.y),
            static_cast<uint32_t>(mcast_end_phys_t.x),
            static_cast<uint32_t>(mcast_end_phys_t.y),
            gather_sem_id,
            done_sem_id,
            static_cast<uint32_t>(mcast_start_phys_t.x),
            static_cast<uint32_t>(mcast_start_phys_t.y),
            in1_cb_index,
            K_tiles * inB_N_tiles_per_core,
            num_groups,
        };

        const std::vector<CoreCoord> sender_cores_t =
            corerange_to_cores(inputA_core_range_set, std::nullopt, true);
        std::map<CoreCoord, uint32_t> sender_id_by_core_t;
        for (uint32_t id = 0; id < sender_cores_t.size(); id++) {
            sender_id_by_core_t[sender_cores_t[id]] = id;
        }
        // The reader runs ONLY on the compute (output) cores -- each has a compute consumer
        // that pops the slice CB every round (so the per-round reserve_back never deadlocks).
        // Senders are a subset of the output cores (a_cores <= out_cores, wrapper-validated).
        // The multicast still targets the full bbox rectangle (num_receivers_t); the few
        // padding cores in the bbox have c_3 allocated and just receive harmless writes -- they
        // run NO reader kernel, so they cannot stall on a consumer-less CB.
        TT_FATAL(
            output_core_range_set.contains(inputA_core_range_set),
            "k_stream: A sender cores must be a subset of the output (compute) cores");
        const std::vector<CoreCoord> all_reader_cores_t =
            corerange_to_cores(output_core_range_set, std::nullopt, true);

        auto build_reader_t = [&](const std::vector<CoreCoord>& cores, NOC noc) {
            std::vector<CoreRange> ranges;
            ranges.reserve(cores.size());
            for (const auto& core : cores) {
                ranges.emplace_back(core, core);
            }
            KernelDescriptor rk;
            rk.kernel_source =
                "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/dataflow/reader_full_width_temporal.cpp";
            rk.source_type = KernelDescriptor::SourceType::FILE_PATH;
            rk.core_ranges = CoreRangeSet(ranges);
            rk.compile_time_args = reader_cta_t;
            rk.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = noc};
            rk.runtime_args.reserve(cores.size());
            for (const auto& core : cores) {
                const auto it = sender_id_by_core_t.find(core);
                const bool is_sender = it != sender_id_by_core_t.end();
                const uint32_t sender_id = is_sender ? it->second : 0;
                const bool is_coordinator = (core == coordinator_logical_t);
                rk.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        static_cast<uint32_t>(is_sender), sender_id, static_cast<uint32_t>(is_coordinator)});
            }
            return rk;
        };

        // Split senders across both NoCs (balance the per-round multicast traffic).
        const size_t num_noc0_t = sender_cores_t.size() / 2;
        const std::vector<CoreCoord> noc0_t(sender_cores_t.begin(), sender_cores_t.begin() + num_noc0_t);
        const std::vector<CoreCoord> noc1_t(sender_cores_t.begin() + num_noc0_t, sender_cores_t.end());
        std::vector<CoreCoord> default_t;
        default_t.reserve(all_reader_cores_t.size());
        for (const auto& core : all_reader_cores_t) {
            if (sender_id_by_core_t.find(core) == sender_id_by_core_t.end()) {
                default_t.push_back(core);
            }
        }
        if (!noc0_t.empty()) {
            desc.kernels.push_back(build_reader_t(noc0_t, NOC::NOC_0));
        }
        if (!noc1_t.empty()) {
            desc.kernels.push_back(build_reader_t(noc1_t, NOC::NOC_1));
        }
        if (!default_t.empty()) {
            desc.kernels.push_back(build_reader_t(default_t, NOC::RISCV_1_default));
        }

        // ---- Compute kernel (temporal) ----
        KernelDescriptor ck;
        ck.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/compute/compute_full_width_temporal.cpp";
        ck.source_type = KernelDescriptor::SourceType::FILE_PATH;
        ck.core_ranges = output_core_range_set;
        ck.compile_time_args = {
            M_tiles,
            K_tiles,
            inB_N_tiles_per_core,
            k_slice_tiles,
            out_subblock_w_t,
            G_temporal,
            DST_CAP,
        };
        // The cross-slice K-reduction accumulates IN-DST; DST_CAP (8 fp32 / 16 bf16) follows
        // the resolved fp32_dest_acc_en so the output rectangle fits one DST group when possible.
        ck.config = ComputeConfigDescriptor{
            .math_fidelity = _t_mf,
            .fp32_dest_acc_en = _t_fp32,
            .dst_full_sync_en = _t_dst_sync,
            .math_approx_mode = _t_approx,
        };
        desc.kernels.push_back(std::move(ck));

        log_debug(
            tt::LogOp,
            "MatmulDecode TEMPORAL: M_tiles={} K_tiles={} N_tpc={} k_slice={} G_temporal={} "
            "num_senders={} out_w={}",
            M_tiles, K_tiles, inB_N_tiles_per_core, k_slice_tiles, G_temporal, num_senders_t,
            out_subblock_w_t);

        return desc;
    }

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
    // deep-plan_13 Phase 1: derive a REAL fat systolic fill (out_subblock_h/w) via the
    // ported native get_subblock_sizes, bounded by the DST cap (8 bf16 / 4 fp32). The
    // previously DEAD compile args 4,5 (last_out_block_h/num_blocks_h, never read by the
    // kernel) are repurposed to carry out_subblock_h/out_subblock_w.
    //
    // P0-A gate: full_in0 is SENDER-MAJOR so M-fill (out_h>1) needs a contiguous
    // out_h x kt A rectangle that the layout does NOT provide. v1 ships out_w-only
    // (out_subblock_h clamped to 1) -- the safe, contiguous N axis. Flip
    // MMD_ENABLE_M_FILL to true once P0-A proves M-fill (then the full helper is used).
    auto [_mf_cfg, _approx_cfg, _fp32_acc_cfg, _l1_acc_cfg, _dst_full_sync_cfg] =
        ttnn::get_compute_kernel_config_args(
            input_tensor_a.device()->arch(), operation_attributes.compute_kernel_config);
    // deep-plan_14 Lever 1: MMD_ENABLE_M_FILL flips true ONLY after P0-A proves out_h>1 on the
    // Route-A (per-sender K-major-of-M) relayout. Until then v1 auto-derives out_w-only; the
    // explicit-override path (used by the P0-A probe) can still SET out_h>1 to falsify the layout.
    // deep-plan_14 P0-A RESOLVED: out_h>1 M-fill PROVEN via Route-B (per-row rt_dim=1 matmul_block
    // into adjacent DST slots; sender-major A stays). fc1 M_tiles=8: {2x1} fp32 PCC 1.000008,
    // {2x3}/{4x1} bf8_b PCC 0.999842 (>=0.999). Auto-derive the full (out_h,out_w) fat rectangle.
    constexpr bool MMD_ENABLE_M_FILL = true;
    uint32_t out_subblock_h = 1, out_subblock_w = 1;
    if (operation_attributes.out_subblock_h.has_value() && operation_attributes.out_subblock_w.has_value()) {
        // Explicit override (Lever-0-wired, sweep tunable).
        out_subblock_h = *operation_attributes.out_subblock_h;
        out_subblock_w = *operation_attributes.out_subblock_w;
    } else if (MMD_ENABLE_M_FILL) {
        std::tie(out_subblock_h, out_subblock_w) =
            mmd_get_subblock_sizes(M_tiles, inB_N_tiles_per_core, _fp32_acc_cfg);
    } else {
        std::tie(out_subblock_h, out_subblock_w) =
            mmd_get_subblock_sizes_out_w_only(M_tiles, inB_N_tiles_per_core, _fp32_acc_cfg);
    }
    // deep-plan_14 Lever 0: in0_block_w now threaded (default 1 == byte-identical). It must
    // evenly divide K_tiles. NOTE: in0_block_w>1 requires the B (in1) K-tiles within a block
    // to be contiguous for matmul_block's kt_dim; the WIDTH in1 layout is [K_tiles x N_tpc]
    // row-major (K-stride = N_tpc), so >1 is only correct with a matching in1 relayout -- left
    // at the caller's risk and PCC-gated. Default 1 is the validated production path.
    uint32_t in0_block_w = operation_attributes.in0_block_w == 0 ? 1u : operation_attributes.in0_block_w;
    if (K_tiles % in0_block_w != 0) {
        in0_block_w = 1;  // graceful degrade: never produce an invalid K-block count
    }

    log_debug(
        tt::LogOp,
        "MatmulDecode: fat-fill out_subblock_h={}, out_subblock_w={} (fp32_acc={})",
        out_subblock_h,
        out_subblock_w,
        _fp32_acc_cfg);
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
        out_subblock_h,  // arg 4 (was DEAD last_out_block_h)
        out_subblock_w,  // arg 5 (was DEAD num_blocks_h)
        in0_block_w,     // arg 6 (deep-plan_14 Lever 0: was hardcoded constexpr 1)
    };
    // PRECISION (now OPT-IN, mirroring ttnn.matmul): fp32 DST accumulation keeps the
    // K-reduction accumulating in fp32 DST instead of packing each matmul_block partial
    // back to bf16 between K-blocks, recovering the ~0.007 bf16 reduction-order drift on
    // the deep pi0.5 SigLIP-tower / Tier-4 path. It is NO LONGER hardcoded -- it is now
    // driven by the resolved compute_kernel_config threaded through the device op
    // (default OFF; opt in via a DeviceComputeKernelConfig with fp32_dest_acc_en=true).
    // BLACKHOLE DST-CAPACITY NOTE (deep-plan_13): the kernel now holds out_h*out_w DST
    // tiles per tile_regs_acquire; mmd_get_subblock_sizes(_out_w_only) bounds that to
    // <=4 (fp32_dest_acc) / <=8 (bf16) so the DST cap is respected at any fill.
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = _mf_cfg,
        .fp32_dest_acc_en = _fp32_acc_cfg,
        .dst_full_sync_en = _dst_full_sync_cfg,
        .math_approx_mode = _approx_cfg,
    };
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::operations::matmul_decode
