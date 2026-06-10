// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/shape.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <cstdlib>
#include <map>
#include <optional>
#include <set>
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

    // K-STREAMING (plan_5 s3): full_in0 holds either the WHOLE gathered A (one-shot gather,
    // M_tiles*K_tiles tiles) or -- when stream_k -- only ONE K-slice, CB-double-buffered (2 *
    // M_tiles * K_slice_tiles tiles). K_slice_tiles is clamped to be <= inA_K_tiles_per_core (so
    // each stream step is owned by a single sender) and to evenly divide K_tiles.
    const bool stream_k = operation_attributes.stream_k;
    uint32_t K_slice_tiles = operation_attributes.k_slice_tiles;
    if (stream_k) {
        if (K_slice_tiles > inA_K_tiles_per_core) {
            K_slice_tiles = inA_K_tiles_per_core;
        }
        while (K_slice_tiles > 1 && (K_tiles % K_slice_tiles != 0 || inA_K_tiles_per_core % K_slice_tiles != 0)) {
            K_slice_tiles -= 1;
        }
        TT_FATAL(
            K_tiles % K_slice_tiles == 0 && inA_K_tiles_per_core % K_slice_tiles == 0,
            "stream_k: K_slice_tiles {} must divide both K_tiles {} and inA_K_tiles_per_core {}",
            K_slice_tiles,
            K_tiles,
            inA_K_tiles_per_core);
    }
    const uint32_t full_in0_num_tiles = stream_k ? (2u * M_tiles * K_slice_tiles) : (M_tiles * K_tiles);

    ProgramDescriptor desc;

    // ---- Circular buffers (allocated on every participating core) ----
    // Input A Real Input
    constexpr uint32_t in0_cb_index = CBIndex::c_0;
    constexpr uint32_t in1_cb_index = CBIndex::c_1;
    constexpr uint32_t out_cb_index = CBIndex::c_2;
    constexpr uint32_t full_in0_cb_index = CBIndex::c_3;
    constexpr uint32_t acc_cb_index = CBIndex::c_4;  // deep-plan_7: fp32 L1 K-accumulator (stream_k only)
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
        .total_size = full_in0_num_tiles * in0_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = full_in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
        }}},
    });
    // deep-plan_7 §3 STEP-5.1: NEW fp32 L1 K-accumulator CB (c_4), stream_k only. Holds the full
    // M x N partial state (M_tiles * inB_N_tiles_per_core fp32 tiles) that the PACKER accumulates
    // into across the K-OUTER-once loop (pack_reconfig_l1_acc). Lives ONLY on the compute-consumer
    // cores (output_core_range_set, NOT the bbox -- orphan cores never run the streamed compute
    // path), is NOT buffer-backed (scratch, like full_in0), and uses the Float32 (4096B) tile so
    // the cross-slice K-reduction is lossless. Sized for ONE region (the K-OUTER-once path reserves
    // it once and pushes it once at the very end). Non-streamed shapes skip it entirely.
    if (stream_k) {
        const uint32_t acc_tile_size = tt::tile_size(tt::DataFormat::Float32);  // 4096 B
        desc.cbs.push_back(CBDescriptor{
            .total_size = M_tiles * inB_N_tiles_per_core * acc_tile_size,
            .core_ranges = output_core_range_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = acc_cb_index,
                .data_format = tt::DataFormat::Float32,
                .page_size = acc_tile_size,
            }}},
        });
    }
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

    // The one-shot gather reader and the streaming reader take DIFFERENT CT-arg lists.
    const KernelDescriptor::CompileTimeArgs gather_reader_compile_time_args = {
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
    // ---- Compute kernel DST-block geometry (deep-plan_6 §2.2/§5) ----
    // out_block_h = the number of M-tiles per DST-block. The streamed compute kernel holds
    // block_h*N_tiles_per_core live DST tiles per tile_regs_acquire; the proven fp32 DST cap on
    // this Blackhole P150 is 8. For the streamed wide-N case (Npc>1: gate/up, SigLIP fc1/QKV)
    // shrink out_block_h to max(1, 8/Npc) so block_h*Npc <= 8. For Npc==1 keep 8 (down/O M=288 ->
    // num_blocks_h=2). num_blocks_h==1 (M_tiles<=8) -> compute & reader byte-identical to today.
    uint32_t out_block_h = 8;
    if (stream_k && inB_N_tiles_per_core > 1) {
        out_block_h = std::max(1u, 8u / inB_N_tiles_per_core);
    }
    const uint32_t num_blocks_h = tt::div_up(M_tiles, out_block_h);
    const uint32_t last_out_block_h = (M_tiles % out_block_h == 0) ? out_block_h : (M_tiles % out_block_h);

    const KernelDescriptor::CompileTimeArgs stream_reader_compile_time_args = {
        in0_cb_index,
        full_in0_cb_index,
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
        K_tiles * inB_N_tiles_per_core,
        M_tiles,
        K_tiles,
        K_slice_tiles,
        inA_K_tiles_per_core,
        num_blocks_h,  // NEW reader CT arg [19] (deep-plan_6 §2.3 M-block re-stream count)
    };
    const KernelDescriptor::CompileTimeArgs& reader_compile_time_args =
        stream_k ? stream_reader_compile_time_args : gather_reader_compile_time_args;
    const char* reader_kernel_source = stream_k ? "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/dataflow/"
                                                  "reader_full_width_sharded_stream.cpp"
                                                : "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/dataflow/"
                                                  "reader_full_width_sharded.cpp";

    // Map each A-holding core to its K-slice index (== semaphore id).  Cores are
    // walked in row-major order so the slice ordering matches the width-sharded
    // layout of input A across `inputA_core_range_set`.
    const std::vector<CoreCoord> sender_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, true);
    std::map<CoreCoord, uint32_t> sender_id_by_core;
    for (uint32_t id = 0; id < sender_cores.size(); id++) {
        sender_id_by_core[sender_cores[id]] = id;
    }

    // The OUTPUT (compute-consumer) cores. In the streaming reader, only these cores have a
    // compute kernel draining full_in0, so only they may run the per-step reserve_back/push_back
    // loop -- "orphan" cores inside the mcast bbox that are neither senders nor output cores would
    // otherwise block forever on the 3rd reserve_back (CB full, no consumer). is_consumer gates
    // the CB loop for the streaming reader (ignored by the one-shot gather reader).
    const std::vector<CoreCoord> output_cores_vec = corerange_to_cores(output_core_range_set, std::nullopt, true);
    std::set<CoreCoord> output_core_set(output_cores_vec.begin(), output_cores_vec.end());

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
        reader_kernel_desc.kernel_source = reader_kernel_source;
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
            const bool is_consumer = output_core_set.count(core) > 0;
            reader_kernel_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    static_cast<uint32_t>(is_sender),
                    sender_id,
                    static_cast<uint32_t>(is_coordinator),
                    static_cast<uint32_t>(is_consumer)});
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
    // DST-block geometry (out_block_h / num_blocks_h / last_out_block_h) computed above next to
    // the stream-reader CT args so the reader and compute kernel share num_blocks_h in lockstep.
    // matmul_block PER-TILE geometry (in0_block_w / out_block_w) is 1/1 by default (byte-identical
    // single-tile path). The §6c K-reuse/N-reuse widening EXPERIMENT can raise them via the
    // TT_MMD_IN0_BLOCK_W / TT_MMD_OUT_BLOCK_W env knobs (host-read here, passed as compute CT
    // [9]/[10]); in0_block_w must divide K_slice_tiles (streamed) and K_tiles (non-streamed),
    // out_block_w must divide N_tiles_per_core. Defaults 1 leave the proven path untouched.
    uint32_t mm_in0_block_w = 1;
    uint32_t mm_out_block_w = 1;
    // deep-plan_7 §3 STEP-5.2: rt_dim (DST M-block fill) -- previously hardcoded 1 in the compute
    // kernel, now a swept CT arg [11] env-gated by TT_MMD_OUT_BLOCK_H (default 1 = byte-identical
    // matmul_block geometry). It is clamped below to honour the <=8 fp32 DST cap (rt_dim*Npc<=8).
    uint32_t mm_out_block_h = 1;
    if (const char* e = std::getenv("TT_MMD_IN0_BLOCK_W")) {
        const uint32_t v = static_cast<uint32_t>(std::atoi(e));
        if (v >= 1) {
            mm_in0_block_w = v;
        }
    }
    if (const char* e = std::getenv("TT_MMD_OUT_BLOCK_W")) {
        const uint32_t v = static_cast<uint32_t>(std::atoi(e));
        if (v >= 1) {
            mm_out_block_w = v;
        }
    }
    if (const char* e = std::getenv("TT_MMD_OUT_BLOCK_H")) {
        const uint32_t v = static_cast<uint32_t>(std::atoi(e));
        if (v >= 1) {
            mm_out_block_h = v;
        }
    }
    // Safety: only apply widening when it divides cleanly; otherwise fall back to 1 (the kernel
    // loops step by these values, so a non-divisor would drop K/N tiles -> wrong result).
    const uint32_t k_inner = stream_k ? K_slice_tiles : K_tiles;
    if (k_inner % mm_in0_block_w != 0) {
        mm_in0_block_w = 1;
    }
    if (inB_N_tiles_per_core % mm_out_block_w != 0) {
        mm_out_block_w = 1;
    }
    // deep-plan_7 §3 STEP-5.2 + §6 risk-11: clamp rt_dim so a single matmul_block acquire never
    // exceeds the proven fp32 DST cap of 8 (rt_dim*Npc<=8) and never exceeds the per-M-block height
    // (out_block_h). The kernel's inner loop is tail-safe via min(), but keep rt_dim <= out_block_h
    // so a single block isn't over-filled. Defaults to 1 (byte-identical).
    if (inB_N_tiles_per_core * mm_out_block_h > 8u) {
        mm_out_block_h = std::max(1u, 8u / inB_N_tiles_per_core);
    }
    if (mm_out_block_h > out_block_h) {
        mm_out_block_h = out_block_h;
    }

    log_debug(
        tt::LogOp,
        "MatmulDecode: M_tiles: {}, K_tiles: {}, inB_N_tiles_per_core: {}, inA_K_tiles_per_core: {}",
        M_tiles,
        K_tiles,
        inB_N_tiles_per_core,
        inA_K_tiles_per_core);
    log_debug(tt::LogOp, "MatmulDecode: num_blocks_h: {}, last_out_block_h: {}", num_blocks_h, last_out_block_h);
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
        last_out_block_h,
        num_blocks_h,
        static_cast<uint32_t>(stream_k ? 1u : 0u),
        stream_k ? K_slice_tiles : 1u,
        out_block_h,     // NEW compute CT [8] (deep-plan_6 §2.2 DST M-block height)
        mm_in0_block_w,  // NEW compute CT [9] (§6c K-reuse widen; default 1)
        mm_out_block_w,  // NEW compute CT [10] (§6c N-reuse widen; default 1)
        mm_out_block_h,  // NEW compute CT [11] (deep-plan_7 rt_dim M-block fill; env TT_MMD_OUT_BLOCK_H, default 1)
    };
    // PRECISION (now OPT-IN, mirroring ttnn.matmul): fp32 DST accumulation keeps the
    // K-reduction accumulating in fp32 DST instead of packing each matmul_block partial
    // back to bf16 between K-blocks, recovering the ~0.007 bf16 reduction-order drift on
    // the deep pi0.5 SigLIP-tower / Tier-4 path. It is NO LONGER hardcoded -- it is now
    // driven by the resolved compute_kernel_config threaded through the device op
    // (default OFF; opt in via a DeviceComputeKernelConfig with fp32_dest_acc_en=true).
    // BLACKHOLE DST-CAPACITY NOTE (deep-plan_7 §3 STEP-3/STEP-5.8, rewritten): the NON-streamed
    // branch holds 1 live DST tile per tile_regs_acquire (packs slot 0). The STREAMED branch is now
    // K-slice-OUTER-once / M-block+N-INNER: it holds the FULL M x N partial state in the fp32 L1
    // accumulator CB (c_4) that the PACKER accumulates into across the K-OUTER loop
    // (pack_reconfig_l1_acc(s==0?0:1)); DST holds ONLY the <= block_h*N_tiles_per_core (<=8 fp32 DST
    // cap, clamped via rt_dim) transient tiles of ONE acquire (pure micro-tiling). The M loop is
    // tiled into num_blocks_h DST-blocks to keep each acquire <=8. For num_blocks_h==1 (M_tiles<=8:
    // SigLIP M=8, every M<=96) the compute takes a byte-identical iter-6 single-acquire fast-path
    // (no acc CB, no PACKER_L1_ACC). The fp32 c_4 accumulator keeps out_cb (c_2) at its bf16
    // buffer-backed contract UNCHANGED -- there is NO output-dtype ripple.
    auto [_mf, _approx, _fp32_acc, _l1_acc, _dst_full_sync] = ttnn::get_compute_kernel_config_args(
        input_tensor_a.device()->arch(), operation_attributes.compute_kernel_config);
    // FORCED dst_full_sync_en for the streamed MULTI-BLOCK path (deep-plan_6 §2.2 correctness):
    // in SyncHalf mode DST is split into two banks that PING-PONG across consecutive
    // tile_regs_acquire blocks, so the per-M-block acquires of the num_blocks_h>1 streamed loop
    // land in alternating banks while pack reads the wrong bank -> the observed odd-M-block
    // corruption (down M-tile 8 = 0.70, gate/up odd M-tiles = 0.58). Forcing full-sync makes DST
    // a single contiguous bank with no cross-acquire ping-pong, recovering all M-tiles to >=0.9999.
    // This is the same "internal force only when needed" discipline as fp32-on-stream_k; the
    // num_blocks_h==1 path is unaffected (single acquire, no ping-pong) so it stays caller-driven.
    const bool force_full_sync = stream_k && (num_blocks_h > 1);
    // deep-plan_7 §3 STEP-5.5 (PACKER_L1_ACC): the iter-7 streamed K-OUTER-once path accumulates
    // K-slice partials in the fp32 c_4 CB via pack_reconfig_l1_acc(s==0?0:1). On THIS fork the
    // ComputeConfigDescriptor has NO packer_l1_acc field (verified: program_descriptors.hpp:98) and
    // the kernel-internal pack_reconfig_l1_acc()/llk_pack_reconfig_l1_acc() calls are UNCONDITIONAL
    // runtime calls (no #ifdef PACKER_L1_ACC gate exists in matmul_decode), so PACKER L1 accumulation
    // is enabled purely by those kernel calls -- no host descriptor flag is required or available.
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = _mf,
        .fp32_dest_acc_en = _fp32_acc,
        .dst_full_sync_en = force_full_sync ? true : _dst_full_sync,
        .math_approx_mode = _approx,
    };
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::operations::matmul_decode
