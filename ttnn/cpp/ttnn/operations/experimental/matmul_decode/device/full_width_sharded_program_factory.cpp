// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_device_operation.hpp"
#include "ring_walk.hpp"
#include "all_gather_writer.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_program_utils.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/shape.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include <map>
#include <memory>
#include <optional>
#include <algorithm>
#include <array>
#include <set>
#include <vector>

namespace ttnn::operations::experimental::matmul_decode {

using namespace tt;
using namespace tt::tt_metal;

namespace {
// Ring-gather implementation of FullWidthSharded. Replaces the two-hub gather-then-broadcast
// with a pipelined ring all-gather on in0. Kept separate from the hub-gather path so we can
// keep the ENABLE_GLOBAL_CB weight streaming logic on the old path unchanged; the ring-gather
// path is used only when `ring_gather` is requested on L1-resident weights (plain or packed).
ProgramDescriptor create_descriptor_ring_gather_full(
    const MatmulDecodeDeviceOperation::operation_attributes_t& operation_attributes,
    const MatmulDecodeDeviceOperation::tensor_args_t& tensor_args,
    MatmulDecodeDeviceOperation::tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate,
    const GlobalSemaphore* out_ready_semaphore = nullptr,
    const GlobalSemaphore* barrier_semaphore = nullptr);
}  // namespace

// Full width-sharded: B/output are width(N)-sharded; reader gathers full A onto every core.
ProgramDescriptor MatmulDecodeDeviceOperation::FullWidthSharded::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    if (operation_attributes.mesh_coords.has_value() &&
        (!mesh_dispatch_coordinate.has_value() ||
         std::find(
             operation_attributes.mesh_coords->begin(),
             operation_attributes.mesh_coords->end(),
             *mesh_dispatch_coordinate) == operation_attributes.mesh_coords->end())) {
        return {};
    }
    // Ring gather is opt-in (`ring_gather`) on the L1-resident weight paths (plain and
    // packed_weight). The two-hub gather remains the default, and the GCB (prefetcher) path
    // always uses it because the reader<->prefetcher handshake is baked into
    // reader_full_width_sharded.cpp.
    if (operation_attributes.ring_gather && !operation_attributes.global_cb.has_value()) {
        const auto& a_grid = tensor_args.input_tensor_a.memory_config().shard_spec().value().grid;
        const auto& b_grid = operation_attributes.packed_weight.has_value()
                                 ? operation_attributes.packed_weight->cores
                                 : tensor_args.input_tensor_b.memory_config().shard_spec().value().grid;
        log_info(
            tt::LogOp,
            "matmul_decode FullWidthSharded: in0 gather = ring ({}, S={} cores, C={} cores)",
            a_grid.intersects(b_grid) ? "overlapping S/C" : "disjoint S/C",
            a_grid.num_cores(),
            b_grid.num_cores());
        return create_descriptor_ring_gather_full(
            operation_attributes, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
    }
    (void)mesh_dispatch_coordinate;
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
    const bool use_global_cb = operation_attributes.global_cb.has_value();
    // A packed weight is a region of a fused height-sharded tensor whose shard spec spans the
    // whole chip, so this weight's grid comes from the spec.
    const auto& packed = operation_attributes.packed_weight;
    // A prefetcher-fed weight is ND-sharded in DRAM (no legacy shard spec), so the receiver
    // grid is the GCB's receiver set. Otherwise it is the weight's own shard grid.
    auto inputB_core_range_set = use_global_cb        ? operation_attributes.global_cb->receiver_cores()
                                 : packed.has_value() ? packed->cores
                                                      : input_tensor_b.memory_config().shard_spec().value().grid;
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

    uint32_t inB_N_tiles_per_core;
    if (use_global_cb || packed.has_value()) {
        // Neither a GCB-fed nor a packed weight has a legacy [K, N/n] shard spec to read the
        // per-core width from; it is N split across the weight's cores by definition.
        const uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);
        const uint32_t num_weight_cores = inputB_core_range_set.num_cores();
        TT_FATAL(
            N_tiles % num_weight_cores == 0,
            "full_width_sharded matmul_decode with {} requires N in tiles ({}) to be divisible by the weight core "
            "count ({})",
            use_global_cb ? "global_cb" : "packed_weight",
            N_tiles,
            num_weight_cores);
        inB_N_tiles_per_core = N_tiles / num_weight_cores;
    } else {
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
        inB_N_tiles_per_core = inputB_shard_shape[1] / tt::constants::TILE_WIDTH;
    }
    ProgramDescriptor desc;

    // These are this op's own CB indices; every kernel receives them as named "cb_*" compile-time
    // args, so op fusion can pool-allocate different hardware slots for two instances sharing a
    // core without either factory having to know about the other.
    const uint32_t in0_cb_index = CBIndex::c_0;
    const uint32_t in1_cb_index = CBIndex::c_1;
    const uint32_t out_cb_index = CBIndex::c_2;
    const uint32_t full_in0_cb_index = CBIndex::c_3;
    // GCB path only: sync_cb carries "compute is done reading in1" back to the reader so it can
    // release the GCB page; remote_cb is the remote (GCB) index aliased onto the local in1 CB.
    const uint32_t sync_cb_index = CBIndex::c_4;
    const uint32_t remote_cb_index = CBIndex::c_31;
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

    // A GCB page is `num_k_blocks`-th of a receiver's [K, N/num_receivers] weight slab: a whole
    // number of K-rows, contiguous in the slab. num_k_blocks == 1 makes the page the whole slab
    // (one credit per invocation, the GCB must hold a slab); higher values stream the slab in and
    // let the GCB be smaller than it. The local alias (in1_cb_index) stays tile-paged so the
    // compute kernel indexes tiles within the page it is holding; the remote index is page-paged
    // so one page-credit == one K-block.
    const uint32_t num_k_blocks = operation_attributes.global_cb_k_blocks;
    TT_FATAL(
        K_tiles % num_k_blocks == 0,
        "full_width_sharded matmul_decode with global_cb_k_blocks={} requires K in tiles ({}) to be divisible by it, "
        "because a GCB page is a whole number of K-rows of the weight slab",
        num_k_blocks,
        K_tiles);
    // Streaming completes each output tile across several pages, and the running sum lives in the
    // output CB itself because the packer accumulates into it. A block-float output cannot be read
    // back and added to, so it has to be rejected rather than quietly dropping partial sums.
    TT_FATAL(
        num_k_blocks == 1 || out_data_format == tt::DataFormat::Float32 ||
            out_data_format == tt::DataFormat::Float16_b || out_data_format == tt::DataFormat::Float16,
        "full_width_sharded matmul_decode with global_cb_k_blocks={} accumulates partial sums in the output CB, so "
        "the output dtype must be float32/bfloat16/float16, but it is {}",
        num_k_blocks,
        out_data_format);
    const uint32_t in1_k_block_tiles = K_tiles / num_k_blocks;
    const uint32_t in1_slab_num_tiles = K_tiles * inB_N_tiles_per_core;
    const uint32_t in1_slab_bytes = in1_slab_num_tiles * in1_tile_size;
    const uint32_t in1_page_num_tiles = in1_k_block_tiles * inB_N_tiles_per_core;
    const uint32_t in1_page_bytes = in1_page_num_tiles * in1_tile_size;
    if (use_global_cb) {
        const auto& gcb = *operation_attributes.global_cb;
        // Round the window down to a whole number of pages; the remote CB requires its total
        // size to be a multiple of its page size, and the local alias only wraps in step with
        // the remote ring if it spans whole pages too.
        const uint32_t gcb_window_bytes = (gcb.size() / in1_page_bytes) * in1_page_bytes;
        // Streaming keeps one page un-acked while the next is published, so the ring has to hold
        // two. With one page it would deadlock: the reader waits for a page the sender cannot
        // write until the reader returns the credit it is still holding.
        const uint32_t min_pages = num_k_blocks > 1 ? 2 : 1;
        TT_FATAL(
            gcb_window_bytes >= min_pages * in1_page_bytes,
            "full_width_sharded matmul_decode with global_cb_k_blocks={} needs a GCB of at least {} page(s) per "
            "receiver ({} B), but the GCB holds {} B",
            num_k_blocks,
            min_pages,
            min_pages * in1_page_bytes,
            gcb.size());
        desc.cbs.push_back(CBDescriptor{
            .total_size = gcb_window_bytes,
            .core_ranges = inputB_core_range_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = in1_cb_index,
                .data_format = in1_data_format,
                .page_size = in1_tile_size,
                .tile = in1_tile_desc,
            }}},
            .remote_format_descriptors = {{CBFormatDescriptor{
                .buffer_index = remote_cb_index,
                .data_format = in1_data_format,
                .page_size = in1_page_bytes,
            }}},
            .global_circular_buffer = std::addressof(gcb),
        });
        // Compute -> reader release signal: one 16 B page (one credit) per in1 page. Deliberately
        // one page deep -- it is what bounds compute to a single un-acked GCB page, which is the
        // invariant the two-page ring minimum above is derived from.
        constexpr uint32_t sync_cb_page_bytes = 16;
        desc.cbs.push_back(CBDescriptor{
            .total_size = sync_cb_page_bytes,
            .core_ranges = inputB_core_range_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = sync_cb_index,
                .data_format = tt::DataFormat::UInt16,
                .page_size = sync_cb_page_bytes,
            }}},
        });
    } else {
        // Globally allocated over the resident weight. For a packed weight the buffer is the
        // fused tensor's, and the region's byte offset into every core's shard re-bases the CB
        // onto this weight's slab -- the kernels are none the wiser.
        desc.cbs.push_back(CBDescriptor{
            .total_size = in1_slab_bytes,
            .core_ranges = all_compute_cores_with_bbox,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = in1_cb_index,
                .data_format = in1_data_format,
                .page_size = in1_tile_size,
                .tile = in1_tile_desc,
            }}},
            .buffer = input_tensor_b.buffer(),
            .address_offset = packed.has_value() ? packed->tile_offset * in1_tile_size : 0,
        });
    }

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
        in1_page_num_tiles,
        num_k_blocks,
    };

    // Every CB index travels as a named "cb_*" arg: op fusion pool-allocates hardware CB slots
    // across the phases it merges and rewrites exactly these args, so positional or hard-coded
    // indices would leave the kernels pointing at pre-remap slots.
    const KernelDescriptor::NamedCompileTimeArgs reader_named_args = {
        {"cb_in0", in0_cb_index},
        {"cb_full_in0", full_in0_cb_index},
        {"cb_in1", in1_cb_index},
        {"cb_in1_remote", remote_cb_index},
        {"cb_sync", sync_cb_index},
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
        reader_kernel_desc.named_compile_time_args = reader_named_args;
        reader_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_1,
            // The GCB path pins every reader to NOC 0. remote_cb_pop_front acks the page with a
            // non-posted atomic increment into the DRISC sender's L1, and that ack only comes
            // back on NOC 0 -- on NOC 1 the following atomic barrier never drains and the core
            // hangs after the matmul has otherwise finished. Every other GCB consumer in the
            // repo likewise runs its remote-CB traffic on the DRAM-read NOC. This costs the
            // two-hub gather its NOC split (both hubs mcast on NOC 0) in the GCB path only.
            .noc = use_global_cb ? NOC::NOC_0 : noc,
        };
        if (use_global_cb) {
            reader_kernel_desc.defines.emplace_back("ENABLE_GLOBAL_CB", "1");
        }

        reader_kernel_desc.runtime_args.reserve(cores.size());
        for (const auto& core : cores) {
            const auto it = sender_id_by_core.find(core);
            const bool is_sender = it != sender_id_by_core.end();
            const uint32_t sender_id = is_sender ? it->second : 0;
            // The reader runs on the merged A-and-B bounding box, but only B cores are GCB
            // receivers and only they have the in1 / sync CBs configured.
            const bool is_in1_receiver = inputB_core_range_set.contains(core);
            reader_kernel_desc.runtime_args.emplace_back(
                core,
                KernelDescriptor::CoreRuntimeArgs{
                    static_cast<uint32_t>(is_sender),
                    sender_id,
                    static_cast<uint32_t>(role_of(core)),
                    static_cast<uint32_t>(is_in1_receiver)});
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
        num_k_blocks,
    };
    compute_kernel_desc.named_compile_time_args = {
        {"cb_full_in0", full_in0_cb_index},
        {"cb_in1", in1_cb_index},
        {"cb_out", out_cb_index},
        {"cb_sync", sync_cb_index},
    };
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    if (use_global_cb) {
        compute_kernel_desc.defines.emplace_back("ENABLE_GLOBAL_CB", "1");
    }
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

namespace {

using ring_walk::arriving_sender_ids_at;
using ring_walk::build_ring_walk;
using ring_walk::RG_ROLE_IDLE;
using ring_walk::RingWalk;

ProgramDescriptor create_descriptor_ring_gather_full(
    const MatmulDecodeDeviceOperation::operation_attributes_t& operation_attributes,
    const MatmulDecodeDeviceOperation::tensor_args_t& tensor_args,
    MatmulDecodeDeviceOperation::tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate,
    const GlobalSemaphore* out_ready_semaphore,
    const GlobalSemaphore* barrier_semaphore) {
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
    TT_FATAL(
        inputA_tile.get_width() == tt::constants::TILE_WIDTH &&
            inputB_tile.get_height() == tt::constants::TILE_HEIGHT &&
            inputB_tile.get_width() == tt::constants::TILE_WIDTH &&
            output_tile.get_width() == tt::constants::TILE_WIDTH && inputA_tile_height == output_tile.get_height(),
        "matmul_decode ring gather: unexpected tile geometry");

    const uint32_t M_tiles = div_up(operation_attributes.M, inputA_tile_height);
    const uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);

    IDevice* device = input_tensor_a.device();
    const bool all_gather = operation_attributes.all_gather;
    TT_FATAL(
        !all_gather || mesh_dispatch_coordinate.has_value(),
        "matmul_decode all_gather requires a mesh dispatch coordinate");

    const auto& packed = operation_attributes.packed_weight;
    auto inputA_core_range_set = input_tensor_a.memory_config().shard_spec().value().grid;
    auto inputB_core_range_set =
        packed.has_value() ? packed->cores : input_tensor_b.memory_config().shard_spec().value().grid;
    auto output_core_range_set = output_tensor.memory_config().shard_spec().value().grid;
    TT_FATAL(
        inputB_core_range_set == output_core_range_set,
        "matmul_decode ring gather: input tensor B and output tensor must have the same core range set");

    // Per-core sharding geometry. inA_K_tiles_per_core (= K_local in tiles) is a source shard's
    // width; each shard is M_tiles x inA_K_tiles_per_core.
    const std::array<uint32_t, 2> inputA_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;
    TT_FATAL(
        inputA_shard_shape[0] == (M_tiles * inputA_tile_height) &&
            inputA_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "matmul_decode ring gather: input A shard shape must be [M_tiles*tile_h, k * tile_w]");
    const uint32_t inA_K_tiles_per_core = inputA_shard_shape[1] / tt::constants::TILE_WIDTH;

    uint32_t inB_N_tiles_per_core;
    if (packed.has_value()) {
        const uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);
        const uint32_t num_weight_cores = inputB_core_range_set.num_cores();
        TT_FATAL(
            N_tiles % num_weight_cores == 0,
            "matmul_decode ring gather (packed_weight): N in tiles ({}) must be divisible by weight core count ({})",
            N_tiles,
            num_weight_cores);
        inB_N_tiles_per_core = N_tiles / num_weight_cores;
    } else {
        const std::array<uint32_t, 2> inputB_shard_shape = input_tensor_b.memory_config().shard_spec().value().shape;
        TT_FATAL(
            inputB_shard_shape[0] == (K_tiles * tt::constants::TILE_HEIGHT) &&
                inputB_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
            "matmul_decode ring gather: input B shard shape must be [K, per_core_N]");
        inB_N_tiles_per_core = inputB_shard_shape[1] / tt::constants::TILE_WIDTH;
    }

    // No topological requirement on either grid: the ring walk is a logical sequence over
    // corerange_to_cores(..., row_wise=true), and NoC unicasts route through intermediate cores
    // at the hardware level without any relay kernel. Row-major snake grids (as used by the
    // packed-weight deepseek layer test) work as well as rectangles.
    TT_FATAL(
        inputA_core_range_set.num_cores() > 0 && output_core_range_set.num_cores() > 0,
        "matmul_decode ring gather: both input A grid and output grid must be non-empty");

    const std::vector<CoreCoord> S_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, /*row_wise=*/true);
    const std::vector<CoreCoord> C_cores = corerange_to_cores(output_core_range_set, std::nullopt, /*row_wise=*/true);
    const uint32_t num_senders = static_cast<uint32_t>(S_cores.size());

    TT_FATAL(
        num_senders == K_tiles / inA_K_tiles_per_core,
        "matmul_decode ring gather: sender count ({}) must equal K_tiles / inA_K_tiles_per_core ({} / {})",
        num_senders,
        K_tiles,
        inA_K_tiles_per_core);

    // Phase 1: single CW ring only. Bidirectional (CCW on RISCV_0/NOC_1) is a straightforward
    // extension -- build a second RingWalk over the second half of the senders, allocate a
    // cb_in2_ccw, and instantiate a second reader kernel.
    const RingWalk cw = build_ring_walk(device, /*sources=*/S_cores, /*hops=*/C_cores);

    // Kernels and CBs are placed on the walk's cores (S ∪ C) only. Cores that fall inside the
    // bounding box but neither shard nor compute get no kernel, no CB, and no L1 reservation.
    std::vector<CoreCoord> S_or_C = S_cores;
    for (const auto& c : C_cores) {
        bool dup = false;
        for (const auto& s : S_cores) {
            if (s == c) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            S_or_C.push_back(c);
        }
    }
    const CoreRangeSet all_ring_cores = CoreRangeSet(S_or_C);

    ProgramDescriptor desc;

    const uint32_t in0_cb_index = CBIndex::c_0;
    const uint32_t in1_cb_index = CBIndex::c_1;
    const uint32_t out_cb_index = CBIndex::c_2;
    const uint32_t in2_cw_cb_index = CBIndex::c_3;
    const uint32_t in2_ccw_cb_index = CBIndex::c_4;

    // cb_in0: local shard, allocated only on source cores over the input tensor's L1.
    desc.cbs.push_back(CBDescriptor{
        .total_size = M_tiles * inA_K_tiles_per_core * in0_tile_size,
        .core_ranges = inputA_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
        .buffer = input_tensor_a.buffer(),
    });

    // cb_in1: full weight, L1-resident, allocated on compute cores. In the packed case the
    // buffer is the fused tensor and address_offset re-bases it onto this weight's slab.
    desc.cbs.push_back(CBDescriptor{
        .total_size = K_tiles * inB_N_tiles_per_core * in1_tile_size,
        .core_ranges = output_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_tile_size,
            .tile = in1_tile_desc,
        }}},
        .buffer = input_tensor_b.buffer(),
        .address_offset = packed.has_value() ? packed->tile_offset * in1_tile_size : 0,
    });

    // cb_out: on compute cores. Aliased over the output tensor unless all_gather is on,
    // in which case it is scratch (local N-shard) and the writer copies into the gathered tensor.
    CBDescriptor out_cb_desc{
        .total_size = M_tiles * inB_N_tiles_per_core * out_tile_size,
        .core_ranges = output_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    };
    if (!all_gather) {
        out_cb_desc.buffer = output_tensor.buffer();
    }
    desc.cbs.push_back(std::move(out_cb_desc));

    // cb_in2_cw: remote shard ring buffer for the CW direction. Sized to hold every shard this
    // core receives (max = num_senders across all cores in the ring walk). This is the CB whose
    // size scales with num_senders, but each core only fills num_recv slots.
    const uint32_t in2_cb_num_tiles = num_senders * M_tiles * inA_K_tiles_per_core;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in2_cb_num_tiles * in0_tile_size,
        .core_ranges = all_ring_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in2_cw_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
    });

    // cb_in2_ccw: zero-sized placeholder while CCW is not yet wired up. Sized to a single tile
    // so CB allocation succeeds; the compute kernel skips CCW when num_shards_ccw == 0.
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_tile_size,
        .core_ranges = all_ring_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in2_ccw_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
    });

    // Ring semaphores. sig_cw carries the pipeline credit for the CW ring. Allocated on the
    // whole ring so every core can be atomically bumped from its predecessor.
    constexpr uint32_t sig_cw_sem_id = 0;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sig_cw_sem_id,
        .core_ranges = all_ring_cores,
        .initial_value = 0,
    });
    if (all_gather) {
        TT_FATAL(
            out_ready_semaphore != nullptr && barrier_semaphore != nullptr,
            "matmul_decode all_gather requires workload-scoped semaphores");
        // Cross-device semaphores are allocated once by AllGatherFullWidth and
        // kept alive by its WorkloadDescriptor. The activation-ring semaphore
        // above remains a normal per-program semaphore.
    }

    const uint32_t shard_num_tiles = M_tiles * inA_K_tiles_per_core;
    const uint32_t local_shard_num_tiles = shard_num_tiles;

    // ---- Reader kernel (CW) ----
    KernelDescriptor reader_cw;
    reader_cw.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/reader_full_width_ring_gather.cpp";
    reader_cw.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_cw.core_ranges = all_ring_cores;
    reader_cw.compile_time_args = {
        shard_num_tiles,
        in0_tile_size,
        sig_cw_sem_id,
        local_shard_num_tiles,
    };
    reader_cw.named_compile_time_args = {
        {"cb_in0", in0_cb_index},
        {"cb_in2", in2_cw_cb_index},
    };
    reader_cw.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_0,
    };
    // Position of a core in the CW walk (if any).
    std::map<CoreCoord, uint32_t> cw_pos;
    for (uint32_t p = 0; p < cw.cores.size(); ++p) {
        cw_pos[cw.cores[p]] = p;
    }
    // Reader on every core in S ∪ C. `signal_in0` publishes cb_in0 for compute to consume:
    // - source-only: harmless (push_back with no compute consumer, cb_in0 is allocated).
    // - source-and-compute (overlap): required so compute.wait_front(cb_in0) resolves.
    // - hop (compute-only, no own shard): 0 -- cb_in0 isn't allocated on this core.
    reader_cw.runtime_args.reserve(S_or_C.size());
    for (const auto& core : S_or_C) {
        const auto it = cw_pos.find(core);
        if (it == cw_pos.end()) {
            reader_cw.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{RG_ROLE_IDLE, 0, 0, 0, 0, 0});
            continue;
        }
        const uint32_t p = it->second;
        const CoreCoord next = cw.next_phys[p];
        const uint32_t signal_in0 = cw.is_source[p] ? 1u : 0u;
        // Reader kernel only distinguishes "has own shard" (SOURCE) vs "no own shard" (HOP);
        // it doesn't care whether we're also a compute core. Fold the roles here so the kernel
        // stays simple.
        const uint32_t reader_role = cw.is_source[p] ? /*SOURCE*/ 1u : /*HOP*/ 2u;
        reader_cw.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                reader_role,
                cw.num_recv[p],
                cw.num_sends[p],
                static_cast<uint32_t>(next.x),
                static_cast<uint32_t>(next.y),
                signal_in0});
    }
    desc.kernels.push_back(std::move(reader_cw));

    // ---- Compute kernel ----
    TT_FATAL(
        M_tiles <= 8,
        "matmul_decode ring gather: M_tiles must be <= 8 so the output block fits in DST, but got {}",
        M_tiles);

    KernelDescriptor compute_kd;
    compute_kd.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/compute/"
        "compute_full_width_ring_gather.cpp";
    compute_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kd.core_ranges = output_core_range_set;

    compute_kd.compile_time_args = {
        M_tiles,
        K_tiles,
        inB_N_tiles_per_core,
        inA_K_tiles_per_core,
        // num_senders is the max any core's num_arriving_cw can be (compute-only cores see all
        // S shards; overlap cores see S-1). Passed as a compile-time constant only so the
        // kernel can size compile-time init decisions; the actual per-launch loop bound is a
        // runtime arg (num_arriving_cw) since it varies per core.
        /*num_senders=*/num_senders,
    };
    compute_kd.named_compile_time_args = {
        {"cb_in0", in0_cb_index},
        {"cb_in2_cw", in2_cw_cb_index},
        {"cb_in2_ccw", in2_ccw_cb_index},
        {"cb_in1", in1_cb_index},
        {"cb_out", out_cb_index},
    };
    compute_kd.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
    };
    if (all_gather) {
        compute_kd.defines.emplace_back("ENABLE_ALL_GATHER", "1");
    }
    // Per-core runtime args:
    //   [0] has_local_shard  (1 iff this core is also in S -- overlap)
    //   [1] local_sender_id  (source index if has_local_shard, else don't-care)
    //   [2] num_arriving_cw  (S for compute-only, S-1 for overlap)
    //   [3] num_arriving_ccw (0 while CCW ring is not wired up)
    //   [4..4+num_arriving_cw-1] cw_sender_ids (arrival order)
    //   [4+num_arriving_cw..] ccw_sender_ids (currently empty)
    compute_kd.runtime_args.reserve(C_cores.size());
    for (const auto& core : C_cores) {
        const auto it = cw_pos.find(core);
        TT_FATAL(
            it != cw_pos.end(),
            "matmul_decode ring gather: compute core {} is not on the CW walk (internal error)",
            core.str());
        const uint32_t p = it->second;
        const std::vector<uint32_t> ids = arriving_sender_ids_at(p, cw);
        const uint32_t expected = num_senders - cw.is_source[p];
        TT_FATAL(
            ids.size() == expected,
            "matmul_decode ring gather: compute core at walk pos {} sees {} sender IDs, expected {} "
            "(is_source={})",
            p,
            ids.size(),
            expected,
            cw.is_source[p]);
        KernelDescriptor::CoreRuntimeArgs rt;
        rt.reserve(4 + ids.size());
        rt.push_back(cw.is_source[p] ? 1u : 0u);                   // has_local_shard
        rt.push_back(cw.is_source[p] ? cw.own_sender_id[p] : 0u);  // local_sender_id
        rt.push_back(static_cast<uint32_t>(ids.size()));           // num_arriving_cw
        rt.push_back(0);                                           // num_arriving_ccw
        rt.insert(rt.end(), ids.begin(), ids.end());
        compute_kd.runtime_args.emplace_back(core, std::move(rt));
    }
    desc.kernels.push_back(std::move(compute_kd));

    if (all_gather) {
        const auto route = make_all_gather_fabric_route(input_tensor_a, *mesh_dispatch_coordinate);
        const auto available_cores =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().at(0));
        const auto available_core_list = corerange_to_cores(available_cores, std::nullopt, true);
        const auto output_core_list = corerange_to_cores(output_core_range_set, std::nullopt, true);
        const std::set<CoreCoord> ring_core_set(S_or_C.begin(), S_or_C.end());
        std::set<CoreCoord> output_core_set(output_core_list.begin(), output_core_list.end());
        std::vector<CoreCoord> mux_cores;
        for (const auto& core : available_core_list) {
            if (!output_core_set.contains(core) && !ring_core_set.contains(core)) {
                mux_cores.push_back(core);
            }
        }
        const bool use_mux = C_cores.size() > 1;
        std::array<std::vector<uint32_t>, 2> mux_links_by_dir;
        uint32_t dst_index = 0;
        const auto sender_node = input_tensor_a.device()->get_fabric_node_id(*mesh_dispatch_coordinate);
        for (uint32_t dir = 0; dir < 2; ++dir) {
            const uint32_t range = dir == 0 ? route.range_fwd : route.range_bwd;
            if (range == 0) {
                continue;
            }
            const auto valid_links =
                tt::tt_fabric::get_forwarding_link_indices(sender_node, route.dst_nodes[dst_index++]);
            TT_FATAL(!valid_links.empty(), "matmul_decode all_gather: no forwarding link for mux direction {}", dir);
            mux_links_by_dir[dir] = valid_links;
        }
        const uint32_t required_mux_cores =
            static_cast<uint32_t>(mux_links_by_dir[0].size() + mux_links_by_dir[1].size());
        TT_FATAL(
            C_cores.size() <= 1 || mux_cores.size() >= required_mux_cores,
            "matmul_decode all_gather needs {} free mux cores, but only {} are available",
            required_mux_cores,
            mux_cores.size());
        std::array<std::vector<CoreCoord>, 2> mux_core_groups;
        uint32_t next_mux_core = 0;
        uint32_t max_clients_per_mux = 0;
        for (uint32_t dir = 0; dir < 2; ++dir) {
            for (uint32_t mux_idx = 0; mux_idx < mux_links_by_dir[dir].size(); ++mux_idx) {
                mux_core_groups[dir].push_back(mux_cores[next_mux_core++]);
            }
            if (!mux_links_by_dir[dir].empty()) {
                max_clients_per_mux = std::max(
                    max_clients_per_mux,
                    div_up(static_cast<uint32_t>(C_cores.size()), static_cast<uint32_t>(mux_links_by_dir[dir].size())));
            }
        }
        tt::tt_fabric::FabricMuxConfig mux_config(
            /*num_full_size_channels=*/max_clients_per_mux,
            /*num_header_only_channels=*/0,
            /*num_buffers_full_size_channel=*/2,
            /*num_buffers_header_only_channel=*/0,
            /*buffer_size_bytes_full_size_channel=*/tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
            /*base_l1_address=*/device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
        KernelDescriptor writer;
        writer.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/matmul_decode/device/kernels/dataflow/"
            "writer_full_width_all_gather.cpp";
        writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer.core_ranges = output_core_range_set;
        writer.named_compile_time_args = all_gather_named_compile_time_args(
            out_cb_index,
            M_tiles,
            inB_N_tiles_per_core,
            route.ring_index,
            operation_attributes.ring_size,
            static_cast<uint32_t>(out_ready_semaphore->address()),
            static_cast<uint32_t>(barrier_semaphore->address()),
            route.start_fwd,
            route.range_fwd,
            route.start_bwd,
            route.range_bwd,
            /*ag_rt_arg_base=*/0,
            /*num_shards=*/1,
            /*shard_sem_id=*/0,
            /*staging_cb_id=*/0,
            use_mux,
            mux_config.get_num_buffers(tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL),
            mux_config.get_buffer_size_bytes(tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL),
            mux_config.get_status_address(),
            mux_config.get_termination_signal_address(),
            max_clients_per_mux);
        writer.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_1,
        };
        const KernelHandle writer_id = static_cast<KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(writer));
        CoreRangeSet mux_core_range;
        if (use_mux) {
            std::vector<CoreRange> active_mux_ranges;
            for (uint32_t dir = 0; dir < 2; ++dir) {
                for (const auto& mux_core : mux_core_groups[dir]) {
                    active_mux_ranges.emplace_back(mux_core);
                }
            }
            mux_core_range = CoreRangeSet(active_mux_ranges);
            KernelDescriptor mux_kernel;
            mux_kernel.kernel_source = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
            mux_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
            mux_kernel.core_ranges = mux_core_range;
            mux_kernel.compile_time_args = mux_config.get_fabric_mux_compile_time_args();
            mux_kernel.config =
                DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0};
            mux_kernel.runtime_args.reserve(required_mux_cores);
            uint32_t route_dst_index = 0;
            for (uint32_t dir = 0; dir < 2; ++dir) {
                if ((dir == 0 && route.range_fwd == 0) || (dir == 1 && route.range_bwd == 0)) {
                    continue;
                }
                const auto dst = route.dst_nodes[route_dst_index++];
                for (uint32_t mux_idx = 0; mux_idx < mux_core_groups[dir].size(); ++mux_idx) {
                    const auto mux_core = mux_core_groups[dir][mux_idx];
                    mux_kernel.runtime_args.emplace_back(
                        mux_core,
                        mux_config.get_fabric_mux_run_time_args(
                            input_tensor_a.device()->get_fabric_node_id(*mesh_dispatch_coordinate),
                            dst,
                            mux_links_by_dir[dir][mux_idx],
                            desc,
                            mux_core));
                }
            }
            desc.kernels.push_back(std::move(mux_kernel));
        }
        const uint32_t gathered_n_tiles = inB_N_tiles_per_core * operation_attributes.ring_size;
        const uint32_t local_n_tiles = inB_N_tiles_per_core * static_cast<uint32_t>(C_cores.size());
        for (uint32_t i = 0; i < C_cores.size(); ++i) {
            // The gathered tensor remains WIDTH_SHARDED across C_cores. A local result from
            // compute core i belongs at global N tile
            //   ring_index * N_local_tiles + i * N_tiles_per_core.
            // Convert that global tile to the destination output shard and its in-shard offset.
            const uint32_t global_n_tile = route.ring_index * local_n_tiles + i * inB_N_tiles_per_core;
            const uint32_t output_core_idx = global_n_tile / gathered_n_tiles;
            const uint32_t output_tile_offset = global_n_tile % gathered_n_tiles;
            TT_FATAL(
                output_core_idx < C_cores.size() && output_tile_offset + inB_N_tiles_per_core <= gathered_n_tiles,
                "matmul_decode all_gather output placement is invalid: source core {}, global tile {}, "
                "destination core {}, offset {}, shard width {}",
                i,
                global_n_tile,
                output_core_idx,
                output_tile_offset,
                gathered_n_tiles);
            const CoreCoord output_core = C_cores[output_core_idx];
            const auto forwarding_links =
                all_gather_forwarding_links(input_tensor_a, *mesh_dispatch_coordinate, route, i);
            if (use_mux) {
                std::array<CoreCoord, 2> mux_core_by_dir{};
                std::array<CoreCoord, 2> termination_masters{};
                std::array<uint32_t, 2> mux_worker_ids{};
                std::array<uint32_t, 2> mux_client_counts{};
                for (uint32_t dir = 0; dir < 2; ++dir) {
                    if ((dir == 0 && route.range_fwd == 0) || (dir == 1 && route.range_bwd == 0)) {
                        continue;
                    }
                    const uint32_t mux_group = i % static_cast<uint32_t>(mux_core_groups[dir].size());
                    const uint32_t mux_group_start = mux_group;
                    const uint32_t mux_group_count = div_up(
                        static_cast<uint32_t>(C_cores.size()) - mux_group_start,
                        static_cast<uint32_t>(mux_core_groups[dir].size()));
                    mux_core_by_dir[dir] = mux_core_groups[dir][mux_group];
                    termination_masters[dir] = C_cores[mux_group_start];
                    mux_worker_ids[dir] = i / static_cast<uint32_t>(mux_core_groups[dir].size());
                    mux_client_counts[dir] = mux_group_count;
                }
                set_all_gather_writer_runtime_args(
                    desc,
                    writer_id,
                    C_cores[i],
                    device,
                    input_tensor_a,
                    *mesh_dispatch_coordinate,
                    output_tensor,
                    route,
                    output_core,
                    output_tile_offset,
                    {},
                    {},
                    {},
                    mux_config,
                    mux_core_by_dir,
                    termination_masters,
                    mux_worker_ids,
                    mux_client_counts);
            } else {
                set_all_gather_writer_runtime_args(
                    desc,
                    writer_id,
                    C_cores[i],
                    device,
                    input_tensor_a,
                    *mesh_dispatch_coordinate,
                    output_tensor,
                    route,
                    output_core,
                    output_tile_offset,
                    forwarding_links);
            }
        }
    }

    log_debug(
        tt::LogOp,
        "matmul_decode ring gather: num_senders={}, num_compute={}, M_tiles={}, K_tiles={}, "
        "inA_K_tiles_per_core={}, inB_N_tiles_per_core={}",
        num_senders,
        C_cores.size(),
        M_tiles,
        K_tiles,
        inA_K_tiles_per_core,
        inB_N_tiles_per_core);

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor MatmulDecodeDeviceOperation::AllGatherFullWidth::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    TT_FATAL(operation_attributes.all_gather, "internal error: all-gather workload factory used without all_gather");
    TT_FATAL(
        !operation_attributes.global_cb.has_value(),
        "matmul_decode all_gather is not supported with global_cb (tensor prefetcher)");

    auto* mesh_device = tensor_args.input_tensor_a.device();
    const auto subdevice_id = mesh_device->get_sub_device_ids().at(0);
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);

    tt::tt_metal::WorkloadDescriptor workload;
    workload.semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));
    workload.semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));
    const auto& out_ready_semaphore = workload.semaphores[0];
    const auto& barrier_semaphore = workload.semaphores[1];

    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);

    workload.programs.reserve(tensor_coords.coords().size());
    for (const auto& coord : tensor_coords.coords()) {
        auto descriptor = create_descriptor_ring_gather_full(
            operation_attributes, tensor_args, tensor_return_value, coord, &out_ready_semaphore, &barrier_semaphore);
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(descriptor)});
    }
    return workload;
}

}  // namespace ttnn::operations::experimental::matmul_decode
