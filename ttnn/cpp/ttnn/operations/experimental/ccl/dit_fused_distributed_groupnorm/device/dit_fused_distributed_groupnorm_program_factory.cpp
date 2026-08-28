// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_groupnorm_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/normalization/groupnorm/device/groupnorm_program_utils.hpp"
#include "ttnn/operations/normalization/groupnorm/groupnorm_grid_utils.hpp"
#include "dit_fused_distributed_groupnorm_device_operation_types.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace {

uint32_t float_to_u32(float v) {
    uint32_t out;
    std::memcpy(&out, &v, sizeof(float));
    return out;
}

}  // namespace

// Multi-core mcast GroupNorm across the device grid — the on-device reduction is identical to the
// stock ttnn::group_norm mcast path (groupnorm_mcast_program_factory.cpp): cores split into
// num_virtual_cols group-columns × spatial-row stacks; one mcast-group master per column reduces
// its column over the spatial rows, NoC-reads its peers' partials, and Welford-combines them into
// the device-global (mean, var). The ONLY additions for the distributed op are, on the master
// reader: (1) after the intra-device combine and BEFORE the mcast-back, the masters coalesce their
// per-group (mean, var) sub-sticks over one forwarder, ring all-gather every device's stats, and
// Chan-merge them into the GLOBAL (mean, var); (2) the master then mcasts the GLOBAL stat (not the
// device-local one) back to its column. Compute + writer run verbatim on every reduction core and
// are agnostic to sender/receiver role and to whether the stat is device-local or global. At
// ring_size==1 there is no fabric and the result is bit-exact with stock local GroupNorm.
DitFusedDistributedGroupnormMeshWorkloadFactory::cached_program_t
DitFusedDistributedGroupnormMeshWorkloadFactory::create_at(
    const DitFusedDistributedGroupnormParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const DitFusedDistributedGroupnormInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    Tensor& output_tensor = tensor_return_value.at(0);
    Tensor* stats_dram_tensor = tensor_return_value.size() > 1 ? &tensor_return_value[1] : nullptr;
    const auto& input_tensor = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto& input_mask = tensor_args.input_mask;

    Program program = CreateProgram();

    IDevice* device = input_tensor.device();
    auto* mesh_device = input_tensor.device();

    const bool is_local = (args.ring_size <= 1);
    const bool use_mux = !is_local;

    TT_FATAL(input_mask.has_value(), "dit_fused_distributed_groupnorm requires an input_mask (welford GroupNorm)");

    // ------------------------------------------------------------------------
    // Shapes
    // ------------------------------------------------------------------------
    const uint32_t tile_h = TILE_HEIGHT;
    const uint32_t tile_w = TILE_WIDTH;
    const uint32_t datum_size_bytes = 2;  // bfloat16

    const auto& shape = input_tensor.padded_shape();
    const uint32_t num_batches = shape[0];
    const uint32_t H = shape[1] * shape[2] * num_batches;  // folded spatial (per-device shard)
    const uint32_t Ht = H / tile_h;
    const uint32_t W = shape[3];  // channels
    const uint32_t Wt = W / tile_w;
    const uint32_t num_groups = args.num_groups;

    TT_FATAL(num_batches == 1, "dit_fused_distributed_groupnorm currently supports N==1 (got {})", num_batches);
    TT_FATAL(W % tile_w == 0, "channels ({}) must be divisible by {}", W, tile_w);
    TT_FATAL(W % num_groups == 0, "channels ({}) must be divisible by num_groups ({})", W, num_groups);
    TT_FATAL(H % tile_h == 0, "folded HW ({}) must be divisible by {}", H, tile_h);

    const uint32_t num_channels_per_group = W / num_groups;
    const uint32_t num_channels_per_group_mod_tile_w =
        (num_channels_per_group % tile_w == 0) ? tile_w : (num_channels_per_group % tile_w);

    // ------------------------------------------------------------------------
    // Multi-core mcast grid (mirror groupnorm_mcast_program_factory.cpp:118-164)
    // ------------------------------------------------------------------------
    const auto grid_size = device->compute_with_storage_grid_size();
    const CoreRange core_grid({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    const uint32_t max_cores = core_grid.size();

    const uint32_t num_virtual_cols =
        ttnn::operations::normalization::compute_num_virtual_cols(grid_size.x, static_cast<int>(num_groups), W);
    const uint32_t col_rep_max = grid_size.x / num_virtual_cols;  // ≥ 1

    // Spatial-row split. Unlike stock GroupNorm — which fills the whole grid.y and FATALs when the
    // input is too short — the per-device shard height Ht may be small (H is sharded across the
    // cluster). Pick the largest num_virtual_rows that (a) divides Ht, (b) factors as
    // col_replication × num_actual_rows within the grid (col_replication ≤ grid.x/num_virtual_cols,
    // num_actual_rows ≤ grid.y), and (c) leaves a core free for the fabric forwarder. This maximizes
    // the reduction parallelism while keeping per_core_Mt integral. vr==1 is always feasible.
    const uint32_t reduction_core_budget = max_cores - (use_mux ? 1u : 0u);
    uint32_t num_virtual_rows = 1, num_actual_rows = 1, col_replication = 1;
    for (uint32_t cr = 1; cr <= col_rep_max; ++cr) {
        for (uint32_t nr = 1; nr <= grid_size.y; ++nr) {
            const uint32_t vr = cr * nr;
            if (vr <= Ht && (Ht % vr) == 0 && (num_virtual_cols * vr) <= reduction_core_budget &&
                vr > num_virtual_rows) {
                num_virtual_rows = vr;
                num_actual_rows = nr;
                col_replication = cr;
            }
        }
    }
    const uint32_t num_actual_cols = num_virtual_cols * col_replication;
    const uint32_t num_reduction_cores = num_actual_cols * num_actual_rows;
    TT_FATAL(Ht % num_virtual_rows == 0, "internal: num_virtual_rows ({}) must divide Ht ({})", num_virtual_rows, Ht);

    const uint32_t per_core_Mt = Ht / num_virtual_rows;
    const uint32_t per_core_M = per_core_Mt * tile_h;
    const uint32_t per_core_N = W / num_virtual_cols;
    const uint32_t per_core_Nt = (per_core_N + tile_w - 1) / tile_w;
    const uint32_t num_shards_c = W / per_core_N;  // == num_virtual_cols
    const uint32_t num_groups_per_core = (num_groups > num_shards_c) ? (num_groups / num_shards_c) : 1u;
    const uint32_t num_batches_per_core = 1;         // N == 1
    const uint32_t num_rows_per_group = per_core_M;  // per-core rows per group (N == 1, one batch)

    auto [block_wt, num_groups_per_reset] = ttnn::prim::find_max_tile_span(per_core_N, num_channels_per_group);
    const uint32_t block_ht = per_core_Mt;  // num_batches_per_core == 1
    const uint32_t block_wt_last = (per_core_Nt + num_groups_per_core - 1) / num_groups_per_core;
    const uint32_t input_mask_num_tiles_per_core = block_wt * num_groups_per_core;
    const uint32_t per_core_N_bytes_padded =
        tt::round_up(per_core_N * datum_size_bytes, output_tensor.buffer()->alignment());

    // num_out_blocks: L1 safety net that splits the (now per-core) block height so height-scaled CBs
    // fit L1. With the multi-core split per_core_M is small, so this is typically 1. The stock
    // heuristic is keyed on the whole-tensor volume divided across cores; pass the per-core
    // footprint with a core count of 1 for the same result, then clamp to the block height.
    const uint32_t num_out_blocks =
        std::min(ttnn::prim::groupnorm_heuristic_num_out_blocks(per_core_M * per_core_N, 1u), block_ht);
    TT_FATAL(
        num_out_blocks >= 1 && num_out_blocks <= block_ht,
        "num_out_blocks ({}) must be in [1, block_ht ({})]",
        num_out_blocks,
        block_ht);

    TT_FATAL(num_groups_per_core <= 16, "num_groups_per_core ({}) must be <= 16 for welford", num_groups_per_core);
    TT_FATAL(
        input_mask->padded_shape()[3] == block_wt * tile_w,
        "input mask width ({}) must equal block_wt * TILE_WIDTH ({})",
        input_mask->padded_shape()[3],
        block_wt * tile_w);

    // ------------------------------------------------------------------------
    // Data formats
    // ------------------------------------------------------------------------
    const tt::DataFormat in_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat out_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    // Pinned to bf16: the cross-device gather packs each group's [mean, var] as a bf16 pair
    // (stick_bytes = num_groups_per_core * 4), so fp32 stats would need a different stick layout.
    const tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    const tt::DataFormat fp32_df = tt::DataFormat::Float32;
    tt::DataFormat gamma_beta_cb_data_format = tt::DataFormat::Float16_b;
    if (gamma.has_value()) {
        gamma_beta_cb_data_format = datatype_to_dataformat_converter(gamma->dtype());
    } else if (beta.has_value()) {
        gamma_beta_cb_data_format = datatype_to_dataformat_converter(beta->dtype());
    }
    const tt::DataFormat in_mask_cb_data_format = datatype_to_dataformat_converter(input_mask->dtype());

    const uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    const uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    const uint32_t gamma_beta_single_tile_size = tt::tile_size(gamma_beta_cb_data_format);
    const uint32_t in_mask_single_tile_size = tt::tile_size(in_mask_cb_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);
    TT_FATAL(fp32_dest_acc_en, "dit_fused_distributed_groupnorm requires fp32_dest_acc_en=true");

    const bool welford_unpack_fp32_active = (in_data_format == tt::DataFormat::Float32) && fp32_dest_acc_en;
    const bool welford_fp32_alias = welford_unpack_fp32_active;  // tilize_in == false
    const uint32_t cb_in0_welford_index =
        welford_fp32_alias ? static_cast<uint32_t>(tt::CBIndex::c_19) : static_cast<uint32_t>(tt::CBIndex::c_0);
    // V1 accepts tiled BF16 input, so the FP32 aliases remain inactive. Its input CB holds one streamed
    // block rather than the complete local input, requiring the reader to supply both statistics passes.
    const bool fp32_sfpu_normalizer = welford_unpack_fp32_active;
    constexpr uint32_t cb_ex_global_index = tt::CBIndex::c_15;
    constexpr uint32_t cb_ex2pe_index = tt::CBIndex::c_27;
    constexpr bool sfpu_two_pass_l1_replay = false;

    const bool has_gamma = gamma.has_value();
    const bool has_beta = beta.has_value();
    const uint32_t groupnorm_mode = static_cast<uint32_t>(ttnn::prim::GroupNormMode::TWO_PASS);

    // ------------------------------------------------------------------------
    // Ring topology (unchanged from CCL scaffold)
    // ------------------------------------------------------------------------
    std::optional<ttnn::MeshCoordinate> forward_coord = std::nullopt;
    std::optional<ttnn::MeshCoordinate> backward_coord = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;
    uint32_t device_index = 0;
    if (use_mux) {
        forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, mesh_coordinate, /*offset=*/1, args.topology, args.cluster_axis);
        backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, mesh_coordinate, /*offset=*/-1, args.topology, args.cluster_axis);
        device_index =
            ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, mesh_coordinate, args.cluster_axis);
        if (forward_coord.has_value()) {
            forward_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
        }
        if (backward_coord.has_value()) {
            backward_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
        }
    }

    uint32_t num_targets_forward = 0;
    uint32_t num_targets_backward = 0;
    if (use_mux) {
        std::tie(num_targets_forward, num_targets_backward) = ttnn::ccl::get_forward_backward_line_mcast_distance(
            args.ring_size, device_index, args.topology, /*static_alternate=*/true);
    }

    // ------------------------------------------------------------------------
    // Cores: reduction grid (num_actual_cols × num_actual_rows) + 1 forwarder when distributed.
    // ------------------------------------------------------------------------
    const std::vector<CoreCoord> core_coords =
        grid_to_cores(num_reduction_cores, num_actual_cols, num_actual_rows, /*row_wise=*/false);
    const std::vector<CoreCoord> virtual_core_coords =
        grid_to_cores(num_reduction_cores, num_virtual_cols, num_virtual_rows, /*row_wise=*/false);

    std::set<CoreRange> reduction_core_ranges;
    for (const auto& c : core_coords) {
        reduction_core_ranges.insert(CoreRange(c));
    }
    const CoreRangeSet all_reduction_cores(reduction_core_ranges);

    // Forwarder: one spare core outside the reduction grid. When the reduction layout leaves a
    // spare column/row (num_virtual_cols not dividing grid.x, the common case), it is free; if the
    // reduction grid exactly fills the device this shape can't host a forwarder → clear FATAL.
    const uint32_t num_forwarders = use_mux ? 1u : 0u;
    std::vector<CoreCoord> forwarder_cores;
    if (use_mux) {
        std::unordered_set<uint32_t> reduction_lookup;
        for (const auto& c : core_coords) {
            reduction_lookup.insert(c.y * grid_size.x + c.x);
        }
        const auto grid_cores = corerange_to_cores(core_grid, max_cores, /*row_wise=*/true);
        for (const auto& c : grid_cores) {
            if (!reduction_lookup.contains(c.y * grid_size.x + c.x)) {
                forwarder_cores.push_back(c);
                break;
            }
        }
        TT_FATAL(
            !forwarder_cores.empty(),
            "distributed dit_fused_distributed_groupnorm needs a spare core for the fabric forwarder, but the "
            "reduction grid ({}x{}={}) fills the whole device ({} cores).",
            num_actual_cols,
            num_actual_rows,
            num_reduction_cores,
            max_cores);
    }

    CoreRangeSet forwarder_core_set;
    for (const auto& c : forwarder_cores) {
        forwarder_core_set = forwarder_core_set.merge(CoreRangeSet({CoreRange(c, c)}));
    }

    // ------------------------------------------------------------------------
    // Mcast groups (mirror groupnorm_mcast_program_factory.cpp:324-364). For N==1 the senders are
    // core_coords[0], [num_virtual_rows], [2*num_virtual_rows], … — one master per group-column.
    // ------------------------------------------------------------------------
    const uint32_t num_cores_per_batch = num_virtual_rows;  // N == 1
    const uint32_t sender_stride = num_virtual_rows;
    std::set<CoreRange> mcast_sender_core_ranges;
    {
        uint32_t core_index = 0;
        for (uint32_t j = 0; j < num_groups / num_groups_per_core; ++j) {
            mcast_sender_core_ranges.insert(CoreRange(core_coords[core_index]));
            core_index += sender_stride;
        }
    }
    (void)num_cores_per_batch;

    std::vector<std::vector<CoreCoord>> mcast_groups;
    std::vector<std::vector<CoreCoord>> mcast_virtual_groups;
    int group_index = -1;
    for (size_t i = 0; i < core_coords.size(); ++i) {
        if (mcast_sender_core_ranges.contains(CoreRange(core_coords[i]))) {
            group_index += 1;
        }
        if (group_index >= static_cast<int>(mcast_groups.size())) {
            mcast_groups.emplace_back();
            mcast_virtual_groups.emplace_back();
        }
        mcast_groups[group_index].push_back(core_coords[i]);
        mcast_virtual_groups[group_index].push_back(virtual_core_coords[i]);
    }
    const uint32_t num_cores_per_mcast_group = mcast_groups[0].size();
    const uint32_t num_masters = static_cast<uint32_t>(mcast_groups.size());

    std::set<CoreRange> mcast_receiver_core_ranges;
    for (const auto& c : core_coords) {
        if (!mcast_sender_core_ranges.contains(CoreRange(c))) {
            mcast_receiver_core_ranges.insert(CoreRange(c));
        }
    }
    const CoreRangeSet mcast_sender_cores(mcast_sender_core_ranges);
    const CoreRangeSet mcast_receiver_cores(mcast_receiver_core_ranges);
    const bool has_receivers = !mcast_receiver_core_ranges.empty();

    // ------------------------------------------------------------------------
    // Fabric geometry (single source of truth shared with create_stats_buffer / validate).
    // ------------------------------------------------------------------------
    const auto sizing = compute_sizing(args, input_tensor);
    TT_FATAL(
        !use_mux || sizing.num_masters == num_masters,
        "internal: sizing masters ({}) != program masters ({})",
        sizing.num_masters,
        num_masters);
    const uint32_t stick_bytes = sizing.stick_bytes;  // per-master, 64 B-aligned
    const uint32_t num_chunks_per_device = sizing.num_chunks_per_device;
    constexpr uint32_t max_rounds = 1u;

    TT_FATAL(
        !use_mux || stats_dram_tensor != nullptr,
        "create_at requires stats DRAM scratch at tensor_return_value[1] when ring_size > 1");
    tt::tt_metal::Buffer* stats_dram_buffer = use_mux ? stats_dram_tensor->buffer() : nullptr;

    if (use_mux) {
        const uint32_t max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
        TT_FATAL(
            num_masters * stick_bytes <= max_payload,
            "coalesced GN stats packet ({} masters × {} B = {} B) exceeds fabric max payload ({} B)",
            num_masters,
            stick_bytes,
            num_masters * stick_bytes,
            max_payload);
    }

    // ------------------------------------------------------------------------
    // Circular buffers
    // ------------------------------------------------------------------------
    const uint32_t in0_block_tiles = block_ht / num_out_blocks * block_wt;
    const uint32_t interm_block_tiles = in0_block_tiles;

    // c_0 input (+ c_19 alias on fp32 path)
    if (welford_fp32_alias) {
        const uint32_t in0_cbs[] = {static_cast<uint32_t>(tt::CBIndex::c_0), static_cast<uint32_t>(tt::CBIndex::c_19)};
        create_cb(in0_cbs, program, all_reduction_cores, in_single_tile_size, in0_block_tiles, in_data_format);
    } else {
        create_cb(tt::CBIndex::c_0, program, all_reduction_cores, in_single_tile_size, in0_block_tiles, in_data_format);
    }
    create_cb(tt::CBIndex::c_16, program, all_reduction_cores, out_single_tile_size, in0_block_tiles, out_data_format);
    create_cb(tt::CBIndex::c_29, program, all_reduction_cores, in_single_tile_size, in0_block_tiles, in_data_format);
    create_cb(tt::CBIndex::c_2, program, all_reduction_cores, single_tile_size, 1, cb_data_format);
    create_cb(tt::CBIndex::c_3, program, all_reduction_cores, single_tile_size, 1, cb_data_format);  // eps
    create_cb(tt::CBIndex::c_4, program, all_reduction_cores, single_tile_size, 1, cb_data_format);
    if (has_gamma) {
        create_cb(
            tt::CBIndex::c_5,
            program,
            all_reduction_cores,
            gamma_beta_single_tile_size,
            per_core_Nt,
            gamma_beta_cb_data_format);
    }
    if (has_beta) {
        create_cb(
            tt::CBIndex::c_6,
            program,
            all_reduction_cores,
            gamma_beta_single_tile_size,
            per_core_Nt,
            gamma_beta_cb_data_format);
    }
    create_cb(
        tt::CBIndex::c_28,
        program,
        all_reduction_cores,
        in_mask_single_tile_size,
        input_mask_num_tiles_per_core,
        in_mask_cb_data_format);
    create_cb(tt::CBIndex::c_24, program, all_reduction_cores, single_tile_size, 1, cb_data_format);  // x
    create_cb(tt::CBIndex::c_25, program, all_reduction_cores, single_tile_size, 3, cb_data_format);  // xmm
    create_cb(
        tt::CBIndex::c_23, program, all_reduction_cores, single_tile_size, interm_block_tiles, cb_data_format);  // xmm2
    create_cb(
        tt::CBIndex::c_22, program, all_reduction_cores, single_tile_size, interm_block_tiles, cb_data_format);  // xmm3
    create_cb(tt::CBIndex::c_8, program, all_reduction_cores, single_tile_size, 2, cb_data_format);  // ex_partial
    // ex_global: dual buffer index c_15 / c_9.
    {
        const uint32_t ex_global_cbs[] = {
            static_cast<uint32_t>(tt::CBIndex::c_15), static_cast<uint32_t>(tt::CBIndex::c_9)};
        create_cb(
            ex_global_cbs, program, all_reduction_cores, single_tile_size, 2u * num_groups_per_core, cb_data_format);
    }
    create_cb(
        tt::CBIndex::c_27,
        program,
        all_reduction_cores,
        single_tile_size,
        num_groups_per_core,
        cb_data_format);  // ex2pe

    // Fabric AG staging CBs (masters) + packet CB (grid-uniform L1 addr) + header CB (forwarder).
    // fp32 is a byte container here; the payload is bf16 [mean, var] pairs (see stick_bytes).
    constexpr uint32_t stats_local_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t stats_gathered_cb_id = tt::CBIndex::c_10;
    constexpr uint32_t packet_cb_id = tt::CBIndex::c_7;
    constexpr uint32_t reserved_packet_header_cb_id = tt::CBIndex::c_11;
    if (use_mux) {
        create_cb(stats_local_cb_id, program, mcast_sender_cores, stick_bytes, 2, fp32_df);
        create_cb(stats_gathered_cb_id, program, mcast_sender_cores, stick_bytes, args.ring_size, fp32_df);
        // One coalesced packet = num_masters sub-sticks; ping-ponged (depth 2). Created on the whole
        // grid so masters and the forwarder see the same L1 base address.
        const uint32_t packet_bytes = num_masters * stick_bytes;
        create_cb(packet_cb_id, program, CoreRangeSet({core_grid}), packet_bytes, 2, fp32_df);
        const uint32_t header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        create_cb(reserved_packet_header_cb_id, program, forwarder_core_set, header_size, 4, tt::DataFormat::RawUInt32);
    }

    // ------------------------------------------------------------------------
    // Semaphores: reduce_sender (id 0) + reduce_receiver (id 1); fabric arrival / go on full grid.
    // ------------------------------------------------------------------------
    const uint32_t reduce_sender_semaphore_id = CreateSemaphore(program, all_reduction_cores, 0u);
    const uint32_t reduce_receiver_semaphore_id = CreateSemaphore(program, all_reduction_cores, 0u);
    const uint32_t arrival_sem_id = use_mux ? CreateSemaphore(program, CoreRangeSet({core_grid}), 0u) : 0u;
    const uint32_t go_sem_id = use_mux ? CreateSemaphore(program, CoreRangeSet({core_grid}), 0u) : 0u;

    // Kernels are the stock welford GroupNorm kernels plus the shared fused-norm CCL forwarder. The
    // only fused-specific behaviour lives behind GN_DISTRIBUTED_AG in the two reader kernels; the
    // compute/writer kernels are used verbatim. At ring_size == 1 the define is not set, so the
    // fused op compiles the exact stock kernel text.
    const std::string gn_kernel_base = "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/";
    const std::string ccl_kernel_base = "ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_norm_common/kernels/";

    // Refers to the welford stats CBs (ex_partial / ex_global / ex2pe), not the fabric staging CBs.
    const bool stats_is_fp32 = cb_data_format == tt::DataFormat::Float32;
    // cb_reciprocals is excluded: it's fp32 here but the reconfigs never touch it.
    const bool enable_fp32_reconfig = ttnn::prim::groupnorm_needs_fp32_reconfig(
        {in_data_format, out_data_format, cb_data_format, gamma_beta_cb_data_format, in_mask_cb_data_format});

    // ------------------------------------------------------------------------
    // Reader compile-time args. Named args mirror the stock welford GN reader exactly; the
    // TensorAccessor blocks stay positional. The master appends the stats-DRAM accessor after
    // src0 (the stock reader only declares src0, so the extra block is inert unless
    // GN_DISTRIBUTED_AG is set).
    // ------------------------------------------------------------------------
    const std::unordered_map<std::string, uint32_t> reader_named_ct = {
        {"reduce_receiver_semaphore_id", reduce_receiver_semaphore_id},
        {"reduce_sender_semaphore_id", reduce_sender_semaphore_id},
        {"num_cores_per_mcast_group", num_cores_per_mcast_group},
        {"num_batch_group", num_groups_per_core * num_batches_per_core},
        {"num_batches", num_batches_per_core},
        {"per_core_N", per_core_Nt},
        {"per_core_N_bytes", per_core_N_bytes_padded},
        {"per_core_N_bytes_with_stride", per_core_Nt * tile_w * datum_size_bytes},
        {"datum_size_bytes", datum_size_bytes},
        {"per_core_M", per_core_Mt},
        {"TILE_HEIGHT", tile_h},
        {"TILE_WIDTH", tile_w},
        {"block_h", block_ht},
        {"block_w", block_wt},
        {"block_hw", block_ht * block_wt},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"num_tiles_per_batch", per_core_Mt * Wt / num_batches_per_core},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         static_cast<uint32_t>((num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0)},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", static_cast<uint32_t>(num_channels_per_group < tile_w)},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * tile_w)},
        {"num_out_blocks", num_out_blocks},
        {"num_channels_per_group", num_channels_per_group},
        {"num_rows_per_group", num_rows_per_group},
        {"welford_fp32_alias", static_cast<uint32_t>(welford_fp32_alias)},
        {"cb_in0_welford", cb_in0_welford_index},
        {"stats_is_fp32", static_cast<uint32_t>(stats_is_fp32)},
        {"sfpu_two_pass_l1_replay", static_cast<uint32_t>(sfpu_two_pass_l1_replay)},
    };

    std::map<std::string, std::string> reader_defines;
    if (has_gamma) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (has_beta) {
        reader_defines["FUSE_BETA"] = "1";
    }
    // Only the cross-device path enables the fabric AG + batched handshake. ring_size == 1 leaves
    // the stock per-group lock-step untouched, so the local path is bit-exact with ttnn.group_norm.
    if (use_mux) {
        reader_defines["GN_DISTRIBUTED_AG"] = "1";
    }

    // Master / sender reader (fabric AG when distributed).
    std::unordered_map<std::string, uint32_t> sender_reader_named_ct = reader_named_ct;
    if (use_mux) {
        sender_reader_named_ct.emplace("ring_size", args.ring_size);
        sender_reader_named_ct.emplace("stick_bytes", stick_bytes);
        sender_reader_named_ct.emplace("num_chunks_per_device", num_chunks_per_device);
        sender_reader_named_ct.emplace("fwd_arrival_sem_id", arrival_sem_id);
        sender_reader_named_ct.emplace("go_sem_id", go_sem_id);
        sender_reader_named_ct.emplace("packet_cb_id", packet_cb_id);
        sender_reader_named_ct.emplace("stats_local_cb_id", stats_local_cb_id);
        sender_reader_named_ct.emplace("stats_gathered_cb_id", stats_gathered_cb_id);
    }
    std::vector<uint32_t> sender_reader_ct;
    TensorAccessorArgs(input_tensor.buffer()).append_to(sender_reader_ct);
    TensorAccessorArgs(use_mux ? stats_dram_buffer : input_tensor.buffer()).append_to(sender_reader_ct);

    KernelHandle sender_reader_kernel_id = CreateKernel(
        program,
        gn_kernel_base + "dataflow/welford_reader_mcast_sender_unary_gn.cpp",
        mcast_sender_cores,
        ReaderDataMovementConfig(sender_reader_ct, reader_defines, sender_reader_named_ct));

    // Receiver reader (signal + wait; no fabric).
    KernelHandle receiver_reader_kernel_id = 0;
    if (has_receivers) {
        std::vector<uint32_t> receiver_reader_ct;
        TensorAccessorArgs(input_tensor.buffer()).append_to(receiver_reader_ct);
        receiver_reader_kernel_id = CreateKernel(
            program,
            gn_kernel_base + "dataflow/welford_reader_mcast_receiver_unary_gn.cpp",
            mcast_receiver_cores,
            ReaderDataMovementConfig(receiver_reader_ct, reader_defines, reader_named_ct));
    }

    // ------------------------------------------------------------------------
    // Writer (stock welford GN gamma/beta + output) — identical on every reduction core.
    // Runtime args and CB indices already match stock, so this kernel is used unmodified.
    // ------------------------------------------------------------------------
    const std::unordered_map<std::string, uint32_t> writer_named_ct = {
        {"is_mcast_sender", 1u},  // stock hardcodes 1 on all cores
        {"fuse_gamma", static_cast<uint32_t>(has_gamma)},
        {"fuse_beta", static_cast<uint32_t>(has_beta)},
        {"num_cols_tile_gamma_beta", per_core_Nt},
        {"per_core_M", per_core_Mt},
        {"per_core_N", per_core_Nt},
        {"per_core_N_bytes", per_core_N * datum_size_bytes},
        {"per_core_N_bytes_with_stride", per_core_Nt * tile_w * datum_size_bytes},
        {"num_groups_per_core", num_groups_per_core},
        {"num_batches_per_core", num_batches_per_core},
        {"num_cols_per_group", num_channels_per_group_mod_tile_w},
        {"num_tiles_per_batch", per_core_Mt * Wt / num_batches_per_core},
        {"block_w_last", block_wt_last},
        {"GROUP_SIZE_IS_POWER_OF_2",
         static_cast<uint32_t>((num_channels_per_group_mod_tile_w & (num_channels_per_group_mod_tile_w - 1)) == 0)},
        {"GROUP_SIZE_SMALLER_THAN_TILE_W", static_cast<uint32_t>(num_channels_per_group < tile_w)},
        {"group_row_offset", num_channels_per_group - ((block_wt - 1) * tile_w)},
        {"num_out_blocks", num_out_blocks},
        {"block_h", block_ht},
        {"block_w", block_wt},
        {"block_hw", block_ht * block_wt},
        {"groupnorm_mode", groupnorm_mode},
        {"TILE_WIDTH", tile_w},
    };
    std::vector<uint32_t> writer_ct;
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct);
    TensorAccessorArgs(has_gamma ? gamma->buffer() : nullptr).append_to(writer_ct);
    TensorAccessorArgs(has_beta ? beta->buffer() : nullptr).append_to(writer_ct);
    TensorAccessorArgs(input_mask->buffer()).append_to(writer_ct);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        gn_kernel_base + "dataflow/welford_writer_unary_gn_rm_gb.cpp",
        all_reduction_cores,
        WriterDataMovementConfig(writer_ct, {}, writer_named_ct));

    // ------------------------------------------------------------------------
    // Forwarder (single fabric ring all-gather, coalescing all masters' sub-sticks).
    // ------------------------------------------------------------------------
    std::vector<KernelHandle> forwarder_kernel_ids(num_forwarders, 0);
    for (uint32_t f = 0; f < num_forwarders; f++) {
        std::vector<uint32_t> fwd_ct = {
            packet_cb_id,
            reserved_packet_header_cb_id,
            args.ring_size,
            device_index,
            num_targets_forward,
            num_targets_backward,
            f,
            num_forwarders,
            num_masters,  // group_size (masters coalesced by this forwarder)
            max_rounds,
            stick_bytes,
            num_chunks_per_device,
            arrival_sem_id,
            go_sem_id,
        };
        TensorAccessorArgs(stats_dram_buffer).append_to(fwd_ct);
        // The coalescing forwarder is fully parameterized by these CT args (stick_bytes,
        // max_rounds, num_chunks_per_device), so it is shared verbatim with the fused rmsnorm op.
        forwarder_kernel_ids[f] = CreateKernel(
            program,
            ccl_kernel_base + "dataflow/dit_fused_norm_forwarder.cpp",
            CoreRangeSet({CoreRange(forwarder_cores[f], forwarder_cores[f])}),
            WriterDataMovementConfig(fwd_ct));
    }

    // ------------------------------------------------------------------------
    // Compute (stock welford GroupNorm PRE/POST) — identical on every reduction core.
    // ------------------------------------------------------------------------
    const std::unordered_map<std::string, uint32_t> compute_named_ct = {
        {"do_gamma", static_cast<uint32_t>(has_gamma)},
        {"do_beta", static_cast<uint32_t>(has_beta)},
        {"num_cores_per_mcast_group", num_cores_per_mcast_group},
        {"batch", num_batches_per_core},
        {"group", num_groups_per_core},
        {"block_h", block_ht},
        {"block_w", block_wt},
        {"per_core_M", per_core_Mt},
        {"per_core_N", per_core_Nt},
        {"per_core_MN", per_core_Mt * per_core_Nt},
        {"single_tile_size_bytes", single_tile_size},
        {"num_tiles_input_mask", num_groups_per_core * block_wt},
        {"num_out_blocks", num_out_blocks},
        {"num_channels_per_group", num_channels_per_group},
        {"TILE_WIDTH", tile_w},
        {"cb_in0_welford", cb_in0_welford_index},
        {"welford_fp32_alias", static_cast<uint32_t>(welford_fp32_alias)},
        {"welford_unpack_fp32_active", static_cast<uint32_t>(welford_unpack_fp32_active)},
        {"enable_fp32_reconfig", static_cast<uint32_t>(enable_fp32_reconfig)},
        {"fp32_sfpu_normalizer", static_cast<uint32_t>(fp32_sfpu_normalizer)},
        {"cb_normalize_in_fp32", cb_in0_welford_index},
        {"cb_ex_global_fp32", cb_ex_global_index},
        {"cb_ex2pe_fp32", cb_ex2pe_index},
        {"sfpu_two_pass_l1_replay", static_cast<uint32_t>(sfpu_two_pass_l1_replay)},
        {"sfpu_two_pass_reciprocal",
         std::bit_cast<uint32_t>(
             1.0f / static_cast<float>(std::max(1U, num_channels_per_group * num_rows_per_group / tile_w)))},
    };
    // Optional fused unary activation. "dst0" is the DEST index the final output tile lives in
    // when the compute kernel applies the activation (see welford_groupnorm.cpp).
    std::map<std::string, std::string> compute_defines;
    if (args.fused_activation.has_value()) {
        const auto& act = args.fused_activation.value();
        for (auto& [key, val] : ttnn::operations::unary::utils::get_defines(
                 act.op_type, act.params, "ACTIVATION", "dst0", output_tensor.dtype())) {
            compute_defines[key] = val;
        }
    }

    ComputeConfig compute_config{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
        .defines = compute_defines,
        .named_compile_args = compute_named_ct,
    };
    if (welford_unpack_fp32_active) {
        std::vector<UnpackToDestMode> unpack_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
        if (welford_fp32_alias) {
            unpack_mode[static_cast<uint32_t>(tt::CBIndex::c_19)] = UnpackToDestMode::UnpackToDestFp32;
        }
        compute_config.unpack_to_dest_mode = unpack_mode;
    }
    KernelHandle compute_kernel_id =
        CreateKernel(program, gn_kernel_base + "compute/welford_groupnorm.cpp", all_reduction_cores, compute_config);

    // ------------------------------------------------------------------------
    // Runtime args
    // ------------------------------------------------------------------------
    const uint32_t input_addr = input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();
    const uint32_t gamma_addr = has_gamma ? gamma->buffer()->address() : 0u;
    const uint32_t beta_addr = has_beta ? beta->buffer()->address() : 0u;
    const uint32_t input_mask_addr = input_mask->buffer()->address();
    const uint32_t stats_dram_addr = use_mux ? stats_dram_buffer->address() : 0u;

    uint32_t out_ready_sem_bank_addr = 0;
    if (use_mux) {
        TT_FATAL(!args.multi_device_global_semaphore.empty(), "ring_size>1 requires a GlobalSemaphore");
        out_ready_sem_bank_addr = args.multi_device_global_semaphore.at(0).address();
    }

    const tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    CoreCoord forwarder_virtual = {};
    if (use_mux) {
        forwarder_virtual = mesh_device->worker_core_from_logical_core(forwarder_cores[0]);
    }

    std::vector<size_t> stats_dram_addr_reader_arg_idx;
    std::vector<CoreCoord> sender_cores_out;
    std::vector<CoreCoord> receiver_cores_out;

    for (size_t i = 0; i < mcast_groups.size(); ++i) {
        std::vector<CoreCoord> group = mcast_groups[i];  // non-const: split_and_form_rectangle_grids mutates it
        const auto& virtual_group = mcast_virtual_groups[i];
        const bool rectangle_grid = ttnn::prim::is_rectangle_grid(group);

        for (size_t j = 0; j < group.size(); ++j) {
            const CoreCoord core = group[j];
            const CoreCoord virtual_core = virtual_group[j];
            const uint32_t in0_start_id = per_core_Mt * Wt * virtual_core.y + per_core_Nt * virtual_core.x;
            const uint32_t out_tile_start_id = in0_start_id;

            if (j == 0) {  // mcast-group master (sender reader + fabric)
                sender_cores_out.push_back(core);
                std::vector<CoreCoord> mcast_group_first;
                std::vector<CoreCoord> mcast_group_mid(group);
                std::vector<CoreCoord> mcast_group_last;
                if (!rectangle_grid) {
                    ttnn::prim::split_and_form_rectangle_grids(
                        group, mcast_group_first, mcast_group_mid, mcast_group_last);
                }

                CoreCoord mcast_start = device->worker_core_from_logical_core(mcast_group_mid.front());
                CoreCoord mcast_end = device->worker_core_from_logical_core(mcast_group_mid.back());
                if (reader_noc == NOC::NOC_1) {
                    std::swap(mcast_start, mcast_end);
                }

                std::vector<uint32_t> rt;
                rt.push_back(input_addr);
                rt.push_back(output_addr);
                rt.push_back(in0_start_id);
                rt.push_back(out_tile_start_id);
                rt.push_back(Wt);
                rt.push_back(static_cast<uint32_t>(!mcast_group_first.empty()));
                rt.push_back(static_cast<uint32_t>(!mcast_group_last.empty()));
                rt.push_back(mcast_start.x);
                rt.push_back(mcast_start.y);
                rt.push_back(mcast_end.x);
                rt.push_back(mcast_end.y);
                rt.push_back(mcast_group_first.empty() ? (mcast_group_mid.size() - 1) : mcast_group_mid.size());

                if (!mcast_group_first.empty()) {
                    CoreCoord fs = device->worker_core_from_logical_core(mcast_group_first.front());
                    CoreCoord fe = device->worker_core_from_logical_core(mcast_group_first.back());
                    rt.push_back(fs.x);
                    rt.push_back(fs.y);
                    rt.push_back(fe.x);
                    rt.push_back(fe.y);
                    rt.push_back(mcast_group_first.size() - 1);
                }
                if (!mcast_group_last.empty()) {
                    CoreCoord ls = device->worker_core_from_logical_core(mcast_group_last.front());
                    CoreCoord le = device->worker_core_from_logical_core(mcast_group_last.back());
                    rt.push_back(ls.x);
                    rt.push_back(ls.y);
                    rt.push_back(le.x);
                    rt.push_back(le.y);
                    rt.push_back(mcast_group_last.size());
                }

                std::vector<uint32_t> noc_x;
                std::vector<uint32_t> noc_y;
                for (const auto& gcore : group) {
                    CoreCoord vc = device->worker_core_from_logical_core(gcore);
                    noc_x.push_back(vc.x);
                    noc_y.push_back(vc.y);
                }
                rt.insert(rt.end(), noc_x.begin(), noc_x.end());
                rt.insert(rt.end(), noc_y.begin(), noc_y.end());

                if (use_mux) {
                    // welford_reader_mcast_sender_unary_gn.cpp locates the fabric block from the
                    // end of the noc_coord_x/y lists, whose offset depends on which optional mcast
                    // sub-group blocks this core emitted. Keep the two layouts in agreement.
                    const size_t kernel_noc_arg_base = (!mcast_group_first.empty() && !mcast_group_last.empty()) ? 22u
                                                       : (!mcast_group_first.empty() || !mcast_group_last.empty())
                                                           ? 17u
                                                           : 12u;
                    const size_t kernel_fabric_base = kernel_noc_arg_base + 2u * group.size();
                    TT_FATAL(
                        rt.size() == kernel_fabric_base,
                        "sender reader runtime-arg layout mismatch: host writes the fabric block at index {}, "
                        "the kernel reads it at index {}",
                        rt.size(),
                        kernel_fabric_base);
                    TT_FATAL(
                        group.size() == num_cores_per_mcast_group,
                        "mcast group {} has {} cores but the kernel is compiled with num_cores_per_mcast_group={}",
                        i,
                        group.size(),
                        num_cores_per_mcast_group);
                    stats_dram_addr_reader_arg_idx.push_back(rt.size());
                    rt.push_back(stats_dram_addr);
                    rt.push_back(static_cast<uint32_t>(forwarder_virtual.x));
                    rt.push_back(static_cast<uint32_t>(forwarder_virtual.y));
                    rt.push_back(static_cast<uint32_t>(i));  // my_slot = master index
                    rt.push_back(0u);                        // my_forwarder_index (single forwarder)
                }
                SetRuntimeArgs(program, sender_reader_kernel_id, core, rt);
            } else {  // receiver reader
                receiver_cores_out.push_back(core);
                CoreCoord sender_virtual = device->worker_core_from_logical_core(group.front());
                std::vector<uint32_t> rt = {
                    input_addr,
                    output_addr,
                    in0_start_id,
                    out_tile_start_id,
                    Wt,
                    static_cast<uint32_t>(sender_virtual.x),
                    static_cast<uint32_t>(sender_virtual.y)};
                SetRuntimeArgs(program, receiver_reader_kernel_id, core, rt);
            }
        }
    }

    // Writer RT args (welford GN) per reduction core, with per-column gamma/beta/mask offsets.
    uint32_t gamma_tile_start_id = 0;
    uint32_t beta_tile_start_id = 0;
    uint32_t input_mask_tile_start_id = 0;
    uint32_t curr_virtual_core_x = 0;
    const uint32_t gamma_beta_num_cols_tile_per_core = per_core_Nt;
    for (size_t i = 0; i < core_coords.size(); ++i) {
        const CoreCoord core = core_coords[i];
        const CoreCoord virtual_core = virtual_core_coords[i];
        const uint32_t out_tile_start_id = per_core_Mt * Wt * virtual_core.y + per_core_Nt * virtual_core.x;

        if (virtual_core.x > curr_virtual_core_x) {
            curr_virtual_core_x++;
            if (has_gamma) {
                gamma_tile_start_id =
                    (gamma_tile_start_id + gamma_beta_num_cols_tile_per_core) % (gamma->physical_volume() / tile_w);
            }
            if (has_beta) {
                beta_tile_start_id =
                    (beta_tile_start_id + gamma_beta_num_cols_tile_per_core) % (beta->physical_volume() / tile_w);
            }
            input_mask_tile_start_id = (input_mask_tile_start_id + input_mask_num_tiles_per_core) %
                                       (input_mask->physical_volume() / (tile_h * tile_w));
        }

        std::vector<uint32_t> writer_rt = {
            float_to_u32(args.eps),
            output_addr,
            gamma_addr,
            beta_addr,
            input_mask_addr,
            out_tile_start_id,  // per-core output page start
            gamma_tile_start_id,
            beta_tile_start_id,
            input_mask_tile_start_id,
            Wt,
        };
        SetRuntimeArgs(program, writer_kernel_id, core, writer_rt);
    }

    // Forwarder RT args: gather all masters' NoC coords (slot order), present_count = num_masters.
    if (num_forwarders > 0) {
        const auto local_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
        for (uint32_t f = 0; f < num_forwarders; f++) {
            const auto& core = forwarder_cores[f];
            std::vector<uint32_t> fwd_rt = {stats_dram_addr, out_ready_sem_bank_addr};
            for (uint32_t m = 0; m < num_masters; m++) {
                CoreCoord mv = device->worker_core_from_logical_core(mcast_groups[m].front());
                fwd_rt.push_back(static_cast<uint32_t>(mv.x));
                fwd_rt.push_back(static_cast<uint32_t>(mv.y));
            }
            fwd_rt.push_back(num_masters);  // present_count[0]
            fwd_rt.push_back(forward_fabric_node_id.has_value() ? 1u : 0u);
            if (forward_fabric_node_id.has_value()) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    local_node_id, forward_fabric_node_id.value(), /*link_idx=*/f, program, {core}, fwd_rt);
            }
            fwd_rt.push_back(backward_fabric_node_id.has_value() ? 1u : 0u);
            if (backward_fabric_node_id.has_value()) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    local_node_id, backward_fabric_node_id.value(), /*link_idx=*/f, program, {core}, fwd_rt);
            }
            SetRuntimeArgs(program, forwarder_kernel_ids[f], core, fwd_rt);
        }
    }

    std::vector<KernelHandle> reader_ids = {sender_reader_kernel_id};
    if (has_receivers) {
        reader_ids.push_back(receiver_reader_kernel_id);
    }

    return {
        std::move(program),
        DitFusedDistributedGroupnormSharedVariables{
            .reader_kernel_ids = reader_ids,
            .writer_kernel_ids = {writer_kernel_id},
            .compute_kernel_ids = {compute_kernel_id},
            .forwarder_kernel_ids = forwarder_kernel_ids,
            .forwarder_cores = forwarder_cores,
            .cores = core_coords,
            .sender_cores = sender_cores_out,
            .receiver_cores = receiver_cores_out,
            .stats_dram_addr_reader_arg_idx = stats_dram_addr_reader_arg_idx,
        }};
}

DitFusedDistributedGroupnormMeshWorkloadFactory::cached_mesh_workload_t
DitFusedDistributedGroupnormMeshWorkloadFactory::create_mesh_workload(
    const DitFusedDistributedGroupnormParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const DitFusedDistributedGroupnormInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& coord : range) {
            auto cached = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
            workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached.program));
            shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached.shared_variables));
        }
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void DitFusedDistributedGroupnormMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const DitFusedDistributedGroupnormParams& operation_attributes,
    const DitFusedDistributedGroupnormInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t output_addr = tensor_return_value.at(0).buffer()->address();
    const uint32_t gamma_addr = tensor_args.gamma.has_value() ? tensor_args.gamma->buffer()->address() : 0u;
    const uint32_t beta_addr = tensor_args.beta.has_value() ? tensor_args.beta->buffer()->address() : 0u;
    const uint32_t input_mask_addr =
        tensor_args.input_mask.has_value() ? tensor_args.input_mask->buffer()->address() : 0u;
    const uint32_t stats_dram_addr = tensor_return_value.size() > 1 ? tensor_return_value[1].buffer()->address() : 0u;
    const uint32_t out_ready_sem_addr = operation_attributes.multi_device_global_semaphore.empty()
                                            ? 0u
                                            : operation_attributes.multi_device_global_semaphore.at(0).address();

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared = cached_workload.shared_variables.at(range);

        // Master / sender reader: input, output, and (fabric) stats DRAM address.
        auto& sender_args_by_core = GetRuntimeArgs(program, shared.reader_kernel_ids[0]);
        const bool has_stats_arg = !shared.stats_dram_addr_reader_arg_idx.empty();
        TT_FATAL(
            !has_stats_arg || shared.stats_dram_addr_reader_arg_idx.size() == shared.sender_cores.size(),
            "stats_dram_addr_reader_arg_idx has {} entries but there are {} sender cores",
            shared.stats_dram_addr_reader_arg_idx.size(),
            shared.sender_cores.size());
        for (size_t s = 0; s < shared.sender_cores.size(); ++s) {
            const auto& core = shared.sender_cores[s];
            auto& a = sender_args_by_core.at(core.x).at(core.y);
            a[0] = input_addr;
            a[1] = output_addr;
            if (has_stats_arg) {
                a[shared.stats_dram_addr_reader_arg_idx[s]] = stats_dram_addr;
            }
        }

        // Receiver reader: input, output.
        if (shared.reader_kernel_ids.size() > 1 && !shared.receiver_cores.empty()) {
            auto& recv_args_by_core = GetRuntimeArgs(program, shared.reader_kernel_ids[1]);
            for (const auto& core : shared.receiver_cores) {
                auto& a = recv_args_by_core.at(core.x).at(core.y);
                a[0] = input_addr;
                a[1] = output_addr;
            }
        }

        // Writer (all reduction cores): output, gamma, beta, mask.
        auto& writer_args_by_core = GetRuntimeArgs(program, shared.writer_kernel_ids[0]);
        for (const auto& core : shared.cores) {
            auto& a = writer_args_by_core.at(core.x).at(core.y);
            a[1] = output_addr;
            a[2] = gamma_addr;
            a[3] = beta_addr;
            a[4] = input_mask_addr;
        }

        for (size_t f = 0; f < shared.forwarder_kernel_ids.size(); f++) {
            auto& fwd_args_by_core = GetRuntimeArgs(program, shared.forwarder_kernel_ids[f]);
            const auto& fc = shared.forwarder_cores[f];
            fwd_args_by_core.at(fc.x).at(fc.y)[0] = stats_dram_addr;
            fwd_args_by_core.at(fc.x).at(fc.y)[1] = out_ready_sem_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim
