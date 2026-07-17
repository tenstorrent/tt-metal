// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of the non-width-sharded conv2d program factory.
//
// Scope (per CB_TAXONOMY_ANALYSIS.md / METAL2_PORT_REPORT.md):
//   - height-sharded conv
//   - block-sharded conv WITHOUT split_reader_cb_shared (the shared-overlap second-writer side-channel
//     co-fill is a hidden SPSC violation — resolution #3 — and is TT_FATAL-rejected here and deferred)
//   - 1D depthwise conv (height-sharded)
//
// Mirrors the already-ported width-sharded factory (conv2d_op_width_sharded_program_factory.cpp):
// get_cb_info() for sizing, DFBs from CBInfo, op_owned_tensors for the indices tensor,
// SemaphoreSpec/SemaphoreBinding for the act- and weights-mcast semaphores, mcast NoC rects -> named
// CTAs/RTAs, per-node X/Y NoC lookup tables -> runtime varargs.
//
// CB resolutions applied:
//   #1 OUT       -> borrowed_from = OUTPUT, compute packer is PRODUCER, the degenerate CONSUMER is bound
//                   to the DM output kernel (the writer_tiled_out / reader_writer_tiled_out kernel, which
//                   exists here unlike width-sharded).
//   #2 PARTIALS  -> self-loop DFB borrowed_from = OUTPUT when partials_cb_uses_output (in-place accumulate).
//   #3 ACT split_reader_cb_shared -> TT_FATAL-rejected (deferred); single-DM-fill ACT only.

#include <cstdint>
#include <cstdlib>  // std::getenv (Option C split-tilize test toggle)
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/experimental/quasar/conv2d/device/conv2d_op_sharded_program_factory.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include "ttnn/operations/compute_throttle_utils.hpp"

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
namespace unary = ttnn::operations::unary;
using ttnn::operations::conv::conv_skip_mcast;
using ttnn::operations::conv::is_1d_depthwise_conv;
using ttnn::operations::conv::should_coalesce_1d_depthwise_conv_reads;
using ttnn::operations::conv::SkipMcast;
using ttnn::prim::CBInfo;
using ttnn::prim::Conv2dCb;
using ttnn::prim::get_cb_info;
using ttnn::prim::get_cb_info_by_name;

// Compute kernel addressing mode divides addresses with 16
constexpr uint32_t COMPUTE_KERNEL_ADDRESS_DIVISOR = 16;

struct ActivationReuseConfig {
    uint32_t image_width_tiles = 0;
    uint32_t image_width_mod_tile = 0;
    uint32_t act_cb_num_tiles_split = 0;
    uint32_t act_cb_num_tiles_split_last = 0;
    uint32_t reuse_window_offset = 0;
    bool readers_process_full_image_widths = false;
    uint32_t tilized_cb_row_offset = 0;
    uint32_t tilized_cb_second_reader_offset = 0;
    // Configuration needed to handle cores with non-meaningful work
    uint32_t num_cores_with_non_meaningful_work = 0;
    std::set<CoreCoord> cores_with_non_meaningful_work;
    bool has_partial_core = false;
    CoreCoord partial_work_core{0, 0};
    uint32_t partial_core_reader_tiles_to_push = 0;
    uint32_t partial_core_writer_remaining_tiles_to_push_to_push = 0;
    bool single_core_processes_multiple_batches = false;
};

ActivationReuseConfig calculate_activation_reuse_params(
    uint32_t output_image_height,
    uint32_t output_image_width,
    uint32_t filter_w,
    uint32_t filter_h,
    uint32_t conv_act_c_read_bytes,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t act_block_h_nsubblocks_split,
    uint32_t act_block_h_nsubblocks_split_last,
    uint32_t tilized_act_tile_size,
    uint32_t act_block_w_ntiles,
    uint32_t act_block_h_ntiles,
    uint32_t single_core_height_ntiles,
    uint32_t total_output_height_ntiles,
    uint32_t padded_total_output_height_ntiles,
    const std::vector<CBInfo>& cb_info,
    bool enable_split_reader,
    const CoreRangeSet& input_cores,
    uint32_t batch) {
    ActivationReuseConfig config;

    // Calculate compile time args needed for activation reuse feature
    config.image_width_tiles = tt::div_up(output_image_width, tt::constants::TILE_HEIGHT);
    config.image_width_mod_tile = output_image_width % tt::constants::TILE_HEIGHT;
    const uint32_t image_width_tile_leftover =
        config.image_width_mod_tile == 0 ? 0 : tt::constants::TILE_HEIGHT - config.image_width_mod_tile;

    // We rely that double buffering is turned off here
    // TODO(sjovic): avoid this assumption
    config.act_cb_num_tiles_split = get_cb_info_by_name(cb_info, Conv2dCb::ACT).num_pages;
    if (enable_split_reader) {
        config.act_cb_num_tiles_split_last = get_cb_info_by_name(cb_info, Conv2dCb::ACT_SECOND_READER).num_pages;
    }

    // Number of bytes to move the CB read pointer when passing on to the new output image row;
    // We need to skip the first kernel_w*in_channels_padded elements
    config.reuse_window_offset = filter_w * conv_act_c_read_bytes;
    // In case the output image width is not a multiple of the tile height, we need to skip the additional elements
    // we read to fill in the tile height
    if (image_width_tile_leftover) {
        config.reuse_window_offset +=
            (filter_w * filter_h * conv_act_c_read_bytes + act_block_w_extra_align_bytes) * image_width_tile_leftover;
    }

    // Precompute happy path for the feature - if each reader processes full image rows only, we can skip many if
    // conditions in the kernel. There are two cases which can affect this:
    // - shards are split in such way that one shard ends in the middle of the image width
    // - shards contain full image widths only, but split reader splits shard in the middle of the image width
    // - output image width is not a multiple of the tile height, so we need to process more than one image width at
    // once
    config.readers_process_full_image_widths = act_block_h_nsubblocks_split % config.image_width_tiles == 0 &&
                                               act_block_h_nsubblocks_split_last % config.image_width_tiles == 0 &&
                                               image_width_tile_leftover == 0;

    // Compute kernel interleaves tilizing data coming from two readers so it needs to calculate the address in the
    // tilized CB
    config.tilized_cb_row_offset = tilized_act_tile_size * act_block_w_ntiles;
    config.tilized_cb_second_reader_offset = tilized_act_tile_size * act_block_h_nsubblocks_split * act_block_w_ntiles;

    // Last cores sometime have less work to do, but we still need to push the same number of tiles
    // to avoid blocking compute kernels; Here we compute how many cores will be pushing the remaining tiles
    uint32_t total_remaining_tiles_to_push = padded_total_output_height_ntiles - total_output_height_ntiles;

    config.num_cores_with_non_meaningful_work = tt::div_up(total_remaining_tiles_to_push, single_core_height_ntiles);

    std::vector<CoreCoord> all_input_cores;
    for (const CoreRange& range : input_cores.ranges()) {
        for (const CoreCoord& core : range) {
            all_input_cores.push_back(core);
        }
    }

    // Calculate tiles for the partial core (the one core that may have less than full work)
    uint32_t partial_core_remaining_tiles = total_remaining_tiles_to_push % single_core_height_ntiles;
    config.has_partial_core = partial_core_remaining_tiles > 0;
    config.partial_core_reader_tiles_to_push = partial_core_remaining_tiles;

    if (partial_core_remaining_tiles > 0) {
        if (enable_split_reader) {
            uint32_t partial_core_act_blocks_to_push = partial_core_remaining_tiles / act_block_h_ntiles;
            config.partial_core_reader_tiles_to_push = partial_core_act_blocks_to_push * act_block_h_nsubblocks_split;
            config.partial_core_writer_remaining_tiles_to_push_to_push =
                partial_core_act_blocks_to_push * act_block_h_nsubblocks_split_last;

            uint32_t partial_core_leftover_tiles = partial_core_remaining_tiles % act_block_h_ntiles;
            if (partial_core_leftover_tiles > act_block_h_nsubblocks_split_last) {
                config.partial_core_writer_remaining_tiles_to_push_to_push += act_block_h_nsubblocks_split_last;
                config.partial_core_reader_tiles_to_push +=
                    partial_core_leftover_tiles - act_block_h_nsubblocks_split_last;
            } else {
                config.partial_core_writer_remaining_tiles_to_push_to_push += partial_core_leftover_tiles;
            }
        }
        uint32_t partial_core_idx = all_input_cores.size() - config.num_cores_with_non_meaningful_work;
        config.partial_work_core = all_input_cores[partial_core_idx];
    }

    // Put all cores with non-meaningful work to the set
    uint32_t start_idx = all_input_cores.size() - config.num_cores_with_non_meaningful_work;
    for (uint32_t i = start_idx; i < all_input_cores.size(); i++) {
        config.cores_with_non_meaningful_work.insert(all_input_cores[i]);
    }

    // If we have cores processing data from more than 1 batch, we need to trigger a check inside the kernel
    // to 'restart' the optimization and fill in the whole output image width
    // before the reuse for the new batch starts
    const uint32_t per_core_out_hw = single_core_height_ntiles * tt::constants::TILE_HEIGHT;
    const uint32_t total_batch_out_hw = output_image_height * output_image_width;
    config.single_core_processes_multiple_batches = (batch > 1) && (total_batch_out_hw % per_core_out_hw != 0);

    return config;
}

namespace {

// ---- Metal 2.0 resource names (ProgramSpec scope) ----
// DFB accessor names surface kernel-side as dfb::<name> tokens; the ported sharded kernels reference
// these exact spellings.
const m2::DFBSpecName DFB_ACT{"act"};                              // mcast result / direct act (reader -> compute)
const m2::DFBSpecName DFB_ACT_SECOND_READER{"act_second_reader"};  // split-reader second half (writer -> compute)
const m2::DFBSpecName DFB_ACT_RM{"act_row_major"};                 // 2D reader -> compute (tilize input)
const m2::DFBSpecName DFB_ACT_TILIZED{"act_tilized"};          // compute (tilize) -> 2D reader (mcast src) / depthwise
const m2::DFBSpecName DFB_WEIGHTS{"weights"};                  // weights reader/writer -> compute
const m2::DFBSpecName DFB_BIAS{"bias"};                        // weights writer -> compute (optional)
const m2::DFBSpecName DFB_MATMUL_PARTIALS{"matmul_partials"};  // compute self-loop (borrows OUTPUT when aliased)
const m2::DFBSpecName DFB_OUT{"out"};                          // compute packer -> OUTPUT (degenerate DM consumer)
const m2::DFBSpecName DFB_READER_INDICES{"reader_indices"};    // fresh L1 DMA landing (DRAM-config path only)
const m2::DFBSpecName DFB_IN_SCALAR{"in_scalar"};              // reduce scalar srcB (unpack-tilize probe only)

const m2::TensorParamName TP_INPUT{"input"};
const m2::TensorParamName TP_OUTPUT{"output"};
const m2::TensorParamName TP_WEIGHTS{"weights"};
const m2::TensorParamName TP_BIAS{"bias"};
const m2::TensorParamName TP_READER_INDICES{"reader_indices"};

const m2::SemaphoreSpecName SEM_ACT_MCAST_SENDER{"act_mcast_sender"};
const m2::SemaphoreSpecName SEM_ACT_MCAST_RECEIVER{"act_mcast_receiver"};
const m2::SemaphoreSpecName SEM_WEIGHTS_MCAST_SENDER{"weights_mcast_sender"};
const m2::SemaphoreSpecName SEM_WEIGHTS_MCAST_RECEIVER{"weights_mcast_receiver"};

const m2::KernelSpecName KERNEL_READER{"reader"};
const m2::KernelSpecName KERNEL_WRITER_SENDER{"writer_mcast_sender"};
const m2::KernelSpecName KERNEL_WRITER_RECEIVER{"writer_mcast_receiver"};
const m2::KernelSpecName KERNEL_COMPUTE{"compute"};

}  // namespace

ttnn::device_operation::ProgramArtifacts Conv2dShardedProgramFactory::create_program_artifacts(
    const Conv2dParams& operation_attributes, const Conv2dInputs& tensor_args, Tensor& output_tensor) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    const auto& ashape = ttnn::Shape(operation_attributes.input_tensor_shape);
    const auto& bias = tensor_args.bias;
    const auto& sliding_window_config = operation_attributes.sliding_window_config;

    ttnn::operations::sliding_window::ParallelConfig parallel_config{
        .grid = a.shard_spec().value().grid,
        .shard_scheme = a.memory_config().memory_layout(),
        .shard_orientation = a.shard_spec().value().orientation};

    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config);

    const auto output_channels = operation_attributes.output_channels;
    const auto groups = operation_attributes.groups;
    const auto untilize_out = operation_attributes.untilize_out;
    const auto has_bias = operation_attributes.has_bias;
    const auto& fused_activation = operation_attributes.activation;
    const auto& parallelization_config = operation_attributes.parallelization_config;
    const auto& block_config = operation_attributes.block_config;
    const auto transpose_mcast = a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR;
    auto& output = output_tensor;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    const auto enable_act_double_buffer = operation_attributes.enable_act_double_buffer;
    const auto enable_weights_double_buffer = operation_attributes.enable_weights_double_buffer;
    const auto full_inner_dim = operation_attributes.full_inner_dim;
    // enable_activation_reuse requires the split reader (deferred in this Metal 2.0 port). It is a
    // read optimization (reuse of overlapping halo rows across blocks) — the non-reuse path produces
    // the same conv result, just with more activation reads. Downgrade the request to the non-reuse
    // path instead of aborting so callers that pass activation_reuse=1 (e.g. resnet50) run correctly.
    if (operation_attributes.enable_activation_reuse) {
        log_warning(
            tt::LogOp,
            "conv2d Metal 2.0 sharded factory: enable_activation_reuse is not yet supported (it needs the "
            "split reader); falling back to the non-reuse path (correct, slightly slower).");
    }
    const auto enable_activation_reuse = false;
    const auto config_tensors_in_dram = operation_attributes.config_tensors_in_dram;
    const auto& force_split_reader = operation_attributes.force_split_reader;

    distributed::MeshDevice* device = a.device();
    TT_FATAL(a.layout() == Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    TT_FATAL(a.memory_config().is_sharded(), "Conv activation must be sharded.");
    TT_FATAL(output_channels <= b.padded_shape()[3], "Invalid weight shape. Incorrect weight tensor.");
    const uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    const uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    const uint32_t weight_block_w_ntiles = parallelization_config.per_core_out_matrix_width_ntile;
    const uint32_t out_block_h_ntiles = parallelization_config.per_core_out_matrix_height_ntile;
    // WORKAROUND (Quasar), from sjovic/quasar-resnet 057e1792f0a: force out_subblock to 1x1. The Quasar
    // compute dest-sync (MATH_PACK / SrcA handshake) in conv_bmm_tilize_metal2 deadlocks the three compute
    // threads whenever the matmul_partials spill/reload handles more than one tile per subblock
    // (out_subblock_num_tiles > 1). Constraining to 1x1 makes partials flow one tile at a time so the
    // compute pipeline drains. Verified there: the resnet stem conv passes single-core AND full 32-core
    // (was hanging). Applies to every conv this factory builds (stem + bottleneck 3x3), so it also covers
    // the L1-path convs a DRAM slice_config cannot reach. Gated to Quasar; WH/BH keep the tuned subblock.
    // Remove once the LLK dest-sync limitation is fixed (tt-metal #48679 / tt-llk #48504).
    const bool arch_is_quasar = device->arch() == tt::ARCH::QUASAR;
    const uint32_t out_subblock_h_ntiles = arch_is_quasar ? 1 : block_config.out_subblock_h_ntiles;
    const uint32_t out_subblock_w_ntiles = arch_is_quasar ? 1 : block_config.out_subblock_w_ntiles;

    const SkipMcast skip_mcast = conv_skip_mcast(parallelization_config, a.memory_config().memory_layout());
    const bool skip_activation_mcast = skip_mcast.skip_activation_mcast;
    const bool skip_weights_mcast = skip_mcast.skip_weights_mcast;

    const tt::DataFormat tilized_act_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    TT_FATAL(
        out_block_h_ntiles >= act_block_h_ntiles,
        "Output block height (in # of tiles) ({}) should be greater than or equal to activation block height (in # of "
        "tiles) ({})",
        out_block_h_ntiles,
        act_block_h_ntiles);

    // Tensor b has weights and it should be tiled layout after converting conv weights into weight matrix
    TT_FATAL(b.layout() == Layout::TILE, "Conv weights should be in tiled layout");
    TT_FATAL(b.padded_shape()[0] == 1, "Conv weight matrix shape is invalid");
    TT_FATAL(b.padded_shape()[1] == 1, "Conv weight matrix shape is invalid");
    uint32_t weight_matrix_height = b.padded_shape()[2];
    uint32_t weight_matrix_width = b.padded_shape()[3];
    uint32_t weight_matrix_width_ntiles = weight_matrix_width / tt::constants::TILE_WIDTH;

    const std::array<uint32_t, 2> shard_shape = a.shard_spec().value().shape;

    const bool block_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool height_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;

    // [#47797] Force full DEST sync on the block-sharded 2D-mcast path. With half-sync DEST the BH MATH
    // coprocessor overlaps and backs up at the height-block boundary (h's matmul/output DEST not drained
    // before h+1's tilize-init), deadlocking the cross-core tilize->mcast->matmul loop from the second
    // height-block on (works on WH, hangs on BH). Full sync drains DEST per op, preventing the backup.
    // (This also selects the standard tilize, since can_use_fast_tilize() requires !dst_full_sync.)
    // Numerically identical; localized perf cost on this conv only.
    // Also force it on height-sharded to route WH/BH off the racy fast_tilize path onto regular tilize_block
    // (the fixed-latency PACK nop race-guard could not mask the fast_tilize dest handshake race).
    //
    // EXPERIMENT (Quasar height-sharded): do NOT force full DEST sync here. On Quasar the height-sharded
    // conv still 0x19s (Risc IB interrupt on MATH) in the per-tile tilize datacopy<->pack DEST[0] reuse.
    // Full sync uses a single DEST section, so tile t+1's datacopy MOP collides with tile t's pack drain;
    // half sync double-buffers DEST across two sections, which can hide that reuse. Full sync was only
    // forced on height-sharded to route WH/BH off fast_tilize -- Quasar has no fast_tilize (can_use_fast
    // is already false), so the force serves no purpose on Quasar and may be causing the race. Let Quasar
    // height-sharded fall back to half sync. WH/BH and block-sharded keep full sync.
    if (block_sharded || (height_sharded && !arch_is_quasar)) {
        dst_full_sync_en = true;
    }

    // parallelization config
    CoreRangeSet input_cores = a.memory_config().shard_spec().value().grid;
    CoreRangeSet output_cores = output.memory_config().shard_spec().value().grid;
    TT_ASSERT(
        input_cores == output_cores || block_sharded,
        "For height sharded convs input and output cores must be the same");

    // Anchor the compute / kernel-placement rectangle on the INPUT shard grid's bounding box rather than a
    // (0,0)-origin grid_size rectangle.  For height-sharded, input_cores == output_cores and their bbox
    // equals (grid_size.x, grid_size.y) whenever the grid starts at (0,0) — so this is byte-identical on
    // the default grid.  On an offset grid (conv_config.core_grid placed away from (0,0), e.g. a 2-core
    // sub-grid at logical y=1) it keeps the kernels on the tensor's actual cores instead of requesting
    // non-existent logical coords (e.g. (1,1) on a 2x1 grid).
    CoreRangeSet all_cores;
    if (height_sharded) {
        all_cores = CoreRangeSet(input_cores.bounding_box());
    } else {
        all_cores = input_cores.merge(output_cores);
    }

    const uint32_t num_cores_x = parallelization_config.grid_size.x;
    const uint32_t num_cores_y = parallelization_config.grid_size.y;
    const uint32_t total_num_cores = all_cores.num_cores();

    const uint32_t per_core_out_matrix_width_ntiles = parallelization_config.per_core_out_matrix_width_ntile;
    const uint32_t per_core_out_matrix_height_ntiles = parallelization_config.per_core_out_matrix_height_ntile;

    // full_inner_dim => keep the whole reduction dim in one K-block (num_blocks_act_w = 1, no matmul-partials
    // accumulate). For height-sharded this is the Quasar fit-guarded "no-spill" path (set upstream in conv2d.cpp
    // for small-K convs whose full window fits L1); block-sharded already used it. When full_inner_dim is false we
    // slice by kernel row and spill as before.
    const bool slice_inner_dim =
        (height_sharded && !enable_activation_reuse && !full_inner_dim) || (block_sharded && !full_inner_dim);

    uint32_t conv_act_c_blocks = 1;
    uint32_t out_conv_c_blocks = 1;
    uint32_t input_channels_padded = shard_shape[1];
    if (block_sharded) {
        TT_ASSERT(input_cores.size() == 1, "Block sharded convs should have only one input core range!");
        const uint32_t in_num_cores_x = input_cores.bounding_box().end_coord.x + 1;
        const uint32_t in_num_cores_y = input_cores.bounding_box().end_coord.y + 1;

        conv_act_c_blocks =
            a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR ? in_num_cores_x : in_num_cores_y;
        out_conv_c_blocks =
            output.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR ? num_cores_x : num_cores_y;
        if (transpose_mcast) {
            TT_FATAL(conv_act_c_blocks == in_num_cores_y, "Expected conv_act_c_blocks to be equal to height of grid");
            input_channels_padded = shard_shape[1] * in_num_cores_y;
        } else {
            TT_FATAL(conv_act_c_blocks == in_num_cores_x, "Expected conv_act_c_blocks to be equal to width of grid");
            input_channels_padded = shard_shape[1] * in_num_cores_x;
        }
    }

    const ttnn::Shape ashape_with_channels_padded({ashape[0], ashape[1], ashape[2], input_channels_padded});
    uint32_t conv_act_size_w = ashape_with_channels_padded[2];
    const uint32_t conv_act_size_c = ashape_with_channels_padded[3];
    const uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;   // filter_h
    const uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;  // filter_W
    uint32_t pad_w = (uint32_t)sliding_window_config.get_pad_w();
    const uint32_t dilation_h = (uint32_t)sliding_window_config.dilation_hw.first;
    const uint32_t dilation_w = (uint32_t)sliding_window_config.dilation_hw.second;
    const uint32_t stride_w = sliding_window_config.is_transpose ? 1 : (uint32_t)sliding_window_config.stride_hw.second;

    if (sliding_window_config.is_transpose) {
        auto input_shape = sliding_window_config.get_transposed_full_input_shape();
        conv_act_size_w = input_shape[2];
        pad_w = 0;
    }

    const bool is_conv_1d_depthwise_conv =
        is_1d_depthwise_conv(groups, ashape[3], output_channels, filter_h, ashape[1], has_bias);
    const uint32_t conv_act_c_read_bytes = conv_act_size_c * a.element_size() / conv_act_c_blocks;
    const bool coalesce_1d_depthwise_kw_reads = should_coalesce_1d_depthwise_conv_reads(
        is_conv_1d_depthwise_conv,
        a.memory_config().memory_layout(),
        input_channels_padded,
        filter_w,
        dilation_w,
        a.dtype());
    TT_FATAL(
        !coalesce_1d_depthwise_kw_reads || filter_h == 1,
        "Coalesced 1D depthwise reads require filter_h == 1, got {}",
        filter_h);

    // ---- Split reader: deferred in the Metal 2.0 sharded port (single-DM-fill ACT only) ----
    // The legacy split reader has the WRITER kernel co-read the second half of the activation block,
    // sharing the borrowed ACT_SHARDED (input shard) and READER_INDICES address-source CBs with the
    // READER kernel on the SAME node.  Under Quasar's per-node SPSC DFB invariant that is two
    // producer/consumer instances of those borrowed DFBs on one node — structurally unsupported.
    // Per resolution #3 the sanctioned fallback is single-DM-fill: the reader fills the entire ACT block
    // and the writer does weights/bias only.  We therefore force split_reader off here.  When the caller
    // explicitly forces it on we reject with a clear message (deferred optimization).
    const bool split_reader_would_enable =
        is_split_reader_supported(a.memory_config().memory_layout(), is_conv_1d_depthwise_conv, act_block_h_ntiles) &&
        force_split_reader.value_or(is_split_reader_viable(
            a.memory_config().memory_layout(),
            act_block_h_ntiles,
            input_channels_padded,
            filter_w,
            tt::tt_metal::hal::get_arch(),
            a.dtype(),
            parallelization_config.per_core_out_matrix_width_ntile * block_config.act_block_w_ntiles,
            tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(b.dtype())),
            dilation_w,
            per_core_out_matrix_height_ntiles / block_config.act_block_h_ntiles,
            act_block_w_ntiles,
            fp32_dest_acc_en,
            output.dtype(),
            enable_activation_reuse));
    TT_FATAL(
        !force_split_reader.value_or(false) || !split_reader_would_enable,
        "conv2d Metal 2.0 sharded factory: split reader is deferred (single-DM-fill ACT only). The legacy "
        "split reader shares the borrowed ACT_SHARDED / READER_INDICES address-source CBs between the reader "
        "and writer kernels on the same node, which violates the Quasar per-node SPSC DFB invariant. Disable "
        "force_split_reader (and enable_activation_reuse, which requires it).");
    // Force single-DM-fill: the reader fills the whole ACT block; the writer does weights/bias only.
    const bool enable_split_reader = false;
    (void)split_reader_would_enable;
    log_debug(
        tt::LogOp,
        "force_split_reader: {}, enable_split_reader (forced single-DM-fill): {}, num_blocks_act_h: {}, "
        "per_core_out_matrix_height_ntiles: {}, act_block_h_ntiles: {}",
        force_split_reader,
        enable_split_reader,
        per_core_out_matrix_height_ntiles / block_config.act_block_h_ntiles,
        per_core_out_matrix_height_ntiles,
        block_config.act_block_h_ntiles);

    // Activation reuse depends on split reader, which is deferred above — reject it here.
    TT_FATAL(
        !enable_activation_reuse,
        "conv2d Metal 2.0 sharded factory: enable_activation_reuse is deferred (it requires the split reader, "
        "which is not supported in this port — see the split-reader note above).");

    TT_FATAL(input_channels_padded >= ashape[3], "Incorrect padding of input channels!");
    if (is_conv_1d_depthwise_conv && height_sharded) {
        const uint32_t expected_act_block_w_ntiles =
            tt::round_up(
                input_channels_padded * (coalesce_1d_depthwise_kw_reads ? filter_w : 1), tt::constants::TILE_WIDTH) /
            tt::constants::TILE_WIDTH;
        TT_FATAL(
            act_block_w_ntiles == expected_act_block_w_ntiles,
            "1D depthwise activation block width mismatch. Got {} tiles, expected {} tiles",
            act_block_w_ntiles,
            expected_act_block_w_ntiles);
    }
    // check is for 16-byte alignment
    TT_FATAL(
        // Since fp16 is smalleset data format used for halo output, 8 input_channels is enough for 16 byte alignment
        input_channels_padded % 8 == 0,
        "Expected input channels to be padded for 16 byte alignment in L1 ({} % 16 != 0)",
        input_channels_padded);

    const uint32_t act_matrix_height_ntiles = out_block_h_ntiles * parallelization_config.num_cores_nhw;
    const uint32_t act_matrix_height = act_matrix_height_ntiles * tt::constants::TILE_HEIGHT;

    if (has_bias) {
        if (is_conv_1d_depthwise_conv) {
            TT_THROW("Bias is not supported for depthwise conv1d");
        }
        // Tensor bias is of shape {output_channels}
        TT_FATAL(bias.has_value(), "Bias tensor must be provided when has_bias is true");
        TT_FATAL(bias.value().buffer() != nullptr, "Bias tensor buffer must not be null");
        auto bias_shape_without_padding = bias.value().logical_shape();
        TT_FATAL(bias_shape_without_padding[0] == 1, "Bias should have batch == 1");
    }

    // Tile size divisibility checks
    TT_FATAL(
        weight_matrix_height % tt::constants::TILE_HEIGHT == 0, "Height of weight matrix needs to be divisible by 32");
    TT_FATAL(
        weight_matrix_width % tt::constants::TILE_WIDTH == 0, "Width of weight matrix needs to be divisible by 32");

    // Device compatibility checks
    TT_FATAL(
        a.storage_type() == StorageType::DEVICE && b.storage_type() == StorageType::DEVICE,
        "Operands to large matmul need to be on device!");
    TT_FATAL(a.device() == b.device(), "Operands to conv need to be on the same device!");
    TT_FATAL(
        a.buffer() != nullptr && b.buffer() != nullptr, "Operands to conv need to be allocated in buffers on device!");
    if (has_bias) {
        TT_FATAL(bias.value().storage_type() == StorageType::DEVICE, "Bias should be on device");
        TT_FATAL(bias.value().device() == a.device(), "Bias should be on the same device as act tensor");
    }

    TT_FATAL(
        act_matrix_height_ntiles % act_block_h_ntiles == 0,
        "act_matrix_height_ntiles {} should be divisible by act_block_h_ntiles {}",
        act_matrix_height_ntiles,
        act_block_h_ntiles);
    TT_FATAL(
        weight_matrix_width_ntiles % weight_block_w_ntiles == 0,
        "weight_matrix_width_ntiles {} should be divisible by weight_block_w_ntiles {}",
        weight_matrix_width_ntiles,
        weight_block_w_ntiles);
    TT_FATAL(
        act_matrix_height_ntiles % out_block_h_ntiles == 0,
        "act_matrix_height_ntiles {} should be divisible by out_block_h_ntiles {}",
        act_matrix_height_ntiles,
        out_block_h_ntiles);

    const uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    const uint32_t num_blocks_act_w = is_conv_1d_depthwise_conv
                                          ? (coalesce_1d_depthwise_kw_reads ? 1 : filter_h * filter_w)
                                          : (slice_inner_dim ? filter_h : 1);
    const uint32_t num_blocks_weight_w = weight_matrix_width_ntiles / weight_block_w_ntiles;

    // act block info
    uint32_t act_block_h_datums = act_matrix_height / num_blocks_act_h;

    uint32_t act_block_h_nsubblocks_split = block_config.act_block_h_ntiles;
    uint32_t act_block_h_nsubblocks_split_last = 0;
    if (enable_split_reader) {
        act_block_h_nsubblocks_split_last = block_config.act_block_h_ntiles / 2;
        act_block_h_nsubblocks_split = block_config.act_block_h_ntiles - act_block_h_nsubblocks_split_last;
    }
    uint32_t act_block_h_datums_split = act_block_h_nsubblocks_split * tt::constants::TILE_HEIGHT;

    uint32_t act_block_num_tiles_split = act_block_h_nsubblocks_split * act_block_w_ntiles;
    uint32_t act_block_num_tiles_split_last = act_block_h_nsubblocks_split_last * act_block_w_ntiles;

    // weight block info
    uint32_t weight_block_w_datums = weight_matrix_width / num_blocks_weight_w;
    TT_FATAL(
        weight_block_w_ntiles % out_subblock_w_ntiles == 0,
        "weight_block_w_ntiles {} should be divisible by weight_block_w_ntiles {}",
        weight_block_w_ntiles,
        out_subblock_w_ntiles);
    uint32_t weight_num_subblocks = weight_block_w_ntiles / out_subblock_w_ntiles;
    uint32_t weight_block_h_ntiles = is_conv_1d_depthwise_conv
                                         ? act_block_h_ntiles * (coalesce_1d_depthwise_kw_reads ? filter_w : 1)
                                         : act_block_w_ntiles;
    uint32_t weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles;

    // writer of conv op partially removes padding on the width
    // it removes the padding done for block width but it doesn't remove padding done for tiled width
    uint32_t output_channels_padded_to_tile_width = tt::round_up(output_channels, tt::constants::TILE_WIDTH);
    TT_FATAL(
        output_channels_padded_to_tile_width <= weight_matrix_width,
        "output_channels_padded_to_tile_width {} should be less than or equal to weight_matrix_width {}",
        output_channels_padded_to_tile_width,
        weight_matrix_width);
    uint32_t last_block_width_datums = (output_channels_padded_to_tile_width % weight_block_w_datums == 0)
                                           ? weight_block_w_datums
                                           : (output_channels_padded_to_tile_width % weight_block_w_datums);
    TT_FATAL(
        last_block_width_datums % tt::constants::TILE_WIDTH == 0,
        "last_block_width_datums {} should be divisible by TILE_WIDTH",
        last_block_width_datums);

    TT_FATAL(output.is_sharded(), "Output buffer must be sharded!");

    // out
    uint32_t out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;
    TT_FATAL(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    TT_FATAL(
        act_block_h_ntiles % out_subblock_h_ntiles == 0,
        "act_block_h_ntiles {} should be divisible by out_subblock_h_ntiles {}",
        act_block_h_ntiles,
        out_subblock_h_ntiles);

    const uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    const uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;
    const uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    const uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    const uint32_t in0_num_blocks_w = conv_act_c_blocks * num_blocks_act_w;

    // bias
    uint32_t bias_ntiles = 0;
    if (has_bias) {
        bias_ntiles =
            bias.value().padded_shape()[3] / tt::constants::TILE_WIDTH;  // TODO: support non tile multiple sizes
    }

    const uint32_t window_outer = num_blocks_act_w;
    const uint32_t window_inner = block_sharded ? filter_h : filter_h * filter_w / num_blocks_act_w;
    log_debug(tt::LogOp, "window_outer: {}, window_inner: {}", window_outer, window_inner);

    TT_FATAL(
        weight_matrix_width_ntiles % per_core_out_matrix_width_ntiles == 0,
        "weight_matrix_width_ntiles {} should be divisible by per_core_out_matrix_width_ntiles {}",
        weight_matrix_width_ntiles,
        per_core_out_matrix_width_ntiles);
    TT_FATAL(
        per_core_out_matrix_width_ntiles % weight_block_w_ntiles == 0,
        "per_core_out_matrix_width_ntiles {} should be divisible by weight_block_w_ntiles {}",
        per_core_out_matrix_width_ntiles,
        weight_block_w_ntiles);
    uint32_t num_blocks_weight_w_per_core = per_core_out_matrix_width_ntiles / weight_block_w_ntiles;
    if (height_sharded) {
        TT_FATAL(
            num_blocks_weight_w_per_core == num_blocks_weight_w,
            "num_blocks_weight_w_per_core {} should be equal to num_blocks_weight_w {}",
            num_blocks_weight_w_per_core,
            num_blocks_weight_w);
    }
    uint32_t num_weight_slices_width = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
    uint32_t total_num_cores_per_weight_slice = 0;
    if (block_sharded) {
        if (transpose_mcast) {
            TT_FATAL(
                num_cores_y % num_weight_slices_width == 0,
                "num_cores_y {} should be divisible by num_weight_slices_width {}",
                num_cores_y,
                num_weight_slices_width);
            uint32_t num_cores_y_per_weight_slice_width = num_cores_y / num_weight_slices_width;
            total_num_cores_per_weight_slice = num_cores_y_per_weight_slice_width * num_cores_x;
        } else {
            TT_FATAL(
                num_cores_x % num_weight_slices_width == 0,
                "num_cores_x {} should be divisible by num_weight_slices_width {}",
                num_cores_x,
                num_weight_slices_width);
            uint32_t num_cores_x_per_weight_slice_width = num_cores_x / num_weight_slices_width;
            total_num_cores_per_weight_slice = num_cores_x_per_weight_slice_width * num_cores_y;
        }
        TT_FATAL(
            total_num_cores_per_weight_slice * per_core_out_matrix_height_ntiles == act_matrix_height_ntiles,
            "total_num_cores_per_weight_slice {} * per_core_out_matrix_height_ntiles {} should be equal to "
            "act_matrix_height_ntiles {}",
            total_num_cores_per_weight_slice,
            per_core_out_matrix_height_ntiles,
            act_matrix_height_ntiles);
    } else {
        TT_FATAL(
            num_cores_y % num_weight_slices_width == 0,
            "num_cores_y {} should be divisible by num_weight_slices_width {}",
            num_cores_y,
            num_weight_slices_width);
        uint32_t num_cores_y_per_weight_slice_width = num_cores_y / num_weight_slices_width;
        total_num_cores_per_weight_slice = num_cores_y_per_weight_slice_width * num_cores_x;
        TT_FATAL(
            total_num_cores * per_core_out_matrix_height_ntiles >= act_matrix_height_ntiles,
            "total_num_cores {} * per_core_out_matrix_height_ntiles {} should be greater than or equal to "
            "act_matrix_height_ntiles {}",
            total_num_cores,
            per_core_out_matrix_height_ntiles,
            act_matrix_height_ntiles);
    }
    TT_FATAL(
        per_core_out_matrix_height_ntiles % act_block_h_ntiles == 0,
        "per_core_out_matrix_height_ntiles {} should be divisible by act_block_h_ntiles {}",
        per_core_out_matrix_height_ntiles,
        act_block_h_ntiles);
    uint32_t num_blocks_act_h_per_core = per_core_out_matrix_height_ntiles / act_block_h_ntiles;

    // OPTION C (split tilize/matmul) test toggle. When TT_METAL_QSR_CONV_SPLIT_TILIZE is set, select the
    // conv_bmm_split_tilize_metal2.cpp compute kernel, which tilizes ALL height blocks first (one contiguous
    // tilize phase) then matmuls them, so the compute engine transitions tilize->matmul only once (diagnostic
    // for the Quasar per-block tilize<->matmul DEST-handshake 0x19 race). Gated to the height-sharded,
    // single-K-block (in0_num_blocks_w == 1), single-output-width-block (num_blocks_weight_w_per_core == 1),
    // no-split-reader / no-activation-reuse / non-depthwise path -- the resnet stem / 1x1 conv shape the split
    // kernel implements. Requires act_tilized to hold all height blocks at once (resized below). Everything
    // else falls back to the fused kernel even when the env is set.
    const bool split_tilize_matmul = (std::getenv("TT_METAL_QSR_CONV_SPLIT_TILIZE") != nullptr) && height_sharded &&
                                     !is_conv_1d_depthwise_conv && !enable_split_reader && !enable_activation_reuse &&
                                     (in0_num_blocks_w == 1) && (num_blocks_weight_w_per_core == 1);

    // OPTION B — PROGRAM A (tilize-only, standalone). When TT_METAL_QSR_CONV_SPLIT_PROGRAM is set, this conv
    // op runs ONLY the gather+tilize half in a fresh tilize-oriented Metal program (conv_tilize_only_metal2.cpp)
    // and OUTPUTS the tilized activations — no matmul, no weights reader, no output writer. This isolates the
    // tilize into its own program (its own tilize-oriented compute_kernel_hw_startup, no preceding matmul) to
    // confirm the Quasar 0x19 (stale MATH DEST-dvalid left by the matmul rejecting the tilize datacopy) clears.
    // Program B (the matmul over these tilized activations) is chained separately at the host level in a later
    // increment. Same eligibility as Option C: the resnet stem / 1x1 height-sharded single-K-block shape.
    // OPTION B — UNPACK-TILIZE PROBE (TT_METAL_QSR_CONV_UNPACK_TILIZE). Runs the conv's gathered activation
    // through the MAXPOOL-style unpack-tilize path (unpack_tilizeA_B_block + reduce) instead of tilize_block,
    // to localize the intrinsic Quasar 0x19 to tilize_block vs unpack-tilize. Shares ALL the standalone
    // Program-A plumbing below (folded into split_program_tilize_only); differs only in the compute kernel,
    // an extra reduce-scalar (srcB) DFB, and REDUCE_OP/REDUCE_DIM compute defines.
    const bool split_program_unpack_tilize = (std::getenv("TT_METAL_QSR_CONV_UNPACK_TILIZE") != nullptr) &&
                                             height_sharded && !is_conv_1d_depthwise_conv && !enable_split_reader &&
                                             !enable_activation_reuse && (in0_num_blocks_w == 1) &&
                                             (num_blocks_weight_w_per_core == 1);

    // "Standalone Program A" plumbing flag — TRUE for BOTH the tilize-only kernel (TT_METAL_QSR_CONV_SPLIT_PROGRAM)
    // AND the unpack-tilize probe (above): reader gather + tilize into a PLAIN ACT_TILIZED, no matmul / weights /
    // writer, borrowed OUT kept only to satisfy TP_OUTPUT. They differ only where split_program_unpack_tilize is
    // consulted (compute kernel selection, scalar DFB, compute defines, compute bindings).
    const bool split_program_tilize_only =
        ((std::getenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM") != nullptr) || split_program_unpack_tilize) &&
        height_sharded && !is_conv_1d_depthwise_conv && !enable_split_reader && !enable_activation_reuse &&
        (in0_num_blocks_w == 1) && (num_blocks_weight_w_per_core == 1);

    // Program A (tilize-only) EXPERIMENT: the pure tilize (conv_tilize_only_metal2.cpp) faults with ERROR_TRISC1
    // 0x19 mid-stream (after several 4-wide blocks tilize cleanly, config identical to the PASSING standalone
    // tilize), which matches the residual Quasar tilize DEST-bank-release LLK issue in HALF sync (syncfull=0 —
    // the Quasar height-sharded default here). Force FULL DEST sync for the split tilize path: full sync
    // collapses DEST to one section, eliminating the per-tile half-sync bank rotation whose release appears to
    // stop freeing banks. This was never actually run on the conv (the earlier "half sync" try just re-set the
    // default). If it clears the 0x19, full sync is the workaround; if not, this is a clean LLK escalation.
    // dst_full_sync: the FULL-sync force below was added for the OLD MATH-A2D-datacopy tilize path (collapse
    // DEST to one section to dodge the half-sync bank-rotation leak). It is HARMFUL for the UnpackToDestEn path:
    // there `_llk_unpack_dest_dvalid_section_done_` in SyncFull RESETS the dest bank to 0 every tile
    // (llk_unpack_common.h:94), so every tile reuses DEST bank 0 with NO double-buffering — a tight UNPACK<->PACK
    // ping-pong on bank 0 that races/laps once there are enough tiles (exposed by the K-fix's filter_h x tile
    // count -> TRISC0 0x19). SyncHalf instead double-buffers across 2 banks (unpack tile t+1 into bank1 while
    // pack drains tile t from bank0). So only force full-sync when NOT using unpack-to-dest.
    if (split_program_tilize_only && std::getenv("TT_METAL_QSR_TILIZE_UNPACK_TO_DEST") == nullptr) {
        dst_full_sync_en = true;
    }
    // SyncFull EXPERIMENT for the UnpackToDestEn tilize. The comment above warned SyncFull is harmful for UTD —
    // but that predates ARMING the DEST-dvalid handshake (set_up_dest_dvalid_per_thread in tilize_init). UNARMED,
    // SyncFull's single-bank ping-pong raced (nothing gated the refill). ARMED, a single DEST bank is exactly
    // what serializes UNPACK<->PACK: the unpacker physically cannot refill bank 0 until the packer's dvalid-clear,
    // so it CANNOT lap the packer (the non-deterministic multi-section race in SyncHalf's 2-bank double-buffer).
    // Applies to the fused conv too — the matmul then runs SyncFull (single-buffered DEST; valid, slower). Gated
    // to the Quasar UTD path only. If this makes the fused conv deterministic, SyncFull is the workaround.
    if (arch_is_quasar && std::getenv("TT_METAL_QSR_TILIZE_UNPACK_TO_DEST") != nullptr) {
        dst_full_sync_en = true;
    }

    // OPTION B — Program A must produce the FULL im2col contraction dim K, not just one act_block_w K-sub-block.
    // On Quasar force_conv_no_spill is off, so full_inner_dim does NOT bump act_block_w to the full K; here
    // act_block_w_ntiles = in_ch*kw/32 (ONE window row) while the true matmul K = in_ch*kh*kw/32 = the prepared
    // weights' K-height (b_shape[-2]) = act_block_w_ntiles * filter_h. The fused conv contracts the extra
    // filter_h window-rows via a window-accumulation loop; the split's single tilize must instead tilize ALL
    // filter_h K-sub-blocks into a contiguous [M, full_K] activation so Program B's plain matmul matches the
    // [full_K, N] weights. The reader already gathers the full window (ACTFILL nt = full_K tiles per tile-row);
    // we just tilize all of it. (RISK: the reader's gathered K-ordering — window-position x channel — must match
    // the weights' [r][s][c] flattening; if not, PCC needs a weight reorder. Verified by the e2e PCC test.)
    const uint32_t full_k_ntiles = act_block_w_ntiles * filter_h;

    TT_FATAL(
        act_matrix_height_ntiles % per_core_out_matrix_height_ntiles == 0,
        "Activation matrix height in tiles ({}) must be divisible by per-core output matrix height in tiles ({})",
        act_matrix_height_ntiles,
        per_core_out_matrix_height_ntiles);
    uint32_t total_noop_cores = total_num_cores_per_weight_slice - parallelization_config.num_cores_nhw;
    uint32_t total_active_num_cores = parallelization_config.num_cores_nhw * num_weight_slices_width;
    TT_FATAL(!block_sharded || total_noop_cores == 0, "All cores should be active for block sharded convs");

    if (has_bias) {
        TT_FATAL(
            bias_ntiles == weight_matrix_width_ntiles,
            "Bias tiles ({}) must equal weight matrix width in tiles ({})",
            bias_ntiles,
            weight_matrix_width_ntiles);
    }
    uint32_t bias_ntiles_per_core = bias_ntiles / num_weight_slices_width;

    const uint32_t act_block_w_logical_scalars =
        is_conv_1d_depthwise_conv
            ? shard_shape[1] * (coalesce_1d_depthwise_kw_reads ? filter_w : 1)
            : (!slice_inner_dim ? shard_shape[1] * filter_h * filter_w : shard_shape[1] * filter_w);
    uint32_t act_block_w_extra_align_bytes =
        (tt::round_up(act_block_w_logical_scalars, tt::constants::TILE_WIDTH) - act_block_w_logical_scalars) *
        a.element_size();
    const uint32_t act_block_w_extra_align_scalars = act_block_w_extra_align_bytes / a.element_size();
    // When using block float format, we must handle cases where the data doesn't align to 16-scalar boundaries.
    const bool needs_act_block_zero_out =
        act_block_w_extra_align_scalars % 16 != 0 && tt::tt_metal::is_block_float(output.dtype());

    const uint32_t tilized_act_tile_size = tt::tile_size(tilized_act_df);

    // Only enable packer l1 accumulation when there are in0_num_blocks_w > 2.
    // QSR: the Quasar hardware packer-L1-accumulate pack path (PACR0_TILE_INC in-place accumulate combined
    // with the QSR_RESTORE_WR / g_dfb ring rewind between K-blocks) mis-addresses the matmul_partials CB and
    // overruns it -> OOB L1 write -> ERROR_TRISC1 fault on the pack thread (opcode 0x19 = PACR0_TILE_INC).
    // Unlike WH/BH (address derived from fifo_wr_ptr), Quasar's packer DST_TILE_FACE_ROW_IDX counter is not
    // resynced to the rewound descriptor. Force off so K-accumulation goes through the FPU-reload path
    // (copy_block_matmul_partials reload + re-accumulate), which IS ported/validated on Quasar. This only
    // drops a perf optimization; correctness is preserved. Remove once the LLK packer-L1-acc + ring-rewind
    // counter resync is fixed. (This factory is Quasar-only, so no arch guard is needed.)
    const bool packer_l1_acc_en = false;
    (void)determine_packer_l1_acc(packer_l1_acc, has_bias, in0_num_blocks_w);
    const uint32_t batch = sliding_window_config.get_output_shape()[0];
    const uint32_t output_image_width = sliding_window_config.get_output_shape()[2];
    const uint32_t output_image_height = sliding_window_config.get_output_shape()[1];
    const uint32_t total_output_height_ntiles =
        (batch * output_image_height * output_image_width) / tt::constants::TILE_HEIGHT;

    Conv2dConfig conv_config = Conv2dConfig{
        .weights_dtype = b.dtype(),
        .config_tensors_in_dram = config_tensors_in_dram,
        .shard_layout = a.memory_config().memory_layout(),
        .output_layout = (untilize_out ? Layout::ROW_MAJOR : Layout::TILE),
        .enable_act_double_buffer = enable_act_double_buffer,
        .enable_weights_double_buffer = enable_weights_double_buffer,
        .enable_activation_reuse = enable_activation_reuse,
        // This factory forces single-DM-fill (enable_split_reader = false below): the reader fills the
        // WHOLE ACT block into the ACT CB alone. get_cb_info() independently decides split-reader via
        // is_split_reader_viable() when force_split_reader is nullopt, and would then size ACT for the
        // split layout (half-blocks + a separate ACT_SECOND_READER CB) — inconsistent with the reader,
        // which deadlocks the act producer/consumer. Force it off here so the CB sizing matches.
        .force_split_reader = false};

    // ---- Determine split_reader_cb_shared up front so we can reject it (resolution #3) ----
    // get_cb_info needs the indices page size first; compute the shared-overlap flag after CB info.

    // ---- Allocate the op-owned conv_reader_indices tensor ----
    // This intermediate config tensor must outlive the cached program; it is parked on
    // ProgramArtifacts::op_owned_tensors.  move_config_tensor_to_device() picks the correct
    // (HEIGHT_SHARDED/L1_SMALL, BLOCK_SHARDED-replicated, or INTERLEAVED/DRAM) layout per shard scheme;
    // we then release_mesh_tensor() the moved device Tensor to hand back a SOLE-OWNER MeshTensor
    // (#44565) without the ~Tensor force-deallocate hazard the legacy WorkloadDescriptor::buffers
    // parking guarded against.
    const uint32_t reader_act_block_h_datums = enable_split_reader ? act_block_h_datums_split : act_block_h_datums;
    const uint32_t reader_act_block_h_datums_last =
        enable_split_reader ? act_block_h_nsubblocks_split_last * tt::constants::TILE_HEIGHT : 0;
    std::vector<std::vector<uint16_t>> conv_sharded_input_top_left_indices =
        ttnn::operations::sliding_window::generate_sliding_window_op_config(
            op_trace_metadata,
            shard_boundaries,
            stride_w,
            true,
            reader_act_block_h_datums,
            reader_act_block_h_datums_last);
    Tensor conv_reader_indices_tensor = ttnn::operations::sliding_window::construct_on_host_config_tensor(
        conv_sharded_input_top_left_indices, parallel_config, config_tensors_in_dram);
    conv_reader_indices_tensor = ttnn::operations::sliding_window::move_config_tensor_to_device(
        conv_reader_indices_tensor, parallel_config, block_sharded, a.device(), config_tensors_in_dram);
    log_trace(tt::LogOp, "Conv2D Config Tensor : {}", conv_reader_indices_tensor);

    tt::tt_metal::Buffer* conv_reader_indices_buffer = conv_reader_indices_tensor.buffer();
    const uint32_t reader_indices_actual_page_size = conv_reader_indices_buffer->page_size();
    // Release the sole-owner MeshTensor (the source Tensor is left deallocated; ~Tensor will not
    // force-free the device buffer that the cached program still references).
    tt::tt_metal::MeshTensor reader_indices_mesh_tensor =
        conv_reader_indices_tensor.device_storage().release_mesh_tensor();

    // ---- Query CB sizing/format/backing via the shared conv2d helper ----
    std::vector<CBInfo> cb_info = get_cb_info(
        compute_kernel_config,
        block_config,
        parallelization_config,
        b.padded_shape(),
        {filter_h, filter_w},
        {sliding_window_config.input_hw.first, sliding_window_config.input_hw.second},
        {dilation_h, dilation_w},
        conv_config,
        a.dtype(),
        output.dtype(),
        shard_shape,
        output_image_width,
        has_bias,
        is_conv_1d_depthwise_conv,
        skip_activation_mcast,
        input_channels_padded,
        reader_indices_actual_page_size);

    // split_reader_cb_shared (block-sharded split-reader overlapped ACT CB, CB_TAXONOMY_ANALYSIS.md §2) is
    // subsumed by the split-reader deferral above (enable_split_reader is forced false), so it can never
    // arise here.

    // 1D depthwise compute uses dest-reuse for accumulation — no MATMUL_PARTIALS CB is allocated.
    const bool partials_cb_uses_output =
        !is_conv_1d_depthwise_conv && get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).is_globally_allocated;
    log_debug(tt::LogOp, "partials_cb_uses_output: {}", partials_cb_uses_output);

    const bool reader_indices_globally_allocated =
        get_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).is_globally_allocated;

    // Convenience accessor for CB sizing.
    auto cb = [&](Conv2dCb name) -> const CBInfo& { return get_cb_info_by_name(cb_info, name); };

    // QSR: the packer RELU (llk_pack_relu_config(ReluConfig::zero())) leaves ~one 16x16 face per output tile
    // UNCLAMPED on Quasar -- proven by bisect (test_conv2d_correctness_bisect): a height-sharded conv is
    // correct with the activation off (PCC 0.99997) but drops to 0.8547 with RELU on, uniform / tap-count-
    // independent / fidelity-independent, with stale negatives surviving (output absmax > golden). WH is fine.
    // Route RELU through the SFPU activation path on Quasar instead (full DEST tile; the same path GELU uses)
    // by NOT taking the packer-relu fast-path here -- the `!pack_relu` branch below then merges the SFPU relu
    // defines (SFPU_OP_*_ACTIVATION), applied per output tile in the compute kernel after the bias add.
    // TODO(LLK): fix the Quasar packer relu face coverage so the faster packer clamp can be used again.
    // (arch_is_quasar is declared earlier, near the out_subblock 1x1 workaround.)
    bool pack_relu =
        fused_activation.has_value() && fused_activation.value().op_type == unary::UnaryOpType::RELU && !arch_is_quasar;

    const bool check_skip_compute = input_cores != output_cores;
    // populate_skipped_work_cores is only reachable with split reader (deferred), so it is always false.
    const bool populate_skipped_work_cores = false;

    // ============================================================================
    //  Mcast geometry (mirrors the legacy build_program_descriptor_sharded)
    // ============================================================================
    const uint32_t in_num_cores_x = input_cores.bounding_box().end_coord.x + 1;
    const uint32_t in_num_cores_y = input_cores.bounding_box().end_coord.y + 1;

    // Anchor mcast geometry on the actual grid origin (input shard bbox top-left) so an offset core_grid
    // (e.g. a sub-grid at logical y=1) works; grid_start == (0,0) on the default grid, so every derived
    // coord below is unchanged there.
    const CoreCoord grid_start = input_cores.bounding_box().start_coord;
    const CoreCoord top_left_core = grid_start;
    const CoreCoord top_left_core_plus_one = {grid_start.x + 1, grid_start.y + 1};
    const CoreCoord bottom_right_core = {(std::size_t)in_num_cores_x - 1, (std::size_t)in_num_cores_y - 1};
    const CoreCoord top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    // top_left_core_plus_one (logical origin + {1,1}) is consumed ONLY by the block-sharded 2D
    // weights-mcast sender path.  Resolving its physical coord unconditionally throws on grids that don't
    // contain that logical coord (a height-sharded 1D conv on a 2x1 grid, or any 2-core sub-grid), so only
    // resolve it when block-sharded; height-sharded leaves it unused.
    const CoreCoord top_left_core_plus_one_physical =
        block_sharded ? device->worker_core_from_logical_core(top_left_core_plus_one) : CoreCoord{0, 0};
    const CoreCoord bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    CoreRangeSet mcast_sender_cores =
        CoreRangeSet(CoreRange(top_left_core, top_left_core));  // If single core, this kernel doesn't do mcasting
    CoreRangeSet mcast_receiver_cores;

    if (block_sharded) {
        const CoreCoord out_bottom_right_core = {(std::size_t)num_cores_x - 1, (std::size_t)num_cores_y - 1};
        if (transpose_mcast) {
            mcast_sender_cores = CoreRangeSet(CoreRange(top_left_core, CoreCoord(0, num_cores_y - 1)));
            if (!skip_weights_mcast) {
                mcast_receiver_cores = CoreRange(CoreCoord(1, 0), out_bottom_right_core);
            }
        } else {
            mcast_sender_cores = CoreRangeSet(CoreRange(top_left_core, CoreCoord(num_cores_x - 1, 0)));
            if (!skip_weights_mcast) {
                mcast_receiver_cores = CoreRange(CoreCoord(0, 1), out_bottom_right_core);
            }
        }
        if (populate_skipped_work_cores) {
            mcast_sender_cores = input_cores.subtract(mcast_receiver_cores);
        }
    } else {
        if (!skip_weights_mcast) {
            // Height-sharded / depthwise: place the receiver only on the ACTIVE (compute) cores minus the
            // sender, NOT on the full grid.  The legacy placed it on all_cores and let noop cores early-
            // return, but under the Metal 2.0 DFB model a writer on a noop node would be a WEIGHTS producer
            // with no compute consumer there (producer-without-consumer).  Active cores = input_cores
            // (reader/compute placement); the noop cores never consumed weights anyway and never
            // incremented the mcast semaphore, so dropping them is behaviorally identical.
            mcast_receiver_cores = input_cores.subtract(mcast_sender_cores);
        }
    }

    const bool create_writer_mcast_receiver = !skip_weights_mcast;

    uint32_t reader_arg_act_block_h_datums = (enable_split_reader ? act_block_h_datums_split : act_block_h_datums);
    TT_FATAL(reader_arg_act_block_h_datums % 2 == 0, "2 Indices are packed in one uint32_t word.");

    ActivationReuseConfig activation_reuse_config;
    if (enable_activation_reuse) {
        activation_reuse_config = calculate_activation_reuse_params(
            output_image_height,
            output_image_width,
            filter_w,
            filter_h,
            conv_act_c_read_bytes,
            act_block_w_extra_align_bytes,
            act_block_h_nsubblocks_split,
            act_block_h_nsubblocks_split_last,
            tilized_act_tile_size,
            act_block_w_ntiles,
            act_block_h_ntiles,
            out_block_h_ntiles,
            total_output_height_ntiles,
            act_matrix_height_ntiles,
            cb_info,
            enable_split_reader,
            input_cores,
            batch);
    }

    // ---- compute defines ----
    std::map<std::string, std::string> compute_defines;
    if (fused_activation.has_value() && !pack_relu) {
        // Pass the activation input dtype: several unary ops (RELU, SIGNBIT, ...) branch float-vs-int in
        // get_op_init_and_func and TT_FATAL if it is absent. On Quasar RELU takes this SFPU path (packer-relu
        // is disabled above), so the dtype is now required. The activation runs on the accumulated conv
        // result that becomes the output, so use the output dtype (bf16 => the float relu_tile path).
        compute_defines.merge(ttnn::operations::unary::utils::get_defines(
            fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i", output.dtype()));
    }
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), output_cores.num_cores(), compute_defines, ttnn::get_throttle_level(compute_kernel_config));

    // UnpackToDestEn bypass (tt-metal #49445): when TT_METAL_QSR_TILIZE_UNPACK_TO_DEST is set, inject
    // QSR_TILIZE_UNPACK_TO_DEST so tilize.h routes the tilize unpacker straight into DEST (UNP_DEST) and turns
    // the MATH A2D datacopy into a no-MOP sync forwarder — sidestepping the faulting Quasar tilize datacopy MOP
    // (the 0x19: the semaphore-scheme tilize never frees the FPU dest-dvalid ring; unpack-to-dest removes the
    // ring-advancing MOVA2D). Applies to any Quasar tilize_block path (the fused conv kernel AND the split
    // Program-A tilize-only kernel) — excludes depthwise (no tilize_block) and the unpack-tilize reduce probe
    // (uses tilizeA_B_reduce, not tilize_block).
    if (arch_is_quasar && !is_conv_1d_depthwise_conv && !split_program_unpack_tilize &&
        (std::getenv("TT_METAL_QSR_TILIZE_UNPACK_TO_DEST") != nullptr)) {
        compute_defines.insert({"QSR_TILIZE_UNPACK_TO_DEST", "1"});
    }

    // ============================================================================
    //  Build the ProgramSpec
    // ============================================================================
    m2::ProgramSpec spec;
    spec.name = is_conv_1d_depthwise_conv ? "conv2d_depthwise"
                                          : (block_sharded ? "conv2d_block_sharded" : "conv2d_height_sharded");

    // ---- Tensor parameters ----
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = TP_INPUT, .spec = a.mesh_tensor().tensor_spec()});
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = TP_OUTPUT, .spec = output.mesh_tensor().tensor_spec()});
    // Program A (split_program_tilize_only) has no weights reader / writer / matmul, so NO kernel binds
    // TP_WEIGHTS or TP_BIAS. The Metal-2.0 spec validator (program_spec.cpp:575) rejects any TensorParameter
    // that is declared but bound by no kernel, so we must not declare them in the tilize-only program.
    if (!split_program_tilize_only) {
        spec.tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = TP_WEIGHTS, .spec = b.mesh_tensor().tensor_spec()});
        if (has_bias) {
            spec.tensor_parameters.push_back(
                m2::TensorParameter{.unique_id = TP_BIAS, .spec = bias.value().mesh_tensor().tensor_spec()});
        }
    }
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = TP_READER_INDICES, .spec = reader_indices_mesh_tensor.tensor_spec()});

    // ---- Semaphores ----
    // act mcast: block-sharded reader only.  weights mcast: always declared because the writer kernels
    // construct Semaphore(sem::weights_mcast_*) unconditionally (the mcast bodies are #ifndef SKIP_MCAST,
    // but the semaphore objects are built up-front), so the tokens must exist even when mcast is skipped.
    // target_nodes spans all kernel-placement cores (sender ∪ receiver ⊆ all_cores).
    if (block_sharded) {
        spec.semaphores.push_back(m2::SemaphoreSpec{.unique_id = SEM_ACT_MCAST_SENDER, .target_nodes = all_cores});
        spec.semaphores.push_back(m2::SemaphoreSpec{.unique_id = SEM_ACT_MCAST_RECEIVER, .target_nodes = all_cores});
    }
    // Program A (split_program_tilize_only) registers no writer kernels, so nothing constructs the weights
    // mcast Semaphore objects — do not declare their tokens (an unbound semaphore is at best dead, and the
    // spec validator may reject it the same way it rejects unbound tensor parameters).
    if (!split_program_tilize_only) {
        spec.semaphores.push_back(m2::SemaphoreSpec{.unique_id = SEM_WEIGHTS_MCAST_SENDER, .target_nodes = all_cores});
        spec.semaphores.push_back(
            m2::SemaphoreSpec{.unique_id = SEM_WEIGHTS_MCAST_RECEIVER, .target_nodes = all_cores});
    }

    // ---- Dataflow buffers ----
    auto make_dfb = [&](const m2::DFBSpecName& id, Conv2dCb name) {
        const CBInfo& info = cb(name);
        return m2::DataflowBufferSpec{
            .unique_id = id,
            .entry_size = info.page_size,
            .num_entries = info.num_pages,
            .data_format_metadata = info.data_format,
        };
    };

    // ACT: real FIFO (height-sharded direct reader->compute, or block-sharded mcast result).
    spec.dataflow_buffers.push_back(make_dfb(DFB_ACT, Conv2dCb::ACT));
    // WEIGHTS: weights reader/writer -> compute.  (Program A / tilize-only has no matmul: no weights/bias.)
    if (!split_program_tilize_only) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_WEIGHTS, Conv2dCb::WEIGHTS));
        // BIAS (optional).
        if (has_bias) {
            spec.dataflow_buffers.push_back(make_dfb(DFB_BIAS, Conv2dCb::BIAS));
        }
    }
    // ACT_SECOND_READER (split reader, non-shared): writer fills, compute consumes.
    if (enable_split_reader) {
        spec.dataflow_buffers.push_back(make_dfb(DFB_ACT_SECOND_READER, Conv2dCb::ACT_SECOND_READER));
    }
    // ACT_ROW_MAJOR + ACT_TILIZED: block-sharded path (reader tilize input / mcast source) and the
    // depthwise path (tilized_in0).  Height-sharded conv reads directly into ACT (no separate RM/tilized
    // handoff in the reader); still emit ACT_TILIZED for depthwise.  Emit both when block_sharded; emit
    // ACT_TILIZED for depthwise.
    if (block_sharded) {
        // get_cb_info() overlaps ACT_ROW_MAJOR onto ACT (num_pages = 0) when conv_input_df == output_df
        // (the legacy `overlap_im2col_cb` L1 optimization — true for bf16-in/bf16-out). The Metal 2.0
        // aliased-DFB form of that overlap isn't wired up in this port yet, and a 0-entry DFB is rejected.
        // DEFER the optimization: give ACT_ROW_MAJOR its own buffer (sized like ACT, which is what the
        // non-overlapped path uses) so it is a valid DFB. Slightly more L1; identical numerics, and the
        // existing reader-producer / compute-consumer bindings + HAS_ACT_ROW_MAJOR define already assume
        // a real ACT_ROW_MAJOR DFB.
        const CBInfo& act_rm_info = cb(Conv2dCb::ACT_ROW_MAJOR_BFLOAT16);
        if (act_rm_info.num_pages > 0) {
            spec.dataflow_buffers.push_back(make_dfb(DFB_ACT_RM, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16));
        } else {
            spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
                .unique_id = DFB_ACT_RM,
                .entry_size = act_rm_info.page_size,
                .num_entries = cb(Conv2dCb::ACT).num_pages,
                .data_format_metadata = act_rm_info.data_format,
            });
        }
        spec.dataflow_buffers.push_back(make_dfb(DFB_ACT_TILIZED, Conv2dCb::ACT_TILIZED));
    } else {
        // Non-block-sharded paths consume ACT_TILIZED from compute:
        // - depthwise path emits/consumes ACT_TILIZED in compute
        // - height-sharded non-depthwise path tilizes ACT internally in compute
        // (The 2x ACT_TILIZED double-buffering diagnostic was reverted: a 392-tile K-block is ~800 KB
        //  = 50,176 units of 16 B, and 2x = 100,352 units overflows the uint16_t DFB ring-extent field
        //  (max 65,536 = 1 MB). Capacity double-buffering of a full K-block is structurally impossible
        //  on Quasar — see dfb_conv2d_analysis.md.)
        if (split_program_unpack_tilize) {
            // UNPACK-TILIZE PROBE (diagnostic): tilize into a PLAIN (non-borrowed) ACT_TILIZED DFB sized to hold
            // the WHOLE per-core tilized activation (num_blocks_act_h_per_core x one ACT_TILIZED block = M*K
            // tiles); the probe reduce writes here (throwaway). Plus a reduce-scalar (srcB) DFB — one tile, face
            // geometry {face_r_dim=1, num_faces=4} mirroring the pool's scalar CB — credited as a compute
            // self-loop, value unfilled.
            const CBInfo& tilized_info = cb(Conv2dCb::ACT_TILIZED);
            spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
                .unique_id = DFB_ACT_TILIZED,
                .entry_size = tilized_info.page_size,
                .num_entries = tilized_info.num_pages * num_blocks_act_h_per_core,
                .data_format_metadata = tilized_info.data_format,
            });
            spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
                .unique_id = DFB_IN_SCALAR,
                .entry_size = tilized_info.page_size,
                .num_entries = 1,
                .data_format_metadata = tilized_info.data_format,
                .unpack_face_geometry_metadata = FaceGeometry{.face_r_dim = 1, .num_faces = 4},
            });
        } else if (split_program_tilize_only) {
            // OPTION B / Program A: the tilize writes STRAIGHT INTO the borrowed OUT (sized to M*K below,
            // borrowed_from OUTPUT — the op's output IS the tilized activation). No separate ACT_TILIZED DFB.
        } else if (split_tilize_matmul) {
            // OPTION C: hold ALL height blocks of tilized activation at once (num_blocks_act_h_per_core x
            // one block) so Phase 1 can tilize every block before Phase 2's matmul consumes them. NB: the
            // ring extent (page_size_units x num_entries) must stay under the uint16_t limit (65,536 units
            // = 1 MB); if the full per-core tilized activation exceeds that, the DFB spec is rejected at
            // program creation and this path cannot be used for that conv (fall back to the fused kernel).
            const CBInfo& tilized_info = cb(Conv2dCb::ACT_TILIZED);
            spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
                .unique_id = DFB_ACT_TILIZED,
                .entry_size = tilized_info.page_size,
                .num_entries = tilized_info.num_pages * num_blocks_act_h_per_core,
                .data_format_metadata = tilized_info.data_format,
            });
        } else {
            spec.dataflow_buffers.push_back(make_dfb(DFB_ACT_TILIZED, Conv2dCb::ACT_TILIZED));
        }
    }

    // MATMUL_PARTIALS: self-loop accumulator (resolution #2).  Borrowed-from OUTPUT when
    // partials_cb_uses_output.  1D depthwise allocates no partials CB (dest-reuse), so skip it there.
    if (!is_conv_1d_depthwise_conv && !split_program_tilize_only) {
        auto dfb = make_dfb(DFB_MATMUL_PARTIALS, Conv2dCb::MATMUL_PARTIALS);
        if (partials_cb_uses_output) {
            dfb.borrowed_from = TP_OUTPUT;
        }
        spec.dataflow_buffers.push_back(std::move(dfb));
    }

    // OUT: borrowed from the OUTPUT tensor — compute packs the conv result directly into the output shard.
    {
        m2::DataflowBufferSpec dfb;
        if (split_program_tilize_only) {
            // OPTION B / Program A: the op output IS the FULL tilized im2col activation — M tile-rows x full_K
            // tile-cols (full_K = act_block_w_ntiles * filter_h = the prepared weights' K-height), so Program B's
            // matmul [M, full_K] x [full_K, N] matches. compute_output_specs sizes the output tensor to the same
            // [M*32, full_K*32] shard, so the borrowed OUT matches. (Was mis-sized to M x act_block_w = one
            // window-row's K-sub-block, causing the K=128 vs 512 matmul mismatch.)
            const CBInfo& tilized_info = cb(Conv2dCb::ACT_TILIZED);
            dfb = m2::DataflowBufferSpec{
                .unique_id = DFB_OUT,
                .entry_size = tilized_info.page_size,
                .num_entries = per_core_out_matrix_height_ntiles * full_k_ntiles,
                .data_format_metadata = tilized_info.data_format,
            };
            // QSR OUT headroom REMOVED (red herring): the borrowed-output exact-fill it guarded doesn't stall
            // (reserve waits free >= n; last block has free == n) and the matching +1 shard row broke the strict
            // emulator sharded readback. num_entries = M*full_K exact.
        } else {
            dfb = make_dfb(DFB_OUT, Conv2dCb::OUT);
            // QSR OUT headroom REMOVED (red herring): the borrowed-output exact-fill it guarded doesn't stall
            // (reserve waits free >= n; the last block has free == n, so it passes — same as WH/BH's borrowed
            // conv OUT, which uses no headroom). The matching +1 shard row broke the strict emulator sharded
            // readback (tile_read_bytes 80 vs 81). The real stalls were the DEST-dvalid race (fixed via
            // set_up_dest_dvalid_per_thread) and the trailing unpacker pop. num_entries stays exact.
        }
        dfb.borrowed_from = TP_OUTPUT;
        spec.dataflow_buffers.push_back(std::move(dfb));
    }

    // ACT_SHARDED: the resident input shard is read by address via tensor::act_sharded (a local
    // TensorAccessor) in the reader — no borrowed self-loop DFB.

    // READER_INDICES: L1-resident indices are read by address via tensor::reader_indices. Only the
    // DRAM-config path needs a fresh L1 landing DFB (the reader DMAs the slice into it, then reads it);
    // that path is not exercised by resnet.
    if (!reader_indices_globally_allocated) {
        auto dfb = make_dfb(DFB_READER_INDICES, Conv2dCb::READER_INDICES);
        spec.dataflow_buffers.push_back(std::move(dfb));
    }

    const tt::tt_metal::NOC writer_mcast_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    const tt::tt_metal::NOC reader_noc =
        writer_mcast_noc == tt::tt_metal::NOC::NOC_0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;

    // grid (for the block-sharded act-mcast Y/X lookup varargs)

    // ============================================================================
    //  Kernel sources
    // ============================================================================
    const std::string reader_kernel = [&]() -> std::string {
        if (!is_conv_1d_depthwise_conv && block_sharded) {
            return "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                   "reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2_metal2.cpp";
        } if (is_conv_1d_depthwise_conv) {
            return "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                   "reader_depthwise_conv1d_metal2.cpp";
        }
        return "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
               "reader_conv_activations_padded_with_halo_3x3_weights_v2_metal2.cpp";
    }();
    const std::string compute_kernel =
        is_conv_1d_depthwise_conv     ? "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                                        "compute_depthwise_conv1d_metal2.cpp"
        : split_program_unpack_tilize ? "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                                        "conv_unpack_tilize_probe_metal2.cpp"
        : split_program_tilize_only   ? "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                                        "conv_tilize_only_metal2.cpp"
        : split_tilize_matmul         ? "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                                        "conv_bmm_split_tilize_metal2.cpp"
                                      : "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                                        "conv_bmm_tilize_metal2.cpp";
    const std::string writer_sender_kernel =
        block_sharded ? "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                        "writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp"
                      : "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                        "reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks_metal2.cpp";
    const std::string writer_receiver_kernel =
        block_sharded ? "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                        "writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks_metal2.cpp"
                      : "ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/"
                        "reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks_metal2.cpp";

    // ============================================================================
    //  READER kernel
    // ============================================================================
    // DFB bindings differ per sub-layout.
    std::vector<m2::DFBBinding> reader_dfb_bindings;
    std::vector<m2::TensorBinding> reader_tensor_bindings;
    std::vector<m2::SemaphoreBinding> reader_sem_bindings;
    if (block_sharded) {
        // 2D reader: produces ACT_ROW_MAJOR + ACT (mcast), consumes ACT_TILIZED (mcast source); self-loops
        // borrowed ACT_SHARDED + READER_INDICES; uses act mcast semaphores.
        reader_dfb_bindings = {
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_RM,
                .accessor_name = "act_row_major",
                .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_TILIZED,
                .accessor_name = "act_tilized",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
        };
        reader_sem_bindings = {
            m2::SemaphoreBinding{.semaphore_spec_name = SEM_ACT_MCAST_SENDER, .accessor_name = "act_mcast_sender"},
            m2::SemaphoreBinding{.semaphore_spec_name = SEM_ACT_MCAST_RECEIVER, .accessor_name = "act_mcast_receiver"},
        };
    } else {
        // Height-sharded / depthwise reader: produces ACT (direct). act_sharded + reader_indices are read
        // by address via tensor bindings (no self-loop DFBs).
        reader_dfb_bindings = {
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        };
    }
    // The resident activation shard and (L1-resident) reader-indices are read by address via local
    // TensorAccessors — tensor::act_sharded and tensor::reader_indices — rather than borrowed self-loop CBs.
    reader_tensor_bindings.push_back(
        m2::TensorBinding{.tensor_parameter_name = TP_INPUT, .accessor_name = "act_sharded"});
    reader_tensor_bindings.push_back(
        m2::TensorBinding{.tensor_parameter_name = TP_READER_INDICES, .accessor_name = "reader_indices"});

    // The depthwise reader has no CONFIG_TENSOR_IN_DRAM path (its indices are always L1-resident); only
    // the height-/block-sharded readers DMA the config slice into the fresh READER_INDICES DFB.
    TT_FATAL(
        !(config_tensors_in_dram && is_conv_1d_depthwise_conv),
        "conv2d Metal 2.0 sharded factory: config_tensors_in_dram is not supported for the 1D depthwise path.");
    if (!reader_indices_globally_allocated) {
        // DRAM-config: the reader DMAs the indices slice from tensor::reader_indices into this fresh L1
        // DFB and then reads it (a self-loop on the reader; this path is not exercised by resnet).
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = DFB_READER_INDICES,
            .accessor_name = "reader_indices",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = DFB_READER_INDICES,
            .accessor_name = "reader_indices",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }

    m2::KernelSpec reader_kernel_spec{
        .unique_id = KERNEL_READER,
        .source = std::filesystem::path(reader_kernel),
        .dfb_bindings = std::move(reader_dfb_bindings),
        .semaphore_bindings = std::move(reader_sem_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        // QSR: this conv activation reader (all 3 variants: height-sharded / 2D-mcast / depthwise) fills the
        // ACT (and ACT_ROW_MAJOR) DFB via per-window "stick" NOC reads — read_sticks() in conv_reader_common.hpp
        // issues many sub-tile (conv_act_c_read_bytes each) async reads per act block, then reserve_back/push_back.
        // That sub-tile stick fill is exactly the pattern that stalls the DFB implicit-sync credit accounting
        // (reader pinned at NRBW/cb_reserve_back, compute idle, writer at NWFW). Mirror the tilize/transpose
        // HC-sharded workaround: opt out of implicit sync so explicit reserve/push credits stay authoritative.
        // (The writer/weights kernels do full-tile page reads into WEIGHTS/BIAS only — no opt-out needed; and
        // the reader is the only DM kernel on the ACT/ACT_RM producer + ACT_TILIZED consumer sides, so the
        // cross-kernel agreement rule is satisfied.)
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = reader_noc},
                .gen2_config = m2::DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true},
            },
    };

    // Reader compile-time args (per sub-layout).
    auto& r_cta = reader_kernel_spec.compile_time_args;
    if (block_sharded) {
        r_cta = {
            {"dilation_h", (uint32_t)dilation_h},
            {"dilation_w", (uint32_t)dilation_w},
            {"stride_w", (uint32_t)stride_w},
            {"conv_act_c_read_bytes", (uint32_t)conv_act_c_read_bytes},
            {"window_outer", (uint32_t)window_outer},
            {"act_block_num_tiles_read",
             (uint32_t)(enable_split_reader ? act_block_num_tiles_split : act_block_num_tiles)},
            {"weight_size_h", (uint32_t)filter_h},
            {"weight_size_w", (uint32_t)filter_w},
            {"padded_conv_act_size_w", (uint32_t)conv_act_size_w + pad_w},
            {"act_block_w_extra_align_bytes", (uint32_t)act_block_w_extra_align_bytes},
            {"act_num_blocks_h", (uint32_t)num_blocks_act_h_per_core},
            {"act_block_num_tiles", (uint32_t)act_block_num_tiles},
            {"act_w_num_outer", (uint32_t)conv_act_c_blocks},
            {"act_mcast_num_dests", (uint32_t)(transpose_mcast ? num_cores_y - 1 : num_cores_x - 1)},
            {"act_mcast_num_cores", (uint32_t)(transpose_mcast ? num_cores_y - 1 : num_cores_x - 1)},
            {"act_mcast_tile_size_bytes", (uint32_t)tilized_act_tile_size},
            {"transpose_mcast", (uint32_t)(transpose_mcast ? 1 : 0)},
            {"needs_act_block_zero_out", (uint32_t)needs_act_block_zero_out},
            {"split_reader_enabled", (uint32_t)enable_split_reader},
        };
        if (config_tensors_in_dram) {
            r_cta.insert({"config_page_size", conv_reader_indices_buffer->page_size()});
        }
    } else if (is_conv_1d_depthwise_conv) {
        // Depthwise reader (reader_depthwise_conv1d_metal2): no dilation/zero-out/reuse args; carries
        // window_inner and the coalesce flag.
        r_cta = {
            {"stride_w", (uint32_t)stride_w},
            {"conv_act_c_read_bytes", (uint32_t)conv_act_c_read_bytes},
            {"window_outer", (uint32_t)window_outer},
            {"window_inner", (uint32_t)window_inner},
            {"act_block_num_tiles", (uint32_t)act_block_num_tiles},
            {"weight_size_h", (uint32_t)filter_h},
            {"weight_size_w", (uint32_t)filter_w},
            {"conv_act_size_w_padded", (uint32_t)conv_act_size_w + pad_w},
            {"act_block_w_extra_align_bytes", (uint32_t)act_block_w_extra_align_bytes},
            {"act_num_blocks_h", (uint32_t)num_blocks_act_h_per_core},
            {"coalesce_kw_reads", (uint32_t)coalesce_1d_depthwise_kw_reads},
        };
    } else {
        // Height-sharded reader (reader_conv_activations_padded_with_halo_..._metal2): no window_inner;
        // carries the activation-reuse block (present-but-inert, reuse deferred).
        r_cta = {
            {"dilation_h", (uint32_t)dilation_h},
            {"dilation_w", (uint32_t)dilation_w},
            {"stride_w", (uint32_t)stride_w},
            {"conv_act_c_read_bytes", (uint32_t)conv_act_c_read_bytes},
            {"window_outer", (uint32_t)window_outer},
            {"act_block_num_tiles", (uint32_t)act_block_num_tiles},
            {"weight_size_h", (uint32_t)filter_h},
            {"weight_size_w", (uint32_t)filter_w},
            {"conv_act_size_w_padded", (uint32_t)conv_act_size_w + pad_w},
            {"act_block_w_extra_align_bytes", (uint32_t)act_block_w_extra_align_bytes},
            {"act_num_blocks_h", (uint32_t)num_blocks_act_h_per_core},
            {"needs_act_block_zero_out", (uint32_t)needs_act_block_zero_out},
            {"split_reader_enabled", (uint32_t)enable_split_reader},
            {"activation_reuse_enabled", (uint32_t)enable_activation_reuse},
        };
        if (config_tensors_in_dram) {
            r_cta.insert({"config_page_size", conv_reader_indices_buffer->page_size()});
        }
        // Activation-reuse block (8 args), present-but-inert (reuse deferred -> 0).
        r_cta.insert({"act_reuse_cb_tiles", 0u});
        r_cta.insert({"act_block_w_tiles", 0u});
        r_cta.insert({"readers_process_full_image_widths", 0u});
        r_cta.insert({"image_width_tiles", 0u});
        r_cta.insert({"output_image_width", 0u});
        r_cta.insert({"window_reuse_offset", 0u});
        r_cta.insert({"need_to_push_remaining_tiles", 0u});
        r_cta.insert({"single_core_processes_multiple_batches", 0u});
    }

    // Reader runtime-arg schema / defines.
    if (block_sharded) {
        reader_kernel_spec.runtime_arg_schema = {
            .runtime_arg_names =
                {"mcast_dest_noc_start_x",
                 "mcast_dest_noc_start_y",
                 "mcast_dest_noc_end_x",
                 "mcast_dest_noc_end_y",
                 "act_mcast_sender_id",
                 "act_mcast_sender_noc_x",
                 "is_receiver_core",
                 "is_sender_core",
                 "dram_config_reader_index"},
        };
        // Per-node Y (or X, for non-transpose) NoC lookup table -> runtime varargs.
        reader_kernel_spec.advanced_options.num_runtime_varargs = transpose_mcast ? in_num_cores_y : in_num_cores_x;
    } else if (is_conv_1d_depthwise_conv) {
        // Depthwise reader reads no runtime args.
    } else {
        reader_kernel_spec.runtime_arg_schema = {
            .runtime_arg_names = {"core_index", "remaining_tiles_to_push"},
        };
    }
    if (skip_activation_mcast && block_sharded) {
        // SKIP_MCAST is only meaningful on the block-sharded (act-mcast) reader.
        reader_kernel_spec.compiler_options.defines.insert({"SKIP_MCAST", "1"});
    }
    if (config_tensors_in_dram && !is_conv_1d_depthwise_conv) {
        reader_kernel_spec.compiler_options.defines.insert({"CONFIG_TENSOR_IN_DRAM", "1"});
    }
    // SPLIT_READER / ACTIVATION_REUSE are deferred (enable_split_reader / enable_activation_reuse are
    // forced false above), so those reader defines are never set.

    // ============================================================================
    //  WRITER (weights/bias mcast) sender + receiver kernels
    // ============================================================================
    // Writer compile-time args.  The 1D (height-sharded/depthwise) and 2D (block-sharded) writer forks
    // read DIFFERENT named-CTA sets, and the sender reads a few geometry CTAs the receiver does not.  These
    // are NAMED CTAs (codegen produces args::<name>), so every name a kernel reads must be emitted.  Split
    // reader is deferred, so the split-reader/activation-reuse CTAs are present-but-inert (value 0).
    auto append_reuse_dummy = [&](m2::KernelSpec::CompileTimeArgs& ctas) {
        ctas.insert({"act_reuse_cb_tiles", 0u});
        ctas.insert({"act_block_w_tiles", 0u});
        ctas.insert({"readers_process_full_image_widths", 0u});
        ctas.insert({"image_width_tiles", 0u});
        ctas.insert({"output_image_width", 0u});
        ctas.insert({"window_reuse_offset", 0u});
        ctas.insert({"need_to_push_remaining_tiles", 0u});
        ctas.insert({"single_core_processes_multiple_batches", 0u});
    };
    auto build_writer_ctas = [&](bool is_sender) {
        m2::KernelSpec::CompileTimeArgs ctas;
        if (block_sharded) {
            // 2D writer CTA set (writer_tiled_out_2d_..._metal2).
            ctas.insert({"num_blocks_weight_h", num_blocks_act_w});
            ctas.insert({"weight_block_num_tiles", weight_block_num_tiles});
            ctas.insert({"weight_block_height_num_outer", out_conv_c_blocks});
            if (is_sender) {
                ctas.insert({"weight_block_height_ntiles", weight_block_h_ntiles});
                ctas.insert({"weight_block_width_ntiles", weight_block_w_ntiles});
                ctas.insert({"weight_stride_h", weight_matrix_width_ntiles});
                ctas.insert({"weight_next_block_stride_w", weight_block_w_ntiles});
            }
            ctas.insert({"bias_ntiles", bias_ntiles_per_core});
            ctas.insert({"out_num_blocks_h", num_blocks_act_h_per_core});
            ctas.insert({"out_num_blocks_w", num_blocks_weight_w_per_core});
            if (is_sender) {
                ctas.insert({"weight_block_height_num_outer_in", out_conv_c_blocks});
            }
            ctas.insert({"fuse_bias", (uint32_t)has_bias});
            ctas.insert({"split_reader_enabled", (uint32_t)enable_split_reader});
            ctas.insert({"window_outer", (uint32_t)window_outer});
            ctas.insert({"act_block_num_tiles_split_last", (uint32_t)act_block_num_tiles_split_last});
            ctas.insert({"conv_act_c_read_bytes", (uint32_t)conv_act_c_read_bytes});
            ctas.insert({"weight_size_w", (uint32_t)filter_w});
            ctas.insert({"padded_conv_act_size_w", (uint32_t)(conv_act_size_w + pad_w)});
            ctas.insert({"act_block_w_extra_align_bytes", (uint32_t)act_block_w_extra_align_bytes});
            ctas.insert({"needs_act_block_zero_out", (uint32_t)needs_act_block_zero_out});
            ctas.insert({"dilation_h", (uint32_t)dilation_h});
            ctas.insert({"dilation_w", (uint32_t)dilation_w});
            ctas.insert({"stride_w", (uint32_t)stride_w});
            ctas.insert({"weight_size_h", (uint32_t)filter_h});
        } else {
            // 1D writer CTA set (reader_writer_tiled_out_1d_..._metal2).
            ctas.insert({"num_blocks_weight_h", num_blocks_act_w});
            ctas.insert({"weight_block_num_tiles", weight_block_num_tiles});
            if (is_sender) {
                ctas.insert({"weight_block_height_num_outer", out_conv_c_blocks});
                ctas.insert({"weight_block_height_ntiles", weight_block_h_ntiles});
                ctas.insert({"weight_block_width_ntiles", weight_block_w_ntiles});
                ctas.insert({"weight_stride_h", weight_matrix_width_ntiles});
                ctas.insert({"weight_next_block_stride_h", weight_matrix_width_ntiles * weight_block_h_ntiles});
            }
            ctas.insert({"bias_ntiles", bias_ntiles_per_core});
            ctas.insert({"out_num_blocks_h", num_blocks_act_h_per_core});
            ctas.insert({"fuse_bias", (uint32_t)has_bias});
            ctas.insert({"split_reader_enabled", (uint32_t)enable_split_reader});
            ctas.insert({"activation_reuse_enabled", (uint32_t)(enable_activation_reuse && height_sharded)});
            ctas.insert({"act_block_num_tiles", (uint32_t)act_block_num_tiles});
            ctas.insert({"conv_act_c_read_bytes", (uint32_t)conv_act_c_read_bytes});
            ctas.insert({"weight_size_w", (uint32_t)filter_w});
            ctas.insert({"conv_act_size_w_padded", (uint32_t)(conv_act_size_w + pad_w)});
            ctas.insert({"act_block_w_extra_align_bytes", (uint32_t)act_block_w_extra_align_bytes});
            ctas.insert({"needs_act_block_zero_out", (uint32_t)needs_act_block_zero_out});
            ctas.insert({"dilation_h", (uint32_t)dilation_h});
            ctas.insert({"dilation_w", (uint32_t)dilation_w});
            ctas.insert({"stride_w", (uint32_t)stride_w});
            ctas.insert({"weights_size_h", (uint32_t)filter_h});
        }
        // Split reader is deferred -> activation-reuse block is present-but-inert.
        append_reuse_dummy(ctas);
        return ctas;
    };

    // Writer DFB / tensor bindings.
    // WEIGHTS/BIAS mcast: BOTH the sender and the receiver are PRODUCERS of their own local WEIGHTS/BIAS CB
    // — the sender reads from DRAM and the receiver's CB is filled by the inbound mcast; each kernel does
    // reserve_back/push_back to publish the block to compute (the WEIGHTS/BIAS consumer).  Sender and
    // receiver run on disjoint node sets, so per node there is exactly one WEIGHTS producer.
    // OUT (resolution #1): packer-into-output, producer-only on compute.  The DM writer kernels here do
    // NOT actually drain OUT (they only mcast weights/bias), and they run on all_cores while compute runs
    // on input_cores in the height-sharded path — so binding the degenerate OUT consumer to the writers
    // would leave consumer-without-producer noop nodes.  Instead we bind the degenerate OUT consumer as a
    // compute self-loop (INTRA), exactly as the width-sharded factory does; this is node-coverage-safe and
    // equally faithful (the consumer is degenerate either way).
    // (split reader is deferred, so no ACT_SECOND_READER / borrowed-address-source bindings here.)
    auto build_writer_dfb_bindings = [&]() {
        std::vector<m2::DFBBinding> bindings;
        bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = DFB_WEIGHTS, .accessor_name = "weights", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        if (has_bias) {
            bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = DFB_BIAS, .accessor_name = "bias", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        }
        return bindings;
    };
    auto build_writer_tensor_bindings = [&]() {
        std::vector<m2::TensorBinding> bindings = {
            m2::TensorBinding{.tensor_parameter_name = TP_WEIGHTS, .accessor_name = "weights"},
        };
        if (has_bias) {
            bindings.push_back(m2::TensorBinding{.tensor_parameter_name = TP_BIAS, .accessor_name = "bias"});
        }
        return bindings;
    };

    // ---- writer mcast SENDER ----
    m2::KernelSpec writer_sender_spec{
        .unique_id = KERNEL_WRITER_SENDER,
        .source = std::filesystem::path(writer_sender_kernel),
        .dfb_bindings = build_writer_dfb_bindings(),
        .semaphore_bindings = {},
        .tensor_bindings = build_writer_tensor_bindings(),
        .compile_time_args = build_writer_ctas(/*is_sender=*/true),
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = writer_mcast_noc},
                // The sender does explicit reserve_back/push_back on WEIGHTS/BIAS/ACT_SECOND to publish
                // blocks to compute. On Quasar the implicit-sync ISR would ALSO bump those tile counters
                // -> double-count -> 16-bit counter overflow -> TILE_COUNTERS fault on the compute unpack
                // that consumes WEIGHTS. Opt out so explicit credits stay authoritative (mirrors the reader
                // + matmul mcast fix).
                .gen2_config = m2::DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true},
            },
    };
    // The sender always builds the weights-mcast Semaphore objects (even under SKIP_MCAST), so always bind.
    writer_sender_spec.semaphore_bindings = {
        m2::SemaphoreBinding{.semaphore_spec_name = SEM_WEIGHTS_MCAST_SENDER, .accessor_name = "weights_mcast_sender"},
        m2::SemaphoreBinding{
            .semaphore_spec_name = SEM_WEIGHTS_MCAST_RECEIVER, .accessor_name = "weights_mcast_receiver"},
    };
    if (block_sharded) {
        writer_sender_spec.runtime_arg_schema = {
            .runtime_arg_names =
                {"out_start_tile_id_w",
                 "bias_tile_offset",
                 "mcast_dest_noc_start_x",
                 "mcast_dest_noc_start_y",
                 "mcast_dest_noc_end_x",
                 "mcast_dest_noc_end_y",
                 "weights_mcast_num_dests",
                 "weights_mcast_num_cores",
                 "is_sender_core",
                 "skip_work"},
        };
    } else {
        // 1D sender: bias_tile_offset is read only under #ifdef FUSE_BIAS; remaining_tiles_to_push is
        // always referenced (in a constexpr-false ternary), so its name must exist.
        std::vector<std::string> names = {"out_start_tile_id_w"};
        if (has_bias) {
            names.push_back("bias_tile_offset");
        }
        names.insert(
            names.end(),
            {"mcast_dest_noc_start_x",
             "mcast_dest_noc_start_y",
             "mcast_dest_noc_end_x",
             "mcast_dest_noc_end_y",
             "weights_mcast_num_dests",
             "weights_mcast_num_cores",
             "remaining_tiles_to_push"});
        writer_sender_spec.runtime_arg_schema = {.runtime_arg_names = names};
    }

    // ---- writer mcast RECEIVER ----
    m2::KernelSpec writer_receiver_spec{
        .unique_id = KERNEL_WRITER_RECEIVER,
        .source = std::filesystem::path(writer_receiver_kernel),
        .dfb_bindings = build_writer_dfb_bindings(),
        .semaphore_bindings = {},
        .tensor_bindings = build_writer_tensor_bindings(),
        .compile_time_args = build_writer_ctas(/*is_sender=*/false),
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = writer_mcast_noc},
                // Same as the sender: the receiver does explicit reserve_back/push_back on WEIGHTS/BIAS/
                // ACT_SECOND; opt out of implicit sync so those tile counters aren't double-bumped
                // (else TILE_COUNTERS overflow on the compute unpack consuming WEIGHTS).
                .gen2_config = m2::DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true},
            },
    };
    if (create_writer_mcast_receiver) {
        writer_receiver_spec.semaphore_bindings = {
            m2::SemaphoreBinding{
                .semaphore_spec_name = SEM_WEIGHTS_MCAST_SENDER, .accessor_name = "weights_mcast_sender"},
            m2::SemaphoreBinding{
                .semaphore_spec_name = SEM_WEIGHTS_MCAST_RECEIVER, .accessor_name = "weights_mcast_receiver"},
        };
    }
    if (block_sharded) {
        writer_receiver_spec.runtime_arg_schema = {
            .runtime_arg_names = {"weights_mcast_sender_noc_x", "weights_mcast_sender_noc_y", "is_sender_core"},
        };
    } else {
        // 1D receiver: remaining_tiles_to_push is always referenced (constexpr-false ternary).
        writer_receiver_spec.runtime_arg_schema = {
            .runtime_arg_names = {
                "noop", "weights_mcast_sender_noc_x", "weights_mcast_sender_noc_y", "remaining_tiles_to_push"}};
    }

    // Writer defines.
    auto apply_writer_defines = [&](m2::KernelSpec& k, bool is_sender) {
        if (has_bias) {
            k.compiler_options.defines.insert({"FUSE_BIAS", "1"});
        }
        if (enable_split_reader) {
            k.compiler_options.defines.insert({"SPLIT_READER", "1"});
        }
        if (enable_activation_reuse) {
            k.compiler_options.defines.insert({"ACTIVATION_REUSE", "1"});
        }
        if (config_tensors_in_dram) {
            k.compiler_options.defines.insert({"CONFIG_TENSOR_IN_DRAM", "1"});
        }
        if (is_sender && skip_weights_mcast) {
            k.compiler_options.defines.insert({"SKIP_MCAST", "1"});
        }
    };
    apply_writer_defines(writer_sender_spec, /*is_sender=*/true);
    apply_writer_defines(writer_receiver_spec, /*is_sender=*/false);

    // ============================================================================
    //  COMPUTE kernel
    // ============================================================================
    std::vector<m2::DFBBinding> compute_dfb_bindings;
    if (split_program_unpack_tilize) {
        // UNPACK-TILIZE PROBE (diagnostic): compute consumes ACT + reduces into the PLAIN ACT_TILIZED DFB
        // (PRODUCER + degenerate CONSUMER, throwaway) with the IN_SCALAR reduce-scalar as a self-loop. OUT is
        // dead-but-bound so TP_OUTPUT stays satisfied. No weights / bias / matmul_partials.
        compute_dfb_bindings = {
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_TILIZED,
                .accessor_name = "act_tilized",
                .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_TILIZED,
                .accessor_name = "act_tilized",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_IN_SCALAR,
                .accessor_name = "in_scalar",
                .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_IN_SCALAR,
                .accessor_name = "in_scalar",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
        };
    } else if (split_program_tilize_only) {
        // OPTION B / Program A: compute consumes ACT (gathered by the reader) and tilizes it STRAIGHT INTO OUT
        // (borrowed from the op's output tensor — the op output IS the tilized activation). PRODUCER + the
        // degenerate self-CONSUMER, mirroring the fused packer-into-output idiom. No weights / act_tilized /
        // matmul_partials. Program B (host-level matmul) consumes this tilized activation.
        compute_dfb_bindings = {
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        };
    } else if (is_conv_1d_depthwise_conv) {
        // Depthwise: consumes ACT (raw RM) + WEIGHTS, produces ACT_TILIZED (tilize out) and consumes it,
        // produces OUT (dest-reuse accumulate, packer-into-output; no MATMUL_PARTIALS CB).
        compute_dfb_bindings = {
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_WEIGHTS,
                .accessor_name = "weights",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_TILIZED,
                .accessor_name = "act_tilized",
                .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_TILIZED,
                .accessor_name = "act_tilized",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            // OUT is also self-consumed (dest-reuse accumulate reads prior partial back from out_cb).
            m2::DFBBinding{
                .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        };
    } else {
        compute_dfb_bindings = {
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT, .accessor_name = "act", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_WEIGHTS,
                .accessor_name = "weights",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_TILIZED,
                .accessor_name = "act_tilized",
                .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_MATMUL_PARTIALS,
                .accessor_name = "matmul_partials",
                .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_MATMUL_PARTIALS,
                .accessor_name = "matmul_partials",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        };
        // OUT: compute packs the borrowed OUTPUT shard in place (degenerate self-consumer).
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = DFB_OUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        if (block_sharded) {
            // 2D path: compute consumes ACT_ROW_MAJOR (tilize input) and PRODUCES ACT_TILIZED; the READER
            // consumes ACT_TILIZED (mcast source).  So ACT_TILIZED is producer-only on compute here (the
            // CONSUMER lives on the reader — see the reader bindings), NOT a compute self-loop.
            compute_dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_RM,
                .accessor_name = "act_row_major",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        } else {
            // Height-sharded path: compute tilizes ACT -> ACT_TILIZED then matmuls ACT_TILIZED, so compute is
            // BOTH producer and consumer of ACT_TILIZED (INTRA self-loop; the reader produces ACT directly).
            compute_dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = DFB_ACT_TILIZED,
                .accessor_name = "act_tilized",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        if (has_bias) {
            compute_dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = DFB_BIAS, .accessor_name = "bias", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
    }

    m2::KernelSpec compute_kernel_spec{
        .unique_id = KERNEL_COMPUTE,
        .source = std::filesystem::path(compute_kernel),
        .compiler_options = {.defines = m2::KernelSpec::CompilerOptions::Defines(compute_defines)},
        .dfb_bindings = std::move(compute_dfb_bindings),
        .hw_config =
            m2::ComputeHardwareConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = dst_full_sync_en,
                .math_approx_mode = math_approx_mode,
            },
    };

    if (is_conv_1d_depthwise_conv) {
        compute_kernel_spec.compile_time_args = {
            {"in0_block_w", act_block_w_ntiles},
            {"in0_num_subblocks", act_num_subblocks},
            {"in0_block_num_tiles", act_block_num_tiles},
            {"in0_num_blocks_h", num_blocks_act_h_per_core},
            {"in0_num_blocks_w", in0_num_blocks_w},
            {"kernel_width", filter_w},
            {"coalesce_kw_reads", (uint32_t)coalesce_1d_depthwise_kw_reads},
        };
    } else {
        compute_kernel_spec.compile_time_args = {
            {"in0_block_w", act_block_w_ntiles},
            {"in0_num_subblocks", act_num_subblocks},
            {"in0_block_num_tiles", act_block_num_tiles},
            {"in0_subblock_num_tiles", act_subblock_num_tiles},
            // OPTION B / Program A (tilize-only, NOT the reduce probe): the tilize kernel computes
            // num_blocks = in0_num_blocks_h * reader_num_h_subblocks and tilizes num_blocks x in0_block_w
            // (=act_block_w) tiles. To cover the FULL im2col K (act_block_w * filter_h tiles) rather than just
            // one window-row, scale reader_num_h_subblocks by filter_h so num_blocks = M * filter_h and the
            // tilize produces [M, full_K] into OUT. (Only the tilize-only kernel reads this; the fused kernel and
            // the reduce probe keep the un-scaled value.)
            {"reader_num_h_subblocks",
             (enable_split_reader ? act_block_h_ntiles : act_subblock_h_ntiles * act_num_subblocks) *
                 ((split_program_tilize_only && !split_program_unpack_tilize) ? filter_h : 1u)},
            {"in1_num_subblocks", weight_num_subblocks},
            {"in1_block_num_tiles", weight_block_num_tiles},
            {"in1_block_w", weight_block_w_ntiles},
            {"in0_num_blocks_h", num_blocks_act_h_per_core},
            {"in0_num_blocks_w", in0_num_blocks_w},
            {"in1_num_blocks_w", num_blocks_weight_w_per_core},
            {"out_subblock_h", out_subblock_h_ntiles},
            {"out_subblock_w", out_subblock_w_ntiles},
            {"out_subblock_num_tiles", out_subblock_num_tiles},
            {"height_sharded", (uint32_t)height_sharded},
            {"untilize_out", (uint32_t)untilize_out},
            {"bias_ntiles_w", bias_ntiles_per_core},
            {"partials_cb_uses_output", (uint32_t)partials_cb_uses_output},
            {"in0_nblocks_w_tilize", conv_act_c_blocks},
            {"pack_relu", (uint32_t)pack_relu},
            {"packer_untilize", (uint32_t)(weight_block_w_ntiles <= 8)},
            {"packer_l1_acc", (uint32_t)packer_l1_acc_en},
            {"fuse_bias", (uint32_t)has_bias},
            {"split_reader", (uint32_t)enable_split_reader},
            {"activation_reuse", (uint32_t)enable_activation_reuse},
        };
        if (enable_activation_reuse) {
            compute_kernel_spec.compile_time_args.insert(
                {"image_width_in_tiles", activation_reuse_config.image_width_tiles});
            compute_kernel_spec.compile_time_args.insert(
                {"window_reuse_offset", activation_reuse_config.reuse_window_offset / COMPUTE_KERNEL_ADDRESS_DIVISOR});
            compute_kernel_spec.compile_time_args.insert(
                {"tilized_cb_row_offset",
                 activation_reuse_config.tilized_cb_row_offset / COMPUTE_KERNEL_ADDRESS_DIVISOR});
            compute_kernel_spec.compile_time_args.insert(
                {"tilized_cb_second_reader_offset",
                 activation_reuse_config.tilized_cb_second_reader_offset / COMPUTE_KERNEL_ADDRESS_DIVISOR});
        } else {
            compute_kernel_spec.compile_time_args.insert({"image_width_in_tiles", 0u});
            compute_kernel_spec.compile_time_args.insert({"window_reuse_offset", 0u});
            compute_kernel_spec.compile_time_args.insert({"tilized_cb_row_offset", 0u});
            compute_kernel_spec.compile_time_args.insert({"tilized_cb_second_reader_offset", 0u});
        }
        compute_kernel_spec.compile_time_args.insert({"split_reader_cb_shared", 0u});

        if (has_bias) {
            compute_kernel_spec.compiler_options.defines.insert({"FUSE_BIAS", "1"});
        }
        if (block_sharded) {
            // Only the block-sharded (mcast) path binds dfb::act_row_major on compute; the define guards
            // the kernel's in0_pretilize_cb_id = dfb::act_row_major reference (height-sharded has no
            // act_row_major DFB and tilizes dfb::act directly).
            compute_kernel_spec.compiler_options.defines.insert({"HAS_ACT_ROW_MAJOR", "1"});
        }
        if (enable_split_reader) {
            compute_kernel_spec.compiler_options.defines.insert({"SPLIT_READER", "1"});
        }
        if (enable_activation_reuse) {
            compute_kernel_spec.compiler_options.defines.insert({"ACTIVATION_REUSE", "1"});
        }
        if (check_skip_compute) {
            compute_kernel_spec.compiler_options.defines.insert({"CHECK_SKIP_COMPUTE", "1"});
            compute_kernel_spec.runtime_arg_schema = {.runtime_arg_names = {"skip_compute"}};
        }
        if (split_program_unpack_tilize) {
            // UNPACK-TILIZE PROBE: tilizeA_B_reduce_init + reduce_tile_math read REDUCE_OP / REDUCE_DIM macros
            // (mirroring the pool). MAX / REDUCE_ROW — the reduce output is throwaway; only completion matters.
            compute_kernel_spec.compiler_options.defines.insert({"REDUCE_OP", "PoolType::MAX"});
            compute_kernel_spec.compiler_options.defines.insert({"REDUCE_DIM", "ReduceDim::REDUCE_ROW"});
        }
    }

    // ---- Register kernels ----
    spec.kernels.push_back(std::move(reader_kernel_spec));
    // OPTION B / Program A has no weights/bias mcast: register only reader + compute (no writer kernels).
    if (!split_program_tilize_only) {
        spec.kernels.push_back(std::move(writer_sender_spec));
        if (create_writer_mcast_receiver) {
            spec.kernels.push_back(std::move(writer_receiver_spec));
        }
    }
    spec.kernels.push_back(std::move(compute_kernel_spec));

    // ---- Work units ----
    // reader + compute run on input_cores (height-sharded) / all_cores (block-sharded).  The writer
    // sender runs on mcast_sender_cores, the receiver on mcast_receiver_cores.  We emit one work unit per
    // kernel-placement set (the variant adapter realizes them).
    const CoreRangeSet reader_compute_cores = height_sharded ? input_cores : all_cores;
    // WorkUnitSpecs must have DISJOINT target_nodes; each lists ALL kernels that run on those nodes
    // (a kernel may appear in several WUs — its placement is the union of their nodes). A sender core
    // runs reader+compute+writer_sender; a receiver core runs reader+compute+writer_receiver; any
    // remaining reader/compute cores (block-sharded: output cores outside input_cores) run reader+compute.
    if (split_program_tilize_only) {
        // OPTION B / Program A: reader + compute only (no writer), across all reader/compute cores.
        m2::WorkUnitSpec wu{.name = "reader_compute", .target_nodes = reader_compute_cores};
        wu.kernels = {KERNEL_READER, KERNEL_COMPUTE};
        spec.work_units.push_back(std::move(wu));
    } else {
        {
            m2::WorkUnitSpec wu{.name = "sender", .target_nodes = mcast_sender_cores};
            wu.kernels = {KERNEL_READER, KERNEL_COMPUTE, KERNEL_WRITER_SENDER};
            spec.work_units.push_back(std::move(wu));
        }
        if (create_writer_mcast_receiver) {
            m2::WorkUnitSpec wu{.name = "receiver", .target_nodes = mcast_receiver_cores};
            wu.kernels = {KERNEL_READER, KERNEL_COMPUTE, KERNEL_WRITER_RECEIVER};
            spec.work_units.push_back(std::move(wu));
        }
        {
            CoreRangeSet reader_only_cores = reader_compute_cores.subtract(mcast_sender_cores);
            if (create_writer_mcast_receiver) {
                reader_only_cores = reader_only_cores.subtract(mcast_receiver_cores);
            }
            if (reader_only_cores.num_cores() > 0) {
                m2::WorkUnitSpec wu{.name = "reader_compute", .target_nodes = reader_only_cores};
                wu.kernels = {KERNEL_READER, KERNEL_COMPUTE};
                spec.work_units.push_back(std::move(wu));
            }
        }
    }

    // ============================================================================
    //  ProgramRunArgs
    // ============================================================================
    m2::ProgramRunArgs run_args;

    auto setup_mcast_args = [&](bool is_noc_0, uint32_t start_x, uint32_t start_y, uint32_t end_x, uint32_t end_y) {
        return is_noc_0 ? std::array<uint32_t, 4>{start_x, start_y, end_x, end_y}
                        : std::array<uint32_t, 4>{end_x, end_y, start_x, start_y};
    };

    // ---- Reader RTAs ----
    m2::KernelRunArgs reader_run_args{.kernel = KERNEL_READER};
    if (block_sharded) {
        std::vector<uint32_t> act_mcast_noc_y;  // X table for non-transpose, Y table for transpose
        if (transpose_mcast) {
            act_mcast_noc_y.reserve(in_num_cores_y);
            for (uint32_t core_idx_y = 0; core_idx_y < in_num_cores_y; ++core_idx_y) {
                act_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
            }
        } else {
            act_mcast_noc_y.reserve(in_num_cores_x);
            for (uint32_t core_idx_x = 0; core_idx_x < in_num_cores_x; ++core_idx_x) {
                act_mcast_noc_y.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
            }
        }
        const CoreCoord out_bottom_right_core = {(std::size_t)num_cores_x - 1, (std::size_t)num_cores_y - 1};
        const CoreCoord out_bottom_right_core_physical = device->worker_core_from_logical_core(out_bottom_right_core);
        const bool reader_is_noc_0 = reader_noc == tt::tt_metal::NOC::NOC_0;

        for (const CoreRange& core_range : all_cores.ranges()) {
            for (const CoreCoord& core : core_range) {
                const bool is_receiver_core = output_cores.contains(core);
                const bool is_sender_core = input_cores.contains(core);
                std::array<uint32_t, 4> mcast;
                uint32_t act_mcast_sender_id, act_mcast_sender_noc_x;
                if (transpose_mcast) {
                    CoreCoord bottom_core = {(std::size_t)core.x, (std::size_t)num_cores_y - 1};
                    CoreCoord bottom_core_physical = device->worker_core_from_logical_core(bottom_core);
                    mcast = setup_mcast_args(
                        reader_is_noc_0,
                        bottom_core_physical.x,
                        top_left_core_physical.y,
                        bottom_core_physical.x,
                        bottom_core_physical.y);
                    act_mcast_sender_id = core.y;
                    act_mcast_sender_noc_x = bottom_core_physical.x;
                } else {
                    CoreCoord core_physical = device->worker_core_from_logical_core(core);
                    mcast = setup_mcast_args(
                        reader_is_noc_0,
                        top_left_core_physical.x,
                        core_physical.y,
                        out_bottom_right_core_physical.x,
                        core_physical.y);
                    act_mcast_sender_id = core.x;
                    act_mcast_sender_noc_x = core_physical.y;
                }
                reader_run_args.runtime_arg_values.push_back(m2::KernelRunArgs::NodeRuntimeArgs{
                    .node = core,
                    .args =
                        {
                            {"mcast_dest_noc_start_x", mcast[0]},
                            {"mcast_dest_noc_start_y", mcast[1]},
                            {"mcast_dest_noc_end_x", mcast[2]},
                            {"mcast_dest_noc_end_y", mcast[3]},
                            {"act_mcast_sender_id", act_mcast_sender_id},
                            {"act_mcast_sender_noc_x", act_mcast_sender_noc_x},
                            {"is_receiver_core", (uint32_t)is_receiver_core},
                            {"is_sender_core", (uint32_t)is_sender_core},
                            {"dram_config_reader_index", transpose_mcast ? core.x : core.y},
                        },
                });
                m2::AdvancedKernelRunArgs::Varargs varargs(act_mcast_noc_y.begin(), act_mcast_noc_y.end());
                reader_run_args.advanced_options.runtime_varargs.insert({core, std::move(varargs)});
            }
        }
    } else if (is_conv_1d_depthwise_conv) {
        // Depthwise reader has no runtime args.
    } else {
        // Height-sharded reader: {core_index, remaining_tiles_to_push}.  Activation reuse is deferred, so
        // remaining_tiles_to_push is always 0.
        uint32_t core_index = 0;
        for (const CoreRange& core_range : input_cores.ranges()) {
            for (const CoreCoord& core : core_range) {
                reader_run_args.runtime_arg_values.push_back(m2::KernelRunArgs::NodeRuntimeArgs{
                    .node = core,
                    .args = {{"core_index", core_index}, {"remaining_tiles_to_push", 0u}},
                });
                core_index++;
            }
        }
    }
    run_args.kernel_run_args.push_back(std::move(reader_run_args));

    // ---- Writer SENDER RTAs ----
    // OPTION B / Program A registers no writer kernels, so emit no writer run-args for it.
    if (!split_program_tilize_only) {
        m2::KernelRunArgs writer_sender_run_args{.kernel = KERNEL_WRITER_SENDER};
        for (const CoreRange& core_range : mcast_sender_cores.ranges()) {
            for (const CoreCoord& core : core_range) {
                if (populate_skipped_work_cores && !output_cores.contains(core)) {
                    // Pad-out path: zeros with only the bias-flag/skip slots populated.
                    writer_sender_run_args.runtime_arg_values.push_back(m2::KernelRunArgs::NodeRuntimeArgs{
                        .node = core,
                        .args =
                            {
                                {"out_start_tile_id_w", 0u},
                                {"bias_tile_offset", 0u},
                                {"mcast_dest_noc_start_x", 0u},
                                {"mcast_dest_noc_start_y", 0u},
                                {"mcast_dest_noc_end_x", 0u},
                                {"mcast_dest_noc_end_y", 0u},
                                {"weights_mcast_num_dests", 0u},
                                {"weights_mcast_num_cores", 0u},
                                {"is_sender_core", 1u},
                                {"skip_work", 1u},
                            },
                    });
                    continue;
                }
                uint32_t weight_slice_i = ((block_sharded && transpose_mcast) || !block_sharded) ? core.y : core.x;
                uint32_t out_start_tile_id_w = weight_slice_i * per_core_out_matrix_width_ntiles;
                uint32_t bias_tile_offset = out_start_tile_id_w;
                TT_FATAL(
                    bias_tile_offset < bias_ntiles || !has_bias,
                    "bias_tile_offset {} should be less than bias_ntiles {}",
                    bias_tile_offset,
                    bias_ntiles);

                if (block_sharded) {
                    const bool is_sender_core = input_cores.contains(core);
                    std::array<uint32_t, 4> mcast;
                    if (transpose_mcast) {
                        CoreCoord right_core = {(std::size_t)num_cores_x - 1, (std::size_t)core.y};
                        CoreCoord right_core_physical = device->worker_core_from_logical_core(right_core);
                        TT_FATAL(core.x == 0, "Expected core.x to be 0 for sender in 2D mcast setup");
                        mcast = setup_mcast_args(
                            writer_mcast_noc == tt::tt_metal::NOC::NOC_0,
                            top_left_core_plus_one_physical.x,
                            right_core_physical.y,
                            bottom_right_core_physical.x,
                            right_core_physical.y);
                        writer_sender_run_args.runtime_arg_values.push_back(m2::KernelRunArgs::NodeRuntimeArgs{
                            .node = core,
                            .args =
                                {
                                    {"out_start_tile_id_w", out_start_tile_id_w},
                                    {"bias_tile_offset", bias_tile_offset},
                                    {"mcast_dest_noc_start_x", mcast[0]},
                                    {"mcast_dest_noc_start_y", mcast[1]},
                                    {"mcast_dest_noc_end_x", mcast[2]},
                                    {"mcast_dest_noc_end_y", mcast[3]},
                                    {"weights_mcast_num_dests", num_cores_x - 1},
                                    {"weights_mcast_num_cores", num_cores_x - 1},
                                    {"is_sender_core", (uint32_t)is_sender_core},
                                    {"skip_work", 0u},
                                },
                        });
                    } else {
                        CoreCoord top_core = {(std::size_t)core.x, 0};
                        CoreCoord top_core_physical = device->worker_core_from_logical_core(top_core);
                        TT_FATAL(core.y == 0, "Expected core.y to be 0 for sender in 2D mcast setup");
                        mcast = setup_mcast_args(
                            writer_mcast_noc == tt::tt_metal::NOC::NOC_0,
                            top_core_physical.x,
                            top_left_core_plus_one_physical.y,
                            top_core_physical.x,
                            bottom_right_core_physical.y);
                        writer_sender_run_args.runtime_arg_values.push_back(m2::KernelRunArgs::NodeRuntimeArgs{
                            .node = core,
                            .args =
                                {
                                    {"out_start_tile_id_w", out_start_tile_id_w},
                                    {"bias_tile_offset", bias_tile_offset},
                                    {"mcast_dest_noc_start_x", mcast[0]},
                                    {"mcast_dest_noc_start_y", mcast[1]},
                                    {"mcast_dest_noc_end_x", mcast[2]},
                                    {"mcast_dest_noc_end_y", mcast[3]},
                                    {"weights_mcast_num_dests", num_cores_y - 1},
                                    {"weights_mcast_num_cores", num_cores_y - 1},
                                    {"is_sender_core", (uint32_t)is_sender_core},
                                    {"skip_work", 0u},
                                },
                        });
                    }
                } else {
                    std::array<uint32_t, 4> mcast = setup_mcast_args(
                        writer_mcast_noc == tt::tt_metal::NOC::NOC_0,
                        top_left_core_physical.x,
                        top_left_core_physical.y,
                        bottom_right_core_physical.x,
                        bottom_right_core_physical.y);
                    m2::KernelRunArgs::RuntimeArgValues args;
                    args["out_start_tile_id_w"] = out_start_tile_id_w;
                    if (has_bias) {
                        args["bias_tile_offset"] = bias_tile_offset;
                    }
                    args["mcast_dest_noc_start_x"] = mcast[0];
                    args["mcast_dest_noc_start_y"] = mcast[1];
                    args["mcast_dest_noc_end_x"] = mcast[2];
                    args["mcast_dest_noc_end_y"] = mcast[3];
                    args["weights_mcast_num_dests"] = total_active_num_cores - 1;
                    args["weights_mcast_num_cores"] = total_num_cores - 1;
                    // remaining_tiles_to_push is always in the 1D schema (activation reuse deferred -> 0).
                    args["remaining_tiles_to_push"] = 0u;
                    writer_sender_run_args.runtime_arg_values.push_back(
                        m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = std::move(args)});
                }
            }
        }
        run_args.kernel_run_args.push_back(std::move(writer_sender_run_args));
    }  // if (!split_program_tilize_only) — writer SENDER RTAs

    // ---- Writer RECEIVER RTAs ----
    if (create_writer_mcast_receiver && !split_program_tilize_only) {
        m2::KernelRunArgs writer_receiver_run_args{.kernel = KERNEL_WRITER_RECEIVER};
        for (const CoreRange& core_range : mcast_receiver_cores.ranges()) {
            for (const CoreCoord& core : core_range) {
                if (block_sharded) {
                    uint32_t sender_noc_x, sender_noc_y;
                    if (transpose_mcast) {
                        CoreCoord right_core = {(std::size_t)num_cores_x - 1, (std::size_t)core.y};
                        CoreCoord right_core_physical = device->worker_core_from_logical_core(right_core);
                        sender_noc_x = top_left_core_physical.x;
                        sender_noc_y = right_core_physical.y;
                    } else {
                        CoreCoord top_core = {(std::size_t)core.x, 0};
                        CoreCoord top_core_physical = device->worker_core_from_logical_core(top_core);
                        sender_noc_x = top_core_physical.x;
                        sender_noc_y = top_left_core_physical.y;
                    }
                    const bool is_sender_core = input_cores.contains(core);
                    writer_receiver_run_args.runtime_arg_values.push_back(m2::KernelRunArgs::NodeRuntimeArgs{
                        .node = core,
                        .args =
                            {
                                {"weights_mcast_sender_noc_x", sender_noc_x},
                                {"weights_mcast_sender_noc_y", sender_noc_y},
                                {"is_sender_core", (uint32_t)is_sender_core},
                            },
                    });
                } else {
                    bool is_no_op_core = !input_cores.contains(core);
                    m2::KernelRunArgs::RuntimeArgValues args;
                    args["noop"] = (uint32_t)is_no_op_core;
                    args["weights_mcast_sender_noc_x"] = top_left_core_physical.x;
                    args["weights_mcast_sender_noc_y"] = top_left_core_physical.y;
                    // remaining_tiles_to_push is always in the 1D receiver schema (reuse deferred -> 0).
                    args["remaining_tiles_to_push"] = 0u;
                    writer_receiver_run_args.runtime_arg_values.push_back(
                        m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = std::move(args)});
                }
            }
        }
        run_args.kernel_run_args.push_back(std::move(writer_receiver_run_args));
    }

    // ---- Compute RTAs ----
    m2::KernelRunArgs compute_run_args{.kernel = KERNEL_COMPUTE};
    if (check_skip_compute && !is_conv_1d_depthwise_conv) {
        CoreCoord bottom_right_core_out = output_cores.bounding_box().end_coord;
        uint32_t end_coord_x = bottom_right_core_out.x;
        uint32_t end_coord_y = bottom_right_core_out.y;
        for (const CoreRange& range : all_cores.ranges()) {
            for (const CoreCoord& core : range) {
                bool skip_compute = transpose_mcast ? core.y > end_coord_y : core.x > end_coord_x;
                compute_run_args.runtime_arg_values.push_back(m2::KernelRunArgs::NodeRuntimeArgs{
                    .node = core, .args = {{"skip_compute", (uint32_t)skip_compute}}});
            }
        }
    }
    run_args.kernel_run_args.push_back(std::move(compute_run_args));

    // ---- Op-owned tensors ----
    std::vector<tt::tt_metal::MeshTensor> op_owned_tensors;
    op_owned_tensors.reserve(1);
    op_owned_tensors.push_back(std::move(reader_indices_mesh_tensor));
    const tt::tt_metal::MeshTensor& reader_indices_owned = op_owned_tensors[0];

    // ---- Tensor args ----
    run_args.tensor_args.emplace(TP_INPUT, std::cref(a.mesh_tensor()));
    run_args.tensor_args.emplace(TP_OUTPUT, std::cref(output.mesh_tensor()));
    // Program A declares no TP_WEIGHTS/TP_BIAS parameters (see tensor-parameter block above) — the run args
    // must match the declared parameters, so only pass them on the fused/matmul path.
    if (!split_program_tilize_only) {
        run_args.tensor_args.emplace(TP_WEIGHTS, std::cref(b.mesh_tensor()));
        if (has_bias) {
            run_args.tensor_args.emplace(TP_BIAS, std::cref(bias.value().mesh_tensor()));
        }
    }
    run_args.tensor_args.emplace(TP_READER_INDICES, std::cref(reader_indices_owned));

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
        .op_owned_tensors = std::move(op_owned_tensors),
    };
}

}  // namespace ttnn::prim::qsr
