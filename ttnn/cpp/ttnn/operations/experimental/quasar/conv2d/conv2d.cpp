// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <cstdlib>  // std::getenv (TT_METAL_QSR_CONV_SPLIT_PROGRAM Option-B two-program split)
#include <optional>
#include <string>
#include <utility>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // tt::tile_size (DFB ring-extent cap for no-spill conv)
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include "tt-metalium/math.hpp"
#include "ttnn/operations/sliding_window/op_slicing/op_slicing.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/quasar/conv2d/conv2d.hpp"
#include "ttnn/operations/experimental/quasar/conv2d/device/conv2d_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/experimental/quasar/matmul/matmul.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/experimental/quasar/halo/halo.hpp"
#include "ttnn/operations/experimental/quasar/tilize/tilize.hpp"
#include "ttnn/operations/experimental/quasar/move/move.hpp"
#include "ttnn/operations/experimental/quasar/pad/pad.hpp"
#include "ttnn/operations/experimental/quasar/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/experimental/quasar/to_device/to_device.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"

namespace ttnn::operations::conv {
// get_conv_padded_input_shape_and_mem_config has external linkage but is not declared in
// conv2d_utils.hpp (only used within conv2d_utils.cpp originally). Forward-declare it so the quasar
// shard_or_reshard fork can reuse it; resolves against conv2d_utils.cpp at link time.
std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> get_conv_padded_input_shape_and_mem_config(
    MeshDevice* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv);
}  // namespace ttnn::operations::conv

namespace ttnn::operations::experimental::quasar::detail {

// Shared conv host infrastructure (conv2d_utils.hpp, prepare_conv2d_weights.hpp) is reused from the
// original conv namespaces; bring it into scope so this impl's bare references resolve unchanged.
using namespace ttnn::operations::conv;          // conv2d_utils helpers (get_conv_configs, determine_*, ...)
using namespace ttnn::operations::conv::conv2d;  // prepare_conv2d_weights helpers, get_conv2d_slice_attr
using ttnn::operations::sliding_window::ParallelConfig;  // return/locals of shard_or_reshard fork

using ttnn::operations::experimental::quasar::Conv2dResult;
using ttnn::operations::experimental::quasar::Conv2dResultWithOptions;
using Result = Conv2dResult;
using ResultWithOptions = Conv2dResultWithOptions;

// Mirrors to_layout_op.cpp::requires_padding_change for a RM->TILE conversion: true when tiling the
// tensor would change its padded shape (i.e. needs val-padding, not just a layout repack).
static bool conv_act_requires_tile_padding_qsr(const ttnn::Tensor& tensor) {
    tt::tt_metal::PageConfig page_config = tt::tt_metal::PageConfig(Layout::TILE);
    if (tensor.layout() == Layout::TILE) {
        page_config = tt::tt_metal::PageConfig(Layout::TILE, tensor.tensor_spec().tile());
    }
    tt::tt_metal::TensorSpec padded_spec(
        tensor.padded_shape(), tt::tt_metal::TensorLayout(tensor.dtype(), page_config, tensor.memory_config()));
    return tensor.padded_shape() != padded_spec.padded_shape();
}

// Quasar variant of conv2d_utils::tilize_with_optional_deallocation. The original converts the conv
// activation to TILE via core to_layout, which dispatches the ORIGINAL tilize kernel. Here we route
// the common (already tile-aligned) case through the QUASAR tilize op so that kernel runs from the
// quasar tree. The genuinely-needs-padding case falls back to the shared helper, since quasar has no
// tilize_with_val_padding port.
static void tilize_with_optional_deallocation_qsr(ttnn::Tensor& input_tensor_on_device, bool deallocate) {
    if (input_tensor_on_device.layout() == Layout::TILE) {
        return;
    }
    if (conv_act_requires_tile_padding_qsr(input_tensor_on_device)) {
        tilize_with_optional_deallocation(input_tensor_on_device, deallocate);
        return;
    }
    ttnn::Tensor input_tensor_tilized = ttnn::operations::experimental::quasar::tilize(input_tensor_on_device);
    if (deallocate) {
        input_tensor_on_device.deallocate(/*force*/ true);
    }
    input_tensor_on_device = std::move(input_tensor_tilized);
}

// Relocate a (0,0)-anchored CoreRangeSet onto the same-shaped ranges whose top-left is `offset`. The
// shared conv host helpers (get_conv_padded_input_shape_and_mem_config, determine_*_parallel_config)
// always build shard grids from (0,0); this shifts them onto conv_config.core_grid's origin so the
// conv's activation/output shards — and therefore the program factory's kernel placement, which follows
// the input shard grid — land on the requested cores instead of always starting at (0,0). No-op when
// offset == (0,0), so the default-grid path is unchanged.
static CoreRangeSet offset_core_range_set_qsr(const CoreRangeSet& crs, const CoreCoord& offset) {
    if (offset.x == 0 && offset.y == 0) {
        return crs;
    }
    std::vector<CoreRange> ranges;
    ranges.reserve(crs.ranges().size());
    for (const CoreRange& r : crs.ranges()) {
        ranges.emplace_back(
            CoreCoord(r.start_coord.x + offset.x, r.start_coord.y + offset.y),
            CoreCoord(r.end_coord.x + offset.x, r.end_coord.y + offset.y));
    }
    return CoreRangeSet(ranges);
}

// Rebuild a sharded MemoryConfig with its shard grid relocated to `offset` (shape/shard-shape preserved).
static ttnn::MemoryConfig offset_sharded_mem_config_qsr(const ttnn::MemoryConfig& mem_config, const CoreCoord& offset) {
    if ((offset.x == 0 && offset.y == 0) || !mem_config.shard_spec().has_value()) {
        return mem_config;
    }
    const auto& shard_spec = mem_config.shard_spec().value();
    return tt::tt_metal::MemoryConfig(
        mem_config.memory_layout(),
        mem_config.buffer_type(),
        tt::tt_metal::ShardSpec(
            offset_core_range_set_qsr(shard_spec.grid, offset), shard_spec.shape, shard_spec.orientation));
}

// Quasar variant of conv2d_utils::shard_or_reshard_tensor_if_required. Mirrors the shared function but
// routes the per-conv input pad / reshard / reallocation through the QUASAR pad / to_memory_config / move
// (hence the quasar reshard + interleaved<->sharded kernels). The BLOCK_SHARDED mm-conv tilize workaround
// (#13979) keeps the core to_layout, which quasar tilize does not yet replicate for block-sharded inputs.
// Shared decision helpers (get_conv_padded_input_shape_and_mem_config, flatten_4d_shape, ...) are reused.
static std::tuple<ttnn::Tensor, ParallelConfig, ParallelConfig> shard_or_reshard_tensor_if_required_qsr(
    MeshDevice* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv,
    bool auto_shard) {
    ttnn::Tensor input_tensor = input_tensor_;  // tensor to return
    bool input_tensor_on_device = tt::tt_metal::is_device_tensor(input_tensor_);
    auto compute_grid_size = device->compute_with_storage_grid_size();

    auto [input_padded_shape, input_tensor_sharded_memory_config, needs_shard_or_reshard] =
        get_conv_padded_input_shape_and_mem_config(
            device, input_tensor_, conv_config, batch_size, height, width, in_channels, out_channels, is_mm_conv);

    // Honor the OFFSET of conv_config.core_grid (the shared helper honors only its size, always anchoring
    // at (0,0)). Shift the activation shard grid onto that origin so the conv runs on the requested cores
    // — e.g. a 2-core sub-grid at logical y=1 on a small (emulated) device. For HEIGHT_SHARDED the output
    // parallel config == the input parallel config, so this offset propagates to the output shard (and
    // thus the program factory's placement) automatically. No-op when core_grid is unset or starts at (0,0).
    const CoreCoord core_grid_offset =
        conv_config.core_grid.has_value() ? conv_config.core_grid.value().bounding_box().start_coord : CoreCoord{0, 0};
    input_tensor_sharded_memory_config =
        offset_sharded_mem_config_qsr(input_tensor_sharded_memory_config, core_grid_offset);

    ParallelConfig parallel_config = {
        .grid = input_tensor_sharded_memory_config.shard_spec().value().grid,
        .shard_scheme = input_tensor_sharded_memory_config.memory_layout(),
        .shard_orientation = input_tensor_sharded_memory_config.shard_spec().value().orientation};

    auto output_compute_grid_size = get_output_compute_grid_size(compute_grid_size, conv_config, parallel_config);
    ParallelConfig output_parallel_config = determine_output_parallel_config(
        parallel_config, output_compute_grid_size, out_channels, parallel_config.shard_orientation, is_mm_conv);

    // We can have flat and unflattened (n, h, w, c) tensors here
    const auto flattened_input_shape = flatten_4d_shape(input_tensor.logical_shape());
    const auto flattened_padded_input_shape = flatten_4d_shape(input_tensor.padded_shape());

    input_tensor = ttnn::operations::experimental::quasar::reshape(
        input_tensor, flattened_input_shape, flattened_padded_input_shape);
    const ttnn::Shape& input_shape = flattened_input_shape;

    if (needs_shard_or_reshard) {
        uint32_t tensor_height = input_shape[2];
        uint32_t tensor_width = input_shape[3];
        if (!input_tensor_on_device) {
            if (input_padded_shape[-2] != tensor_height || input_padded_shape[-1] != tensor_width) {
                input_tensor = ttnn::operations::experimental::quasar::pad(
                    input_tensor,
                    tt::tt_metal::Array4D(
                        {input_shape[0], input_shape[1], input_padded_shape[-2], input_padded_shape[-1]}),
                    tt::tt_metal::Array4D({0, 0, 0, 0}),
                    0);
            }
        }

        // In case we are in auto sharded codepath and convolution maps to matmul
        // Skip sharding of the input tensor and run the matmul out of interleaved tensor.
        bool auto_shard_mm = auto_shard && is_mm_conv;
        if (input_tensor_on_device) {
            if (is_mm_conv && input_tensor.layout() == Layout::ROW_MAJOR &&
                parallel_config.shard_scheme != TensorMemoryLayout::HEIGHT_SHARDED) {
                // Workaround #13979 ttnn::tilize doesn't support BLOCK_SHARDED layout.
                // Route the (already tile-aligned) common case through quasar tilize so its kernels run
                // from the quasar tree; fall back to quasar to_layout only when val-padding is required.
                Tensor input_tensor_tilized =
                    conv_act_requires_tile_padding_qsr(input_tensor)
                        ? ttnn::operations::experimental::quasar::to_layout(input_tensor, Layout::TILE)
                        : ttnn::operations::experimental::quasar::tilize(input_tensor);
                if (conv_config.deallocate_activation && !input_tensor.memory_config().is_dram()) {
                    input_tensor.deallocate(/*force*/ true);
                    input_tensor_tilized = ttnn::operations::experimental::quasar::move(input_tensor_tilized);
                }
                input_tensor = input_tensor_tilized;
            }
            if (!auto_shard_mm) {
                ttnn::MemoryConfig input_tensor_sharded_memory_config_to_layout = input_tensor_sharded_memory_config;
                tt::tt_metal::Alignment alignment = {};
                if (!input_tensor.is_sharded()) {
                    // In case we need to run Interleaved2Sharded, adjust the shard spec,
                    // in order to get smaller allocation size of sharded buffer.
                    const auto& shard_spec = input_tensor_sharded_memory_config.shard_spec().value();
                    input_tensor_sharded_memory_config_to_layout = tt::tt_metal::MemoryConfig(
                        input_tensor_sharded_memory_config_to_layout.memory_layout(),
                        input_tensor_sharded_memory_config_to_layout.buffer_type(),
                        tt::tt_metal::ShardSpec(shard_spec.grid, shard_spec.shape, shard_spec.orientation));
                    alignment = tt::tt_metal::Alignment{shard_spec.shape[0], shard_spec.shape[1]};
                }
                Tensor resharded_input_tensor = tt::tt_metal::create_device_tensor(
                    TensorSpec(
                        input_tensor.logical_shape(),
                        tt::tt_metal::TensorLayout(
                            input_tensor.dtype(),
                            tt::tt_metal::PageConfig(input_tensor.layout()),
                            input_tensor_sharded_memory_config_to_layout,
                            alignment)),
                    input_tensor.device());
                ttnn::operations::experimental::quasar::to_memory_config(
                    input_tensor, input_tensor_sharded_memory_config_to_layout, std::nullopt, resharded_input_tensor);
                if (conv_config.deallocate_activation && !input_tensor.memory_config().is_dram()) {
                    input_tensor.deallocate(/*force*/ true);
                    resharded_input_tensor = ttnn::operations::experimental::quasar::move(resharded_input_tensor);
                }
                input_tensor = resharded_input_tensor;
            }
        } else {
            input_tensor = ttnn::operations::experimental::quasar::to_device(
                input_tensor, device, (auto_shard_mm ? ttnn::DRAM_MEMORY_CONFIG : input_tensor_sharded_memory_config));
        }
    }
    return {input_tensor, parallel_config, output_parallel_config};
}

// Quasar variant of determine_matmul_op_config_from_conv_op_config (conv2d_utils): builds the
// QUASAR matmul program config (ttnn::operations::experimental::quasar::matmul) so the 1x1-conv
// GEMM runs on the quasar matmul/linear instead of the original. Mirrors the shared helper
// field-for-field; the config structs are identical copies.
static ttnn::operations::experimental::quasar::matmul::MatmulProgramConfig
determine_matmul_op_config_from_conv_op_config_qsr(
    Conv2dParallelizationConfig conv_parallelization_config,
    Conv2dBlockConfig conv_blocking_config,
    bool height_sharded,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    bool transpose_mcast,
    uint32_t /*grid_size_along_c*/) {
    namespace qmm = ttnn::operations::experimental::quasar::matmul;
    if (height_sharded) {
        qmm::MatmulMultiCoreReuseMultiCast1DProgramConfig matmul_config = {
            .compute_with_storage_grid_size = conv_parallelization_config.grid_size,
            .in0_block_w = conv_blocking_config.act_block_w_ntiles,
            .out_subblock_h = conv_blocking_config.out_subblock_h_ntiles,
            .out_subblock_w = conv_blocking_config.out_subblock_w_ntiles,
            .out_block_h = conv_parallelization_config.per_core_out_matrix_height_ntile,
            .out_block_w = conv_parallelization_config.per_core_out_matrix_width_ntile,
            .per_core_M = conv_parallelization_config.per_core_out_matrix_height_ntile,
            .per_core_N = conv_parallelization_config.per_core_out_matrix_width_ntile,
            .fuse_batch = true,
            .mcast_in0 = false};
        if (activation.has_value()) {
            matmul_config.fused_activation = activation.value();
        }
        return matmul_config;
    }
    qmm::MatmulMultiCoreReuseMultiCastProgramConfig matmul_config = {
        .compute_with_storage_grid_size = conv_parallelization_config.grid_size,
        .in0_block_w = conv_blocking_config.act_block_w_ntiles,
        .out_subblock_h = conv_blocking_config.out_subblock_h_ntiles,
        .out_subblock_w = conv_blocking_config.out_subblock_w_ntiles,
        .out_block_h = conv_parallelization_config.per_core_out_matrix_height_ntile,
        .out_block_w = conv_parallelization_config.per_core_out_matrix_width_ntile,
        .per_core_M = conv_parallelization_config.per_core_out_matrix_height_ntile,
        .per_core_N = conv_parallelization_config.per_core_out_matrix_width_ntile,
        .transpose_mcast = transpose_mcast};
    if (activation.has_value()) {
        matmul_config.fused_activation = activation.value();
    }
    return matmul_config;
}

Result conv2d_L1(
    const ttnn::Tensor& input_tensor_,
    const ttnn::Tensor& weight_tensor_,
    MeshDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor_,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const DataType output_dtype = dtype.value_or(input_tensor_.dtype());
    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    const auto& weight_tensor = weight_tensor_;
    std::optional<ttnn::Tensor> bias_tensor = bias_tensor_;
    bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    // Store the original stride size for weight folding
    auto orig_stride = stride;

    auto input_tensor = fold_input_tensor_if_required(
        input_tensor_,
        device,
        batch_size,
        input_height,
        input_width,
        in_channels,
        kernel_size,
        stride,
        dilation,
        padding_n4,
        mm_conv,
        conv_config);

    if (conv_config.enable_activation_reuse) {
        if (conv_config.enable_act_double_buffer) {
            conv_config.enable_act_double_buffer = false;
            log_debug(
                tt::LogOp,
                "Activation double buffering is currently not supported when activation reuse optimization is enabled, "
                "disabling double buffering.");
        }

        if (conv_config.enable_weights_double_buffer) {
            conv_config.enable_weights_double_buffer = false;
            log_debug(
                tt::LogOp,
                "Weights are already fully buffered when activation reuse optimization is enabled, disabling weights "
                "double buffering.");
        }
    }
    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);

    // Use weights_dtype from config if set, otherwise use weight tensor's dtype
    DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor_.dtype());
    DeviceComputeKernelConfig compute_config =
        compute_config_.value_or(get_conv_default_compute_kernel_config(device, input_tensor_.dtype(), weight_dtype));

    const auto compute_grid_size = device->compute_with_storage_grid_size();

    bool auto_shard = false;
    if (!input_tensor.is_sharded() && !conv_config.shard_layout.has_value()) {
        if (!conv_config.weights_dtype.has_value()) {
            conv_config.weights_dtype = weight_tensor.dtype();
        }
        // In this case we deduce the shard layout.
        conv_config = determine_conv_config_for_auto_shard(
            conv_config,
            mm_conv,
            batch_size,
            in_channels,
            out_channels,
            output_height,
            output_width,
            kernel_size[1],
            input_height,
            input_width,
            compute_grid_size,
            input_tensor.layout(),
            input_tensor.dtype(),
            output_dtype,
            tt::tt_metal::is_device_tensor(input_tensor) ? std::make_optional(input_tensor.memory_config())
                                                         : std::nullopt,
            kernel_size,
            stride,
            dilation,
            padding_n4,
            groups,
            bias_tensor.has_value(),
            compute_config);
        auto_shard = true;
    }
    const bool should_deallocate_act = conv_config.deallocate_activation && !input_tensor.memory_config().is_dram();
    auto [input_tensor_post_tm, parallel_config, output_parallel_config] = shard_or_reshard_tensor_if_required_qsr(
        device,
        input_tensor,
        conv_config,
        batch_size,
        output_height,
        output_width,
        in_channels,
        out_channels,
        mm_conv,
        auto_shard);

    const uint32_t input_channels_alignment = get_input_channels_alignment(
        input_tensor_post_tm.memory_config().memory_layout(),
        input_tensor_post_tm.layout(),
        false,
        mm_conv,
        input_tensor_post_tm.memory_config());
    const uint32_t in_channels_padded = tt::round_up(
        in_channels, get_num_cores_channels_from_parallel_config(parallel_config) * input_channels_alignment);

    const bool conv_is_1d_depthwise =
        is_1d_depthwise_conv(groups, in_channels, out_channels, kernel_size[0], input_height, bias_tensor.has_value());
    const bool coalesce_1d_depthwise_kw_reads = should_coalesce_1d_depthwise_conv_reads(
        conv_is_1d_depthwise,
        parallel_config.shard_scheme,
        in_channels_padded,
        kernel_size[1],
        dilation[1],
        input_tensor_post_tm.dtype());

    auto [opt_conv_op_parallel_config, opt_conv_op_block_config, conv_out_memory_config] = get_conv_configs(
        conv_config,
        compute_config,
        parallel_config,
        output_parallel_config,
        in_channels_padded,
        out_channels,
        batch_size,
        output_height,
        output_width,
        kernel_size,
        compute_grid_size,
        conv_is_1d_depthwise,
        coalesce_1d_depthwise_kw_reads);

    // ---- FIT-GUARDED Quasar "no-spill" (single K-block) conv path ----
    // The Quasar matmul-partials K-accumulate (RESTORE_PARTIALS_WR / QSR_RESTORE_WR g_dfb ring rewind in
    // conv_bmm_tilize_metal2.cpp) is a known-broken LLK path: the packer PACR0_TILE_INC in-place accumulate
    // mis-addresses the matmul_partials CB after the ring rewind and writes OOB in L1 (ERROR_TRISC1, opcode
    // 0x19). The reduction dim K is spilled into filter_h blocks (num_blocks_act_w = filter_h) only because a
    // height-sharded conv slices its inner dim by kernel row. When the WHOLE reduction dim
    // (filter_h*filter_w*in_channels) fits in one small K-block we can instead run a single matmul block, so
    // matmul_partials is never accumulated (compute kernel spill = in0_num_blocks_w > 1 stays false).
    //
    // Lever = full_inner_dim (same "don't slice the reduction dim" semantics block-sharded already uses):
    //   (i)   override block_config.act_block_w_ntiles to the FULL-K tile count (was one filter row),
    //   (ii)  conv_config.full_inner_dim = true  -> factory slice_inner_dim = false -> num_blocks_act_w = 1
    //         and the reader reads the full window per block,
    //   (iii) force the weight matrix into its full-inner-dim (single-block) layout below.
    // FIT GUARD: height-sharded, real conv (not 1x1-matmul, not depthwise), filter_h > 1 (so the sliced path
    // would actually spill), and the full single K-block is small enough to fit L1 / the uint16 DFB ring.
    // Deep/large-K convs (e.g. resnet layer2+ >32 K-tiles) fall through and keep the (validated) spilling path.
    // NOTE: this is Quasar-only (experimental/quasar/conv2d tree); the shared conv2d_utils / prepare_conv2d_weights
    // decisions are left untouched, so non-Quasar convs (which set full_inner_dim=true on height-sharded layers,
    // e.g. resnet50) are unaffected.
    constexpr uint32_t kQuasarConvNoSpillMaxKTiles = 32;  // stem full-K ~= 16 tiles; conservative L1/DFB headroom
    const bool height_sharded_conv = parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED;
    const uint32_t full_inner_dim_k_ntiles =
        tt::round_up(in_channels_padded * kernel_size[0] * kernel_size[1], tt::constants::TILE_WIDTH) /
        tt::constants::TILE_HEIGHT;
    // Quasar PIVOT (2026-07-15): the no-spill path packs the WHOLE reduction into ONE K-block, so the
    // in-kernel tilize_block runs over the full-K width (e.g. stem in0_block_w=16 tiles). That long tilize
    // trips the Quasar per-tile datacopy<->pack DEST-reuse race -> ERROR_TRISC1 0x19. Historically DISABLED on
    // Quasar in favor of K-SPILL + out_subblock 1x1.
    //
    // UPDATE (2026-07-17, split path): the datacopy tilize's 0x19 is now FIXED (per-tile FPU dest-dvalid clear,
    // tilize.h). In the SPLIT program (TT_METAL_QSR_CONV_SPLIT_PROGRAM) the tilize also runs in its OWN Metal
    // program (no interleaved matmul), so the full-K datacopy tilize is safe. And the split path REQUIRES
    // no-spill: with K-spill on Quasar, act_block_w stays a window-row but the reader gathers the full K per
    // row -> it writes act_block_h*full_K tiles into an act CB sized act_block_h*act_block_w -> 4x ACT-CB
    // OVERRUN (dprint_spe3: end-wptr=64 tiles vs nent=16) -> hang. So force no-spill on Quasar TOO when the
    // split is requested, matching the WH/BH config (act_block_w == full_K, reader gather == CB size).
    const bool arch_is_quasar = device->arch() == tt::ARCH::QUASAR;
    const bool split_env_requested = (std::getenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM") != nullptr);
    const bool force_conv_no_spill = (!arch_is_quasar || split_env_requested) && height_sharded_conv && !mm_conv &&
                                     !conv_is_1d_depthwise && (kernel_size[0] > 1) &&
                                     (full_inner_dim_k_ntiles <= kQuasarConvNoSpillMaxKTiles);
    if (force_conv_no_spill) {
        opt_conv_op_block_config.act_block_w_ntiles = full_inner_dim_k_ntiles;
        conv_config.full_inner_dim = true;

        // The no-spill path grows act_block_w from one filter row to the full K (filter_h*filter_w*Cin).
        // The ACT and ACT_TILIZED CBs are sized act_block_h * act_block_w tiles, so a large per-core output
        // height (act_block_h, e.g. the stem's 112 tiles) would push a CB's Quasar DFB TRISC ring extent
        // (capacity * entry_size, in 16-byte units) past the uint16_t limit (65536 units = 1 MB). Cap
        // act_block_h (more, smaller M-blocks: num_blocks_act_h grows) so the largest activation CB ring
        // stays under the limit. This splits only the OUTPUT-HEIGHT (M) axis; the reduction dim
        // (K / act_block_w / num_blocks_act_w) is untouched, so num_blocks_act_w stays 1 and NO
        // matmul-partials accumulate is reintroduced.
        const uint32_t act_in_units_per_tile =
            tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(input_tensor_post_tm.dtype())) / 16u;
        const uint32_t act_tilized_units_per_tile =
            tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(output_dtype)) / 16u;
        const uint32_t act_db = conv_config.enable_act_double_buffer ? 2u : 1u;
        // Worst-case ring units per act_block_h tile-row = act_block_w(full-K) * max(ACT CB, ACT_TILIZED CB).
        // ACT uses the input data format and may be double-buffered; ACT_TILIZED uses the output data format.
        const uint32_t act_db_units = act_db * act_in_units_per_tile;
        const uint32_t worst_units_per_h =
            full_inner_dim_k_ntiles *
            (act_db_units > act_tilized_units_per_tile ? act_db_units : act_tilized_units_per_tile);
        constexpr uint32_t kDfbRingUnitCap = 65535u;  // strictly below the 65536 uint16_t DFB ring limit
        const uint32_t per_core_h_ntiles = opt_conv_op_parallel_config.per_core_out_matrix_height_ntile;
        const uint32_t max_act_block_h =
            worst_units_per_h == 0 ? per_core_h_ntiles : (kDfbRingUnitCap / worst_units_per_h);
        uint32_t hi = max_act_block_h < per_core_h_ntiles ? max_act_block_h : per_core_h_ntiles;
        // Honor a user act_block_h_override (in rows) as an UPPER bound on the height block. QUASAR reader bug:
        // the no-spill full-window gather (read_activation_data) mis-reads output rows beyond ~4 M-tiles in a
        // SINGLE gather -> M-tiles >=4 come out wrong (tzrb_qsr: PCC by M-quarter 1,0,1,0). WH never hit it
        // (per_core_M=4). Capping act_block_h (=> more, smaller height blocks; the reader gathers each block
        // separately) keeps every gather inside the validated <=4-M-tile range. The real fix belongs in
        // read_activation_data; this bounds the gather until then.
        if (conv_config.act_block_h_override > 0) {
            const uint32_t override_tiles = conv_config.act_block_h_override / tt::constants::TILE_HEIGHT;
            if (override_tiles > 0 && override_tiles < hi) {
                hi = override_tiles;
            }
        }
        // Keep act_block_h a multiple of the (already-valid) out_subblock height AND a divisor of the per-core
        // output height, so the compute subblocking stays valid (factory asserts act_block_h % out_subblock_h
        // == 0) and no partial M-block is produced. out_subblock_h divides the per-core height, so it is
        // always a valid fallback.
        const uint32_t osh = opt_conv_op_block_config.out_subblock_h_ntiles;
        uint32_t capped_act_block_h = osh;
        for (uint32_t h = (hi / osh) * osh; h >= osh; h -= osh) {
            if (per_core_h_ntiles % h == 0) {
                capped_act_block_h = h;
                break;
            }
        }
        opt_conv_op_block_config.act_block_h_ntiles = capped_act_block_h;

        log_debug(
            tt::LogOp,
            "conv2d Quasar: no-spill full-inner-dim path (act_block_w_ntiles={} K-tiles, num_blocks_act_w=1) to "
            "avoid the broken matmul-partials K-accumulate; capped act_block_h_ntiles={} (per-core height {} "
            "tiles, num_blocks_act_h={}) to keep every activation CB DFB ring under the uint16_t limit.",
            full_inner_dim_k_ntiles,
            capped_act_block_h,
            per_core_h_ntiles,
            per_core_h_ntiles / capped_act_block_h);
    }

    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;

    // Configure weight and bias preparation parameters
    Conv2dWeightsBiasPrepConfig params(
        input_channels_alignment,
        conv_config.weights_dtype,
        opt_conv_op_block_config.act_block_w_ntiles,
        opt_conv_op_block_config.out_subblock_w_ntiles,
        parallel_config,
        output_parallel_config,
        groups,
        opt_conv_op_block_config.act_block_h_ntiles,
        input_height,
        input_width,
        mm_conv && auto_shard,
        out_channels,
        bias_tensor.has_value(),
        conv_config.enable_kernel_stride_folding.value(),
        conv_config.full_inner_dim,
        // Height-sharded weight layout picks full-inner-dim (single-block, [r][s][c] flattened, no per-kernel-row
        // tile padding) vs sliced-by-kernel-row via this activation-reuse arg (to_weight_special_padding_tile_layout).
        // For the no-spill path we need the full-inner-dim layout so it matches act_block_w_ntiles = full K and the
        // factory's single-block reader. We only borrow the layout here; enable_activation_reuse is NOT passed on to
        // the device op (line below), so the deferred split-reader reuse optimization stays off.
        conv_config.enable_activation_reuse || force_conv_no_spill,
        coalesce_1d_depthwise_kw_reads,
        orig_stride);

    // Prepare weights and move to device if necessary
    if (!is_device_tensor(weight_tensor)) {
        log_trace(tt::LogOp, "conv2d: Preprocessing weights on host and moving to device.");
        std::tie(weight_tensor_on_device, bias_tensor_on_device) =
            prepare_conv_weights_biases_and_move_to_device(weight_tensor, bias_tensor, params, device);
    } else {
        // Check if device weights are properly prepared
        if (is_valid_device_conv_weights(
                weight_tensor_on_device, in_channels, out_channels, conv_config.weights_dtype)) {
            log_debug(tt::LogOp, "conv2d: Using preprocessed weights from device.");
        } else {
            log_warning(
                tt::LogOp,
                "conv2d: Device weights not properly prepared, pulling back to host and trying to reprocess.");
            // Pull weights back to host, prepare them, and push back to device
            ttnn::Tensor host_weight_tensor = ttnn::operations::core::from_device(weight_tensor_on_device);
            std::tie(weight_tensor_on_device, bias_tensor_on_device) =
                prepare_conv_weights_biases_and_move_to_device(host_weight_tensor, bias_tensor, params, device);
        }
    }

    // Prepare bias tensor if it exists and is not yet on device
    if (bias_tensor_on_device.has_value()) {
        if (!is_device_tensor(bias_tensor_on_device.value())) {
            log_trace(tt::LogOp, "conv2d: Preprocessing bias on host and moving to device.");

            bias_tensor_on_device = prepare_conv_bias_internal(
                bias_tensor_on_device, out_channels, params, weight_tensor_on_device.dtype(), device);
        } else {
            // Check if device bias is properly prepared
            if (is_valid_device_conv_bias(bias_tensor_on_device.value(), out_channels, conv_config.weights_dtype)) {
                log_debug(tt::LogOp, "conv2d: Using preprocessed bias from device.");
            } else {
                log_warning(
                    tt::LogOp, "conv2d: Device bias not properly prepared, pulling back to host and reprocessing.");
                // Pull bias back to host, prepare it, and push back to device
                ttnn::Tensor host_bias_tensor = ttnn::operations::core::from_device(bias_tensor_on_device.value());
                bias_tensor_on_device = prepare_conv_bias_internal(
                    std::optional<const ttnn::Tensor>(host_bias_tensor),
                    out_channels,
                    params,
                    weight_tensor_on_device.dtype(),
                    device);
            }
        }
    }

    // call conv op or matmul micro op
    bool input_is_on_device = tt::tt_metal::is_device_tensor(input_tensor_post_tm);
    TT_ASSERT(input_is_on_device);

    if (!mm_conv) {
        // call halo op
        sliding_window::SlidingWindowConfig sliding_window_config = sliding_window::SlidingWindowConfig{
            .batch_size = batch_size,
            .input_hw = {input_height, input_width},
            .window_hw = {kernel_size[0], kernel_size[1]},
            .stride_hw = {stride[0], stride[1]},
            .padding = {{padding_n4[0], padding_n4[1], padding_n4[2], padding_n4[3]}},
            .dilation_hw = {dilation[0], dilation[1]},
            .num_cores_nhw = opt_conv_op_parallel_config.num_cores_nhw,
            .core_range_set = input_tensor_post_tm.memory_config().shard_spec().value().grid,
            .snap_to_tile = true,
            .padding_mode = conv_config.padding_mode,
        };

        if (parallel_config.shard_scheme != TensorMemoryLayout::WIDTH_SHARDED ||
            input_tensor_post_tm.layout() != Layout::ROW_MAJOR || sliding_window_config.get_pad_h() != 0 ||
            sliding_window_config.get_pad_w() != 0) {
            ttnn::Tensor halo_output = ttnn::operations::experimental::quasar::halo(
                input_tensor_post_tm,
                sliding_window_config,
                compute_config,
                0,
                false,
                parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
                true,
                conv_config.config_tensors_in_dram);

            // In cases where input tensor is in DRAM and it gets sharded, we need to deallocate the sharded input
            // tensor at this point (it will be deallocated automatically because nothing is using it, but reallocating
            // halo output will be affected so we need to deallocate it manually before reallocating halo output)
            if (conv_config.deallocate_activation && !input_tensor_post_tm.memory_config().is_dram()) {
                input_tensor_post_tm.deallocate(/*force*/ true);
            }

            input_tensor_post_tm = std::move(halo_output);

            if (conv_config.reallocate_halo_output) {
                input_tensor_post_tm = ttnn::operations::experimental::quasar::move(input_tensor_post_tm);
            }
        }

        const std::array<std::uint32_t, 4> input_tensor_shape = {
            batch_size,
            input_height,
            input_width,
            in_channels,
        };

        // call conv micro op
        auto conv_output = ttnn::prim::qsr::conv2d(
            input_tensor_post_tm,
            weight_tensor_on_device,
            bias_tensor_on_device,
            sliding_window_config,
            out_channels,
            groups,
            conv_config.output_layout == Layout::ROW_MAJOR,
            conv_config.activation,
            opt_conv_op_parallel_config,
            opt_conv_op_block_config,
            conv_out_memory_config,
            output_dtype,
            input_tensor_shape,
            compute_config,
            conv_config.enable_act_double_buffer,
            conv_config.enable_weights_double_buffer,
            conv_config.full_inner_dim,
            conv_config.enable_activation_reuse,
            conv_config.config_tensors_in_dram,
            conv_config.force_split_reader);

        // OPTION B — PROGRAM B (two-program split). Under TT_METAL_QSR_CONV_SPLIT_PROGRAM the conv op above ran
        // TILIZE-ONLY (Program A: reader gather + UnpackToDestEn tilize) and `conv_output` is the tilized
        // activation [M, K] (K = in0_block_w). The tilize and matmul can't share one kernel on Quasar (the
        // dvalid-synced tilize + semaphore-synced matmul re-fault the 0x19), so the matmul runs here as a
        // SEPARATE op: conv_output @ weights -> conv result [M, N], reusing the same quasar matmul::linear the
        // 1x1-conv path uses (bias + activation folded into the matmul program config). Gate must match the
        // factory's split_program_tilize_only eligibility (height-sharded + full_inner_dim single-K-block; this
        // is the !mm_conv, non-depthwise branch already).
        // DIAGNOSTIC (tilize isolation): TT_METAL_QSR_CONV_TILIZE_ONLY_NO_MATMUL stops after Program A and returns
        // the tilized activation [M, K] ITSELF (skips the Program B matmul), so a test can read it back and diff
        // against a host golden — isolating the UnpackToDestEn tilize from the matmul. Program A still ran
        // tilize-only above (the factory keys off TT_METAL_QSR_CONV_SPLIT_PROGRAM). See test_conv2d_tilize_readback.py.
        const bool tilize_only_no_matmul = (std::getenv("TT_METAL_QSR_CONV_TILIZE_ONLY_NO_MATMUL") != nullptr);
        // NOTE: the arch==QUASAR restriction was REMOVED so the split runs on WH/BH too (bring-up/validation with
        // working LLK). It MUST match the sharded factory's split_program_tilize_only gate, which is env-only (no
        // arch check) — otherwise the factory builds the tilize-only Program A but conv2d.cpp skips Program B, and
        // the op returns the raw tilized activation [M,K] instead of the conv result [M,N]. The env var is the
        // explicit opt-in (only tests set it), so production convs on any arch are unaffected.
        const bool split_program_active = (std::getenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM") != nullptr) &&
                                          parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED &&
                                          conv_config.full_inner_dim && !tilize_only_no_matmul;
        if (split_program_active) {
            std::optional<ttnn::operations::experimental::quasar::matmul::MatmulProgramConfig> program_config =
                std::nullopt;
            std::optional<MemoryConfig> mm_output_memory_config = std::nullopt;
            if (conv_output.is_sharded()) {
                uint32_t num_cores_c = get_num_cores_channels_from_parallel_config(parallel_config);
                program_config = determine_matmul_op_config_from_conv_op_config_qsr(
                    opt_conv_op_parallel_config,
                    opt_conv_op_block_config,
                    parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
                    conv_config.activation,
                    parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
                    num_cores_c);
                // QUASAR: Program A tilized the FULL im2col K (act_block_w * filter_h) contiguously into
                // conv_output. The linear MUST contract it in ONE K-block (in0_block_w == full_K -> num_blocks
                // == 1) to avoid the Quasar matmul interm0/mm-partials K-SPILL ACCUMULATE, which faults (the
                // same constraint that made test_linear pass with a single K-block). The shared helper sets
                // in0_block_w = act_block_w_ntiles (== full_K only for 1x1, filter_h=1); here full_K =
                // act_block_w * filter_h, so override. mcast_in0 is false on the 1D height-sharded path, so K is
                // NOT width-sharded across the grid -> in0_block_w == full_K really does give num_blocks == 1.
                // Program B contracts the FULL K in ONE block (num_blocks == 1) to dodge the Quasar matmul
                // K-spill accumulate. in0_block_w MUST be the intrinsic full K (= in_ch_padded*kh*kw/32), NOT
                // act_block_w * kernel_size[0]: on WH/BH act_block_w is ALREADY the full K, so multiplying by
                // kernel_size[0] over-counted 4x and made Program B's K (64) disagree with the weights' K (16).
                // full_inner_dim_k_ntiles is that intrinsic K on every arch.
                const uint32_t full_k_ntiles_mm = full_inner_dim_k_ntiles;
                // The helper always builds a MatmulMultiCoreReuseMultiCast1DProgramConfig (height-sharded, 1D
                // mcast_in0=false). Only that alternative carries in0_block_w, so target it directly rather than a
                // std::visit lambda (which would have to compile for every variant, most of which lack the field).
                auto* mm1d_cfg = std::get_if<
                    ttnn::operations::experimental::quasar::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
                    &program_config.value());
                TT_FATAL(
                    mm1d_cfg != nullptr,
                    "QUASAR split conv Program B expected a MatmulMultiCoreReuseMultiCast1DProgramConfig");
                mm1d_cfg->in0_block_w = full_k_ntiles_mm;

                // Program B's OUTPUT is [M, N] (N = out_channels), but the helper derived per_core_N / out_block_w
                // from per_core_out_matrix_width_ntile, which on the split path is the tilized-activation width K
                // (16 tiles), NOT N (out_channels tiles). Uncorrected, the matmul writes a K-wide output and the
                // real N columns are scattered -> correct values, scrambled layout (sorted-flat PCC~1, exact~0,
                // raw out.shape [1,1,512,512] instead of [1,1,512,64]). Override the N dims to true out_channels
                // and pick a valid out_subblock_w divisor (<=8, DEST limit).
                const uint32_t n_ntiles_mm =
                    tt::round_up(out_channels, tt::constants::TILE_WIDTH) / tt::constants::TILE_WIDTH;
                // Cap out_block_w to a divisor <= 4 of per_core_N. The Quasar FUSED-bias matmul stalls at
                // program completion when out_block_w is large in a single N-block (N=8 tiles / out_channels=256
                // + bias HANGS; N<=4 passes, pure N=8 passes). out_block_w<=4 splits large N into multiple 4-wide
                // blocks (num_blocks_x>1) == the working N=4 geometry, sidestepping the wide-N fused-bias epilogue
                // stall (the pack_untilize_dest<obw,obw> + bias path). out_subblock_w = out_block_w (1 x obw <= 8).
                uint32_t obw_mm = n_ntiles_mm;
                while (obw_mm > 4 || (n_ntiles_mm % obw_mm) != 0) {
                    obw_mm--;
                }
                mm1d_cfg->per_core_N = n_ntiles_mm;
                mm1d_cfg->out_block_w = obw_mm;
                mm1d_cfg->out_subblock_w = obw_mm;
                mm1d_cfg->out_subblock_h = 1;  // 1 x obw_mm subblock is always valid (<=8) and correct

                // Build the matching [M, N] output memory config: same grid/orientation as Program A's [M, K]
                // output, but shard WIDTH = N (out_channels padded), not K. Reusing conv_out_memory_config (the
                // [M, K] tilized-activation config) is what forced the K-wide output.
                const auto& a_shard = conv_out_memory_config.shard_spec().value();
                const std::array<uint32_t, 2> mm_shard_shape = {
                    a_shard.shape[0], n_ntiles_mm * tt::constants::TILE_WIDTH};
                mm_output_memory_config = tt::tt_metal::MemoryConfig(
                    conv_out_memory_config.memory_layout(),
                    conv_out_memory_config.buffer_type(),
                    tt::tt_metal::ShardSpec{a_shard.grid, mm_shard_shape, a_shard.orientation});
            }
            ttnn::Tensor matmul_output = ttnn::operations::experimental::quasar::matmul::linear(
                conv_output,  // Program A output = tilized activation [M, K]
                weight_tensor_on_device,
                bias_tensor_on_device,
                false,
                false,
                mm_output_memory_config,
                output_dtype,
                program_config,
                conv_output.is_sharded() ? std::nullopt : conv_config.activation,
                compute_config);
            if (memory_config.has_value() && memory_config.value() != matmul_output.memory_config()) {
                matmul_output = ttnn::operations::experimental::quasar::to_memory_config(
                    matmul_output, memory_config.value(), std::nullopt);
            }
            return {matmul_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
        }

        if (memory_config.has_value() && memory_config.value() != conv_output.memory_config()) {
            conv_output = ttnn::operations::experimental::quasar::to_memory_config(
                conv_output, memory_config.value(), std::nullopt);
        }
        return {conv_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    }  // Matmul expects inputs to be in Tile Layout
    tilize_with_optional_deallocation_qsr(input_tensor_post_tm, should_deallocate_act);

    // run conv as matmul
    std::optional<ttnn::operations::experimental::quasar::matmul::MatmulProgramConfig> program_config = std::nullopt;
    std::optional<MemoryConfig> mm_output_memory_config = std::nullopt;

    if (input_tensor_post_tm.is_sharded()) {
        uint32_t num_cores_c = get_num_cores_channels_from_parallel_config(parallel_config);
        program_config = determine_matmul_op_config_from_conv_op_config_qsr(
            opt_conv_op_parallel_config,
            opt_conv_op_block_config,
            parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
            conv_config.activation,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            num_cores_c);
        mm_output_memory_config = conv_out_memory_config;
    }

    ttnn::Tensor matmul_output = ttnn::operations::experimental::quasar::matmul::linear(
        input_tensor_post_tm,
        weight_tensor_on_device,
        bias_tensor_on_device,
        false,
        false,
        mm_output_memory_config,
        output_dtype,
        program_config,
        // for sharded input, activation is set on program config
        input_tensor_post_tm.is_sharded() ? std::nullopt : conv_config.activation,
        compute_config);

    if (should_deallocate_act) {
        input_tensor_post_tm.deallocate(/*force*/ true);
    }
    if (memory_config.has_value() && memory_config.value() != matmul_output.memory_config()) {
        matmul_output = ttnn::operations::experimental::quasar::to_memory_config(
            matmul_output, memory_config.value(), std::nullopt);
    }

    return {matmul_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
}

ResultWithOptions result_to_result_with_options(
    const Result& result, const bool return_output_dim, const bool return_weights_and_bias) {
    if (return_output_dim && return_weights_and_bias) {
        return std::make_tuple(
            std::get<0>(result),
            std::make_tuple(std::get<1>(result), std::get<2>(result)),
            std::make_tuple(std::get<3>(result), std::get<4>(result)));
    }
    if (return_output_dim) {
        return std::make_tuple(std::get<0>(result), std::make_tuple(std::get<1>(result), std::get<2>(result)));
    }
    if (return_weights_and_bias) {
        return std::make_tuple(std::get<0>(result), std::make_tuple(std::get<3>(result), std::get<4>(result)));
    }
    return std::get<0>(result);
}

class Conv2dSliceAttr : public ttnn::operations::op_slicing::OpSliceAttr {
    using OptionalRefTensor = std::optional<std::reference_wrapper<ttnn::Tensor>>;

    Conv2dConfig auto_slice_conv_config;
    uint32_t batch_size;
    IOShape input_shape;
    uint32_t input_channels;
    uint32_t output_channels;
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 2> stride;
    std::array<uint32_t, 4> padding_n4;
    std::array<uint32_t, 2> dilation;
    uint32_t groups;
    tt::tt_metal::Layout input_layout;
    tt::tt_metal::DataType input_dtype;
    tt::tt_metal::DataType output_dtype;
    Tensor& weight_tensor;
    OptionalRefTensor bias_tensor;
    Conv2dConfig conv_config;
    DeviceComputeKernelConfig compute_config;
    MeshDevice* device;

public:
    Conv2dSliceAttr(
        uint32_t batch_size,
        IOShape input_shape,
        uint32_t input_channels,
        uint32_t output_channels,
        std::array<uint32_t, 2> kernel_size,
        std::array<uint32_t, 2> stride,
        std::array<uint32_t, 4> padding_n4,
        std::array<uint32_t, 2> dilation,
        uint32_t groups,
        tt::tt_metal::Layout input_layout,
        tt::tt_metal::DataType input_dtype,
        tt::tt_metal::DataType output_dtype,
        Tensor& weight_tensor,
        OptionalRefTensor bias_tensor,
        const Conv2dConfig& conv_config,
        const DeviceComputeKernelConfig& compute_config,
        MeshDevice* device) :
        batch_size(batch_size),
        input_shape(input_shape),
        input_channels(input_channels),
        output_channels(output_channels),
        kernel_size(kernel_size),
        stride(stride),
        padding_n4(padding_n4),
        dilation(dilation),
        groups(groups),
        input_layout(input_layout),
        input_dtype(input_dtype),
        output_dtype(output_dtype),
        weight_tensor(weight_tensor),
        bias_tensor(bias_tensor),
        conv_config(conv_config),
        compute_config(compute_config),
        device(device) {}

    std::tuple<std::tuple<IOShape, IOShape>, std::array<uint32_t, 4>> get_input_slice_and_padding(
        const IOShape& output_slice_start, const IOShape& output_slice_end) const {
        auto [output_slice_height_start, output_slice_width_start] = output_slice_start;
        auto [output_slice_height_end, output_slice_width_end] = output_slice_end;
        auto [input_height, input_width] = input_shape;

        // Calculate required input slice range based on output slice
        // Formula: input_start = (output_start * stride) - padding
        // Formula: input_end = ((output_end - 1) * stride) - padding + dilated_kernel_size
        int input_slice_height_start = (output_slice_height_start * stride[0]) - padding_n4[0];
        int input_slice_height_end = ((output_slice_height_end - 1) * stride[0]) - padding_n4[0] +
                                     ((kernel_size[0] - 1) * (dilation[0] - 1)) + kernel_size[0];
        int input_slice_width_start = (output_slice_width_start * stride[1]) - padding_n4[2];
        int input_slice_width_end = ((output_slice_width_end - 1) * stride[1]) - padding_n4[2] +
                                    ((kernel_size[1] - 1) * (dilation[1] - 1)) + kernel_size[1];

        // Calculate padding needed if input slice extends beyond input tensor
        uint32_t pad_top = std::max<int>(0, -input_slice_height_start);
        uint32_t pad_bottom = std::max<int>(0, input_slice_height_end - input_height);
        uint32_t pad_left = std::max<int>(0, -input_slice_width_start);
        uint32_t pad_right = std::max<int>(0, input_slice_width_end - input_width);

        // Clamp input slice to valid input tensor bounds
        input_slice_height_start = std::max<int>(0, input_slice_height_start);
        input_slice_height_end = std::min<int>(input_height, input_slice_height_end);
        input_slice_width_start = std::max<int>(0, input_slice_width_start);
        input_slice_width_end = std::min<int>(input_width, input_slice_width_end);

        // Calculate full output dimensions
        auto [output_height, output_width] = calculate_output_image_size(
            std::array<uint32_t, 2>{input_height, input_width}, kernel_size, stride, padding_n4, dilation);

        // Special handling for edges: if output slice starts/ends at tensor boundary,
        // use the full original padding and reset input slice to tensor boundary
        if (output_slice_height_start == 0) {
            pad_top = padding_n4[0];
            input_slice_height_start = 0;
        }
        if (output_slice_height_end == output_height) {
            pad_bottom = padding_n4[1];
            input_slice_height_end = input_height;
        }
        if (output_slice_width_start == 0) {
            pad_left = padding_n4[2];
            input_slice_width_start = 0;
        }
        if (output_slice_width_end == output_width) {
            pad_right = padding_n4[3];
            input_slice_width_end = input_width;
        }
        uint32_t input_slice_height = input_slice_height_end - input_slice_height_start;
        uint32_t input_slice_width = input_slice_width_end - input_slice_width_start;
        uint32_t output_slice_width = output_slice_width_end - output_slice_width_start;
        // Apply width rounding and adjust right padding if necessary
        uint32_t width_rounding_value =
            (conv_config.output_layout == tt::tt_metal::Layout::TILE) ? tt::constants::TILE_HEIGHT : 1;

        bool single_slice =
            (input_slice_height == std::get<0>(input_shape)) && (input_slice_width == std::get<1>(input_shape));

        if (output_slice_width % width_rounding_value != 0 && !single_slice) {
            uint32_t additional_padded_width = width_rounding_value - (output_slice_width % width_rounding_value);
            log_trace(
                tt::LogOp,
                "Conv2d DRAM Slicing: Additional padding of {} added to the right side.",
                additional_padded_width);
            pad_right += additional_padded_width * stride[1];  // Adjust right padding
        }

        return {
            {{input_slice_height_start, input_slice_width_start}, {input_slice_height_end, input_slice_width_end}},
            {pad_top, pad_bottom, pad_left, pad_right}};
    }

    std::tuple<IOShape, IOShape> get_input_slice(
        const IOShape& output_slice_start, const IOShape& output_slice_end) const override {
        return std::get<0>(get_input_slice_and_padding(output_slice_start, output_slice_end));
    }

    uint32_t get_L1_usage(
        const IOShape& output_slice_start,
        const IOShape& output_slice_end,
        const op_slicing::Op2DSliceConfig& slice_config) const override {
        // Remove this->conv_config from scope so that for each slice, conv_config can be calculated independently.
        auto conv_config = this->conv_config;
        bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
        TT_FATAL(!mm_conv, "Conv2D DRAM with matmul should never use the slicing code path.");

        auto [input_slicing, slice_padding] = get_input_slice_and_padding(output_slice_start, output_slice_end);
        auto [input_slice_start, input_slice_end] = input_slicing;
        auto [input_slice_height_start, input_slice_width_start] = input_slice_start;
        auto [input_slice_height_end, input_slice_width_end] = input_slice_end;
        auto input_slice_height = input_slice_height_end - input_slice_height_start;
        auto input_slice_width = input_slice_width_end - input_slice_width_start;

        auto [output_slice_height, output_slice_width] = calculate_output_image_size(
            {input_slice_height, input_slice_width}, kernel_size, stride, slice_padding, dilation);
        auto compute_grid = device->compute_with_storage_grid_size();
        log_trace(
            tt::LogOp,
            "Conv2D DRAM Auto Slice Max Input Size : {}x{}, Max Output Size : {}x{}",
            input_slice_height,
            input_slice_width,
            output_slice_height,
            output_slice_width);

        auto sliced_input_tensor_memory_config = get_input_memory_config(output_slice_start, output_slice_end);
        if (!conv_config.shard_layout.has_value()) {
            conv_config.shard_layout = sliced_input_tensor_memory_config.memory_layout();
        }
        auto conv_L1_usage = calculate_L1_usage_for_conv_op(
            batch_size,
            input_channels,
            output_channels,
            input_slice_height,
            input_slice_width,
            output_slice_height,
            output_slice_width,
            kernel_size,
            stride,
            slice_padding,
            dilation,
            groups,
            bias_tensor.has_value(),
            input_dtype,
            output_dtype,
            input_layout,
            compute_grid,
            false,
            conv_config.shard_layout.value(),
            compute_config,
            conv_config,
            sliced_input_tensor_memory_config);

        log_trace(
            tt::LogOp,
            "Conv DRAM Auto slicing: num_slices = {}, input_memory_config = {}, L1 usage = {}",
            slice_config.num_slices,
            sliced_input_tensor_memory_config,
            conv_L1_usage);
        return std::max(conv_L1_usage.halo_input_size + conv_L1_usage.halo_output_size, conv_L1_usage.total_size);
    }

    tt::tt_metal::MemoryConfig get_input_memory_config(
        const IOShape& output_slice_start, const IOShape& output_slice_end) const override {
        auto compute_grid_size = device->compute_with_storage_grid_size();
        auto conv_config = this->conv_config;

        auto [input_slicing, slice_padding] = get_input_slice_and_padding(output_slice_start, output_slice_end);
        auto [input_start, input_end] = input_slicing;
        uint32_t input_slice_height = std::get<0>(input_end) - std::get<0>(input_start);
        uint32_t input_slice_width = std::get<1>(input_end) - std::get<1>(input_start);
        // Use padded output dimensions to match what the halo op actually produces.
        // The halo output is tile-aligned, so edge slices get additional padding
        // (e.g., output width 4 pads to 32). Without this, the shard spec is computed
        // for the unpadded dimensions, leading to L1 underestimation.
        auto [output_slice_height, output_slice_width] = calculate_output_image_size(
            {input_slice_height, input_slice_width}, kernel_size, stride, slice_padding, dilation);

        bool single_slice =
            (input_slice_height == std::get<0>(input_shape)) && (input_slice_width == std::get<1>(input_shape));

        if (!conv_config.shard_layout.has_value()) {
            if (!conv_config.weights_dtype.has_value()) {
                conv_config.weights_dtype = weight_tensor.dtype();
            }
            conv_config = determine_conv_config_for_auto_shard(
                conv_config,
                false,
                batch_size,
                input_channels,
                output_channels,
                output_slice_height,
                output_slice_width,
                weight_tensor.logical_shape()[3],
                input_slice_height,
                input_slice_width,
                device->compute_with_storage_grid_size(),
                input_layout,
                input_dtype,
                output_dtype,
                std::nullopt,
                kernel_size,
                stride,
                dilation,
                padding_n4,
                groups,
                bias_tensor.has_value(),
                compute_config);
        }
        TT_FATAL(conv_config.shard_layout.has_value(), " Conv2D DRAM Slicing must have a shard layout set.");

        ShardOrientation shard_orientation =
            conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;
        auto sliced_input_tensor_memory_config = std::get<1>(determine_input_memory_config(
            conv_config.shard_layout.value(),
            shard_orientation,
            batch_size,
            ttnn::Shape({batch_size, input_slice_height, input_slice_width, input_channels}),
            ttnn::Shape({batch_size, output_slice_height, output_slice_width, output_channels}),
            false,
            compute_grid_size,
            input_layout,
            single_slice ? BufferType::L1 : BufferType::DRAM));
        return sliced_input_tensor_memory_config;
    }

    std::string name() const override { return "Conv2D"; }

    std::vector<ttnn::Tensor> run_L1_op(
        const ttnn::Tensor& sliced_input_tensor,
        const IOShape& output_slice_start,
        const IOShape& output_slice_end) override {
        // Use helper function to calculate slice bounds and padding
        auto [input_slicing, this_op_padding] = get_input_slice_and_padding(output_slice_start, output_slice_end);
        auto [input_slice_start, input_slice_end] = input_slicing;
        auto [input_slice_height_start, input_slice_width_start] = input_slice_start;
        auto [input_slice_height_end, input_slice_width_end] = input_slice_end;
        // Calculate dimensions directly from result
        uint32_t input_slice_height = input_slice_height_end - input_slice_height_start;
        uint32_t input_slice_width = input_slice_width_end - input_slice_width_start;

        if (!conv_config.shard_layout.has_value() && sliced_input_tensor.is_sharded()) {
            conv_config.shard_layout = sliced_input_tensor.memory_config().memory_layout();
        }
        auto conv_config_l1 = conv_config;

        conv_config_l1.deallocate_activation = true;
        conv_config_l1.reallocate_halo_output = true;

        // Force Conv2d_L1 to always output tiled layout to reduce CB Memory usage.
        conv_config_l1.output_layout = Layout::TILE;

        auto conv2d_result = conv2d_L1(
            sliced_input_tensor,
            weight_tensor,
            device,
            input_channels,
            output_channels,
            batch_size,
            input_slice_height,
            input_slice_width,
            kernel_size,
            stride,
            this_op_padding,
            dilation,
            groups,
            output_dtype,
            bias_tensor,
            conv_config_l1,
            compute_config,
            std::nullopt);
        weight_tensor = std::get<3>(conv2d_result);
        if (bias_tensor.has_value()) {
            bias_tensor->get() = std::get<4>(conv2d_result).value();
        }
        return {std::get<0>(conv2d_result)};
    }
};

// This function is used for DRAM Slicing
// It divides the output tensor into slices, and calculates the corresponding input slices.
// Uses ttnn::slice to slice the input tensor and bring it to L1.
// Calls conv2d_L1 to perform the convolution on the sliced input tensor.
// Finally, it uses ttnn::experimental::slice_write to write the output tensor back to DRAM.
// The function is called in a loop for each slice of the output tensor.
// The Conv2dSliceConfig is used to determine the slicing configuration. The dimension along which it is sliced, and the
// number of such slices.
// Conv2dConfig does not control the final output, but rather the conv2d_L1 function that is called internally.
Result conv2d_DRAM(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config_,
    const std::optional<const Conv2dSliceConfig>& dram_slice_config_) {
    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const DataType output_dtype = dtype.value_or(input_tensor.dtype());
    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    // Use weights_dtype from config if set, otherwise use weight tensor's dtype
    DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor.dtype());
    DeviceComputeKernelConfig compute_config =
        compute_config_.value_or(get_conv_default_compute_kernel_config(device, input_tensor.dtype(), weight_dtype));
    TT_FATAL(
        !conv_config.override_output_sharding_config,
        "Conv2D DRAM slicing doesn't support override_output_sharding_config.");

    // Fold the input tensor if required - this may update mm_conv after folding
    ttnn::Tensor input_tensor_on_device = fold_input_tensor_if_required(
        input_tensor,
        device,
        batch_size,
        input_height,
        input_width,
        in_channels,
        kernel_size,
        stride,
        dilation,
        padding_n4,
        mm_conv,
        conv_config);
    if (!is_device_tensor(input_tensor_on_device)) {
        input_tensor_on_device =
            ttnn::operations::core::to_device(input_tensor_on_device, device, ttnn::DRAM_MEMORY_CONFIG);
    }

    // After folding, check if this can be implemented as matmul and delegate to conv2d_L1
    // Note: mm_conv may have been updated by fold_input_tensor_if_required
    if (mm_conv) {
        return conv2d_L1(
            input_tensor_on_device,
            weight_tensor,
            device,
            in_channels,
            out_channels,
            batch_size,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding_n4,
            dilation,
            groups,
            output_dtype,
            bias_tensor,
            conv_config,
            compute_config_,
            memory_config_);
    }

    // DRAM slicing path - only executed when mm_conv is false
    const bool should_deallocate_act = conv_config.deallocate_activation && !input_tensor.memory_config().is_dram();
    ttnn::Tensor weight_tensor_on_device;
    std::optional<ttnn::Tensor> bias_tensor_on_device;
    if (memory_config_.has_value()) {
        log_warning(
            tt::LogOp,
            "Conv2D DRAM doesn't support specifying memory config, as the output will always be DRAM Interleaved");
    }

    TT_FATAL(
        !(conv_config.output_layout == Layout::ROW_MAJOR && output_dtype == DataType::BFLOAT8_B),
        "Conv output can't be in Row Major if output dtype is BFloat8_B.");

    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);

    if (!conv_config.weights_dtype.has_value()) {
        conv_config.weights_dtype = weight_tensor.dtype();
    }

    const auto unflattened_input_shape = ttnn::Shape{batch_size, input_height, input_width, in_channels};
    input_tensor_on_device = ttnn::operations::experimental::quasar::reshape(
        input_tensor_on_device, unflattened_input_shape, unflattened_input_shape);
    TT_FATAL(input_tensor_on_device.memory_config().is_dram(), "Conv DRAM expects the input tensor to be in DRAM.");
    TT_FATAL(
        input_tensor_on_device.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input Tensor to Conv DRAM should be in Interleaved Memory Layout");

    ttnn::Tensor dram_output_tensor = tt::tt_metal::create_device_tensor(
        tt::tt_metal::TensorSpec(
            ttnn::Shape({batch_size, output_height, output_width, out_channels}),
            tt::tt_metal::TensorLayout(
                output_dtype,
                tt::tt_metal::PageConfig(conv_config.output_layout),
                MemoryConfig{
                    TensorMemoryLayout::INTERLEAVED,
                    BufferType::DRAM,
                })),
        device);

    weight_tensor_on_device = weight_tensor;
    bias_tensor_on_device = bias_tensor;
    auto slice_attr = Conv2dSliceAttr(
        batch_size,
        {input_height, input_width},
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding_n4,
        dilation,
        groups,
        input_tensor.layout(),
        input_tensor.dtype(),
        output_dtype,
        std::ref(weight_tensor_on_device),
        bias_tensor_on_device.has_value() ? std::make_optional(std::ref(bias_tensor_on_device.value())) : std::nullopt,
        conv_config,
        compute_config,
        device);

    std::vector<std::reference_wrapper<Tensor>> output_tensors = {std::ref(dram_output_tensor)};
    ttnn::operations::op_slicing::run_sliced_op(
        input_tensor_on_device, output_tensors, &slice_attr, dram_slice_config_);

    if (should_deallocate_act) {
        input_tensor_on_device.deallocate(true);
    }
    const auto flattened_output_shape = flatten_4d_shape(dram_output_tensor.logical_shape());
    const auto flattened_padded_output_shape = flatten_4d_shape(dram_output_tensor.padded_shape());

    dram_output_tensor = ttnn::operations::experimental::quasar::reshape(
        dram_output_tensor, flattened_output_shape, flattened_padded_output_shape);

    return {dram_output_tensor, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
}

}  // namespace ttnn::operations::experimental::quasar::detail

namespace ttnn::operations::experimental::quasar {

Conv2dResultWithOptions conv2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    const std::optional<const DataType>& dtype,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const DeviceComputeKernelConfig>& compute_config_,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const Conv2dSliceConfig>& slice_config_,
    bool return_output_dim,
    bool return_weights_and_bias) {
    using namespace detail;
    using operations::conv::Conv2dExecutionPath;
    using operations::conv::determine_conv2d_execution_path;
    // Determine execution path based on configuration and input properties
    Conv2dExecutionPath path = determine_conv2d_execution_path(input_tensor, slice_config_);

    // Execute L1 path
    if (path == Conv2dExecutionPath::L1) {
        log_trace(tt::LogOp, "Conv2d L1 {}", slice_config_.has_value() ? "with slice config" : "without slice config");
        return result_to_result_with_options(
            conv2d_L1(
                input_tensor,
                weight_tensor,
                device,
                in_channels,
                out_channels,
                batch_size,
                input_height,
                input_width,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                dtype,
                bias_tensor,
                conv_config_,
                compute_config_,
                memory_config),
            return_output_dim,
            return_weights_and_bias);
    }

    // Execute DRAM path
    log_trace(tt::LogOp, "Conv2d DRAM {}", slice_config_.has_value() ? "with slice config" : "without slice config");
    return result_to_result_with_options(
        conv2d_DRAM(
            input_tensor,
            weight_tensor,
            device,
            in_channels,
            out_channels,
            batch_size,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            dtype,
            bias_tensor,
            conv_config_,
            compute_config_,
            memory_config,
            slice_config_),
        return_output_dim,
        return_weights_and_bias);
}

}  // namespace ttnn::operations::experimental::quasar

namespace ttnn::operations::experimental::quasar::detail {

std::unique_ptr<op_slicing::OpSliceAttr> get_conv2d_slice_attr(
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t in_channels,
    uint32_t out_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding_n4,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    Layout input_layout,
    DataType input_dtype,
    DataType conv_output_dtype,
    Tensor& weight_tensor,
    std::optional<std::reference_wrapper<Tensor>> bias_tensor,
    const Conv2dConfig& conv_config_,
    const DeviceComputeKernelConfig& compute_config,
    MeshDevice* device) {
    return std::unique_ptr<op_slicing::OpSliceAttr>(new Conv2dSliceAttr(
        batch_size,
        {input_height, input_width},
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding_n4,
        dilation,
        groups,
        input_layout,
        input_dtype,
        conv_output_dtype,
        weight_tensor,
        bias_tensor,
        conv_config_,
        compute_config,
        device));
}
}  // namespace ttnn::operations::experimental::quasar::detail
