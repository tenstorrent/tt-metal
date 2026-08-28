// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm.hpp"
#include "device/groupnorm_device_operation.hpp"
#include "device/groupnorm_program_utils.hpp"
#include "groupnorm_grid_utils.hpp"
#include "groupnorm_input_mask.hpp"

#include <mutex>
#include <optional>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace {

using ttnn::operations::normalization::compute_num_virtual_cols;
using ttnn::operations::normalization::find_expected_dram_grid;

// Stats/intermediate CB format: fp32 input uses fp32 stats, bf16 input uses bf16.
ttnn::DataType group_norm_im_data_format(ttnn::DataType input_dtype) {
    return input_dtype == ttnn::DataType::FLOAT32 ? ttnn::DataType::FLOAT32 : ttnn::DataType::BFLOAT16;
}

// Validates that the requested core grid satisfies the DRAM group-norm constraints.
// If the requested grid is invalid, fatals with an error suggesting the largest valid sub-grid.
void validate_dram_grid(
    const ttnn::CoreGrid& requested, uint32_t W, uint32_t Ht, int num_groups, uint32_t num_batches) {
    uint32_t nvc = compute_num_virtual_cols(requested.x, num_groups, W);
    if (nvc > 0) {
        uint32_t rows_per_y = requested.x / nvc;
        if (rows_per_y > 0) {
            uint32_t num_virtual_rows = rows_per_y * requested.y;
            if (Ht >= num_virtual_rows && Ht % num_virtual_rows == 0 &&
                (num_virtual_rows < num_batches || num_virtual_rows % num_batches == 0)) {
                return;
            }
        }
    }

    auto suggested =
        find_expected_dram_grid(requested.x, requested.y, W, num_groups, Ht * ttnn::types::TILE_SIZE, num_batches);

    uint32_t nvc_for_msg = compute_num_virtual_cols(requested.x, num_groups, W);
    uint32_t nvr_for_msg = nvc_for_msg > 0 ? (requested.x / nvc_for_msg) * requested.y : 0;

    // User supplied invalid grid, but suggested a valid grid
    if (suggested.has_value()) {
        TT_THROW(
            "group_norm: Requested core_grid (x={}, y={}) is invalid for the input dimensions "
            "(Ht={}, W={}, num_groups={}, num_batches={}). The grid must satisfy: "
            "num_virtual_rows = (grid_x / num_virtual_cols) * grid_y <= Ht, "
            "Ht must be divisible by num_virtual_rows, and "
            "num_virtual_rows must be divisible by num_batches (for uniform multicast groups). "
            "Computed num_virtual_rows={}, num_virtual_cols={}. "
            "The largest valid grid that fits is (x={}, y={}).",
            requested.x,
            requested.y,
            Ht,
            W,
            num_groups,
            num_batches,
            nvr_for_msg,
            nvc_for_msg,
            suggested->x,
            suggested->y);
    } else {
        TT_THROW(
            "group_norm: Cannot find any valid core grid for the given configuration. "
            "Input height in tiles (Ht={}) is too small for any grid with W={}, num_groups={}, "
            "num_batches={}. Computed num_virtual_rows={}, num_virtual_cols={}. "
            "Requested grid was (x={}, y={}).",
            Ht,
            W,
            num_groups,
            num_batches,
            nvr_for_msg,
            nvc_for_msg,
            requested.x,
            requested.y);
    }
}

// Decides whether the sharded program needs the negative-mask CB-overlap trick to fit in L1.
bool needs_negative_mask_overlap(
    const ttnn::Tensor& input_tensor,
    const ttnn::prim::GroupNormShardedMultiCoreProgramConfig& program_config,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::Tensor>& gamma,
    const std::optional<ttnn::Tensor>& beta,
    const std::optional<ttnn::Tensor>& input_mask,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    float eps,
    bool use_welford,
    uint32_t num_groups) {
    using tt::tt_metal::BufferType;
    using tt::tt_metal::HalMemType;
    using tt::tt_metal::Layout;

    // Only the ROW_MAJOR-in / ROW_MAJOR-out non-Welford sharded kernels have a negative-mask
    // path.
    if (use_welford || input_tensor.layout() != Layout::ROW_MAJOR ||
        program_config.output_layout != Layout::ROW_MAJOR) {
        return false;
    }

    auto* device = input_tensor.device();
    const auto& allocator = device->allocator();
    const uint32_t l1_base = allocator->get_base_allocator_addr(HalMemType::L1);
    // L1 tensor buffers grow down from the top of L1 while circular buffers grow up from
    // l1_base, so the gap between the two is what the CB region has to fit into. nullopt means
    // no L1 tensor is placed at all, which cannot happen on this path.
    const auto lowest_occupied = device->lowest_occupied_compute_l1_address();
    if (!lowest_occupied.has_value() || lowest_occupied.value() <= l1_base) {
        return false;
    }
    uint32_t available = lowest_occupied.value() - l1_base;

    // The output buffer is allocated after this point, so account for it up front.
    if (!program_config.inplace && output_mem_config.buffer_type() == BufferType::L1) {
        const auto output_spec = ttnn::prim::GroupNormDeviceOperation::compute_output_specs(
            ttnn::prim::GroupNormParams{
                .eps = eps,
                .num_groups = num_groups,
                .output_mem_config = output_mem_config,
                .program_config = program_config,
                .compute_kernel_config = compute_kernel_config,
                .use_welford = use_welford},
            ttnn::prim::GroupNormInputs{.input = input_tensor});
        const uint32_t output_per_bank = static_cast<uint32_t>(output_spec.compute_consumed_memory_bytes_per_bank(
            allocator->get_alignment(BufferType::L1), allocator->get_num_banks(BufferType::L1)));
        if (output_per_bank >= available) {
            // Nothing left for CBs either way.
            return false;
        }
        available -= output_per_bank;
    }

    const auto pad = ttnn::prim::make_group_norm_pad_correction(
        static_cast<uint32_t>(input_tensor.logical_shape()[2]),
        static_cast<uint32_t>(input_tensor.padded_shape()[2]),
        use_welford);
    const auto cb_sizes = ttnn::prim::compute_sharded_gn_static_cb_sizes(
        input_tensor,
        program_config.im_data_format,
        gamma.has_value() ? std::make_optional(gamma->dtype()) : std::nullopt,
        beta.has_value() ? std::make_optional(beta->dtype()) : std::nullopt,
        input_mask.has_value() ? std::make_optional(input_mask->dtype()) : std::nullopt,
        /*negative_mask_dtype=*/std::nullopt,  // synthesized, hence bfloat16
        use_welford,
        // c_7 sizing matches the factory by construction now: the sharded writer streams a single
        // (double-buffered) mask set even under the pad correction -- the row mask is composed on
        // device (c_18/c_19) -- so there is no second-set factor to keep in sync here.
        num_groups);
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    ttnn::prim::GroupNormShardedCbFlags flags{
        .with_negative_mask = false,
        .untilize_out = true,  // guaranteed by the ROW_MAJOR output check above
        .has_gamma = gamma.has_value(),
        .has_beta = beta.has_value(),
        .reader_repack_output = (input_tensor.shard_spec().value().shape[1] % tile_width) != 0,
        .use_welford = use_welford,
        .pad_correction_active = pad.active};
    const uint32_t cb_total = cb_sizes.total(flags);

    const bool needed = cb_total > available;
    if (needed) {
        flags.with_negative_mask = true;
        log_debug(
            tt::LogOp,
            "group_norm: enabling the negative-mask CB overlap -- {} B of CBs do not fit in {} B of L1, "
            "the overlap needs {} B",
            cb_total,
            available,
            cb_sizes.total(flags));
    } else {
        log_debug(
            tt::LogOp, "group_norm: no negative-mask overlap needed ({} B of CBs fit in {} B)", cb_total, available);
    }
    return needed;
}

int64_t get_group_norm_cores_across_channel(
    tt::tt_metal::TensorMemoryLayout memory_layout,
    const ttnn::CoreGrid& core_grid,
    const std::optional<tt::tt_metal::ShardOrientation>& shard_orientation) {
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::TensorMemoryLayout;
    if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        if (shard_orientation == ShardOrientation::ROW_MAJOR) {
            return static_cast<int64_t>(core_grid.x);
        }
        return static_cast<int64_t>(core_grid.y);
    }
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        return 1;
    }
    return static_cast<int64_t>(core_grid.x * core_grid.y);
}

}  // namespace

namespace ttnn::operations::normalization {

ttnn::Tensor get_mask_tensor(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& input_mask,
    const std::optional<ttnn::Tensor>& negative_mask,
    const CoreGrid& core_grid,
    const int num_groups,
    const int64_t rows_in_last_tile) {
    ttnn::Tensor mask = input_mask.value_or(ttnn::Tensor());
    if (!input_mask.has_value() and !negative_mask.has_value()) {
        // create input mask
        int64_t num_channel = input_tensor.padded_shape()[-1];
        int64_t num_cores_across_channel;
        if (input_tensor.memory_config().buffer_type() == BufferType::L1) {
            const auto mem_layout = input_tensor.memory_config().memory_layout();
            const auto shard_spec = input_tensor.memory_config().shard_spec();
            std::optional<ShardOrientation> shard_orientation = std::nullopt;
            if (shard_spec.has_value()) {
                shard_orientation = shard_spec->orientation;
            }
            num_cores_across_channel = get_group_norm_cores_across_channel(mem_layout, core_grid, shard_orientation);
        } else {
            uint32_t num_virtual_cols =
                compute_num_virtual_cols(core_grid.x, num_groups, static_cast<uint32_t>(num_channel));
            TT_FATAL(
                num_virtual_cols > 0,
                "group_norm: Cannot determine num_virtual_cols for core_grid x={}, num_groups={}, "
                "num_channels={}. num_virtual_cols must satisfy (num_channels / nvc) % TILE_SIZE == 0 "
                "and num_groups % nvc == 0.",
                core_grid.x,
                num_groups,
                num_channel);
            num_cores_across_channel = static_cast<int64_t>(num_virtual_cols);
        }
        // rows_in_last_tile != 0 builds the doubled mask; free here, since the mask is assembled
        // on host and uploaded once either way.
        mask = create_group_norm_input_mask(
            num_channel,
            num_groups,
            num_cores_across_channel,
            tt::tt_metal::DataType::BFLOAT16,
            input_tensor.tensor_spec().tile().get_height(),
            input_tensor.tensor_spec().tile().get_width(),
            rows_in_last_tile);
        mask = mask.to_device(input_tensor.device());
    }
    return mask;
}

}  // namespace ttnn::operations::normalization

namespace ttnn {

Tensor group_norm(
    const Tensor& input_tensor,
    const int num_groups,
    const float epsilon,
    const std::optional<Tensor>& input_mask,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& reciprocals,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType> dtype,
    std::optional<CoreGrid> core_grid,
    std::optional<bool> inplace,
    std::optional<Layout> output_layout,
    std::optional<int> num_out_blocks,
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<Tensor>& negative_mask,
    bool use_welford) {
    if (input_tensor.layout() == Layout::TILE and inplace.has_value()) {
        TT_FATAL(
            !inplace.value(),
            "In-place operation not supported: Tile layout requires non-inplace tensors. (inplace={})",
            inplace.value());
    }

    if (output_layout.has_value() and inplace.has_value()) {
        if (output_layout != input_tensor.layout()) {
            TT_FATAL(
                !inplace.value(),
                "In-place operation not allowed: Input and output tensor layouts differ. (input_layout={}, "
                "output_layout={})",
                input_tensor.layout(),
                output_layout.value());
        }
    }

    TT_FATAL(
        input_tensor.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout: Input tensor cannot be width-sharded.");


    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(
        input_shape.rank() == 4, "Invalid tensor shape: Input tensor must have rank 4. (rank={})", input_shape.rank());

    TT_FATAL(
        input_shape[-1] % num_groups == 0,
        "Invalid channel configuration: Number of channels ({}) must be divisible by the number of groups ({}).",
        input_shape[-1],
        num_groups);

    const auto& input_padded_shape = input_tensor.padded_shape();
    const auto nhw = input_padded_shape[0] * input_padded_shape[1] * input_padded_shape[2];
    TT_FATAL(
        (nhw % ttnn::types::TILE_SIZE) == 0,
        "Invalid tensor dimensions: The product of NHW dimensions ({}) must be divisible by the tile size ({}).",
        nhw,
        ttnn::types::TILE_SIZE);

    // The #50682 correction exists only on the two-pass path: Welford transposes H*W into the tile
    // columns and counts in tile units, so the padding rows cannot be excluded. Route here and drop
    // the Welford-only reciprocals LUT. Only place that enforces it -- a new direct
    // ttnn::prim::group_norm caller would need its own guard.
    std::optional<Tensor> effective_reciprocals = reciprocals;
    const uint32_t tile_height_align = input_tensor.tensor_spec().tile().get_height();
    if (use_welford && (input_shape[2] % tile_height_align != 0)) {
        // Once per process: group_norm runs inside per-step model loops.
        static std::once_flag welford_fallback_warned;
        std::call_once(welford_fallback_warned, [&] {
            log_warning(
                tt::LogOp,
                "group_norm: use_welford is not supported for non-tile-aligned H*W ({} % {} != 0); "
                "falling back to the two-pass path, which handles this case correctly (#50682). "
                "This warning is emitted once per process.",
                input_shape[2],
                tile_height_align);
        });
        use_welford = false;
        effective_reciprocals = std::nullopt;
    }

    // For 0V tensors
    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return ttnn::clone(input_tensor, /*dtype=*/std::nullopt, memory_config, /*compute_kernel_config=*/std::nullopt);
    }

    const std::optional<ttnn::Tensor>& gamma =
        weight.has_value() ? std::optional<ttnn::Tensor>(ttnn::unsqueeze_to_4D(weight.value())) : std::nullopt;
    const std::optional<ttnn::Tensor>& beta =
        bias.has_value() ? std::optional<ttnn::Tensor>(ttnn::unsqueeze_to_4D(bias.value())) : std::nullopt;

    const MemoryConfig& dram_memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    const MemoryConfig& output_mem_config = memory_config.value_or(dram_memory_config);

    // Initialize compute kernel config
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Invalid input tensor storage type: Input tensor must be on device. (storage type={})",
        input_tensor.storage_type());
    if (input_mask.has_value()) {
        TT_FATAL(
            input_mask->storage_type() == StorageType::DEVICE,
            "Input mask must be on device, got storage type: {}",
            input_mask->storage_type());
        TT_FATAL(input_mask->buffer() != nullptr, "Input mask must be allocated in buffers on device!");
        TT_FATAL(input_tensor.device() == input_mask->device(), "Input and input mask tensors must be on same device");
    }

    // Program factories require output dtype == input dtype.
    const auto out_dtype = dtype.value_or(input_tensor.dtype());
    TT_FATAL(
        out_dtype == input_tensor.dtype(),
        "group_norm output dtype must match input dtype ({}), got dtype={}",
        input_tensor.dtype(),
        out_dtype);

    const auto arch = input_tensor.device()->arch();

    // Interleaved (non-sharded) ROW_MAJOR is Wormhole-only: it is a perf regression on Blackhole; see #52279.
    if (!input_tensor.is_sharded() && arch != tt::ARCH::WORMHOLE_B0) {
        TT_FATAL(
            input_tensor.layout() == Layout::TILE,
            "group_norm: interleaved (non-sharded) input must be in TILE layout, got ROW_MAJOR. "
            "Convert the input with ttnn.to_layout(input, ttnn.TILE_LAYOUT) before calling group_norm. "
            "ROW_MAJOR is supported only for sharded inputs, or on Wormhole.");
        TT_FATAL(
            output_layout.value_or(Layout::TILE) == Layout::TILE,
            "group_norm: interleaved (non-sharded) output must be in TILE layout, got ROW_MAJOR output_layout. "
            "Request TILE output and convert it yourself with ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT). "
            "ROW_MAJOR output is supported only for sharded inputs, or on Wormhole.");
    }

    const auto math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    const auto approx_mode = true;
    // fp32 input accumulates in the fp32 DEST (like LayerNorm); welford already forces it. A
    // user-supplied compute_kernel_config still overrides this default.
    const auto fp32_acc = use_welford || (input_tensor.dtype() == DataType::FLOAT32);
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, math_fidelity, approx_mode, fp32_acc);

    // Reciprocals must be sharded to L1 via the legacy ShardSpec representation:
    // the program factory reads shard_spec().value().numel() as the compile-time
    // reciprocal_size and binds the cb_reciprocals CB to the per-bank addresses
    // of the buffer. Interleaved, DRAM-sharded, or NdShardSpec reciprocals would
    // either trip bad_optional_access or violate the per-core-L1-bank assumption
    // downstream.
    if (reciprocals.has_value()) {
        TT_FATAL(
            reciprocals->is_sharded() && reciprocals->memory_config().buffer_type() == BufferType::L1,
            "group_norm: reciprocals tensor must be sharded to L1 (got is_sharded={}, buffer_type={}); "
            "interleaved or DRAM-sharded reciprocals are not supported.",
            reciprocals->is_sharded(),
            reciprocals->memory_config().buffer_type());
        TT_FATAL(
            reciprocals->shard_spec().has_value(),
            "group_norm: reciprocals tensor must use the legacy ShardSpec "
            "representation (NdShardSpec sharding is not currently supported).");
    }

    const bool core_grid_auto_selected = !core_grid.has_value();

    if (core_grid_auto_selected) {
        if (input_tensor.is_sharded()) {
            const auto& shard_spec_opt = input_tensor.shard_spec();
            TT_FATAL(
                shard_spec_opt.has_value(),
                "group_norm: Sharded input must have a shard spec when core_grid is not provided.");
            // Derive the grid directly from the tensor's existing shard layout
            // rather than recomputing from scratch, so that program_config's
            // grid_size matches the cores where kernels are actually placed.
            const auto bbox = shard_spec_opt->grid.bounding_box();
            core_grid = ttnn::operations::normalization::core_grid_from_shard_bounding_box(bbox);
        } else if (reciprocals.has_value() && reciprocals->is_sharded()) {
            // The reciprocals LUT is sharded on a specific grid; its length
            // encodes num_virtual_rows which must match the compute grid.
            // Infer the grid from the reciprocals tensor so the kernel sees a
            // consistent LUT.
            const auto bbox = reciprocals->shard_spec()->grid.bounding_box();
            core_grid = ttnn::operations::normalization::core_grid_from_shard_bounding_box(bbox);
        } else {
            const auto dev_grid = input_tensor.device()->compute_with_storage_grid_size();
            auto dram_grid = ttnn::operations::normalization::find_expected_dram_grid(
                dev_grid.x, dev_grid.y, input_padded_shape[3], num_groups, nhw, input_padded_shape[0]);
            TT_FATAL(
                dram_grid.has_value(),
                "group_norm: Could not determine a valid DRAM core grid for channels={}, num_groups={}, nhw={}. "
                "Specify core_grid explicitly or adjust configuration.",
                input_padded_shape[3],
                num_groups,
                nhw);
            core_grid = dram_grid.value();
        }
    }

    // num_out_blocks only affects the non-sharded (interleaved/DRAM) program factory
    // (as stated in docstring in nanobind documents), so the assert that "auto-grid implies
    // no explicit chunking" rule only applies to the non-sharded path.
    TT_FATAL(
        (input_tensor.is_sharded() || !(core_grid_auto_selected && num_out_blocks.has_value())),
        "group_norm: num_out_blocks cannot be specified when core_grid is auto-selected. "
        "Either provide an explicit core_grid or omit num_out_blocks.");

    if (!input_tensor.is_sharded() && num_out_blocks.has_value()) {
        // Reject obviously-out-of-range values of num_out_blocks up front, but only for
        // non-sharded inputs, where num_out_blocks actually has effect.
        // Accepted user-facing values are -1 (use the program-factory's auto-heuristic)
        // or an explicit chunk count in [1, block_h]; the upper bound and >= 1 are checked
        // later against the resolved block_h.
        // Catching < -1 here prevents the value, which gets reinterpreted as a huge
        // uint32_t at some point, from producing a confusing downstream error.
        TT_FATAL(
            *num_out_blocks >= -1,
            "group_norm: num_out_blocks ({}) is invalid. Use -1 to request the auto-heuristic, "
            "or an explicit chunk count in [1, block_h].",
            *num_out_blocks);
    }

    // The per-shard numel (consumed by the compute kernel as the compile-time
    // `reciprocal_size`) and the per-bank addresses bound to the reciprocals CB
    // are baked for a specific grid. The compute kernel runs on `core_grid`,
    // so the reciprocals must be sharded on that same grid; otherwise the LUT
    // is the wrong length and/or lives on the wrong banks. This covers all
    // three paths to picking core_grid (sharded input, reciprocals inference,
    // and an explicit user-provided core_grid).
    if (reciprocals.has_value()) {
        // Precondition above guarantees is_sharded() and shard_spec().has_value()
        // whenever reciprocals is provided.
        const auto recip_bbox = reciprocals->shard_spec()->grid.bounding_box();
        const auto recip_grid = ttnn::operations::normalization::core_grid_from_shard_bounding_box(recip_bbox);
        TT_FATAL(
            recip_grid.x == core_grid->x && recip_grid.y == core_grid->y,
            "group_norm: reciprocals shard grid (x={}, y={}) must match the compute core_grid "
            "(x={}, y={}). The reciprocals LUT length and per-bank addresses are baked for a "
            "specific grid; running the kernel on a different grid will read past the LUT or "
            "use unallocated banks.",
            recip_grid.x,
            recip_grid.y,
            core_grid->x,
            core_grid->y);
    }

    // For non-sharded DRAM tensors, validate that the requested core grid is not too
    // large for the input spatial dimensions. The constraint is Ht >= num_virtual_rows,
    // where num_virtual_rows = (grid_x / num_virtual_cols) * grid_y and Ht = NHW / TILE_SIZE.
    // Violating this leads to per_core_Mt == 0, causing division-by-zero in kernels.
    if (!input_tensor.is_sharded()) {
        const uint32_t W = input_padded_shape[3];
        const uint32_t Ht = nhw / ttnn::types::TILE_SIZE;
        validate_dram_grid(core_grid.value(), W, Ht, num_groups, input_padded_shape[0]);
    }

    // Tile-aligned H*W: the caller's optional passes straight through; when it is empty the
    // writer kernel synthesizes the per-group {0.0, 1.0} selector directly in L1 (see
    // groupnorm_mask_synthesize.hpp).
    //
    // Non-tile-aligned H*W: the tile-padding rows are not guaranteed to hold zeros (reshape,
    // slice and exp all leave non-zero bytes there), so both accumulation passes must exclude
    // them via a row-masked variant of the mask on each batch's final row-tile. An empty
    // optional still means "synthesize". Welford is unaffected: non-tile-aligned H*W already
    // fell back to the two-pass path above.
    //
    // A caller-supplied mask is honoured as the column selector (synthesis is bf16-only, while
    // callers commonly supply BFLOAT8_B). A doubled mask built with rows_in_last_tile is
    // accepted; a single-set mask is used for both sets.
    const uint32_t rows_in_last_tile =
        (input_padded_shape[2] != input_shape[2] && !use_welford) ? (input_shape[2] % tile_height_align) : 0;

    std::optional<ttnn::Tensor> effective_input_mask = input_mask;
    if (rows_in_last_tile != 0 && input_mask.has_value() &&
        input_mask.value().padded_shape()[1] == static_cast<uint32_t>(num_groups)) {
        // Caller supplied a single-set mask for a non-tile-aligned input. Derive the row-masked
        // second set. Costs a host build, an upload, a multiply and a concat per call, so prefer
        // passing rows_in_last_tile to create_group_norm_input_mask -- or omitting the mask
        // entirely, which is now free on this path.
        ttnn::Tensor mask = input_mask.value();
        const auto row_mask =
            operations::normalization::create_group_norm_row_mask(
                rows_in_last_tile, mask.padded_shape()[1], mask.padded_shape()[3], mask.dtype(), tile_height_align)
                .to_device(mask.device());
        const auto row_masked = ttnn::multiply(mask, row_mask, mask.dtype());
        effective_input_mask = ttnn::concat({mask, row_masked}, 1);
    }

    if (input_tensor.is_sharded()) {
        const ttnn::prim::GroupNormShardedMultiCoreProgramConfig program_config = {
            .compute_with_storage_grid_size = core_grid.value().to_CoreCoord(),
            .im_data_format = group_norm_im_data_format(input_tensor.dtype()),
            .out_data_format = out_dtype,
            .inplace = inplace.value_or(false),
            .output_layout = output_layout.value_or(input_tensor.layout())};
        // A caller-supplied negative_mask always wins; otherwise the op decides for itself
        // whether it needs it.
        const bool synthesize_negative_mask = !negative_mask.has_value() && needs_negative_mask_overlap(
                                                                                input_tensor,
                                                                                program_config,
                                                                                output_mem_config,
                                                                                gamma,
                                                                                beta,
                                                                                input_mask,
                                                                                kernel_config_val,
                                                                                epsilon,
                                                                                use_welford,
                                                                                static_cast<uint32_t>(num_groups));
        return ttnn::prim::group_norm(
            input_tensor,
            epsilon,
            static_cast<uint32_t>(num_groups),
            output_mem_config,
            program_config,
            kernel_config_val,
            use_welford,
            gamma,
            beta,
            effective_input_mask,
            negative_mask,
            effective_reciprocals,
            synthesize_negative_mask);
    }

    const uint32_t per_batch_hw = input_padded_shape[1] * input_padded_shape[2];
    const uint32_t tile_h = input_tensor.tensor_spec().tile().get_height();
    const bool per_batch_hw_tile_aligned = per_batch_hw % tile_h == 0;
    const Layout requested_out_layout = output_layout.value_or(input_tensor.layout());

    // L1-fit estimate shared by the input and output decisions below.
    const uint32_t Ht = nhw / ttnn::types::TILE_SIZE;
    const uint32_t tile_w = input_tensor.tensor_spec().tile().get_width();
    const uint64_t base_l1 = input_tensor.device()->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint64_t available_l1 = input_tensor.device()->l1_size_per_core() - base_l1;
    const auto legacy_rm_fits_l1 = [&](bool tilize_in, bool untilize_out) {
        return ttnn::prim::groupnorm_legacy_rm_input_fits_l1(
            Ht,
            input_padded_shape[3],
            per_batch_hw,
            input_padded_shape[0],
            core_grid->x,
            core_grid->y,
            static_cast<uint32_t>(num_groups),
            core_grid_auto_selected ? -1 : num_out_blocks.value_or(1),
            tile_w,
            tile_h * tile_w * input_tensor.element_size(),
            gamma.has_value(),
            beta.has_value(),
            tilize_in,
            untilize_out,
            available_l1);
    };

    // Composite fallback: L1 does not fit, or fused RM is expected to be slower.
    Tensor gn_input = input_tensor;
    if (!use_welford && input_tensor.layout() == Layout::ROW_MAJOR && per_batch_hw_tile_aligned) {
        const bool fits_l1 =
            legacy_rm_fits_l1(/*tilize_in=*/true, /*untilize_out=*/requested_out_layout == Layout::ROW_MAJOR);
        const uint32_t num_virtual_cols = compute_num_virtual_cols(core_grid->x, num_groups, input_padded_shape[3]);
        const uint32_t num_virtual_rows = num_virtual_cols == 0 ? 0 : (core_grid->x / num_virtual_cols) * core_grid->y;
        const uint32_t num_cores = num_virtual_cols * num_virtual_rows;
        const bool prefer_composite = ttnn::prim::groupnorm_legacy_rm_prefer_composite_for_perf(
            num_cores, num_virtual_rows, input_padded_shape[0]);
        if (!fits_l1 || prefer_composite) {
            log_debug(
                tt::LogOp,
                "group_norm: tilizing ROW_MAJOR input on host and running the TILE path (composite) -- "
                "reason: {}.",
                !fits_l1 ? "resident tilized group does not fit L1" : "composite preferred for perf");
            // Keep the input's memory config; output_mem_config may well be different.
            gn_input = ttnn::tilize_with_zero_padding(input_tensor, input_tensor.memory_config());
        }
    }

    // Likewise for the output: the fused untilize needs extra CBs, so fall back to untilizing on host.
    Layout device_out_layout = requested_out_layout;
    bool untilize_out_on_host = false;
    if (!use_welford && requested_out_layout == Layout::ROW_MAJOR && per_batch_hw_tile_aligned) {
        if (!legacy_rm_fits_l1(/*tilize_in=*/gn_input.layout() == Layout::ROW_MAJOR, /*untilize_out=*/true)) {
            log_debug(
                tt::LogOp,
                "group_norm: running the TILE-output path and untilizing on host (composite output) -- "
                "the fused ROW_MAJOR-output CBs do not fit L1.");
            device_out_layout = Layout::TILE;
            untilize_out_on_host = true;
        }
    }

    // When the user did not pin a core grid, defer num_out_blocks to the program
    // factory's heuristic via the -1 sentinel (see GroupNormMultiCoreProgramConfig).
    // Otherwise honor the explicit num_out_blocks (defaulting to 1 = no chunking).
    const ttnn::prim::GroupNormMultiCoreProgramConfig program_config = {
        .compute_with_storage_grid_size = core_grid.value().to_CoreCoord(),
        .im_data_format = group_norm_im_data_format(input_tensor.dtype()),
        .out_data_format = out_dtype,
        .inplace = inplace.value_or(false),
        .output_layout = device_out_layout,
        .num_out_blocks = core_grid_auto_selected ? -1 : num_out_blocks.value_or(1)};
    Tensor output = ttnn::prim::group_norm(
        gn_input,
        epsilon,
        static_cast<uint32_t>(num_groups),
        output_mem_config,
        program_config,
        kernel_config_val,
        use_welford,
        gamma,
        beta,
        effective_input_mask,
        negative_mask,
        effective_reciprocals,
        // The interleaved factories have no negative-mask code path at all.
        /*synthesize_negative_mask=*/false);
    if (untilize_out_on_host) {
        output = ttnn::to_layout(output, Layout::ROW_MAJOR);
    }
    return output;
}

}  // namespace ttnn
