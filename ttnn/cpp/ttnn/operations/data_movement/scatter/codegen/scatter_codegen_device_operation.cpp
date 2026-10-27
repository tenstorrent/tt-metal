// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_codegen_device_operation.hpp"

#include <algorithm>
#include <variant>

#include <tt_stl/assert.hpp>

#include "scatter_codegen_supported.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {
using namespace tt::tt_metal;
namespace scatter_ns = ttnn::operations::data_movement::scatter;

// ROW_MAJOR dispatches on dtype+reduction_mode alone (both already part of the default hash), so no
// live L1 read is needed to pick between the two RM factories. TILE prefers the row-buffered
// interleaved plan whenever its footprint fits the live L1 frontier, UNLESS the tile-row count
// underfills the grid while more than one output column exists -- there, splitting by output COLUMN
// (streaming) reaches every core instead of leaving most of them idle even though the interleaved
// plan would also fit.
ScatterCodegenDeviceOperation::program_factory_t ScatterCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        const bool bf16_reduce = attributes.reduction_mode == scatter_ns::kReductionAdd ||
                                 attributes.reduction_mode == scatter_ns::kReductionMultiply;
        if (bf16_reduce) {
            return ScatterCodegenProgramFactoryBf16ReduceRowMajor{};
        }
        return ScatterCodegenProgramFactoryRowMajor{};
    }

    // Output dtype/tile always equal the input's (compute_output_specs), but the ALIGNED page size
    // also depends on Buffer::alignment(), which is BufferType-dependent -- so the output estimate is
    // only right when it accounts for the caller's actual output placement, not the input's.
    const uint64_t output_page =
        scatter_output_aligned_page_size(input_tensor, attributes.output_mem_config, tensor_args.output_tensor);
    const uint64_t input_page = input_tensor.buffer()->aligned_page_size();
    const uint64_t index_page = tensor_args.index_tensor.buffer()->aligned_page_size();
    const uint64_t src_page = tensor_args.src_tensor.buffer()->aligned_page_size();
    const bool interleaved_fits = scatter_interleaved_fits_l1(
        scatter_usable_l1(input_tensor), attributes.Wt_output, output_page, input_page, index_page, src_page);

    auto* device = input_tensor.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t device_cores = static_cast<uint32_t>(grid.x * grid.y);
    const uint32_t candidate_cores = attributes.sub_core_grids.has_value()
                                         ? static_cast<uint32_t>(attributes.sub_core_grids->num_cores())
                                         : device_cores;
    const uint32_t column_cores = std::min(attributes.Wt_output, candidate_cores);
    const bool use_column_parallel = attributes.Ht < column_cores && attributes.Wt_output > 1;

    if (interleaved_fits && !use_column_parallel) {
        return ScatterCodegenProgramFactoryInterleaved{};
    }
    return ScatterCodegenProgramFactoryStreaming{};
}

// The default key is the attributes and tensor specs, and a foreign L1 buffer is in neither. Both the
// TILE factory choice and the RM chunk depth are read off the live L1 frontier, and
// Program::validate_circular_buffer_region re-checks a cached program's baked CB region against the
// CURRENT frontier on every enqueue, so a plan cached against a clear frontier throws once an
// unrelated L1 tensor moves it. Keying on the derived plan rather than the frontier itself keeps
// entries down to genuinely distinct plans.
//
// Calling select_program_factory() here is sound because create_output_tensors() has already run by
// the time the key is computed (ttnn/device_operation.hpp), so this sees the same frontier the
// factory's create_descriptor() will.
//
// Because the key depends on live L1, a trace captured under one L1 state replays that state's
// plan; a capture whose frontier differs from the warm-up's misses the cache, and the framework
// throws on a miss during capture rather than replaying a stale plan.
ttsl::hash::hash_t ScatterCodegenDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto factory = select_program_factory(attributes, tensor_args);
    uint32_t chunk_elems = 0;
    const bool is_row_major = std::holds_alternative<ScatterCodegenProgramFactoryRowMajor>(factory);
    const bool is_bf16_reduce_row_major =
        std::holds_alternative<ScatterCodegenProgramFactoryBf16ReduceRowMajor>(factory);
    if (is_row_major || is_bf16_reduce_row_major) {
        const auto& in_t = tensor_args.input_tensor;
        // The same page sizes and fixed bytes the matching create_descriptor() computes, so this key
        // never diverges from the chunk depth it bakes into the program.
        const uint64_t input_page_bytes = scatter_rm_aligned_stick_bytes(in_t, attributes.input_stick_elems);
        const uint64_t output_page_bytes =
            scatter_output_aligned_page_size(in_t, attributes.output_mem_config, tensor_args.output_tensor);
        chunk_elems = scatter_rm_chunk_elems(
            scatter_usable_l1(in_t),
            scatter_rm_fixed_bytes(input_page_bytes, output_page_bytes, is_bf16_reduce_row_major),
            attributes.index_stick_elems,
            tensor_args.index_tensor.element_size(),
            tensor_args.src_tensor.element_size());
    }
    // Mirrors the default key and appends to it, rather than naming fields explicitly, so no
    // discrimination the default traversal makes is dropped here by omission.
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<ScatterCodegenDeviceOperation>, attributes, tensor_args, factory.index(), chunk_elems);
}

void ScatterCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& index_tensor = tensor_args.index_tensor;
    const auto& src_tensor = tensor_args.src_tensor;

    // Checked ahead of supported_by_codegen() (which also rejects it) so a direct prim caller gets the
    // specific reason: a zero extent divides the CB page or page-map arithmetic by zero on the host.
    TT_FATAL(
        input_tensor.logical_shape().volume() != 0 && index_tensor.logical_shape().volume() != 0 &&
            src_tensor.logical_shape().volume() != 0,
        "scatter_codegen: zero-volume operands are not supported (input {}, index {}, src {}).",
        input_tensor.logical_shape(),
        index_tensor.logical_shape(),
        src_tensor.logical_shape());

    // Every reader addresses src pages with index's geometry (ttnn::scatter slices src down to
    // index's shape before calling), so a wider src would read the wrong pages and a narrower one
    // past its buffer.
    TT_FATAL(
        src_tensor.logical_shape() == index_tensor.logical_shape() &&
            src_tensor.padded_shape() == index_tensor.padded_shape(),
        "scatter_codegen: src must carry index's shape (index {}, src {}).",
        index_tensor.logical_shape(),
        src_tensor.logical_shape());

    // The readers walk the input's pages and clip each page coordinate against the index's extent
    // (map_scatter_input_page), so an index wider than the input on a leading axis has pages no
    // reader ever visits: its updates would be dropped with no error. The prim only ever holds
    // normalized tensors with the scatter axis last, so every other axis is a leading one.
    {
        const auto& input_shape = input_tensor.logical_shape();
        const auto& index_shape = index_tensor.logical_shape();
        TT_FATAL(
            input_shape.rank() == index_shape.rank(),
            "scatter_codegen: input and index must have equal rank (input {}, index {}).",
            input_shape,
            index_shape);
        for (int32_t i = 0; i + 1 < static_cast<int32_t>(input_shape.rank()); ++i) {
            TT_FATAL(
                index_shape[i] <= input_shape[i],
                "scatter_codegen: index extent {} on axis {} exceeds the input's {} (input {}, index {}).",
                index_shape[i],
                i,
                input_shape[i],
                input_shape,
                index_shape);
        }
    }

    // Repeated ahead of supported_by_codegen(), which also rejects each, so a direct prim caller
    // gets the specific reason.
    TT_FATAL(
        scatter_page_rank(input_tensor.logical_shape()) <= kScatterMaxPageRank,
        "scatter_codegen: at most {} page dimensions are supported; got {}",
        kScatterMaxPageRank,
        scatter_page_rank(input_tensor.logical_shape()));
    TT_FATAL(
        input_tensor.dtype() == src_tensor.dtype(),
        "scatter_codegen: input_dtype differs from src_dtype (input_dtype: {}, src_dtype: {}).",
        input_tensor.dtype(),
        src_tensor.dtype());
    TT_FATAL(
        !input_tensor.is_sharded() && !index_tensor.is_sharded() && !src_tensor.is_sharded(),
        "scatter_codegen: sharded tensors are not supported.");

    // The prim only ever holds an already-normalized tensor (transpose-to-last-dim already applied),
    // so the scatter axis here is always the last dim; -1 says that without re-deriving rank.
    TT_FATAL(
        ttnn::operations::data_movement::scatter::supported_by_codegen(
            input_tensor, /*dim=*/-1, index_tensor, src_tensor, attributes.reduction_mode),
        "scatter_codegen: operands are outside the codegen scope (see supported_by_codegen(): bfloat16 input/src "
        "with an int32/uint32 index in one shared TILE or ROW_MAJOR layout, all non-sharded, non-empty and on the "
        "device, default untransposed 32x32 tiles and standard padding, add/multiply reduction only on ROW_MAJOR, "
        "at most {} page dimensions, and for ROW_MAJOR resident input/output sticks plus the smallest index/src "
        "chunk fitting the device's static per-core L1 bank).",
        kScatterMaxPageRank);

    // The descriptor launches on input_tensor.device() while embedding index/src's raw buffer
    // addresses, so every operand must live on that same device.
    TT_FATAL(
        input_tensor.device() == index_tensor.device() && input_tensor.device() == src_tensor.device(),
        "scatter_codegen: input, index and src tensors must all be on the same device.");

    // Second belt-and-suspenders gate, over the output half of the call contract
    // supported_by_codegen() cannot see: the requested memory config, and a caller-supplied output
    // tensor's own spec (compute_output_specs() hands that spec straight back, so it -- not the
    // input -- is what every CB and per-tile/per-stick transfer would be cut from).
    TT_FATAL(
        ttnn::operations::data_movement::scatter::supported_execution_controls(
            input_tensor, attributes.output_mem_config, tensor_args.output_tensor),
        "scatter_codegen: requested output placement is not supported (sharded memory config, a ROW_MAJOR output "
        "whose buffer type differs from the input's, or a preallocated output tensor whose shape, layout, dtype or "
        "tile differ from what this op would create for itself).");
    if (tensor_args.output_tensor.has_value()) {
        const auto& out = tensor_args.output_tensor.value();
        TT_FATAL(
            out.device() == input_tensor.device(),
            "scatter_codegen: preallocated output tensor must be on the same device as the input.");
        TT_FATAL(
            out.buffer() != nullptr, "scatter_codegen: preallocated output tensor has no device buffer allocated.");
        TT_FATAL(
            out.buffer() != src_tensor.buffer() && out.buffer() != index_tensor.buffer(),
            "scatter_codegen: preallocated output tensor must not alias src or index (the readers scan the whole "
            "source and index rows while the writers update the output).");
    }
}

// A hit re-runs the miss checks. The key pins the tensor specs through their attribute values, and
// a Tile's leave out its transpose flags, so a transposed-tile operand hashes like a default-tile one
// and only a re-check keeps it off the fixed-face readers; the checks also cover what no key carries
// at all, an operand on another device. The cost is a few spec constructions and allocator reads
// per call, small next to a dispatch.
void ScatterCodegenDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

ScatterCodegenDeviceOperation::spec_return_value_t ScatterCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value().tensor_spec();
    }
    return tt::tt_metal::TensorSpec(
        tensor_args.input_tensor.logical_shape(),
        TensorLayout(
            tensor_args.input_tensor.dtype(),
            PageConfig(tensor_args.input_tensor.layout(), tensor_args.input_tensor.tensor_spec().tile()),
            attributes.output_mem_config));
}

ScatterCodegenDeviceOperation::tensor_return_value_t ScatterCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<ScatterCodegenDeviceOperation::tensor_return_value_t>
ScatterCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input_tensor, output);
    return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
}

namespace {

// Fills either the TILE or the ROW_MAJOR geometry fields (the other side's are zero) plus the page
// map every factory shares. The only constructor of a ScatterCodegenParams, so the page map the
// readers fold ordinals through is always the one the tensors describe.
ScatterCodegenParams build_scatter_codegen_params(
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    uint32_t reduction_mode,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    const bool tiled = input_tensor.layout() == Layout::TILE;
    const auto page_map = compute_scatter_page_map(input_tensor.logical_shape(), index_tensor.logical_shape(), tiled);

    if (tiled) {
        const auto geometry = compute_scatter_tile_geometry(input_tensor, index_tensor, src_tensor);
        return ScatterCodegenParams{
            geometry.Ht,
            geometry.Wt_output,
            geometry.Wt_index,
            geometry.output_logical_w,
            geometry.idx_valid_h_last,
            geometry.idx_valid_w_last,
            geometry.Ht_per_batch_index,
            /*num_sticks=*/0,
            /*input_stick_elems=*/0,
            /*index_stick_elems=*/0,
            page_map,
            reduction_mode,
            output_mem_config,
            sub_core_grids};
    }
    const auto geometry = compute_scatter_rm_geometry(input_tensor, index_tensor);
    return ScatterCodegenParams{
        /*Ht=*/0,
        /*Wt_output=*/0,
        /*Wt_index=*/0,
        /*output_logical_w=*/0,
        /*idx_valid_h_last=*/0,
        /*idx_valid_w_last=*/0,
        /*Ht_per_batch_index=*/0,
        geometry.num_sticks,
        geometry.input_stick_elems,
        geometry.index_stick_elems,
        page_map,
        reduction_mode,
        output_mem_config,
        sub_core_grids};
}

}  // namespace

Tensor scatter_codegen(
    const Tensor& input_tensor,
    const Tensor& index_tensor,
    const Tensor& src_tensor,
    uint32_t reduction_mode,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<Tensor>& output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::device_operation::launch<ScatterCodegenDeviceOperation>(
        build_scatter_codegen_params(
            input_tensor, index_tensor, src_tensor, reduction_mode, output_mem_config, sub_core_grids),
        ScatterCodegenInputs{input_tensor, index_tensor, src_tensor, output_tensor});
}

}  // namespace ttnn::prim
