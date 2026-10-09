// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar (Metal 2.0) port of ttnn::concat (operations/data_movement/concat/concat.cpp plus its
// concat_impl). The flow is the original's, with these differences:
//  - every helper op is the experimental/quasar copy (to_memory_config, to_layout,
//    untilize_with_unpadding, tilize_with_val_padding, transpose, slice, reshape);
//  - the device op has only the generic TensorAccessor factory, so sharded inputs never take the
//    original's zero-copy shard-as-CB path: TILE (and height-sharded ROW_MAJOR) shards are read and
//    written in place by that factory, other ROW_MAJOR shards are staged through interleaved DRAM;
//  - a tile-padded width concat always takes the untilize -> row-major concat -> retilize route
//    (the original's native tiled-unaligned factory is not ported), and groups must be 1;
//  - the unaligned-last-dim fallback transposes row-major tensors through tile layout
//    (transpose_through_tiles), and input lists are batched at 16 rather than 47 (DM stack, see
//    max_inputs_per_concat_program).

#include "ttnn/operations/experimental/quasar/concat/concat.hpp"

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-logger/tt-logger.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/experimental/quasar/concat/device/concat_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/quasar/slice/slice.hpp"
#include "ttnn/operations/experimental/quasar/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"
#include "ttnn/operations/experimental/quasar/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/experimental/quasar/transpose/transpose.hpp"
#include "ttnn/operations/experimental/quasar/untilize_with_unpadding/untilize_with_unpadding.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::quasar {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using tt::tt_metal::BufferType;
using tt::tt_metal::Layout;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::TensorMemoryLayout;

using OwnedConcatArgs = std::tuple<std::vector<ttnn::Tensor>, int, unsigned int>;
using MassagedConcat = ttnn::operations::data_movement::
    MassagedOperation<ttnn::Tensor, const std::vector<ttnn::Tensor>&, int, unsigned int>;
using MassagedConcatParams = ttnn::operations::data_movement::
    MassagedOperationParams<ttnn::Tensor, const std::vector<ttnn::Tensor>&, int, unsigned int>;

// Longer input lists are concatenated in batches (the original batches at 47; see the constant).
constexpr uint32_t max_tensors_per_concat = ttnn::prim::qsr::max_inputs_per_concat_program;

// ROW_MAJOR pages are whole rows only for interleaved and height-sharded buffers; width-, block- and
// ND-sharded ones page by shard width, which the device op cannot assemble rows from.
bool rm_pages_are_full_rows(const MemoryConfig& memory_config) {
    return !memory_config.is_sharded() || memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
}

// quasar::transpose on a ROW_MAJOR tensor returns wrong data on Quasar (every element, all shapes
// tried on craq-sim: its row-major WH path tilizes, transposes and untilizes in one compute kernel),
// while its TILE transpose is exact. The original transposes row-major tensors directly; here they go
// tilize -> TILE transpose -> untilize instead. Only the unaligned-last-dim fallback below needs it.
Tensor transpose_through_tiles(
    const Tensor& tensor, int dim1, int dim2, const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    if (tensor.layout() != Layout::ROW_MAJOR) {
        return quasar::transpose(tensor, dim1, dim2, memory_config);
    }
    const auto tiled = quasar::tilize_with_val_padding(
        tensor,
        ttnn::operations::data_movement::compute_padded_shape(tensor.logical_shape()),
        0.0f,
        tensor.memory_config());
    const auto transposed = quasar::transpose(tiled, dim1, dim2, tiled.memory_config());
    ttsl::SmallVector<uint32_t> ends(transposed.logical_shape().cbegin(), transposed.logical_shape().cend());
    std::transform(ends.begin(), ends.end(), ends.begin(), [](const auto l) { return l - 1; });
    return quasar::untilize_with_unpadding(
        transposed, ttnn::Shape(ends), memory_config.value_or(tensor.memory_config()));
}

Tensor concat_impl(
    const std::vector<Tensor>& input_tensors,
    const std::int64_t dim,
    const MemoryConfig& output_mem_config,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids = std::nullopt) {
    using namespace tt::constants;

    TT_FATAL(!input_tensors.empty(), "need 1 or more tensors");

    for (const auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    }
    if (input_tensors.size() == 1) {
        // Single tensor case - just ensure it has the correct memory config
        const auto& input = input_tensors[0];
        if (input.memory_config() != output_mem_config) {
            return quasar::to_memory_config(input, output_mem_config);
        }
        return input;
    }

    // Handle large number of tensors by splitting into batches of the safe limit.
    if (input_tensors.size() > max_tensors_per_concat) {
        std::vector<Tensor> intermediate_results;
        const size_t num_batches = tt::div_up(input_tensors.size(), max_tensors_per_concat);
        intermediate_results.reserve(num_batches);

        log_debug(
            tt::LogOp,
            "ttnn.experimental.quasar.concat: Processing {} tensors in {} batches of up to {} tensors each",
            input_tensors.size(),
            num_batches,
            max_tensors_per_concat);

        for (size_t i = 0; i < input_tensors.size(); i += max_tensors_per_concat) {
            const size_t batch_size = std::min(static_cast<size_t>(max_tensors_per_concat), input_tensors.size() - i);
            std::vector<Tensor> batch(input_tensors.begin() + i, input_tensors.begin() + i + batch_size);
            intermediate_results.push_back(concat_impl(batch, dim, output_mem_config, sub_core_grids));
        }

        // Final concat
        return concat_impl(intermediate_results, dim, output_mem_config, sub_core_grids);
    }

    const uint32_t ref_rank = input_tensors[0].logical_shape().rank();
    const uint32_t normalized_dim = input_tensors[0].logical_shape().get_normalized_index(dim);

    if (input_tensors[0].is_sharded()) {
        // The generic factory reads and writes shards in place through TensorAccessor, as the
        // original does for DRAM-sharded tensors -- except ROW_MAJOR shards that page by shard
        // width, on either side.
        if (input_tensors[0].layout() == Layout::ROW_MAJOR &&
            !(rm_pages_are_full_rows(input_tensors[0].memory_config()) && rm_pages_are_full_rows(output_mem_config))) {
            // Unshard to interleaved and re-enter: the interleaved path below already knows how to
            // stage a width/block-sharded RM output through an interleaved result.
            log_debug(
                tt::LogOp,
                "ttnn.experimental.quasar.concat: {} ROW_MAJOR inputs page by shard width, which the device op "
                "cannot read directly; unsharding to interleaved first.",
                input_tensors[0].memory_config().memory_layout());
            const auto interleaved_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
            std::vector<Tensor> interleaved_inputs;
            interleaved_inputs.reserve(input_tensors.size());
            for (const auto& input_tensor : input_tensors) {
                interleaved_inputs.push_back(quasar::to_memory_config(input_tensor, interleaved_config));
            }
            return concat_impl(interleaved_inputs, dim, output_mem_config, sub_core_grids);
        }
        return ttnn::prim::qsr::concat(input_tensors, dim, output_mem_config);
    }
    if (input_tensors[0].layout() == Layout::ROW_MAJOR && normalized_dim == ref_rank - 1) {
        for (const auto& input_tensor : input_tensors) {
            TT_FATAL(
                (input_tensor.padded_shape()[dim] * input_tensor.element_size()) % input_tensor.buffer()->alignment() ==
                    0,
                "Current concat implementation requires aligned last dim when concatting on last dim");
        }
    }
    // Determine target layout by checking all inputs
    // Start with first input's layout, but may need to fall back to ROW_MAJOR
    Layout target_layout = input_tensors[0].layout();

    // Check all inputs - if any ROW_MAJOR input cannot be tiled, use ROW_MAJOR for all
    for (const auto& input_tensor : input_tensors) {
        if (input_tensor.layout() == Layout::ROW_MAJOR) {
            const auto& input_shape = input_tensor.padded_shape();
            if (input_shape.rank() < 2 || input_shape[-2] % TILE_HEIGHT != 0 || input_shape[-1] % TILE_WIDTH != 0) {
                target_layout = Layout::ROW_MAJOR;
                break;
            }
        }
    }

    // Format all inputs to target layout
    std::vector<Tensor> formatted_tensors;
    formatted_tensors.reserve(input_tensors.size());

    for (const auto& input_tensor : input_tensors) {
        if (input_tensor.layout() == target_layout) {
            formatted_tensors.push_back(input_tensor);
        } else {
            formatted_tensors.push_back(
                quasar::to_layout(input_tensor, target_layout, std::nullopt, std::nullopt, sub_core_grids));
        }
    }

    if (output_mem_config.is_sharded() && target_layout == Layout::ROW_MAJOR &&
        output_mem_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        // For width/block-sharded RM output the buffer page width equals the shard width,
        // which is narrower than the full-row pages the concat pipeline produces.
        // Fall back to interleaved concat + to_memory_config for these cases.
        // Height-sharded RM pages span the full tensor width (same as interleaved),
        // so they flow through the device op natively via TensorAccessor.
        const auto interleaved_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
        auto interleaved_result = ttnn::prim::qsr::concat(formatted_tensors, dim, interleaved_config, sub_core_grids);
        return quasar::to_memory_config(interleaved_result, output_mem_config);
    }
    return ttnn::prim::qsr::concat(formatted_tensors, dim, output_mem_config, sub_core_grids);
}

MassagedConcat build_untilize_rm_retilize_concat(const MemoryConfig& output_memory_config) {
    return MassagedConcat(MassagedConcatParams{
        .predicate = [](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int /*groups*/) -> bool {
            // untilize_rm_retilize if the concat dim is padded for tilized tensors
            return std::any_of(tensors.begin(), tensors.end(), [&](const ttnn::Tensor& tensor) {
                return tensor.layout() == Layout::TILE and (tensor.logical_shape()[dim] != tensor.padded_shape()[dim] or
                                                            tensor.logical_shape().rank() == 1);
            });
        },
        .pre_transform = [](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> OwnedConcatArgs {
            std::vector<ttnn::Tensor> itensors;
            itensors.reserve(tensors.size());
            std::transform(
                tensors.begin(),
                tensors.end(),
                std::back_inserter(itensors),
                [](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                    TT_FATAL(
                        input_tensor.layout() == Layout::TILE,
                        "ttnn.experimental.quasar.concat: expected all input tensors to be in tile layout");
                    ttsl::SmallVector<uint32_t> ends(
                        input_tensor.logical_shape().cbegin(), input_tensor.logical_shape().cend());
                    std::transform(ends.begin(), ends.end(), ends.begin(), [](const auto l) { return l - 1; });
                    return quasar::untilize_with_unpadding(input_tensor, ttnn::Shape(ends), std::nullopt);
                });
            return std::make_tuple(itensors, dim, groups);
        },
        .post_transform = [](const ttnn::Tensor& output) -> ttnn::Tensor {
            // now we have a rm tensor, so we need to re-tilize it
            if (output.layout() != Layout::TILE) {
                return quasar::tilize_with_val_padding(
                    output,
                    ttnn::operations::data_movement::compute_padded_shape(output.padded_shape()),
                    0.0f,
                    output.memory_config());
            }
            return output;
        },
        .operation = [output_memory_config](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int /*groups*/)
            -> ttnn::Tensor { return concat_impl(tensors, dim, output_memory_config); }});
}

MassagedConcat build_prepost_transpose_concat(const MemoryConfig& output_memory_config, int dim1, int dim2) {
    return MassagedConcat(MassagedConcatParams{
        .predicate = [dim1, dim2](const std::vector<ttnn::Tensor>& /*tensors*/, int /*dim*/, unsigned int /*groups*/)
            -> bool { return dim1 != dim2; },
        .pre_transform =
            [dim1, dim2](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> OwnedConcatArgs {
            std::vector<ttnn::Tensor> itensors;
            itensors.reserve(tensors.size());
            std::transform(
                tensors.begin(),
                tensors.end(),
                std::back_inserter(itensors),
                [dim1, dim2](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                    return transpose_through_tiles(input_tensor, dim1, dim2);
                });
            const auto& first_shape = tensors.front().logical_shape();
            auto norm_dim1 = first_shape.get_normalized_index(dim1);
            auto norm_dim2 = first_shape.get_normalized_index(dim2);
            int swapped_dim;
            if (dim == norm_dim1) {
                swapped_dim = norm_dim2;
            } else if (dim == norm_dim2) {
                swapped_dim = norm_dim1;
            } else {
                swapped_dim = dim;
            }
            return std::make_tuple(itensors, swapped_dim, groups);
        },
        .post_transform = [dim1, dim2, output_memory_config](const ttnn::Tensor& output) -> ttnn::Tensor {
            return transpose_through_tiles(output, dim1, dim2, output_memory_config);
        },
        .operation = [output_memory_config](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int /*groups*/)
            -> ttnn::Tensor { return concat_impl(tensors, dim, output_memory_config); }});
}

MassagedConcat build_non_aligned_last_dim_concat(const MemoryConfig& output_memory_config) {
    // this is a special case of pre-post transpose concat where we're
    // concatting on the last dim and the last dims of the input tensors are
    // not all aligned
    auto dim_aligned = [](const std::vector<ttnn::Tensor>& tensors, int dim) -> bool {
        return std::all_of(tensors.begin(), tensors.end(), [&](const ttnn::Tensor& tensor) {
            auto storage_type = tensor.storage_type();
            if (storage_type == StorageType::DEVICE) {
                return tensor.padded_shape()[dim] * tensor.element_size() % tensor.buffer()->alignment() == 0;
            }
            TT_THROW(
                "ttnn.experimental.quasar.concat: expected a tensor with device storage, but got a tensor with "
                "storage type {}",
                tensor.storage_type());
        });
    };

    auto predicate = [dim_aligned](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int /*groups*/) -> bool {
        auto last_dim = tensors.front().logical_shape().rank() - 1;
        if (dim == last_dim) {
            return !dim_aligned(tensors, dim);
        }
        return false;
    };

    auto transpose_concat = build_prepost_transpose_concat(output_memory_config, -2, -1);
    transpose_concat.set_predicate(predicate);
    return transpose_concat;
}

MassagedConcat build_unsqueeze_squeeze_1D_rm_unaligned_concat(const MemoryConfig& output_memory_config) {
    return MassagedConcat(MassagedConcatParams{
        .predicate = [](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int /*groups*/) -> bool {
            if (dim != 0) {
                return false;
            }
            return std::any_of(tensors.begin(), tensors.end(), [](const ttnn::Tensor& tensor) {
                return tensor.layout() == Layout::ROW_MAJOR and tensor.logical_shape().rank() == 1 and
                       tensor.logical_shape()[0] * tensor.element_size() % tensor.buffer()->alignment() != 0;
            });
        },
        .pre_transform = [](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> OwnedConcatArgs {
            std::vector<ttnn::Tensor> itensors;
            itensors.reserve(tensors.size());
            std::transform(
                tensors.begin(),
                tensors.end(),
                std::back_inserter(itensors),
                [](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                    TT_FATAL(
                        input_tensor.logical_shape().rank() == 1, "Expected 1D tensor for unsqueeze_squeeze_1D_concat");
                    // unsqueeze(0): [N] -> [1, N], a row-major view
                    return quasar::reshape(input_tensor, ttnn::Shape({1, input_tensor.logical_shape()[0]}));
                });
            return std::make_tuple(itensors, dim + 1, groups);
        },
        .post_transform = [](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto shape = output.logical_shape();
            TT_FATAL(shape.rank() == 2 && shape[0] == 1, "Expected 2D tensor with first dim=1, got shape {}", shape);
            // squeeze(0): [1, N] -> [N]
            return quasar::reshape(output, ttnn::Shape({shape[1]}));
        },
        .operation = [output_memory_config](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int /*groups*/)
            -> ttnn::Tensor { return concat_impl(tensors, dim, output_memory_config); }});
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::Tensor concat(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    unsigned int groups,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    TT_FATAL(!input_tensors.empty(), "ttnn.experimental.quasar.concat: expected a non-empty list of Tensors!");
    TT_FATAL(!optional_output_tensor.has_value(), "optional output tensor currently unsupported!");
    TT_FATAL(
        groups == 1,
        "ttnn.experimental.quasar.concat: groups > 1 (got {}) is implemented only by the L1 height-sharded "
        "zero-copy concat factory, which is not ported to Quasar",
        groups);
    // Same default as the original op: DRAM interleaved, not the inputs' memory config.
    const auto mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

    if (input_tensors.size() == 1) {
        return quasar::to_memory_config(input_tensors.at(0), mem_config);
    }

    const ttnn::Tensor& first_tensor = input_tensors.front();
    const int rank = first_tensor.logical_shape().rank();

    dim = first_tensor.logical_shape().get_normalized_index(dim);

    TT_FATAL(
        dim >= 0 and dim < rank,
        "ttnn: Dimension out of range: dim {} cannot be used for tensors of rank {}",
        dim,
        rank);

    const bool shapes_match =
        std::all_of(input_tensors.begin(), input_tensors.end(), [&first_tensor, dim](const ttnn::Tensor& t) {
            const auto& ft_shape = first_tensor.logical_shape();
            const auto& t_shape = t.logical_shape();

            const bool ranks_match = ft_shape.rank() == t_shape.rank();
            bool non_concat_dims_match = true;
            for (int i = 0; i < ft_shape.rank(); i++) {
                non_concat_dims_match &= dim == i or t_shape[i] == ft_shape[i];
            }
            return ranks_match and non_concat_dims_match;
        });

    TT_FATAL(
        shapes_match,
        "All dimensions must be the same size except for the dimension along which the contenation is taking place.");

    // For interleaved outputs, if sub_core_grids is provided, use direct path to avoid massaged operations
    // which don't currently support sub_core_grids
    if (sub_core_grids.has_value() && !first_tensor.is_sharded() &&
        (mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED)) {
        return concat_impl(input_tensors, dim, mem_config, sub_core_grids);
    }

    // Issue #43371: When concat is on the last dim and the last dim is not buffer-aligned,
    // the fallback path transposes dims -2/-1 so that the (small) last dim moves to dim[-2]
    // and concat proceeds along the new last dim.  If dim[-2] is very large the transposed
    // page size (element_size * dim[-2]) overflows L1.  Fix: chunk along dim[-2], concat
    // each chunk independently, then concat the results along dim[-2].
    // This applies to both TILE_LAYOUT (untilize -> RM -> transpose path) and ROW_MAJOR
    // (direct transpose path) when the last dim is not buffer-aligned.
    if (rank >= 2 && dim == rank - 1 && ttnn::is_device_tensor(first_tensor)) {
        const uint64_t second_last_dim = first_tensor.logical_shape()[rank - 2];
        const uint64_t elem_size = first_tensor.element_size();
        tt::tt_metal::distributed::MeshDevice* device = first_tensor.device();
        // Match the factory's DFB page alignment (common_align_len = max(input_alignment, output_alignment)).
        const uint64_t buf_align = std::max<uint64_t>(
            first_tensor.buffer()->alignment(), device->allocator()->get_alignment(mem_config.buffer_type()));
        const uint64_t l1_capacity =
            device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

        // Determine whether the transpose fallback will fire:
        // - TILE_LAYOUT: untilize produces RM first, then transpose fires if the RM last
        //   dim is non-aligned. This can only happen when tile padding exists on the concat dim.
        // - ROW_MAJOR: transpose fires directly when the last dim is not buffer-aligned.
        bool would_transpose = false;
        if (first_tensor.layout() == Layout::TILE) {
            bool has_tile_padding_on_concat_dim =
                std::any_of(input_tensors.begin(), input_tensors.end(), [dim](const ttnn::Tensor& tensor) {
                    return tensor.logical_shape()[dim] != tensor.padded_shape()[dim];
                });
            if (has_tile_padding_on_concat_dim) {
                would_transpose = std::any_of(
                    input_tensors.begin(), input_tensors.end(), [dim, buf_align](const ttnn::Tensor& tensor) {
                        return (static_cast<uint64_t>(tensor.logical_shape()[dim]) * tensor.element_size()) %
                                   buf_align !=
                               0;
                    });
            }
        } else if (first_tensor.layout() == Layout::ROW_MAJOR) {
            would_transpose =
                std::any_of(input_tensors.begin(), input_tensors.end(), [dim, buf_align](const ttnn::Tensor& tensor) {
                    return (static_cast<uint64_t>(tensor.logical_shape()[dim]) * tensor.element_size()) % buf_align !=
                           0;
                });
        }

        // Account for buffer alignment when estimating the post-transpose page size.
        const uint64_t raw_page = elem_size * second_last_dim;
        const uint64_t estimated_page_size = ((raw_page + buf_align - 1) / buf_align) * buf_align;

        if (would_transpose && estimated_page_size > l1_capacity) {
            // TILE_LAYOUT: the final dim[-2] concat operates on tiled chunk outputs, so
            // chunks must be tile-height-aligned. Read from tensor spec rather than
            // hardcoding. ROW_MAJOR: no tile boundary required, use alignment of 1.
            const uint32_t tile_h =
                (first_tensor.layout() == Layout::TILE) ? first_tensor.tensor_spec().tile().get_height() : 1;
            const uint32_t max_chunk_rows = static_cast<uint32_t>(l1_capacity / (2 * elem_size));
            const uint32_t chunk_rows = (max_chunk_rows / tile_h) * tile_h;
            TT_FATAL(
                chunk_rows > 0,
                "ttnn.experimental.quasar.concat: double-buffered tile-height chunk (2 x {} x {} = {} B) exceeds L1 "
                "capacity ({} B)",
                tile_h,
                elem_size,
                2 * tile_h * elem_size,
                l1_capacity);

            const uint32_t total_rows = second_last_dim;
            std::vector<ttnn::Tensor> chunk_outputs;
            chunk_outputs.reserve((total_rows + chunk_rows - 1) / chunk_rows);

            for (uint32_t row_start = 0; row_start < total_rows; row_start += chunk_rows) {
                const uint32_t row_end = std::min(row_start + chunk_rows, total_rows);

                std::vector<ttnn::Tensor> chunk_inputs;
                chunk_inputs.reserve(input_tensors.size());
                for (const auto& t : input_tensors) {
                    ttsl::SmallVector<uint32_t> starts(rank, 0);
                    ttsl::SmallVector<uint32_t> ends(rank);
                    for (int i = 0; i < rank; i++) {
                        ends[i] = t.logical_shape()[i];
                    }
                    starts[rank - 2] = row_start;
                    ends[rank - 2] = row_end;
                    ttsl::SmallVector<uint32_t> step(rank, 1);
                    chunk_inputs.push_back(quasar::slice(t, starts, ends, step, mem_config));
                }

                chunk_outputs.push_back(
                    quasar::concat(chunk_inputs, dim, memory_config, std::nullopt, groups, sub_core_grids));
            }

            if (chunk_outputs.size() == 1) {
                return chunk_outputs[0];
            }
            return quasar::concat(chunk_outputs, rank - 2, memory_config, std::nullopt, 1, sub_core_grids);
        }
    }

    auto untilize_rm_retilize_concat = build_untilize_rm_retilize_concat(mem_config);
    auto non_aligned_last_dim_concat = build_non_aligned_last_dim_concat(mem_config);
    auto unsqueeze_squeeze_1D_concat = build_unsqueeze_squeeze_1D_rm_unaligned_concat(mem_config);
    auto massaged_concat =
        untilize_rm_retilize_concat.sequence(unsqueeze_squeeze_1D_concat.sequence(non_aligned_last_dim_concat));

    return massaged_concat(input_tensors, dim, groups);
}

}  // namespace ttnn::operations::experimental::quasar
