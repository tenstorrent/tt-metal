// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/device.hpp>

#include "ttnn/operations/data_movement/concat/device/concat_device_operation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.hpp"

#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

#include <ranges>
#include <utility>

// toggle this to enable debug prints
constexpr bool debug_concat = false;
inline void concat_db_print(bool condition, const std::string& msg) {
    if constexpr (debug_concat) {
        if (condition) {
            std::cout << "[DEBUG] concat: " << msg << std::endl;
        }
    }
}

namespace ttnn::operations::data_movement {

using OwnedConcatArgs = std::tuple<std::vector<ttnn::Tensor>, int, unsigned int>;

using MassagedConcat = MassagedOperation<ttnn::Tensor, const std::vector<ttnn::Tensor>&, int, unsigned int>;
using MassagedConcatParams = MassagedOperationParams<ttnn::Tensor, const std::vector<ttnn::Tensor>&, int, unsigned int>;

// FIXME: this papers over an issue in pad, so we should probably move the
// fix there.
MassagedConcat build_unsqueeze_concat(int input_rank, const MemoryConfig& output_memory_config) {
    return MassagedConcat(MassagedConcatParams{
        .predicate = [input_rank](
                         const std::vector<ttnn::Tensor>& tensors, int /*dim*/, unsigned int /*groups*/) -> bool {
            bool inputs_are_device_tensors =
                std::all_of(tensors.begin(), tensors.end(), [](const ttnn::Tensor& tensor) {
                    return tt::tt_metal::is_device_tensor(tensor);
                });
            bool res = input_rank < 4 && inputs_are_device_tensors;  // pad only rejects rank != 4 for device tensors
            concat_db_print(res, "unsqueeze to 4D required");
            return res;
        },
        .pre_transform =
            [input_rank](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> OwnedConcatArgs {
            std::vector<ttnn::Tensor> itensors;
            itensors.reserve(tensors.size());
            std::transform(
                tensors.begin(),
                tensors.end(),
                std::back_inserter(itensors),
                [](const ttnn::Tensor& input_tensor) -> ttnn::Tensor { return ttnn::unsqueeze_to_4D(input_tensor); });
            return std::make_tuple(itensors, dim + 4 - input_rank, groups);
        },
        .post_transform = [input_rank](const ttnn::Tensor& output) -> ttnn::Tensor {
            ttnn::Tensor res = output;
            while (res.logical_shape().rank() > input_rank) {
                const auto shape = res.logical_shape();
                const auto full_shape = res.padded_shape();
                ttsl::SmallVector<uint32_t> shape_vec{};
                ttsl::SmallVector<uint32_t> full_shape_vec{};
                for (int i = 1; i < shape.rank(); i++) {
                    shape_vec.push_back(shape[i]);
                    full_shape_vec.push_back(full_shape[i]);
                }
                res = ttnn::reshape(res, ttnn::Shape(std::move(shape_vec)), ttnn::Shape(std::move(full_shape_vec)));
            }
            return res;
        },
        .operation = [output_memory_config](
                         const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> ttnn::Tensor {
            const std::vector<ttnn::Tensor>& itensors(tensors);
            return ttnn::operations::data_movement::concat_impl(
                itensors, dim, groups, output_memory_config, std::nullopt);
        }});
}

MassagedConcat build_untilize_rm_retilize_concat(
    const MemoryConfig& output_memory_config, ttnn::Shape& logical_output_shape) {
    return MassagedConcat(MassagedConcatParams{
        .predicate = [](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int /*groups*/) -> bool {
            // untilize_rm_retilize if the concat dim is padded for tilized tensors
            bool res = std::any_of(tensors.begin(), tensors.end(), [&](const ttnn::Tensor& tensor) {
                return tensor.layout() == ttnn::TILE_LAYOUT and
                       (tensor.logical_shape()[dim] != tensor.padded_shape()[dim] or
                        tensor.logical_shape().rank() == 1);
            });
            concat_db_print(res, "untilize_rm_retilize required");
            return res;
        },
        .pre_transform = [output_memory_config](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups)
            -> OwnedConcatArgs {
            std::vector<ttnn::Tensor> itensors;
            itensors.reserve(tensors.size());
            std::transform(
                tensors.begin(),
                tensors.end(),
                std::back_inserter(itensors),
                [=](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                    TT_FATAL(
                        input_tensor.layout() == ttnn::TILE_LAYOUT,
                        "ttnn.concat: expected all input tensors to be in tile layout");
                    ttsl::SmallVector<uint32_t> ends(
                        input_tensor.logical_shape().cbegin(), input_tensor.logical_shape().cend());
                    std::transform(ends.begin(), ends.end(), ends.begin(), [](const auto l) { return l - 1; });
                    return ttnn::untilize_with_unpadding(input_tensor, ttnn::Shape(ends), std::nullopt);
                });
            return std::make_tuple(itensors, dim, groups);
        },
        .post_transform = [&logical_output_shape](const ttnn::Tensor& output) -> ttnn::Tensor {
            // now we have a rm tensor, so we need to re-tilize it
            if (output.layout() != ttnn::TILE_LAYOUT) {
                return ttnn::tilize_with_val_padding(
                    output, compute_padded_shape(output.padded_shape()), 0.0f, output.memory_config());
            }
            concat_db_print(true, "[DEBUG] already tilized");
            return output;
        },
        .operation = [&output_memory_config](
                         const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> ttnn::Tensor {
            const std::vector<ttnn::Tensor>& itensors(tensors);
            auto res = concat_impl(itensors, dim, groups, output_memory_config, std::nullopt);
            return res;
        }});
}

MassagedConcat build_prepost_transpose_concat(const MemoryConfig& output_memory_config, int dim1, int dim2) {
    return MassagedConcat(MassagedConcatParams{
        .predicate = [dim1, dim2](
                         const std::vector<ttnn::Tensor>& /*tensors*/, int /*dim*/, unsigned int /*groups*/) -> bool {
            bool res = dim1 != dim2;
            concat_db_print(res, "[DEBUG] pre-post transpose required");
            concat_db_print(!res, "[DEBUG] pre-post transpose not required");
            return res;
        },
        .pre_transform =
            [dim1, dim2](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> OwnedConcatArgs {
            std::vector<ttnn::Tensor> itensors;
            itensors.reserve(tensors.size());
            std::transform(
                tensors.begin(),
                tensors.end(),
                std::back_inserter(itensors),
                [dim1, dim2](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                    return ttnn::transpose(input_tensor, dim1, dim2);
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
        .post_transform = [dim1, dim2, &output_memory_config](const ttnn::Tensor& output) -> ttnn::Tensor {
            return ttnn::transpose(output, dim1, dim2, output_memory_config);
        },
        .operation = [output_memory_config](
                         const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> ttnn::Tensor {
            const std::vector<ttnn::Tensor>& itensors(tensors);
            return ttnn::operations::data_movement::concat_impl(
                itensors, dim, groups, output_memory_config, std::nullopt);
        }});
}

MassagedConcat build_non_aligned_last_dim_concat(
    const std::vector<ttnn::Tensor>& /*tensors*/, const MemoryConfig& output_memory_config) {
    // this is a special case of pre-post transpose concat where we're
    // concatting on the last dim and the last dims of the input tensors are
    // not all aligned
    auto dim_aligned = [](const std::vector<ttnn::Tensor>& tensors, int dim) -> bool {
        return std::all_of(tensors.begin(), tensors.end(), [&](const ttnn::Tensor& tensor) {
            auto storage_type = tensor.storage_type();
            if (storage_type == tt::tt_metal::StorageType::DEVICE) {
                return tensor.padded_shape()[dim] * tensor.element_size() % tensor.buffer()->alignment() == 0;
            }
            TT_THROW(
                "ttnn.concat: expected a tensor with device storage, but got a tensor with storage type"
                " {}",
                tensor.storage_type());
        });
    };

    auto predicate = [dim_aligned](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int /*groups*/) -> bool {
        auto last_dim = tensors.front().logical_shape().rank() - 1;
        if (dim == last_dim) {
            bool res = !dim_aligned(tensors, dim);
            concat_db_print(res, "[DEBUG] alignment fixedup required");
            return res;
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
            bool res = std::any_of(tensors.begin(), tensors.end(), [](const ttnn::Tensor& tensor) {
                return tensor.layout() == ttnn::ROW_MAJOR_LAYOUT and tensor.logical_shape().rank() == 1 and
                       tensor.logical_shape()[0] * tensor.element_size() % tensor.buffer()->alignment() != 0;
            });
            concat_db_print(res, "unsqueeze_squeeze_1D_concat required");
            return res;
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
                    return ttnn::unsqueeze(input_tensor, 0);
                });
            return std::make_tuple(itensors, dim + 1, groups);
        },
        .post_transform = [](const ttnn::Tensor& output) -> ttnn::Tensor {
            auto shape = output.logical_shape();
            TT_FATAL(shape.rank() == 2 && shape[0] == 1, "Expected 2D tensor with first dim=1, got shape {}", shape);
            return ttnn::squeeze(output, 0);
        },
        .operation = [output_memory_config](
                         const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> ttnn::Tensor {
            const std::vector<ttnn::Tensor>& itensors(tensors);
            return ttnn::operations::data_movement::concat_impl(
                itensors, dim, groups, output_memory_config, std::nullopt);
        }});
}

}  // namespace ttnn::operations::data_movement

namespace ttnn {

// Wrapper for TTDNN
ttnn::Tensor concat(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    unsigned int groups,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids) {
    TT_FATAL(!input_tensors.empty(), "ttnn.concat: expected a non-empty list of Tensors!");
    TT_FATAL(!optional_output_tensor.has_value(), "optional output tensor currently unsupported!");
    const auto mem_config =
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);  // should match input tensor memory config when unpopulated
                                                           // but causes CI errors for now

    if (input_tensors.size() == 1) {
        return ttnn::to_memory_config(input_tensors.at(0), mem_config, std::nullopt);
    }

    // TODO: Issue #8426: Add validation for ttnn.concat for sharded inputs
    // const bool all_tensors_are_tile_layout_without_padding = std::all_of(input_tensors.begin(), input_tensors.end(),
    // [dim](const ttnn::Tensor& input_tensor){
    //    return input_tensor.layout() == ttnn::TILE_LAYOUT and not has_tile_padding(input_tensor, dim);
    //});
    // TT_FATAL(all_tensors_are_tile_layout_without_padding, "Not Implemented");

    const ttnn::Tensor& first_tensor = input_tensors.front();
    const int rank = first_tensor.logical_shape().rank();

    dim = first_tensor.logical_shape().get_normalized_index(dim);

    TT_FATAL(
        dim >= 0 and dim < rank,
        "ttnn: Dimension out of range: dim {} cannot be used for tensors of rank {}",
        dim,
        rank);

    const bool shapes_match =
        std::all_of(input_tensors.begin(), input_tensors.end(), [first_tensor, dim](const ttnn::Tensor& t) {
            const auto& ft_shape = first_tensor.logical_shape();
            const auto& t_shape = t.logical_shape();

            const bool ranks_match = ft_shape.rank() == t_shape.rank();
            bool non_concat_dims_match = true;
            for (int i = 0; i < ft_shape.rank(); i++) {
                non_concat_dims_match &= dim == i or t_shape[i] == ft_shape[i];
            }
            // bool non_concat_padded_dims_match = true;
            // for(int i = 0; i < ft_shape.rank(); i++) {
            //     non_concat_padded_dims_match &= dim == i or t_shape.with_tile_padding()[i] ==
            //     ft_shape.with_tile_padding()[i];
            // }
            return ranks_match and non_concat_dims_match;  // and non_concat_padded_dims_match;
        });

    TT_FATAL(
        shapes_match,
        "All dimensions must be the same size except for the dimension along which the contenation is taking place.");

    auto compute_output_shape = [](const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Shape {
        ttnn::Shape shape_out = tensors[0].logical_shape();
        shape_out[dim] = 0;
        for (const Tensor& in_ref : tensors) {
            ttnn::Shape curr_shape = in_ref.logical_shape();
            shape_out[dim] += curr_shape[dim];
        }
        return shape_out;
    };

    ttnn::Shape logical_output_shape = compute_output_shape(input_tensors, dim);

    // For interleaved outputs, if sub_core_grids is provided, use direct path to avoid massaged operations
    // which don't currently support sub_core_grids
    if (sub_core_grids.has_value() && !first_tensor.is_sharded() &&
        (mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED)) {
        return ttnn::operations::data_movement::concat_impl(input_tensors, dim, groups, mem_config, sub_core_grids);
    }

    // Issue #43371: When concat is on the last dim and the last dim is not buffer-aligned,
    // the fallback path transposes dims -2/-1 so that the (small) last dim moves to dim[-2]
    // and concat proceeds along the new last dim.  If dim[-2] is very large the transposed
    // page size (element_size * dim[-2]) overflows L1.  Fix: chunk along dim[-2], concat
    // each chunk independently, then concat the results along dim[-2].
    // This applies to both TILE_LAYOUT (untilize -> RM -> transpose path) and ROW_MAJOR
    // (direct transpose path) when the last dim is not buffer-aligned.
    if (rank >= 2 && dim == rank - 1 && tt::tt_metal::is_device_tensor(first_tensor)) {
        const uint64_t second_last_dim = first_tensor.logical_shape()[rank - 2];
        const uint64_t elem_size = first_tensor.element_size();
        tt::tt_metal::IDevice* device = first_tensor.device();
        // Match the factory's CB page alignment (concat_program_factory.cpp uses
        // common_align_len = max(input_alignment, output_alignment)).
        const uint64_t buf_align = std::max<uint64_t>(
            first_tensor.buffer()->alignment(), device->allocator()->get_alignment(mem_config.buffer_type()));
        const uint64_t l1_capacity =
            device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

        // Determine whether the transpose fallback will fire:
        // - TILE_LAYOUT: untilize produces RM first, then transpose fires if the RM last
        //   dim is non-aligned. This can only happen when tile padding exists on the concat dim.
        // - ROW_MAJOR: transpose fires directly when the last dim is not buffer-aligned.
        bool would_transpose = false;
        if (first_tensor.layout() == ttnn::TILE_LAYOUT) {
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
        } else if (first_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) {
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
                (first_tensor.layout() == ttnn::TILE_LAYOUT) ? first_tensor.tensor_spec().tile().get_height() : 1;
            const uint32_t max_chunk_rows = static_cast<uint32_t>(l1_capacity / (2 * elem_size));
            const uint32_t chunk_rows = (max_chunk_rows / tile_h) * tile_h;
            TT_FATAL(
                chunk_rows > 0,
                "ttnn.concat: double-buffered tile-height chunk (2 x {} x {} = {} B) exceeds L1 capacity ({} B)",
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
                    chunk_inputs.push_back(ttnn::slice(t, starts, ends, step, mem_config));
                }

                chunk_outputs.push_back(
                    ttnn::concat(chunk_inputs, dim, memory_config, std::nullopt, groups, sub_core_grids));
            }

            if (chunk_outputs.size() == 1) {
                return chunk_outputs[0];
            }
            return ttnn::concat(chunk_outputs, rank - 2, memory_config, std::nullopt, 1, sub_core_grids);
        }
    }

    auto untilize_rm_retilize_concat =
        ttnn::operations::data_movement::build_untilize_rm_retilize_concat(mem_config, logical_output_shape);
    auto non_aligned_last_dim_concat =
        ttnn::operations::data_movement::build_non_aligned_last_dim_concat(input_tensors, mem_config);
    auto unsqueeze_squeeze_1D_concat =
        ttnn::operations::data_movement::build_unsqueeze_squeeze_1D_rm_unaligned_concat(mem_config);
    auto massaged_concat =
        untilize_rm_retilize_concat.sequence(unsqueeze_squeeze_1D_concat.sequence(non_aligned_last_dim_concat));

    const std::vector<ttnn::Tensor>& itensors(input_tensors);
    auto res = massaged_concat(itensors, dim, groups);
    return res;
}

}  // namespace ttnn
