// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/math.hpp>

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
        .predicate = [input_rank](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> bool {
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
                SmallVector<uint32_t> shape_vec{};
                SmallVector<uint32_t> full_shape_vec{};
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
            return concat_impl(itensors, dim, groups, output_memory_config);
        }});
}

MassagedConcat build_untilize_rm_retilize_concat(
    const MemoryConfig& output_memory_config, ttnn::Shape& logical_output_shape) {
    return MassagedConcat(MassagedConcatParams{
        .predicate = [](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> bool {
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
                    ttnn::SmallVector<uint32_t> ends(
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
            auto res = concat_impl(itensors, dim, groups, output_memory_config);
            return res;
        }});
}

MassagedConcat build_prepost_transpose_concat(const MemoryConfig& output_memory_config, int dim1, int dim2) {
    return MassagedConcat(MassagedConcatParams{
        .predicate = [dim1, dim2](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> bool {
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
            return concat_impl(itensors, dim, groups, output_memory_config);
        }});
}

MassagedConcat build_non_aligned_last_dim_concat(
    const std::vector<ttnn::Tensor>& tensors, const MemoryConfig& output_memory_config) {
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

    auto predicate = [dim_aligned](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> bool {
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
        .predicate = [](const std::vector<ttnn::Tensor>& tensors, int dim, unsigned int groups) -> bool {
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
            return concat_impl(itensors, dim, groups, output_memory_config);
        }});
}

// Wrapper for TTDNN
ttnn::Tensor ConcatOperation::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    unsigned int groups) {
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

    auto untilize_rm_retilize_concat = build_untilize_rm_retilize_concat(mem_config, logical_output_shape);
    auto non_aligned_last_dim_concat = build_non_aligned_last_dim_concat(input_tensors, mem_config);
    auto unsqueeze_squeeze_1D_concat = build_unsqueeze_squeeze_1D_rm_unaligned_concat(mem_config);
    auto massaged_concat =
        untilize_rm_retilize_concat.sequence(unsqueeze_squeeze_1D_concat.sequence(non_aligned_last_dim_concat));

    const std::vector<ttnn::Tensor>& itensors(input_tensors);
    auto res = massaged_concat(itensors, dim, groups);
    return res;
}

}  // namespace ttnn::operations::data_movement
