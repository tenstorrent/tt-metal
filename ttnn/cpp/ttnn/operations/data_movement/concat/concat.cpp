// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "tt_metal/common/math.hpp"

#include "ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"

#include <ranges>

// toggle this to enable debug prints
constexpr bool debug_concat = false;
inline void concat_db_print(bool condition, const std::string& msg) {
    if constexpr (debug_concat) {
        if (condition) {
            std::cout << "[DEBUG] concat: " << msg << std::endl;
        }
    }
}

namespace ttnn {
namespace operations {
namespace data_movement {
    using ConcatArgs = std::tuple<const std::vector<ttnn::Tensor>&, int>;
    using OwnedConcatArgs = std::tuple<std::vector<ttnn::Tensor>, int>;

    // FIXME: this papers over an issue in pad, so we should probably move the
    // fix there.
    MassagedOperation<ttnn::Tensor,
                      const std::vector<ttnn::Tensor>&,
                      int> build_unsqueeze_concat(int input_rank, ttnn::MemoryConfig& output_memory_config) {
        return MassagedOperation<ttnn::Tensor,
                                 const std::vector<ttnn::Tensor>&,
                                 int>(
            MassagedOperationParams<ttnn::Tensor,
                                    const std::vector<ttnn::Tensor>&,
                                    int> {
                .predicate = [input_rank](const std::vector<ttnn::Tensor>& tensors, int dim) -> bool {
                    bool inputs_are_device_tensors = std::all_of(tensors.begin(), tensors.end(), [](const ttnn::Tensor& tensor) {
                        return tensor.storage_type() != ttnn::StorageType::BORROWED && tensor.storage_type() != ttnn::StorageType::OWNED;
                    });
                    bool res = input_rank < 4 && inputs_are_device_tensors; // pad only rejects rank != 4 for device tensors
                    concat_db_print(res, "unsqueeze to 4D required");
                    return res;
                },
                .pre_transform = [input_rank](const std::vector<ttnn::Tensor>& tensors, int dim) -> OwnedConcatArgs {
                    std::vector<ttnn::Tensor> itensors;
                    itensors.reserve(tensors.size());
                    std::transform(
                        tensors.begin(),
                        tensors.end(),
                        std::back_inserter(itensors),
                        [](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                            return ttnn::unsqueeze_to_4D(input_tensor);
                        }
                    );
                    return std::make_tuple(itensors, dim + 4 - input_rank);
                },
                .post_transform = [input_rank](const ttnn::Tensor& output, const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                    ttnn::Tensor res = output;
                    while (res.get_shape().rank() > input_rank) {
                        const auto shape = res.get_shape();
                        const auto full_shape = res.get_shape().with_tile_padding();
                        std::vector<uint32_t> shape_vec{};
                        std::vector<uint32_t> full_shape_vec{};
                        for (int i = 1; i < shape.rank(); i++) {
                            shape_vec.push_back(shape[i]);
                            full_shape_vec.push_back(full_shape[i]);
                        }
                        res = ttnn::reshape(res, ttnn::Shape(shape_vec, full_shape_vec));
                    }
                    return res;
                },
                .operation = [output_memory_config](const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                    std::vector<ttnn::Tensor> itensors(tensors);
                    return concat_impl(itensors, dim, output_memory_config);
                }
            }
        );
    }

    MassagedOperation<ttnn::Tensor,
                      const std::vector<ttnn::Tensor>&,
                      int> build_untilize_rm_retilize_concat(uint8_t queue_id, MemoryConfig &output_memory_config) {
        return MassagedOperation<ttnn::Tensor,
                                 const std::vector<ttnn::Tensor>&,
                                 int>(
            MassagedOperationParams<ttnn::Tensor, const std::vector<ttnn::Tensor>&, int> {
                .predicate = [](const std::vector<ttnn::Tensor>& tensors, int dim) -> bool {
                    // untilize_rm_retilize if the concat dim is padded for tilized tensors
                    bool res = std::any_of(tensors.begin(), tensors.end(), [&](const ttnn::Tensor& tensor) {
                        return tensor.get_layout() == ttnn::TILE_LAYOUT and tensor.get_logical_shape()[dim] != tensor.get_padded_shape()[dim];
                    });
                    concat_db_print(res, "untilize_rm_retilize required");
                    return res;
                },
                .pre_transform = [](const std::vector<ttnn::Tensor>& tensors, int dim) -> OwnedConcatArgs {
                    std::vector<ttnn::Tensor> itensors;
                    itensors.reserve(tensors.size());
                    std::transform(
                        tensors.begin(),
                        tensors.end(),
                        std::back_inserter(itensors),
                        [](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                            TT_FATAL(input_tensor.get_layout() == ttnn::TILE_LAYOUT, "ttnn.concat: expected all input tensors to be in tile layout");
                            auto untilized_tensor = ttnn::untilize(input_tensor);
                            // untilized, so now we have a padded rm tensor
                            untilized_tensor.set_shape(ttnn::Shape {input_tensor.get_logical_shape().as_vector(),
                                                                    untilized_tensor.get_padded_shape().as_vector()});
                            return untilized_tensor;
                        }
                    );
                    return std::make_tuple(itensors, dim);
                },
                .post_transform = [queue_id](const ttnn::Tensor& output, const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                    // now we have a rm tensor, so we need ensure its's padded to tile size and re-tilize it
                    if (output.get_layout() != ttnn::TILE_LAYOUT) {
                        auto padded = pad_to_tile_vol(queue_id,
                                                      output,
                                                      0.0f,
                                                      true,
                                                      output.memory_config());
                        concat_db_print(true, "[DEBUG] padded to tile layout, now tilizing.");
                        auto tilized = ttnn::tilize_with_val_padding(padded, padded.get_legacy_shape(), 0.0f, output.memory_config());
                        concat_db_print(true, "[DEBUG] tilized");
                        return tilized;
                    }
                    concat_db_print(true, "[DEBUG] already tilized");
                    return output;
                },
                .operation = [output_memory_config](const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                    std::vector<ttnn::Tensor> itensors(tensors);
                    return concat_impl(itensors, dim, output_memory_config);
                }
            }
        );
    }

    MassagedOperation<ttnn::Tensor,
                      const std::vector<ttnn::Tensor>&,
                      int> build_prepost_transpose_concat(uint8_t queue_id, MemoryConfig &output_memory_config, int dim1, int dim2) {
        return MassagedOperation<ttnn::Tensor,
                                 const std::vector<ttnn::Tensor>&,
                                 int>(
            MassagedOperationParams<ttnn::Tensor, const std::vector<ttnn::Tensor>&, int> {
                .predicate = [dim1, dim2](const std::vector<ttnn::Tensor>& tensors, int dim) -> bool {
                    bool res = dim1 != dim2;
                    concat_db_print(res, "[DEBUG] pre-post transpose required");
                    concat_db_print(!res, "[DEBUG] pre-post transpose not required");
                    return res;
                },
                .pre_transform = [dim1, dim2](const std::vector<ttnn::Tensor>& tensors, int dim) -> OwnedConcatArgs {
                    std::vector<ttnn::Tensor> itensors;
                    itensors.reserve(tensors.size());
                    std::transform(
                        tensors.begin(),
                        tensors.end(),
                        std::back_inserter(itensors),
                        [dim1, dim2](const ttnn::Tensor& input_tensor) -> ttnn::Tensor {
                            return ttnn::transpose(input_tensor, dim1, dim2, std::nullopt);
                        }
                    );
                    auto norm_dim1 = tensors.front().get_shape().get_normalized_index(dim1);
                    auto norm_dim2 = tensors.front().get_shape().get_normalized_index(dim2);
                    int swapped_dim;
                    if (dim == norm_dim1) {
                        swapped_dim = norm_dim2;
                    } else if (dim == norm_dim2) {
                        swapped_dim = norm_dim1;
                    } else {
                        swapped_dim = dim;
                    }
                    return std::make_tuple(itensors, swapped_dim);
                },
                .post_transform = [dim1, dim2, &output_memory_config](const ttnn::Tensor& output, const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                    return ttnn::transpose(output, dim1, dim2, output_memory_config);
                },
                .operation = [output_memory_config](const std::vector<ttnn::Tensor>& tensors, int dim) -> ttnn::Tensor {
                    std::vector<ttnn::Tensor> itensors(tensors);
                    return concat_impl(itensors, dim, output_memory_config);
                }
            }
        );
    }

    MassagedOperation<ttnn::Tensor,
                      const std::vector<ttnn::Tensor>&,
                      int> build_non_aligned_last_dim_concat(const std::vector<ttnn::Tensor>& tensors, uint8_t queue_id, MemoryConfig &output_memory_config) {
        // this is a special case of pre-post transpose concat where we're
        // concatting on the last dim and the last dims of the input tensors are
        // not all aligned
        auto dim_aligned = [](const std::vector<ttnn::Tensor>& tensors, int dim) -> bool {
            return std::all_of(tensors.begin(), tensors.end(), [&](const ttnn::Tensor& tensor) {
                return tensor.get_padded_shape()[dim] * tensor.element_size() % tensor.buffer()->alignment() == 0;
            });
        };

        auto predicate = [dim_aligned](const std::vector<ttnn::Tensor>& tensors, int dim) -> bool {
            auto last_dim = tensors.front().get_shape().rank() - 1;
            if (dim == last_dim) {
                bool res = !dim_aligned(tensors, dim);
                concat_db_print(res, "[DEBUG] alignment fixedup required");
                return res;
            }
            return false;
        };

        auto transpose_concat = build_prepost_transpose_concat(queue_id,
                                                               output_memory_config,
                                                               -2,
                                                               -1);
        transpose_concat.set_predicate(predicate);
        return transpose_concat;
    }

// Wrapper for TTDNN
ttnn::Tensor ConcatOperation::invoke(
    uint8_t queue_id,
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::Tensor> optional_output_tensor,
    unsigned int groups) {
    TT_FATAL(input_tensors.size() > 0, "ttnn.concat: expected a non-empty list of Tensors!");
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
    //    return input_tensor.get_layout() == ttnn::TILE_LAYOUT and not has_tile_padding(input_tensor, dim);
    //});
    // TT_FATAL(all_tensors_are_tile_layout_without_padding, "Not Implemented");

    const ttnn::Tensor& first_tensor = input_tensors.front();
    const int rank = first_tensor.get_shape().rank();

    dim = first_tensor.get_legacy_shape().get_normalized_index(dim);

    TT_FATAL(
        dim >= 0 and dim < rank,
        "ttnn: Dimension out of range: dim {} cannot be used for tensors of rank {}",
        dim,
        rank);

    const bool shapes_match =
        std::all_of(input_tensors.begin(), input_tensors.end(), [first_tensor, dim](const ttnn::Tensor& t) {
            const auto& ft_shape = first_tensor.get_shape();
            const auto& t_shape = t.get_shape();

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

        auto output_memory_config = memory_config.value_or(first_tensor.memory_config());
        auto unsqueeze_concat = build_unsqueeze_concat(rank, output_memory_config);
        auto untilize_rm_retilize_concat = build_untilize_rm_retilize_concat(queue_id, output_memory_config);
        auto non_aligned_last_dim_concat = build_non_aligned_last_dim_concat(input_tensors, queue_id, output_memory_config);
        std::vector<ttnn::Tensor> itensors(input_tensors);
        auto massaged_concat = untilize_rm_retilize_concat.sequence(non_aligned_last_dim_concat);
        return massaged_concat(input_tensors, dim);
    }

ttnn::Tensor ConcatOperation::invoke(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::Tensor> optional_output_tensor,
    unsigned int groups) {
    return invoke(DefaultQueueId, input_tensors, dim, memory_config, optional_output_tensor, groups);
}

}  // namespace data_movement
}  // namespace operations
}  // namespace ttnn
