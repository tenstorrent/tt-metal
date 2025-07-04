// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "../device/cumulation_device_operation_types.hpp"
#include "../device/cumulation_device_operation.hpp"

#include "cumsum.hpp"

#include <algorithm>
#include <iterator>
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/shape.hpp"
#include <tt_stl/small_vector.hpp>
#include "tt-metalium/tile.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::reduction::cumulation {

uint64_t compute_padded_volume(const Shape& logical_shape, const tt::tt_metal::Tile& tile) {
    unsigned tile_width = tile.get_width();
    unsigned tile_height = tile.get_height();

    unsigned width = 1;
    unsigned height = 1;

    int rank = logical_shape.rank();

    if (rank >= 1) {
        width = logical_shape[-1];
    }
    if (rank >= 2) {
        height = logical_shape[-2];
    }

    // Round up width to the next multiple of tile_width
    uint64_t padded_width = ((width + tile_width - 1) / tile_width) * tile_width;

    // Round up height to the next multiple of tile height
    uint64_t padded_height = ((height + tile_height - 1) / tile_height) * tile_height;

    // Compute new padded volume: padded_width * padded_height * old_channels * old_batches * ...
    uint64_t volume = padded_width * padded_height;
    for (int i = 0; i < rank - 2; i++) {
        volume *= logical_shape[i];
    }

    return volume;
}

Tensor CumsumOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> optional_output_tensor,
    const bool& flip,
    const std::optional<MemoryConfig>& memory_config) {
    const auto& input_shape = input_tensor.logical_shape();
    int tensor_rank = input_shape.rank();

    Tensor adjusted_input_tensor = input_tensor;  // Tensor copy, but simplifies code (temporary solution)
    const auto& input_dtype = input_tensor.dtype();

    if (dtype.has_value() && input_dtype != dtype.value()) {
        // auto converted_tensor = ttnn::to_dtype(input_tensor, DataType::BFLOAT16);
        // adjusted_input_tensor = converted_tensor;

        // Ideally, we would use `ttnn::to_dtype()` directly on input_tensor (DeviceStorage)
        // However, as of writing `ttnn::to_dtype()` does not support this.
        // The (provisional) workaround is to move the tensor to CPU, do the type conversion
        // and bring it back to the device.
        Tensor cpu_tensor = input_tensor.cpu();
        Tensor cpu_converted_tensor = ttnn::to_dtype(cpu_tensor, dtype.value());

        Tensor converted_tensor = cpu_converted_tensor.to_device(input_tensor.device(), input_tensor.memory_config());

        adjusted_input_tensor = converted_tensor;
    }

    constexpr uint32_t FOUR_DIMENSIONS{4};
    constexpr uint32_t FIRST_DIMENSION{0};

    if (tensor_rank == 0 || adjusted_input_tensor.logical_volume() == 0) {  // empty input tensor => nothing to do
        return adjusted_input_tensor;
    }

    // Normalize negative dim
    int32_t cum_axis{dim};
    if (cum_axis < 0) {
        cum_axis += tensor_rank;
    }

    // If dim is x or y axis (last or second last dimension)
    if (tensor_rank - cum_axis < FOUR_DIMENSIONS) {
        auto opt_output = optional_output_tensor;

        int initial_tensor_rank = tensor_rank;

        if (initial_tensor_rank < FOUR_DIMENSIONS) {
            ttnn::SmallVector<uint32_t> new_dims = {};
            for (int i{initial_tensor_rank}; i < FOUR_DIMENSIONS; ++i) {
                new_dims.push_back(1);
            }
            new_dims.insert(new_dims.end(), input_shape.cbegin(), input_shape.cend());
            ttnn::Shape new_shape(new_dims);

            adjusted_input_tensor = ttnn::reshape(adjusted_input_tensor, new_shape);

            // Update params
            tensor_rank = FOUR_DIMENSIONS;
            cum_axis += (FOUR_DIMENSIONS - initial_tensor_rank);
        }

        // For now, the cumsum does not support `dim` == x or y-axis.
        // For now, we make the operation compatible by permuting axes if `dim` is either x or y axes.

        // If input tensor is 1D (or 2D with few rows), then permuting it can significantly increase its
        // memory footprint if using a tile layout
        // For instance (dim = -1):
        // input shape = [32, 32] (1 tile)
        // output shape = [32, 32, 1] (32 tiles)

        // To detect problems, we compute footprint of new tensor ahead of its permutation
        // and display error if it exceeds 30% of all DRAM
        // Note: This is a 'temporary' limitation: permute-free accumulation on x and y axes is planned
        const Shape tensor_shape = adjusted_input_tensor.logical_shape();
        const uint64_t old_volume = adjusted_input_tensor.physical_volume();

        ttnn::SmallVector<uint32_t> new_dims(tensor_shape.cbegin(), tensor_shape.cend());
        std::swap(new_dims[0], new_dims[cum_axis]);
        const Shape new_shape(new_dims);

        const uint64_t new_volume = compute_padded_volume(new_shape, adjusted_input_tensor.tensor_spec().tile());

        if (new_volume > old_volume) {
            const uint64_t element_size = adjusted_input_tensor.element_size();
            const uint64_t growth = (new_volume - old_volume) * element_size;
            constexpr uint64_t ONE_MB = (1024 * 1024);
            constexpr uint64_t EIGHT_KB = (1024 * 4);
            constexpr uint64_t MAX_ALLOWED_GROWTH = 800 * ONE_MB;  // 800 MiB

            TT_FATAL(old_volume > 0, "Can not compute permuted tensor for cumsum operation if input is empty");
            if (new_volume >= 2 * old_volume && growth >= EIGHT_KB) {
                log_warning(
                    tt::LogOp,
                    "Intermediate tensor of cumsum exceeds input by a factor of {}, input size = {} MiB, intermediate "
                    "size = {} MiB",
                    new_volume / old_volume,
                    old_volume / ONE_MB,
                    new_volume / ONE_MB);
            }

            TT_FATAL(
                growth <= MAX_ALLOWED_GROWTH,
                "Permuted tensor for cumsum would have size of {} MiB, which exceeds initial input of size {} MiB by "
                "{} MiB, which is more than {} MiB limit",
                new_volume * element_size / ONE_MB,
                old_volume * element_size / ONE_MB,
                growth / ONE_MB,
                MAX_ALLOWED_GROWTH / ONE_MB);
        }

        // Create permutation that just swaps dim with dim=0
        ttnn::SmallVector<int64_t> permutation(tensor_rank);
        std::iota(permutation.begin(), permutation.end(), FIRST_DIMENSION);  // Initialize to [0,1,2,...]
        permutation[FIRST_DIMENSION] = cum_axis;                             // Swap dim with dim=0
        permutation[cum_axis] = FIRST_DIMENSION;

        Tensor permuted_tensor =
            ttnn::permute(adjusted_input_tensor, permutation, adjusted_input_tensor.memory_config());

        // Compute cumsum on permuted tensor (now accumulation is on dim=0)
        Tensor output_tensor = ttnn::prim::cumulation(
            queue_id, permuted_tensor, FIRST_DIMENSION, dtype, std::nullopt, memory_config, flip, CumulationOp::CUMSUM);

        // Apply backward permutation to restore initial shape
        output_tensor = ttnn::permute(output_tensor, permutation, output_tensor.memory_config());

        if (initial_tensor_rank < FOUR_DIMENSIONS) {
            output_tensor = ttnn::reshape(output_tensor, input_shape);
        }

        if (opt_output.has_value()) {
            auto& out_tensor = optional_output_tensor.value();
            out_tensor.storage() = output_tensor.storage();
        }

        return output_tensor;
    }

    // For other dimensions, proceed with original cumsum
    return ttnn::prim::cumulation(
        queue_id,
        adjusted_input_tensor,
        cum_axis,
        dtype,
        optional_output_tensor,
        memory_config ? memory_config : adjusted_input_tensor.memory_config(),
        flip,
        CumulationOp::CUMSUM);
}

Tensor CumsumBackwardOperation::invoke(
    QueueId queue_id,
    const Tensor& input,
    int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> optional_output_tensor,
    const std::optional<MemoryConfig>& memory_config) {
    return CumsumOperation::invoke(queue_id, input, dim, dtype, optional_output_tensor, true, memory_config);
}

}  // namespace ttnn::operations::reduction::cumulation
