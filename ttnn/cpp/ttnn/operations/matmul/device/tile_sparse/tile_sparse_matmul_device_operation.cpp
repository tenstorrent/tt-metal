// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/tile_sparse/tile_sparse_matmul_device_operation.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"

#include <tt-metalium/work_split.hpp>

namespace {

/**
 * @brief Computes the output shape of a tile-sparse matmul operation.
 *
 * Output shape follows standard matmul rules: [M, K] @ [K, N] = [M, N]
 * with batch dimensions preserved.
 */
ttnn::Shape compute_tile_sparse_matmul_output_shape(
    const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b) {
    const auto& input_shape_a = input_tensor_a.logical_shape();
    const auto& input_shape_b = input_tensor_b.logical_shape();

    const auto a_rank = input_shape_a.rank();
    const auto b_rank = input_shape_b.rank();

    // Determine batch dimensions
    uint32_t a_batched_dims = (a_rank <= 2) ? 0 : (a_rank - 2);
    uint32_t b_batched_dims = (b_rank <= 2) ? 0 : (b_rank - 2);

    // Output rank includes batch dims from both inputs
    uint32_t output_rank = std::max(a_batched_dims, b_batched_dims) + 2;

    // Initialize output shape
    ttnn::Shape output_shape(std::vector<uint32_t>(output_rank, 1));

    // Set M and N dimensions
    output_shape[-2] = input_shape_a[-2];  // M from A
    output_shape[-1] = input_shape_b[-1];  // N from B

    // Broadcast batch dimensions
    for (uint32_t i = 0; i < std::max(a_batched_dims, b_batched_dims); ++i) {
        uint32_t a_dim = (i < a_batched_dims) ? input_shape_a[a_batched_dims - 1 - i] : 1;
        uint32_t b_dim = (i < b_batched_dims) ? input_shape_b[b_batched_dims - 1 - i] : 1;
        output_shape[-3 - i] = std::max(a_dim, b_dim);
    }

    return output_shape;
}

}  // namespace

namespace ttnn::prim {

void TileSparseMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace operations::matmul::utilities;

    const auto& input_tensor_a = tensor_args.input_tensors.at(0);
    const auto& input_tensor_b = tensor_args.input_tensors.at(1);

    const auto& a_shape_padded = get_matmul_tensor_padded_shape(input_tensor_a, /*transpose=*/false);
    const auto& b_shape_padded = get_matmul_tensor_padded_shape(input_tensor_b, /*transpose=*/false);
    auto in0_tile = get_matmul_tile(input_tensor_a, /*transpose=*/false);
    auto in1_tile = get_matmul_tile(input_tensor_b, /*transpose=*/false);

    // Validate K dimension match
    TT_FATAL(
        a_shape_padded[-1] == b_shape_padded[-2],
        "Dimension K (A.shape[-1] {}) and B.shape[-2] ({}) must match for A and B",
        a_shape_padded[-1],
        b_shape_padded[-2]);

    // Validate tile alignment for A
    TT_FATAL(
        a_shape_padded[-2] % in0_tile.get_height() == 0,
        "A's rows ({}) must be divisible by tile height ({})",
        a_shape_padded[-2],
        in0_tile.get_height());
    TT_FATAL(
        a_shape_padded[-1] % in0_tile.get_width() == 0,
        "A's cols ({}) must be divisible by tile width ({})",
        a_shape_padded[-1],
        in0_tile.get_width());

    // Validate tile alignment for B
    TT_FATAL(
        b_shape_padded[-2] % in1_tile.get_height() == 0,
        "B's rows ({}) must be divisible by tile height ({})",
        b_shape_padded[-2],
        in1_tile.get_height());
    TT_FATAL(
        b_shape_padded[-1] % in1_tile.get_width() == 0,
        "B's cols ({}) must be divisible by tile width ({})",
        b_shape_padded[-1],
        in1_tile.get_width());

    // Validate sparsity masks if provided
    if (operation_attributes.input_a_sparsity_mask.has_value()) {
        const auto& mask_a = operation_attributes.input_a_sparsity_mask.value();
        uint32_t expected_tile_rows = a_shape_padded[-2] / operation_attributes.tile_height;
        uint32_t expected_tile_cols = a_shape_padded[-1] / operation_attributes.tile_width;
        TT_FATAL(
            mask_a.tile_rows == expected_tile_rows && mask_a.tile_cols == expected_tile_cols,
            "Sparsity mask A dimensions ({}, {}) must match tensor A tile dimensions ({}, {})",
            mask_a.tile_rows,
            mask_a.tile_cols,
            expected_tile_rows,
            expected_tile_cols);
    }

    if (operation_attributes.input_b_sparsity_mask.has_value()) {
        const auto& mask_b = operation_attributes.input_b_sparsity_mask.value();
        uint32_t expected_tile_rows = b_shape_padded[-2] / operation_attributes.tile_height;
        uint32_t expected_tile_cols = b_shape_padded[-1] / operation_attributes.tile_width;
        TT_FATAL(
            mask_b.tile_rows == expected_tile_rows && mask_b.tile_cols == expected_tile_cols,
            "Sparsity mask B dimensions ({}, {}) must match tensor B tile dimensions ({}, {})",
            mask_b.tile_rows,
            mask_b.tile_cols,
            expected_tile_rows,
            expected_tile_cols);
    }
}

TileSparseMatmulDeviceOperation::spec_return_value_t TileSparseMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace operations::matmul::utilities;

    TT_FATAL(tensor_args.optional_output_tensors.size() <= 1, "At most one optional output tensor can be provided");

    const bool is_output_tensor_given =
        !tensor_args.optional_output_tensors.empty() && tensor_args.optional_output_tensors.at(0).has_value();

    if (is_output_tensor_given) {
        return {tensor_args.optional_output_tensors.at(0)->tensor_spec()};
    }

    const auto& input_tensor_a = tensor_args.input_tensors.at(0);
    const auto& input_tensor_b = tensor_args.input_tensors.at(1);

    const auto output_shape = compute_tile_sparse_matmul_output_shape(input_tensor_a, input_tensor_b);

    const auto output_dtype = operation_attributes.output_dtype.has_value() ? operation_attributes.output_dtype.value()
                                                                            : input_tensor_a.dtype();

    auto in0_tile = get_matmul_tile(input_tensor_a, /*transpose=*/false);
    auto in1_tile = get_matmul_tile(input_tensor_b, /*transpose=*/false);

    tt::tt_metal::Tile output_tile = operations::matmul::utilities::get_output_tile(
        operation_attributes.output_mem_config,
        in0_tile,
        in1_tile,
        operation_attributes.output_tile,
        /*optional_output_tensor_tile=*/std::nullopt);

    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            output_dtype,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, output_tile),
            operation_attributes.output_mem_config))};
}

TileSparseMatmulDeviceOperation::tensor_return_value_t TileSparseMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TileSparseMatmulDeviceOperation::tensor_return_value_t output_tensors;
    const auto& optional_output_tensors = tensor_args.optional_output_tensors;
    const auto& input_tensors = tensor_args.input_tensors;

    if (!optional_output_tensors.empty() && optional_output_tensors[0].has_value()) {
        output_tensors.reserve(optional_output_tensors.size());
        for (const auto& optional_output_tensor : optional_output_tensors) {
            TT_FATAL(
                optional_output_tensor.has_value(),
                "If using optional output tensors, all output tensors must have a value");
            output_tensors.emplace_back(optional_output_tensor.value());
        }
        // Initialize output to zeros for accumulation
        for (auto& output_tensor : output_tensors) {
            output_tensor = ttnn::zeros_like(
                output_tensor,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::optional<Tensor>(output_tensor));
        }
        return output_tensors;
    }

    auto* device = input_tensors.at(0).device();
    const auto& output_specs = compute_output_specs(operation_attributes, tensor_args);
    output_tensors.reserve(output_specs.size());
    for (const auto& output_spec : output_specs) {
        output_tensors.emplace_back(create_device_tensor(output_spec, device));
    }
    // Initialize output to zeros for accumulation
    for (auto& output_tensor : output_tensors) {
        output_tensor = ttnn::zeros_like(
            output_tensor,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::optional<Tensor>(output_tensor));
    }
    return output_tensors;
}

std::tuple<TileSparseMatmulDeviceOperation::operation_attributes_t, TileSparseMatmulDeviceOperation::tensor_args_t>
TileSparseMatmulDeviceOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& sparsity_mask_a,
    const std::optional<const Tensor>& sparsity_mask_b,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<const MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreCoord>& user_core_coord,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    auto tile_sparse_attributes = TileSparseMatmulParams{
        std::nullopt,  // input_a_sparsity_mask - will be computed if mask tensor provided
        std::nullopt,  // input_b_sparsity_mask - will be computed if mask tensor provided
        program_config,
        memory_config.has_value() ? memory_config.value() : ttnn::DRAM_MEMORY_CONFIG,
        dtype,
        compute_kernel_config,
        user_core_coord,
        output_tile,
        global_cb,
        sub_device_id,
        32,  // tile_height
        32   // tile_width
    };

    // Parse user-provided sparsity mask tensors
    if (sparsity_mask_a.has_value()) {
        tile_sparse_attributes.input_a_sparsity_mask = parse_tile_sparsity_mask_tensor(
            sparsity_mask_a.value(), tile_sparse_attributes.tile_height, tile_sparse_attributes.tile_width);
    }
    if (sparsity_mask_b.has_value()) {
        tile_sparse_attributes.input_b_sparsity_mask = parse_tile_sparsity_mask_tensor(
            sparsity_mask_b.value(), tile_sparse_attributes.tile_height, tile_sparse_attributes.tile_width);
    }

    auto parameters = create_tile_sparse_matmul_attributes(
        input_tensor_a,
        input_tensor_b,
        sparsity_mask_a,
        sparsity_mask_b,
        tile_sparse_attributes,
        {optional_output_tensor});

    // Sparsity masks are parsed into TileSparseMatmulParams (host-side) and not needed
    // as device tensors. The factory reads from operation_attributes.input_*_sparsity_mask.
    return {
        parameters,
        TileSparseMatmulInputs{
            {input_tensor_a, input_tensor_b}, {std::nullopt, std::nullopt}, {optional_output_tensor}}};
}

TileSparseMatmulParams create_tile_sparse_matmul_attributes(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    [[maybe_unused]] const std::optional<const Tensor>& sparsity_mask_a,
    [[maybe_unused]] const std::optional<const Tensor>& sparsity_mask_b,
    const TileSparseMatmulParams& parameters,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    // Create standard matmul attributes for program config selection
    auto matmul_attributes = MatmulParams{
        parameters.program_config,
        /*bcast_batch=*/std::nullopt,
        parameters.output_mem_config,
        parameters.output_dtype,
        parameters.compute_kernel_config,
        /*untilize_out=*/false,
        parameters.user_core_coord,
        /*user_fused_activation=*/std::nullopt,
        /*user_run_batched=*/false,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        parameters.output_tile,
        parameters.global_cb,
        parameters.sub_device_id};

    auto matmul_struct = create_matmul_attributes(
        input_tensor_a,
        input_tensor_b,
        matmul_attributes,
        {optional_output_tensors.empty() ? std::nullopt : optional_output_tensors.at(0)});

    return TileSparseMatmulParams{
        parameters.input_a_sparsity_mask,
        parameters.input_b_sparsity_mask,
        matmul_struct.program_config,
        matmul_struct.output_mem_config,
        matmul_struct.output_dtype,
        matmul_struct.compute_kernel_config,
        matmul_struct.user_core_coord,
        matmul_struct.output_tile,
        matmul_struct.global_cb,
        matmul_struct.sub_device_id,
        parameters.tile_height,
        parameters.tile_width};
}

TileSparsityMask create_tile_sparsity_mask(
    const Tensor& dense_tensor, [[maybe_unused]] float threshold, uint32_t tile_height, uint32_t tile_width) {
    // Get tensor shape
    const auto& shape = dense_tensor.padded_shape();
    TT_FATAL(shape.rank() >= 2, "Tensor must have at least 2 dimensions");

    uint32_t M = shape[-2];
    uint32_t K = shape[-1];
    uint32_t tile_rows = M / tile_height;
    uint32_t tile_cols = K / tile_width;

    TT_FATAL(M % tile_height == 0, "Tensor height {} must be divisible by tile height {}", M, tile_height);
    TT_FATAL(K % tile_width == 0, "Tensor width {} must be divisible by tile width {}", K, tile_width);

    // For now, return a dense mask (all tiles non-zero)
    // Full implementation would analyze tensor contents
    std::vector<uint32_t> tile_indices;
    tile_indices.reserve(tile_rows * tile_cols);
    for (uint32_t i = 0; i < tile_rows * tile_cols; ++i) {
        tile_indices.push_back(i);
    }

    // Create mask tensor on host
    std::vector<uint8_t> mask_data(tile_rows * tile_cols, 1);
    TensorSpec spec(
        ttnn::Shape({tile_rows, tile_cols}),
        TensorLayout(DataType::UINT8, PageConfig(ttnn::ROW_MAJOR_LAYOUT), MemoryConfig{}));
    auto mask_tensor = Tensor::from_vector(mask_data, spec);

    return TileSparsityMask{
        std::move(mask_tensor),
        static_cast<uint32_t>(tile_indices.size()),
        std::move(tile_indices),
        tile_rows,
        tile_cols};
}

TileSparsityMask parse_tile_sparsity_mask_tensor(
    const Tensor& mask_tensor, [[maybe_unused]] uint32_t tile_height, [[maybe_unused]] uint32_t tile_width) {
    // Validate mask tensor is on host
    TT_FATAL(mask_tensor.storage_type() == StorageType::HOST, "Sparsity mask tensor must be on host");

    const auto& shape = mask_tensor.logical_shape();
    TT_FATAL(shape.rank() == 2, "Sparsity mask must be a 2D tensor, got rank {}", shape.rank());

    uint32_t tile_rows = shape[0];
    uint32_t tile_cols = shape[1];

    // Read mask values and build tile_indices
    std::vector<uint32_t> tile_indices;
    tile_indices.reserve(tile_rows * tile_cols);

    // Get mask data based on dtype
    auto dtype = mask_tensor.dtype();
    if (dtype == DataType::UINT8) {
        auto mask_data = mask_tensor.to_vector<uint8_t>();
        for (uint32_t i = 0; i < mask_data.size(); ++i) {
            if (mask_data[i] != 0) {
                tile_indices.push_back(i);
            }
        }
    } else if (dtype == DataType::UINT32) {
        auto mask_data = mask_tensor.to_vector<uint32_t>();
        for (uint32_t i = 0; i < mask_data.size(); ++i) {
            if (mask_data[i] != 0) {
                tile_indices.push_back(i);
            }
        }
    } else if (dtype == DataType::BFLOAT16) {
        auto mask_data = mask_tensor.to_vector<bfloat16>();
        for (uint32_t i = 0; i < mask_data.size(); ++i) {
            if (static_cast<float>(mask_data[i]) != 0.0f) {
                tile_indices.push_back(i);
            }
        }
    } else if (dtype == DataType::FLOAT32) {
        auto mask_data = mask_tensor.to_vector<float>();
        for (uint32_t i = 0; i < mask_data.size(); ++i) {
            if (mask_data[i] != 0.0f) {
                tile_indices.push_back(i);
            }
        }
    } else {
        TT_FATAL(false, "Unsupported sparsity mask dtype: {}", dtype);
    }

    uint32_t nnz_tiles = static_cast<uint32_t>(tile_indices.size());

    return TileSparsityMask{
        mask_tensor,  // Keep original mask tensor
        nnz_tiles,
        std::move(tile_indices),
        tile_rows,
        tile_cols};
}

}  // namespace ttnn::prim
