// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"

#include <tt-metalium/work_split.hpp>

namespace {

/**
 * @brief Computes the output shape of a sparse matmul operation given two input tensors.
 *
 * The output shape for a sparse matmul is the same as for a dense matmul, but allows for
 * batching on both input tensors.
 * The final output shape as batched dimensions from input B first (inner), then input A (outer).
 * @param input_tensor_a First input tensor
 * @param input_tensor_b Second input tensor
 * @return Shape of the resulting tensor after sparse matmul
 */
ttnn::Shape compute_sparse_matmul_output_shape(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    bool is_input_a_sparse,
    bool is_input_b_sparse) {
    const auto& input_shape_a = input_tensor_a.logical_shape();
    const auto& input_shape_b = input_tensor_b.logical_shape();

    const auto a_rank = input_shape_a.rank();
    const auto b_rank = input_shape_b.rank();

    // Decide the rank of the output shape based on batch dimensions in input tensors
    // Find batched dimensions in both. Add batched dimensions from both to output rank and then add 2
    // Batched dimensions are all dimensions except the last two
    uint32_t a_batched_dims = ((is_input_a_sparse && is_input_b_sparse) || (a_rank <= 2)) ? 0 : (a_rank - 2);
    uint32_t b_batched_dims = ((is_input_a_sparse && !is_input_b_sparse) || (b_rank <= 2)) ? 0 : (b_rank - 2);
    uint32_t output_rank = a_batched_dims + b_batched_dims + 2;

    // Initialize output shape with zeros based on the output rank
    ttnn::Shape output_shape(std::vector<uint32_t>(output_rank, 0));

    // First pick the M and N dimensions from the input tensors
    output_shape[-2] = input_shape_a[-2];
    output_shape[-1] = input_shape_b[-1];

    // Add batched dims from input B to output shape
    for (uint32_t i = 0; i < b_batched_dims; ++i) {
        output_shape[-3 - i] = input_shape_b[-3 - i];
    }

    // Add batched dims from input A to output shape
    for (uint32_t i = 0; i < a_batched_dims; ++i) {
        output_shape[-3 - b_batched_dims - i] = input_shape_a[-3 - i];
    }

    return output_shape;
}
}  // namespace

namespace ttnn::prim {
SparseMatmulDeviceOperation::program_factory_t SparseMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return SparseMatmulMultiCoreReuseMcast1DProgramFactory{};
}

void SparseMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace operations::matmul::utilities;
    const auto& input_tensor_a = tensor_args.input_tensors.at(0);
    const auto& input_tensor_b = tensor_args.input_tensors.at(1);
    const auto& sparsity = tensor_args.input_tensors.at(2);

    const auto& a_shape_padded = get_matmul_tensor_padded_shape(input_tensor_a, /*transpose=*/false);
    const auto& b_shape_padded = get_matmul_tensor_padded_shape(input_tensor_b, /*transpose=*/false);
    auto in0_tile = get_matmul_tile(input_tensor_a, /*transpose=*/false);
    auto in1_tile = get_matmul_tile(input_tensor_b, /*transpose=*/false);

    TT_FATAL(
        a_shape_padded[-1] == b_shape_padded[-2],
        "Dimension K (A.shape[-1] {}) and B.shape[-2] ({}) must match for A and B",
        a_shape_padded[-1],
        b_shape_padded[-2]);
    TT_FATAL(
        a_shape_padded[-2] % in0_tile.get_height() == 0,
        "a_shape_padded[-2] (A's rows: {}) must be divisible by in0_tile.get_height() (A's tile height: {}) for "
        "tilization. "
        "a_shape_padded: {}, in0_tile: {}",
        a_shape_padded[-2],
        in0_tile.get_height(),
        a_shape_padded,
        in0_tile);
    TT_FATAL(
        a_shape_padded[-1] % in0_tile.get_width() == 0,
        "a_shape_padded[-1] (A's cols: {}) must be divisible by in0_tile.get_width() (A's tile width: {}) for "
        "tilization. "
        "a_shape_padded: "
        "{}, in0_tile: {}",
        a_shape_padded[-1],
        in0_tile.get_width(),
        a_shape_padded,
        in0_tile);
    TT_FATAL(
        b_shape_padded[-2] % in1_tile.get_height() == 0,
        "b_shape_padded[-2] (B's rows: {}) must be divisible by in1_tile.get_height() (B's tile height: {}) for "
        "tilization. "
        "b_shape_padded: {}, in1_tile_shape: {}",
        b_shape_padded[-2],
        in1_tile.get_height(),
        b_shape_padded,
        in1_tile);
    TT_FATAL(
        b_shape_padded[-1] % in1_tile.get_width() == 0,
        "b_shape_padded[-1] (B's cols: {}) must be divisible by in1_tile_shape[1] (B's tile width: {}) for tilization. "
        "b_shape_padded: "
        "{}, in1_tile: {}",
        b_shape_padded[-1],
        in1_tile.get_width(),
        b_shape_padded,
        in1_tile);
    TT_FATAL(
        operation_attributes.nnz.value_or(1) > 0, "nnz ({}) must be greater than 0", operation_attributes.nnz.value());

    // Check that nnz is less than or equal to the length of all batch dimensions
    uint32_t batch_length_A = 1;
    if (a_shape_padded.rank() > 2) {
        for (int i = 0; i < a_shape_padded.rank() - 2; ++i) {
            batch_length_A *= a_shape_padded[i];
        }
    }

    uint32_t batch_length_B = 1;
    if (b_shape_padded.rank() > 2) {
        for (int i = 0; i < b_shape_padded.rank() - 2; ++i) {
            batch_length_B *= b_shape_padded[i];
        }
    }

    uint32_t batch_length = 0;
    if (operation_attributes.is_input_a_sparse && operation_attributes.is_input_b_sparse) {
        batch_length = batch_length_B;
    } else if (operation_attributes.is_input_a_sparse) {
        batch_length = batch_length_A;
    } else {
        batch_length = batch_length_A * batch_length_B;
    }

    // Check that sparsity has enough entries
    TT_FATAL(
        sparsity.logical_volume() == batch_length,
        "sparsity.logical_volume() ({}) must be equal to the product of all batch dimensions ({})",
        sparsity.logical_volume(),
        batch_length);

    TT_FATAL(
        operation_attributes.nnz.value_or(1) <= batch_length,
        "nnz ({}) must be less than or equal to the length of all batch dimensions ({})",
        operation_attributes.nnz,
        batch_length);
}

SparseMatmulDeviceOperation::spec_return_value_t SparseMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace operations::matmul::utilities;
    TT_FATAL(
        tensor_args.optional_output_tensors.size() <= 1,
        "None or One Optional output tensor can be passed when accessing it "
        "for computing SparseMatmul's output specs");

    const bool is_output_tensor_given =
        !tensor_args.optional_output_tensors.empty() && tensor_args.optional_output_tensors.at(0).has_value();

    if (is_output_tensor_given) {
        return {tensor_args.optional_output_tensors.at(0)->tensor_spec()};
    }

    const auto& input_tensor_a = tensor_args.input_tensors.at(0);
    const auto& input_tensor_b = tensor_args.input_tensors.at(1);

    const auto output_shape = compute_sparse_matmul_output_shape(
        input_tensor_a, input_tensor_b, operation_attributes.is_input_a_sparse, operation_attributes.is_input_b_sparse);

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

SparseMatmulDeviceOperation::tensor_return_value_t SparseMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    SparseMatmulDeviceOperation::tensor_return_value_t output_tensors;
    const auto& optional_output_tensors = tensor_args.optional_output_tensors;
    const auto& input_tensors = tensor_args.input_tensors;

    if (!optional_output_tensors.empty() and optional_output_tensors[0].has_value()) {
        output_tensors.reserve(optional_output_tensors.size());
        for (const auto& optional_output_tensor : optional_output_tensors) {
            TT_FATAL(
                optional_output_tensor.has_value(),
                "If using optional output tensors, all output tensors must have a value");
            output_tensors.emplace_back(optional_output_tensor.value());
        }
        return output_tensors;
    }
    const auto& device = input_tensors.at(0).device();
    const auto& output_specs = compute_output_specs(operation_attributes, tensor_args);
    output_tensors.reserve(output_specs.size());
    for (const auto& output_spec : output_specs) {
        output_tensors.emplace_back(create_device_tensor(output_spec, device));
    }
    return output_tensors;
}

void SparseMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

// static tt::stl::hash::hash_t SparseMatmulDeviceOperation::compute_program_hash(
//     const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

std::tuple<SparseMatmulDeviceOperation::operation_attributes_t, SparseMatmulDeviceOperation::tensor_args_t>
SparseMatmulDeviceOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& sparsity,
    const std::optional<Tensor>& optional_output_tensor,
    std::optional<uint32_t> nnz,
    bool is_input_a_sparse,
    bool is_input_b_sparse,
    const std::optional<const MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    const std::optional<const operations::matmul::MatmulProgramConfig>& program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreCoord>& user_core_coord,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    auto sparse_matmul_attributes = SparseMatmulParams{
        nnz,
        is_input_a_sparse,
        is_input_b_sparse,
        program_config,
        memory_config.has_value() ? memory_config.value() : ttnn::DRAM_MEMORY_CONFIG,
        dtype,
        compute_kernel_config,
        user_core_coord,
        output_tile,
        global_cb,
        sub_device_id};

    auto parameters = create_sparse_matmul_attributes(
        input_tensor_a, input_tensor_b, sparsity, sparse_matmul_attributes, {optional_output_tensor});

    return {parameters, SparseMatmulInputs{{input_tensor_a, input_tensor_b, sparsity}, {}, {optional_output_tensor}}};
}

SparseMatmulParams create_sparse_matmul_attributes(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& /*sparsity*/,
    const SparseMatmulParams& parameters,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
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

    auto matmul_struct =
        create_matmul_attributes(input_tensor_a, input_tensor_b, matmul_attributes, {optional_output_tensors.at(0)});
    return SparseMatmulParams{
        parameters.nnz,
        parameters.is_input_a_sparse,
        parameters.is_input_b_sparse,
        matmul_struct.program_config,
        matmul_struct.output_mem_config,
        matmul_struct.output_dtype,
        matmul_struct.compute_kernel_config,
        matmul_struct.user_core_coord,
        matmul_struct.output_tile,
        matmul_struct.global_cb,
        matmul_struct.sub_device_id};
}
}  // namespace ttnn::prim
