// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"

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
    bool is_input_b_sparse,
    std::optional<uint32_t> num_active = std::nullopt) {
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

    // Indexed/gather mode: the expert/batch axis is COMPACT (only the num_active selected experts).
    // For every supported mode here input B is sparse with layout [..., E, K, N], so the E batch
    // length lives at output_shape[-3]; overwrite it with num_active. (M/N are unchanged.)
    if (num_active.has_value()) {
        output_shape[-3] = num_active.value();
    }

    return output_shape;
}
}  // namespace

namespace ttnn::prim {

void SparseMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace operations::matmul::utilities;
    const auto& input_tensor_a = tensor_args.input_tensors.at(0);
    const auto& input_tensor_b = tensor_args.input_tensors.at(1);
    const auto& sparsity = tensor_args.input_tensors.at(2);

    TT_FATAL(
        input_tensor_a.storage_type() == ttnn::StorageType::DEVICE &&
            input_tensor_b.storage_type() == ttnn::StorageType::DEVICE &&
            sparsity.storage_type() == ttnn::StorageType::DEVICE,
        "All sparse matmul inputs must be on device");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr && input_tensor_b.buffer() != nullptr && sparsity.buffer() != nullptr,
        "All sparse matmul inputs must be allocated in buffers");
    TT_FATAL(
        input_tensor_a.device() == input_tensor_b.device() && input_tensor_a.device() == sparsity.device(),
        "All sparse matmul inputs must be on the same device");
    TT_FATAL(
        input_tensor_a.layout() == ttnn::Layout::TILE,
        "Input tensor A must be TILE layout, got {}",
        input_tensor_a.layout());
    TT_FATAL(
        input_tensor_b.layout() == ttnn::Layout::TILE,
        "Input tensor B must be TILE layout, got {}",
        input_tensor_b.layout());
    TT_FATAL(
        is_floating_point(input_tensor_a.dtype()),
        "Input tensor A must be a floating point type, got {}",
        input_tensor_a.dtype());
    TT_FATAL(
        is_floating_point(input_tensor_b.dtype()),
        "Input tensor B must be a floating point type, got {}",
        input_tensor_b.dtype());
    TT_FATAL(
        sparsity.layout() == ttnn::Layout::ROW_MAJOR,
        "Sparsity tensor must be ROW_MAJOR layout, got {}",
        sparsity.layout());
    TT_FATAL(
        operation_attributes.is_input_a_sparse || operation_attributes.is_input_b_sparse,
        "sparse_matmul requires at least one of is_input_a_sparse or is_input_b_sparse to be true");

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
        operation_attributes.nnz.value_or(1) > 0,
        "nnz ({}) must be greater than 0",
        operation_attributes.nnz.value_or(1));

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
        "sparsity logical_volume ({}) must equal batch_length ({}) "
        "[sparsity_shape={}, is_input_a_sparse={}, is_input_b_sparse={}]",
        sparsity.logical_volume(),
        batch_length,
        sparsity.logical_shape(),
        operation_attributes.is_input_a_sparse,
        operation_attributes.is_input_b_sparse);

    TT_FATAL(
        operation_attributes.nnz.value_or(1) <= batch_length,
        "nnz ({}) must be less than or equal to the length of all batch dimensions ({})",
        operation_attributes.nnz.value_or(1),
        batch_length);

    // When nnz is supplied, the receiver and compute kernels loop exactly nnz times while the in0 sender
    // only multicasts once per non-zero sparsity entry. The op therefore requires
    // count_nonzero(sparsity) == nnz; a mismatch deadlocks the device (see issue #45943).
    // count_nonzero(sparsity) is data-dependent and lives on device, so it cannot be checked here on the
    // host -- it is the caller's responsibility to pass an exact nnz, and the contract is validated
    // on-device in reader_bmm_tile_layout_in0_sender_padding.cpp (asserts loudly under watcher instead of
    // hanging).
    // Indexed/gather mode validation. `indices` (optional_input_tensors[0]) is a compacted list of
    // active expert ids; the kernels iterate it directly (bB = indices[i]) instead of scanning all
    // batch slots, and the output expert axis becomes num_active = indices.logical_volume().
    if (operation_attributes.use_indices) {
        TT_FATAL(
            !tensor_args.optional_input_tensors.empty() && tensor_args.optional_input_tensors.at(0).has_value(),
            "use_indices is set but no indices tensor was provided");
        const auto& indices = tensor_args.optional_input_tensors.at(0).value();
        TT_FATAL(
            operation_attributes.is_input_b_sparse,
            "Indexed/gather mode requires is_input_b_sparse=true (the indexed operand is the expert "
            "weight tensor B, laid out as [..., E, K, N]).");
        TT_FATAL(
            indices.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "indices must be ROW_MAJOR layout, got {}",
            indices.layout());
        TT_FATAL(
            indices.dtype() == tt::tt_metal::DataType::UINT16, "indices must be UINT16 dtype, got {}", indices.dtype());
        TT_FATAL(indices.is_allocated(), "indices tensor must be allocated on device");
        TT_FATAL(
            indices.logical_volume() <= batch_length,
            "indices length / num_active ({}) must be <= the length of all batch dimensions ({})",
            indices.logical_volume(),
            batch_length);
    }
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

    // Indexed/gather mode -> compact output: the expert axis shrinks from E to num_active (the
    // length of the indices operand carried in optional_input_tensors[0]).
    std::optional<uint32_t> num_active = std::nullopt;
    if (operation_attributes.use_indices && !tensor_args.optional_input_tensors.empty() &&
        tensor_args.optional_input_tensors.at(0).has_value()) {
        num_active = tensor_args.optional_input_tensors.at(0)->logical_volume();
    }

    const auto output_shape = compute_sparse_matmul_output_shape(
        input_tensor_a,
        input_tensor_b,
        operation_attributes.is_input_a_sparse,
        operation_attributes.is_input_b_sparse,
        num_active);

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
    const auto& device = input_tensors.at(0).device();
    const auto& output_specs = compute_output_specs(operation_attributes, tensor_args);
    output_tensors.reserve(output_specs.size());
    for (const auto& output_spec : output_specs) {
        output_tensors.emplace_back(create_device_tensor(output_spec, device));
    }
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

// static ttsl::hash::hash_t SparseMatmulDeviceOperation::compute_program_hash(
//     const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

std::tuple<SparseMatmulParams, SparseMatmulInputs> sparse_matmul_build_operation_args(
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
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const std::optional<Tensor>& indices) {
    auto sparse_matmul_attributes = SparseMatmulParams{
        nnz,
        is_input_a_sparse,
        is_input_b_sparse,
        indices.has_value(),  // use_indices
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

    // The indices operand (if any) rides in optional_input_tensors[0]; presence there is the sole
    // trigger for indexed/gather mode. When absent, optional_input_tensors stays empty and every
    // downstream path is byte-for-byte identical to the dense sparsity-scan behavior.
    std::vector<std::optional<const Tensor>> optional_inputs;
    if (indices.has_value()) {
        optional_inputs.emplace_back(indices);
    }

    return {
        parameters,
        SparseMatmulInputs{{input_tensor_a, input_tensor_b, sparsity}, optional_inputs, {optional_output_tensor}}};
}

SparseMatmulDeviceOperation::tensor_return_value_t sparse_matmul(
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
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const std::optional<Tensor>& indices) {
    auto [params, inputs] = sparse_matmul_build_operation_args(
        input_tensor_a,
        input_tensor_b,
        sparsity,
        optional_output_tensor,
        nnz,
        is_input_a_sparse,
        is_input_b_sparse,
        memory_config,
        std::move(dtype),
        program_config,
        std::move(compute_kernel_config),
        user_core_coord,
        output_tile,
        global_cb,
        sub_device_id,
        indices);
    return ttnn::device_operation::launch<SparseMatmulDeviceOperation>(params, inputs);
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
    if (matmul_struct.program_config.has_value()) {
        auto device_grid = input_tensor_a.device()->compute_with_storage_grid_size();
        operations::matmul::normalize_program_config(matmul_struct.program_config.value(), device_grid);
    }
    return SparseMatmulParams{
        parameters.nnz,
        parameters.is_input_a_sparse,
        parameters.is_input_b_sparse,
        parameters.use_indices,
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
