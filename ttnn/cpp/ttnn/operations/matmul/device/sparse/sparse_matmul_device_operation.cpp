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

#include <variant>

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

ttnn::Shape compute_sparse_matmul_compact_output_shape(
    const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b, uint32_t nnz) {
    return ttnn::Shape{1U, nnz, input_tensor_a.logical_shape()[-2], input_tensor_b.logical_shape()[-1]};
}
}  // namespace

namespace ttnn::prim {

void SparseMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

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

    const bool is_output_tensor_given =
        !tensor_args.optional_output_tensors.empty() && tensor_args.optional_output_tensors.at(0).has_value();
    if (is_output_tensor_given) {
        // The program factory derives the writer's page geometry from the input tiles
        // ({in0 tile height, in1 tile width}), not from the output tensor, so an optional
        // output with any other tile would be paged differently than the writer writes it.
        const auto& optional_output_tile = tensor_args.optional_output_tensors.at(0)->tensor_spec().tile();
        TT_FATAL(
            optional_output_tile.get_height() == in0_tile.get_height() &&
                optional_output_tile.get_width() == in1_tile.get_width(),
            "Optional output tensor tile {}x{} must match the output tile {}x{} derived from the input tiles",
            optional_output_tile.get_height(),
            optional_output_tile.get_width(),
            in0_tile.get_height(),
            in1_tile.get_width());

        const auto& optional_output_shape = tensor_args.optional_output_tensors.at(0)->logical_shape();
        const auto expanded_output_shape = compute_sparse_matmul_output_shape(
            input_tensor_a,
            input_tensor_b,
            operation_attributes.is_input_a_sparse,
            operation_attributes.is_input_b_sparse);

        if (operation_attributes.nnz.has_value()) {
            const auto compact_output_shape = compute_sparse_matmul_compact_output_shape(
                input_tensor_a, input_tensor_b, operation_attributes.nnz.value());
            TT_FATAL(
                optional_output_shape == expanded_output_shape || optional_output_shape == compact_output_shape,
                "Optional output tensor shape {} must match expanded output shape {} or compact output shape {} "
                "when nnz={}",
                optional_output_shape,
                expanded_output_shape,
                compact_output_shape,
                operation_attributes.nnz.value());
        } else {
            TT_FATAL(
                optional_output_shape == expanded_output_shape,
                "Optional output tensor shape {} must match expanded output shape {} when nnz is not provided",
                optional_output_shape,
                expanded_output_shape);
        }
    }

    // The mcast_in0 kernel derives in1_num_subblocks = out_block_w / out_subblock_w and bakes it in
    // as a compute-kernel compile-time arg. When out_subblock_w does not divide out_block_w the
    // integer division yields 0, the compute kernel's in1_subblock loop runs zero times and never
    // pushes cb_out, and the in1 writer parks forever on cb_out.wait_front() -- a silent device hang
    // requiring a board reset. The same expression also underflows uint32 in the last-block padded
    // skip count. Reject on the host instead, matching the dense matmul path's
    // validate_matmul_block_and_subblock_configuration.
    //
    // Check order matters: every value used as a divisor below is rejected as zero first, so a
    // malformed program config fails as a TT_FATAL rather than a host divide-by-zero.
    if (operation_attributes.program_config.has_value()) {
        if (const auto* pc = std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
                &operation_attributes.program_config.value())) {
            TT_FATAL(
                pc->out_subblock_w != 0 && pc->out_subblock_h != 0,
                "sparse_matmul: out_subblock_w and out_subblock_h must be non-zero");
            TT_FATAL(
                pc->out_block_w != 0 && pc->out_block_h != 0,
                "sparse_matmul: out_block_w and out_block_h must be non-zero");
            TT_FATAL(
                pc->out_block_w % pc->out_subblock_w == 0,
                "sparse_matmul: out_block_w ({}) must be divisible by out_subblock_w ({}); otherwise "
                "in1_num_subblocks becomes 0 and the mcast_in0 kernel deadlocks",
                pc->out_block_w,
                pc->out_subblock_w);
            TT_FATAL(
                pc->out_block_h % pc->out_subblock_h == 0,
                "sparse_matmul: out_block_h ({}) must be divisible by out_subblock_h ({})",
                pc->out_block_h,
                pc->out_subblock_h);
            TT_FATAL(
                pc->per_core_M % pc->out_block_h == 0,
                "sparse_matmul: per_core_M ({}) must be divisible by out_block_h ({})",
                pc->per_core_M,
                pc->out_block_h);
            TT_FATAL(
                pc->per_core_N % pc->out_block_w == 0,
                "sparse_matmul: per_core_N ({}) must be divisible by out_block_w ({})",
                pc->per_core_N,
                pc->out_block_w);
        }
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

    return {tt::tt_metal::TensorSpec(
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
    // A compact output is fully written by the in1 writer (one block per non-zero sparsity entry),
    // so it skips the zero-fill below; expanded outputs rely on it to zero the skipped blocks.
    const bool compact_output = !optional_output_tensors.empty() && optional_output_tensors[0].has_value() &&
                                operation_attributes.nnz.has_value() &&
                                optional_output_tensors[0]->logical_shape() ==
                                    compute_sparse_matmul_compact_output_shape(
                                        input_tensors.at(0), input_tensors.at(1), operation_attributes.nnz.value());

    if (!optional_output_tensors.empty() and optional_output_tensors[0].has_value()) {
        output_tensors.reserve(optional_output_tensors.size());
        for (const auto& optional_output_tensor : optional_output_tensors) {
            TT_FATAL(
                optional_output_tensor.has_value(),
                "If using optional output tensors, all output tensors must have a value");
            output_tensors.emplace_back(optional_output_tensor.value());
        }
        if (!compact_output) {
            for (auto& output_tensor : output_tensors) {
                output_tensor = ttnn::zeros_like(
                    output_tensor,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::optional<Tensor>(output_tensor));
            }
        }
        return output_tensors;
    }
    const auto& device = input_tensors.at(0).device();
    const auto& output_specs = compute_output_specs(operation_attributes, tensor_args);
    output_tensors.reserve(output_specs.size());
    for (const auto& output_spec : output_specs) {
        output_tensors.emplace_back(create_device_tensor(output_spec, device));
    }
    // Compact output requires a caller-supplied tensor, so this path is never compact.
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
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
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
        sub_device_id);
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
