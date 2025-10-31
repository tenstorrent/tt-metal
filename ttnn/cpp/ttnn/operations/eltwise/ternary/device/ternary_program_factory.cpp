// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_device_operation.hpp"
#include "ternary_op_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include <cmath>

namespace {
// Helper function to add sharding defines for dataflow kernels
void add_sharding_defines(
    std::map<std::string, std::string>& defines,
    bool predicate_sharded,
    bool value_true_sharded,
    bool value_false_sharded) {
    defines["SRC_SHARDED_A"] = predicate_sharded ? "1" : "0";    // CB0 sharding
    defines["SRC_SHARDED_B"] = value_true_sharded ? "1" : "0";   // CB1 sharding
    defines["SRC_SHARDED_C"] = value_false_sharded ? "1" : "0";  // CB2 sharding
}

// Helper function to add broadcast defines for dataflow kernels
void add_broadcast_defines(
    std::map<std::string, std::string>& defines, bool pred_is_bcast, bool true_is_bcast, bool false_is_bcast) {
    defines["SRC_BCAST_A"] = pred_is_bcast ? "1" : "0";   // Predicate tensor (CB0)
    defines["SRC_BCAST_B"] = true_is_bcast ? "1" : "0";   // True tensor (CB1)
    defines["SRC_BCAST_C"] = false_is_bcast ? "1" : "0";  // False tensor (CB2)
}

// Helper function to get sharding status for tensors
std::tuple<bool, bool, bool> get_tensor_sharding_status(
    const ttnn::Tensor& predicate_tensor,
    const std::optional<ttnn::Tensor>& value_true_tensor,
    const std::optional<ttnn::Tensor>& value_false_tensor) {
    bool predicate_sharded = predicate_tensor.memory_config().is_sharded();
    bool value_true_sharded = value_true_tensor.has_value() && value_true_tensor->memory_config().is_sharded();
    bool value_false_sharded = value_false_tensor.has_value() && value_false_tensor->memory_config().is_sharded();
    return {predicate_sharded, value_true_sharded, value_false_sharded};
}

// Helper function to detect broadcast patterns for different variants and broadcast types
void detect_broadcasts(
    bool& pred_is_bcast,
    bool& true_is_bcast,
    bool& false_is_bcast,
    ttnn::operations::ternary::TernaryVariant variant,
    ttnn::operations::ternary::TernaryBroadcastType broadcast_type,
    const ttnn::Tensor& predicate_tensor,
    const std::optional<ttnn::Tensor>& value_true_tensor,
    const std::optional<ttnn::Tensor>& value_false_tensor) {
    auto pred_shape = predicate_tensor.logical_shape();

    using TernaryBroadcastType = ttnn::operations::ternary::TernaryBroadcastType;
    using TernaryVariant = ttnn::operations::ternary::TernaryVariant;

    if (broadcast_type == TernaryBroadcastType::OUTER_BCAST || broadcast_type == TernaryBroadcastType::NONE) {
        pred_is_bcast = true_is_bcast = false_is_bcast = false;
    } else if (broadcast_type == TernaryBroadcastType::COL_BCAST) {
        // Column broadcast detection
        auto pred_w = pred_shape[pred_shape.rank() - 1];

        if (variant == TernaryVariant::TTS) {
            auto true_shape = value_true_tensor.value().logical_shape();
            auto true_w = true_shape[true_shape.rank() - 1];
            pred_is_bcast = (pred_w == 1 && true_w > 1);
            true_is_bcast = (true_w == 1 && pred_w > 1);
            false_is_bcast = false;  // False is scalar for TTS
        } else if (variant == TernaryVariant::TST) {
            auto false_shape = value_false_tensor.value().logical_shape();
            auto false_w = false_shape[false_shape.rank() - 1];
            pred_is_bcast = (pred_w == 1 && false_w > 1);
            false_is_bcast = (false_w == 1 && pred_w > 1);
            true_is_bcast = false;  // True is scalar for TST
        } else {                    // TTT
            auto true_shape = value_true_tensor.value().logical_shape();
            auto false_shape = value_false_tensor.value().logical_shape();
            auto true_w = true_shape[true_shape.rank() - 1];
            auto false_w = false_shape[false_shape.rank() - 1];
            pred_is_bcast = (pred_w == 1 && (true_w > 1 || false_w > 1));
            true_is_bcast = (true_w == 1 && (pred_w > 1 || false_w > 1));
            false_is_bcast = (false_w == 1 && (pred_w > 1 || true_w > 1));
        }
    } else if (broadcast_type == TernaryBroadcastType::ROW_BCAST) {
        // Row broadcast detection
        auto pred_h = pred_shape[pred_shape.rank() - 2];

        if (variant == TernaryVariant::TTS) {
            auto true_shape = value_true_tensor.value().logical_shape();
            auto true_h = true_shape[true_shape.rank() - 2];
            pred_is_bcast = (pred_h == 1 && true_h > 1);
            true_is_bcast = (true_h == 1 && pred_h > 1);
            false_is_bcast = false;  // False is scalar for TTS
        } else if (variant == TernaryVariant::TST) {
            auto false_shape = value_false_tensor.value().logical_shape();
            auto false_h = false_shape[false_shape.rank() - 2];
            pred_is_bcast = (pred_h == 1 && false_h > 1);
            true_is_bcast = false;  // True is scalar for TST
            false_is_bcast = (false_h == 1 && pred_h > 1);
        } else {  // TTT
            auto true_shape = value_true_tensor.value().logical_shape();
            auto false_shape = value_false_tensor.value().logical_shape();
            auto true_h = true_shape[true_shape.rank() - 2];
            auto false_h = false_shape[false_shape.rank() - 2];
            pred_is_bcast = (pred_h == 1 && (true_h > 1 || false_h > 1));
            true_is_bcast = (true_h == 1 && (pred_h > 1 || false_h > 1));
            false_is_bcast = (false_h == 1 && (pred_h > 1 || true_h > 1));
        }
    } else if (broadcast_type == TernaryBroadcastType::SCALAR_BCAST) {
        // Scalar broadcast detection (H and W dimensions must be broadcast for TTT)
        auto true_shape = value_true_tensor.value().logical_shape();
        auto false_shape = value_false_tensor.value().logical_shape();

        auto pred_w = pred_shape[-1], pred_h = pred_shape[-2];
        auto true_w = true_shape[-1], true_h = true_shape[-2];
        auto false_w = false_shape[-1], false_h = false_shape[-2];

        auto max_h = std::max({pred_h, true_h, false_h});
        auto max_w = std::max({pred_w, true_w, false_w});

        bool pred_row_bcast = (pred_h == 1 && max_h > 1);
        bool true_row_bcast = (true_h == 1 && max_h > 1);
        bool false_row_bcast = (false_h == 1 && max_h > 1);

        bool pred_col_bcast = (pred_w == 1 && max_w > 1);
        bool true_col_bcast = (true_w == 1 && max_w > 1);
        bool false_col_bcast = (false_w == 1 && max_w > 1);

        pred_is_bcast = (pred_row_bcast && pred_col_bcast);
        true_is_bcast = (true_row_bcast && true_col_bcast);
        false_is_bcast = (false_row_bcast && false_col_bcast);
    } else if (broadcast_type == TernaryBroadcastType::SCALAR_A_BCAST) {
        // Scalar broadcast detection (H and W dimensions of condition tensor must be broadcast for TTS/TST)
        pred_is_bcast = true;
        true_is_bcast = false;
        false_is_bcast = false;
    } else if (broadcast_type == TernaryBroadcastType::SCALAR_B_BCAST) {
        // Scalar broadcast detection (H and W dimensions of true/false tensor must be broadcast for TTS/TST)
        pred_is_bcast = false;
        true_is_bcast = true;
        false_is_bcast = true;
    }
}

// Helper function to set up reader defines based on variant and broadcast type
void setup_reader_defines(
    std::map<std::string, std::string>& reader_defines,
    ttnn::operations::ternary::TernaryVariant variant,
    ttnn::operations::ternary::TernaryBroadcastType broadcast_type,
    const ttnn::Tensor& predicate_tensor,
    const std::optional<ttnn::Tensor>& value_true_tensor,
    const std::optional<ttnn::Tensor>& value_false_tensor,
    bool pred_is_bcast,
    bool true_is_bcast,
    bool false_is_bcast) {
    using TernaryVariant = ttnn::operations::ternary::TernaryVariant;
    if (variant == TernaryVariant::TTT) {
        reader_defines = ttnn::operations::ternary::make_dataflow_defines(
            predicate_tensor.dtype(), value_true_tensor.value().dtype(), value_false_tensor.value().dtype());
        auto [predicate_sharded, value_true_sharded, value_false_sharded] =
            get_tensor_sharding_status(predicate_tensor, value_true_tensor, value_false_tensor);
        add_sharding_defines(reader_defines, predicate_sharded, value_true_sharded, value_false_sharded);
        add_broadcast_defines(reader_defines, pred_is_bcast, true_is_bcast, false_is_bcast);
    } else if (variant == TernaryVariant::TTS) {
        reader_defines = ttnn::operations::ternary::make_dataflow_defines(
            predicate_tensor.dtype(), value_true_tensor.value().dtype());
        auto [predicate_sharded, value_true_sharded, value_false_sharded] =
            get_tensor_sharding_status(predicate_tensor, value_true_tensor, std::nullopt);
        add_sharding_defines(reader_defines, predicate_sharded, value_true_sharded, value_false_sharded);
        add_broadcast_defines(reader_defines, pred_is_bcast, true_is_bcast, false_is_bcast);
    } else if (variant == TernaryVariant::TST) {
        reader_defines = ttnn::operations::ternary::make_dataflow_defines(
            predicate_tensor.dtype(), value_false_tensor.value().dtype());
        auto [predicate_sharded, value_true_sharded, value_false_sharded] =
            get_tensor_sharding_status(predicate_tensor, std::nullopt, value_false_tensor);
        add_sharding_defines(reader_defines, predicate_sharded, value_true_sharded, value_false_sharded);

        // For TST the two tensors are predicate and false_tensors and are SRC_A and SRC_B respectively
        add_broadcast_defines(reader_defines, pred_is_bcast, false_is_bcast, true_is_bcast);
    }
}

namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::ternary;

// Get operation-specific compute kernel defines
std::map<std::string, std::string> get_ternary_compute_defines(
    const TernaryDeviceOperation::operation_attributes_t& operation_attributes, ttnn::DataType dtype) {
    return get_compute_defines(operation_attributes.ternary_op_type, dtype);
}

// For rank > 5 dims will be collapsed into a single dim
uint32_t extract_nD_dims(const ttnn::Tensor& x, const int out_rank) {
    const auto& shape = x.logical_shape();
    uint32_t nD_dim = 1;
    if (out_rank >= 6 && shape.rank() >= 6) {
        for (int i = -6; i >= -out_rank; --i) {
            auto dim = shape[i];
            nD_dim *= dim;
        }
    }
    return nD_dim;
}

// Helper struct to hold tensor dimensions
struct TensorDimensions {
    uint32_t D = 1, N = 1, C = 1, Ht = 1, Wt = 1, ND = 1;
    uint32_t num_tiles = 0;
};

// Helper function to extract tensor dimensions
TensorDimensions extract_tensor_dimensions(const ttnn::Tensor& tensor, int out_rank, uint32_t tile_h, uint32_t tile_w) {
    TensorDimensions dims;
    const auto& shape = tensor.padded_shape();
    dims.ND = extract_nD_dims(tensor, out_rank);
    dims.D = shape.rank() >= 5 ? shape[-5] : 1;
    dims.N = shape[-4];
    dims.C = shape[-3];
    dims.Ht = shape[-2] / tile_h;
    dims.Wt = shape[-1] / tile_w;

    return dims;
}

// Helper function to calculate strides for broadcast operations
struct Strides {
    uint32_t nD_stride = 0, d_stride = 0, n_stride = 0, c_stride = 0;
};

Strides calculate_strides(const TensorDimensions& dims) {
    Strides strides;
    strides.nD_stride = dims.Ht * dims.Wt * dims.C * dims.N * dims.D * (dims.ND > 1);
    strides.d_stride = dims.Ht * dims.Wt * dims.C * dims.N * (dims.D > 1);
    strides.n_stride = dims.Ht * dims.Wt * dims.C * (dims.N > 1);
    strides.c_stride = dims.Ht * dims.Wt * (dims.C > 1);
    return strides;
}

// Helper function to set up reader runtime arguments and dimensions for TTS/TST variants
template <size_t num_reader_args>
void setup_ts_reader_args_and_dims(
    std::array<uint32_t, num_reader_args>& reader_runtime_args,
    const ttnn::Tensor& predicate_tensor,
    const ttnn::Tensor& tensor_operand,  // true tensor for TTS, false tensor for TST
    const ttnn::Tensor& output,
    uint32_t num_tiles_per_core,
    uint32_t start_tile_id,
    uint32_t& a_num_tiles,
    uint32_t& tensor_num_tiles,  // number of tiles for the tensor operand
    uint32_t& c_current_shard_width,
    ttnn::operations::ternary::TernaryBroadcastType broadcast_type,
    bool& has_sharding,
    int out_rank,
    uint32_t tile_h,
    uint32_t tile_w) {
    has_sharding = predicate_tensor.memory_config().is_sharded() || tensor_operand.memory_config().is_sharded() ||
                   output.memory_config().is_sharded();
    auto output_dims = extract_tensor_dimensions(output, out_rank, tile_h, tile_w);
    auto pred_dims = extract_tensor_dimensions(predicate_tensor, out_rank, tile_h, tile_w);
    auto tensor_dims = extract_tensor_dimensions(tensor_operand, out_rank, tile_h, tile_w);

    auto pred_strides = calculate_strides(pred_dims);
    auto tensor_strides = calculate_strides(tensor_dims);

    // Only set tile counts if sharding is enabled
    if (has_sharding) {
        a_num_tiles = pred_dims.Ht * pred_dims.Wt;           // predicate tiles per core
        tensor_num_tiles = tensor_dims.Ht * tensor_dims.Wt;  // tensor operand tiles per core
        c_current_shard_width = output_dims.Wt;
    }
    // Standard first 5 arguments
    reader_runtime_args[0] = predicate_tensor.buffer()->address();  // 0: src0_addr (predicate)
    reader_runtime_args[1] = tensor_operand.buffer()->address();    // 1: src1_addr (tensor operand)
    reader_runtime_args[2] = 0u;                                    // 2: src2_addr (scalar operand)
    reader_runtime_args[3] = num_tiles_per_core;                    // 3: num_tiles (per core)
    reader_runtime_args[4] = start_tile_id;                         // 4: start_id
    // Extended broadcast arguments
    if (broadcast_type != ttnn::operations::ternary::TernaryBroadcastType::NONE) {
        reader_runtime_args[5] = pred_strides.nD_stride;  // 5: nD_stride
        reader_runtime_args[6] = pred_strides.d_stride;   // 6: d_stride
        reader_runtime_args[7] = pred_strides.n_stride;   // 7: n_stride
        reader_runtime_args[8] = pred_strides.c_stride;   // 8: c_stride
        reader_runtime_args[9] = output_dims.D;           // 9: D
        reader_runtime_args[10] = output_dims.N;          // 10: N
        reader_runtime_args[11] = output_dims.C;          // 11: C
        reader_runtime_args[12] = output_dims.Ht;         // 12: Ht
        reader_runtime_args[13] = output_dims.Wt;         // 13: Wt
        reader_runtime_args[14] = output_dims.ND;         // 14: cND
        // Both TTS and TST: tensor operand strides in 15-19, scalar operand zeros in 20-24
        reader_runtime_args[15] = tensor_strides.nD_stride;  // 15: tensor_operand_nD_stride
        reader_runtime_args[16] = tensor_strides.d_stride;   // 16: tensor_operand_d_stride
        reader_runtime_args[17] = tensor_strides.n_stride;   // 17: tensor_operand_n_stride
        reader_runtime_args[18] = tensor_strides.c_stride;   // 18: tensor_operand_c_stride
        reader_runtime_args[19] = tensor_num_tiles;          // 19: tensor_operand_num_tiles
        // Scalar operand has no strides
        reader_runtime_args[20] = 0u;  // 20: scalar_operand_nD_stride
        reader_runtime_args[21] = 0u;  // 21: scalar_operand_d_stride
        reader_runtime_args[22] = 0u;  // 22: scalar_operand_n_stride
        reader_runtime_args[23] = 0u;  // 23: scalar_operand_c_stride
        reader_runtime_args[24] = 0u;  // 24: scalar_operand_num_tiles

        reader_runtime_args[25] = c_current_shard_width;  // 25: dst_shard_width
        reader_runtime_args[26] = a_num_tiles;            // 26: src_num_tiles (predicate)
    }
}

template <typename F>
void set_or_update_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    tt::tt_metal::KernelHandle compute_kernel_id,
    CoreCoord compute_with_storage_grid_size,
    const TernaryDeviceOperation::operation_attributes_t& operation_attributes,
    const TernaryDeviceOperation::tensor_args_t& tensor_args,
    TernaryDeviceOperation::tensor_return_value_t& output,
    ttnn::operations::ternary::TernaryBroadcastType broadcast_type,
    F handle_args) {
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;
    TernaryVariant variant = operation_attributes.ternary_variant;
    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    constexpr size_t num_reader_args = 27;
    constexpr size_t num_writer_args = 3;
    constexpr size_t num_kernel_args = 4;

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            handle_args(program, reader_kernel_id, core, std::array<uint32_t, num_reader_args>{0});
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, num_writer_args>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, num_kernel_args>{0});
            continue;
        }

        // Declare variables common to all variants
        uint32_t a_num_tiles = 0, b_num_tiles = 0, f_num_tiles = 0,
                 c_current_shard_width = 0;  // Initialize to 0 like binary_ng
        const auto out_rank = output.logical_shape().rank();
        const auto& tile = output.tensor_spec().tile();
        const uint32_t tile_h = tile.get_height();
        const uint32_t tile_w = tile.get_width();
        auto output_dims = extract_tensor_dimensions(output, out_rank, tile_h, tile_w);
        bool has_sharding = false;
        // calculate has_sharding when support is added
        // has_sharding = predicate_tensor.memory_config().is_sharded() ||
        //                 value_true_tensor.value().memory_config().is_sharded() ||
        //                 value_false_tensor.value().memory_config().is_sharded() ||
        //                 output.memory_config().is_sharded();

        // Set reader runtime arguments based on variant
        if (variant == ttnn::operations::ternary::TernaryVariant::TTS ||
            variant == ttnn::operations::ternary::TernaryVariant::TST) {
            // Get the appropriate tensor operand
            const auto& tensor_operand = (variant == ttnn::operations::ternary::TernaryVariant::TTS)
                                             ? value_true_tensor.value()
                                             : value_false_tensor.value();
            uint32_t& tensor_num_tiles_ref =
                (variant == ttnn::operations::ternary::TernaryVariant::TTS) ? b_num_tiles : f_num_tiles;

            std::array<uint32_t, num_reader_args> reader_runtime_args{};
            setup_ts_reader_args_and_dims<num_reader_args>(
                reader_runtime_args,
                predicate_tensor,
                tensor_operand,
                output,
                num_tiles_per_core,
                start_tile_id,
                a_num_tiles,
                tensor_num_tiles_ref,
                c_current_shard_width,
                broadcast_type,
                has_sharding,
                out_rank,
                tile_h,
                tile_w);
            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        } else if (variant == TernaryVariant::TTT) {
            auto pred_dims = extract_tensor_dimensions(predicate_tensor, out_rank, tile_h, tile_w);
            auto true_dims = extract_tensor_dimensions(value_true_tensor.value(), out_rank, tile_h, tile_w);
            auto false_dims = extract_tensor_dimensions(value_false_tensor.value(), out_rank, tile_h, tile_w);
            auto pred_strides = calculate_strides(pred_dims);
            auto true_strides = calculate_strides(true_dims);
            auto false_strides = calculate_strides(false_dims);
            // Match binary_ng logic: only set tile counts if sharding is enabled
            // For non-sharded (interleaved) mode, these remain 0 like binary_ng
            if (has_sharding) {
                a_num_tiles = pred_dims.Ht * pred_dims.Wt;    // predicate tiles per core
                b_num_tiles = true_dims.Ht * true_dims.Wt;    // value_true tiles per core
                f_num_tiles = false_dims.Ht * false_dims.Wt;  // value_false tiles per core
                c_current_shard_width = output_dims.Wt;
                /* If not sharded, a_num_tiles, b_num_tiles, f_num_tiles, c_current_shard_width remain 0 (like
                   binary_ng) Match binary_ng sharding logic for c_start_id calculation NOTE: This requires shard shape
                   info thats need to be implemented */
            }
            std::array<uint32_t, num_reader_args> reader_runtime_args{};  // zero-initialized
            // Standard first 5 arguments
            reader_runtime_args[0] = predicate_tensor.buffer()->address();            // 0: src0_addr (predicate)
            reader_runtime_args[1] = value_true_tensor.value().buffer()->address();   // 1: src1_addr (true tensor)
            reader_runtime_args[2] = value_false_tensor.value().buffer()->address();  // 2: src2_addr (false tensor)
            reader_runtime_args[3] = num_tiles_per_core;                              // 3: num_tiles (per core)
            reader_runtime_args[4] = start_tile_id;                                   // 4: start_id
            // Extended broadcast arguments (only when broadcast != NONE)
            if (broadcast_type != ttnn::operations::ternary::TernaryBroadcastType::NONE) {
                reader_runtime_args[5] = pred_strides.nD_stride;    // 5: nD_stride
                reader_runtime_args[6] = pred_strides.d_stride;     // 6: d_stride
                reader_runtime_args[7] = pred_strides.n_stride;     // 7: n_stride
                reader_runtime_args[8] = pred_strides.c_stride;     // 8: c_stride
                reader_runtime_args[9] = output_dims.D;             // 9: D
                reader_runtime_args[10] = output_dims.N;            // 10: N
                reader_runtime_args[11] = output_dims.C;            // 11: C
                reader_runtime_args[12] = output_dims.Ht;           // 12: Ht
                reader_runtime_args[13] = output_dims.Wt;           // 13: Wt
                reader_runtime_args[14] = output_dims.ND;           // 14: cND
                reader_runtime_args[15] = true_strides.nD_stride;   // 15: true_nD_stride
                reader_runtime_args[16] = true_strides.d_stride;    // 16: true_d_stride
                reader_runtime_args[17] = true_strides.n_stride;    // 17: true_n_stride
                reader_runtime_args[18] = true_strides.c_stride;    // 18: true_c_stride
                reader_runtime_args[19] = b_num_tiles;              // 19: true_num_tiles
                reader_runtime_args[20] = false_strides.nD_stride;  // 20: false_nD_stride
                reader_runtime_args[21] = false_strides.d_stride;   // 21: false_d_stride
                reader_runtime_args[22] = false_strides.n_stride;   // 22: false_n_stride
                reader_runtime_args[23] = false_strides.c_stride;   // 23: false_c_stride
                reader_runtime_args[24] = f_num_tiles;              // 24: false_num_tiles
                reader_runtime_args[25] = c_current_shard_width;    // 25: dst_shard_width
                reader_runtime_args[26] = a_num_tiles;              // 26: src_num_tiles (predicate)
            }

            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        } else {
            TT_FATAL(false, "Unsupported Where variant in TernaryDeviceOperation. Supported: TTS, TST, TTT");
        }

        // Writer runtime args - use simple unary format (3 args: dst_addr, num_tiles, start_id)
        std::array writer_runtime_args = {
            output.buffer()->address(),  // dst_addr
            num_tiles_per_core,          // num_tiles
            start_tile_id                // start_id
        };
        handle_args(program, writer_kernel_id, core, writer_runtime_args);

        // Compute runtime args
        uint32_t scalar_arg = 0u;
        if (variant == TernaryVariant::TTS) {
            scalar_arg = pack_scalar_runtime_arg(operation_attributes.scalar_input_b.value(), output.dtype());
        } else if (variant == TernaryVariant::TST) {
            scalar_arg = pack_scalar_runtime_arg(operation_attributes.scalar_input_a.value(), output.dtype());
        }
        auto [freq, counter] = [&] {
            switch (broadcast_type) {
                // TODO: test for TTS and TST
                // case TernaryBroadcastType::ROW_B_COL_A:
                // case TernaryBroadcastType::ROW_A_COL_B:
                case TernaryBroadcastType::COL_BCAST: {
                    uint32_t start_t = start_tile_id % (output_dims.Ht * output_dims.Wt);
                    uint32_t start_tw = start_t % output_dims.Wt;
                    return std::pair{output_dims.Wt, start_tw};
                }
                case TernaryBroadcastType::SCALAR_BCAST:
                case TernaryBroadcastType::SCALAR_A_BCAST:
                case TernaryBroadcastType::SCALAR_B_BCAST: {
                    uint32_t HtWt = output_dims.Ht * output_dims.Wt;
                    uint32_t start_t = start_tile_id % HtWt;
                    return std::pair{HtWt, start_t};
                }
                case TernaryBroadcastType::NONE:
                case TernaryBroadcastType::OUTER_BCAST:
                case TernaryBroadcastType::ROW_BCAST: return std::pair{0u, 0u};
                default: __builtin_unreachable();
            }
        }();
        std::array<uint32_t, num_kernel_args> compute_runtime_args = {num_tiles_per_core, freq, counter, scalar_arg};
        handle_args(program, compute_kernel_id, core, compute_runtime_args);

        start_tile_id += num_tiles_per_core;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::ternary {
TernaryDeviceOperation::TernaryProgramFactory::cached_program_t TernaryDeviceOperation::TernaryProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;
    TernaryVariant variant = operation_attributes.ternary_variant;
    TernaryBroadcastType broadcast_type = operation_attributes.broadcast_type;

    // Use TernaryKernelConfig to get the appropriate kernel names
    TernaryKernelConfig kernel_config(operation_attributes.ternary_op_type, variant, broadcast_type);

    auto program = CreateProgram();

    auto* device = predicate_tensor.device();

    auto predicate_data_format = datatype_to_dataformat_converter(predicate_tensor.dtype());
    /*  The where_llk uses UINT16 instrn set for bfloat16 inputs.
        If the bfloat16 inputs' CBs are set to UINT16 dataformat this will enable us to get 'NaN' in the outputs even
       for bfloat16 dtype. We need to test the impact of this hack on the composite ops that use where op and on the
       models, since bfloat16 packs NaN as inf as this is an expected behaviour and a known HW limitation of bfloat16
       dtype. Ex: Force the dataformat of all the bfloat16 inputs' CBs to be of UINT16 dataformat
        (predicate_tensor.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : predicate_tensor.dtype());
        datatype_to_dataformat_converter((output.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : output.dtype()); */

    // Handle data formats based on variant and tensor availability
    DataFormat value_true_data_format, value_false_data_format;
    if (variant == TernaryVariant::TTS) {
        // TTS: only value_true tensor exists
        value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.value().dtype());
        value_false_data_format = value_true_data_format;
    } else if (variant == TernaryVariant::TST) {
        // TST: only value_false tensor exists
        value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.value().dtype());
        value_true_data_format = value_false_data_format;
    } else {
        // TTT: both tensors exist
        value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.value().dtype());
        value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.value().dtype());
    }
    auto output_data_format = datatype_to_dataformat_converter(output.dtype());

    uint32_t predicate_single_tile_size = tt::tile_size(predicate_data_format);
    uint32_t value_true_single_tile_size = tt::tile_size(value_true_data_format);
    uint32_t value_false_single_tile_size = tt::tile_size(value_false_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Number of tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    // Input buffers - Create predicate CB (always c_0)
    auto [predicate_tensor_cb, predicate_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_0,
        program,
        all_device_cores,
        predicate_single_tile_size,
        num_tiles_per_cb,
        predicate_data_format);  // predicate_tensor

    // Create c_1 based on variant - this is the primary tensor CB
    uint32_t value_true_tensor_cb = 0;
    tt::tt_metal::CBHandle value_true_tensor_cb_handle;
    uint32_t value_false_tensor_cb = 0;
    tt::tt_metal::CBHandle value_false_tensor_cb_handle;

    if (variant == TernaryVariant::TTS) {
        // TTS: c_1 = value_true tensor (value_false is scalar)
        auto [cb, cb_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_true_single_tile_size,
            num_tiles_per_cb,
            value_true_data_format);
        value_true_tensor_cb = cb;
        value_true_tensor_cb_handle = cb_handle;
    } else if (variant == TernaryVariant::TST) {
        // TST: c_1 = value_false tensor (value_true is scalar)
        auto [cb, cb_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_false_single_tile_size,
            num_tiles_per_cb,
            value_false_data_format);
        value_false_tensor_cb = cb;
        value_false_tensor_cb_handle = cb_handle;
    } else if (variant == TernaryVariant::TTT) {
        auto [cb1, cb1_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_true_single_tile_size,
            num_tiles_per_cb,
            value_true_data_format);
        value_true_tensor_cb = cb1;
        value_true_tensor_cb_handle = cb1_handle;

        // Create CB for value_false (using actual false tensor)
        auto [cb2, cb2_handle] = create_cb(
            tt::CBIndex::c_2,
            program,
            all_device_cores,
            value_false_single_tile_size,  // Using actual false tensor size
            num_tiles_per_cb,
            value_false_data_format);  // Using actual false tensor format
        value_false_tensor_cb = cb2;
        value_false_tensor_cb_handle = cb2_handle;
    } else {
        TT_FATAL(false, "Unsupported Where variant in TernaryDeviceOperation. Supported: TTS, TST, TTT");
    }

    // Output buffer - use c_3 for all cases now
    auto output_cb_index = tt::CBIndex::c_3;
    auto [output_tensor_cb, output_tensor_cb_handle] = create_cb(
        output_cb_index,
        program,
        all_device_cores,
        output_single_tile_size,
        num_tiles_per_cb,
        output_data_format);  // output

    // BROADCAST DETECTION - Common for both reader and compute kernels
    bool pred_is_bcast = false, true_is_bcast = false, false_is_bcast = false;
    detect_broadcasts(
        pred_is_bcast,
        true_is_bcast,
        false_is_bcast,
        variant,
        broadcast_type,
        predicate_tensor,
        value_true_tensor,
        value_false_tensor);

    // READER KERNEL - Use kernel path from utils
    std::map<std::string, std::string> reader_defines;
    setup_reader_defines(
        reader_defines,
        variant,
        broadcast_type,
        predicate_tensor,
        value_true_tensor,
        value_false_tensor,
        pred_is_bcast,
        true_is_bcast,
        false_is_bcast);

    tt_metal::ReaderDataMovementConfig reader_config;

    if (variant == TernaryVariant::TTS) {
        // TTS: c_0 = predicate, c_1 = value_true tensor
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)predicate_tensor_cb, (std::uint32_t)value_true_tensor_cb};

        TensorAccessorArgs(*predicate_tensor.buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_true_tensor.value().buffer()).append_to(reader_compile_time_args);
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines);

    } else if (variant == TernaryVariant::TST) {
        // TST: c_0 = predicate, c_1 = value_false tensor
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)predicate_tensor_cb, (std::uint32_t)value_false_tensor_cb};
        TensorAccessorArgs(*predicate_tensor.buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_false_tensor.value().buffer()).append_to(reader_compile_time_args);
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines);
    } else if (variant == TernaryVariant::TTT) {
        // TTT: c_0 = predicate, c_1 = value_true, c_2 = value_false
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)predicate_tensor_cb,
            (std::uint32_t)value_true_tensor_cb,
            (std::uint32_t)value_false_tensor_cb};
        TensorAccessorArgs(*predicate_tensor.buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_true_tensor.value().buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_false_tensor.value().buffer()).append_to(reader_compile_time_args);
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines);
    }

    auto reader_kernel_id = tt_metal::CreateKernel(
        program, get_kernel_file_path(kernel_config.reader_kernel), all_device_cores, reader_config);

    // Use unary writer config for all cases (consistent with other writer variants)
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_tensor_cb};
    tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);
    tt_metal::WriterDataMovementConfig writer_config = tt_metal::WriterDataMovementConfig(writer_compile_time_args);

    auto writer_kernel_id = tt_metal::CreateKernel(
        program, get_kernel_file_path(kernel_config.writer_kernel), all_device_cores, writer_config);

    // COMPUTE KERNEL - Use kernel path from utils
    bool fp32_dest_acc_en = output_data_format == tt::DataFormat::UInt32 ||
                            output_data_format == tt::DataFormat::Int32 ||
                            output_data_format == tt::DataFormat::Float32;

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    // c_0 is always predicate
    unpack_to_dest_mode[tt::CBIndex::c_0] = (predicate_tensor.dtype() == DataType::FLOAT32)
                                                ? UnpackToDestMode::UnpackToDestFp32
                                                : UnpackToDestMode::Default;

    // c_1 assignment depends on variant
    if (variant == TernaryVariant::TTS) {
        // TTS: c_1 = value_true tensor
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_true_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    } else if (variant == TernaryVariant::TST) {
        // TST: c_1 = value_false tensor
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_false_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    } else {
        // TTT: c_1 = value_true tensor, c_2 = value_false tensor (including column broadcast)
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_true_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
        unpack_to_dest_mode[tt::CBIndex::c_2] = (value_false_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    }

    // Output CB depends on variant: c_2 for binary_ng compatibility (TTT col bcast), c_3 for other cases
    unpack_to_dest_mode[output_cb_index] =
        (output.dtype() == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;

    constexpr uint32_t num_tiles_per_cycle = 1;  // we produce 1 output tile per read-compute-write cycle

    // All variants use the same compile args now
    uint32_t scalar_is_true_value = (variant == TernaryVariant::TST) ? 1 : 0;
    std::vector<uint32_t> compute_kernel_args = {
        num_tiles_per_cycle, scalar_is_true_value};  // {num_tiles_per_cycle, scalar_is_true_value}
    std::map<std::string, std::string> kernel_defines;

    // Add binary_ng style defines for TTT column broadcast case
    if (variant == TernaryVariant::TTT &&
        (broadcast_type == TernaryBroadcastType::COL_BCAST || broadcast_type == TernaryBroadcastType::SCALAR_BCAST)) {
        // 3-tensor broadcast configuration - set defines for each tensor independently
        kernel_defines["BCAST_A"] = pred_is_bcast ? "1" : "0";
        kernel_defines["BCAST_B"] = true_is_bcast ? "1" : "0";
        kernel_defines["BCAST_C"] = false_is_bcast ? "1" : "0";
    } else if (
        (variant == TernaryVariant::TTS || variant == TernaryVariant::TST) &&
        broadcast_type == TernaryBroadcastType::COL_BCAST) {
        // Unified TTS/TST column broadcast configuration
        kernel_defines["BCAST_A"] = pred_is_bcast ? "1" : "0";
        if (variant == TernaryVariant::TTS) {
            kernel_defines["BCAST_B"] = true_is_bcast ? "1" : "0";
            kernel_defines["BCAST_C"] = "0";  // False is scalar for TTS
        } else {                              // TST
            kernel_defines["BCAST_B"] = "0";  // True is scalar for TST
            kernel_defines["BCAST_C"] = false_is_bcast ? "1" : "0";
        }
    }
    if ((variant == TernaryVariant::TTS || variant == TernaryVariant::TST) &&
        (broadcast_type == TernaryBroadcastType::SCALAR_A_BCAST ||
         broadcast_type == TernaryBroadcastType::SCALAR_B_BCAST)) {
        // Unified TTS/TST scalar broadcast configuration
        kernel_defines["BCAST_A"] = (broadcast_type == TernaryBroadcastType::SCALAR_A_BCAST) ? "1" : "0";
        if (variant == TernaryVariant::TTS) {
            kernel_defines["BCAST_B"] = (broadcast_type == TernaryBroadcastType::SCALAR_B_BCAST) ? "1" : "0";
            kernel_defines["BCAST_C"] = "0";  // False is scalar for TTS
        } else {                              // TST
            kernel_defines["BCAST_B"] = "0";  // True is scalar for TST
            kernel_defines["BCAST_C"] = (broadcast_type == TernaryBroadcastType::SCALAR_B_BCAST) ? "1" : "0";
        }
    }
    // Get operation-specific compute kernel defines
    auto op_defines =
        CMAKE_UNIQUE_NAMESPACE::get_ternary_compute_defines(operation_attributes, predicate_tensor.dtype());
    for (const auto& [key, value] : op_defines) {
        kernel_defines[key] = value;
    }

    // Add common fill defines
    kernel_defines["FILL_LLK"] = "fill_tile";
    if (predicate_tensor.dtype() == DataType::INT32) {
        kernel_defines["FILL_LLK"] = "fill_tile_int";
        kernel_defines["FILL_WITH_VALUE_INT"] = "1";
    } else {
        kernel_defines["FILL_WITH_VALUE_FLOAT"] = "1";
    }

    tt_metal::ComputeConfig compute_config{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .compile_args = compute_kernel_args,
        .defines = kernel_defines};

    auto compute_kernel_id = tt_metal::CreateKernel(
        program, get_kernel_file_path(kernel_config.compute_kernel), all_device_cores, compute_config);

    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        broadcast_type,
        set_runtime_args);

    return {
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, compute_with_storage_grid_size}};
}

void TernaryDeviceOperation::TernaryProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto update_args =
        [](tt::tt_metal::Program& program, tt::tt_metal::KernelHandle kernel_id, CoreCoord core, auto&& args) {
            auto& all_args = GetRuntimeArgs(program, kernel_id);
            auto& core_args = all_args.at(core.x).at(core.y);
            std::copy(args.begin(), args.end(), core_args.data());
        };

    // Detect broadcast type for the cached program
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;
    TernaryBroadcastType broadcast_type = TernaryBroadcastType::NONE;
    if (operation_attributes.ternary_variant == TernaryVariant::TTT) {
        broadcast_type = get_broadcast_type(
            predicate_tensor.logical_shape(),
            value_true_tensor.value().logical_shape(),
            value_false_tensor.value().logical_shape());
    }
    if (operation_attributes.ternary_variant == TernaryVariant::TTS) {
        broadcast_type =
            get_broadcast_type(predicate_tensor.logical_shape(), value_true_tensor.value().logical_shape());
    }
    if (operation_attributes.ternary_variant == TernaryVariant::TST) {
        broadcast_type =
            get_broadcast_type(predicate_tensor.logical_shape(), value_false_tensor.value().logical_shape());
    }

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        cached_program.program,
        cached_program.shared_variables.reader_kernel_id,
        cached_program.shared_variables.writer_kernel_id,
        cached_program.shared_variables.compute_kernel_id,
        cached_program.shared_variables.compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        broadcast_type,
        update_args);
}

}  // namespace ttnn::operations::ternary
