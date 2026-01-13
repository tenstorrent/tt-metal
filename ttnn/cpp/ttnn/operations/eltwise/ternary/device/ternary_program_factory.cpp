// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_device_operation.hpp"
#include "ternary_op_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_utils.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.hpp"

using namespace tt::tt_metal;
namespace {
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
    const ttnn::Tensor& predicate_tensor,
    const std::optional<ttnn::Tensor>& value_true_tensor,
    const std::optional<ttnn::Tensor>& value_false_tensor,
    bool pred_is_bcast,
    bool true_is_bcast,
    bool false_is_bcast,
    bool has_sharding) {
    using TernaryVariant = ttnn::operations::ternary::TernaryVariant;

    // Get sharding status for all tensors
    auto [predicate_sharded, value_true_sharded, value_false_sharded] =
        get_tensor_sharding_status(predicate_tensor, value_true_tensor, value_false_tensor);

    // Set up dataflow defines first
    if (variant == TernaryVariant::TTT) {
        reader_defines = ttnn::operations::ternary::make_dataflow_defines(
            predicate_tensor.dtype(), value_true_tensor.value().dtype(), value_false_tensor.value().dtype());
        add_broadcast_defines(reader_defines, pred_is_bcast, true_is_bcast, false_is_bcast);
    } else if (variant == TernaryVariant::TTS) {
        reader_defines = ttnn::operations::ternary::make_dataflow_defines(
            predicate_tensor.dtype(), value_true_tensor.value().dtype());
        add_broadcast_defines(reader_defines, pred_is_bcast, true_is_bcast, false_is_bcast);
    } else if (variant == TernaryVariant::TST) {
        reader_defines = ttnn::operations::ternary::make_dataflow_defines(
            predicate_tensor.dtype(), value_false_tensor.value().dtype());
        // For TST the two tensors are predicate and false_tensors and are SRC_A and SRC_B respectively
        add_broadcast_defines(reader_defines, pred_is_bcast, false_is_bcast, true_is_bcast);
    }

    // Set sharding defines after dataflow defines
    reader_defines["SRC_SHARDED_A"] = (predicate_sharded && has_sharding) ? "1" : "0";
    reader_defines["SRC_SHARDED_B"] = (value_true_sharded && has_sharding) ? "1" : "0";
    reader_defines["SRC_SHARDED_C"] = (value_false_sharded && has_sharding) ? "1" : "0";
}

namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::ternary;

std::uint32_t* copy_common_runtime_args(const tt::tt_metal::Buffer& buffer, std::uint32_t* dst) {
    const auto src = tt::tt_metal::TensorAccessorArgs(buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
                         .get_common_runtime_args();
    return std::copy(src.begin(), src.end(), dst);
}

bool is_llk_bcast(
    const ttnn::operations::ternary::TernaryBroadcastType broadcast_type,
    const ttnn::DataType a_dtype,
    const ttnn::DataType b_dtype,
    const ttnn::DataType c_dtype) {
    if (broadcast_type == TernaryBroadcastType::ROW_BCAST) {
        if (a_dtype == ttnn::DataType::BFLOAT16 && b_dtype == ttnn::DataType::BFLOAT16 &&
            c_dtype == ttnn::DataType::BFLOAT16) {
            return true;
        }
    }
    return false;
}

void overwrite_compute_kernel_name_and_defines(
    ttnn::operations::ternary::KernelName& kernel_name,
    const ttnn::operations::ternary::TernaryBroadcastType broadcast_type,
    const ttnn::operations::ternary::TernaryOpType op_type) {
    if (broadcast_type == TernaryBroadcastType::ROW_BCAST) {
        kernel_name =
            op_type == TernaryOpType::ADDCMUL ? KernelName::ComputeRowBcastAddcmul : KernelName::ComputeRowBcastTTT;
    }
}

// Get operation-specific compute kernel defines
std::map<std::string, std::string> get_ternary_compute_defines(
    const TernaryDeviceOperation::operation_attributes_t& operation_attributes, ttnn::DataType dtype) {
    return get_compute_defines(operation_attributes.ternary_op_type, dtype);
}

// Helper function to get shape dimensions
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_shape_dims(const ttnn::Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {
        shape.rank() >= 5 ? shape[-5] : 1,
        shape[-4],
        shape[-3],
        shape[-2] / tile.get_height(),
        shape[-1] / tile.get_width()};
}

// Helper function to get memory layout
TensorMemoryLayout get_memory_layout(
    const ttnn::Tensor& predicate_tensor,
    const std::optional<ttnn::Tensor>& value_true_tensor,
    const std::optional<ttnn::Tensor>& value_false_tensor,
    const ttnn::Tensor& output) {
    if (output.memory_config().is_sharded()) {
        return output.memory_config().memory_layout();
    }

    if (predicate_tensor.memory_config().is_sharded()) {
        return predicate_tensor.memory_config().memory_layout();
    }

    if (value_true_tensor.has_value() && value_true_tensor->memory_config().is_sharded()) {
        return value_true_tensor->memory_config().memory_layout();
    }

    if (value_false_tensor.has_value() && value_false_tensor->memory_config().is_sharded()) {
        return value_false_tensor->memory_config().memory_layout();
    }

    return TensorMemoryLayout::INTERLEAVED;
}

// Get number of shards per width dimension
uint32_t get_shards_per_width(const tt::tt_metal::ShardSpec& shard_spec, TensorMemoryLayout memory_layout) {
    auto num_cores = shard_spec.grid.num_cores();
    if (memory_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
        return 1;
    }

    if (memory_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        return num_cores;
    }

    const auto& bbox = shard_spec.grid.bounding_box();
    const auto& start = bbox.start_coord;
    const auto& end = bbox.end_coord;
    return (shard_spec.orientation == ShardOrientation::ROW_MAJOR ? end.x - start.x : end.y - start.y) + 1;
}

// Helper functions for TensorSpec
const std::optional<ShardSpec>& get_shard_spec(const TensorSpec& tensor_spec) {
    return tensor_spec.memory_config().shard_spec();
}

inline auto is_uneven(const TensorSpec& t) {
    if (not t.memory_config().is_sharded()) {
        return false;
    }

    const auto& shape = t.padded_shape();
    const auto& shard = get_shard_spec(t)->shape;
    const auto rank = shape.rank();

    // Compute product of all dimensions except the last
    uint64_t volume_except_last = 1;
    for (int i = 0; i < static_cast<int>(rank) - 1; ++i) {
        volume_except_last *= shape[i];
    }

    return (volume_except_last % shard[0]) != 0 or (shape[-1] % shard[1]) != 0;
}

bool is_native_L1_sharding(
    const TensorSpec& predicate_spec,
    const std::optional<TensorSpec>& true_spec,
    const std::optional<TensorSpec>& false_spec,
    const TensorSpec& output_spec) {
    // Only support TTT variant
    if (!true_spec.has_value() || !false_spec.has_value()) {
        return false;
    }

    // Output must be sharded
    if (!output_spec.memory_config().is_sharded()) {
        return false;
    }

    // All shapes must be identical and predicate/true/false must have matching layouts
    if (predicate_spec.logical_shape() == true_spec->logical_shape() &&
        predicate_spec.logical_shape() == false_spec->logical_shape() &&
        predicate_spec.memory_config() == true_spec->memory_config() &&
        predicate_spec.memory_config() == false_spec->memory_config()) {
        if (is_uneven(predicate_spec) || is_uneven(*true_spec) || is_uneven(*false_spec) || is_uneven(output_spec)) {
            return false;
        }
        if (predicate_spec.memory_config().buffer_type() == BufferType::DRAM ||
            true_spec->memory_config().buffer_type() == BufferType::DRAM ||
            false_spec->memory_config().buffer_type() == BufferType::DRAM ||
            output_spec.memory_config().buffer_type() == BufferType::DRAM) {
            return false;
        }
        if ((predicate_spec.memory_config().is_sharded() &&
             predicate_spec.memory_config().buffer_type() == BufferType::L1)) {
            return true;
        }
        if ((true_spec->memory_config().is_sharded() && true_spec->memory_config().buffer_type() == BufferType::L1)) {
            return true;
        }
        if ((false_spec->memory_config().is_sharded() && false_spec->memory_config().buffer_type() == BufferType::L1)) {
            return true;
        }
        if ((output_spec.memory_config().is_sharded() && output_spec.memory_config().buffer_type() == BufferType::L1)) {
            return true;
        }
    }

    return false;
}

std::optional<AllShardSpecs> get_shard_specs(
    const TensorSpec& predicate_spec,
    const std::optional<TensorSpec>& true_spec,
    const std::optional<TensorSpec>& false_spec,
    const TensorSpec& output_spec) {
    // Only support TTT variant
    if (!true_spec.has_value() || !false_spec.has_value()) {
        return std::nullopt;
    }

    bool predicate_sharded = predicate_spec.memory_config().is_sharded();
    bool true_sharded = true_spec->memory_config().is_sharded();
    bool false_sharded = false_spec->memory_config().is_sharded();
    bool output_sharded = output_spec.memory_config().is_sharded();

    if ((!predicate_sharded && !true_sharded && !false_sharded) && !output_sharded) {
        return std::nullopt;
    }

    if (!is_native_L1_sharding(predicate_spec, true_spec, false_spec, output_spec)) {
        // treat as interleaved
        return std::nullopt;
    }

    const auto& predicate_shape = predicate_spec.padded_shape();
    const auto& true_shape = true_spec->padded_shape();
    const auto& false_shape = false_spec->padded_shape();
    const auto& output_shape = output_spec.padded_shape();

    // Output must have a shard spec
    TT_FATAL(get_shard_spec(output_spec).has_value(), "Output must have a shard spec");
    return AllShardSpecs{
        predicate_sharded ? *get_shard_spec(predicate_spec)
                          : adjust_to_shape(*get_shard_spec(output_spec), output_shape, predicate_shape),
        true_sharded ? *get_shard_spec(*true_spec)
                     : adjust_to_shape(*get_shard_spec(output_spec), output_shape, true_shape),
        false_sharded ? *get_shard_spec(*false_spec)
                      : adjust_to_shape(*get_shard_spec(output_spec), output_shape, false_shape),
        *get_shard_spec(output_spec)};
}

// ShardShapeGenerator class
class ShardShapeGenerator {
    CoreCoord end_core;
    bool row_major{};
    TensorMemoryLayout memory_layout{TensorMemoryLayout::INTERLEAVED};
    std::array<uint32_t, 2> shard_shape{};
    std::array<uint32_t, 2> last_shard_shape{};

public:
    ShardShapeGenerator() = default;

    ShardShapeGenerator(const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Tensor& tensor) :
        // core ranges are sorted, so the last one is indeed the last core
        end_core(shard_spec.grid.ranges().rbegin()->end_coord),
        row_major(shard_spec.orientation == ShardOrientation::ROW_MAJOR),
        memory_layout(tensor.memory_config().memory_layout()) {
        auto tile_height = tensor.tensor_spec().tile().get_height();
        auto tile_width = tensor.tensor_spec().tile().get_width();

        shard_shape = {
            tt::round_up(shard_spec.shape[0], tile_height) / tile_height,
            tt::round_up(shard_spec.shape[1], tile_width) / tile_width};

        TT_FATAL(
            shard_shape[0] != 0 and shard_shape[1] != 0,
            "Shard shape must not contain zero dimensions but got {{{}, {}}}",
            shard_shape[0],
            shard_shape[1]);

        const auto [D, N, C, Ht, Wt] = get_shape_dims(tensor);
        const auto unrolled_Ht = D * N * C * Ht;
        last_shard_shape = {
            shard_shape[0] - (tt::round_up(unrolled_Ht, shard_shape[0]) - unrolled_Ht),
            shard_shape[1] - (tt::round_up(Wt, shard_shape[1]) - Wt),
        };
    }
    std::array<uint32_t, 2> operator()(CoreCoord core) const {
        const unsigned majorDim = row_major ? 1 : 0;
        const unsigned minorDim = row_major ? 0 : 1;
        auto current_shape = shard_shape;
        if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            if (core == end_core) {
                current_shape[majorDim] = last_shard_shape[majorDim];
                current_shape[minorDim] = last_shard_shape[minorDim];
            }
        } else if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            // For BLOCK_SHARDED, edges can have uneven shards
            if (row_major) {
                if (core.x == end_core.x) {
                    current_shape[1] = last_shard_shape[1];  // width
                }
                if (core.y == end_core.y) {
                    current_shape[0] = last_shard_shape[0];  // height
                }
            } else {  // col_major
                if (core.y == end_core.y) {
                    current_shape[1] = last_shard_shape[1];  // width
                }
                if (core.x == end_core.x) {
                    current_shape[0] = last_shard_shape[0];  // height
                }
            }
        }
        return current_shape;
    }
};

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
    bool has_sharding,
    int out_rank,
    uint32_t tile_h,
    uint32_t tile_w) {
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
    const TernaryDeviceOperation::operation_attributes_t& operation_attributes,
    const TernaryDeviceOperation::tensor_args_t& tensor_args,
    TernaryDeviceOperation::tensor_return_value_t& output,
    ttnn::operations::ternary::TernaryBroadcastType broadcast_type,
    F handle_args) {
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;
    TernaryVariant variant = operation_attributes.ternary_variant;
    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    // Get shard specs early
    const auto shard_specs = get_shard_specs(
        predicate_tensor.tensor_spec(),
        value_true_tensor.has_value() ? value_true_tensor->tensor_spec() : std::optional<TensorSpec>{},
        value_false_tensor.has_value() ? value_false_tensor->tensor_spec() : std::optional<TensorSpec>{},
        output.tensor_spec());
    const bool has_sharding = shard_specs.has_value();
    auto grid = has_sharding ? shard_specs->predicate_shard_spec.grid : CoreRangeSet{};

    const auto row_major =
        has_sharding ? shard_specs->predicate_shard_spec.orientation == ShardOrientation::ROW_MAJOR : true;

    // zero_start_grid is a flag to indicate that we are using a single rectangular grid that starts at (0, 0)
    // as well as having the sharded tensors (if any) start at (0, 0)
    bool zero_start_grid = false;
    CoreCoord compute_with_storage_grid;
    const auto& all_device_cores = operation_attributes.worker_grid;
    if (grid.size() == 1) {
        const auto& cr = *all_device_cores.ranges().begin();
        if (cr.start_coord.x == 0 && cr.start_coord.y == 0) {
            if (has_sharding) {
                const auto& shard_start_coord = grid.ranges()[0].start_coord;
                if (shard_start_coord.x == 0 && shard_start_coord.y == 0) {
                    zero_start_grid = true;
                    compute_with_storage_grid = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
                }
            } else {
                zero_start_grid = true;
                compute_with_storage_grid = CoreCoord(cr.end_coord.x + 1, cr.end_coord.y + 1);
            }
        }
    }
    const uint32_t num_cores_total =
        zero_start_grid ? compute_with_storage_grid.x * compute_with_storage_grid.y : all_device_cores.num_cores();

    uint32_t num_tiles_per_core_group_1{}, num_tiles_per_core_group_2{};
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cores;
    std::vector<CoreCoord> cores;

    const uint32_t tile_height = output.tensor_spec().tile().get_height();
    const uint32_t tile_width = output.tensor_spec().tile().get_width();
    uint32_t c_shard_height{}, c_shard_width{}, num_shards_per_width{};

    ShardShapeGenerator predicate_shard_shape_generator;
    ShardShapeGenerator true_shard_shape_generator;
    ShardShapeGenerator false_shard_shape_generator;
    ShardShapeGenerator output_shard_shape_generator;

    if (has_sharding) {
        core_group_1 = grid;
        predicate_shard_shape_generator = ShardShapeGenerator(shard_specs->predicate_shard_spec, predicate_tensor);
        if (value_true_tensor.has_value()) {
            true_shard_shape_generator = ShardShapeGenerator(shard_specs->true_shard_spec, *value_true_tensor);
        }
        if (value_false_tensor.has_value()) {
            false_shard_shape_generator = ShardShapeGenerator(shard_specs->false_shard_spec, *value_false_tensor);
        }
        output_shard_shape_generator = ShardShapeGenerator(shard_specs->output_shard_spec, output);
        c_shard_height = shard_specs->output_shard_spec.shape[0] / tile_height;
        c_shard_width = shard_specs->output_shard_spec.shape[1] / tile_width;
        num_shards_per_width = get_shards_per_width(
            shard_specs->output_shard_spec,
            get_memory_layout(predicate_tensor, value_true_tensor, value_false_tensor, output));

        if (zero_start_grid) {
            auto bbox = core_group_1.bounding_box();
            cores = grid_to_cores_with_noop(
                bbox.end_coord.x,
                bbox.end_coord.y,
                compute_with_storage_grid.x,
                compute_with_storage_grid.y,
                row_major);
        } else {
            cores = grid_to_cores_with_noop(core_group_1, all_device_cores, row_major);
        }
    } else if (zero_start_grid) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid, num_output_tiles, row_major);
        cores = grid_to_cores(num_cores_total, compute_with_storage_grid.x, compute_with_storage_grid.y, row_major);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(all_device_cores, num_output_tiles, row_major);
        cores = corerange_to_cores(all_device_cores, {}, row_major);
    }
    constexpr size_t num_reader_args = 27;
    constexpr size_t num_writer_args = 11;
    constexpr size_t num_kernel_args = 4;

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core = 0;
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
        uint32_t a_num_tiles = 0, b_num_tiles = 0, f_num_tiles = 0, c_current_shard_width = 0;
        const auto out_rank = output.logical_shape().rank();
        const auto& tile = output.tensor_spec().tile();
        const uint32_t tile_h = tile.get_height();
        const uint32_t tile_w = tile.get_width();
        auto output_dims = extract_tensor_dimensions(output, out_rank, tile_h, tile_w);

        uint32_t c_start_id = 0;
        if (has_sharding) {
            auto output_shard_shape = output_shard_shape_generator(core);
            num_tiles_per_core = output_shard_shape[0] * output_shard_shape[1];  // actual
            c_current_shard_width = output_shard_shape[1];                       // actual
            auto predicate_shard_shape = predicate_shard_shape_generator(core);
            a_num_tiles = predicate_shard_shape[0] * predicate_shard_shape[1];  // actual
            if (value_true_tensor.has_value()) {
                auto true_shard_shape = true_shard_shape_generator(core);
                b_num_tiles = true_shard_shape[0] * true_shard_shape[1];  // actual
            }
            if (value_false_tensor.has_value()) {
                auto false_shard_shape = false_shard_shape_generator(core);
                f_num_tiles = false_shard_shape[0] * false_shard_shape[1];  // actual
            }
            c_start_id = (i / num_shards_per_width) * (c_shard_height * output_dims.Wt) +
                         (i % num_shards_per_width) * c_shard_width;
        } else {
            c_start_id = start_tile_id;
        }

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
                c_start_id,
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
            // Tile counts are already calculated above if has_sharding
            std::array<uint32_t, num_reader_args> reader_runtime_args{};              // zero-initialized
            reader_runtime_args[0] = predicate_tensor.buffer()->address();            // 0: src0_addr (predicate)
            reader_runtime_args[1] = value_true_tensor.value().buffer()->address();   // 1: src1_addr (true tensor)
            reader_runtime_args[2] = value_false_tensor.value().buffer()->address();  // 2: src2_addr (false tensor)
            reader_runtime_args[3] = num_tiles_per_core;                              // 3: num_tiles (per core)
            reader_runtime_args[4] = c_start_id;                                      // 4: start_id
            // Set arguments for both broadcast and no-bcast cases (same arguments needed for width sharding)
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

            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        } else {
            TT_FATAL(false, "Unsupported Where variant in TernaryDeviceOperation. Supported: TTS, TST, TTT");
        }

        // Writer runtime args
        std::array writer_runtime_args = {
            output.buffer()->address(),  // 0: dst_addr
            num_tiles_per_core,          // 1: num_tiles
            c_start_id,                  // 2: start_id
            c_current_shard_width,       // 3: dst_shard_width
            output_dims.D,               // 4: D
            output_dims.N,               // 5: N
            output_dims.C,               // 6: C
            output_dims.Ht,              // 7: Ht
            output_dims.Wt,              // 8: Wt
            output_dims.ND,              // 9: cND
            0u                           // 10: padding
        };
        handle_args(program, writer_kernel_id, core, writer_runtime_args);

        // Compute runtime args
        uint32_t scalar_arg = 0u;
        if (variant == TernaryVariant::TTS) {
            scalar_arg = pack_scalar_runtime_arg(operation_attributes.scalar_input_b.value(), output.dtype());
        } else if (
            variant == TernaryVariant::TST || (operation_attributes.ternary_op_type == TernaryOpType::ADDCMUL &&
                                               operation_attributes.scalar_input_a.has_value())) {
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

std::optional<AllShardVolumes> get_shard_volumes(
    const TensorSpec& predicate_spec,
    const std::optional<TensorSpec>& true_spec,
    const std::optional<TensorSpec>& false_spec,
    const TensorSpec& output_spec) {
    const auto shard_specs =
        CMAKE_UNIQUE_NAMESPACE::get_shard_specs(predicate_spec, true_spec, false_spec, output_spec);

    if (not shard_specs.has_value()) {
        return std::nullopt;
    }

    const auto predicate_sharded = predicate_spec.memory_config().is_sharded();
    const auto true_sharded = true_spec.has_value() and true_spec->memory_config().is_sharded();
    const auto false_sharded = false_spec.has_value() and false_spec->memory_config().is_sharded();
    const auto output_sharded = output_spec.memory_config().is_sharded();
    const auto tile_hw = output_spec.tile().get_tile_hw();

    return AllShardVolumes{
        .predicate_shard_volume =
            predicate_sharded ? shard_specs->predicate_shard_spec.numel() / tile_hw : std::optional<std::uint32_t>{},
        .true_shard_volume =
            true_sharded ? shard_specs->true_shard_spec.numel() / tile_hw : std::optional<std::uint32_t>{},
        .false_shard_volume =
            false_sharded ? shard_specs->false_shard_spec.numel() / tile_hw : std::optional<std::uint32_t>{},
        .output_shard_volume =
            output_sharded ? shard_specs->output_shard_spec.numel() / tile_hw : std::optional<std::uint32_t>{},
    };
}

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
    auto reader_kernel = kernel_config.reader_kernel;
    auto compute_kernel = kernel_config.compute_kernel;
    auto writer_kernel = kernel_config.writer_kernel;

    auto program = CreateProgram();

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

    // Get shard volumes (using TensorSpec like binary_ng)
    const auto shard_volumes = get_shard_volumes(
        predicate_tensor.tensor_spec(),
        value_true_tensor.has_value() ? value_true_tensor->tensor_spec() : std::optional<TensorSpec>{},
        value_false_tensor.has_value() ? value_false_tensor->tensor_spec() : std::optional<TensorSpec>{},
        output.tensor_spec());
    const bool has_sharding = shard_volumes.has_value();
    const auto predicate_sharded = has_sharding and shard_volumes->predicate_shard_volume.has_value();
    const auto true_sharded = has_sharding and shard_volumes->true_shard_volume.has_value();
    const auto false_sharded = has_sharding and shard_volumes->false_shard_volume.has_value();
    const auto output_sharded = has_sharding and shard_volumes->output_shard_volume.has_value();
    const auto predicate_num_tiles_per_shard = has_sharding ? shard_volumes->predicate_shard_volume : std::nullopt;
    const auto true_num_tiles_per_shard = has_sharding ? shard_volumes->true_shard_volume : std::nullopt;
    const auto false_num_tiles_per_shard = has_sharding ? shard_volumes->false_shard_volume : std::nullopt;
    const auto output_num_tiles_per_shard = has_sharding ? shard_volumes->output_shard_volume : std::nullopt;

    const auto& all_device_cores = operation_attributes.worker_grid;

    constexpr uint32_t num_tiles_per_cb = 2;
    // Input buffers - Create predicate CB (always c_0)
    auto [predicate_tensor_cb, predicate_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_0,
        program,
        all_device_cores,
        predicate_single_tile_size,
        predicate_num_tiles_per_shard.value_or(num_tiles_per_cb),
        predicate_data_format,
        predicate_sharded ? predicate_tensor.buffer() : nullptr);

    // Create c_1 based on variant - this is the primary tensor CB
    uint32_t value_true_tensor_cb = 0;
    [[maybe_unused]] tt::tt_metal::CBHandle value_true_tensor_cb_handle{};
    uint32_t value_false_tensor_cb = 0;
    [[maybe_unused]] tt::tt_metal::CBHandle value_false_tensor_cb_handle{};

    if (variant == TernaryVariant::TTS) {
        // TTS: c_1 = value_true tensor (value_false is scalar)
        auto [cb, cb_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_true_single_tile_size,
            true_num_tiles_per_shard.value_or(num_tiles_per_cb),
            value_true_data_format,
            true_sharded ? value_true_tensor->buffer() : nullptr);
        value_true_tensor_cb = cb;
        value_true_tensor_cb_handle = cb_handle;
    } else if (variant == TernaryVariant::TST) {
        // TST: c_1 = value_false tensor (value_true is scalar)
        auto [cb, cb_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_false_single_tile_size,
            false_num_tiles_per_shard.value_or(num_tiles_per_cb),
            value_false_data_format,
            false_sharded ? value_false_tensor->buffer() : nullptr);
        value_false_tensor_cb = cb;
        value_false_tensor_cb_handle = cb_handle;
    } else if (variant == TernaryVariant::TTT) {
        auto [cb1, cb1_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_true_single_tile_size,
            true_num_tiles_per_shard.value_or(num_tiles_per_cb),
            value_true_data_format,
            true_sharded ? value_true_tensor->buffer() : nullptr);
        value_true_tensor_cb = cb1;
        value_true_tensor_cb_handle = cb1_handle;

        // Create CB for value_false (using actual false tensor)
        auto [cb2, cb2_handle] = create_cb(
            tt::CBIndex::c_2,
            program,
            all_device_cores,
            value_false_single_tile_size,
            false_num_tiles_per_shard.value_or(num_tiles_per_cb),
            value_false_data_format,
            false_sharded ? value_false_tensor->buffer() : nullptr);
        value_false_tensor_cb = cb2;
        value_false_tensor_cb_handle = cb2_handle;
    } else {
        TT_FATAL(false, "Unsupported Where variant in TernaryDeviceOperation. Supported: TTS, TST, TTT");
    }

    if (variant == TernaryVariant::TTT && broadcast_type == TernaryBroadcastType::ROW_BCAST) {
        create_cb(
            tt::CBIndex::c_4,
            program,
            all_device_cores,
            predicate_single_tile_size,
            num_tiles_per_cb,
            predicate_data_format);
        create_cb(
            tt::CBIndex::c_5,
            program,
            all_device_cores,
            value_true_single_tile_size,
            num_tiles_per_cb,
            value_true_data_format);
        create_cb(
            tt::CBIndex::c_6,
            program,
            all_device_cores,
            value_false_single_tile_size,
            num_tiles_per_cb,
            value_false_data_format);
    }
    // Output buffer - use c_3 for all cases now
    auto output_cb_index = tt::CBIndex::c_3;
    auto [output_tensor_cb, output_tensor_cb_handle] = create_cb(
        output_cb_index,
        program,
        all_device_cores,
        output_single_tile_size,
        output_num_tiles_per_shard.value_or(num_tiles_per_cb),
        output_data_format,
        output_sharded ? output.buffer() : nullptr);

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
        predicate_tensor,
        value_true_tensor,
        value_false_tensor,
        pred_is_bcast,
        true_is_bcast,
        false_is_bcast,
        has_sharding);

    std::map<std::string, std::string> kernel_defines;
    if (variant == TernaryVariant::TTT) {
        if (CMAKE_UNIQUE_NAMESPACE::is_llk_bcast(
                broadcast_type,
                predicate_tensor.dtype(),
                value_true_tensor.value().dtype(),
                value_false_tensor.value().dtype())) {
            CMAKE_UNIQUE_NAMESPACE::overwrite_compute_kernel_name_and_defines(
                compute_kernel, broadcast_type, operation_attributes.ternary_op_type);
            reader_defines["BCAST_LLK"] = "1";
        } else {
            reader_defines["BCAST_LLK"] = "0";
        }
    }
    tt_metal::ReaderDataMovementConfig reader_config;
    std::vector<uint32_t> reader_common_runtime_args;

    if (variant == TernaryVariant::TTS) {
        // TTS: c_0 = predicate, c_1 = value_true tensor
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)predicate_tensor_cb, (std::uint32_t)value_true_tensor_cb};

        tt::tt_metal::TensorAccessorArgs(*predicate_tensor.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
            .append_to(reader_compile_time_args, reader_common_runtime_args);
        tt::tt_metal::TensorAccessorArgs(
            *value_true_tensor.value().buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
            .append_to(reader_compile_time_args, reader_common_runtime_args);
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines);

    } else if (variant == TernaryVariant::TST) {
        // TST: c_0 = predicate, c_1 = value_false tensor
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)predicate_tensor_cb, (std::uint32_t)value_false_tensor_cb};
        tt::tt_metal::TensorAccessorArgs(*predicate_tensor.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
            .append_to(reader_compile_time_args, reader_common_runtime_args);
        tt::tt_metal::TensorAccessorArgs(
            *value_false_tensor.value().buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
            .append_to(reader_compile_time_args, reader_common_runtime_args);
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines);
    } else if (variant == TernaryVariant::TTT) {
        std::vector<uint32_t> reader_compile_time_args;
        reader_compile_time_args.push_back((std::uint32_t)predicate_tensor_cb);
        reader_compile_time_args.push_back((std::uint32_t)value_true_tensor_cb);
        reader_compile_time_args.push_back((std::uint32_t)value_false_tensor_cb);
        tt::tt_metal::TensorAccessorArgs(*predicate_tensor.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
            .append_to(reader_compile_time_args, reader_common_runtime_args);
        tt::tt_metal::TensorAccessorArgs(
            *value_true_tensor.value().buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
            .append_to(reader_compile_time_args, reader_common_runtime_args);
        tt::tt_metal::TensorAccessorArgs(
            *value_false_tensor.value().buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
            .append_to(reader_compile_time_args, reader_common_runtime_args);
        reader_compile_time_args.push_back(static_cast<uint32_t>(has_sharding));
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines);
    }

    bool is_fpu = false;

    if (operation_attributes.ternary_op_type == TernaryOpType::ADDCMUL) {
        is_fpu = (predicate_tensor.dtype() == value_true_tensor.value().dtype()) &&
                 (predicate_tensor.dtype() == value_false_tensor.value().dtype()) &&
                 (predicate_tensor.dtype() != DataType::FLOAT32 && predicate_tensor.dtype() != DataType::INT32 &&
                  predicate_tensor.dtype() != DataType::UINT32);
    }

    auto reader_kernel_id =
        tt_metal::CreateKernel(program, get_kernel_file_path(reader_kernel, is_fpu), all_device_cores, reader_config);
    tt_metal::SetCommonRuntimeArgs(program, reader_kernel_id, reader_common_runtime_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_tensor_cb};
    std::vector<uint32_t> writer_common_runtime_args;
    tt::tt_metal::TensorAccessorArgs(*output.buffer(), tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_compile_time_args, writer_common_runtime_args);
    writer_compile_time_args.push_back(static_cast<uint32_t>(has_sharding));
    std::map<std::string, std::string> writer_defines;
    writer_defines["DST_SHARDED"] = (output_sharded && has_sharding) ? "1" : "0";
    tt_metal::WriterDataMovementConfig writer_config =
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines);

    auto writer_kernel_id =
        tt_metal::CreateKernel(program, get_kernel_file_path(writer_kernel, is_fpu), all_device_cores, writer_config);
    tt_metal::SetCommonRuntimeArgs(program, writer_kernel_id, writer_common_runtime_args);

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

    // Add binary_ng style defines for TTT column broadcast case
    if (variant == TernaryVariant::TTT) {
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

    auto compute_kernel_id =
        tt_metal::CreateKernel(program, get_kernel_file_path(compute_kernel, is_fpu), all_device_cores, compute_config);

    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        operation_attributes,
        tensor_args,
        output,
        broadcast_type,
        set_runtime_args);

    // Update dynamic circular buffer addresses for sharded tensors
    if (has_sharding) {
        TT_FATAL(
            value_true_tensor.has_value() && value_false_tensor.has_value(),
            "Dynamic buffer updates only supported for TTT variant");
        if (predicate_tensor.is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, predicate_tensor_cb_handle, *predicate_tensor.buffer());
        }
        if (value_true_tensor->is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, value_true_tensor_cb_handle, *value_true_tensor->buffer());
        }
        if (value_false_tensor->is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, value_false_tensor_cb_handle, *value_false_tensor->buffer());
        }
        if (output.is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, output_tensor_cb_handle, *output.buffer());
        }
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id}};
}

void TernaryDeviceOperation::TernaryProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    {
        const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;
        auto* predicate_buffer = predicate_tensor.buffer();
        auto* output_buffer = output.buffer();

        // Update common runtime args for reader kernel
        auto* args = GetCommonRuntimeArgs(program, reader_kernel_id).data();
        if (operation_attributes.ternary_variant == TernaryVariant::TTS) {
            args = CMAKE_UNIQUE_NAMESPACE::copy_common_runtime_args(*predicate_buffer, args);
            CMAKE_UNIQUE_NAMESPACE::copy_common_runtime_args(*value_true_tensor.value().buffer(), args);
        } else if (operation_attributes.ternary_variant == TernaryVariant::TST) {
            args = CMAKE_UNIQUE_NAMESPACE::copy_common_runtime_args(*predicate_buffer, args);
            CMAKE_UNIQUE_NAMESPACE::copy_common_runtime_args(*value_false_tensor.value().buffer(), args);
        } else {  // TTT
            args = CMAKE_UNIQUE_NAMESPACE::copy_common_runtime_args(*predicate_buffer, args);
            args = CMAKE_UNIQUE_NAMESPACE::copy_common_runtime_args(*value_true_tensor.value().buffer(), args);
            CMAKE_UNIQUE_NAMESPACE::copy_common_runtime_args(*value_false_tensor.value().buffer(), args);
        }

        // Update common runtime args for writer kernel
        args = GetCommonRuntimeArgs(program, writer_kernel_id).data();
        CMAKE_UNIQUE_NAMESPACE::copy_common_runtime_args(*output_buffer, args);
    }

    auto update_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
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
        program,
        reader_kernel_id,
        writer_kernel_id,
        cached_program.shared_variables.compute_kernel_id,
        operation_attributes,
        tensor_args,
        output,
        broadcast_type,
        update_args);
}

}  // namespace ttnn::operations::ternary
