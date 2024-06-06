// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_op.hpp"
#include "binary_program_factory.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"



namespace ttnn::operations::binary {

namespace utils {
using namespace tt::tt_metal;

std::map<string, string> get_defines(
    BinaryOpType op_type, const std::optional<DataType> input_dtype, const std::optional<DataType> output_dtype, const std::optional<std::vector<tt::tt_metal::UnaryWithParam>> fused_activations) {
    std::map<string, string> defines;
    string op_name = "sub_tiles";
    string op_binary_type = "EltwiseBinaryType::ELWSUB";
    string idst = "i";

    switch (op_type) {
        case BinaryOpType::ADD:
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            break;
        case BinaryOpType::SUB:
            op_name = "sub_tiles";
            op_binary_type = "EltwiseBinaryType::ELWSUB";
            break;
        case BinaryOpType::MUL:
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        case BinaryOpType::GT:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::LT:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LTZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::GTE:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GEZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::LTE:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LEZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::EQ:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EQZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::NE:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::SQUARED_DIFFERENCE:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::SQUARE, std::nullopt, "0", idst));
            break;
        case BinaryOpType::LOGICAL_AND:
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::BIAS_GELU:
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GELU, std::vector<float>{0}, "0", idst));
            break;
        case BinaryOpType::LOGADDEXP:
            // PRE_IN0_0 ===> Applies prescaling for first input
            // PRE_IN1_0 ====> Applies prescaling for second input
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN0_0"));
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LOG, std::nullopt, "0", idst));
            break;
        case BinaryOpType::DIV_FAST:
            // Divide by a non-zero tensor
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::RECIP, std::nullopt, "PRE_IN1_0"));
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        case BinaryOpType::LOGICAL_OR:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN0_0"));
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::NEZ, std::nullopt, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::GTZ, std::nullopt, "0", idst));
            break;
        case BinaryOpType::LDEXP:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0"));
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        case BinaryOpType::LOGADDEXP2:
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN0_0"));
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::EXP2, std::nullopt, "PRE_IN1_0"));
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            defines.merge(eltwise_unary_op_utils::get_defines(UnaryOpType::LOG2, std::nullopt, "0", idst));
            break;
        default: TT_ASSERT(false && "Undefined op type");
    }

    if(input_dtype.has_value() && output_dtype.has_value() &&
        ((input_dtype.value() == DataType::BFLOAT16 && output_dtype.value() == DataType::UINT32) ||
        (input_dtype.value() == DataType::BFLOAT16 && output_dtype.value() == DataType::UINT16) ||
        (input_dtype.value() == DataType::BFLOAT16 && output_dtype.value() == DataType::INT32) ||
        (input_dtype.value() == DataType::UINT16 && output_dtype.value() == DataType::BFLOAT16) ||
        (input_dtype.value() == DataType::INT32 && output_dtype.value() == DataType::BFLOAT16))){
        TT_ASSERT(defines.count("SFPU_OP_CHAIN_0") == 0 && "SFPU_OP_CHAIN_0 already defined");

        auto in_dataformat =  std::to_string((uint32_t)datatype_to_dataformat_converter(input_dtype.value()));
        auto out_dataformat = std::to_string((uint32_t)datatype_to_dataformat_converter(output_dtype.value()));
        defines.insert({"SFPU_OP_CHAIN_0",
                        fmt::format("typecast_tile_init(); typecast_tile<{0}u, {1}u>(i);", in_dataformat, out_dataformat)});
        defines.insert({"SFPU_OP_TYPECAST_INCLUDE", "1"});
    }

    defines["ELTWISE_OP"] = op_name.c_str();
    defines["ELTWISE_OP_TYPE"] = op_binary_type.c_str();
    if (fused_activations.has_value()) {
        if (op_type == BinaryOpType::ADD and fused_activations.value().size() == 1 and
            fused_activations.value().at(0).op_type == UnaryOpType::RELU) {
            defines["PACK_RELU"] = "1";
        } else {
            defines.merge(eltwise_unary_op_utils::get_block_defines(fused_activations.value(), "0", idst));
        }
    }

    return defines;
}

}  // namespace utils


enum class BinaryProgramType {
    ElementWiseMultiCore,
    BroadcastWidthMultiCore,
    BroadcastHeightMultiCore,
    BroadcastHeightAndWidthMultiCore,
};

inline BinaryProgramType get_program_type(const Binary& operation, const std::vector<Tensor>& input_tensors) {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    const auto& input_shape_a = input_tensor_a.get_shape();
    const auto& input_shape_b = input_tensor_b.get_shape();

    auto batch_size_0_a = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
    auto batch_size_1_a = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto batch_size_0_b = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
    auto batch_size_1_b = input_shape_b.rank() >= 3 ? input_shape_b[-3] : 1;
    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    /*
    fmt::print("input_shape_a: {}, input_shape_b: {}\n", input_shape_a, input_shape_b);
    fmt::print(
        "batch_size_0_a: {}, batch_size_1_a: {}, height_a: {}, width_a: {}\n",
        batch_size_0_a,
        batch_size_1_a,
        height_a,
        width_a);
    fmt::print(
        "batch_size_0_b: {}, batch_size_1_b: {}, height_b: {}, width_b: {}\n",
        batch_size_0_b,
        batch_size_1_b,
        height_b,
        width_b);
    */

    if (batch_size_0_a == batch_size_0_b and batch_size_1_a == batch_size_1_b and height_a == height_b and
        width_a == width_b) {
        return BinaryProgramType::ElementWiseMultiCore;
    } else if (height_b == 1 or width_b == 1) {
        if (operation.dtype != input_tensor_a.get_dtype()) {
            TT_THROW("ttnn::operations::binary::Binary: cannot change dtype when broadcasting");
        }
        if (height_b == 1 and width_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastHeightAndWidthMultiCore\n");
            return BinaryProgramType::BroadcastHeightAndWidthMultiCore;
        } else if (height_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastHeightMultiCore\n");
            return BinaryProgramType::BroadcastHeightMultiCore;
        } else if (width_b == 1) {
            // fmt::print("BinaryProgramType::BroadcastWidthMultiCore\n");
            return BinaryProgramType::BroadcastWidthMultiCore;
        }
    }
    TT_THROW("ttnn::operations::binary::Binary: unsupported broadcast");
}

void Binary::validate_with_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    auto program_type = get_program_type(*this, input_tensors);

    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    const auto& input_shape_a = input_tensor_a.get_shape();
    const auto& input_shape_b = input_tensor_b.get_shape();

    auto batch_size_0_a = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
    auto batch_size_1_a = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    auto batch_size_0_b = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
    auto batch_size_1_b = input_shape_b.rank() >= 3 ? input_shape_b[-3] : 1;
    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    // Input shape b must be the same as or broadcastable to input shape a
    if (batch_size_0_a != batch_size_0_b) {
        TT_ASSERT(
            batch_size_0_a > batch_size_0_b and batch_size_0_b == 1,
            "ttnn::operations::binary::Binary: batch size mismatch");
    }
    if (batch_size_1_a != batch_size_1_b) {
        TT_ASSERT(
            batch_size_1_a > batch_size_1_b and batch_size_1_b == 1,
            "ttnn::operations::binary::Binary: batch size mismatch");
    }
    if (height_a != height_b) {
        TT_ASSERT(height_a > height_b and height_b == 1, "ttnn::operations::binary::Binary: height mismatch");
    }
    if (width_a != width_b) {
        TT_ASSERT(width_a > width_b and width_b == 1, "ttnn::operations::binary::Binary: width mismatch");
    }

    TT_FATAL(
        input_tensor_a.device() == input_tensor_b.device(),
        "Operands to eltwise binary need to be on the same device!");
    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to eltwise binary must be tilized");
    if (this->in_place) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == this->memory_config.memory_layout);
        TT_FATAL(input_tensor_a.memory_config().buffer_type == this->memory_config.buffer_type);
        TT_FATAL(input_tensor_a.get_dtype() == this->dtype);
    }
    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            // If we aren't height sharded, we require all sharding schemes to match until we add blocked
            // reader/writers for width and block sharding
            TT_FATAL((input_tensor_b.memory_config().is_sharded()));
            TT_FATAL(input_tensor_a.shard_spec().value().grid.ranges().size() == 1);
        }
        if (input_tensor_b.memory_config().is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == input_tensor_b.memory_config().memory_layout);
            TT_FATAL(input_tensor_a.shard_spec().value() == input_tensor_b.shard_spec().value());
        }
        if (this->memory_config.is_sharded()) {
            TT_FATAL(input_tensor_a.memory_config().memory_layout == this->memory_config.memory_layout);
        } else {
            TT_FATAL(this->memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else if (input_tensor_b.memory_config().is_sharded()) {
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (this->memory_config.is_sharded()) {
            TT_FATAL(input_tensor_b.memory_config().memory_layout == this->memory_config.memory_layout);
        } else {
            TT_FATAL(this->memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (this->memory_config.is_sharded()) {
            TT_FATAL(this->memory_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED);
            uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
            auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
            uint32_t num_cores = core_grid.x * core_grid.y;
            TT_FATAL(num_blocks < num_cores or num_blocks % num_cores == 0);

        } else {
            TT_FATAL(this->memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        }
    }

    if (program_type != BinaryProgramType::ElementWiseMultiCore) {
        TT_FATAL(not this->activations.has_value());
    }

    if (!output_tensors.empty()) {
        TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");

        if(output_tensors.at(0).has_value()) {
            TT_FATAL(!this->in_place, "Operation is configured as in_place. First input is used as output. Provided output tensor is ignored");
        }
    }
}

std::vector<tt::tt_metal::Shape> Binary::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    if (input_tensor_a.get_shape().rank() >= input_tensor_b.get_shape().rank()) {
        return {input_tensor_a.get_legacy_shape()};
    }
    return {input_tensor_b.get_legacy_shape()};
}

std::vector<Tensor> Binary::create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    if (this->in_place) {
        return {input_tensor_a};
    } else {
        if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
            return {output_tensors.at(0).value()};
        }

        auto program_type = get_program_type(*this, input_tensors);

        if (program_type == BinaryProgramType::ElementWiseMultiCore) {
            if (this->memory_config.is_sharded()) {
                ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
                if (input_tensor_a.memory_config().is_sharded()) {
                    shard_spec = input_tensor_a.shard_spec().value();
                } else if (input_tensor_b.memory_config().is_sharded()) {
                    shard_spec = input_tensor_b.shard_spec().value();
                } else {
                    uint32_t num_blocks = input_tensor_a.volume() / input_tensor_a.get_legacy_shape()[-1] / TILE_HEIGHT;
                    auto core_grid = input_tensor_a.device()->compute_with_storage_grid_size();
                    uint32_t num_grid_cores = core_grid.x * core_grid.y;
                    uint32_t target_num_cores = num_blocks < num_grid_cores ? num_blocks : num_grid_cores;
                    shard_spec.grid = num_cores_to_corerange_set(target_num_cores, core_grid, true);
                    shard_spec.shape = {
                        num_blocks / target_num_cores * TILE_HEIGHT, input_tensor_a.get_legacy_shape()[-1]};
                    shard_spec.orientation = ShardOrientation::ROW_MAJOR;
                }
                auto memory_config = this->memory_config;
                memory_config.shard_spec = shard_spec;
                return {create_device_tensor(
                    this->compute_output_shapes(input_tensors).at(0),
                    this->dtype,
                    Layout::TILE,
                    input_tensor_a.device(),
                    memory_config)};
            }
        } else {
            if (this->memory_config.is_sharded()) {
                ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
                if (input_tensor_a.memory_config().is_sharded()) {
                    // Derive output shard_spec based on input
                    shard_spec = input_tensor_a.shard_spec().value();
                }
                auto memory_config = this->memory_config;
                memory_config.shard_spec = shard_spec;
                return {create_device_tensor(
                    this->compute_output_shapes(input_tensors).at(0),
                    this->dtype,
                    Layout::TILE,
                    input_tensor_a.device(),
                    memory_config)};
            }
        }
        return operation::generic_create_output_tensors(
            *this, input_tensors, this->dtype, Layout::TILE, this->memory_config);
    }
}

const std::optional<tt::tt_metal::BcastOpMath> binary_op_type_to_bcast_op_math(const BinaryOpType binary_op_type) {
    switch (binary_op_type) {
        case BinaryOpType::ADD: return tt::tt_metal::BcastOpMath::ADD;
        case BinaryOpType::SUB: return tt::tt_metal::BcastOpMath::SUB;
        case BinaryOpType::MUL: return tt::tt_metal::BcastOpMath::MUL;
        default: return std::nullopt;
    }
}

operation::ProgramWithCallbacks Binary::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& output_tensor = output_tensors.at(0);

    std::vector<UnaryWithParam> activations;
    if (this->program_config.activations.has_value()) {
        activations = this->program_config.activations.value();
    }

    auto program_type = get_program_type(*this, input_tensors);
    auto bcast_op_math = binary_op_type_to_bcast_op_math(this->binary_op_type);
    if (bcast_op_math.has_value()) {
        switch (program_type) {
            case BinaryProgramType::ElementWiseMultiCore:
                return eltwise_binary_multi_core(
                    input_tensor_a, input_tensor_b, output_tensor, this->binary_op_type, activations);
            case BinaryProgramType::BroadcastHeightAndWidthMultiCore:
                return bcast_multi_core_hw(
                    input_tensor_a, input_tensor_b, output_tensor, bcast_op_math.value(), false /* in-place */);
            case BinaryProgramType::BroadcastHeightMultiCore:
                return bcast_multi_core_h(input_tensor_a, input_tensor_b, output_tensor, bcast_op_math.value());
            case BinaryProgramType::BroadcastWidthMultiCore:
                return bcast_multi_core_w(input_tensor_a, input_tensor_b, output_tensor, bcast_op_math.value());
            default: TT_THROW("Invalid program type");
        }
    } else {
        switch (program_type) {
            case BinaryProgramType::ElementWiseMultiCore:
                return eltwise_binary_multi_core(
                    input_tensor_a, input_tensor_b, output_tensor, this->binary_op_type, activations);
            default: TT_THROW("Invalid program type");
        }
    }
}

const operation::Hash Binary::compute_program_hash(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto program_type = get_program_type(*this, input_tensors);
    operation::Hash hash = operation::hash_operation<Binary>(
        this->program_config,
        program_type,
        input_tensor_a.dtype(),
        std::get<DeviceStorage>(input_tensor_a.storage()).memory_config(),
        input_tensor_b.dtype(),
        std::get<DeviceStorage>(input_tensor_b.storage()).memory_config());
    return hash;
}

operation::OpPerformanceModel Binary::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    // GS specific parameters
    // 80 B/cycle unpacker BW shared
    // 128 datums per cycle math, but unpacker cant keep up
    constexpr int num_cores = 9 * 12;

    int total_bytes = 0;
    for (const auto& t : input_tensors) {
        total_bytes += t.volume() * t.element_size();
    }
    int ideal_eltwise_cycles = total_bytes / 80 / num_cores;

    operation::OpPerformanceModel result(input_tensors, output_tensors, ideal_eltwise_cycles);
#if 0
        tt::log_info(tt::LogOp, "Binary PerfModel:");
        tt::log_info(tt::LogOp, "\t Data (Bytes): {}", total_bytes);
        tt::log_info(tt::LogOp, "\t ideal_eltwise_cycles: {}", ideal_eltwise_cycles);
#endif
    return result;
}

}  // namespace ttnn::operations::binary
