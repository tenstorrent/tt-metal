// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_ng_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include <cstdint>
#include <bit>
#include <fmt/core.h>

namespace ttnn::operations::unary_ng {

using unary::EltwiseUnaryWithParam;
using unary::UnaryOpType;

namespace {

// Macro define name for kernel includes. Only migrated ops are defined here; add others when migrating.
std::string get_macro_definition(UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::ABS:
        case UnaryOpType::ABS_INT32: return "SFPU_OP_COMPUTE_KERNEL_API_INCLUDE";
        default:
            TT_FATAL(
                false,
                "UnaryNg: op type {} not yet migrated; add define in unary_ng_op_utils",
                static_cast<int>(op_type));
    }
}

// Init and func strings for SFPU op chain. Only migrated ops (e.g. ABS) are implemented here.
std::pair<std::string, std::string> get_op_init_and_func(
    UnaryOpType op_type, const std::string& idst, std::optional<DataType> /*input_dtype*/) {
    switch (op_type) {
        case UnaryOpType::ABS: return {"abs_tile_init();", fmt::format("abs_tile({});", idst)};
        case UnaryOpType::ABS_INT32: return {"abs_tile_init();", fmt::format("abs_tile_int32({});", idst)};
        default:
            TT_FATAL(
                false,
                "UnaryNg: op type {} not yet migrated; add init/func in unary_ng_op_utils",
                static_cast<int>(op_type));
    }
}

std::map<std::string, std::string> get_defines_impl(
    UnaryOpType op_type,
    const std::string& idst,
    const std::string& init_def,
    const std::string& func_def,
    std::optional<DataType> input_dtype) {
    auto [init_str, func_str] = get_op_init_and_func(op_type, idst, input_dtype);
    std::map<std::string, std::string> defines = {
        {init_def, init_str},
        {func_def, func_str},
    };
    defines[get_macro_definition(op_type)] = "1";
    return defines;
}

}  // namespace

std::string_view get_compute_kernel_path(UnaryOpType op_type, std::optional<DataType> input_dtype) {
    switch (op_type) {
        case UnaryOpType::LGAMMA: return "lgamma_kernel.cpp";
        case UnaryOpType::MISH: return "mish_kernel.cpp";
        case UnaryOpType::TANHSHRINK:
            if (input_dtype.has_value() && input_dtype.value() == DataType::FLOAT32) {
                return "tanhshrink_sfpu_kernel.cpp";
            }
            return "tanhshrink_kernel.cpp";
        case UnaryOpType::IDENTITY: return "eltwise_identity_kernel.cpp";
        case UnaryOpType::WHERE_TSS: return "where_tss_kernel.cpp";
        case UnaryOpType::LOGIT: return "logit_kernel.cpp";
        case UnaryOpType::HARDSHRINK:
            if (input_dtype.has_value() && input_dtype.value() == DataType::FLOAT32) {
                return "hardshrink_kernel_sfpu.cpp";
            }
            return "hardshrink_kernel.cpp";
        case UnaryOpType::HARDSWISH:
            if (input_dtype.has_value() && input_dtype.value() == DataType::FLOAT32) {
                return "hardswish_kernel_sfpu.cpp";
            }
            return "hardswish_kernel.cpp";
        case UnaryOpType::LOGSIGMOID: return "logsigmoid_kernel.cpp";
        default: return "eltwise_sfpu.cpp";
    }
}

uint32_t pack_scalar_runtime_arg_impl(float param, DataType dtype) {
    if (dtype == DataType::UINT32 || dtype == DataType::INT32) {
        return std::bit_cast<std::uint32_t>(static_cast<std::int32_t>(param));
    }
    return std::bit_cast<std::uint32_t>(param);
}

uint32_t pack_scalar_runtime_arg_impl(std::uint32_t param, DataType dtype) {
    if (dtype == DataType::INT32 || dtype == DataType::UINT32) {
        return param;
    }
    return std::bit_cast<std::uint32_t>(static_cast<float>(param));
}

uint32_t pack_scalar_runtime_arg_impl(std::int32_t param, DataType dtype) {
    if (dtype == DataType::INT32 || dtype == DataType::UINT32) {
        return std::bit_cast<std::uint32_t>(param);
    }
    return std::bit_cast<std::uint32_t>(static_cast<float>(param));
}

uint32_t pack_scalar_runtime_arg(const EltwiseUnaryWithParam& op, size_t index, DataType dtype) {
    return std::visit(
        [index, dtype](const auto& op_specialization) -> uint32_t {
            if (auto param = op_specialization.get_param_if(index)) {
                return pack_scalar_runtime_arg_impl(*param, dtype);
            }
            TT_THROW("Unsupported parameter type for index {}", index);
        },
        op.base);
}

bool get_op_approx_mode(UnaryOpType) { return false; }

std::map<std::string, std::string> get_block_defines(
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const std::string& block_id,
    const std::string& idst,
    std::optional<DataType> input_dtype) {
    std::map<std::string, std::string> block_defines;
    std::string block_define;
    for (uint32_t i = 0; i < op_chain.size(); i++) {
        std::string init_def = fmt::format("SFPU_OP_CHAIN_{}_INIT_{}", block_id, i);
        std::string func_def = fmt::format("SFPU_OP_CHAIN_{}_FUNC_{}", block_id, i);
        block_define += init_def + " " + func_def + " ";
        block_defines.merge(get_defines_impl(op_chain[i].type(), idst, init_def, func_def, input_dtype));
    }
    block_defines[fmt::format("SFPU_OP_CHAIN_{}", block_id)] = block_define;
    return block_defines;
}

}  // namespace ttnn::operations::unary_ng
