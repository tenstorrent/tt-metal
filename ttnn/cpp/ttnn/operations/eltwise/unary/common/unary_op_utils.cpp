// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_op_utils.hpp"

namespace ttnn::operations::unary::utils {

namespace {
union Converter {
   public:
    float f;
    uint32_t u;

    Converter(float f_) : f(f_){};

    static std::string to_hex(float f_) {
        Converter obj(f_);
        std::stringstream ss;
        ss << "0x" << std::hex << obj.u;
        return ss.str();
    }
};


// update split eltwise ops include macros
void update_macro_defines(UnaryOpType op_type, std::map<std::string, std::string>& defines) {
    switch (op_type) {
        case UnaryOpType::EXP: defines["SFPU_OP_EXP_INCLUDE"] = "1"; break;
        case UnaryOpType::GELU: defines["SFPU_OP_GELU_INCLUDE"] = "1"; break;
        case UnaryOpType::RECIP: defines["SFPU_OP_RECIP_INCLUDE"] = "1"; break;
        case UnaryOpType::SQRT: defines["SFPU_OP_SQRT_INCLUDE"] = "1"; break;
        case UnaryOpType::ERFINV: defines["SFPU_OP_ERFINV_INCLUDE"] = "1"; break;
        case UnaryOpType::ERFC:
        case UnaryOpType::ERF: defines["SFPU_OP_ERF_ERFC_INCLUDE"] = "1"; break;
        case UnaryOpType::ELU: defines["SFPU_OP_ELU_INCLUDE"] = "1"; break;
        case UnaryOpType::RELU:
        case UnaryOpType::RELU6:
        case UnaryOpType::RELU_MAX:
        case UnaryOpType::RELU_MIN:
        case UnaryOpType::LEAKY_RELU: defines["SFPU_OP_RELU_FAMILY_INCLUDE"] = "1"; break;
        case UnaryOpType::ADD_UNARY_SFPU:
        case UnaryOpType::SUB_UNARY_SFPU:
        case UnaryOpType::MUL_UNARY_SFPU:
        case UnaryOpType::DIV_UNARY_SFPU: defines["SFPU_OP_BINOP_WITH_SCALAR_INCLUDE"] = "1"; break;
        case UnaryOpType::IDENTITY:
        case UnaryOpType::IDENTITY_UINT32: defines["SFPU_OP_IDENTITY_INCLUDE"] = "1"; break;
        case UnaryOpType::RDIV: break;
        case UnaryOpType::RSUB: defines["SFPU_OP_REVERSE_FAMILY_INCLUDE"] = "1";
        case UnaryOpType::ISINF:
        case UnaryOpType::ISNAN:
        case UnaryOpType::ISNEGINF:
        case UnaryOpType::ISPOSINF:
        case UnaryOpType::ISFINITE: defines["SFPU_OP_ISINF_ISNAN_INCLUDE"] = "1"; break;
        case UnaryOpType::LOGICAL_NOT_UNARY: defines["SFPU_OP_LOGICAL_NOT_NOTI_INCLUDE"] = "1"; break;
        case UnaryOpType::I0: defines["SFPU_OP_I0_INCLUDE"] = "1"; break;
        case UnaryOpType::COS:
        case UnaryOpType::SIN:
        case UnaryOpType::TAN: defines["SFPU_OP_TRIG_FAMILY_INCLUDE"] = "1"; break;
        case UnaryOpType::NEG: defines["SFPU_OP_NEG_INCLUDE"] = "1"; break;
        case UnaryOpType::SOFTPLUS: defines["SFPU_OP_SOFTPLUS_INCLUDE"] = "1"; break;
        case UnaryOpType::TYPECAST: defines["SFPU_OP_TYPECAST_INCLUDE"] = "1"; break;
        case UnaryOpType::BITWISE_XOR: defines["SFPU_OP_BITWISE_XOR_INCLUDE"] = "1"; break;
        case UnaryOpType::BITWISE_NOT: defines["SFPU_OP_BITWISE_NOT_INCLUDE"] = "1"; break;
        case UnaryOpType::BITWISE_AND: defines["SFPU_OP_BITWISE_AND_INCLUDE"] = "1"; break;
        case UnaryOpType::BITWISE_OR: defines["SFPU_OP_BITWISE_OR_INCLUDE"] = "1"; break;
        case UnaryOpType::RIGHT_SHIFT: defines["SFPU_OP_RIGHT_SHIFT_INCLUDE"] = "1"; break;
        case UnaryOpType::FLOOR: defines["SFPU_OP_FLOOR_INCLUDE"] = "1"; break;
        case UnaryOpType::CEIL: defines["SFPU_OP_CEIL_INCLUDE"] = "1"; break;
        case UnaryOpType::LEFT_SHIFT: defines["SFPU_OP_LEFT_SHIFT_INCLUDE"] = "1"; break;
        case UnaryOpType::REMAINDER: defines["SFPU_OP_REMAINDER_INCLUDE"] = "1"; break;
        case UnaryOpType::FMOD: defines["SFPU_OP_FMOD_INCLUDE"] = "1"; break;
        case UnaryOpType::DROPOUT: defines["SFPU_OP_DROPOUT_INCLUDE"] = "1"; break;
        case UnaryOpType::FILL: defines["SFPU_OP_FILL_INCLUDE"] = "1"; break;
        default: defines["SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"] = "1"; break;
    };
}

std::pair<std::string, std::string> get_op_init_and_func_parameterized(
    UnaryOpType op_type, const std::vector<float>& params, const std::string& idst) {
    std::pair<std::string, std::string> op_init_and_name;
    TT_FATAL(is_parametrized_type(op_type), "operator should support at least one parameter", "Error");
    float param0 = params[0];
    switch (op_type) {
        case UnaryOpType::FILL:
            op_init_and_name = {
                "fill_tile_init();", fmt::format("fill_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::RELU_MAX:
            op_init_and_name = {
                "relu_max_tile_init();", fmt::format("relu_max_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::RELU_MIN:
            op_init_and_name = {
                "relu_min_tile_init();", fmt::format("relu_min_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::POWER:
            op_init_and_name = {
                "power_tile_init();", fmt::format("power_tile({}, {}u);", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::LEAKY_RELU:
            op_init_and_name = {
                "leaky_relu_tile_init();", fmt::format("leaky_relu_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::ELU:
            op_init_and_name = {"elu_tile_init();", fmt::format("elu_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::GELU:
            op_init_and_name = {
                fmt::format("gelu_tile_init<{}u>();", std::to_string((uint32_t)param0)),
                fmt::format("gelu_tile<{1}u>({0});", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::RSQRT:
            op_init_and_name = {
                fmt::format("rsqrt_tile_init<{}u>();", std::to_string((uint32_t)param0)),
                fmt::format("rsqrt_tile<{1}u>({0});", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::HEAVISIDE:
            op_init_and_name = {
                "heaviside_tile_init();", fmt::format("heaviside_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::BITWISE_XOR:
            op_init_and_name = {
                "bitwise_xor_tile_init();", fmt::format("bitwise_xor_tile({}, {}u);", idst, std::to_string((uint)param0))};
            break;
        case UnaryOpType::BITWISE_NOT:
            op_init_and_name = {
                "bitwise_not_tile_init();", fmt::format("bitwise_not_tile({}, {}u);", idst, std::to_string((uint)param0))};
            break;
        case UnaryOpType::BITWISE_AND:
            op_init_and_name = {
                "bitwise_and_tile_init();", fmt::format("bitwise_and_tile({}, {}u);", idst, std::to_string((uint)param0))};
            break;
        case UnaryOpType::BITWISE_OR:
            op_init_and_name = {
                "bitwise_or_tile_init();", fmt::format("bitwise_or_tile({}, {}u);", idst, std::to_string((uint)param0))};
            break;
        case UnaryOpType::RIGHT_SHIFT:
            op_init_and_name = {
                "right_shift_tile_init();",
                fmt::format("right_shift_tile({}, {}u);", idst, std::to_string((uint)param0))};
            break;
        case UnaryOpType::LEFT_SHIFT:
            op_init_and_name = {
                "left_shift_tile_init();",
                fmt::format("left_shift_tile({}, {}u);", idst, std::to_string((uint)param0))};
            break;
        case UnaryOpType::REMAINDER:
            op_init_and_name = {
                "remainder_tile_init();",
                fmt::format(
                    "remainder_tile({}, {}u, {}u);",
                    idst,
                    Converter::to_hex(param0),
                    Converter::to_hex(1.0f / param0))};
            break;
        case UnaryOpType::FMOD:
            op_init_and_name = {
                "fmod_tile_init();",
                fmt::format("fmod_tile({}, {}u, {}u);", idst, Converter::to_hex(param0), Converter::to_hex(1.0f/param0))};
            break;
        case UnaryOpType::EXP:
            op_init_and_name = {
                fmt::format("exp_tile_init<{}u>();", std::to_string((uint32_t)param0)),
                fmt::format("exp_tile<{1}u>({0});", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::ERF:
            op_init_and_name = {
                fmt::format("erf_tile_init<{}u>();", std::to_string((uint32_t)param0)),
                fmt::format("erf_tile<{1}u>({0});", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::ERFC:
            op_init_and_name = {
                fmt::format("erfc_tile_init<{}u>();", std::to_string((uint32_t)param0)),
                fmt::format("erfc_tile<{1}u>({0});", idst, std::to_string((uint32_t)param0))};
            break;
        case UnaryOpType::RDIV: op_init_and_name = {}; break;
        case UnaryOpType::RSUB:
            op_init_and_name = {
                "rsub_tile_init();", fmt::format("rsub_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::SUB_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("sub_unary_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::ADD_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("add_unary_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::MUL_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("mul_unary_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::DIV_UNARY_SFPU:
            op_init_and_name = {
                "binop_with_scalar_tile_init();",
                fmt::format("div_unary_tile({}, {}u);", idst, Converter::to_hex(1.0f / param0))};
            break;
        case UnaryOpType::UNARY_NE:
            op_init_and_name = {
                "unary_ne_tile_init();", fmt::format("unary_ne_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::UNARY_GT:
            op_init_and_name = {
                "unary_gt_tile_init();", fmt::format("unary_gt_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::UNARY_LT:
            op_init_and_name = {
                "unary_lt_tile_init();", fmt::format("unary_lt_tile({}, {}u);", idst, Converter::to_hex(param0))};
            break;
        case UnaryOpType::SOFTPLUS: {
            TT_ASSERT(params.size() == 2, "Expected softplus to take 2 parameters");
            float param1 = params[1];
            op_init_and_name = {
                "softplus_tile_init();",
                fmt::format(
                    "softplus_tile({}, {}u, {}u, {}u);",
                    idst,
                    Converter::to_hex(param0),
                    Converter::to_hex(1.0f / param0),  // Pass reciprocal to avoid doing it on device
                    Converter::to_hex(param1))};
            break;
        }
        case UnaryOpType::TYPECAST:
            TT_ASSERT(params.size() == 2, "Expected eltwise_typecast to take 2 parameters");
            op_init_and_name = {
                "typecast_tile_init();",
                fmt::format(
                    "typecast_tile<{1}u, {2}u>({0});",
                    idst,
                    std::to_string((uint32_t)datatype_to_dataformat_converter((DataType)params[0])),
                    std::to_string((uint32_t)datatype_to_dataformat_converter((DataType)params[1])))};
            break;
        case UnaryOpType::DROPOUT: {
            TT_ASSERT(params.size() == 3, "Expected Dropout to take 3 parameters: seed, probability and scale factor");
            float prob = params[1];
            float scale = params[2];
            uint32_t uprob = static_cast<uint32_t>((double)INT_MAX * prob); // kernel requirement, please read it in the kernel comments
            op_init_and_name = {
                // DO NOT ADD seed support till runtime args support will be added.
                // Current approach doesn't work with dropout unary op because we will compile a new file for each new seed
                "",//fmt::format("dropout_tile_init({}u);", (uint32_t)param0),

                fmt::format("dropout_tile({}, {}u, {}u);", idst, uprob, Converter::to_hex(scale))
            };
            break;
        }
        default: TT_ASSERT(false && "unexpected parameterized type");
    };
    return op_init_and_name;
}

std::pair<string, string> get_op_init_and_func_default(UnaryOpType op_type, std::string idst) {
    std::pair<std::string, std::string> op_init_and_name;
    switch (op_type) {
        case UnaryOpType::RECIP: op_init_and_name = {"recip_tile_init();", fmt::format("recip_tile({});", idst)}; break;
        case UnaryOpType::RELU: op_init_and_name = {"relu_tile_init();", fmt::format("relu_tile({});", idst)}; break;
        case UnaryOpType::SQRT: op_init_and_name = {"sqrt_tile_init();", fmt::format("sqrt_tile({});", idst)}; break;
        case UnaryOpType::SIGMOID:
            op_init_and_name = {"sigmoid_tile_init();", fmt::format("sigmoid_tile({});", idst)};
            break;
        case UnaryOpType::LOG: op_init_and_name = {"log_tile_init();", fmt::format("log_tile({});", idst)}; break;
        case UnaryOpType::TANH: op_init_and_name = {"tanh_tile_init();", fmt::format("tanh_tile({});", idst)}; break;
        case UnaryOpType::SIGNBIT:
            op_init_and_name = {"signbit_tile_init();", fmt::format("signbit_tile({});", idst)};
            break;
        case UnaryOpType::FLOOR: op_init_and_name = {"floor_tile_init();", fmt::format("floor_tile({});", idst)}; break;
        case UnaryOpType::CEIL: op_init_and_name = {"ceil_tile_init();", fmt::format("ceil_tile({});", idst)}; break;
        case UnaryOpType::SIN: op_init_and_name = {"sin_tile_init();", fmt::format("sin_tile({});", idst)}; break;
        case UnaryOpType::COS: op_init_and_name = {"cos_tile_init();", fmt::format("cos_tile({});", idst)}; break;
        case UnaryOpType::ISFINITE:
            op_init_and_name = {"isfinite_tile_init();", fmt::format("isfinite_tile({});", idst)};
            break;
        case UnaryOpType::ISINF: op_init_and_name = {"isinf_tile_init();", fmt::format("isinf_tile({});", idst)}; break;
        case UnaryOpType::ISPOSINF:
            op_init_and_name = {"isposinf_tile_init();", fmt::format("isposinf_tile({});", idst)};
            break;
        case UnaryOpType::ISNEGINF:
            op_init_and_name = {"isneginf_tile_init();", fmt::format("isneginf_tile({});", idst)};
            break;
        case UnaryOpType::ISNAN: op_init_and_name = {"isnan_tile_init();", fmt::format("isnan_tile({});", idst)}; break;
        case UnaryOpType::LOGICAL_NOT_UNARY:
            op_init_and_name = {"logical_not_unary_tile_init();", fmt::format("logical_not_unary_tile({});", idst)};
            break;
        case UnaryOpType::I0: op_init_and_name = {"i0_tile_init();", fmt::format("i0_tile({});", idst)}; break;
        case UnaryOpType::ERFINV:
            op_init_and_name = {"erfinv_tile_init();", fmt::format("erfinv_tile({});", idst)};
            break;
        case UnaryOpType::LOG10:
            // log10[x] = log[x]/log[10] = log[x]*0.4342944819032518; FP32@U32 0x3ede5bd9; FP16@U16 0x36f3;
            op_init_and_name = {"log_with_base_tile_init();", fmt::format("log_with_base_tile({}, 0x36f3u);", idst)};
            break;
            break;
        case UnaryOpType::LOG2:  // log2[x] = log[x]*1.4426950408889634f; FP32@U32 0x3fb8aa3b; FP16@U16 0x3dc5;
            op_init_and_name = {"log_with_base_tile_init();", fmt::format("log_with_base_tile({}, 0x3dc5u);", idst)};
            break;
            break;
        case UnaryOpType::ABS: op_init_and_name = {"abs_tile_init();", fmt::format("abs_tile({});", idst)}; break;
        case UnaryOpType::SIGN: op_init_and_name = {"sign_tile_init();", fmt::format("sign_tile({});", idst)}; break;
        case UnaryOpType::SQUARE:
            op_init_and_name = {"square_tile_init();", fmt::format("square_tile({});", idst)};
            break;
        case UnaryOpType::TILED_PROD:
            op_init_and_name = {"tiled_prod_tile_init();", fmt::format("tiled_prod_tile({});", idst)};
            break;
        case UnaryOpType::EQZ: op_init_and_name = {"eqz_tile_init();", fmt::format("eqz_tile({});", idst)}; break;
        case UnaryOpType::NEZ: op_init_and_name = {"nez_tile_init();", fmt::format("nez_tile({});", idst)}; break;
        case UnaryOpType::LTZ: op_init_and_name = {"ltz_tile_init();", fmt::format("ltz_tile({});", idst)}; break;
        case UnaryOpType::GTZ: op_init_and_name = {"gtz_tile_init();", fmt::format("gtz_tile({});", idst)}; break;
        case UnaryOpType::LEZ: op_init_and_name = {"lez_tile_init();", fmt::format("lez_tile({});", idst)}; break;
        case UnaryOpType::GEZ: op_init_and_name = {"gez_tile_init();", fmt::format("gez_tile({});", idst)}; break;
        case UnaryOpType::EXP2: op_init_and_name = {"exp2_tile_init();", fmt::format("exp2_tile({});", idst)}; break;
        case UnaryOpType::EXPM1: op_init_and_name = {"expm1_tile_init();", fmt::format("expm1_tile({});", idst)}; break;
        case UnaryOpType::ASIN: op_init_and_name = {"asin_tile_init();", fmt::format("asin_tile({});", idst)}; break;
        case UnaryOpType::ACOS: op_init_and_name = {"acos_tile_init();", fmt::format("acos_tile({});", idst)}; break;
        case UnaryOpType::ATAN: op_init_and_name = {"atan_tile_init();", fmt::format("atan_tile({});", idst)}; break;
        case UnaryOpType::TAN: op_init_and_name = {"tan_tile_init();", fmt::format("tan_tile({});", idst)}; break;
        case UnaryOpType::SILU: op_init_and_name = {"silu_tile_init();", fmt::format("silu_tile({});", idst)}; break;
        case UnaryOpType::IDENTITY:
            op_init_and_name = {"identity_tile_init();", fmt::format("identity_tile({});", idst)};
            break;
        case UnaryOpType::IDENTITY_UINT32:
            op_init_and_name = {"identity_tile_init();", fmt::format("identity_tile_uint32({});", idst)};
            break;
        case UnaryOpType::RELU6:
            op_init_and_name = {"relu_max_tile_init();", fmt::format("relu_max_tile({}, 0x40c00000u);", idst)};
            break;
        case UnaryOpType::NEG:
            op_init_and_name = {"negative_tile_init();", fmt::format("negative_tile({});", idst)};
            break;
        default: TT_ASSERT(false && "Undefined non-parametrized op type");
    }
    return op_init_and_name;
}

std::map<string, string> get_defines_impl(
    UnaryOpType op_type,
    const std::vector<float>& params,
    std::string idst,
    std::string init_def,
    std::string func_def) {
    std::pair<string, string> op_init_and_name = get_op_init_and_func(op_type, params, idst);
    std::map<string, string> defines = {{init_def, op_init_and_name.first}, {func_def, op_init_and_name.second}};
    update_macro_defines(op_type, defines);
    return defines;
}
}

bool get_op_approx_mode(UnaryOpType op_type) {
    switch (op_type) {
        default: return false;
    }
}

UnaryWithParam string_to_unary_with_param(const std::string& name) {
    if (name == "relu")
        return UnaryWithParam(UnaryOpType::RELU);
    else if (name == "gelu")
        return UnaryWithParam(UnaryOpType::GELU, static_cast<float>(true));
    else if (name == "silu")
        return UnaryWithParam(UnaryOpType::SILU);
    else if (name == "sigmoid")
        return UnaryWithParam(UnaryOpType::SIGMOID);
    else if (name == "sqrt")
        return UnaryWithParam(UnaryOpType::SQRT);
    else if (name == "exp")
        return UnaryWithParam(UnaryOpType::EXP, static_cast<float>(true));
    else if (name == "recip")
        return UnaryWithParam(UnaryOpType::RECIP);
    else if (name == "log")
        return UnaryWithParam(UnaryOpType::LOG);
    else if (name == "tanh")
        return UnaryWithParam(UnaryOpType::TANH);
    else if (name == "log2")
        return UnaryWithParam(UnaryOpType::LOG2);
    else if (name == "log10")
        return UnaryWithParam(UnaryOpType::LOG10);
    else if (name == "sin")
        return UnaryWithParam(UnaryOpType::SIN);
    else if (name == "cos")
        return UnaryWithParam(UnaryOpType::COS);
    else if (name == "abs")
        return UnaryWithParam(UnaryOpType::ABS);
    else if (name == "sign")
        return UnaryWithParam(UnaryOpType::SIGN);
    else if (name == "square")
        return UnaryWithParam(UnaryOpType::SQUARE);
    else if (name == "softplus")
        return UnaryWithParam(UnaryOpType::SOFTPLUS);
    TT_THROW("Unknown unary op: {}", name);
}

std::map<string, string> get_defines(
    UnaryOpType op_type, const std::optional<std::vector<float>>& params, const std::string& id, const std::string& idst) {
    std::string init_def = fmt::format("SFPU_OP_INIT_{}", id);
    std::string func_def = fmt::format("SFPU_OP_FUNC_{}", id);
    return get_defines_impl(
        op_type, params.has_value() ? params.value() : std::vector<float>{}, idst, init_def, func_def);
}

std::pair<string, string> get_op_init_and_func(UnaryOpType op_type, const std::vector<float>& params, const std::string& idst) {
    return params.size() > 0 ? get_op_init_and_func_parameterized(op_type, params, idst)
                             : get_op_init_and_func_default(op_type, idst);
}

std::map<string, string> get_block_defines(
    const std::vector<UnaryWithParam>& op_chain, const std::string& block_id, const std::string& idst) {
    std::vector<std::pair<string, string>> op_init_and_name;
    std::map<string, string> block_defines;
    std::string block_define = "";
    for (uint32_t i = 0; i < op_chain.size(); i++) {
        std::string init_def = fmt::format("SFPU_OP_CHAIN_{}_INIT_{}", block_id, i);
        std::string func_def = fmt::format("SFPU_OP_CHAIN_{}_FUNC_{}", block_id, i);
        block_define += init_def + " " + func_def + " ";
        block_defines.merge(get_defines_impl(op_chain[i].op_type, op_chain[i].params, idst, init_def, func_def));
    }
    block_defines[fmt::format("SFPU_OP_CHAIN_{}", block_id)] = block_define;
    return block_defines;
}

}
