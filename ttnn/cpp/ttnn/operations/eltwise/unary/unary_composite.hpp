// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations {
namespace unary {

struct PowerOperation{
     static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        uint32_t exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        uint32_t exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        float exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        float exponent,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

struct RdivOperation {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        float value,
        const std::string& round_mode = "None",
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

#define DEFINE_UNARY_OPERATION(op_name) \
struct op_name##Operation { \
    static Tensor invoke(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt); \
};

#define DEFINE_UNARY_OPERATION_WITH_FLOAT(op_name) \
struct op_name##Operation { \
    static Tensor invoke( \
        const Tensor &input_tensor, \
        float param1, \
        const std::optional<MemoryConfig> &memory_config = std::nullopt); \
};

#define DEFINE_UNARY_OPERATION_WITH_2_FLOATS(op_name) \
struct op_name##Operation { \
    static Tensor invoke( \
        const Tensor &input_tensor, \
        float param1, \
        float param2, \
        const std::optional<MemoryConfig> &memory_config = std::nullopt); \
};

#define DEFINE_UNARY_OPERATION_WITH_INT(op_name) \
struct op_name##Operation { \
    static Tensor invoke( \
        const Tensor &input_tensor, \
        int32_t param1, \
        const std::optional<MemoryConfig> &memory_config = std::nullopt); \
};

DEFINE_UNARY_OPERATION(Acosh)
DEFINE_UNARY_OPERATION(Asinh)
DEFINE_UNARY_OPERATION(Atanh)
DEFINE_UNARY_OPERATION(Cbrt)
DEFINE_UNARY_OPERATION(Cosh)
DEFINE_UNARY_OPERATION(Deg2rad)
DEFINE_UNARY_OPERATION(Digamma)
DEFINE_UNARY_OPERATION(Frac)
DEFINE_UNARY_OPERATION(Lgamma)
DEFINE_UNARY_OPERATION(Log1p)
DEFINE_UNARY_OPERATION(LogicalNot)
DEFINE_UNARY_OPERATION(Mish)
DEFINE_UNARY_OPERATION(Multigammaln)
DEFINE_UNARY_OPERATION(NormalizeGlobal)
DEFINE_UNARY_OPERATION(NormalizeHw)
DEFINE_UNARY_OPERATION(Rad2deg)
DEFINE_UNARY_OPERATION(Sinh)
DEFINE_UNARY_OPERATION(Softsign)
DEFINE_UNARY_OPERATION(StdHw)
DEFINE_UNARY_OPERATION(Swish)
DEFINE_UNARY_OPERATION(Tanhshrink)
DEFINE_UNARY_OPERATION(Trunc)
DEFINE_UNARY_OPERATION(VarHw)

DEFINE_UNARY_OPERATION_WITH_INT(Geglu)
DEFINE_UNARY_OPERATION_WITH_INT(Glu)
DEFINE_UNARY_OPERATION_WITH_INT(Polygamma)
DEFINE_UNARY_OPERATION_WITH_INT(Reglu)
DEFINE_UNARY_OPERATION_WITH_INT(Round)
DEFINE_UNARY_OPERATION_WITH_INT(Swiglu)
DEFINE_UNARY_OPERATION_WITH_INT(Tril)
DEFINE_UNARY_OPERATION_WITH_INT(Triu)

DEFINE_UNARY_OPERATION_WITH_FLOAT(Celu)
DEFINE_UNARY_OPERATION_WITH_FLOAT(Hardshrink)
DEFINE_UNARY_OPERATION_WITH_FLOAT(Logit)
DEFINE_UNARY_OPERATION_WITH_FLOAT(Rpow)
DEFINE_UNARY_OPERATION_WITH_FLOAT(Softshrink)

DEFINE_UNARY_OPERATION_WITH_2_FLOATS(Hardswish)
DEFINE_UNARY_OPERATION_WITH_2_FLOATS(Hardsigmoid)
DEFINE_UNARY_OPERATION_WITH_2_FLOATS(Hardtanh)
DEFINE_UNARY_OPERATION_WITH_2_FLOATS(Clip)
DEFINE_UNARY_OPERATION_WITH_2_FLOATS(Clamp)
DEFINE_UNARY_OPERATION_WITH_2_FLOATS(Selu)
DEFINE_UNARY_OPERATION_WITH_2_FLOATS(Threshold)


}  // namespace unary
}  // namespace operations

// Custom Structs
constexpr auto pow = ttnn::register_operation_with_auto_launch_op<"ttnn::pow", operations::unary::PowerOperation>();
constexpr auto rdiv = ttnn::register_operation_with_auto_launch_op<"ttnn::rdiv", operations::unary::RdivOperation>();

// Single Tensor arg
constexpr auto tanhshrink = ttnn::register_operation_with_auto_launch_op<"ttnn::tanhshrink", operations::unary::TanhshrinkOperation>();
constexpr auto deg2rad = ttnn::register_operation_with_auto_launch_op<"ttnn::deg2rad", operations::unary::Deg2radOperation>();
constexpr auto rad2deg = ttnn::register_operation_with_auto_launch_op<"ttnn::rad2deg", operations::unary::Rad2degOperation>();
constexpr auto acosh = ttnn::register_operation_with_auto_launch_op<"ttnn::acosh", operations::unary::AcoshOperation>();
constexpr auto asinh = ttnn::register_operation_with_auto_launch_op<"ttnn::asinh", operations::unary::AsinhOperation>();
constexpr auto atanh = ttnn::register_operation_with_auto_launch_op<"ttnn::atanh", operations::unary::AtanhOperation>();
constexpr auto cbrt = ttnn::register_operation_with_auto_launch_op<"ttnn::cbrt", operations::unary::CbrtOperation>();
constexpr auto cosh = ttnn::register_operation_with_auto_launch_op<"ttnn::cosh", operations::unary::CoshOperation>();
constexpr auto digamma = ttnn::register_operation_with_auto_launch_op<"ttnn::digamma", operations::unary::DigammaOperation>();
constexpr auto lgamma = ttnn::register_operation_with_auto_launch_op<"ttnn::lgamma", operations::unary::LgammaOperation>();
constexpr auto log1p = ttnn::register_operation_with_auto_launch_op<"ttnn::log1p", operations::unary::Log1pOperation>();
constexpr auto mish = ttnn::register_operation_with_auto_launch_op<"ttnn::mish", operations::unary::MishOperation>();
constexpr auto multigammaln = ttnn::register_operation_with_auto_launch_op<"ttnn::multigammaln", operations::unary::MultigammalnOperation>();
constexpr auto sinh = ttnn::register_operation_with_auto_launch_op<"ttnn::sinh", operations::unary::SinhOperation>();
constexpr auto softsign = ttnn::register_operation_with_auto_launch_op<"ttnn::softsign", operations::unary::SoftsignOperation>();
constexpr auto swish = ttnn::register_operation_with_auto_launch_op<"ttnn::swish", operations::unary::SwishOperation>();
constexpr auto trunc = ttnn::register_operation_with_auto_launch_op<"ttnn::trunc", operations::unary::TruncOperation>();
constexpr auto var_hw = ttnn::register_operation_with_auto_launch_op<"ttnn::var_hw", operations::unary::VarHwOperation>();
constexpr auto std_hw = ttnn::register_operation_with_auto_launch_op<"ttnn::std_hw", operations::unary::StdHwOperation>();
constexpr auto normalize_hw = ttnn::register_operation_with_auto_launch_op<"ttnn::normalize_hw", operations::unary::NormalizeHwOperation>();
constexpr auto normalize_global = ttnn::register_operation_with_auto_launch_op<"ttnn::normalize_global", operations::unary::NormalizeGlobalOperation>();
constexpr auto frac = ttnn::register_operation_with_auto_launch_op<"ttnn::frac", operations::unary::FracOperation>();
constexpr auto logical_not_ = ttnn::register_operation_with_auto_launch_op<"ttnn::logical_not_", operations::unary::LogicalNotOperation>();

// Tensor + Float Param 1 + Float Param 2
constexpr auto hardswish = ttnn::register_operation_with_auto_launch_op<"ttnn::hardswish", operations::unary::HardswishOperation>();
constexpr auto hardsigmoid = ttnn::register_operation_with_auto_launch_op<"ttnn::hardsigmoid", operations::unary::HardsigmoidOperation>();
constexpr auto hardtanh = ttnn::register_operation_with_auto_launch_op<"ttnn::hardtanh", operations::unary::HardtanhOperation>();
constexpr auto clip = ttnn::register_operation_with_auto_launch_op<"ttnn::clip", operations::unary::ClipOperation>();
constexpr auto clamp = ttnn::register_operation_with_auto_launch_op<"ttnn::clamp", operations::unary::ClampOperation>();
constexpr auto selu = ttnn::register_operation_with_auto_launch_op<"ttnn::selu", operations::unary::SeluOperation>();
constexpr auto threshold = ttnn::register_operation_with_auto_launch_op<"ttnn::threshold", operations::unary::ThresholdOperation>();

// Tensor + int Dim
constexpr auto glu = ttnn::register_operation_with_auto_launch_op<"ttnn::glu", operations::unary::GluOperation>();
constexpr auto reglu = ttnn::register_operation_with_auto_launch_op<"ttnn::reglu", operations::unary::RegluOperation>();
constexpr auto geglu = ttnn::register_operation_with_auto_launch_op<"ttnn::geglu", operations::unary::GegluOperation>();
constexpr auto swiglu = ttnn::register_operation_with_auto_launch_op<"ttnn::swiglu", operations::unary::SwigluOperation>();

// Tensor + Float
constexpr auto hardshrink = ttnn::register_operation_with_auto_launch_op<"ttnn::hardshrink", operations::unary::HardshrinkOperation>();
constexpr auto softshrink = ttnn::register_operation_with_auto_launch_op<"ttnn::softshrink", operations::unary::SoftshrinkOperation>();
constexpr auto logit = ttnn::register_operation_with_auto_launch_op<"ttnn::logit", operations::unary::LogitOperation>();
constexpr auto celu = ttnn::register_operation_with_auto_launch_op<"ttnn::celu", operations::unary::CeluOperation>();
constexpr auto rpow = ttnn::register_operation_with_auto_launch_op<"ttnn::rpow", operations::unary::RpowOperation>();

// Tensor + Int
constexpr auto tril = ttnn::register_operation_with_auto_launch_op<"ttnn::tril", operations::unary::TrilOperation>();
constexpr auto triu = ttnn::register_operation_with_auto_launch_op<"ttnn::triu", operations::unary::TriuOperation>();
constexpr auto round = ttnn::register_operation_with_auto_launch_op<"ttnn::round", operations::unary::RoundOperation>();
constexpr auto polygamma = ttnn::register_operation_with_auto_launch_op<"ttnn::polygamma", operations::unary::PolygammaOperation>();




}  // namespace ttnn
