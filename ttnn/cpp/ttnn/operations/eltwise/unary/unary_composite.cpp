// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_composite.hpp"

#include "tt_metal/common/bfloat16.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape/reshape.hpp"
#include "ttnn/deprecated/tt_numpy/functions.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary_composite.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::unary {

namespace {
// Function variance of whole tensor.
// Tensor variance(const Tensor& y,const Tensor& mean_y);
Tensor _variance_impl(
    const Tensor& y,
    const Tensor& mean_y,
    Tensor& y_minus_mean_y,
    const std::optional<MemoryConfig>& memory_config) {
    std::vector<int> dims = { 2, 3 };
    constexpr float correction = 0.0f;
    auto shape_wh = y.get_legacy_shape();
    float scale = 1.0f / ((float)(shape_wh[3] * shape_wh[2]) - correction);
    Tensor sqr_y_minus_mean_y = ttnn::square(y_minus_mean_y, memory_config);
    return ttnn::sum(sqr_y_minus_mean_y, dims, true, std::nullopt, std::nullopt, scale);
}
Tensor _variance_impl(const Tensor& y, const Tensor& mean_y, const std::optional<MemoryConfig>& memory_config) {
    Tensor y_minus_mean_y = ttnn::bcast(0, y, mean_y, ttnn::BcastOpMath::SUB, ttnn::BcastOpDim::HW);
    return _variance_impl(y, mean_y, y_minus_mean_y, memory_config);
}

Tensor is_odd(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    Tensor result = ttnn::multiply(input, (1.0f/2.0f), std::nullopt, memory_config);
    Tensor floor_res = ttnn::floor(result, memory_config);
    return ttnn::ne(result, floor_res, std::nullopt, memory_config);
}

Tensor _std(const Tensor& y, const Tensor& mean_y, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::sqrt(_variance_impl(y, mean_y, memory_config));
}

Tensor _std(const Tensor& y, const Tensor& mean_y, Tensor& y_minus_mean_y, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::sqrt(_variance_impl(y, mean_y, y_minus_mean_y, memory_config));
}

}

Tensor PowerOperation::invoke(
    uint8_t queue_id,
    const Tensor& input,
    uint32_t exponent,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> preallocated_output) {

    return invoke(queue_id, input, static_cast<float>(exponent), memory_config, preallocated_output);
}

Tensor PowerOperation::invoke(
    const Tensor& input,
    uint32_t exponent,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> preallocated_output) {
    return invoke(DefaultQueueId, input, exponent, memory_config, preallocated_output);
}

Tensor PowerOperation::invoke(
    const Tensor& input,
    float exponent,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> preallocated_output) {

    return invoke(DefaultQueueId, input, exponent, memory_config, preallocated_output);
}

Tensor PowerOperation::invoke(
    uint8_t queue_id,
    const Tensor& input,
    float exponent,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> preallocated_output) {

    TT_FATAL(exponent >= 0.0f, "works for positive exponents only");
    const uint32_t exponent_floor = static_cast<uint32_t>(std::floor(exponent));
    if (static_cast<float>(exponent_floor) == exponent) {
        if(preallocated_output.has_value()){
            ttnn::power(queue_id, input, exponent_floor, memory_config, preallocated_output);
            return preallocated_output.value();
        }
        return ttnn::power(queue_id, input, exponent_floor, memory_config);
    }
    const float exponent_trunc = exponent - static_cast<float>(exponent_floor);
    Tensor pow_trunc_log = ttnn::multiply(queue_id, ttnn::log(queue_id, input, memory_config), exponent_trunc, std::nullopt, memory_config);
    Tensor pow_frac = ttnn::exp(queue_id, pow_trunc_log, false, memory_config);
    pow_trunc_log.deallocate();
    float t_nan = std::nanf("");
    Tensor result = ttnn::multiply(queue_id, ttnn::power(queue_id, input, exponent_floor, memory_config), pow_frac, std::nullopt, memory_config);
    // To handle negative inputs:
    // in torch For -ve inputs with float exponent power returns nan
    auto output_memory_config = preallocated_output.has_value() ? preallocated_output.value().memory_config() : memory_config.value_or(input.memory_config());
    result = ttnn::where(ttnn::ltz(queue_id, input, memory_config), t_nan, result, output_memory_config, preallocated_output);
    return result;
}

Tensor RdivOperation::invoke(
    uint8_t queue_id,
    const Tensor& input,
    float value,
    const std::string& round_mode,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> preallocated_output) {

    float t_inf = std::numeric_limits<float>::infinity();
    Tensor recip_result = ttnn::reciprocal(queue_id, input, memory_config, preallocated_output);
    Tensor result = ttnn::multiply(queue_id, recip_result, value, std::nullopt, memory_config, preallocated_output);

    if(round_mode == "trunc"){
       result = ttnn::trunc(result);
    }
    else if(round_mode == "floor"){
       result = ttnn::floor(result);
    }
    return ttnn::where(ttnn::eqz(queue_id, input, memory_config), t_inf, result, memory_config, preallocated_output);
}

// Tensor
// // tanhshrink(x) = x - tanh(x)
Tensor TanhshrinkOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    Tensor tan_x = ttnn::tanh(input, memory_config);
    Tensor result = ttnn::subtract(input, tan_x, std::nullopt, memory_config);
    return result;
}

Tensor Deg2radOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::multiply(input, (float)(M_PI / 180.0), std::nullopt, memory_config.value_or(input.memory_config()));
}

Tensor Rad2degOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::multiply(input, (float)(180.0 / M_PI), std::nullopt, memory_config.value_or(input.memory_config()));
}

// acosh(x) = log(x + sqrt(x^2 - 1))
Tensor AcoshOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
   Tensor t_one = ttnn::full_like(input, 1.0f);
   Tensor t_result(input);
   {
       Tensor ln_res(input);
       {
           Tensor x_abs = ttnn::abs(input, memory_config);
           Tensor x_sq_m1(input);
           {
               Tensor x_sq = ttnn::square(x_abs, memory_config);
               x_sq_m1 = ttnn::subtract(x_sq, 1.0f, std::nullopt, memory_config);
           }
           ln_res = ttnn::log(
               ttnn::add(x_abs, ttnn::sqrt(x_sq_m1, memory_config), std::nullopt, memory_config), memory_config);
       }
       // To handle inputs <= 1
       // input < 1, output is nan
       // input > 1, output is acosh(input)
       Tensor nan_res = ttnn::multiply(
           ttnn::le(input, t_one, std::nullopt, memory_config), std::nanf(""), std::nullopt, memory_config);
       t_result = ttnn::multiply(
           ttnn::gt(input, t_one, std::nullopt, memory_config), ln_res, std::nullopt, memory_config);
       t_result = ttnn::add(nan_res, t_result, std::nullopt, memory_config);
   }
   // input == 1, output is 0
   Tensor result = ttnn::where(ttnn::eq(input, t_one, std::nullopt, memory_config), 0.0f, t_result);
   return result;
}

// asinh(x) = log(x + sqrt(x^2 + 1))
Tensor AsinhOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
   Tensor ln_res(input);
   {
       Tensor x_abs = ttnn::abs(input, memory_config);
       Tensor x_sq_p1(input);
       {
           Tensor x_sq = ttnn::square(input, memory_config);
           x_sq_p1 = ttnn::add(x_sq, 1.0f, std::nullopt, memory_config);
       }
       ln_res =
           ttnn::log(ttnn::add(x_abs, ttnn::sqrt(x_sq_p1, memory_config), std::nullopt, memory_config), memory_config);
   }
   // input is negative, output is -asinh(input)
   Tensor result = ttnn::where(input, ln_res, ttnn::neg(ln_res, memory_config));
   return result;
}

// atanh[x] = 0.5 * ln((1 + x) / (1 - x))
Tensor AtanhOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
   Tensor comp_result(input);
   {
       Tensor nr_term(input);
       {
           Tensor pos_x = ttnn::add(input, 1.0f, std::nullopt, memory_config);
           Tensor neg_x = ttnn::subtract(input, 1.0f, std::nullopt, memory_config);
           nr_term = ttnn::log(
               ttnn::multiply(
                   pos_x, ttnn::reciprocal(ttnn::neg(neg_x, memory_config), memory_config), std::nullopt, memory_config),
               memory_config);
       }
       comp_result = ttnn::multiply(nr_term, 0.5f, std::nullopt, memory_config);
   }
   // Input is -1 > value > 1, output is nan
   // Input is -1 < value < 1, output is atanh(input)
   float t_nan = std::nanf("");
   Tensor abs_temp = ttnn::subtract(ttnn::abs(input, memory_config), 1.0f, std::nullopt, memory_config);
   Tensor result = ttnn::where(ttnn::ltz(abs_temp, memory_config), comp_result, t_nan);
   return result;
}

// cbrt(a) = pow(a,1/3) or (cbrt(a))**3 = a.
//         = exp[ (1/3)*log[a] ]
Tensor CbrtOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
   constexpr float scale = (float)(1.0 / 3.0);
   Tensor t_ln_input =
       ttnn::log(ttnn::abs(input, memory_config), memory_config);  // negative log is not useful here
   Tensor t1 = ttnn::multiply(t_ln_input, scale, std::nullopt, memory_config);
   t_ln_input.deallocate();
   Tensor t2 = ttnn::exp(t1, false, memory_config);
   t1.deallocate();
   Tensor t3 = ttnn::multiply(t2, ttnn::sign(input, memory_config), std::nullopt, memory_config);
   return t3;
}

// cosh[x] = (exp[x] + exp[-x])/2
Tensor CoshOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
   Tensor e_pos_x = ttnn::exp(input, false, memory_config);
   Tensor e_neg_x = ttnn::exp(ttnn::neg(input, memory_config), false, memory_config);
   Tensor nr_term = ttnn::add(e_pos_x, e_neg_x, std::nullopt, memory_config);
   e_pos_x.deallocate();
   e_neg_x.deallocate();
   Tensor scalar = ttnn::full_like(input, 0.5f);
   return ttnn::multiply(nr_term, scalar, std::nullopt, memory_config);
}

// TODO: In future will uplift the op once the floor and tan has supported.
// digamma support for the range of (1, inf)
Tensor DigammaOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
   Tensor t_log_out = ttnn::log(input, memory_config);  // negative log is not useful here

   // 1/2(z)
   Tensor output = ttnn::multiply(ttnn::reciprocal(input, memory_config), 0.5f, std::nullopt, memory_config);
   Tensor tmp = ttnn::square(ttnn::reciprocal(input, memory_config), memory_config);
   Tensor val_square = tmp;
   // (1/12) * x^2
   output = ttnn::subtract(output, ttnn::multiply(tmp, 0.083333333f), std::nullopt, memory_config);

   // (1/120) * x^4
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, memory_config);
   output =
       ttnn::add(output, ttnn::multiply(tmp, 0.008333333333333333f, std::nullopt, memory_config), std::nullopt, memory_config);

   //(1/252) * x^6
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, memory_config);
   output = ttnn::subtract(output, ttnn::multiply(tmp, 0.003968253968253968f, std::nullopt, memory_config), std::nullopt, memory_config);

   // (1/240) *x^8
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, memory_config);
   output = ttnn::add(output, ttnn::multiply(tmp, 0.004166666666666667f, std::nullopt, memory_config), std::nullopt, memory_config);

   //(1/132) * x^10
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, memory_config);
   output = ttnn::subtract(
       output, ttnn::multiply(tmp, 0.007575757575757576), std::nullopt, memory_config);

   //(691/32760) * x^12
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, memory_config);
   output =
       ttnn::add(output, ttnn::multiply(tmp, 0.021092796092796094, std::nullopt, memory_config), std::nullopt, memory_config);

   //(1/12) * x^14
   tmp = ttnn::multiply(tmp, val_square, std::nullopt, memory_config);
   output = ttnn::subtract(output, ttnn::multiply(tmp, 0.08333333333333333, std::nullopt, memory_config), std::nullopt, memory_config);

   return ttnn::subtract(t_log_out, output, std::nullopt, memory_config);
}

Tensor LgammaOperation::invoke(const Tensor& x, const std::optional<MemoryConfig>& memory_config) {
    Tensor result(x);
    {
        Tensor t(x);
        {
            Tensor temp_log(x);
            {
                Tensor temp(x);
                Tensor input = ttnn::subtract(x, 1.0f, std::nullopt, memory_config);
                {
                    Tensor z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 1.0f, std::nullopt, memory_config), memory_config),
                        76.18009172947146f,
                        std::nullopt,
                        memory_config);
                    temp = ttnn::add(z1, 1.0f, std::nullopt, memory_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 2.0f, std::nullopt, memory_config), memory_config),
                        -86.50532032941677f,
                        std::nullopt,
                        memory_config);
                    temp = ttnn::add(temp, z1, std::nullopt, memory_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 3.0f, std::nullopt, memory_config), memory_config),
                        24.01409824083091f,
                        std::nullopt,
                        memory_config);
                    temp = ttnn::add(temp, z1, std::nullopt, memory_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 4.0f, std::nullopt, memory_config), memory_config),
                        -1.231739572450155f,
                        std::nullopt,
                        memory_config);
                    temp = ttnn::add(temp, z1, std::nullopt, memory_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 5.0f, std::nullopt, memory_config), memory_config),
                        0.1208650973866179e-2f,
                        std::nullopt,
                        memory_config);
                    temp = ttnn::add(temp, z1, std::nullopt, memory_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 6.0f, std::nullopt, memory_config), memory_config),
                        -0.5395239384953e-5f,
                        std::nullopt,
                        memory_config);
                    temp = ttnn::add(temp, z1, std::nullopt, memory_config);
                }
                {
                    Tensor t_log(x);
                    {
                        t = ttnn::add(input, 5.5f, std::nullopt, memory_config);
                        t_log = ttnn::log(t, memory_config);
                    }
                    temp_log = ttnn::log(temp, memory_config);
                    result = ttnn::add(
                        ttnn::multiply(
                            ttnn::add(input, 0.5f, std::nullopt, memory_config), t_log, std::nullopt, memory_config),
                        0.918938531357171f,
                        std::nullopt,
                        memory_config);
                }
            }
            result = ttnn::add(result, temp_log, std::nullopt, memory_config);
        }
        result = ttnn::subtract(result, t, std::nullopt, memory_config);
        {
            {
                Tensor t_one = ttnn::full_like(x, 1.0f);
                result = ttnn::where(ttnn::eq(x, t_one, std::nullopt, memory_config), 0.0f, result);
            }
            {
                Tensor t_two = ttnn::full_like(x, 2.0f);
                result = ttnn::where(ttnn::eq(x, t_two, std::nullopt, memory_config), 0.0f, result);
            }
        }
    }
    return result;
}

// log1p 1
// use transformation y = log(1.0 + x) by broadcast
Tensor Log1pOperation::invoke(const Tensor& x, const std::optional<MemoryConfig>& memory_config) {
    Tensor t_one = ttnn::full_like(x, 1.0f);
    Tensor x_1 = ttnn::add(t_one, x, std::nullopt, memory_config);
    Tensor result_log1p = ttnn::log(x_1, memory_config);
    return result_log1p;
}

// mish[x] = x*tanh[softplus[x]]
// use transformation y = x*tanh[softplus[x]] by broadcast
// Ref: https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/
Tensor MishOperation::invoke(const Tensor& x, const std::optional<MemoryConfig>& memory_config) {
    Tensor sp_x = ttnn::softplus(x, 1.0f, 20.0f, memory_config);
    Tensor tanh_x = ttnn::tanh(sp_x, memory_config);
    sp_x.deallocate();
    Tensor mish_x = ttnn::multiply(x, tanh_x, std::nullopt, memory_config);
    return mish_x;
}

// multivariate log-gamma function
// Ref : https://pytorch.org/docs/stable/special.html#torch.special.multigammaln
Tensor MultigammalnOperation::invoke(const Tensor& x, const std::optional<MemoryConfig>& memory_config) {
    Tensor result = ttnn::lgamma(x, memory_config);
    result = ttnn::add(
        result, ttnn::lgamma(ttnn::subtract(x, 0.5f, std::nullopt, memory_config), memory_config), std::nullopt, memory_config);
    result = ttnn::add(
        result, ttnn::lgamma(ttnn::subtract(x, 1.0f, std::nullopt, memory_config), memory_config), std::nullopt, memory_config);
    result = ttnn::add(
        result, ttnn::lgamma(ttnn::subtract(x, 1.5f, std::nullopt, memory_config), memory_config), std::nullopt, memory_config);
    result = ttnn::add(result, 3.434189657547f, std::nullopt, memory_config);
    return result;
}

Tensor SinhOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    Tensor e_pos_x = ttnn::exp(input, false, memory_config);
    Tensor e_neg_x = ttnn::exp(ttnn::neg(input, memory_config), false, memory_config);
    Tensor nr_term = ttnn::subtract(e_pos_x, e_neg_x, std::nullopt, memory_config);
    e_pos_x.deallocate();
    e_neg_x.deallocate();
    Tensor scalar = ttnn::full_like(input, 0.5f);
    return ttnn::multiply(nr_term, scalar, std::nullopt, memory_config);
}

// Function: softsign
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
Tensor SoftsignOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
     Tensor result = ttnn::multiply(
        input,
        ttnn::reciprocal(
            ttnn::add(
                ttnn::abs(input, memory_config),
                1.0f,
                std::nullopt, memory_config), memory_config),
        std::nullopt,
        memory_config);

    return result;
}

// x / (1.0f + exp(-x))
Tensor SwishOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::silu(input, memory_config);
}

Tensor TruncOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor floor_res = ttnn::floor(input, memory_config);
    Tensor trunc_res = ttnn::where(ttnn::ne(input, floor_res), ttnn::add(floor_res, 1.0f, std::nullopt, memory_config), floor_res);
    Tensor result = ttnn::where(ttnn::gtz(input, memory_config), floor_res, trunc_res);
    return result;
}

// Function variance of whole tensor.
// Tensor variance(const Tensor& y,const Tensor& mean_y);
Tensor VarHwOperation::invoke(const Tensor& y, const std::optional<MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(y.memory_config());
    std::vector<int> dims = { 2, 3 };
    Tensor mean_y = ttnn::mean(y, dims, true);
    return _variance_impl(y, mean_y, output_memory_config);
}

// Function std
// compute standard deviation of tensor y = sqrt( E((y-<y>)^2)/ y.volume() )
//  Ref: torch.std
Tensor StdHwOperation::invoke(const Tensor& y, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::sqrt(ttnn::var_hw(y, memory_config));
}

// Function normalize
// use transformation y = (y - mean(y))/std(y) by broadcast
Tensor NormalizeHwOperation::invoke(const Tensor& y, const std::optional<MemoryConfig>& memory_config) {
    std::vector<int> dims = { 2, 3 };
    Tensor mean_y = ttnn::mean(y, dims, true);
    Tensor y_minus_mean_y = ttnn::bcast(0, y, mean_y, ttnn::BcastOpMath::SUB, ttnn::BcastOpDim::HW);
    Tensor std_y = _std(y, mean_y, y_minus_mean_y, memory_config);
    Tensor recip_std_y = ttnn::reciprocal(std_y, memory_config);
    Tensor z = ttnn::multiply(y_minus_mean_y, recip_std_y, std::nullopt, memory_config);
    return z;
}

Tensor NormalizeGlobalOperation::invoke(const Tensor& y, const std::optional<MemoryConfig>& memory_config) {
    TT_FATAL(y.get_legacy_shape().rank() == 4, "Cannot support non-rank 4 Tensor");

    // format to HW
    Tensor y_hw = ttnn::reshape_on_device(
        y, 1, 1, y.get_legacy_shape()[2], y.get_legacy_shape()[3] * y.get_legacy_shape()[1] * y.get_legacy_shape()[0]);

    // compute
    Tensor z_0 = ttnn::normalize_hw(y_hw, memory_config);
    TT_FATAL(y_hw.get_legacy_shape() == z_0.get_legacy_shape(), "shape match");
    y_hw.deallocate();

    // reformat
    Tensor z_1 = ttnn::reshape_on_device(
        z_0, y.get_legacy_shape()[0], y.get_legacy_shape()[1], y.get_legacy_shape()[2], y.get_legacy_shape()[3]);
    z_0.deallocate();

    return z_1;
}

Tensor FracOperation::invoke(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor trunc_res = ttnn::trunc(input);
    Tensor result = ttnn::subtract(input, trunc_res, std::nullopt, memory_config);
    return result;
}

// tanhshrink(x) = x - tanh(x)
Tensor LogicalNotOperation::invoke(const Tensor& x, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::logical_not(x, memory_config, x);
}


// Tensor + Int

// tril : select lower triangular region of input matrix
Tensor TrilOperation::invoke(const Tensor& input, int32_t diag, const std::optional<MemoryConfig>& memory_config) {
    Tensor index_l = tt::numpy::index_tril<::bfloat16>(
        input.get_legacy_shape(), diag, DataType::BFLOAT16, Layout::TILE, input.device(), memory_config.value());
    return ttnn::multiply(input, index_l, std::nullopt, memory_config);
}

// triu : select upper triangular region of input matrix
Tensor TriuOperation::invoke(const Tensor& input, int32_t diag, const std::optional<MemoryConfig>& memory_config) {
    Tensor index_u = tt::numpy::index_triu<::bfloat16>(
        input.get_legacy_shape(), diag, DataType::BFLOAT16, Layout::TILE, input.device(), memory_config.value());
    return ttnn::multiply(input, index_u, std::nullopt, memory_config);
}

Tensor RoundOperation::invoke(const Tensor& input, int32_t decimals, const std::optional<MemoryConfig>& memory_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor floor_res = ttnn::floor(input, memory_config);
    if (decimals != 0) {  // TODO: For decimal value!=0
        Tensor power_10 =
            ttnn::power(ttnn::full_like(input, 10.0f), decimals, memory_config);
        Tensor rounded_non_half = ttnn::floor(
            ttnn::add(ttnn::multiply(input, power_10, std::nullopt, memory_config), 0.5, std::nullopt, memory_config),
            memory_config);
        rounded_non_half = ttnn::div(rounded_non_half, power_10);
        return rounded_non_half;
    } else {  // Bankers' Rounding
        Tensor rounded_non_half = ttnn::floor(
            ttnn::add(
                input,
                ttnn::where(ttnn::logical_and(ttnn::ge(input, 0.4), ttnn::le(input, 0.5)), 0.4f, 0.5f, memory_config.value()),
                std::nullopt,
                memory_config),
            memory_config.value());
        Tensor fractional_part = ttnn::subtract(input, floor_res, std::nullopt, memory_config);
        Tensor is_half = ttnn::eq(fractional_part, 0.5, std::nullopt, memory_config);
        Tensor rounded_half =
            ttnn::add(floor_res, is_odd(floor_res, memory_config), std::nullopt, memory_config);
        return ttnn::where(is_half, rounded_half, rounded_non_half, memory_config.value());
    }
}

// polygamma support for the range of input(1, 10) and n(1, 10)
Tensor PolygammaOperation::invoke(const Tensor& input, int32_t k, const std::optional<MemoryConfig>& memory_config) {
    float k_der = 1.0f + k;
    float fact_val = std::tgamma(k_der);
    float pos_neg = 1.0f;
    if (k == 2 || k == 4 || k == 6 || k == 8 || k == 10) {
        pos_neg = -1.0f;
    }
    Tensor temp(input);
    {
        Tensor z1 = ttnn::reciprocal(ttnn::power(input, k_der, memory_config), memory_config);
        temp = z1;
        for (int idx = 1; idx < 11; idx++) {
            z1 = ttnn::reciprocal(ttnn::power(ttnn::add(input, idx, std::nullopt, memory_config), k_der, memory_config), memory_config);
            temp = ttnn::add(temp, z1, std::nullopt, memory_config);
        }
    }
    fact_val *= pos_neg;
    return ttnn::multiply(temp, fact_val, std::nullopt, memory_config);
}

std::vector<Tensor> split_tensor_for_glu(const Tensor& input, int32_t dim, const std::optional<MemoryConfig>& memory_config) {
    std::vector<Tensor> t_split;
    Shape inshape(input.get_legacy_shape());
    TT_FATAL(((inshape[dim] / 2) % TILE_WIDTH == 0), "Split tensor dimension should be in full tile");
    std::vector<uint32_t> s_a = {0, 0, 0, 0};
    std::vector<uint32_t> e_a = {input.get_legacy_shape()[0] - 1, inshape[1] - 1, inshape[2] - 1, inshape[3] / 2 - 1};

    std::vector<uint32_t> s_b = {0, 0, 0, inshape[3] / 2};
    std::vector<uint32_t> e_b = {inshape[0] - 1, inshape[1] - 1, inshape[2] - 1, inshape[3] - 1};

    Tensor t_a = ttnn::slice(0, input, s_a, e_a, memory_config);
    Tensor t_b = ttnn::slice(0, input, s_b, e_b, memory_config);

    t_split.emplace_back(t_a);
    t_split.emplace_back(t_b);

    return t_split;
}

// Gated Linear Unit activation: matmul(split[0],sigmoid(split[1]))
Tensor GluOperation::invoke(const Tensor& input, int32_t dim, const std::optional<MemoryConfig>& memory_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim GLU only supported at this time ");
    if (dim == -1)
        dim = 3;
    std::vector<Tensor> ab = split_tensor_for_glu(input, dim, memory_config);
    Tensor sigmoid_b = ttnn::sigmoid(ab[1], memory_config);
    Tensor glu_result = ttnn::multiply(ab[0], sigmoid_b, std::nullopt, memory_config);
    return glu_result;
}

// ReLU Gated Linear Unit activation: matmul(split[0],relu(split[1]))
Tensor RegluOperation::invoke(const Tensor& input, int32_t dim, const std::optional<MemoryConfig>& memory_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim REGLU only supported at this time ");
    if (dim == -1)
        dim = 3;
    std::vector<Tensor> ab = split_tensor_for_glu(input, dim, memory_config);
    Tensor relu_b = ttnn::relu(ab[1], memory_config);
    Tensor reglu_result = ttnn::multiply(ab[0], relu_b, std::nullopt, memory_config);
    return reglu_result;
}

// Gaussian Error Gated Linear Unit activation: matmul(split[0],gelu(split[1]))
Tensor GegluOperation::invoke(const Tensor& input, int32_t dim, const std::optional<MemoryConfig>& memory_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim GEGLU only supported at this time ");
    if (dim == -1)
        dim = 3;

    std::vector<Tensor> ab = split_tensor_for_glu(input, dim, memory_config);

    constexpr bool fast_appx = true;
    Tensor gelu_b = ttnn::gelu(ab[1], fast_appx, memory_config);
    Tensor geglu_result = ttnn::multiply(ab[0], gelu_b, std::nullopt, memory_config);
    return geglu_result;
}

// Swish Gated Linear Unit activation: matmul(split[0],swish(split[1]))
Tensor SwigluOperation::invoke(const Tensor& input, int32_t dim, const std::optional<MemoryConfig>& memory_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim SWIGLU only supported at this time ");
    if (dim == -1)
        dim = 3;

    std::vector<Tensor> ab = split_tensor_for_glu(input, dim, memory_config);

    Tensor swish_b = ttnn::swish(ab[1], memory_config);
    Tensor swiglu_result = ttnn::multiply(ab[0], swish_b, std::nullopt, memory_config);
    return swiglu_result;
}

// Tensor + Float

// Function: hardshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html
Tensor HardshrinkOperation::invoke(const Tensor& a, float param, const std::optional<MemoryConfig>& memory_config) {
    TT_ASSERT(param >= 0);
    Tensor t1 = ttnn::multiply(ttnn::ltz(ttnn::add(a, param, std::nullopt, memory_config)), a, std::nullopt, memory_config);
    Tensor t2 = ttnn::multiply(ttnn::gtz(ttnn::subtract(a, param, std::nullopt, memory_config)), a, std::nullopt, memory_config);
    return ttnn::add(t1, t2, std::nullopt, memory_config);
}

// Function: softshrink
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html
Tensor SoftshrinkOperation::invoke(const Tensor& a, float param, const std::optional<MemoryConfig>& memory_config) {
    TT_ASSERT(param >= 0);
    Tensor t_a_plus_param = ttnn::add(a, param, std::nullopt, memory_config);
    Tensor t1 = ttnn::multiply(ttnn::ltz(t_a_plus_param, memory_config), t_a_plus_param, std::nullopt, memory_config);
    t_a_plus_param.deallocate();
    Tensor t_a_minus_param = ttnn::subtract(a, param, std::nullopt, memory_config);
    Tensor t2 = ttnn::multiply(ttnn::gtz(t_a_minus_param, memory_config), t_a_minus_param, std::nullopt, memory_config);
    t_a_minus_param.deallocate();
    return ttnn::add(t1, t2, std::nullopt, memory_config);
}

// logit(input, eps)=log(input / 1 - input)
Tensor LogitOperation::invoke(const Tensor& input, float eps, const std::optional<MemoryConfig>& memory_config) {
    Tensor t_eps = ttnn::full_like(input, eps);
    Tensor t1m_eps = ttnn::full_like(input, (1 - eps));
    Tensor logit_input = ttnn::where(
        ttnn::ltz(t_eps, memory_config),
        input,
        ttnn::where(
            ttnn::lt(input, t_eps, std::nullopt, memory_config),
            t_eps,
            ttnn::where(ttnn::gt(input, t1m_eps, std::nullopt, memory_config), t1m_eps, input)
            )
        );
    Tensor linput_m1 = ttnn::rsub(logit_input, 1.0, memory_config);
    Tensor log_input = ttnn::multiply(logit_input, ttnn::reciprocal(linput_m1, memory_config), std::nullopt, memory_config);
    linput_m1.deallocate();
    Tensor t_inf = ttnn::multiply(ttnn::sign(input, memory_config), std::numeric_limits<float>::infinity(), std::nullopt, memory_config);
    Tensor logit_result = ttnn::where(
        ttnn::eq(logit_input, 1.0, std::nullopt, memory_config),
        t_inf,
        ttnn::where(ttnn::ltz(log_input, memory_config), std::nanf(" "), ttnn::log(log_input, memory_config))
        );
    return logit_result;
}

// Celu
// torch.where(x > 0, x, alpha * (torch.exp(x / alpha) - 1))
Tensor CeluOperation::invoke(const Tensor& input, float alpha, const std::optional<MemoryConfig>& memory_config) {
    float recip_val = 1.0f / alpha;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, recip_val},
    UnaryWithParam{UnaryOpType::EXP, 1.0f},
    UnaryWithParam{UnaryOpType::SUB_UNARY_SFPU, 1.0f}, UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, alpha} };

    Tensor result = ttnn::unary_chain(input, ops_chain, memory_config);
    result = ttnn::where(ttnn::gtz(input, memory_config), input, result);
    return result;
}

// rpow: y = k**(a) = exp( a**log(k) )
Tensor RpowOperation::invoke(const Tensor& a, float k, const std::optional<MemoryConfig>& memory_config) {
    TT_ASSERT(k > 0.0, "rpow cannot be calcualted for non-positive numbers");
    float log_k = logf(k);

    Tensor result = ttnn::multiply(a, log_k);
    return ttnn::exp(result, false);
}

// Tensor + 2 Floats

// Function @hard_swish
// use transformation y = x * hardsigmoid( x ) by broadcast
// Ref: PyTorch
// hard swish(x) = x*hardsigmoid(x,scale,shift)
Tensor HardswishOperation::invoke(const Tensor& a, float scale, float shift, const std::optional<MemoryConfig>& memory_config) {
   Tensor a_sigmoid = ttnn::hardsigmoid(a, scale, shift, memory_config);
   Tensor result_sq = ttnn::multiply(a_sigmoid, a, std::nullopt, memory_config);
   return result_sq;
}

// Function Hard Sigmoid
//     Ref: https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py
//
//     slope = tensor.constant(0.2, dtype=out_dtype)
//     shift = tensor.constant(0.5, dtype=out_dtype)
//
//     x1 = (x * slope) + shift
//     y = tensor.clip(x1, 0, 1)
//
// PyTorch version:
// hard sigmoid(x) = { x <= -3: 0, x >= +3: +3, x/6 + 0.5 otherwise}
Tensor HardsigmoidOperation::invoke(const Tensor& a, float param1, float param2, const std::optional<MemoryConfig>& memory_config) {
   Tensor a_t = ttnn::full_like(a,param1);
   Tensor b_t = ttnn::full_like(a,param2);
   Tensor a_mac = ttnn::mac(a, a_t, b_t);  // multiply and add.
   Tensor a_clip = relu_max(a_mac, 1.0f);
   return a_clip;
}

Tensor HardtanhOperation::invoke(const Tensor& a, float low, float high, const std::optional<MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(a.memory_config());
    return ttnn::clip(a, low, high, output_memory_config);
}


// Function Clip
// use clip y = min( max( x, min_value), max_value) by broadcast
// Ref: https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp
Tensor ClipOperation::invoke(const Tensor& a, float low, float high, const std::optional<MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(a.memory_config());
    const Tensor h_const = full_like(a, high);
    Tensor a_max = ttnn::minimum(a, h_const, output_memory_config);
    if (low == 0.0f) {
        return ttnn::relu(a_max, output_memory_config);
    } else {
        const Tensor l_const = full_like(a, low);
        return ttnn::maximum(a_max, l_const, output_memory_config);
    }
}

Tensor ClampOperation::invoke(const Tensor& a, float low, float high, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::clip(a, low, high, memory_config);
}

// Theano defines this differently...
/**
 *
 *   alpha = 1.6732632423543772848170429916717
 *    scale = 1.0507009873554804934193349852946
 *    return scale * elu(x, alpha)
 *
 */
// Function Selu - scaled exponential linear
// use transformation y = scale *(max(0,x)) + min(0,alpha * (exp(X)-1)) by broadcast
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
Tensor SeluOperation::invoke(const Tensor& x, float scale, float alpha, const std::optional<MemoryConfig>& memory_config) {
    // term 2
    Tensor x_Exp = ttnn::exp(x, false, memory_config);
    Tensor x_Exp_minus_1 =ttnn::subtract(x_Exp , -1.0f, std::nullopt, memory_config);
    x_Exp.deallocate();
    Tensor result_t2_ = ttnn::multiply(x_Exp_minus_1, alpha, std::nullopt, memory_config);
    x_Exp_minus_1.deallocate();
    Tensor result_term2 =
        ttnn::multiply(ttnn::gtz(result_t2_, memory_config), result_t2_, std::nullopt, memory_config);
    result_t2_.deallocate();

    // term 1
    Tensor x_relu = ttnn::relu(x, memory_config);
    Tensor result_term1 = ttnn::multiply(x_relu, scale, std::nullopt, memory_config);
    x_relu.deallocate();
    Tensor result_selu = ttnn::add(result_term1, result_term2, std::nullopt, memory_config);

    return result_selu;
}

// threshold(a,t,v) = (a <= t)*v + (a > t)*a
Tensor ThresholdOperation::invoke(const Tensor& input, float threshold, float value, const std::optional<MemoryConfig>& memory_config) {
    Tensor t0 = ttnn::subtract(input, threshold, std::nullopt, memory_config);
    Tensor t1 = ttnn::multiply(ttnn::lez(t0), value, std::nullopt, memory_config);
    Tensor t2 = ttnn::multiply(ttnn::gtz(t0, memory_config), input, std::nullopt, memory_config);
    return ttnn::add(t1, t2, std::nullopt, memory_config);
}

}  // namespace ttnn::operations::unary
