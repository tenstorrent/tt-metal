// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <functional>
#include <optional>

#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/deprecated/tt_numpy/functions.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"
#include "unary_composite_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/types.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"

namespace ttnn::operations::unary{

Tensor _deg2rad(const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::multiply(input_tensor, (float)(M_PI / 180.0), std::nullopt, output_mem_config.value_or(input_tensor.memory_config()));
}

Tensor _rad2deg(const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::multiply(input_tensor, (float)(180.0 / M_PI), std::nullopt, output_mem_config.value_or(input_tensor.memory_config()));
}

// // tanhshrink(x) = x - tanh(x)
Tensor _tanhshrink(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor tan_x = ttnn::tanh(x, output_mem_config);
    Tensor result = ttnn::subtract(x, tan_x, std::nullopt, output_mem_config);
    return result;
}

// acosh(x) = log(x + sqrt(x^2 - 1))
Tensor _acosh(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor t_one = ttnn::ones_like(input_a);
   Tensor t_result(input_a);
   {
       Tensor ln_res(input_a);
       {
           Tensor x_abs = ttnn::abs(input_a, output_mem_config);
           Tensor x_sq_m1(input_a);
           {
               Tensor x_sq = ttnn::square(x_abs, output_mem_config);
               x_sq_m1 = ttnn::subtract(x_sq, 1.0f);
           }
           ln_res = ttnn::log(
               ttnn::add(x_abs, ttnn::sqrt(x_sq_m1, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
       }
       // To handle inputs <= 1
       // input < 1, output is nan
       // input > 1, output is acosh(input)
       Tensor nan_res = ttnn::multiply(
           ttnn::le(input_a, t_one, std::nullopt, output_mem_config), std::nanf(""), std::nullopt, output_mem_config);
       t_result = ttnn::multiply(
           ttnn::gt(input_a, t_one, std::nullopt, output_mem_config), ln_res, std::nullopt, output_mem_config);
       t_result = ttnn::add(nan_res, t_result, std::nullopt, output_mem_config);
   }
   // input == 1, output is 0
   Tensor result = where(ttnn::eq(input_a, t_one, std::nullopt, output_mem_config), 0.0f, t_result);
   return result;
}

// asinh(x) = log(x + sqrt(x^2 + 1))
Tensor _asinh(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor ln_res(input_a);
   {
       Tensor x_abs = ttnn::abs(input_a, output_mem_config);
       Tensor x_sq_p1(input_a);
       {
           Tensor x_sq = ttnn::square(input_a, output_mem_config);
           x_sq_p1 = ttnn::add(x_sq, 1.0f);
       }
       ln_res =
           ttnn::log(ttnn::add(x_abs, ttnn::sqrt(x_sq_p1, output_mem_config), std::nullopt, output_mem_config), output_mem_config);
   }
   // input is negative, output is -asinh(input)
   Tensor result = where(input_a, ln_res, ttnn::neg(ln_res, output_mem_config));
   return result;
}

// atanh[x] = 0.5 * ln((1 + x) / (1 - x))
Tensor _atanh(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor comp_result(input_a);
   {
       Tensor nr_term(input_a);
       {
           Tensor pos_x = ttnn::add(input_a, 1.0f);
           Tensor neg_x = ttnn::subtract(input_a, 1.0f);
           nr_term = ttnn::log(
               ttnn::multiply(
                   pos_x, ttnn::reciprocal(ttnn::neg(neg_x, output_mem_config), output_mem_config), std::nullopt, output_mem_config),
               output_mem_config);
       }
       comp_result = ttnn::multiply(nr_term, 0.5f);
   }
   // Input is -1 > value > 1, output is nan
   // Input is -1 < value < 1, output is atanh(input)
   float t_nan = std::nanf("");
   Tensor abs_temp = ttnn::subtract(ttnn::abs(input_a, output_mem_config), 1.0f);
   Tensor result = where(ttnn::ltz(abs_temp, output_mem_config), comp_result, t_nan);
   return result;
}

// cbrt(a) = pow(a,1/3) or (cbrt(a))**3 = a.
//         = exp[ (1/3)*log[a] ]
Tensor _cbrt(const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
   constexpr float scale = (float)(1.0 / 3.0);
   Tensor t_ln_input =
       ttnn::log(ttnn::abs(input_tensor, output_mem_config), output_mem_config);  // negative log is not useful here
   Tensor t1 = ttnn::multiply(t_ln_input, scale, std::nullopt);
   t_ln_input.deallocate();
   Tensor t2 = ttnn::exp(t1, false, output_mem_config);
   t1.deallocate();
   Tensor t3 = ttnn::multiply(t2, ttnn::sign(input_tensor, output_mem_config), std::nullopt);
   return t3;
}

// cosh[x] = (exp[x] + exp[-x])/2
Tensor _cosh(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor e_pos_x = ttnn::exp(input_a, false, output_mem_config);
   Tensor e_neg_x = ttnn::exp(ttnn::neg(input_a, output_mem_config), false, output_mem_config);
   Tensor nr_term = ttnn::add(e_pos_x, e_neg_x, std::nullopt);
   e_pos_x.deallocate();
   e_neg_x.deallocate();
   Tensor scalar = ttnn::full_like(input_a, 0.5f);
    //    ttnn::operations::creation::create_scalar(0.5f, input_a.get_dtype(), Layout::TILE, input_a.device());
   return ttnn::multiply(nr_term, scalar, std::nullopt);
}

// TODO: In future will uplift the op once the floor and tan has supported.
// digamma support for the range of (1, inf)
Tensor _digamma(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor t_log_out = ttnn::log(input_a, output_mem_config);  // negative log is not useful here

   // 1/2(z)
   Tensor output = ttnn::multiply(ttnn::reciprocal(input_a, output_mem_config), 0.5f, std::nullopt, output_mem_config);
   Tensor tmp = ttnn::square(ttnn::reciprocal(input_a, output_mem_config), output_mem_config);
   Tensor val_square = tmp;
   // (1/12) * x^2
   output = ttnn::subtract(output, ttnn::multiply(tmp, 0.083333333f), std::nullopt);

   // (1/120) * x^4
   tmp = ttnn::multiply(tmp, val_square, std::nullopt);
   output =
       ttnn::add(output, ttnn::multiply(tmp, 0.008333333333333333f), std::nullopt);

   //(1/252) * x^6
   tmp = ttnn::multiply(tmp, val_square, std::nullopt);
   output = ttnn::subtract(
       output, ttnn::multiply(tmp, 0.003968253968253968f), std::nullopt);

   // (1/240) *x^8
   tmp = ttnn::multiply(tmp, val_square, std::nullopt);
   output =
       ttnn::add(output, ttnn::multiply(tmp, 0.004166666666666667f), std::nullopt);

   //(1/132) * x^10
   tmp = ttnn::multiply(tmp, val_square, std::nullopt);
   output = ttnn::subtract(
       output, ttnn::multiply(tmp, 0.007575757575757576), std::nullopt);

   //(691/32760) * x^12
   tmp = ttnn::multiply(tmp, val_square, std::nullopt);
   output =
       ttnn::add(output, ttnn::multiply(tmp, 0.021092796092796094), std::nullopt);

   //(1/12) * x^14
   tmp = ttnn::multiply(tmp, val_square, std::nullopt);
   output =
       ttnn::subtract(output, ttnn::multiply(tmp, 0.08333333333333333), std::nullopt);

   return ttnn::subtract(t_log_out, output, std::nullopt);
}

Tensor _lgamma(const Tensor& x,  const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result(x);
    {
        Tensor t(x);
        {
            Tensor temp_log(x);
            {
                Tensor temp(x);
                Tensor input = ttnn::subtract(x, 1.0f, std::nullopt, output_mem_config);
                {
                    Tensor z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 1.0f, std::nullopt, output_mem_config), output_mem_config),
                        76.18009172947146f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(z1, 1.0f, std::nullopt, output_mem_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 2.0f, std::nullopt, output_mem_config), output_mem_config),
                        -86.50532032941677f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 3.0f, std::nullopt, output_mem_config), output_mem_config),
                        24.01409824083091f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 4.0f, std::nullopt, output_mem_config), output_mem_config),
                        -1.231739572450155f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 5.0f, std::nullopt, output_mem_config), output_mem_config),
                        0.1208650973866179e-2f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);

                    z1 = ttnn::multiply(
                        ttnn::reciprocal(ttnn::add(input, 6.0f, std::nullopt, output_mem_config), output_mem_config),
                        -0.5395239384953e-5f,
                        std::nullopt,
                        output_mem_config);
                    temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);
                }
                {
                    Tensor t_log(x);
                    {
                        t = ttnn::add(input, 5.5f, std::nullopt, output_mem_config);
                        t_log = ttnn::log(t, output_mem_config);
                    }
                    temp_log = ttnn::log(temp, output_mem_config);
                    result = ttnn::add(
                        ttnn::multiply(
                            ttnn::add(input, 0.5f, std::nullopt, output_mem_config), t_log, std::nullopt, output_mem_config),
                        0.918938531357171f,
                        std::nullopt,
                        output_mem_config);
                }
            }
            result = ttnn::add(result, temp_log, std::nullopt, output_mem_config);
        }
        result = ttnn::subtract(result, t, std::nullopt, output_mem_config);
        {
            {
                Tensor t_one = ttnn::ones_like(x);
                result = where(ttnn::eq(x, t_one, std::nullopt, output_mem_config), 0.0f, result);
            }
            {
                Tensor t_two = ttnn::full_like(x, 2.0f);
                result = where(ttnn::eq(x, t_two, std::nullopt, output_mem_config), 0.0f, result);
            }
        }
    }
    return result;
}

// log1p 1
// use transformation y = log(1.0 + x) by broadcast
Tensor _log1p(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_one = ttnn::ones_like(x);
    Tensor x_1 = ttnn::add(t_one, x, std::nullopt, output_mem_config);
    Tensor result_log1p = ttnn::log(x_1, output_mem_config);
    return result_log1p;
}

// mish[x] = x*tanh[softplus[x]]
// use transformation y = x*tanh[softplus[x]] by broadcast
// Ref: https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/
Tensor _mish(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({x}))};
    operation::launch_op(
        [output_mem_config](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            const auto& x = input_tensors.at(0);
            Tensor sp_x = ttnn::softplus(x, 1.0f, 20.0f, output_mem_config);
            Tensor tanh_x = ttnn::tanh(sp_x, output_mem_config);
            sp_x.deallocate();
            Tensor mish_x = ttnn::multiply(x, tanh_x, std::nullopt, output_mem_config);
            return {mish_x};
        },
        {x},
        output_tensors);
    return output_tensors.at(0);
}

// multivariate log-gamma function
// Ref : https://pytorch.org/docs/stable/special.html#torch.special.multigammaln
Tensor _multigammaln(const Tensor& x, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor result = _lgamma(x, output_mem_config);
    result = ttnn::add(
        result, _lgamma(ttnn::subtract(x, 0.5f, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::add(
        result, _lgamma(ttnn::subtract(x, 1.0f, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::add(
        result, _lgamma(ttnn::subtract(x, 1.5f, std::nullopt, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    result = ttnn::add(result, 3.434189657547f, std::nullopt, output_mem_config);
    return result;
}

Tensor _sinh(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor e_pos_x = ttnn::exp(input_a, false, output_mem_config);
    Tensor e_neg_x = ttnn::exp(ttnn::neg(input_a, output_mem_config), false, output_mem_config);
    Tensor nr_term = ttnn::subtract(e_pos_x, e_neg_x, std::nullopt, output_mem_config);
    e_pos_x.deallocate();
    e_neg_x.deallocate();
    Tensor scalar = ttnn::full_like(input_a, 0.5f);
    return ttnn::multiply(nr_term, scalar, std::nullopt, output_mem_config);
}

// Function: softsign
// Ref: https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
Tensor _softsign(const Tensor& a, const std::optional<MemoryConfig>& output_mem_config) {
     Tensor result =ttnn::multiply(
        a,
        ttnn::reciprocal(ttnn::add(ttnn::abs(a, output_mem_config), 1.0f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
        return result;
}

Tensor _swish(const Tensor& a, const std::optional<MemoryConfig>& output_mem_config) {
    // x / (1.0f + exp(-x))
    return ttnn::silu(a);
}

Tensor _trunc(const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor floor_res = ttnn::floor(input, output_mem_config);
    Tensor trunc_res = where(ttnn::ne(input, floor_res), ttnn::add(floor_res, 1.0f), floor_res);
    Tensor result = where(ttnn::gtz(input, output_mem_config), floor_res, trunc_res);
    return result;
}

Tensor _variance_impl(
    const Tensor& y, const Tensor& mean_y, Tensor& y_minus_mean_y, const std::optional<MemoryConfig>& output_mem_config) {
    constexpr float correction = 0.0f;
    auto shape_wh = y.get_legacy_shape();
    float scale = 1.0f / ((float)(shape_wh[3] * shape_wh[2]) - correction);
    Tensor sqr_y_minus_mean_y = ttnn::square(y_minus_mean_y, output_mem_config);
    Tensor sum_sqr_y_minus_mean_y =
        reduce(sqr_y_minus_mean_y, ReduceOpMath::SUM, ReduceOpDim::HW, scale);
    return sum_sqr_y_minus_mean_y;  // var
}
Tensor _variance_impl(const Tensor& y, const Tensor& mean_y, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor y_minus_mean_y = bcast(y, mean_y, BcastOpMath::SUB, BcastOpDim::HW);
    return _variance_impl(y, mean_y, y_minus_mean_y, output_mem_config);
}

Tensor _variance(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    auto output_memory_config = output_mem_config.value_or(y.memory_config());
    Tensor mean_y = mean_hw(y);
    return _variance_impl(y, mean_y, output_memory_config);
}

// Function std
// compute standard deviation of tensor y = sqrt( E((y-<y>)^2)/ y.volume() )
//  Ref: torch.std
Tensor _std(const Tensor& y, const Tensor& mean_y, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::sqrt(_variance_impl(y, mean_y, output_mem_config));
}

Tensor _std(const Tensor& y, const Tensor& mean_y, Tensor& y_minus_mean_y, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::sqrt(_variance_impl(y, mean_y, y_minus_mean_y, output_mem_config));
}

Tensor _std_overload(const Tensor& y, const std::optional<MemoryConfig>&  output_mem_config) {
    return ttnn::sqrt(_variance(y, output_mem_config));
}

// Function normalize
// use transformation y = (y - mean(y))/std(y) by broadcast
Tensor _normalize(const Tensor& y, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor mean_y = mean_hw(y);
    Tensor y_minus_mean_y = bcast(y, mean_y, BcastOpMath::SUB, BcastOpDim::HW);
    Tensor std_y = _std(y, mean_y, y_minus_mean_y, output_mem_config);
    Tensor recip_std_y = ttnn::reciprocal(std_y, output_mem_config);
    Tensor z = ttnn::multiply(y_minus_mean_y, recip_std_y);
    return z;
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
Tensor _hardsigmoid(const Tensor& a, float value_1, float value_2, const std::optional<MemoryConfig>& output_mem_config) {

   Tensor a_t = ttnn::full_like(a,value_1);
   Tensor b_t = ttnn::full_like(a,value_2);
   Tensor a_mac = mac(a, a_t, b_t);  // multiply and add.
   Tensor a_clip = relu_max(a_mac, 1.0f);
   return a_clip;
}

// Function @hard_swish
// use transformation y = x * hardsigmoid( x ) by broadcast
// Ref: PyTorch
// hard swish(x) = x*hardsigmoid(x,scale,shift)
Tensor _hardswish(const Tensor& a, float value_1, float value_2, const std::optional<MemoryConfig>& output_mem_config) {
   Tensor a_sigmoid = _hardsigmoid(a, value_1, value_2, output_mem_config);
   Tensor result_sq = ttnn::multiply(a_sigmoid, a, std::nullopt);
   return result_sq;
}

// Function Clip
// use clip y = min( max( x, min_value), max_value) by broadcast
// Ref: https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp
Tensor _clip(const Tensor& a, float low, float high, const std::optional<MemoryConfig>& output_mem_config) {
    auto output_memory_config = output_mem_config.value_or(a.memory_config());
    const Tensor h_const = full_like(a, high);
    Tensor a_max = tt::tt_metal::min(a, h_const, output_memory_config);
    if (low == 0.0f) {
        return ttnn::relu(a_max, output_memory_config);
    } else {
        const Tensor l_const = full_like(a, low);
        return tt::tt_metal::max(a_max, l_const, output_memory_config);
    }
}

// clamp
Tensor _clamp(const Tensor& a, float low, float high, const std::optional<MemoryConfig>& output_mem_config) {
    return _clip(a, low, high, output_mem_config);
}

// hardtanh
Tensor _hardtanh(
    const Tensor& a, float low /* = -1.0f */, float high /* = +1.0f */, const std::optional<MemoryConfig>& output_mem_config) {
        auto output_memory_config = output_mem_config.value_or(a.memory_config());
    return _clip(a, low, high, output_memory_config);
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
Tensor _selu(const Tensor& x, const float scale, const float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    // term 2
    Tensor x_Exp = ttnn::exp(x, false, output_mem_config);
    Tensor minus_one = ttnn::operations::creation::create_scalar(-1.0f, x.get_dtype(), Layout::TILE, x.device());
    Tensor x_Exp_minus_1 =ttnn::subtract(x_Exp , minus_one, std::nullopt, output_mem_config);
    x_Exp.deallocate();
    minus_one.deallocate();
    Tensor t_alpha = ttnn::operations::creation::create_scalar(alpha, x.get_dtype(), Layout::TILE, x.device());
    Tensor result_t2_ = ttnn::multiply(x_Exp_minus_1, t_alpha, std::nullopt, output_mem_config);
    x_Exp_minus_1.deallocate();
    t_alpha.deallocate();
    Tensor result_term2 =
        ttnn::multiply(ttnn::gtz(result_t2_, output_mem_config), result_t2_, std::nullopt, output_mem_config);
    result_t2_.deallocate();

    // term 1
    Tensor t_scale = ttnn::operations::creation::create_scalar(scale, x.get_dtype(), Layout::TILE, x.device());
    Tensor x_relu = ttnn::relu(x, output_mem_config);
    Tensor result_term1 = ttnn::multiply(x_relu, t_scale, std::nullopt, output_mem_config);
    t_scale.deallocate();
    x_relu.deallocate();
    Tensor result_selu = ttnn::add(result_term1, result_term2, std::nullopt, output_mem_config);

    return result_selu;
}

// threshold(a,t,v) = (a <= t)*v + (a > t)*a
Tensor _threshold(const Tensor& input_tensor, float threshold, float value, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_threshold = ttnn::operations::creation::create_scalar(
        threshold, input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
    Tensor t0 = ttnn::subtract(input_tensor, t_threshold, std::nullopt, output_mem_config);
    t_threshold.deallocate();
    Tensor t_value =
        ttnn::operations::creation::create_scalar(value, input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
    Tensor t1 = ttnn::multiply(ttnn::lez(t0), t_value, std::nullopt, output_mem_config);
    t_value.deallocate();
    Tensor t2 = ttnn::multiply(ttnn::gtz(t0, output_mem_config), input_tensor, std::nullopt, output_mem_config);
    return ttnn::add(t1, t2, std::nullopt, output_mem_config);
}


std::vector<Tensor> split_tensor_for_glu(const Tensor& input_a, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> t_split;
    Shape inshape(input_a.get_legacy_shape());
    TT_FATAL(((inshape[dim] / 2) % TILE_WIDTH == 0), "Split tensor dimension should be in full tile");
    std::vector<uint32_t> s_a = {0, 0, 0, 0};
    std::vector<uint32_t> e_a = {input_a.get_legacy_shape()[0] - 1, inshape[1] - 1, inshape[2] - 1, inshape[3] / 2 - 1};

    std::vector<uint32_t> s_b = {0, 0, 0, inshape[3] / 2};
    std::vector<uint32_t> e_b = {inshape[0] - 1, inshape[1] - 1, inshape[2] - 1, inshape[3] - 1};

    Tensor t_a = ttnn::slice(0, input_a, s_a, e_a, output_mem_config);
    Tensor t_b = ttnn::slice(0, input_a, s_b, e_b, output_mem_config);

    t_split.emplace_back(t_a);
    t_split.emplace_back(t_b);

    return t_split;
}

// Gated Linear Unit activation: matmul(split[0],sigmoid(split[1]))
 Tensor _glu(const Tensor& input_a, int32_t dim , const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim GLU only supported at this time ");
    if (dim == -1)
        dim = 3;
    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);
    Tensor sigmoid_b = ttnn::sigmoid(ab[1], output_mem_config);
    Tensor glu_result = ttnn::multiply(ab[0], sigmoid_b, std::nullopt, output_mem_config);
    return glu_result;
}

// ReLU Gated Linear Unit activation: matmul(split[0],relu(split[1]))
Tensor _reglu(
    const Tensor& input_a,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim REGLU only supported at this time ");
    if (dim == -1)
        dim = 3;
    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);
    Tensor relu_b = ttnn::relu(ab[1], output_mem_config);
    Tensor reglu_result = ttnn::multiply(ab[0], relu_b, std::nullopt, output_mem_config);
    return reglu_result;
}

// Gaussian Error Gated Linear Unit activation: matmul(split[0],gelu(split[1]))
Tensor _geglu(
    const Tensor& input_a,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config ) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim GEGLU only supported at this time ");
    if (dim == -1)
        dim = 3;

    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);

    constexpr bool fast_appx = true;
    Tensor gelu_b = ttnn::gelu(ab[1], fast_appx, output_mem_config);
    Tensor geglu_result = ttnn::multiply(ab[0], gelu_b, std::nullopt, output_mem_config);
    return geglu_result;
}

// Swish Gated Linear Unit activation: matmul(split[0],swish(split[1]))
Tensor _swiglu(
    const Tensor& input_a,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config ) {
    TT_ASSERT(dim == -1 || dim == 3, "last dim SWIGLU only supported at this time ");
    if (dim == -1)
        dim = 3;

    std::vector<Tensor> ab = split_tensor_for_glu(input_a, dim, output_mem_config);

    Tensor swish_b = _swish(ab[1], output_mem_config);
    Tensor swiglu_result = ttnn::multiply(ab[0], swish_b, std::nullopt, output_mem_config);
    return swiglu_result;
}

// Function: bias gelu
// Ref: http://www.xavierdupre.fr/app/mlprodict/helpsphinx/onnxops/onnx_commicrosoft_BiasGelu.html
Tensor _bias_gelu_unary(const Tensor& a, float bias, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::gelu(ttnn::add(a, bias), true, output_mem_config);
}

}  // namespace ttnn::operations::unary
