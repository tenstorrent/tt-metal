// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/auto_format.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/concat/concat_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/copy/copy_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/optimizer/optimizer_ops.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reduce/reduce_op.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "tt_numpy/functions.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/ternary/where.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"

namespace tt {

namespace tt_metal {

Tensor mk_zero_tensor_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt) {
    const DataType& dtype =
        output_tensor.has_value() ? output_tensor.value().get_dtype() : reference_tensor.get_dtype();
    Tensor zero = ttnn::operations::creation::create_scalar(0.0f, dtype, Layout::TILE, reference_tensor.device());
    return ttnn::multiply(queue_id, reference_tensor, zero, std::nullopt, output_mem_config, output_tensor);
}

Tensor mk_zero_tensor_like(
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt) {
    uint8_t default_queue_id = 0;
    return mk_zero_tensor_like(default_queue_id, reference_tensor, output_mem_config, output_tensor);
}

// TODO: enable zeroes(), ones() and eye() type functions on-device using this type of logic
template <typename T>
Tensor mk_filled_tensor_like(
    const Tensor& reference_tensor,
    T val,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor = std::nullopt,
    uint8_t queue_id = 0) {
    const DataType& dtype =
        output_tensor.has_value() ? output_tensor.value().get_dtype() : reference_tensor.get_dtype();
    Tensor k = ttnn::operations::creation::create_scalar(val, dtype, Layout::TILE, reference_tensor.device());
    Tensor zero_like = mk_zero_tensor_like(reference_tensor, output_mem_config);
    if (output_tensor.has_value()) {
        return ttnn::add(queue_id, zero_like, k, std::nullopt, output_mem_config, output_tensor);
    } else {
        return ttnn::add(queue_id, zero_like, k, std::nullopt, output_mem_config);
    }
}

// compute polyval by Horner's rule
Tensor _polyval(const Tensor& input_tensor, std::vector<float> coeffs, const MemoryConfig& output_mem_config) {
    TT_ASSERT(coeffs.size() != 0 && "coeffs should be 1 or more coefficients");
    if (coeffs.size() == 1) {
        return mk_filled_tensor_like(input_tensor, coeffs[0], output_mem_config);
    }

    Tensor scalar = ttnn::operations::creation::create_scalar(
        coeffs[0], input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
    Tensor result = ttnn::multiply(input_tensor, scalar, std::nullopt, output_mem_config);
    scalar.deallocate();
    for (int idx = 1; idx < coeffs.size() - 1; idx++) {
        Tensor scalar = ttnn::operations::creation::create_scalar(
            coeffs[idx], input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
        result = ttnn::add(result, scalar, std::nullopt, output_mem_config);
        scalar.deallocate();
        result = ttnn::multiply(input_tensor, result, std::nullopt, output_mem_config);
    }
    Tensor last_coeffs = ttnn::operations::creation::create_scalar(
        coeffs.back(), input_tensor.get_dtype(), Layout::TILE, input_tensor.device());
    Tensor final_tensor = ttnn::add(result, last_coeffs, std::nullopt, output_mem_config);
    last_coeffs.deallocate();
    return final_tensor;
}
Tensor polyval(const Tensor& input_tensor, std::vector<float> coeffs, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _polyval)(input_tensor, coeffs, output_mem_config);
}

// Function: MAC
// compute multiply-accumulate: y = a * b + c,  over various 8 combinations of a, b, c
// being a scalar or tensor
Tensor _mac(const Tensor& a, const Tensor& b, const Tensor& c, const MemoryConfig& output_mem_config) {
    bool a_is_scalar = a.intended_volume() == 1;
    bool b_is_scalar = b.intended_volume() == 1;
    bool c_is_scalar = c.intended_volume() == 1;

    if (!a_is_scalar && !b_is_scalar && !c_is_scalar) {
        // all tensors
        return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && !b_is_scalar && c_is_scalar) {
        // a - tensor, b - tensor, c - is scalar
        return ttnn::add(
            ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && b_is_scalar && !c_is_scalar) {
        // a - tensor, b - scalar, c - is tensor
        return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && b_is_scalar && c_is_scalar) {
        // a - tensor, b - scalar, c - is scalar
        return ttnn::add(
            ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && !b_is_scalar && !c_is_scalar) {
        // a - scalar, b - tensor, c - tensor
        return ttnn::add(ttnn::multiply(b, a, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && !b_is_scalar && c_is_scalar) {
        // a - scalar, b - tensor, c - is scalar
        return ttnn::add(
            ttnn::multiply(b, a, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && b_is_scalar && !c_is_scalar) {
        // a - scalar, b - scalar, c - is tensor
        return ttnn::add(
            c, ttnn::multiply(a, b, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    }

    // all scalars
    // a - scalar, b - scalar, c - is scalar
    TT_ASSERT(a_is_scalar && b_is_scalar && c_is_scalar);
    return ttnn::add(ttnn::multiply(a, b), c);
}
Tensor mac(const Tensor& a, const Tensor& b, const Tensor& c, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _mac)(a, b, c, output_mem_config);
}

Tensor _mac_overload(const Tensor& a, float b, float c, const MemoryConfig& output_mem_config) {
    Tensor t_b = ttnn::operations::creation::create_scalar(b, a.get_dtype(), Layout::TILE, a.device());
    Tensor t_c = ttnn::operations::creation::create_scalar(c, a.get_dtype(), Layout::TILE, a.device());
    Tensor return_tensor = mac(a, t_b, t_c, output_mem_config);
    t_b.deallocate();
    t_c.deallocate();
    return return_tensor;
}
Tensor mac(const Tensor& input_a, float b, float c, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _mac_overload)(input_a, b, c, output_mem_config);
}

// lerp(input, end, weight) = start + weight * (end - start)
Tensor _lerp(const Tensor& input_a, const Tensor& input_b, float value, const MemoryConfig& output_mem_config) {
    Tensor t_value =
        ttnn::operations::creation::create_scalar(value, input_a.get_dtype(), Layout::TILE, input_a.device());
    Tensor t_diff = ttnn::subtract(input_b, input_a, std::nullopt, output_mem_config);
    Tensor t_mul = ttnn::multiply(t_diff, t_value, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input_a, t_mul, std::nullopt, output_mem_config);
    return result;
}
Tensor lerp(const Tensor& input_a, const Tensor& input_b, float value, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _lerp)(input_a, input_b, value, output_mem_config);
}

// lerp(input, end, weight) = start + weight * (end - start)
Tensor _lerp_overload(
    const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, const MemoryConfig& output_mem_config) {
    Tensor t_diff = ttnn::multiply(
        ttnn::subtract(input_b, input_a, std::nullopt, output_mem_config), input_c, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input_a, t_diff, std::nullopt, output_mem_config);
    return result;
}
Tensor lerp(
    const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _lerp_overload)(input_a, input_b, input_c, output_mem_config);
}

// ldexp(input,other)=input * (2^other)
Tensor _ldexp(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(input_a, ttnn::exp2(input_b, output_mem_config), std::nullopt, output_mem_config);
    return result;
}

Tensor _unary_rdiv_trunc(
    float value,
    const Tensor& input,
    const MemoryConfig& output_mem_config) {
    auto arch = input.device()->arch();
    TT_FATAL(arch == tt::ARCH::WORMHOLE_B0, "Op is only supported on Wormhole");
    Tensor result = ttnn::multiply(ttnn::full_like(input, value), ttnn::reciprocal(input));
    return ttnn::trunc(result);
}
Tensor unary_rdiv_trunc(
    float value,
    const Tensor& input,
    const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _unary_rdiv_trunc)(value, input, output_mem_config);
}

Tensor is_odd(const Tensor& input, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(input, (1.0f/2.0f));
    Tensor floor_res = ttnn::floor(result);
    return ttnn::ne(result, floor_res);
}

Tensor _rfloor_div(float value, const Tensor& input, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(ttnn::full_like(input, value), ttnn::reciprocal(input));
    return ttnn::floor(result, output_mem_config);
}
Tensor rfloor_div(float value, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _rfloor_div)(value, input, output_mem_config);
}

Tensor _scatter(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    tt::tt_metal::Array4D start_index = {0, 0, 0, 0};
    ttnn::Tensor input_tensor_4D = ttnn::unsqueeze_to_4D(input_a);

    Tensor index = ttnn::pad(0, ones_like(input_tensor_4D, output_mem_config), input_b.get_legacy_shape().to_array_4D(), start_index, 0, false, std::nullopt);
    Tensor temp_a = ttnn::pad(0, input_tensor_4D,input_b.get_legacy_shape().to_array_4D(), start_index, 0, false, std::nullopt);
    return ttnn::where(index, temp_a, input_b, output_mem_config);
}
Tensor scatter(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _scatter)(input_a, input_b, output_mem_config);
}

// on-device tensor creation 0s like @reference_tensor
Tensor zeros_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return mk_zero_tensor_like(reference_tensor, output_mem_config, output_tensor);
}
Tensor zeros_like(
    const Tensor& reference_tensor, const MemoryConfig& output_mem_config, std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return mk_zero_tensor_like(default_queue_id, reference_tensor, output_mem_config, output_tensor);
}

// on-device tensor creation 1s like @reference_tensor
Tensor ones_like(const Tensor& reference_tensor, const MemoryConfig& output_mem_config) {
    return mk_filled_tensor_like(reference_tensor, 1.0f, output_mem_config);
}

// on-device tensor creation with value like @reference_tensor
Tensor full_like(
    const Tensor& reference_tensor,
    float value,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    uint8_t default_queue_id = 0;
    return mk_filled_tensor_like(reference_tensor, value, output_mem_config, output_tensor, default_queue_id);
}
Tensor full_like(
    uint8_t queue_id,
    const Tensor& reference_tensor,
    float value,
    const MemoryConfig& output_mem_config,
    std::optional<Tensor> output_tensor) {
    return mk_filled_tensor_like(reference_tensor, value, output_mem_config, output_tensor, queue_id);
}

// hardtanh
Tensor _hardtanh(
    const Tensor& a, float low /* = -1.0f */, float high /* = +1.0f */, const MemoryConfig& output_mem_config) {
    return ttnn::clip(a, low, high, output_mem_config);
}
Tensor hardtanh(
    const Tensor& a, float low /* = -1.0f */, float high /* = +1.0f */, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _hardtanh)(a, low, high, output_mem_config);
}

// on-device tensor creation 0s with shape
Tensor zeros(
    const Shape shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return tt::numpy::zeros(shape, data_type, layout, device, output_mem_config);
}

Tensor empty(
    const Shape shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return create_device_tensor(shape, data_type, layout, device, output_mem_config);
}

// on-device tensor creation 1s with shape
Tensor ones(
    const Shape shape, DataType data_type, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return tt::numpy::ones(shape, data_type, layout, device, output_mem_config);
}

// on-device tensor creation with shape and filled with value
Tensor full(
    const Shape shape,
    float value,
    DataType data_type,
    Layout layout,
    Device* device,
    const MemoryConfig& output_mem_config) {
    return tt::numpy::full(shape, value, data_type, layout, device, output_mem_config);
}

// on-device with increment
Tensor arange(int32_t start, int32_t end, int32_t step, Device* device, const MemoryConfig& output_mem_config) {
    return tt::numpy::arange<bfloat16>(start, end, step, Layout::ROW_MAJOR, device, output_mem_config);
}

/**
 * outer product = matrix multiply when a = [1,1,N,1] and b = [1,1,1,M]
 * and result is of size [1,1,N,M].
 * - implementation supports any 1D "squeezable tensor" at input operands
 *   by running reshape.
 */
Tensor _outer(Tensor& a, Tensor& b, const MemoryConfig& output_mem_config) {
    const Shape s_a = a.get_legacy_shape();
    const Shape s_b = b.get_legacy_shape();

    auto num_ones = [](const Shape& s) -> uint32_t {
        uint32_t num1s = 0;
        for (uint32_t idx = 0; idx < 4; idx++) num1s += (uint32_t)(s[idx] == 1);
        return num1s;
    };

    // check if 3 dimensions are 1
    TT_ASSERT(!(num_ones(s_a) < 3), "3 dimensions are required to be 1 for use with outer product");
    TT_ASSERT(!(num_ones(s_b) < 3), "3 dimensions are required to be 1 for use with outer product");

    const bool skip_reshape_a = (s_a[0] == 1 && s_a[1] == 1 && s_a[2] >= 1 && s_a[3] == 1);
    const bool skip_reshape_b = (s_b[0] == 1 && s_b[1] == 1 && s_b[2] == 1 && s_b[3] >= 1);

    Tensor a_slim = a;
    Tensor b_slim = b;

    if (!skip_reshape_a) {
        a_slim = reshape(a, 1, 1, a.volume(), 1, output_mem_config);
    }
    if (!skip_reshape_b) {
        b_slim = reshape(b, 1, 1, 1, b.volume(), output_mem_config);
    }
    a_slim = ttnn::to_layout(a_slim, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    b_slim = ttnn::to_layout(b_slim, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
    Device* device = AutoFormat::GetDefaultDevice();
    if (device != nullptr) {
        if (a_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            a_slim = AutoFormat::move_tensor_to_device(a_slim, device);
        }
        if (b_slim.storage_type() != tt::tt_metal::StorageType::DEVICE) {
            b_slim = AutoFormat::move_tensor_to_device(b_slim, device);
        }
    }

    return ttnn::operations::matmul::matmul(
            a_slim,
            b_slim,
            /*bias=*/std::nullopt,
            tt::operations::primary::Matmul{
            /*program_config=*/std::nullopt,
            /*bcast_batch=*/std::nullopt,
            output_mem_config}
            );
}
Tensor outer(Tensor& a, Tensor& b, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _outer)(a, b, output_mem_config);
}

// on-device tensor creation with shape and filled with value
Tensor _sfpu_eps(const Shape shape, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    float value = device->sfpu_eps();
    return tt::numpy::full(shape, value, DataType::BFLOAT16, layout, device, output_mem_config);
}
Tensor sfpu_eps(const Shape shape, Layout layout, Device* device, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _sfpu_eps)(shape, layout, device, output_mem_config);
}

Tensor create_mask(const Tensor& input_a, const MemoryConfig& output_mem_config) {
    auto& padded_shape = input_a.get_legacy_shape();
    auto& unpadded_shape = padded_shape.without_padding();
    if (padded_shape == unpadded_shape)
        return input_a;
    float t_inf = -std::numeric_limits<float>::infinity();
    Tensor masked_input = tt::numpy::mask_padded_input<bfloat16>(padded_shape, unpadded_shape, DataType::BFLOAT16);
    masked_input = ttnn::where(masked_input, input_a, t_inf, output_mem_config);
    return masked_input;
}

}  // namespace tt_metal

}  // namespace tt
