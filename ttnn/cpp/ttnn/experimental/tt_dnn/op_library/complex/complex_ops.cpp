// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/experimental/tt_dnn/op_library/complex/complex_ops.hpp"
#include "ttnn/experimental/tt_dnn/op_library/concat/concat_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/reshape/reshape_op.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "tt_numpy/functions.hpp"
#include "ttnn/experimental/tensor/tensor_utils.hpp"

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/complex_unary/device/complex_unary_op.hpp"

namespace tt {

namespace tt_metal {

//TODO: add profiling hooks

#define CHECK_FOR_COMPLEX(input) do {\
  TT_ASSERT( utility::is_complex_shape(input), "works for complex shape only"); \
  /* TT_ASSERT( input.get_legacy_shape()[0] == 1, "tensor should have batch size 1"); */ \
  } while(0);

Tensor mk_complex(const Tensor& input_r, const Tensor& input_i, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> inputs = {input_r,input_i};
    return concat(inputs, -1, output_mem_config);
}

Tensor get_real(const Tensor& input, const MemoryConfig& output_mem_config) {
    Shape t_Shape = input.get_legacy_shape();
    std::vector<uint32_t> start = {0, 0, 0, 0} ;
    std::vector<uint32_t> end = {t_Shape[0] - 1,t_Shape[1] - 1 ,t_Shape[2] - 1, (t_Shape[3] / 2) - 1};
    Tensor r_tensor = ttnn::slice(0, input, start, end, output_mem_config);
    return r_tensor;
}

Tensor get_imag(const Tensor& input, const MemoryConfig& output_mem_config) {
    Shape t_Shape = input.get_legacy_shape();
    std::vector<uint32_t> start = {0, 0, 0, (t_Shape[3] / 2)};
    std::vector<uint32_t> end = {t_Shape[0] - 1,t_Shape[1] - 1 ,t_Shape[2] - 1, (t_Shape[3] - 1)};
    Tensor i_tensor = ttnn::slice(0, input, start, end, output_mem_config);
    return i_tensor;
}

namespace utility {
    bool is_complex_shape(const Tensor& input) {
        const Shape& shape = input.get_legacy_shape();
        return shape[-1]%(2*TILE_WIDTH) == 0; //last dim should be partitionable
    }
}


Tensor _is_real(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    Tensor real = get_real(input, output_mem_config);
    return ttnn::eqz(real, output_mem_config); //imaginary portion = 0
}
Tensor is_real(const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _is_real)(input, output_mem_config);
}

Tensor is_imag(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    Tensor imag = get_imag(input, output_mem_config);
    return ttnn::eqz(imag, output_mem_config);
}

Tensor real(const Tensor& input, const MemoryConfig& output_mem_config) {
	CHECK_FOR_COMPLEX(input);
    return get_real(input, output_mem_config);
}

Tensor imag(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    return get_imag(input, output_mem_config);
}

Tensor conj(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    Tensor real = get_real(input, output_mem_config);
    Tensor imag = get_imag(input, output_mem_config);
    return mk_complex(real,ttnn::neg(imag, output_mem_config));
}

Tensor complex_abs(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    Tensor real = get_real(input, output_mem_config);
    Tensor imag = get_imag(input, output_mem_config);
    return hypot(real, imag, output_mem_config);
}

Tensor complex_recip(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);

    Tensor real = get_real(input, output_mem_config);
    Tensor imag = get_imag(input, output_mem_config);

    Tensor a_plus_b = ttnn::add(real,imag, std::nullopt, output_mem_config);
    Tensor a_minus_b = ttnn::subtract(real, imag, std::nullopt,output_mem_config);
    Tensor asqr_plus_bsqr = ttnn::add(ttnn::square(real,output_mem_config),ttnn::square(imag,output_mem_config), std::nullopt, output_mem_config);
    Tensor inv_dr = ttnn::reciprocal( asqr_plus_bsqr, output_mem_config );
    Tensor conj_im = ttnn::multiply( ttnn::neg(imag,output_mem_config), inv_dr, std::nullopt, output_mem_config);
    Tensor conj_re = ttnn::multiply( real, inv_dr, std::nullopt, output_mem_config);
    return mk_complex( conj_re, conj_im, output_mem_config );
}

Tensor complex_add(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    return ttnn::add(input_a,input_b, std::nullopt, output_mem_config);
}

Tensor complex_sub(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    return ttnn::subtract(input_a,input_b, std::nullopt, output_mem_config);
}

Tensor complex_mul(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input_a);
    CHECK_FOR_COMPLEX(input_b);

    Tensor r_a = get_real(input_a, output_mem_config);
    Tensor i_a = get_imag(input_a, output_mem_config);

    Tensor r_b = get_real(input_b, output_mem_config);
    Tensor i_b = get_imag(input_b, output_mem_config);

    Tensor re_part = ttnn::subtract(
        ttnn::multiply(r_a, r_b, std::nullopt, output_mem_config),
        ttnn::multiply(i_a, i_b, std::nullopt, output_mem_config),
        std::nullopt, output_mem_config);

    Tensor im_part = ttnn::add(
        ttnn::multiply(r_a,i_b,std::nullopt,output_mem_config),
        ttnn::multiply(i_a,r_b,std::nullopt,output_mem_config),
        std::nullopt, output_mem_config);
    return mk_complex( re_part, im_part, output_mem_config);
}


// z_a/z_b = z_a*recip(z_b) = z_a*conj(z_b)/(z_b*conj(z_b))
Tensor complex_div(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input_a);
    CHECK_FOR_COMPLEX(input_b);
    return complex_mul( input_a, complex_recip( input_b , output_mem_config ), output_mem_config  );
}

// theta = /_x + iy = atan2(y,x)
Tensor angle(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    Tensor real = get_real(input, output_mem_config);
    Tensor imag = get_imag(input, output_mem_config);
    return ttnn::neg( atan2(imag, real, output_mem_config), output_mem_config );
}

#undef CHECK_FOR_COMPLEX

///// type-2 implementation ////
ComplexTensor conj(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ComplexTensor({input[0], ttnn::neg(input[1],output_mem_config)});
}

ComplexTensor complex_mul(const ComplexTensor& ab, const ComplexTensor& cd,  const MemoryConfig& output_mem_config) {
    // (a + ib)*(c + id) = (ac - bd) + i(bc + ad)
    Tensor re_part = ttnn::subtract(
        ttnn::multiply(ab[0],cd[0],std::nullopt,output_mem_config),
        ttnn::multiply(ab[1],cd[1],std::nullopt,output_mem_config),
        std::nullopt, output_mem_config);

    Tensor im_part = ttnn::add(
        ttnn::multiply(ab[0],cd[1],std::nullopt,output_mem_config),
        ttnn::multiply(ab[1],cd[0],std::nullopt,output_mem_config),
        std::nullopt, output_mem_config);

    return ComplexTensor({ re_part, im_part });
}

ComplexTensor complex_div(const ComplexTensor& input_a, const ComplexTensor& input_b,  const MemoryConfig& output_mem_config) {
    return complex_mul( input_a, complex_recip( input_b , output_mem_config ), output_mem_config  );
}

ComplexTensor complex_recip(const ComplexTensor& ab, const MemoryConfig& output_mem_config) {
    Tensor a_plus_b = ttnn::add(ab[0],ab[1],std::nullopt,output_mem_config);
    Tensor a_minus_b = ttnn::subtract(ab[0],ab[1],std::nullopt,output_mem_config);
    Tensor asqr_plus_bsqr = ttnn::add(ttnn::square(ab[0],output_mem_config),ttnn::square(ab[1],output_mem_config),
                                std::nullopt,output_mem_config);
    Tensor inv_dr = ttnn::reciprocal( asqr_plus_bsqr, output_mem_config );
    Tensor conj_im = ttnn::multiply( ttnn::neg(ab[1],output_mem_config), inv_dr, std::nullopt, output_mem_config);
    Tensor conj_re = ttnn::multiply( ab[0], inv_dr, std::nullopt, output_mem_config);
    return ComplexTensor({ conj_re, conj_im});
}

ComplexTensor complex_add(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ ttnn::add(input_a[0], input_b[0], std::nullopt, output_mem_config),
             ttnn::add(input_a[1], input_b[1], std::nullopt, output_mem_config) });
}

ComplexTensor complex_sub(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ ttnn::subtract(input_a[0], input_b[0], std::nullopt, output_mem_config),
             ttnn::subtract(input_a[1], input_b[1], std::nullopt, output_mem_config) });
}

// level-1 type polar
Tensor polar(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor c = ttnn::cos(input_b,output_mem_config);
    Tensor r = ttnn::multiply(input_a, c ,std::nullopt, output_mem_config);
    c.deallocate();

    Tensor s = ttnn::sin(input_b,output_mem_config);
    Tensor i = ttnn::multiply(input_a, s, std::nullopt, output_mem_config);
    s.deallocate();
    return mk_complex( r, i, output_mem_config);
}

ComplexTensor polar(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    const Tensor& input_a = input.real();
    const Tensor& input_b = input.imag();
    Tensor c = ttnn::cos(input_b,output_mem_config);
    Tensor r = ttnn::multiply(input_a,c,std::nullopt,output_mem_config);
    c.deallocate();

    Tensor s = ttnn::sin(input_b,output_mem_config);
    Tensor i = ttnn::multiply(input_a,s,std::nullopt,output_mem_config);
    s.deallocate();

    return ComplexTensor({r,i});
}

}//namespace tt_metal

}//namespace tt
