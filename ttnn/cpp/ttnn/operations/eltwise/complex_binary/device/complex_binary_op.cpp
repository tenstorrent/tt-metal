// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/complex_unary/device/complex_unary_op.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn::operations::complex_binary {

ComplexTensor _add(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ ttnn::add(input_a[0], input_b[0], std::nullopt, output_mem_config),
             ttnn::add(input_a[1], input_b[1], std::nullopt, output_mem_config) });
}

ComplexTensor _sub(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ ttnn::subtract(input_a[0], input_b[0], std::nullopt, output_mem_config),
             ttnn::subtract(input_a[1], input_b[1], std::nullopt, output_mem_config) });
}

ComplexTensor _mul(const ComplexTensor& ab, const ComplexTensor& cd,  const MemoryConfig& output_mem_config) {
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

ComplexTensor _div(const ComplexTensor& input_a, const ComplexTensor& input_b,  const MemoryConfig& output_mem_config) {
    return ttnn::operations::complex_binary::_mul( input_a, ttnn::reciprocal( input_b , output_mem_config ), output_mem_config  );
}

}  // namespace ttnn::operations::complex_binary
