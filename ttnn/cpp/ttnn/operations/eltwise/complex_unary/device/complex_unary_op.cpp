// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "third_party/magic_enum/magic_enum.hpp"

#include "ttnn/experimental/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/experimental/tt_dnn/op_library/complex/complex_ops.hpp"

namespace ttnn::operations::complex_unary {

Tensor _real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return input[0];
}

Tensor _imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return input[1];
}

Tensor _angle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ttnn::neg( atan2(input[1],input[0],output_mem_config), output_mem_config );
}

Tensor _is_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ttnn::eqz( input[0], output_mem_config);
}

Tensor _is_real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ttnn::eqz( input[1], output_mem_config);
}

Tensor _abs(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return tt::tt_metal::hypot(input[0],input[1],output_mem_config);
}

ComplexTensor _conj(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ComplexTensor({input[0], ttnn::neg(input[1],output_mem_config)});
}

}  // namespace ttnn::operations::complex_unary
