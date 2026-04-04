// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_unary_op.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include <tt_stl/assert.hpp>

namespace ttnn::operations::complex_unary {

Tensor _real(const ComplexTensor& input, const MemoryConfig& /*output_mem_config*/) { return input[0]; }

Tensor _imag(const ComplexTensor& input, const MemoryConfig& /*output_mem_config*/) { return input[1]; }

Tensor _angle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return atan2(input[1], input[0], output_mem_config);
}

Tensor _is_imag(const ComplexTensor& /*input*/, const MemoryConfig& /*output_mem_config*/) {
    TT_THROW("eqz operation not yet reimplemented after nuke");
}

Tensor _is_real(const ComplexTensor& /*input*/, const MemoryConfig& /*output_mem_config*/) {
    TT_THROW("eqz operation not yet reimplemented after nuke");
}

ComplexTensor _conj(const ComplexTensor& /*input*/, const MemoryConfig& /*output_mem_config*/) {
    TT_THROW("neg operation not yet reimplemented after nuke");
}

ComplexTensor _polar(const ComplexTensor& /*input*/, const MemoryConfig& /*output_mem_config*/) {
    TT_THROW("cos/sin operations not yet reimplemented after nuke");
}

}  // namespace ttnn::operations::complex_unary
