// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp"
#include "tt_eager/tt_dnn/op_library/complex/complex_ops.hpp"

namespace ttnn::operations::complex_binary {

ComplexTensor _add(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ ttnn::add(input_a[0], input_b[0], std::nullopt, output_mem_config),
             ttnn::add(input_a[1], input_b[1], std::nullopt, output_mem_config) });
}

ComplexTensor _sub(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ ttnn::subtract(input_a[0], input_b[0], std::nullopt, output_mem_config),
             ttnn::subtract(input_a[1], input_b[1], std::nullopt, output_mem_config) });
}

}  // namespace ttnn::operations::complex_binary
