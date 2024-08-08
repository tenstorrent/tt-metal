// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/composite/composite_ops.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/auto_format.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
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

Tensor _rfloor_div(float value, const Tensor& input, const MemoryConfig& output_mem_config) {
    Tensor result = ttnn::multiply(ttnn::full_like(input, value), ttnn::reciprocal(input));
    return ttnn::floor(result, output_mem_config);
}
Tensor rfloor_div(float value, const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _rfloor_div)(value, input, output_mem_config);
}

}  // namespace tt_metal

}  // namespace tt
