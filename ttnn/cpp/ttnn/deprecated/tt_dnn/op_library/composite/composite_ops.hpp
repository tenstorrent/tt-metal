// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

#include "tt_metal/common/constants.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/eltwise/binary/device/binary_device_operation.hpp"

namespace tt {

namespace tt_metal {

using unary_tensor_op_t = Tensor(const Tensor& a);
using binary_tensor_op_t = Tensor(const Tensor& a, const Tensor& b);

// Note: inline doesn't allow pybind to work well so we keep few function not inlined.


}  // namespace tt_metal

}  // namespace tt
