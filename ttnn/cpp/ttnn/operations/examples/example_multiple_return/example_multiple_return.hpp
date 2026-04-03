
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/example_multiple_return_device_operation.hpp"

namespace ttnn {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using invoke and can be used to call multiple primitive and/or composite operations
std::vector<std::optional<Tensor>> composite_example_multiple_return(
    const Tensor& input_tensor, bool return_output1 = true, bool return_output2 = true);

}  // namespace ttnn
