// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {

/**
 * Identity op for onboarding: returns a new tensor with the same data as the input.
 *
 * The op exercises the full classic dataflow path (reader -> compute -> writer
 * kernels) and logs the entire host-side call trace; the compute kernel DPRINTs
 * "Hello, world!" from every core the workload is placed on. See the Python
 * docstring (hello_world_nanobind.cpp) for how to enable both.
 */
ttnn::Tensor hello_world(const Tensor& input_tensor);

}  // namespace ttnn::operations::experimental
