// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::detail {

using namespace tt::constants;

operation::ProgramWithCallbacks plusone_single_core(const Tensor &input);

}  // namespace ttnn::operations::experimental::detail
