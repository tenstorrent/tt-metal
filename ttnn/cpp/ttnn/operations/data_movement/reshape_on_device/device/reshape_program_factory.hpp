// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks reshape_tile_single_core(const Tensor &a, Tensor &output, int N, int C, int H, int W);
operation::ProgramWithCallbacks reshape_rm_multi_core(const Tensor &a, Tensor& output, int N, int C, int H, int W);

}
