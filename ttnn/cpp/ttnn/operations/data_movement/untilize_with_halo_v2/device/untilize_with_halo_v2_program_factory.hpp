// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks untilize_with_halo_multi_core_v2(Program& program,
                                                                 const Tensor& input_tensor,
                                                                 const uint32_t pad_val,
                                                                 const uint32_t ncores_nhw,
                                                                 const uint32_t max_out_nsticks_per_core,
                                                                 const Tensor& padding_config,
                                                                 const Tensor& local_config,
                                                                 const Tensor& remote_config,
                                                                 const bool remote_read,
                                                                 const bool transpose_mcast,
                                                                 Tensor& output_tensor);

}  // namespace ttnn::operations::data_movement::detail
