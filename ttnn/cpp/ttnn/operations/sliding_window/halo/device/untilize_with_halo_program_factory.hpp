// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks untilize_with_halo_multi_core(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    uint32_t pad_val,
    uint32_t ncores_nhw,
    uint32_t max_out_nsticks_per_core,
    const Tensor& padding_config0,
    const Tensor& padding_config1,
    const Tensor& gather_config0,
    const Tensor& gather_config1,
    const std::vector<uint16_t>& number_of_blocks_per_core,
    bool remote_read,
    bool transpose_mcast,
    Tensor& output_tensor,
    int block_size,
    bool capture_buffers);  // Used by halo op to cache internally created config buffers with the program
                            // Untilize with Halo takes them as inputs from the user, so doesn't capture

}  // namespace ttnn::operations::data_movement::detail
