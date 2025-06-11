// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::data_movement::detail {

tt::tt_metal::operation::ProgramWithCallbacks untilize_with_halo_multi_core(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t max_out_nsticks_per_core,
    const Tensor& padding_config,
    const Tensor& gather_config0,
    const Tensor& gather_config1,
    const std::vector<uint16_t>& number_of_blocks_per_core,
    const bool remote_read,
    const bool transpose_mcast,
    Tensor& output_tensor,
    const int block_size,
    const bool capture_buffers);  // Used by halo op to cache internally created config buffers with the program
                                  // Untilize with Halo takes them as inputs from the user, so doesn't capture

tt::tt_metal::operation::ProgramWithCallbacks inplace_untilize_with_halo_multi_core(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const uint32_t pad_val,
    const bool padding_exists,
    const uint32_t ncores_nhw,
    const uint32_t ncores_c,
    const uint32_t max_out_nsticks_per_core,
    const uint32_t max_ref_size,
    const Tensor& padding_config,
    const Tensor& local_config,
    const Tensor& remote_config,
    const bool remote_read,
    const bool transpose_mcast,
    Tensor& output_tensor,
    const bool capture_buffers);

}  // namespace ttnn::operations::data_movement::detail
