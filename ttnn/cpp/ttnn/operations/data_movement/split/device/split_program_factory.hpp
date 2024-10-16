// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks split_last_dim_two_chunks_tiled(const Tensor &input_tensor,
                                                                std::vector<Tensor> &output_tensors,
                                                                const MemoryConfig &mem_config);

}
