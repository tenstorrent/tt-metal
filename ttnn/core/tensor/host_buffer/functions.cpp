// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/tensor/host_buffer/functions.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

namespace tt::tt_metal::host_buffer {

// get_host_buffer(const HostTensor&) is now inline in tt-metalium/experimental/tensor/tensor_utils.hpp

HostBuffer get_host_buffer(const Tensor& tensor) {
    TT_FATAL(is_cpu_tensor(tensor), "Tensor must have HostStorage");
    return get_host_buffer(tensor.host_tensor());
}

}  // namespace tt::tt_metal::host_buffer
