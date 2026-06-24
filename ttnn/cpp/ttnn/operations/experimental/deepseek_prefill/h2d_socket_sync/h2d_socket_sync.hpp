// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {
class H2DStreamService;
}

namespace ttnn::experimental {

// Wait for the next H2DStreamService transfer to land in the service's backing
// tensor, copy it into a freshly-allocated device tensor, and ack the service
// core.

// Returns a vector: [tokens] when `metadata_size_bytes == 0`, or
// [tokens, metadata] when > 0 (the metadata tensor is [1,1,1,N/4] uint32 DRAM).
// The Python wrapper unpacks this to a single Tensor or a (tokens, metadata)
// tuple to preserve the existing call contract.
std::vector<Tensor> h2d_socket_sync(const tt::tt_metal::H2DStreamService& service, uint32_t metadata_size_bytes = 0);

}  // namespace ttnn::experimental
