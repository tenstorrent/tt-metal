// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
class D2DStreamServiceReceiver;
}  // namespace ttnn

namespace tt::tt_metal {
class H2DStreamService;
using D2DStreamServiceReceiver = ttnn::D2DStreamServiceReceiver;
}  // namespace tt::tt_metal

namespace ttnn::experimental {

// Wait for the next H2DStreamService transfer to land in the service's backing
// tensor, copy it into a freshly-allocated device tensor, and ack the service
// core.

// Returns a vector, in this order:
//   [0] tokens   -- always; the arriving row minus its trailing `overhang_size_bytes`
//   [1] overhang -- only when `overhang_size_bytes > 0`; those trailing bytes as their
//                   own tensor, i.e. the backing spec with the last dim cut in two
//   [2] metadata -- only when `metadata_size_bytes > 0`; [1,1,1,N/4] uint32 DRAM
// Both splits happen in the one copy the op already performs, so asking for the
// overhang separately costs no extra pass over the data.
std::vector<Tensor> inbound_socket_service_sync(
    const tt::tt_metal::H2DStreamService& service, uint32_t metadata_size_bytes = 0, uint32_t overhang_size_bytes = 0);

// Same op, but draining a D2DStreamServiceReceiver (disaggregated-prefill
// device->device path). The receiver exposes the same getters as
// H2DStreamService. Same return contract.
std::vector<Tensor> inbound_socket_service_sync(
    const ttnn::D2DStreamServiceReceiver& service, uint32_t metadata_size_bytes = 0, uint32_t overhang_size_bytes = 0);

}  // namespace ttnn::experimental
