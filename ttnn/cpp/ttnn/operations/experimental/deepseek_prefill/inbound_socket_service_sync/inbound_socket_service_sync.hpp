// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
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
// tensor, copy it into a device tensor, and ack the service core.

// Returns, in order: tokens (the arriving row minus its trailing `overhang_size_bytes`), overhang (when
// `overhang_size_bytes > 0`), and metadata (when `metadata_size_bytes > 0`). Both splits ride the one copy.
//
// `tokens_out` / `metadata_out` are optional caller-owned PERSISTENT
// destinations. Supply them and the op writes into them and allocates nothing,
// which is what makes the dispatch capturable in a ttnn trace: a trace bakes the
// destination address in at capture time and re-patches nothing on replay, so an
// allocate-per-call output would leave every replay writing to a buffer the
// allocator has since reused. Omit them for the eager allocate-per-call path.
//
// They are independent: supplying one and not the other is legal, and is the right
// thing when only that output must outlive the call (e.g. the metadata record is
// consumed by a DOWNSTREAM traced op while the tokens are copied away eagerly). To
// capture THIS op, supply every destination it produces -- one left unset is
// allocated per call and its address cannot survive a replay. The overhang has no
// persistent destination because it is only produced under MTP, which never
// co-executes with trace capture.
// The caller keeps ownership and can hold them for the service's lifetime. The op
// returns handles to the same device buffers -- but not necessarily the same handle
// objects: when the service covers a strict subset of the mesh, launch() re-wraps
// every output through filter_tensor_shards. Compare buffers, not object identity.
//
// `tokens_out` must match the op's FIRST output spec: the service's per-shard spec
// (`service.get_per_shard_spec()`, e.g. via
// `ttnn.allocate_tensor_on_device(spec, mesh_device)`), less the overhang tail when
// `overhang_size_bytes > 0`; `metadata_out` must be
// [1,1,1,metadata_size_bytes/4] uint32 ROW_MAJOR interleaved DRAM. Both are
// validated on every call.
std::vector<Tensor> inbound_socket_service_sync(
    const tt::tt_metal::H2DStreamService& service,
    uint32_t metadata_size_bytes = 0,
    uint32_t overhang_size_bytes = 0,
    const std::optional<Tensor>& tokens_out = std::nullopt,
    const std::optional<Tensor>& metadata_out = std::nullopt);

// Same op, but draining a D2DStreamServiceReceiver (disaggregated-prefill
// device->device path). The receiver exposes the same getters as
// H2DStreamService. Same return contract, and the same optional persistent
// destinations.
std::vector<Tensor> inbound_socket_service_sync(
    const ttnn::D2DStreamServiceReceiver& service,
    uint32_t metadata_size_bytes = 0,
    uint32_t overhang_size_bytes = 0,
    const std::optional<Tensor>& tokens_out = std::nullopt,
    const std::optional<Tensor>& metadata_out = std::nullopt);

}  // namespace ttnn::experimental
