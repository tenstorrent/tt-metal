// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/async_runtime.hpp"

#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/distributed/api.hpp"

using namespace tt::tt_metal;

namespace ttnn {

void queue_synchronize(tt::tt_metal::distributed::MeshCommandQueue& cq) { cq.finish(); }

void event_synchronize(const tt::tt_metal::distributed::MeshEvent& event) {
    tt::tt_metal::distributed::EventSynchronize(event);
}

void wait_for_event(
    tt::tt_metal::distributed::MeshCommandQueue& cq, const tt::tt_metal::distributed::MeshEvent& event) {
    cq.enqueue_wait_for_event(event);
}

tt::tt_metal::distributed::MeshEvent record_event(tt::tt_metal::distributed::MeshCommandQueue& cq) {
    return cq.enqueue_record_event();
}
tt::tt_metal::distributed::MeshEvent record_event_to_host(tt::tt_metal::distributed::MeshCommandQueue& cq) {
    return cq.enqueue_record_event_to_host();
}

}  // namespace ttnn
