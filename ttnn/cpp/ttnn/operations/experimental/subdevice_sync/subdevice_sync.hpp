// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <ttnn/distributed/types.hpp>

namespace ttnn::operations::experimental {

// In-trace cross-sub-device hand-off (GLM prefill pipeline E2). Both ops enqueue a ONE-core program on every chip
// of the mesh (chip-local; no cross-chip traffic); both are trace-capturable and program-cached.
//
// subdevice_signal: issue right after the producer op(s) on the producer's sub-device. Its go signal is sent only
//   after every earlier program on `subdevice_id` has completed, so its atomic `semaphore[consumer_core] += increment`
//   means "producer done". `core` (on the producer's sub-device) runs the kernel; `consumer_core` holds the semaphore word.
// subdevice_wait: issue right before the consumer op(s) on the consumer's sub-device. Spins on its own L1 word
//   until semaphore[core] >= value, then atomically subtracts value. The next program on `subdevice_id`
//   (the consumer) starts only after this program completes.
//
// Issue-order rule (single CQ; deadlock otherwise): the matching subdevice_signal must be enqueued BEFORE the next
// program on the waiter's sub-device after the subdevice_wait. Go signals of one CQ are sent in issue order and a
// go for sub-device X waits for X's previous program, so a later-issued signal could never be released.
// The mesh device is the semaphore's device.
void subdevice_signal(
    const tt::tt_metal::GlobalSemaphore& semaphore,
    tt::tt_metal::SubDeviceId subdevice_id,
    const CoreCoord& core,
    const CoreCoord& consumer_core,
    uint32_t increment = 1);

void subdevice_wait(
    const tt::tt_metal::GlobalSemaphore& semaphore,
    tt::tt_metal::SubDeviceId subdevice_id,
    const CoreCoord& core,
    uint32_t value = 1,
    uint32_t timeout_cycles = 0);

}  // namespace ttnn::operations::experimental
