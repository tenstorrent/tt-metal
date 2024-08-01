// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#pragma once

namespace ttnn {

/*
    We have 2 software command queues available to make it possible to overlap some work and reduce latency.
    For example, Op2 can be prepared in a different queue while the first queue is blocked waiting for data readout by Op1
    TT-NN operations allows to specify which queue should be used.
    The default Queue is 0 and possible values are 0 and 1.
*/
constexpr uint8_t DefaultQueueId = 0;

}
