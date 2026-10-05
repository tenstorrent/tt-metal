// Copyright (c) 2020-2024 Tenstorrent
//
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "accumulation_compute.hpp"

#include <math.h>

#include <cstdint>

void accumulate_block(uint32_t rt0, uint32_t rt1, uint32_t rt2, uint32_t rt3, float* out, float* in, int32_t block_size) {
    float acc = out[0];
    float c = 0.0f;
    for (int32_t i = 0; i < block_size; i++) {
        float y = in[i] - c;
        float t = acc + y;
        c = (t - acc) - y;
        acc = t;
        out[i] = acc;
    }
}

void accumulate_block_compensated(uint32_t rt0, uint32_t rt1, uint32_t rt2, uint32_t rt3, float* out, float* in, int32_t block_size) {
    float acc = out[0];
    float c = 0.0f;
    for (int32_t i = 0; i < block_size; i++) {
        float y = in[i] - c;
        float t = acc + y;
        // Guard against non-finite running total: if acc is inf or nan,
        // compensation calculation would produce nan (inf - inf = nan),
        // poisoning subsequent elements. When t is non-finite, skip
        // compensation update and propagate the IEEE-compliant result.
        if (std::isfinite(t)) {
            c = (t - acc) - y;
        } else {
            c = 0.0f;
        }
        acc = t;
        out[i] = acc;
    }
}

void accumulate_block_with_initial(uint32_t rt0, uint32_t rt1, uint32_t rt2, uint32_t rt3, float* out, float* in, int32_t block_size) {
    float acc = out[0];
    float c = 0.0f;
    for (int32_t i = 0; i < block_size; i++) {
        float y = in[i] - c;
        float t = acc + y;
        c = (t - acc) - y;
        acc = t;
        out[i] = acc;
    }
}

void accumulate_block_with_initial_compensated(uint32_t rt0, uint32_t rt1, uint32_t rt2, uint32_t rt3, float* out, float* in, int32_t block_size) {
    float acc = out[0];
    float c = 0.0f;
    for (int32_t i = 0; i < block_size; i++) {
        float y = in[i] - c;
        float t = acc + y;
        // Guard against non-finite running total: if t is non-finite,
        // skip compensation update to avoid NaN propagation.
        if (std::isfinite(t)) {
            c = (t - acc) - y;
        } else {
            c = 0.0f;
        }
        acc = t;
        out[i] = acc;
    }
}
