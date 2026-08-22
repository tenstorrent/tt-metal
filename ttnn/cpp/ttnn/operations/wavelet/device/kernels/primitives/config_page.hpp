// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

namespace ttnn::operations::wavelet::kernels::primitives {

class ConfigWords {
public:
    template <uint32_t WordCount>
    ALWI explicit ConfigWords(const uint32_t (&words)[WordCount]) : words_(words) {}

    ALWI explicit ConfigWords(const uint32_t* words) : words_(words) {}

    [[nodiscard]] ALWI uint32_t operator[](const uint32_t index) const { return words_[index]; }

private:
    const uint32_t* words_;
};

template <typename Accessor>
ALWI void load_config_page(
    const Accessor& accessor,
    const uint32_t address,
    const uint32_t page_bytes,
    const uint32_t page_index,
    const uint32_t cb,
    uint32_t* words,
    const uint32_t word_count) {
    const auto pages = TensorAccessor(accessor, address, page_bytes);
    CircularBuffer config_buffer(cb);
    Noc noc;

    config_buffer.reserve_back(1);
    noc.async_read(pages, config_buffer, page_bytes, {.page_id = page_index}, {});
    noc.async_read_barrier();
    config_buffer.push_back(1);
    config_buffer.wait_front(1);
    const auto* loaded = reinterpret_cast<const uint32_t*>(config_buffer.get_read_ptr());
    for (uint32_t word = 0; word < word_count; ++word) {
        words[word] = loaded[word];
    }
    config_buffer.pop_front(1);
}

}  // namespace ttnn::operations::wavelet::kernels::primitives
