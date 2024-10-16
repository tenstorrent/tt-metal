// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"

namespace tt::tt_metal {

template <size_t N>
std::tuple<std::array<CB, N>, CBHandle> create_cb(const CB (&cbs)[N],
                                                  Program &program,
                                                  const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
                                                  uint32_t page_size,
                                                  uint32_t num_pages,
                                                  const tt::DataFormat data_format,
                                                  Buffer *buffer = nullptr) {
    std::map<uint8_t, tt::DataFormat> data_format_spec = {};
    for (auto cb : cbs) {
        data_format_spec[cb] = data_format;
    }

    auto cb_config = CircularBufferConfig(num_pages * page_size, data_format_spec);
    for (auto cb : cbs) {
        cb_config.set_page_size(cb, page_size);
    }

    if (buffer != nullptr) {
        cb_config.set_globally_allocated_address(*buffer);
    }

    std::array<CB, N> cbs_out;
    std::copy(cbs, cbs + N, cbs_out.begin());
    return std::make_tuple(cbs_out, tt_metal::CreateCircularBuffer(program, core_spec, cb_config));
}

inline std::tuple<CB, CBHandle> create_cb(CB cb,
                                          Program &program,
                                          const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
                                          uint32_t page_size,
                                          uint32_t num_pages,
                                          const tt::DataFormat data_format,
                                          Buffer *buffer = nullptr) {
    CB cbs[] = {cb};
    auto [_, handle] = create_cb(cbs, program, core_spec, page_size, num_pages, data_format, buffer);
    return std::make_tuple(cb, handle);
}

}  // namespace tt::tt_metal
