// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// An interleaved DRAM buffer addressed in pages of `page` bytes (page p in bank p % banks), for the block's own DRAM
// words and rows (plans, counts, routing tables, x / y rows). A TensorAccessor over an interleaved DRAM spec: no
// host-side accessor args, and the page size is the caller's, which may differ from the buffer's allocation page.
#pragma once

#include <cstdint>
#include "api/tensor/tensor_accessor.h"

using DramPages = TensorAccessor<decltype(tensor_accessor::make_interleaved_dspec</*is_dram=*/true>())>;

FORCE_INLINE DramPages dram_pages(uint32_t base, uint32_t page) {
    return DramPages(tensor_accessor::make_interleaved_dspec</*is_dram=*/true>(), base, page);
}
