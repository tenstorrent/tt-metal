// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement::detail {

// start is inclusive, end is exclusive
struct PageRange {
    uint32_t start;
    uint32_t end;
};

struct Stride {
    CoreCoord core;
    uint32_t data;
};

struct PageStride {
    CoreCoord start_core;
    uint32_t start_data;
    uint32_t stride_size;  // number of pages per stride
    Stride stride;
    uint32_t num_strides;
    bool skip;
};

struct CorePageRange {
    CoreCoord core;
    PageRange range;
};

struct CorePageStride {
    CoreCoord core;
    PageStride page_stride;
};

tt::tt_metal::operation::ProgramWithCallbacks reshard_multi_core(const Tensor& input, Tensor& output);

}  // namespace ttnn::operations::data_movement::detail
