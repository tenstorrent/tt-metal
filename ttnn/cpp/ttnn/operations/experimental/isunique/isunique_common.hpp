// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "isunique_cbs.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <map>

namespace ttnn::operations::experimental::isunique::common {

using namespace tt;
using namespace tt_metal;

struct OptimalHeuristic {
    double total_cost;
    uint64_t input_volume;
    uint32_t num_rows;
    uint32_t row_size;
    uint32_t num_cores;
    // full_like accepts ints as the only integral type
    int fill_value = -1;
};

constexpr int32_t FIRST_DIMENSION = 0;

constexpr uint32_t BAD_MASK = -1;

static const std::map<IsUniqueCB, DataType> PREDEFINED_TENSOR_DTYPES{
    {IsUniqueCB::INDEX_HINT, DataType::INT32},
    {IsUniqueCB::FIRST_OCCURRENCES, DataType::INT32},
    {IsUniqueCB::OUTPUT, DataType::UINT8}};

}  // namespace ttnn::operations::experimental::isunique::common
