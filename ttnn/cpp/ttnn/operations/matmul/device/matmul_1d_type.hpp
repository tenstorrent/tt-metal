// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::matmul {

enum class Matmul1DType { MCAST_IN0, GATHER_IN0, MCAST_IN1 };

}  // namespace ttnn::operations::matmul
