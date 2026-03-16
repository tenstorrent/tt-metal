// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel stub for layer_norm_rm.
// Reads RM input sticks, optional gamma/beta, generates scaler and epsilon.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp"

void kernel_main() {}
