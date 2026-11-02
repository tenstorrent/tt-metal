// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc reader (NCRISC) — regimes R0, R3 and R4/R5: the window / halo fetch and im2col gather
// (conv2d_nhwc_gather.hpp). With the split gather (R3) it takes the even m-blocks; the writer the odd.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#define GATHER_CT_BASE 0
#include "conv2d_nhwc_gather.hpp"

void kernel_main() { conv_gather::gather_m_blocks(0, 0); }
