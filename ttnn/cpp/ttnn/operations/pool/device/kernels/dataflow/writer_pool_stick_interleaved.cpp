// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "experimental/kernel_args.h"
#include "pool_stick_writer_impl.hpp"

template <uint32_t output_stick_size, uint32_t ntiles_c>
TT_KERNEL void writer_pool_stick_interleaved(uint32_t num_sticks, uint32_t start_stick_id) {
    const auto output = TensorAccessor(tensor::output);
    write_pool_sticks<output_stick_size, ntiles_c>(output, dfb::output, num_sticks, start_stick_id);
}
