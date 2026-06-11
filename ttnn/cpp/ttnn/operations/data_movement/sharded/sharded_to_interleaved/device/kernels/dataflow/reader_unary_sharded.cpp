// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/tensor_accessor.h"

// Metal 2.0 (sharded_to_interleaved private copy): the sharded input lives in an L1 buffer that backs the
// input DFB (dfb::in0, borrowed_from the "input" tensor parameter). The reader simply publishes the
// already-resident shard to the DFB's consumer (the writer, or the compute kernel when converting dtype).
// The CB id now comes from the DFB binding token (dfb::in0) instead of a positional compile-time arg.
//
// The ta::input accessor is bound on this kernel purely to satisfy the ProgramSpec referential-integrity
// check for the borrowed_from input DFB (a borrowed tensor counts as "bound" only if some kernel has a
// TensorAccessor for it). It is unused in the kernel body — the shard is reached through dfb::in0.
void kernel_main() {
    const uint32_t num_units_per_core = get_arg(args::num_units_per_core);

    constexpr uint32_t cb_id_in0 = dfb::in0;

    [[maybe_unused]] const auto input = TensorAccessor(ta::input);

    CircularBuffer cb(cb_id_in0);
    cb.push_back(num_units_per_core);
}
