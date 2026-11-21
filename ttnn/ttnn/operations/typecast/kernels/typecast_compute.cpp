// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// typecast compute (convert_block): NEO t of C converts this cluster's tiles t, t+C, ... (its slice of the
// STRIDED rings) in blocks of `block_tiles`: unpack (to DEST for a 32-bit input, through SrcA for bfloat16)
// into the 32-bit DEST, then pack in the output ring's format. One helper call over the NEO's whole share,
// synchronised per block (wait/pop and reserve/push once per block; the partial last block syncs its valid
// tiles). The Typecast SFPU step is derived from the two rings' formats (a no-op for the float pairs).
//
// DataFormatReconfig::Disabled on both sides: one input and one output ring, whose formats
// compute_kernel_hw_startup sets once.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t block_tiles = get_arg(args::block_tiles);
    // The chain clamps a block to one DEST section and syncs the rings per section; the host sizes the rings as
    // whole blocks, so a larger block would let a sync window straddle a ring's end. Refuse it at build time.
    static_assert(
        block_tiles >= 1 && block_tiles <= ckl::DEST_AUTO_LIMIT, "typecast: block_tiles must fit one DEST section");
    const uint32_t t = get_my_thread_id();
    const uint32_t neo_tiles = num_tiles / C + (t < num_tiles % C ? 1u : 0u);

    DataflowBuffer cb_input_tiles(dfb::cb_input_tiles);
    DataflowBuffer cb_output_tiles(dfb::cb_output_tiles);
    compute_kernel_hw_startup(dfb::cb_input_tiles, dfb::cb_output_tiles);

    ckl::typecast<
        ckl::input(
            dfb::cb_input_tiles,
            ckl::WaitPolicy::PerBlockSize,
            ckl::PopPolicy::PerBlockSize,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Disabled),
        ckl::output(
            dfb::cb_output_tiles,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>(
        ckl::IterationShape::tiles(neo_tiles).block_size(block_tiles), cb_input_tiles, cb_output_tiles);
}
