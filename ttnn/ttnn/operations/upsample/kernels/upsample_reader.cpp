// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// upsample reader: block operation `load_block` (op_design.md, Block schedule).
//
// No kernel-library helper moves interleaved row-major pages into a DFB entry at a per-stick offset (the
// design's "Helpers considered and rejected" table): raw Device-2.0 NoC + DFB API, as in the Quasar example
// reader (explicit sync).
//
// This cluster owns input sticks [start_stick, start_stick + core_sticks), cut into blocks of block_sticks
// sticks; block b is one ring entry. Reader thread t of R takes blocks t, t+R, ...; on the STRIDED ring its
// k-th push lands in slot t + k*R, which is where writer thread (b mod W) looks for block b.
//
// Explicit sync (the entry is assembled from block_extent reads): reserve, reads, barrier, push. Every
// transfer carries the user transaction id 0, and the tag is handed back as the last statement.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_stick = get_arg(args::start_stick);
    const uint32_t core_sticks = get_arg(args::core_sticks);
    // The plan's count, not get_num_threads(): were a thread not to run, its blocks would be missing.
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t block_sticks = get_arg(args::block_sticks);
    constexpr uint32_t stick_bytes = get_arg(args::stick_bytes);
    constexpr uint32_t stick_stride = get_arg(args::stick_stride);
    constexpr uint32_t thread_mask = get_arg(args::reader_thread_mask);  // test probe; all ones by default
    constexpr NocOptVals user_txn = {.trid = 0};

    DataflowBuffer ring(dfb::cb_input_sticks);
    Noc noc;
    const auto acc_in = TensorAccessor(tensor::input);

    const uint32_t thread_id = get_my_thread_id();
    const bool moves_bytes = (thread_mask >> thread_id) & 1u;
    const uint32_t num_blocks = (core_sticks + block_sticks - 1) / block_sticks;

    for (uint32_t block = thread_id; block < num_blocks; block += R) {
        const uint32_t first = block * block_sticks;
        const uint32_t remaining = core_sticks - first;
        const uint32_t extent = remaining < block_sticks ? remaining : block_sticks;  // ragged last block
        const uint32_t page0 = start_stick + first;

        ring.reserve_back(1);
        if (moves_bytes) {
            for (uint32_t j = 0; j < extent; ++j) {
                noc.async_read<NocOptions::TXN_ID>(
                    acc_in, ring, stick_bytes, {.page_id = page0 + j}, {.offset_bytes = j * stick_stride}, user_txn);
            }
            noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
        }
        ring.push_back(1);
    }
    noc_async_read_set_trid(0, noc.get_noc_id());
}
