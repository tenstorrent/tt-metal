// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// add reader (load_block): this cluster's pages [start_id, start_id + num_tiles) of a and b into their rings.
//
// Implicit sync (the default protocol of a NoC-filled ring on Quasar): one read tagged NocOptions::TXN_ID per
// page lands at the ring's cursor and the DM0 interrupt handler posts its credit, so there is no reserve,
// barrier or push, and the hart keeps as many reads in flight as the ring has free entries (one block-sized
// handshake per block on the compute side is all that paces it).
// Thread plan: reader hart r of R reads pages r, r+R, ...; the STRIDED rings deal them to the NEOs.
//
// No dataflow helper covers a tensor-page -> ring transfer (op_design.md, "Helpers considered and rejected").

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);
    constexpr uint32_t R = get_arg(args::reader_threads);

    DataflowBuffer cb_input_a_tiles(dfb::cb_input_a_tiles);
    DataflowBuffer cb_input_b_tiles(dfb::cb_input_b_tiles);
    Noc noc;
    const auto acc_a = TensorAccessor(tensor::input_a);
    const auto acc_b = TensorAccessor(tensor::input_b);

    for (uint32_t i = get_my_thread_id(); i < num_tiles; i += R) {
        noc.async_read<NocOptions::TXN_ID>(acc_a, cb_input_a_tiles, {.page_id = start_id + i}, {});
        noc.async_read<NocOptions::TXN_ID>(acc_b, cb_input_b_tiles, {.page_id = start_id + i}, {});
    }
    // No final barrier: firmware credits the last partial batch and drains the rings when the kernel returns.
    // Hand the hart back with the user id in its (sticky) tag register.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
