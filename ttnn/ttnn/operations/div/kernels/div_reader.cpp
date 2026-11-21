// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// div reader (load_block): reader hart p of R fills the input rings with this cluster's tiles p, p+R, ...
// A STRIDED ring deals its entries round-robin over the NEOs, and R | C, so hart p always serves the same
// NEOs. Implicit sync: one tagged NoC read per ring entry, the DM0 interrupt handler posts the credits, no
// reserve / barrier / push. The divisor ring and tensor exist only when the divisor is a tensor
// (DIVISOR_IS_NUMBER == 0); in the same-shape regime page i of b is page i of a.

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

    DataflowBuffer dfb_dividend(dfb::in_a);
    Noc noc;
    const auto acc_dividend = TensorAccessor(tensor::a);
#if !DIVISOR_IS_NUMBER
    DataflowBuffer dfb_divisor(dfb::in_b);
    const auto acc_divisor = TensorAccessor(tensor::b);
#endif

    for (uint32_t i = get_my_thread_id(); i < num_tiles; i += R) {
        noc.async_read<NocOptions::TXN_ID>(acc_dividend, dfb_dividend, {.page_id = start_id + i}, {});
#if !DIVISOR_IS_NUMBER
        noc.async_read<NocOptions::TXN_ID>(acc_divisor, dfb_divisor, {.page_id = start_id + i}, {});
#endif
    }
    // Firmware credits a last partial batch and drains the rings when the kernel returns.
    // Hand the hart's tag register back with the user id: it is sticky, and a program that runs next with
    // untagged transfers (tt-metal's own, e.g. ttnn.to_layout) would otherwise have them counted for a ring.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
