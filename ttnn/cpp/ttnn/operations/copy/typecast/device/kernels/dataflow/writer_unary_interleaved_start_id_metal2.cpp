// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp.
// Identical dataflow logic; only the resource plumbing moves to the Metal 2.0 named bindings: the
// output CB index compile-time arg becomes dfb::out, the destination tensor becomes tensor::output
// (so the dst_addr runtime arg and the TensorAccessorArgs compile-time args both disappear), and the
// page count / start page become named runtime args. The OUT_SHARDED / BACKWARDS variants are kept
// so this fork stays diffable against its legacy twin (typecast defines neither). Forked rather than
// converted in place because the legacy file is instantiated by ~70 factories that are all still on
// the legacy positional-arg API; delete this fork once the eltwise/unary family adopts the rewrite.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "api/debug/dprint.h"  // [TC-DBG #51270] remove after typecast multi-tile isolation

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    Noc noc;
    // dfb::out — the result pages produced upstream, drained to the output tensor here
    DataflowBuffer dfb(dfb::out);

    // Get page size from the DFB (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = dfb.get_entry_size();

#ifdef OUT_SHARDED
    dfb.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(tensor::output);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb.wait_front(onepage);
        // [TC-DBG #51270] output page ready; print the first 4 fp32 datums as RAW uint32 bits. Compare
        // against the reader's TC_RD for the same id: for bf16->fp32 each fp32 bit-pattern must equal the
        // matching bf16 bits << 16 (i.e. bf16bits * 65536, low 16 bits zero). tile 0 should hold; the
        // suspect is tile 1 (the 2nd tile of the single-core loop) being wrong on Quasar.
        const uint32_t dbg_rd_addr = dfb.get_read_ptr();
        volatile tt_l1_ptr uint32_t* dbg_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbg_rd_addr);
        DPRINT("TC_WR id={} fp32bits= {} {} {} {}\n", i, dbg_p[0], dbg_p[1], dbg_p[2], dbg_p[3]);
        noc.async_write(dfb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        dfb.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}
