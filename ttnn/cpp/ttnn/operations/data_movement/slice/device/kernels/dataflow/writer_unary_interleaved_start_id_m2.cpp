// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 (ProgramSpec) port of writer_unary_interleaved_start_id.cpp.
// Used only by SliceTileSpecProgramFactory, so it is a local fork of the legacy writer
// (the legacy file is still consumed unchanged by SliceTileProgramFactory::create_descriptor,
// which ccl/mesh_partition reuses). Logic, #ifdefs, loop bounds and numeric paths are
// UNCHANGED; only the access mechanism moves to named bindings:
//   dst address  -> ta::dst (TensorAccessor)
//   CB id        -> dfb::cb_out
//   num_pages/start_id positional RTAs -> get_arg(args::...)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t cb_id_out = dfb::cb_out;

    // Create objects for Device 2.0 API
    CircularBuffer cb_out(cb_id_out);

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = cb_out.get_tile_size();
    Noc noc;

#ifdef OUT_SHARDED
    cb_out.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(ta::dst);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_out.wait_front(onepage);
        noc.async_write(cb_out, s, page_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_writes_flushed();
        cb_out.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}
