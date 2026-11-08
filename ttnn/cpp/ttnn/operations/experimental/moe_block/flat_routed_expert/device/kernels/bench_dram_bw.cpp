// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microbenchmark: DRAM read bandwidth. Reads N pages (page FIRST + i x STRIDE) of an interleaved DRAM tensor into a
// scratch CB ring, DEPTH reads in flight (transaction ids 1..DEPTH, as the flat expert's weight reads).
// RT: 0 tensor address, 1 N, 2 FIRST, 3 STRIDE, 4 ring pages; CT: 0 scratch CB, 1 DEPTH, 2 page bytes, 3.. accessor
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0), depth = get_compile_time_arg_val(1);
    constexpr uint32_t pbytes = get_compile_time_arg_val(2);
    constexpr auto ta = TensorAccessorArgs<3>();
    const uint32_t n = get_arg_val<uint32_t>(1), first = get_arg_val<uint32_t>(2), stride = get_arg_val<uint32_t>(3);
    const uint32_t ring = get_arg_val<uint32_t>(4);
    const auto t = TensorAccessor(ta, get_arg_val<uint32_t>(0), pbytes);
#ifdef BW_HALF  // (the second reader of a core: the upper half of the scratch ring)
    const uint32_t dst = get_write_ptr(cb) + ring * pbytes;
#else
    const uint32_t dst = get_write_ptr(cb);
#endif
    uint32_t iss = 0, done = 0;
    while (done < n) {
        while (iss < n && iss - done < depth) {
            noc_async_read_set_trid(1 + iss % depth);
            noc_async_read(t.get_noc_addr(first + iss * stride), dst + (iss % ring) * pbytes, pbytes);
            ++iss;
        }
        while (done < iss && !noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
        }
        while (done < iss && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + done % depth)) {
            ++done;
        }
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
}
