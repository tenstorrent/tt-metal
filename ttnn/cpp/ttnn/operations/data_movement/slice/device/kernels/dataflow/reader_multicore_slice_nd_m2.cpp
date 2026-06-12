// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 slice rm-stride nd reader. Logic identical to reader_multicore_slice_nd.cpp
// (that one stays for the legacy/descriptor path); bindings are Metal 2.0:
//   - CB index      -> dfb::cb_out (this kernel is the DFB producer)
//   - rank/element  -> named CTAs  (in the cache key)
//   - input accessor -> ta::src    (address implicit; standard accessor)
//   - num_rows/start_row -> named RTAs
//   - per-dim arrays (input/output dims, slice start/end/step) -> RTA varargs,
//     copied into local arrays (rank is compile-time, so arrays are fixed-size)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_id_out = dfb::cb_out;
    constexpr uint32_t tensor_rank = get_named_compile_time_arg_val("rank");
    constexpr uint32_t element_size = get_named_compile_time_arg_val("element_size");

    const uint32_t num_rows_for_this_core = get_arg(args::num_rows);
    const uint32_t start_row_for_this_core = get_arg(args::start_row);

    // Per-dim arrays as runtime varargs, in the order the factory appends them:
    // [input_dims, output_dims, slice_starts, slice_ends, slice_steps], each `tensor_rank` long.
    uint32_t input_dims[tensor_rank];
    uint32_t output_dims[tensor_rank];
    uint32_t slice_starts[tensor_rank];
    uint32_t slice_ends[tensor_rank];
    uint32_t slice_steps[tensor_rank];
    for (uint32_t i = 0; i < tensor_rank; ++i) {
        input_dims[i] = get_vararg(i);
        output_dims[i] = get_vararg(tensor_rank + i);
        slice_starts[i] = get_vararg(2 * tensor_rank + i);
        slice_ends[i] = get_vararg(3 * tensor_rank + i);
        slice_steps[i] = get_vararg(4 * tensor_rank + i);
    }

    const uint32_t input_bytes_per_row = input_dims[tensor_rank - 1] * element_size;

    const auto s0 = TensorAccessor(ta::src);
    Noc noc;
    CircularBuffer cb_out(cb_id_out);

    uint32_t coords[16];
    for (uint32_t i = 0; i < tensor_rank; ++i) {
        coords[i] = slice_starts[i];
    }

    uint32_t rows_processed = 0;
    uint32_t current_logical_row = 0;
    bool found_start = false;
    bool more_iterations = true;

    while (more_iterations && rows_processed < num_rows_for_this_core) {
        bool should_process_row = false;
        if (!found_start) {
            if (current_logical_row == start_row_for_this_core) {
                found_start = true;
                should_process_row = true;
            }
        } else {
            should_process_row = true;
        }

        if (should_process_row && rows_processed < num_rows_for_this_core) {
            uint32_t input_row_idx = 0;
            if (tensor_rank > 1) {
                for (int32_t dim = 0; dim < (int32_t)(tensor_rank - 1); ++dim) {
                    uint32_t stride = 1;
                    for (int32_t d = dim + 1; d < (int32_t)(tensor_rank - 1); ++d) {
                        stride *= input_dims[d];
                    }
                    input_row_idx += coords[dim] * stride;
                }
            }

            cb_out.reserve_back(1);
            uint32_t l1_write_addr = cb_out.get_write_ptr();

            noc.async_read(
                s0, cb_out, input_bytes_per_row, {.page_id = input_row_idx, .offset_bytes = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();

            uint32_t last_dim = tensor_rank - 1;
            if (slice_starts[last_dim] != 0 || slice_steps[last_dim] != 1 ||
                slice_ends[last_dim] != input_dims[last_dim]) {
                volatile tt_l1_ptr uint8_t* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);
                volatile tt_l1_ptr uint8_t* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);

                uint32_t out_col = 0;
                for (uint32_t input_col = slice_starts[last_dim];
                     input_col < slice_ends[last_dim] && out_col < output_dims[last_dim];
                     input_col += slice_steps[last_dim]) {
                    for (uint32_t byte_idx = 0; byte_idx < element_size; ++byte_idx) {
                        dst_ptr[out_col * element_size + byte_idx] = src_ptr[input_col * element_size + byte_idx];
                    }
                    out_col++;
                }
            }

            cb_out.push_back(1);
            rows_processed++;
        }

        current_logical_row++;

        bool carry = true;
        for (int32_t dim = (int32_t)(tensor_rank - 2); dim >= 0 && carry; --dim) {
            coords[dim] += slice_steps[dim];
            if (coords[dim] < slice_ends[dim]) {
                carry = false;
            } else {
                coords[dim] = slice_starts[dim];
            }
        }
        if (carry) {
            more_iterations = false;
        }

        if (tensor_rank == 1) {
            break;
        }
    }
}
