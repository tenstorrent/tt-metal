// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "api/scratchpad.h"

void kernel_main() {
    constexpr uint32_t N = get_arg(args::N);
    constexpr uint32_t input_cb_page_size = get_arg(args::page_size);
    constexpr uint32_t num_rows = get_arg(args::num_rows);
    constexpr uint32_t x_dim = get_arg(args::x_dim);
    constexpr uint32_t num_blocks_total = get_arg(args::num_blocks_total);
    constexpr uint32_t x_blocks = get_arg(args::x_blocks);
    constexpr uint32_t w_blocks = get_arg(args::w_blocks);
    constexpr uint32_t x_block_size = get_arg(args::x_block_size);
    constexpr uint32_t w_block_size = get_arg(args::w_block_size);
    constexpr uint32_t element_size = get_arg(args::element_size);
    constexpr uint32_t read_alignment = get_arg(args::read_alignment);
    // The bounce scratchpad is bound only when this is set, hence get_token_if_present below.
    constexpr bool bounce_misaligned_rows = static_cast<bool>(get_arg(args::bounce_misaligned_rows));
    constexpr uint32_t read_alignment_minus_one = read_alignment - 1;

    // Precomputed constants: size of a 32 element block along the W dimension (measured in bytes)
    constexpr uint32_t w_block_size_bytes = w_block_size * element_size;

    // One aligned slot per row of an X block, each with room for the row at any offset below read_alignment.
    // Must match the scratchpad size in permute_rm_program_factory.cpp.
    constexpr uint32_t bounce_slot_bytes =
        (input_cb_page_size + read_alignment_minus_one + read_alignment_minus_one) & ~read_alignment_minus_one;
    [[maybe_unused]] uint32_t bounce_window = 0;
    if constexpr (bounce_misaligned_rows) {
        static_assert(x_block_size <= 32, "bounced_rows tracks one X block in a 32-bit mask");
        const Scratchpad<uint32_t> bounce(*scratch::get_token_if_present<"bounce">());
        const uint32_t bounce_base =
            static_cast<uint32_t>(local_mem::detail::AddressAccess::address(bounce.local_mem()));
        bounce_window = (bounce_base + read_alignment_minus_one) & ~read_alignment_minus_one;
    }

    uint32_t start_block = get_arg(args::start_block);
    uint32_t end_block = get_arg(args::end_block);

    // Input shape and strides (excluding W dimension and measured in rows, not bytes).
    // Rank-length arrays (count = N, a CTA) delivered as runtime varargs:
    // input_shape in varargs [0, N), src_strides in [N, 2N).
    uint32_t input_shape[N], src_strides[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_vararg(i);
        src_strides[i] = get_vararg(i + N);
    }

    /**
     * We have a multidimensional tensor:
     * - num_blocks_total = (rows * x_blocks * w_blocks) where rows = num_rows / X
     *   Here, 'rows' represent the combination of all rows before and after X dimension.
     *   So: rows * X * W_dimension = total number of elements (conceptually).
     *
     * For each 'block':
     *   - Compute which w_block and x_block this corresponds to.
     *   - Then compute which row set (xw_block) we are in.
     */

    // x_dim is the dimension along which we are reading the tensor, as it's the new W dimension in the output tensor
    uint32_t X = input_shape[x_dim];
    uint32_t X_stride = src_strides[x_dim];

    const auto s0 = TensorAccessor(tensor::input);
    constexpr bool fast_path = get_arg(args::fast_path) != 0 && decltype(s0)::DSpec::is_interleaved;
    Noc noc;

    uint32_t idxs[N];
    idxs[N - 1] = 0;
    uint32_t non_x_rows = num_rows / X;

    // One DataflowBuffer for the whole kernel: on Quasar a DataflowBuffer drains when it goes out of scope (#60848),
    // so a per-block one would wait after every push for the compute to pop that block (no double buffering).
    DataflowBuffer dfb(dfb::cb_in);

    for (uint32_t block = start_block; block < end_block; ++block) {
        // Decompose block into w_block, x_block, and xw_block indices
        uint32_t rem = block;
        const uint32_t w_block = rem % w_blocks;  // Which W block are we in?
        rem /= w_blocks;

        const uint32_t x_block = rem % x_blocks;  // Which X block?
        rem /= x_blocks;

        uint32_t xw_block = rem % (non_x_rows);  // Which row set (beyond X dimension)?
        uint32_t remainder = xw_block;

        // Compute X block boundaries
        uint32_t x_start = x_block * x_block_size;
        uint32_t x_end = std::min(x_start + x_block_size, X);

        // Compute W block boundaries
        uint32_t w_start = w_block * w_block_size;
        uint32_t w_end = std::min(w_start + w_block_size, input_shape[N - 1]);
        uint32_t w_offset = w_start * element_size;

        uint32_t w_read_size_bytes = (w_end - w_start) * element_size;

        // Map linear index i to multidimensional indices idxs[]
        // We skip x_dim when doing this mapping and set it separately later
        for (int32_t d = N - 2; d >= 0; --d) {  // Exclude W dimension
            if (d == (int32_t)x_dim) {
                idxs[d] = 0;  // Initialize x_dim to zero (will be set in inner loop)
                continue;     // Skip x_dim during mapping
            }
            idxs[d] = remainder % input_shape[d];
            remainder /= input_shape[d];
        }
        idxs[N - 1] = 0;  // Initialize W dimension index to zero if not already set

        // Precompute the base address offset (excluding x_dim)
        uint64_t base_addr_offset = 0;
        for (uint32_t d = 0; d < N; ++d) {
            if (d != x_dim) {
                base_addr_offset += idxs[d] * src_strides[d];
            }
        }

        // Reserve space in the circular buffer for the X-block length
        dfb.reserve_back(x_block_size);
        uint32_t src_buffer_l1_addr = dfb.get_write_ptr();

        // We read in 'x_block_len' chunks along the X dimension
        uint32_t page_offset = 0;
        // Bit k set: row k was read into its bounce slot at offset bounce_offsets[k] and still needs moving.
        [[maybe_unused]] uint32_t bounced_rows = 0;
        [[maybe_unused]] uint8_t bounce_offsets[x_block_size];
        // Read along the X dimension
        for (uint32_t x = x_start; x < x_end; ++x) {
            uint64_t addr_offset = base_addr_offset + x * X_stride;
            if constexpr (bounce_misaligned_rows) {
                const uint32_t l1_dst = src_buffer_l1_addr + page_offset;
                const uint32_t src_offset =
                    (uint32_t)s0.get_noc_addr(static_cast<uint32_t>(addr_offset), w_offset) & read_alignment_minus_one;
                if (src_offset != (l1_dst & read_alignment_minus_one)) {
                    const uint32_t k = x - x_start;
                    noc.async_read(
                        s0,
                        CoreLocalMem<uint32_t>(bounce_window + k * bounce_slot_bytes + src_offset),
                        w_read_size_bytes,
                        {.page_id = static_cast<uint32_t>(addr_offset), .offset_bytes = w_offset},
                        {});
                    bounced_rows |= 1u << k;
                    bounce_offsets[k] = static_cast<uint8_t>(src_offset);
                    page_offset += input_cb_page_size;
                    continue;
                }
            }
            if constexpr (fast_path) {
                noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
                    s0,
                    CoreLocalMem<uint32_t>(src_buffer_l1_addr + page_offset),
                    w_read_size_bytes,
                    {.page_id = static_cast<uint32_t>(addr_offset), .offset_bytes = w_offset},
                    {});
            } else {
                tt::data_movement::common::noc_async_read_sharded(
                    noc, src_buffer_l1_addr + page_offset, s0, addr_offset, w_offset, w_read_size_bytes);
            }

            // Advance output pointer by one page size for next row
            page_offset += input_cb_page_size;
        }
        // Wait for all async reads to complete before proceeding
        noc.async_read_barrier();
        if constexpr (bounce_misaligned_rows) {
            if (bounced_rows != 0) {
                for (uint32_t k = 0; k < x_end - x_start; ++k) {
                    if ((bounced_rows >> k) & 1u) {
                        const uint32_t l1_dst = src_buffer_l1_addr + k * input_cb_page_size;
                        const uint32_t bounce_src = bounce_window + k * bounce_slot_bytes + bounce_offsets[k];
#ifdef ARCH_QUASAR
                        tt::data_movement::common::copy_via_memmove<false>(l1_dst, bounce_src, w_read_size_bytes);
#else
                        tt::data_movement::common::tt_memmove<false, true, false, w_block_size_bytes>(
                            noc, l1_dst, bounce_src, w_read_size_bytes);
#endif
                    }
                }
                noc.async_write_barrier();
            }
        }
        // Push the filled block into the circular buffer
        dfb.push_back(x_block_size);
    }
}
