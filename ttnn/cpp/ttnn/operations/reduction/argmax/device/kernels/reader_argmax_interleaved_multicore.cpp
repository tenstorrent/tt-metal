// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "argmax_common.hpp"
#include "api/numeric/bfloat16.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <cstdint>

/**
 * @brief Finds the argmax (argument of maximum value) for a specific core in a multicore reduction operation.
 *
 * This function performs argmax computation on interleaved data for a single core, supporting various
 * data formats and memory configurations. It processes data in units along inner and outer dimensions
 * and handles both DRAM and SRAM source locations.
 *
 * @tparam src_is_dram Boolean flag indicating whether source data is stored in DRAM (true) or SRAM (false)
 * @tparam reduce_all Boolean flag indicating whether to reduce across all dimensions (true) or specific dimensions
 * (false)
 * @tparam data_format Compile-time data format specification that determines the data type and layout
 *                     (e.g., Float16, BFloat16, Float32, Int32, etc.) used for input values and computations
 *
 * @param inner_dim_units Number of units to process along the inner dimension
 * @param outer_idx Index of the current outer dimension being processed
 * @param src_offset Offset into the source data buffer
 * @param s_src Interleaved address generator for source data access
 * @param src_dfb Dataflow buffer holding the source data
 * @param src_read_size Size of data to read from source in bytes
 * @param red_dim_offset Offset along the reduction dimension
 * @param red_dim_units_this_core Number of reduction dimension units assigned to this core
 * @param in_vals Pointer to L1 memory containing input values with data type determined by data_format template
 * parameter
 */
template <bool reduce_all, DataFormat data_format, typename AddrGen, typename SrcDfb>
inline void find_argmax_for_core(
    const Noc& noc,
    const uint32_t inner_dim_units,
    const uint32_t outer_idx,
    const uint32_t src_offset,
    const AddrGen& s_src,
    const SrcDfb& src_dfb,
    const uint32_t src_read_size,
    const uint32_t red_dim_offset,
    const uint32_t red_dim_units_this_core,
    volatile tt_l1_ptr decltype(get_default_value<data_format>())* in_vals,
    const uint32_t red_dim_units,
    uint32_t& max_idx,
    decltype(get_default_value<data_format>())& max_val,
    volatile tt_l1_ptr uint32_t* red_idxs,
    volatile tt_l1_ptr decltype(get_default_value<data_format>())* red_vals) {
    for (uint32_t j = 0; j < inner_dim_units; ++j) {
        // Cores that sub_core_grids leaves without work skip the empty NoC read but still publish a default partial.
        if (red_dim_units_this_core > 0) {
            noc.async_read(
                s_src,
                src_dfb,
                src_read_size,
                {.page_id = outer_idx * inner_dim_units + j, .offset_bytes = src_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();
        }

        // Reset max_val for each new output
        if constexpr (not reduce_all) {
            max_idx = 0;
            max_val = get_default_value<data_format>();
        }

        for (uint32_t i = red_dim_offset; i < (red_dim_offset + red_dim_units_this_core); ++i) {
            if constexpr (data_format == DataFormat::Float16_b) {
                uint16_t val = in_vals[i - red_dim_offset];
                process_value_comparison<data_format, uint16_t, reduce_all>(
                    val, max_val, max_idx, i, outer_idx, j, inner_dim_units, red_dim_units, [](uint16_t a, uint16_t b) {
                        return bfloat16_greater(a, b);
                    });

            } else if constexpr (data_format == DataFormat::UInt16) {
                uint16_t val = in_vals[i - red_dim_offset];
                process_value_comparison<data_format, uint16_t, reduce_all>(
                    val, max_val, max_idx, i, outer_idx, j, inner_dim_units, red_dim_units, [](uint16_t a, uint16_t b) {
                        return a > b;
                    });

            } else if constexpr (data_format == DataFormat::Float32) {
                uint32_t val = in_vals[i - red_dim_offset];
                process_value_comparison<data_format, uint32_t, reduce_all>(
                    val, max_val, max_idx, i, outer_idx, j, inner_dim_units, red_dim_units, [](uint32_t a, uint32_t b) {
                        return float32_greater(a, b);
                    });

            } else if constexpr (data_format == DataFormat::Int32) {
                int32_t val = in_vals[i - red_dim_offset];
                process_value_comparison<data_format, int32_t, reduce_all>(
                    val, max_val, max_idx, i, outer_idx, j, inner_dim_units, red_dim_units, [](int32_t a, int32_t b) {
                        return int32_greater(a, b);
                    });

            } else if constexpr (data_format == DataFormat::UInt32) {
                uint32_t val = in_vals[i - red_dim_offset];
                process_value_comparison<data_format, uint32_t, reduce_all>(
                    val, max_val, max_idx, i, outer_idx, j, inner_dim_units, red_dim_units, [](uint32_t a, uint32_t b) {
                        return a > b;
                    });

            } else {
                // We need a value-dependent expression (gcc-12) that is not
                // tautologically false (gcc-15)
                static_assert(data_format == DataFormat::Float16_b, "Unsupported data format in find_argmax_for_core");
            }
        }

        if constexpr (not reduce_all) {
            red_idxs[j] = max_idx;
            red_vals[j] = max_val;
        }
    }
}

/**
 * @brief Finds the argument (index) of the maximum value from intermediate reduction outputs across multiple cores.
 *
 * This function performs a final reduction step to find the global argmax by comparing intermediate
 * maximum values computed by different cores. It handles tie-breaking by selecting the smaller index
 * when values are equal.
 *
 * @tparam num_cores Number of cores that participated in the reduction operation
 * @tparam data_format Data format of the values being compared. Supported formats:
 *                     - DataFormat::Float16_b: 16-bit bfloat16 floating point
 *                     - DataFormat::UInt16: 16-bit unsigned integer
 *                     - DataFormat::Float32: 32-bit IEEE 754 floating point
 *                     - DataFormat::Int32: 32-bit signed integer
 *                     - DataFormat::UInt32: 32-bit unsigned integer
 *
 * @param inner_idx Index within each core's output buffer to compare
 * @param red_val_dfb_local_base_addr Base address of the dataflow buffer containing reduction values
 * @param red_idx_dfb_local_base_addr Base address of the dataflow buffer containing reduction indices
 * @param red_val_size_per_core Size in bytes of reduction values buffer per core
 * @param red_idx_size_per_core Size in bytes of reduction indices buffer per core
 *
 * @return The global index of the maximum value across all cores
 *
 * @note The function uses volatile L1 pointers for accessing core-local memory
 * @note Tie-breaking favors the smaller index when multiple cores have the same maximum value
 * @note Compilation will fail with static_assert for unsupported data formats
 */
template <uint32_t num_cores, DataFormat data_format>
inline uint32_t find_argmax_from_intermediate_outputs(
    const uint32_t inner_idx,
    const uint32_t red_val_dfb_local_base_addr,
    const uint32_t red_idx_dfb_local_base_addr,
    const uint32_t red_val_size_per_core,
    const uint32_t red_idx_size_per_core) {
    // Reset max_val for each new output
    auto max_val = get_default_value<data_format>();
    uint32_t max_idx = 0;

    for (uint32_t i = 0; i < num_cores; ++i) {
        volatile tt_l1_ptr auto i_red_idxs =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_idx_dfb_local_base_addr + i * red_idx_size_per_core);

        if constexpr (data_format == DataFormat::Float16_b) {
            volatile tt_l1_ptr auto i_red_vals =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(red_val_dfb_local_base_addr + i * red_val_size_per_core);

            process_core_data<data_format>(
                inner_idx, i_red_vals, i_red_idxs, max_val, max_idx, [](uint16_t a, uint16_t b) {
                    return bfloat16_greater(a, b);
                });

        } else if constexpr (data_format == DataFormat::UInt16) {
            volatile tt_l1_ptr auto i_red_vals =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(red_val_dfb_local_base_addr + i * red_val_size_per_core);

            process_core_data<data_format>(
                inner_idx, i_red_vals, i_red_idxs, max_val, max_idx, [](uint16_t a, uint16_t b) { return a > b; });

        } else if constexpr (data_format == DataFormat::Float32) {
            volatile tt_l1_ptr auto i_red_vals =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_val_dfb_local_base_addr + i * red_val_size_per_core);

            process_core_data<data_format>(
                inner_idx, i_red_vals, i_red_idxs, max_val, max_idx, [](uint32_t a, uint32_t b) {
                    return float32_greater(a, b);
                });

        } else if constexpr (data_format == DataFormat::Int32) {
            volatile tt_l1_ptr auto i_red_vals =
                reinterpret_cast<volatile tt_l1_ptr int32_t*>(red_val_dfb_local_base_addr + i * red_val_size_per_core);

            process_core_data<data_format>(
                inner_idx, i_red_vals, i_red_idxs, max_val, max_idx, [](int32_t a, int32_t b) {
                    return int32_greater(a, b);
                });

        } else if constexpr (data_format == DataFormat::UInt32) {
            volatile tt_l1_ptr auto i_red_vals =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_val_dfb_local_base_addr + i * red_val_size_per_core);

            process_core_data<data_format>(
                inner_idx, i_red_vals, i_red_idxs, max_val, max_idx, [](uint32_t a, uint32_t b) { return a > b; });

        } else {
            // We need a value-dependent expression (gcc-12) that is not
            // tautologically false (gcc-15)
            static_assert(
                data_format == DataFormat::Float16_b,
                "Unsupported data format in find_argmax_from_intermediate_outputs");
        }
    }
    return max_idx;
}

void kernel_main() {
    // Runtime args
    // ------------
    const auto core_id = get_arg(args::core_id);

    // The offset of the input DFB in the input tensor
    const auto src_offset = get_arg(args::src_offset);

    // The offset of the reduction dim in the input DFB
    const auto red_dim_offset = get_arg(args::red_dim_offset);

    // The bytes to read from the input DFB for this core
    const auto src_read_size = get_arg(args::src_read_size);

    // This is the number of elements in the reduction dim processed by this core
    const auto red_dim_units_this_core = get_arg(args::red_dim_units_this_core);

    // Compile time args
    // -----------------
    constexpr auto dst_page_size = get_arg(args::dst_page_size);
    constexpr auto red_idx_size_per_core = get_arg(args::red_idx_size_per_core);
    constexpr auto red_val_size_per_core = get_arg(args::red_val_size_per_core);

    // This is the number of elements in the output, excluding the last two dimensions.
    // i.e. for an input tensor of shape (.., N, C, H, W), this is (.. * N * C)
    // It also depends on the `keepdim`
    constexpr auto outer_dim_units = get_arg(args::outer_dim_units);

    // This is the number of elements in the last dimension of the output
    // i.e. for an input tensor of shape (.., N, C, H, W), this is H.
    // This dictates the page size in the output dfb
    constexpr auto inner_dim_units = get_arg(args::inner_dim_units);

    // This is the number of elements in the input tensor along the reduction dim (W)
    constexpr auto red_dim_units = get_arg(args::red_dim_units);

    // Boolean to indicate if we reduce across _all_ dimensions or just on the reduction dim (last dim)
    constexpr bool reduce_all = (bool)get_arg(args::reduce_all);

    // Total number of cores participating in this op
    constexpr auto num_cores = get_arg(args::num_cores);

    // Pick the core that will collate the intermediate outputs
    constexpr uint32_t reduce_core_id = (bool)get_arg(args::reduce_core_id);

    constexpr auto reduce_core_x = get_arg(args::reduce_core_x);
    constexpr auto reduce_core_y = get_arg(args::reduce_core_y);

    // start and end coordinates of the cores that will be used to compute the intermediate outputs
    // At maximum, there can be two groups of cores (suffix 0 and 1)
    //
    // These four names are the operands of a NOC1 multicast rectangle, which is addressed
    // end-corner first. The host therefore sends the group's *end* coordinate in the "start"
    // argument and its *start* coordinate in the "end" argument, and set_multicast() below
    // consumes them in that order. The inversion is deliberate and load-bearing: swapping these
    // to make the names read naturally would address the wrong rectangle, and multicast has no
    // check that would catch it.
    constexpr auto start_core_x0 = get_arg(args::start_core_x0);
    constexpr auto start_core_y0 = get_arg(args::start_core_y0);
    constexpr auto end_core_x0 = get_arg(args::end_core_x0);
    constexpr auto end_core_y0 = get_arg(args::end_core_y0);

    constexpr auto start_core_x1 = get_arg(args::start_core_x1);
    constexpr auto start_core_y1 = get_arg(args::start_core_y1);
    constexpr auto end_core_x1 = get_arg(args::end_core_x1);
    constexpr auto end_core_y1 = get_arg(args::end_core_y1);

    constexpr auto num_cores0 = get_arg(args::num_cores0);  // Number of cores in group 0
    constexpr auto num_cores1 = get_arg(args::num_cores1);  // Number of cores in group 1

    //-------------------------------------------------------------------------
    // Flag to identify if this core will collate intermediate outputs
    const bool is_reduce_core = (core_id == reduce_core_id);

    const auto s_src = TensorAccessor(tensor::src);
    const auto s_dst = TensorAccessor(tensor::dst);

    Noc noc;
    UnicastEndpoint remote;
    DataflowBuffer src_dfb(dfb::src);
    // This DFB is only used in the reduction core. It is used to store
    // final outputs (indices) after reduction of intermediate outputs.
    DataflowBuffer dst_dfb(dfb::dst);
    // This DFB holds intermediate outputs (indices) in each core
    DataflowBuffer red_idx_dfb(dfb::red_idxs);
    // This DFB holds intermediate outputs (values) in each core
    DataflowBuffer red_val_dfb(dfb::red_vals);

    // DFB in L1 memory for storing input
    constexpr DataFormat src_dfb_addr_data_format = get_dataformat(dfb::src);
    const uint32_t src_dfb_addr = src_dfb.get_write_ptr();
    volatile tt_l1_ptr auto* in_vals = get_tt_l1_ptr_based_on_data_format<src_dfb_addr_data_format>(src_dfb_addr);

    // DFB in L1 memory of reducer core for storing output
    const uint32_t dst_dfb_addr = dst_dfb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* out_idxs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_dfb_addr);

    // DFB in L1 memory for storing partial idx output
    const uint32_t red_idx_dfb_local_base_addr = red_idx_dfb.get_write_ptr();
    const uint32_t red_idx_offset = core_id * red_idx_size_per_core;

    const uint32_t red_idx_dfb_local_addr = red_idx_dfb_local_base_addr + red_idx_offset;
    volatile tt_l1_ptr uint32_t* red_idxs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_idx_dfb_local_addr);

    // DFB in L1 memory for storing partial val output
    const uint32_t red_val_dfb_local_base_addr = red_val_dfb.get_write_ptr();
    constexpr DataFormat red_val_dfb_local_addr_data_format = get_dataformat(dfb::red_vals);
    const uint32_t red_val_offset = core_id * red_val_size_per_core;

    const uint32_t red_val_dfb_local_addr = red_val_dfb_local_base_addr + red_val_offset;
    volatile tt_l1_ptr auto* red_vals =
        get_tt_l1_ptr_based_on_data_format<red_val_dfb_local_addr_data_format>(red_val_dfb_local_addr);

    static_assert(
        red_val_dfb_local_addr_data_format == src_dfb_addr_data_format,
        "Program logic error. "
        "Partial result buffer must use the same data format as values.");

    // Semaphores
    Semaphore start_sem(sem::start);
    Semaphore done_sem(sem::done);

    uint32_t max_idx = 0;
    auto max_val = get_default_value<src_dfb_addr_data_format>();

    // -------------------------------------------------------------------------
    // Main loop - run by all cores
    for (uint32_t k = 0; k < outer_dim_units; ++k) {
        if (is_reduce_core) {
            // done_sem is zero-initialized by the dispatcher before the kernel
            // launches (and restored to zero at the end of this kernel so trace
            // replays see the same state), so the k == 0 iteration needs no
            // reset. Resetting it here
            // at k == 0 would race the worker cores' done_sem.up() increments:
            // those increments are ungated at k == 0 (the start_sem handshake
            // below only orders k > 0), so a reset could clobber an increment that
            // already landed and stall the done_sem.wait(num_cores) below forever.
            // For k > 0 the reset clears the previous iteration's count and is
            // ordered ahead of the increments by the start_sem multicast/wait.
            if (k > 0) {
                done_sem.set(0);
            }
            start_sem.set(k + 1);

            if constexpr (num_cores > 1) {
                if (k > 0) {
                    start_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                        noc, start_core_x0, start_core_y0, end_core_x0, end_core_y0, num_cores0);
                }

                if (num_cores1 > 0) {
                    start_sem.set_multicast(noc, start_core_x1, start_core_y1, end_core_x1, end_core_y1, num_cores1);
                }

                noc.async_write_barrier();
            }
        }

        // Wait to start.  Use wait_min (>=) rather than exact-match: in the
        // reduce_all path there is no per-iteration done_sem back-pressure (it is
        // lifted out of this loop), so the reduce core free-runs and can advance
        // start_sem past (k+1) before a lagging worker samples it.  An exact-match
        // wait(k+1) would then never observe k+1 and deadlock; start_sem is
        // monotonically increasing, so wait_min(k+1) is correct for both paths.
        if (k > 0) {
            start_sem.wait_min(k + 1);
        }

        find_argmax_for_core<reduce_all, src_dfb_addr_data_format>(
            noc,
            inner_dim_units,
            k,
            src_offset,
            s_src,
            src_dfb,
            src_read_size,
            red_dim_offset,
            red_dim_units_this_core,
            in_vals,
            red_dim_units,
            max_idx,
            max_val,
            red_idxs,
            red_vals);

        if constexpr (not reduce_all) {
            // We now write these local values to the equivalent position in the reduction core.
            //
            // A DFB handed to async_write as the source is read at its read pointer. That is the
            // address the partials were written to as well: no DFB in this kernel ever calls
            // reserve_back/push_back/wait_front/pop_front, so every one of them keeps its read and
            // write pointers at the buffer base for the whole kernel.
            if (core_id != reduce_core_id) {
                noc.async_write(
                    red_idx_dfb,
                    remote,
                    red_idx_size_per_core,
                    {.offset_bytes = red_idx_offset},
                    {.noc_x = reduce_core_x, .noc_y = reduce_core_y, .addr = red_idx_dfb_local_addr});
                noc.async_write(
                    red_val_dfb,
                    remote,
                    red_val_size_per_core,
                    {.offset_bytes = red_val_offset},
                    {.noc_x = reduce_core_x, .noc_y = reduce_core_y, .addr = red_val_dfb_local_addr});
                noc.async_write_barrier();
            }

            done_sem.up(noc, reduce_core_x, reduce_core_y, 1);
            noc.async_atomic_barrier();

            // If this is the reducer core, wait for the semaphore to be set
            if (is_reduce_core) {
                if constexpr (num_cores > 1) {
                    done_sem.wait(num_cores);
                }

                // Find argmax from intermediate outputs and write to output
                for (uint32_t j = 0; j < inner_dim_units; ++j) {
                    out_idxs[j] = find_argmax_from_intermediate_outputs<num_cores, src_dfb_addr_data_format>(
                        j,
                        red_val_dfb_local_base_addr,
                        red_idx_dfb_local_base_addr,
                        red_val_size_per_core,
                        red_idx_size_per_core);
                }

                noc.async_write(dst_dfb, s_dst, dst_page_size, {.offset_bytes = 0}, {.page_id = k});
                noc.async_write_barrier();
            }
        }
    }  // for (uint32_t k = 0; k < outer_dim_units; ++k)

    if constexpr (reduce_all) {
        red_idxs[0] = max_idx;
        red_vals[0] = max_val;

        // We now write these local values to the equivalent position in the reduction core
        if (core_id != reduce_core_id) {
            noc.async_write(
                red_idx_dfb,
                remote,
                red_idx_size_per_core,
                {.offset_bytes = red_idx_offset},
                {.noc_x = reduce_core_x, .noc_y = reduce_core_y, .addr = red_idx_dfb_local_addr});
            noc.async_write(
                red_val_dfb,
                remote,
                red_val_size_per_core,
                {.offset_bytes = red_val_offset},
                {.noc_x = reduce_core_x, .noc_y = reduce_core_y, .addr = red_val_dfb_local_addr});
            noc.async_write_barrier();
        }

        done_sem.up(noc, reduce_core_x, reduce_core_y, 1);
        noc.async_atomic_barrier();

        // If this is the reducer core, wait for the semaphore to be set
        if (is_reduce_core) {
            if constexpr (num_cores > 1) {
                done_sem.wait(num_cores);
            }

            out_idxs[0] = find_argmax_from_intermediate_outputs<num_cores, src_dfb_addr_data_format>(
                0,  // For reduce_all, we only have one inner_dim_unit
                red_val_dfb_local_base_addr,
                red_idx_dfb_local_base_addr,
                red_val_size_per_core,
                red_idx_size_per_core);

            noc.async_write(dst_dfb, s_dst, dst_page_size, {.offset_bytes = 0}, {.page_id = 0});
            noc.async_write_barrier();
        }
    }  // if constexpr (reduce_all)

    // Restore both semaphores to their initial (zero) state for the next launch. The k == 0
    // iteration relies on done_sem starting at 0, but that only holds on a fresh enqueue: trace
    // replay does not re-run the dispatcher's semaphore initialization, so without this reset the
    // previous run's final count (num_cores) is still present, the reducer's k == 0 wait passes
    // immediately, and it collates stale partials (observed as argmax indices from the wrong
    // core's slice on the first outer row). These resets are race-free: the final done_sem.wait
    // above guarantees every worker's increment for the last iteration has landed, and workers
    // never read start_sem again after their final wait.
    if (is_reduce_core) {
        done_sem.set(0);
        start_sem.set(0);
        if constexpr (num_cores > 1) {
            start_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
                noc, start_core_x0, start_core_y0, end_core_x0, end_core_y0, num_cores0);
            if (num_cores1 > 0) {
                start_sem.set_multicast(noc, start_core_x1, start_core_y1, end_core_x1, end_core_y1, num_cores1);
            }
            noc.async_write_barrier();
        }
    }
}
