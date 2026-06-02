// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Typecast
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // AddBinary, SubBinary, MulBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"     // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

// Per-stat fused chain computing
//
//   D0 = cb_one - cb_momentum                       // (1 - momentum)
//   D0 = D0 * cb_old_stat                            // (1 - momentum) * old
//   D1 = cb_batch_stat * cb_momentum (via D2)        // momentum * batch
//   D0 = D0 + D1                                     // running result
//   pack D0 -> cb_updated_running_stat               // always
//   [pack D0 -> cb_out0]                             // if AlsoPackToOut
//   [Typecast<TcIn, TcOut>(D0) + pack -> cb_writer]  // if NeedsTypecast
//
// Three DEST slots used (D0, D1, D2); DEST_AUTO_LIMIT is at least 4, so safe.
//
// CB lifecycles:
//   cb_one, cb_momentum   InputLifecycle::CallerManaged + Scalar (held by kernel_main for the
//                                                  whole kernel)
//   cb_old_stat, cb_batch_stat
//                         InputLifecycle::Bulk + Scalar (chain emits 1-tile wait+pop per call
//                                        via window_1d<Scalar>)
//   cb_updated_running_stat
//                         OutputLifecycle::Streaming + Scalar (chain reserves+packs+pushes)
//   cb_out0               OutputLifecycle::CallerManaged on the optional 2nd pack (kernel_main
//                                                  reserves+pushes around the
//                                                  whole per-iter block)
//   cb_writer_updated_stat
//                         OutputLifecycle::Streaming on the typecast-tail pack

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr bool old_running_mean_has_value = get_compile_time_arg_val(0) == 1;
    constexpr bool old_running_var_has_value = get_compile_time_arg_val(1) == 1;

    constexpr auto cb_batch_mean = get_compile_time_arg_val(2);
    constexpr auto cb_batch_var = get_compile_time_arg_val(3);
    constexpr auto cb_out0 = get_compile_time_arg_val(4);
    constexpr auto cb_old_running_mean = get_compile_time_arg_val(5);
    constexpr auto cb_old_running_var = get_compile_time_arg_val(6);
    constexpr auto cb_updated_running_mean = get_compile_time_arg_val(7);
    constexpr auto cb_updated_running_var = get_compile_time_arg_val(8);
    constexpr auto cb_momentum = get_compile_time_arg_val(9);
    constexpr auto cb_one = get_compile_time_arg_val(10);
    // CT-arg slots 11..13 used to be cb_tmp1/2/3 — no longer referenced (the fused chain
    // keeps the running result in DEST). Kept for ABI compatibility with the program factory.
    constexpr auto cb_writer_updated_mean = get_compile_time_arg_val(14);
    constexpr auto cb_writer_updated_var = get_compile_time_arg_val(15);
    constexpr bool stat_needs_typecast = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t tc_in_fmt = get_compile_time_arg_val(17);
    constexpr uint32_t tc_out_fmt = get_compile_time_arg_val(18);
    constexpr bool needs_mean_typecast = old_running_mean_has_value && stat_needs_typecast;
    constexpr bool needs_var_typecast = old_running_var_has_value && stat_needs_typecast;

    // cb_out0 receives the "last computed stat": var if both, else mean.
    constexpr bool mean_packs_to_out0 = old_running_mean_has_value && !old_running_var_has_value;
    constexpr bool var_packs_to_out0 = old_running_var_has_value;

    CircularBuffer cb_batch_mean_obj(cb_batch_mean);
    CircularBuffer cb_out0_obj(cb_out0);

    compute_kernel_hw_startup(cb_batch_mean, cb_momentum, cb_out0);

    cb_wait_front(cb_momentum, 1);
    cb_wait_front(cb_one, 1);

    constexpr uint32_t onetile = 1;

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // cb_batch_mean is wait/popped per iter regardless of branch (original
        // contract — the reader pushes one cb_batch_mean tile per output tile).
        cb_batch_mean_obj.wait_front(onetile);
        cb_out0_obj.reserve_back(onetile);

        if constexpr (old_running_mean_has_value) {
            // updated_running_mean = (1 - momentum) * old_running_mean
            //                      + momentum       * batch_mean
            //
            // Two chain shapes — the only difference is whether we mirror D0 to
            // cb_out0. Done via constexpr-if (not OptionalChainElement) because
            // wrapping a PackTile in OptionalChainElement<false, …> breaks
            // chain_pack_writes_collide's SFINAE probe.
            if constexpr (mean_packs_to_out0) {
                eltwise_chain(
                    onetile,
                    CopyTile<
                        cb_one,
                        Dst::D0,
                        InputLifecycle::CallerManaged,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    CopyTile<
                        cb_momentum,
                        Dst::D1,
                        InputLifecycle::CallerManaged,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
                    CopyTile<
                        cb_old_running_mean,
                        Dst::D1,
                        InputLifecycle::Bulk,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                    CopyTile<
                        cb_batch_mean,
                        Dst::D1,
                        InputLifecycle::CallerManaged,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    CopyTile<
                        cb_momentum,
                        Dst::D2,
                        InputLifecycle::CallerManaged,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    MulBinary<Dst::D1, Dst::D2, Dst::D1>{},
                    AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
                    PackTile<cb_updated_running_mean, OutputLifecycle::Streaming, PackTileReconfig::Output, Dst::D0>{},
                    PackTile<cb_out0, OutputLifecycle::CallerManaged, PackTileReconfig::Output, Dst::D0>{});
            } else {
                eltwise_chain(
                    onetile,
                    CopyTile<
                        cb_one,
                        Dst::D0,
                        InputLifecycle::CallerManaged,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    CopyTile<
                        cb_momentum,
                        Dst::D1,
                        InputLifecycle::CallerManaged,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
                    CopyTile<
                        cb_old_running_mean,
                        Dst::D1,
                        InputLifecycle::Bulk,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                    CopyTile<
                        cb_batch_mean,
                        Dst::D1,
                        InputLifecycle::CallerManaged,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    CopyTile<
                        cb_momentum,
                        Dst::D2,
                        InputLifecycle::CallerManaged,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    MulBinary<Dst::D1, Dst::D2, Dst::D1>{},
                    AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
                    PackTile<cb_updated_running_mean, OutputLifecycle::Streaming, PackTileReconfig::Output>{});
            }

            if constexpr (needs_mean_typecast) {
                eltwise_chain(
                    onetile,
                    CopyTile<
                        cb_updated_running_mean,
                        Dst::D0,
                        InputLifecycle::Bulk,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    Typecast<tc_in_fmt, tc_out_fmt, Dst::D0>{},
                    PackTile<cb_writer_updated_mean, OutputLifecycle::Streaming, PackTileReconfig::Output>{});
            }
        }

        cb_batch_mean_obj.pop_front(onetile);

        if constexpr (old_running_var_has_value) {
            // updated_running_var = (1 - momentum) * old_running_var
            //                     + momentum       * batch_var
            // var_packs_to_out0 is always true when this branch runs, so cb_out0
            // gets the var result.
            eltwise_chain(
                onetile,
                CopyTile<
                    cb_one,
                    Dst::D0,
                    InputLifecycle::CallerManaged,
                    OperandKind::Scalar,
                    CopyTileReconfig::Input>{},
                CopyTile<
                    cb_momentum,
                    Dst::D1,
                    InputLifecycle::CallerManaged,
                    OperandKind::Scalar,
                    CopyTileReconfig::Input>{},
                SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
                CopyTile<
                    cb_old_running_var,
                    Dst::D1,
                    InputLifecycle::Bulk,
                    OperandKind::Scalar,
                    CopyTileReconfig::Input>{},
                MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
                CopyTile<cb_batch_var, Dst::D1, InputLifecycle::Bulk, OperandKind::Scalar, CopyTileReconfig::Input>{},
                CopyTile<
                    cb_momentum,
                    Dst::D2,
                    InputLifecycle::CallerManaged,
                    OperandKind::Scalar,
                    CopyTileReconfig::Input>{},
                MulBinary<Dst::D1, Dst::D2, Dst::D1>{},
                AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
                PackTile<cb_updated_running_var, OutputLifecycle::Streaming, PackTileReconfig::Output, Dst::D0>{},
                PackTile<cb_out0, OutputLifecycle::CallerManaged, PackTileReconfig::Output, Dst::D0>{});

            if constexpr (needs_var_typecast) {
                eltwise_chain(
                    onetile,
                    CopyTile<
                        cb_updated_running_var,
                        Dst::D0,
                        InputLifecycle::Bulk,
                        OperandKind::Scalar,
                        CopyTileReconfig::Input>{},
                    Typecast<tc_in_fmt, tc_out_fmt, Dst::D0>{},
                    PackTile<cb_writer_updated_var, OutputLifecycle::Streaming, PackTileReconfig::Output>{});
            }
        }

        cb_out0_obj.push_back(1);
    }

    cb_pop_front(cb_momentum, 1);
    cb_pop_front(cb_one, 1);
}
