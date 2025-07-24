// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t valid_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t valid_Lt = get_compile_time_arg_val(7);
    constexpr uint32_t padded_Nqt = get_compile_time_arg_val(8);
    constexpr uint32_t padded_Nkt = get_compile_time_arg_val(9);
    constexpr uint32_t padded_Lqt = get_compile_time_arg_val(10);
    constexpr uint32_t padded_Lkt = get_compile_time_arg_val(11);
    constexpr uint32_t unpadded_N = get_compile_time_arg_val(12);
    constexpr uint32_t unpadded_L = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores = get_compile_time_arg_val(14);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(15);
    constexpr uint32_t scale_val = get_compile_time_arg_val(16);
    constexpr bool use_joint_mask = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t mask_chunk_0 = get_compile_time_arg_val(18);
    constexpr uint32_t mask_chunk_1 = get_compile_time_arg_val(19);

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t joint_out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_end = get_arg_val<uint32_t>(argidx++);

    constexpr bool is_dram = true;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr DataFormat data_format = get_dataformat(cb_out);

    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<is_dram> joint_out_writer = {
        .bank_base_address = joint_out_addr, .page_size = tile_bytes, .data_format = data_format};

    const auto output_tile_logical = TensorTileShape(B, NH, valid_Nt, DHt);
    const auto joint_tile_logical = TensorTileShape(B, NH, valid_Lt, DHt);
    const auto cat_out_generator =
        CatAddrGenerator(out_writer, output_tile_logical, padded_Nqt, joint_out_writer, joint_tile_logical, padded_Lqt);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;

    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            for (uint32_t q_chunk = local_q_start; q_chunk < local_q_end; ++q_chunk) {
                if constexpr (use_joint_mask) {
                    /*
                    If `use_joint_mask`, then one or both of input tensors are padded.
                    We already know that input tensors are padded up to Sk_chunk_t.
                    Therefore, for the last K chunk of the first tensor and the last K chunk of the joint tensor,
                    we should generate the vertical mask.
                    */
                    if (mask_chunk_0 != (uint32_t)(-1)) {
                        generate_noncausal_padded_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, unpadded_N);
                    }
                    if (mask_chunk_1 != (uint32_t)(-1)) {
                        generate_noncausal_padded_mask<cb_mask_in>(Sq_chunk_t, Sk_chunk_t, unpadded_L);
                    }
                }

                const uint32_t out_row_start_tile = q_chunk * Sq_chunk_t;
                const auto dst_slice = Slice(nb, nq, out_row_start_tile, out_row_start_tile + Sq_chunk_t, 0, DHt);

                write_block(cat_out_generator, dst_slice, cb_out, tile_bytes, barrier_threshold);
            }
        }
    }
}
