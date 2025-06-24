
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;
    // binary_op_init_common(cb_in, cb_scaler, cb_intermediate);
    reconfig_data_format(cb_in, cb_scaler);
    pack_reconfig_data_format(cb_intermediate);
    mul_tiles_bcast_scalar_init_short(cb_in, cb_scaler);
                // UNPACK(DPRINT <<"0" << ENDL());
                // UNPACK(print_full_tile(tt::CBIndex::c_24, 0, true));
    for (uint32_t tile = 0; tile < cb_length; tile += num_dst_regs) {
        tile_regs_acquire();
        uint32_t blk = tile + num_dst_regs > cb_length ? cb_length - tile : num_dst_regs;
        cb_wait_front(cb_in, blk);
        DPRINT << "blk: " << blk << ENDL();
    UNPACK(DPRINT <<"0" << ENDL());
    UNPACK(print_full_tile(tt::CBIndex::c_24, 0, true));
        for (uint32_t wtr = 0; wtr < blk; wtr++) {
                // UNPACK(print_full_tile(cb_in, wtr, true ));
            mul_tiles_bcast_scalar(cb_in, cb_scaler, wtr, 0, wtr);
        }
    // UNPACK(DPRINT <<"0" << ENDL());
    // UNPACK(print_full_tile(tt::CBIndex::c_24, 0, true));
        cb_pop_front(cb_in, blk);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_intermediate, blk);
        for (uint32_t wtr = 0; wtr < blk; wtr++) {
            pack_tile(wtr, cb_intermediate);
        }
    // UNPACK(DPRINT <<"0" << ENDL());
    // UNPACK(print_full_tile(tt::CBIndex::c_24, 0, true));
        cb_push_back(cb_intermediate, blk);
        tile_regs_release();
    // UNPACK(DPRINT <<"0" << ENDL());
    // UNPACK(print_full_tile(tt::CBIndex::c_24, 0, true));
    }
    reconfig_data_format(cb_intermediate, cb_intermediate);
    pack_reconfig_data_format(cb_intermediate);
    // 4 dst regs if FP32 and 8 is BFLOAT 16
    add_tiles_init(cb_intermediate, cb_intermediate);
    while (cb_length > 1 ) {
        uint32_t dstreg = 0;
        for (uint32_t i = 0; i < cb_length; i += 2) {
            // We acquire dst regs only if we are processing new block
            if (dstreg == 0) {
                tile_regs_acquire();
            }
            cb_wait_front(cb_intermediate, 2);
            add_tiles(cb_intermediate, cb_intermediate, 0, 1, dstreg);
            // dprint_tensix_dest_reg(dstreg);
            cb_pop_front(cb_intermediate, 2);
            // If we have an odd cb_length, we want to add the third tile to the result of the first two
            if (i == 0 && (cb_length & 1) == 1) {
                cb_wait_front(cb_intermediate, 1);
                binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_intermediate);
                binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_intermediate, 0, dst0);
                cb_pop_front(cb_intermediate, 1);
                add_tiles_init(cb_intermediate, cb_intermediate);
                // We decriment here since we no longer have an odd tile, it just is added to dst0
                cb_length--;
            }
            // We commit our registers either when we are finished or we are about to run out of dst registers
            if (dstreg == num_dst_regs - 1 || i + 2 == cb_length) {
                tile_regs_wait();
                tile_regs_commit();
                for (uint32_t dst = 0; dst < dstreg + 1; dst++) {
                    pack_tile(dst, cb_intermediate);
                }
                cb_push_back(cb_intermediate, dstreg + 1);
                tile_regs_release();
                dstreg = 0;
            }
            // increment to the next dst register
            else {
                dstreg++;
            }
        }
        // We are okay with floor divide since we subtracted one if cb_length is odd
        cb_length = cb_length / 2;
    }
    // TODO change this to a cb
    reconfig_data_format(cb_intermediate, cb_scaler);
    pack_reconfig_data_format(cb_out);
    reduce_init_delta<false, reduce_type, reduce_dim>(cb_intermediate, cb_scaler, cb_out);
    tile_regs_acquire();
    cb_wait_front(cb_scaler, 2);
    cb_wait_front(cb_intermediate, 1);
    cb_reserve_back(cb_out, onetile);
    reduce_tile<reduce_type, reduce_dim>(cb_intermediate, cb_scaler, 0, 1, dst0);
    cb_pop_front(cb_intermediate, onetile);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, cb_out);
    cb_push_back(cb_out, 1);
    cb_wait_front(cb_out, onetile);
    // UNPACK(print_full_tile(cb_out, 0, true));
    tile_regs_release();
    reduce_revert_delta<reduce_dim>(cb_out);
