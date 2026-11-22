// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// sdpa_fused_chunk (see recipe_fused_chunk.hpp). Included by recipe_step.inc once per attribute set: the includer
// defines SDPA_FUSED_CHUNK_ATTR (the chunk's function attributes) around the include.

template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t qkt_subblock_w,
    uint32_t qktv_subblock_w,
    uint32_t cb_q_in,
    uint32_t cb_kt_in,
    uint32_t cb_v_in,
    uint32_t cb_qkt_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_exp_max_diff,
    uint32_t cb_col_identity,
    uint32_t cb_recip_scratch,
    uint32_t cb_normalized_out,
    bool independent_q_release>
static SDPA_FUSED_CHUNK_ATTR void sdpa_fused_chunk(
    AccumulatorHalf& prev, AccumulatorHalf& cur, bool is_last_iter, bool release_q) {
    constexpr uint32_t KT = Sk_chunk_t;
    // Rows per group. Dense STANDARD (math-bound) runs a short Q chunk in one-row groups, so the lagged QK / check /
    // PV pipeline has enough groups to overlap. FAST is pack-bound and keeps two-row groups (one-row groups double
    // its per-group pack work), and so do the ring kernels and Wormhole STANDARD.
    constexpr uint32_t kH1MaxQTiles = recipe::lofi || recipe::ring || recipe::wormhole ? 0 : 6;
    constexpr uint32_t H = Sq_chunk_t <= kH1MaxQTiles ? 1 : 2;
    constexpr uint32_t sbw = qkt_subblock_w;
    // A K chunk of 2 mod 4 tiles ends with a two-wide QK subblock (recipe_qk_subblock_width); the chunk's row sums
    // stay sbw tiles wide (the narrow subblock adds into their first columns).
    constexpr bool qk_ragged = Sk_chunk_t % sbw != 0;
    constexpr uint32_t n_kb = (Sk_chunk_t + sbw - 1) / sbw;
    static_assert(!qk_ragged || (sbw == 4 && Sk_chunk_t % 4 == 2));
    static_assert(vDHt % qktv_subblock_w == 0);
    auto kb_w = [](uint32_t kb) -> uint32_t { return qk_ragged && kb == n_kb - 1 ? Sk_chunk_t % sbw : sbw; };
    // PV of a pipelined group runs in n_pieces K pieces, each after QK subblocks of the next group: two when the QK
    // subblocks split evenly, and at least enough that no piece accumulates more than the recipe's fold interval in
    // DEST (recipe::pv_fold; the packer adds each piece to O in FP32). Pieces split the K tiles and the QK subblocks
    // as evenly as they divide.
    constexpr uint32_t n_sched_pieces = n_kb % 2 == 0 ? 2 : 1;
    constexpr uint32_t fold = recipe::pv_fold == 0 ? KT : recipe::pv_fold;
    constexpr uint32_t n_fold_pieces = (KT + fold - 1) / fold;
    constexpr uint32_t n_pieces = n_fold_pieces > n_sched_pieces ? n_fold_pieces : n_sched_pieces;
    auto piece_kb = [](uint32_t piece) -> uint32_t { return piece * n_kb / n_pieces; };
    auto piece_k0 = [](uint32_t piece) -> uint32_t { return piece * KT / n_pieces; };
    constexpr uint32_t pv_w = qktv_subblock_w;
    constexpr uint32_t n_pv = vDHt / pv_w;
    constexpr bool tail = Sq_chunk_t % H != 0;
    constexpr uint32_t G = Sq_chunk_t / H + (tail ? 1 : 0);
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t neg_unit_cb = kFusedNegUnitCb;
    constexpr uint32_t lsum_cb = kRefMaxChunkSumCb;  // tile-shaped sums of a redone group
    constexpr uint32_t psum_cb = kFusedSumCb;
    constexpr uint32_t check_cb = kFusedCheckCb;
    const uint32_t out_cb = cur.out;
    auto rows = [](uint32_t g) -> uint32_t { return tail && g == G - 1 ? 1 : H; };
    // Group index in units of its own height (the helpers address rows as index * height).
    auto gindex = [](uint32_t g) -> uint32_t { return tail && g == G - 1 ? H * g : g; };

    // Block mask rows (the sparse ring's stacked Q blocks, recipe_ring_row_blocks): a group none of whose rows keeps a
    // block of this chunk is skipped (no QK, check, PV or fold: its state is untouched); a group with a row whose state
    // has seen no key (recipe_ring_rows_fresh) skips the fused QK and takes the redo, the reduce path for that group.
    constexpr bool row_masks = recipe::ring && recipe::ring_mask == 3;
    static_assert(!row_masks || G <= 32);
    if constexpr (row_masks) {
        recipe_ring_full_tiles = true;
        for (uint32_t t = 0; t < KT; ++t) {
            recipe_ring_full_tiles = recipe_ring_full_tiles && recipe_ring_tile_valid[t] == 32;
        }
        recipe_ring_skip_groups = 0;
        recipe_ring_fresh_groups = 0;
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t r0 = H * g, h = rows(g);
            recipe_ring_psum_w[g] = 0;
            if (recipe_ring_rows_blocks(r0, h) == 0) {
                recipe_ring_skip_groups |= 1u << g;
            } else if (((recipe_ring_rows_fresh >> r0) & ((1u << h) - 1u)) != 0) {
                recipe_ring_fresh_groups |= 1u << g;
            }
        }
    }

    // Runtime unpack-A / pack format tracking: the chunk interleaves broadcasts, matmuls, reduces and folds.
    uint32_t srca = cb_qkt_im;  // step entry state
    uint32_t pack_cb = cb_qkt_im;
    auto set_srca = [&](uint32_t cb) {
        if (cb != srca) {
            reconfig_data_format_srca(srca, cb);
            srca = cb;
        }
    };
    // Unpack-B: Q is the only srcB operand that may not be BF16 (a BFP8 / BFP4 Q); every other srcB operand here (P,
    // the maxima, the identity scale) is BF16. A BFP Q switches srcB around its matmuls; a BF16 Q compiles this out.
    constexpr bool q_srcb_switch = sdpa_unpack_format_changed<cb_qkt_im, cb_q_in>();
    auto srcb_to_q = [&]() {
        if constexpr (q_srcb_switch) {
            reconfig_data_format_srcb(cb_qkt_im, cb_q_in);
        }
    };
    auto srcb_from_q = [&]() {
        if constexpr (q_srcb_switch) {
            reconfig_data_format_srcb(cb_q_in, cb_qkt_im);
        }
    };
    auto set_pack = [&](uint32_t cb) {
        if (cb != pack_cb) {
            pack_reconfig_data_format(pack_cb, cb);
            pack_cb = cb;
        }
    };
    // The score exp's SFPU program and constants (28-octave headroom). Other SFPU users interleaved with the
    // exps (normalization's reciprocal, a redone group's correction and rescale) reprogram the shared
    // SFPU, so the exp is re-armed after them.
    auto arm_exp = []() { recipe_arm_score_exp<scale_fp32>(); };

    auto pack_to_unpack_barrier = []() {
        PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
        UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
        UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
    };

    CircularBuffer(cb_qkt_im).reserve_back(Sq_chunk_t * KT);
    CircularBuffer(cur.sum).reserve_back(Sq_chunk_t);
    CircularBuffer(lsum_cb).reserve_back(Sq_chunk_t);
    CircularBuffer(psum_cb).reserve_back(Sq_chunk_t * sbw);
    CircularBuffer(out_cb).reserve_back(Sq_chunk_t * vDHt * sdpa_out_stride);
    CircularBuffer(cb_kt_in).wait_front(DHt * KT);
    if constexpr (!row_masks) {
        // Stacked Q blocks start anywhere in their Q CB, whose wrapper waited for all of it.
        CircularBuffer(cb_q_in).wait_front(Sq_chunk_t * DHt);
    }

    // The reference max stays in prev.max for every fused chunk; a redone group updates its rows in place.
    CircularBuffer(prev.max).wait_front(Sq_chunk_t);
    // -e0 in K's format, built once per program from the column identity (the packer converts the format).
    if (!fused_neg_unit_ready) {
        CircularBuffer(cb_col_identity).wait_front(1);
        set_srca(cb_col_identity);
        copy_init(cb_col_identity);
        CircularBuffer(neg_unit_cb).reserve_back(sbw);
        tile_regs_acquire();
        copy_tile(cb_col_identity, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_negate_max, 0, VectorMode::None)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        set_pack(neg_unit_cb);
        configure_single_tile_pack(neg_unit_cb);
        for (uint32_t j = 0; j < sbw; ++j) {
            pack_tile(0, neg_unit_cb);
        }
        tile_regs_release();
        CircularBuffer(neg_unit_cb).push_back(sbw);
        CircularBuffer(neg_unit_cb).wait_front(sbw);
        fused_neg_unit_ready = true;
    }

    // The packer's ReLU clears the negative P of the LLK fast exp (recipe_score_exp's fallback for large scales); the
    // one-instruction score exp emits none.
    constexpr bool score_relu = !recipe_score_exp_one_op<scale_fp32>;
    // QK matmul setup is kept across back-to-back subblocks of the same height (the m step's replay ends
    // with the counter reset, so it leaves nothing behind), and in STANDARD so is the P pack setup (format,
    // row width, ReLU; FAST's exp ring builds have no room for it); every other step calls leave_qk.
    uint32_t qk_setup_h = 0;
    [[maybe_unused]] uint32_t qk_setup_w = sbw;
    // STANDARD keeps the P pack setup across QK subblocks (qk_pack_on); leave_qk resets it.
    bool qk_pack_on = false;
    // Out of line: it is called from every other step.
    auto leave_qk = [&]() __attribute__((noinline)) {
        qk_setup_h = 0;
        if (qk_pack_on) {
            if constexpr (score_relu) {
                PACK((llk_pack_relu_config(ReluConfig::none())));
            }
            qk_pack_on = false;
        }
    };
    // FAST: a plain statement (lambda calls from the -Os cold steps cost code, and FAST's exp ring builds sit
    // at the kernel config buffer limit).
#define SDPA_FUSED_LEAVE_QK()         \
    do {                              \
        if constexpr (recipe::lofi) { \
            qk_setup_h = 0;           \
        } else {                      \
            leave_qk();               \
        }                             \
    } while (0)
    // QK subblock (g, kb): DEST = -m_ref + Q K^T, exp in place, P and its row sums packed.
    auto qk_subblock = [&](uint32_t g, uint32_t kb) SDPA_FUSED_STEP {
        const uint32_t h = rows(g);
        const uint32_t row0 = H * g;
        uint32_t col0 = kb * sbw;
        uint32_t w = kb_w(kb);
        // Row sums: the group's first subblock packs them, the others L1-accumulate.
        bool psum_acc = kb > 0;
        [[maybe_unused]] uint32_t span_end = 0;
        if constexpr (row_masks) {
            // A skipped or fresh group runs no fused QK. A subblock runs only over the span of the group's kept tiles
            // in it (none: not at all; its P is never read, PV runs over the kept tiles only, and its row sums are
            // zero). The group's first computed piece starts the row-sum columns it covers (recipe_ring_psum_w); later
            // pieces are at most that wide, so every pack of the sums accumulates onto started columns.
            if (!recipe_ring_group_live(g)) {
                return;
            }
            const uint32_t bits = recipe_ring_rows_blocks(row0, h);
            const uint32_t b0 = col0 / kRecipeRingBlockDiv, b1 = (col0 + w - 1) / kRecipeRingBlockDiv;
            const uint32_t sub_blocks = ((2u << (b1 - b0)) - 1u) << b0;
            uint32_t t_lo = col0, t_hi = col0 + w;
            if (!recipe_ring_full_tiles || (bits & sub_blocks) != sub_blocks) {
                t_lo = col0 + w;
                t_hi = col0;
                for (uint32_t t = col0; t < col0 + w; ++t) {
                    if (recipe_ring_tile_valid[t] != 0 && ((bits >> (t / kRecipeRingBlockDiv)) & 1u) != 0) {
                        t_lo = t < t_lo ? t : t_lo;
                        t_hi = t + 1;
                    }
                }
                if (t_hi == col0) {
                    return;
                }
                // Even widths only (the m step needs QK subblocks at least two wide): the span grows to even offsets in
                // the subblock (the extra tiles are masked).
                t_lo = col0 + ((t_lo - col0) & ~1u);
                t_hi = col0 + ((t_hi - col0 + 1) & ~1u);
            }
            const uint32_t started = recipe_ring_psum_w[g];
            col0 = t_lo;
            span_end = t_hi;
            w = started != 0 && started < t_hi - t_lo ? started : t_hi - t_lo;
            psum_acc = started != 0;
        }
        while (true) {  // block mask rows: the span's pieces; otherwise once
            if (qk_setup_h != h || ((qk_ragged || row_masks) && qk_setup_w != w)) {
                set_srca(cb_kt_in);
                recipe_mm_reinit(cb_q_in, cb_kt_in, true, w, h, DHt);
                qk_setup_h = h;
                if constexpr (qk_ragged || row_masks) {
                    qk_setup_w = w;
                }
            }
            tile_regs_acquire();
            {
                uint32_t in0_index = row0 * DHt;
                uint32_t in1_index = col0;
                srcb_to_q();
                for (uint32_t inner = 0; inner < DHt; ++inner) {
                    matmul_block_no_mop(cb_q_in, cb_kt_in, in0_index, in1_index, 0, true, w, h, DHt);
                    in0_index++;
                    in1_index += KT;
                }
                srcb_from_q();
                // - m_ref: M (row r's m in column 0) x (-e0 transposed: -1 in row 0). Both operands are exact in one
                // fidelity phase (m_ref has 7 significant bits for srcB, -1 has one for srcA), so a HiFi build replays
                // its image once here, and only its inner 0-15 half (the rest multiplies zeros).
                UNPACK((llk_unpack_AB_matmul(prev.max, neg_unit_cb, row0, 0, w, h, 1)));
                // A LoFi image has no halves; this keeps FAST's matmul instantiations (and code size) as they were.
                constexpr bool m_inner_half = !recipe::lofi;
                MATH((llk_math_matmul_no_mop<MATH_FIDELITY, MM_THROTTLE, 1, m_inner_half>(
                    prev.max, neg_unit_cb, 0, w, h)));
            }
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (recipe::ring) {
                bool masked = recipe_ring_chunk_masked();
                if constexpr (row_masks) {
                    // Only a subblock with a partial tile, or a tile some row of the group does not keep.
                    masked = masked && recipe_ring_tiles_masked(row0, h, col0, w);
                }
                if (masked) {
                    recipe_mask_row0 = row0;
                    mask_recipe_tail(col0, w, h);
                    qk_setup_h = 0;
                }
            } else if constexpr (recipe::k_tail) {
                mask_recipe_tail(col0, w, h);
                qk_setup_h = 0;
            }
            if constexpr (recipe::lofi) {
                if constexpr (score_relu) {
                    PACK((llk_pack_relu_config(ReluConfig::zero())));
                }
                recipe_score_exp<scale_fp32>(0, h * w);
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                set_pack(cb_qkt_im);
                configure_row_pack_width(cb_qkt_im, w);
                pack_contiguous_rows_nocfg(cb_qkt_im, row0, h, KT, col0, w);
                PACK((llk_pack_reconfig_l1_acc(psum_acc ? 1 : 0)));
                pack_contiguous_rows_nocfg(psum_cb, row0, h, sbw, 0, w);
                PACK((llk_pack_reconfig_l1_acc(0)));
                if constexpr (score_relu) {
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                }
            } else {
                if constexpr (score_relu) {
                    if (!qk_pack_on) {
                        PACK((llk_pack_relu_config(ReluConfig::zero())));
                    }
                }
                recipe_score_exp<scale_fp32>(0, h * w);
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                if (!qk_pack_on) {
                    set_pack(cb_qkt_im);
                    configure_row_pack_width(cb_qkt_im, w);
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    qk_pack_on = true;
                } else if constexpr (qk_ragged || row_masks) {
                    if (w != sbw) {
                        configure_row_pack_width(cb_qkt_im, w);
                    }
                }
                pack_contiguous_rows_nocfg(cb_qkt_im, row0, h, KT, col0, w);
                if (psum_acc) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
                pack_contiguous_rows_nocfg(psum_cb, row0, h, sbw, 0, w);
                if (psum_acc) {
                    PACK((llk_pack_reconfig_l1_acc(0)));
                }
                if constexpr (qk_ragged || row_masks) {
                    // The narrow subblock's pack width does not carry over: the next subblock sets the pack up again.
                    if (w != sbw) {
                        leave_qk();
                    }
                }
            }
            tile_regs_release();
            if constexpr (row_masks) {
                if (!psum_acc) {
                    recipe_ring_psum_w[g] = w;
                }
                psum_acc = true;
                col0 += w;
                if (col0 >= span_end) {
                    break;
                }
                w = span_end - col0 < recipe_ring_psum_w[g] ? span_end - col0 : recipe_ring_psum_w[g];
            } else {
                break;
            }
        }
    };

    // Saturation check of groups [g0, g0 + n): max over their chunk row-sum tiles, into the one-tile check CB.
    // STANDARD (math-bound) reduces columns only, FAST (pack-bound) to a scalar; read_check takes the max.
    constexpr ReduceDim check_dim = recipe::lofi ? ReduceDim::REDUCE_SCALAR : ReduceDim::REDUCE_COL;
    // Check unit size (groups per check). STANDARD is math-bound, so halving the checks' fixed cost pays;
    // FAST checks every group.
    constexpr uint32_t CG = recipe::lofi ? 1 : 2;
    auto check_groups = [&](uint32_t g0, uint32_t n) SDPA_FUSED_STEP {
        SDPA_FUSED_LEAVE_QK();
        const uint32_t last = g0 + n - 1;
        const uint32_t row_end = H * last + rows(last);
        CircularBuffer(cb_qkt_im).wait_front(row_end * KT);  // the groups' sums were packed before their push
        set_srca(psum_cb);
        reduce_init<PoolType::MAX, check_dim>(psum_cb, cb_identity_scale_in, check_cb);
        tile_regs_acquire();
        for (uint32_t t = H * g0 * sbw; t < row_end * sbw; ++t) {
            if constexpr (row_masks) {
                // Skipped and fresh groups packed no sums; a group's started columns only.
                const uint32_t g = t / (H * sbw);
                if (!recipe_ring_group_live(g) || t % sbw >= recipe_ring_psum_w[g]) {
                    continue;
                }
            }
            reduce_tile<PoolType::MAX, check_dim>(psum_cb, cb_identity_scale_in, t, 0, 0);
        }
        tile_regs_commit();
        reduce_uninit();
        if constexpr (recipe::lofi) {
            // The reduce may replace the matmul replay image: the next matmul init must re-record it.
            recipe_mm_reuse_a = false;
        }
        CircularBuffer(check_cb).reserve_back(1);
        tile_regs_wait();
        set_pack(check_cb);
        configure_single_tile_pack(check_cb);
        pack_tile(0, check_cb);
        tile_regs_release();
        CircularBuffer(check_cb).push_back(1);
    };
    // The verdict of the oldest issued check: does any of its groups' partial sums reach the redo threshold.
    auto read_check = [&]() SDPA_FUSED_STEP -> bool {
        uint32_t redo = 0;
        UNPACK({
            CircularBuffer(check_cb).wait_front(1);
            auto* check = reinterpret_cast<volatile uint32_t*>(get_tile_l1_byte_address(get_operand_id(check_cb), 0));
            if constexpr (check_dim == ReduceDim::REDUCE_SCALAR) {
                redo = (check[0] & 0x7fffu) >= kFusedRedoSumBf16 ? 1u : 0u;
            } else {
                // Column maxima: row 0 of faces 0 and 1, BF16 pairs at uint32 words 0-7 and 128-135.
                for (uint32_t i = 0; i < 8; ++i) {
                    const uint32_t a = check[i], b = check[128 + i];
                    redo |= ((a & 0x7fffu) >= kFusedRedoSumBf16) | (((a >> 16) & 0x7fffu) >= kFusedRedoSumBf16) |
                            ((b & 0x7fffu) >= kFusedRedoSumBf16) | (((b >> 16) & 0x7fffu) >= kFusedRedoSumBf16);
                }
            }
            CircularBuffer(check_cb).pop_front(1);
            mailbox_write(ckernel::ThreadId::MathThreadId, redo);
            mailbox_write(ckernel::ThreadId::PackThreadId, redo);
        })
        MATH(redo = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        PACK(redo = mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        return redo != 0;
    };

    // PV of group g over K tiles [k0, k0 + k_len), L1-accumulated into O plane `plane` (0: O, vDHt: a redone
    // group's chunk PV, written by the first piece).
    bool v_ready = false;
    // In-place latent V (sdpa_inner_loop_step: the K CB passed as V): V tile (k, d) is K^T tile (d, k).
    constexpr bool v_in_k = cb_v_in == cb_kt_in;
    // The unpacker's column stride needs PV subblocks at least as wide as tall (one-row Q chunks: any width).
    static_assert(!v_in_k || Sq_chunk_t == 1 || pv_w >= H, "in-place V subblocks");
    auto pv_piece = [&](uint32_t g, uint32_t k0, uint32_t k_len, uint32_t plane, bool accumulate) SDPA_FUSED_STEP {
        SDPA_FUSED_LEAVE_QK();
        const uint32_t h = rows(g);
        const uint32_t row0 = H * g;
        if constexpr (recipe::ring_stream_state) {
            if (plane == 0) {
                recipe_stream_wait_rows(row0 + h);
            }
        }
        if (!v_ready) {
            if constexpr (!v_in_k) {
                CircularBuffer(cb_v_in).wait_front(KT * vDHt);
            }
            v_ready = true;
        }
        CircularBuffer(cb_qkt_im).wait_front((row0 + h) * KT);
        set_srca(cb_v_in);
        recipe_mm_reinit(cb_qkt_im, cb_v_in, false, pv_w, h, KT);
        if constexpr (v_in_k) {
            matmul_set_in1_column_stride(cb_v_in, KT);
        }
        set_pack(out_cb);
        PACK((llk_pack_reconfig_l1_acc(accumulate ? 1 : 0)));
        configure_row_pack_width(out_cb, pv_w);
        for (uint32_t v = 0; v < n_pv; ++v) {
            blocked_matmul_and_pack<false, v_in_k ? 1 : vDHt, vDHt>(
                cb_qkt_im,
                cb_v_in,
                out_cb,
                row0 * KT + k0,
                v_in_k ? v * pv_w * KT + k0 : k0 * vDHt + v * pv_w,
                0,
                v * pv_w + plane,
                pv_w,
                h,
                k_len,
                KT,
                /*skip_pack_configure=*/true);
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
        if constexpr (v_in_k) {
            matmul_set_in1_column_stride(cb_v_in, 1);
        }
    };

    // Block mask: PV over the piece's runs of valid K tiles only (the others' P is zero), accumulating onto O. With no
    // valid tile, only the waits a PV would do (V, and the group's scores before the chunk pops them).
    [[maybe_unused]] auto pv_piece_kept = [&](uint32_t g, uint32_t k0, uint32_t k_len) SDPA_FUSED_STEP {
        bool any = false;
        const uint32_t bits =
            recipe_ring_group_live(g) ? recipe_ring_rows_blocks(H * g, rows(g)) : 0u;  // block mask rows only
        auto kept = [&](uint32_t t) -> bool {
            return recipe_ring_tile_valid[t] != 0 && ((bits >> (t / kRecipeRingBlockDiv)) & 1u) != 0;
        };
        for (uint32_t t = k0; t < k0 + k_len;) {
            if (!kept(t)) {
                ++t;
                continue;
            }
            uint32_t e = t + 1;
            while (e < k0 + k_len && kept(e)) {
                ++e;
            }
            pv_piece(g, t, e - t, 0, true);
            any = true;
            t = e;
        }
        if (!any) {
            if (!v_ready) {
                if constexpr (!v_in_k) {
                    CircularBuffer(cb_v_in).wait_front(KT * vDHt);
                }
                v_ready = true;
            }
            CircularBuffer(cb_qkt_im).wait_front((H * g + rows(g)) * KT);
        }
    };

    // l += this chunk's sums for an unchanged group.
    auto fold_sums = [&](uint32_t g) SDPA_FUSED_STEP {
        SDPA_FUSED_LEAVE_QK();
        const uint32_t h = rows(g);
        set_srca(psum_cb);
        copy_init(psum_cb);
        tile_regs_acquire();
        for (uint32_t t = 0; t < h * sbw; ++t) {
            copy_tile(psum_cb, H * g * sbw + t, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        set_pack(cur.sum);
        configure_single_tile_pack(cur.sum);
        PACK((llk_pack_reconfig_l1_acc(1)));
        uint32_t cols = sbw;
        if constexpr (row_masks) {
            cols = recipe_ring_psum_w[g];  // the started row-sum columns
        }
        for (uint32_t i = 0; i < h; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                pack_tile<true>(i * sbw + j, cur.sum, i);
            }
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
        tile_regs_release();
    };

    // Publish group g's O and l rows; the last chunk normalizes them.
    auto finish_group = [&](uint32_t g) SDPA_FUSED_STEP {
        SDPA_FUSED_LEAVE_QK();
        const uint32_t h = rows(g);
        CircularBuffer(cur.sum).push_back(h);
        CircularBuffer(out_cb).push_back(h * vDHt * sdpa_out_stride);
        if constexpr (recipe::ring_stream_state) {
            // The last chunk of a saved block: these O rows are final for this ring iteration.
            if (recipe_stream_save_slot != kNoStreamSlot) {
                recipe_stream_save_rows(out_cb, H * g + h);
            }
        }
        if (is_last_iter) {
            if constexpr (recipe::sink) {
                recipe_sink_max_cb = prev.max;
            }
            normalize_row_streaming<vDHt, dst_size, cb_col_identity, cb_recip_scratch, cb_normalized_out, scale_fp32>(
                cur.sum, out_cb, h);
            srca = cb_col_identity;  // normalize_row_streaming leaves srcA in BF16
            pack_cb = cb_recip_scratch;
            arm_exp();
        }
    };

    // Redo group g on the reduce path (rare): scores, the real max with the theta select, P and sums,
    // its PV into plane 1, then O = O * c + PV and l = l * c + l_chunk row by row.
    auto redo_group = [&](uint32_t g) SDPA_FUSED_REDO {
        SDPA_FUSED_LEAVE_QK();
        const uint32_t h = rows(g);
        const uint32_t row0 = H * g;
        if constexpr (recipe::ring_stream_state) {
            recipe_stream_wait_rows(row0 + h);
        }
        const uint32_t gi = gindex(g);
        set_srca(cb_kt_in);
        recipe_mm_init(cb_q_in, cb_kt_in, true, sbw, h, DHt);
        set_pack(cb_qkt_im);
        configure_row_pack_width(cb_qkt_im, sbw);
        srcb_to_q();
        for (uint32_t kb = 0; kb < n_kb; ++kb) {
            const uint32_t w = kb_w(kb);
            if constexpr (qk_ragged) {
                if (w != sbw) {
                    recipe_mm_reinit(cb_q_in, cb_kt_in, true, w, h, DHt);
                    configure_row_pack_width(cb_qkt_im, w);
                }
            }
            blocked_matmul_and_pack<true, KT, KT>(
                cb_q_in, cb_kt_in, cb_qkt_im, row0 * DHt, kb * sbw, gi, kb * sbw, w, h, DHt, DHt, true);
        }
        srcb_from_q();
        pack_to_unpack_barrier();
        set_srca(cb_qkt_im);
        set_pack(cur.max);
        configure_single_tile_pack(cur.max);
        reduce_c_row_group<cb_qkt_im, cb_identity_scale_in, KT, scale_fp32>(
            cur.max, prev.max, gi, true, h, KT, false, false, row0);
        pack_to_unpack_barrier();
        for (uint32_t kb = 0; kb < n_kb; ++kb) {
            sub_exp_block_bcast_cols<scale_fp32>(
                cb_qkt_im, cur.max, lsum_cb, KT, gi, kb * sbw, h, kb_w(kb), false, /*wait_max=*/false);
        }
        pack_cb = cb_qkt_im;
        srca = cb_qkt_im;
        pack_to_unpack_barrier();
        for (uint32_t piece = 0; piece < n_pieces; ++piece) {
            pv_piece(g, piece_k0(piece), piece_k0(piece + 1) - piece_k0(piece), vDHt, piece > 0);
        }
        pack_to_unpack_barrier();
        const uint32_t read_base = is_last_iter ? 0 : row0;
        for (uint32_t r = 0; r < h; ++r) {
            // c = exp(scale * (m_old - m_new)) for this Q tile row, column-broadcast below.
            set_srca(prev.max);
            sub_init(prev.max, cur.max);
            tile_regs_acquire();
            sub_tiles(prev.max, cur.max, row0 + r, row0 + r, 0);
            tile_regs_commit();
            CircularBuffer(cb_recip_scratch).reserve_back(1);
            tile_regs_wait();
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_correction, (scale_fp32), 0, VectorMode::C)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            set_pack(cb_recip_scratch);
            configure_single_tile_pack(cb_recip_scratch);
            pack_tile(0, cb_recip_scratch);
            tile_regs_release();
            CircularBuffer(cb_recip_scratch).push_back(1);
            CircularBuffer(cb_recip_scratch).wait_front(1);
            // l = l * c + l_chunk
            tile_regs_acquire();
            set_srca(cur.sum);
            copy_init(cur.sum);
            copy_tile(cur.sum, read_base + r, 0);
            set_srca(lsum_cb);
            copy_init(lsum_cb);
            copy_tile(lsum_cb, row0 + r, 1);
            set_srca(cb_recip_scratch);
            unary_bcast_init<BroadcastType::COL>(cb_recip_scratch);
            unary_bcast<BroadcastType::COL>(cb_recip_scratch, 0, 2);
            unary_bcast_uninit<BroadcastType::COL>(cb_recip_scratch);
            tile_regs_commit();
            tile_regs_wait();
            PACK(
                (SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_rescale_add, (1), 0, VectorMode::None)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            set_pack(cur.sum);
            configure_single_tile_pack(cur.sum);
            pack_tile<true>(0, cur.sum, r);
            tile_regs_release();
            // O = O * c + PV, two head-dim tiles at a time
            for (uint32_t j = 0; j < vDHt; j += 2) {
                const bool pair = j + 1 < vDHt;
                tile_regs_acquire();
                set_srca(out_cb);
                copy_init(out_cb);
                const uint32_t rbase = (read_base + r) * 2 * vDHt + j;
                copy_tile(out_cb, rbase, 0);
                if (pair) {
                    copy_tile(out_cb, rbase + 1, 1);
                }
                copy_tile(out_cb, rbase + vDHt, pair ? 2 : 1);
                if (pair) {
                    copy_tile(out_cb, rbase + vDHt + 1, 3);
                }
                set_srca(cb_recip_scratch);
                unary_bcast_init<BroadcastType::COL>(cb_recip_scratch);
                unary_bcast<BroadcastType::COL>(cb_recip_scratch, 0, pair ? 4 : 2);
                unary_bcast_uninit<BroadcastType::COL>(cb_recip_scratch);
                tile_regs_commit();
                tile_regs_wait();
                if (pair) {
                    PACK((SFPU_UNARY_CALL(
                        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_rescale_add, (2), 0, VectorMode::None)));
                } else {
                    PACK((SFPU_UNARY_CALL(
                        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_rescale_add, (1), 0, VectorMode::None)));
                }
                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                set_pack(out_cb);
                configure_single_tile_pack(out_cb);
                const uint32_t wbase = r * 2 * vDHt + j;
                pack_tile<true>(0, out_cb, wbase);
                if (pair) {
                    pack_tile<true>(1, out_cb, wbase + 1);
                }
                tile_regs_release();
            }
            CircularBuffer(cb_recip_scratch).pop_front(1);
        }
        // The new maxima become this group's reference (cur.max was scratch, prev.max is the state).
        set_srca(cur.max);
        copy_init(cur.max);
        tile_regs_acquire();
        for (uint32_t r = 0; r < h; ++r) {
            copy_tile(cur.max, row0 + r, r);
        }
        tile_regs_commit();
        tile_regs_wait();
        set_pack(prev.max);
        configure_single_tile_pack(prev.max);
        for (uint32_t r = 0; r < h; ++r) {
            pack_tile<true>(r, prev.max, row0 + r);
        }
        tile_regs_release();
        pack_to_unpack_barrier();
        arm_exp();
    };

    // Non-last chunks fold every unchanged group's sums once at the end (one pack-format switch); the last
    // chunk folds per group, ahead of each group's normalization.
    const bool late_fold = !is_last_iter;
    uint32_t redone = 0;
    // Groups are checked in units of CG: one reduce, one verdict and one round of handshakes per unit; a unit
    // whose verdict fires is re-checked group by group, so only the groups that need it are redone. Iteration it runs
    // QK of group it, the check of the unit ending at group it - 1 and PV of group it - CG - 1, so a unit's verdicts
    // are read one iteration after its check was issued and before its first group's PV touches O.
    uint32_t unit_redo = 0;  // the current check unit's verdicts, bit j for its group j
    for (uint32_t it = 0; it < G + CG + 1; ++it) {
        const bool do_qk = it < G;
        const uint32_t cg_last = it - 1;
        const bool do_check = it >= 1 && it <= G && ((cg_last + 1) % CG == 0 || cg_last == G - 1);
        bool do_pv = it >= CG + 1;
        const uint32_t pg = it - CG - 1;
        if (do_pv && pg % CG == 0) {
            const uint32_t n = G - pg < CG ? G - pg : CG;
            bool live = true;
            if constexpr (row_masks) {
                live = recipe_ring_unit_live(pg, n);
            }
            unit_redo = live && read_check() ? 1u : 0u;
            if (unit_redo && n > 1) {
                // Rare: find which groups need the redo (redoing a neighbour would change its numerics).
                unit_redo = 0;
                for (uint32_t j = 0; j < n; ++j) {
                    if constexpr (row_masks) {
                        if (!recipe_ring_group_live(pg + j)) {
                            continue;
                        }
                    }
                    check_groups(pg + j, 1);
                    unit_redo |= (read_check() ? 1u : 0u) << j;
                }
            }
            if constexpr (row_masks) {
                unit_redo |= (recipe_ring_fresh_groups >> pg) & ((1u << n) - 1u);
            }
        }
        if (do_pv && ((unit_redo >> (pg % CG)) & 1u)) {
            redo_group(pg);
            finish_group(pg);
            redone |= 1u << pg;
            do_pv = false;
        }
        for (uint32_t piece = 0; piece < n_pieces; ++piece) {
            if (do_qk) {
                for (uint32_t kb = piece_kb(piece); kb < piece_kb(piece + 1); ++kb) {
                    qk_subblock(it, kb);
                }
                if (piece == n_pieces - 1) {
                    cb_push_back_hold_wr_ptr(cb_qkt_im, rows(it) * KT);
                }
            }
            bool check_live = true;
            if constexpr (row_masks) {
                check_live = recipe_ring_unit_live(cg_last - cg_last % CG, cg_last % CG + 1);
            }
            if (do_check && piece == 0 && check_live) {
                check_groups(cg_last - cg_last % CG, cg_last % CG + 1);
            }
            if (do_pv) {
                if constexpr (recipe::ring && recipe::ring_mask == 3) {
                    pv_piece_kept(pg, piece_k0(piece), piece_k0(piece + 1) - piece_k0(piece));
                } else {
                    pv_piece(pg, piece_k0(piece), piece_k0(piece + 1) - piece_k0(piece), 0, true);
                }
            }
        }
        if (do_pv) {
            bool fold = !late_fold;
            if constexpr (row_masks) {
                fold = fold && ((recipe_ring_skip_groups >> pg) & 1u) == 0;
            }
            if (fold) {
                fold_sums(pg);
            }
            finish_group(pg);
        }
    }
    if constexpr (!recipe::lofi) {
        leave_qk();
    }
    if (late_fold) {
        // cur.sum has wrapped back to this chunk's first row (one push per row); fold unchanged groups in place.
        set_srca(psum_cb);
        copy_init(psum_cb);
        set_pack(cur.sum);
        configure_single_tile_pack(cur.sum);
        PACK((llk_pack_reconfig_l1_acc(1)));
        for (uint32_t g = 0; g < G; ++g) {
            if ((redone >> g) & 1u) {
                continue;
            }
            if constexpr (row_masks) {
                if ((recipe_ring_skip_groups >> g) & 1u) {
                    continue;
                }
            }
            const uint32_t h = rows(g);
            tile_regs_acquire();
            for (uint32_t t = 0; t < h * sbw; ++t) {
                copy_tile(psum_cb, H * g * sbw + t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t cols = sbw;
            if constexpr (row_masks) {
                cols = recipe_ring_psum_w[g];  // the started row-sum columns
            }
            for (uint32_t i = 0; i < h; ++i) {
                for (uint32_t j = 0; j < cols; ++j) {
                    pack_tile<true>(i * sbw + j, cur.sum, H * g + i);
                }
            }
            tile_regs_release();
        }
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
    // K and Q stay until here: a redone group recomputes its scores.
    if constexpr (!recipe::kv_hold) {
        CircularBuffer(cb_kt_in).pop_front(DHt * KT);
    }
    if (independent_q_release ? release_q : is_last_iter) {
        sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
    }
    // Chunk sums were read in place; release them as a whole (pointers wrap back to the same slots).
    CircularBuffer(lsum_cb).push_back(Sq_chunk_t);
    CircularBuffer(lsum_cb).wait_front(Sq_chunk_t);
    CircularBuffer(lsum_cb).pop_front(Sq_chunk_t);
    CircularBuffer(psum_cb).push_back(Sq_chunk_t * sbw);
    CircularBuffer(psum_cb).wait_front(Sq_chunk_t * sbw);
    CircularBuffer(psum_cb).pop_front(Sq_chunk_t * sbw);
    if constexpr (!v_in_k && !recipe::kv_hold) {
        CircularBuffer(cb_v_in).pop_front(KT * vDHt);
    }
    CircularBuffer(cb_qkt_im).pop_front(Sq_chunk_t * KT);
    // Leave the unpack/pack formats where the step expects them.
    set_srca(cb_qkt_im);
    set_pack(cb_qkt_im);
}
#undef SDPA_FUSED_LEAVE_QK
