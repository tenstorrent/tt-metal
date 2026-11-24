// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Split-K merge of recipe states: a core's own unnormalized state (after its last K chunk, run non-final) merged with
// the unnormalized states of the cores that ran the other K ranges of the same Q rows, then the recipe's normalization.
// Used by sparse SDPA's split-K TAIL (sparse_sdpa_compute.cpp) and the Gemma4 sliding ring's K-split reducer
// (gemma4 sliding_ring_sdpa.cpp). A received state sits in three CBs the caller's dataflow fills: O rows of vd tiles
// (plane 0 only), the maxima and l, each in the recipe state's format. The merge is the online softmax's own rescale
// (m = max(m_a, m_b), c_x = exp(scale (m_x - m)), O = O_a c_a + O_b c_b, l = l_a c_a + l_b c_b): only the summation
// order differs from one core running every K chunk.
// Include after the recipe headers (recipe/recipe_streaming.hpp).

// SFPI code exists in the MATH and PACK builds only.
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// BF16 recipes. dest tile 0: the own maxima m_a, tile 1: the received m_b (column 0 is read). Tile 0 <- c_a =
// exp(scale * (m_a - m)), tile 1 <- c_b = exp(scale * (m_b - m)), tile 2 <- m = max(m_a, m_b): both states rescaled to
// the common maximum, which becomes the merged state's maxima when another state merges after it. One exp in the code
// (a two-pass loop): the merge runs once per call and its code shares the kernel config buffer with the recipe step.
template <uint32_t scale_bits>
inline void calculate_recipe_merge_corrections() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr float scale = __builtin_bit_cast(float, scale_bits);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[32];
        sfpi::vFloat m = a;
        v_if(b > a) { m = b; }
        v_endif;
        sfpi::dst_reg[64] = m;
#pragma GCC unroll 1
        for (int k = 0; k < 2; ++k) {
            sfpi::vFloat x = sfpi::dst_reg[32 * k];
            sfpi::dst_reg[32 * k] = _sfpu_exp_fp32_accurate_((x - m) * scale);
        }
        sfpi::dst_reg++;
    }
}
// BF16 recipes. dest tiles [0, w): the own state, [w, 2w): the received one, tiles 2w and 2w + 1: column-broadcast
// c_a and c_b. state = state * c_a + received * c_b.
template <int w>
inline void calculate_recipe_merge() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat ca = sfpi::dst_reg[32 * 2 * w];
        sfpi::vFloat cb = sfpi::dst_reg[32 * (2 * w + 1)];
#pragma GCC unroll 3
        for (int t = 0; t < w; ++t) {
            sfpi::vFloat x = sfpi::dst_reg[32 * t];
            sfpi::vFloat y = sfpi::dst_reg[32 * (w + t)];
            sfpi::dst_reg[32 * t] = x * ca + y * cb;
        }
        sfpi::dst_reg++;
    }
}
// FP32 recipes, the first column (VectorMode::C). dest tiles 0-3: the maxima m_a, m_b and the denominators l_a, l_b
// of the own and the received states. Tile 0 <- s_a = c_a / l, tile 1 <- s_b = c_b / l: c_x = exp(scale * (m_x -
// m)), m = max(m_a, m_b), l = l_a c_a + l_b c_b (the merged denominator and its reciprocal as the recipe's
// normalization takes them), each times output_scale (a power of two, e.g. sparse SDPA's fused dequant 2^-OFFSET). The
// merged numerator O_a c_a + O_b c_b is never stored: the normalized output is O_a s_a + O_b s_b.
template <uint32_t scale_bits, uint32_t output_scale_bits>
inline void calculate_recipe_merge_scales() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr float scale = __builtin_bit_cast(float, scale_bits);
    constexpr float output_scale = __builtin_bit_cast(float, output_scale_bits);
    sfpi::vConstFloatPrgm0 = 2.0f;  // sfpu_reciprocal_iter's constant (fp32_state_sfpu.hpp sdpa_state_reciprocal)
    for (int d = 0; d < 4; ++d) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[32];
        sfpi::vFloat m = a;
        v_if(b > a) { m = b; }
        v_endif;
        sfpi::vFloat ca = _sfpu_exp_fp32_accurate_((a - m) * scale);
        sfpi::vFloat cb = _sfpu_exp_fp32_accurate_((b - m) * scale);
        sfpi::vFloat l = sfpi::dst_reg[64] * ca + sfpi::dst_reg[96] * cb;
        sfpi::vFloat r = sfpu_reciprocal_iter<2>(l);
        if constexpr (output_scale_bits != 0x3f800000u) {
            r = r * output_scale;
        }
        sfpi::dst_reg[0] = ca * r;
        sfpi::dst_reg[32] = cb * r;
        sfpi::dst_reg += 2;
    }
}
// FP32 recipes. dest tiles 0, 1: O_a and O_b, tiles 2, 3: column-broadcast s_a and s_b. Tile 0 <- O_a s_a + O_b s_b.
inline void calculate_recipe_merge_normalize() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (int i = 0; i < 32; ++i) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat y = sfpi::dst_reg[32];
        sfpi::dst_reg[0] = x * sfpi::dst_reg[64] + y * sfpi::dst_reg[96];
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace recipe_merge {
// A sender band's state is final in L1 (BF16 recipes: O and l at CB 9's and CB 13's bases, the maxima pushed in
// state.prev.max): hand it to the writer (a one-page CB: words max_cb_word, has_state_word), which signals the reducer.
// The state stays in place until the program ends.
template <uint32_t signal_cb, uint32_t max_cb_word, uint32_t has_state_word>
void publish_state(const RecipeAccumulatorState& state, bool has_state) {
    recipe_pack_to_unpack_barrier();  // every pack of the state has landed in L1
    CircularBuffer signal(signal_cb);
    signal.reserve_back(1);
    PACK({
        auto* words = reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(signal_cb).fifo_wr_ptr << 4);
        words[max_cb_word] = state.prev.max;
        words[has_state_word] = has_state ? 1u : 0u;
    })
    signal.push_back(1);
}

// The CBs of one received state and the merge's correction scratch: O rows of vd tiles, the maxima, l (the recipe
// state's formats) and a q_tiles-tile CB in the maxima's format for c_b (the recipe's exp_max_diff CB takes c_a).
template <uint32_t recv_out_, uint32_t recv_max_, uint32_t recv_sum_, uint32_t corr_>
struct Received {
    static constexpr uint32_t out = recv_out_, max = recv_max_, sum = recv_sum_, corr = corr_;
};

// Copy tile `index` of `cb` into dest tile `dest` (the recipe's state formats; FP32 state CBs unpack to dest).
ALWI void copy_state_tile(uint32_t cb, uint32_t index, uint32_t dest) {
    sdpa_stream_reconfig_srca(cb);
    copy_init(cb);
    copy_tile(cb, index, dest);
}

// The received state was taken from: pop it unmerged (a K range with no keys).
template <uint32_t q_tiles, uint32_t vd_tiles, typename R>
ALWI void discard(R) {
    CircularBuffer(R::out).wait_front(q_tiles * vd_tiles);
    CircularBuffer(R::max).wait_front(q_tiles);
    CircularBuffer(R::sum).wait_front(q_tiles);
    CircularBuffer(R::out).pop_front(q_tiles * vd_tiles);
    CircularBuffer(R::max).pop_front(q_tiles);
    CircularBuffer(R::sum).pop_front(q_tiles);
}

// The BF16 merge's steps out of line, one copy each (the merge runs once per call; its code shares the kernel config
// buffer with the recipe step). Tiles [first, first + n) of `cb` into dest tiles [dst, dst + n), the unpacker set up
// for the state format of `cb`.
inline __attribute__((noinline, optimize("Os"))) void merge_copy_tiles(uint32_t cb, uint32_t first, uint32_t n, uint32_t dst) {
    sdpa_stream_reconfig_srca(cb);
    copy_init(cb);
    for (uint32_t t = 0; t < n; ++t) {
        copy_tile(cb, first + t, dst + t);
    }
}
// Row r's corrections, column-broadcast: c_a (CB a_cb) into dest tile dst, c_b (CB b_cb, the same format) into dst + 1.
inline __attribute__((noinline, optimize("Os"))) void merge_bcast_corrections(uint32_t a_cb, uint32_t b_cb, uint32_t r, uint32_t dst) {
    sdpa_stream_reconfig_srca(a_cb);
    unary_bcast_init<BroadcastType::COL>(a_cb);
    unary_bcast<BroadcastType::COL>(a_cb, r, dst);
    unary_bcast<BroadcastType::COL>(b_cb, r, dst + 1);
    unary_bcast_uninit<BroadcastType::COL>(a_cb);
}
// w tiles of state (dest [0, w)) and received state (dest [w, 2w)) with their corrections (dest 2w, 2w + 1):
// state = state * c_a + received * c_b, packed back to `cb` at tile `index` (absolute: the state's rows).
template <uint32_t w>
ALWI void merge_tiles_sfpu() {
    PACK((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_recipe_merge, (w), 0, VectorMode::None)));
}
template <uint32_t max_w>
__attribute__((noinline, optimize("Os"))) void merge_finish_tiles(uint32_t w, uint32_t cb, uint32_t index) {
    tile_regs_commit();
    tile_regs_wait();
    if (max_w == 1 || w == 1) {
        merge_tiles_sfpu<1>();
    } else if (max_w == 2 || w == 2) {
        merge_tiles_sfpu<2>();
    } else {
        merge_tiles_sfpu<3>();
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    for (uint32_t t = 0; t < w; ++t) {
        pack_tile<true>(t, cb, index + t);
    }
    tile_regs_release();
}
// w (<= max_w <= 3) tiles of the own state (CB a_cb from tile a_index) merged in place with the received ones (CB b_cb
// from b_index), row r's corrections in exp_max_diff (c_a) and corr_cb (c_b).
template <uint32_t max_w>
__attribute__((noinline, optimize("Os"))) void merge_state_tiles(
    uint32_t w,
    uint32_t a_cb,
    uint32_t a_index,
    uint32_t b_cb,
    uint32_t b_index,
    uint32_t exp_max_diff,
    uint32_t corr_cb,
    uint32_t r) {
    tile_regs_acquire();
    merge_copy_tiles(a_cb, a_index, w, 0);
    for (uint32_t t = 0; t < w; ++t) {
        copy_tile(b_cb, b_index + t, w + t);  // the own state's format: the unpacker stays set up
    }
    merge_bcast_corrections(exp_max_diff, corr_cb, r, 2 * w);
    merge_finish_tiles<max_w>(w, a_cb, a_index);
}

// BF16 recipes: the own state after its last chunk (O in CB 9's rows, plane 0, and l in CB 13 at their bases, popped;
// the maxima pushed in state.prev.max) merged in place with the received state. keep_max: the merged maxima replace
// the own ones (another received state merges next); otherwise they are left stale (the normalization follows).
// O tiles merge up to max_w per dest pass (3: 3 + 3 + 2 of the 16-bit dest's 8 tiles).
template <uint32_t q_tiles, uint32_t vd_tiles, uint32_t scale_fp32, uint32_t exp_max_diff, typename R, uint32_t max_w = 3>
void merge_bf16(const RecipeAccumulatorState& state, bool keep_max) {
    static_assert(max_w >= 1 && max_w <= 3);
    constexpr uint32_t o_stride = sdpa_out_stride;  // numerator CB tiles per Q row, over vd
    const uint32_t n_out = state.cur.out, n_sum = state.cur.sum, a_max = state.prev.max;
    CircularBuffer(R::out).wait_front(q_tiles * vd_tiles);
    CircularBuffer(R::max).wait_front(q_tiles);
    CircularBuffer(R::sum).wait_front(q_tiles);
    // Corrections per Q row: c_a into CB exp_max_diff, c_b into R::corr (and m back into the own maxima).
    CircularBuffer(exp_max_diff).reserve_back(q_tiles);
    CircularBuffer(R::corr).reserve_back(q_tiles);
    PACK((llk_pack_reconfig_l1_acc(0)));
    pack_reconfig_data_format(exp_max_diff);
    configure_single_tile_pack(exp_max_diff);
    for (uint32_t r = 0; r < q_tiles; ++r) {
        tile_regs_acquire();
        merge_copy_tiles(a_max, r, 1, 0);
        copy_tile(R::max, r, 1);  // the state format, as the maxima
        tile_regs_commit();
        tile_regs_wait();
        PACK((SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_recipe_merge_corrections, (scale_fp32), 0, VectorMode::None)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_tile<true>(0, exp_max_diff, r);
        pack_tile<true>(1, R::corr, r);  // the state format, as exp_max_diff
        if (keep_max) {
            // The maxima stay pushed (one CB of q_tiles pages): row r at its write pointer, which is the CB's base.
            pack_tile<true>(2, a_max, r);
        }
        tile_regs_release();
    }
    CircularBuffer(exp_max_diff).push_back(q_tiles);
    CircularBuffer(R::corr).push_back(q_tiles);
    CircularBuffer(exp_max_diff).wait_front(q_tiles);
    CircularBuffer(R::corr).wait_front(q_tiles);
    // l = l_a * c_a + l_b * c_b, O = O_a * c_a + O_b * c_b.
    pack_reconfig_data_format(n_sum);
    configure_single_tile_pack(n_sum);
    for (uint32_t r = 0; r < q_tiles; ++r) {
        merge_state_tiles<max_w>(1, n_sum, r, R::sum, r, exp_max_diff, R::corr, r);
    }
    pack_reconfig_data_format(n_out);
    configure_single_tile_pack(n_out);
    for (uint32_t r = 0; r < q_tiles; ++r) {
        for (uint32_t j = 0; j < vd_tiles; j += max_w) {
            const uint32_t w = vd_tiles - j < max_w ? vd_tiles - j : max_w;
            merge_state_tiles<max_w>(
                w, n_out, r * vd_tiles * o_stride + j, R::out, r * vd_tiles + j, exp_max_diff, R::corr, r);
        }
    }
    CircularBuffer(exp_max_diff).pop_front(q_tiles);
    CircularBuffer(R::corr).pop_front(q_tiles);
    CircularBuffer(R::out).pop_front(q_tiles * vd_tiles);
    CircularBuffer(R::max).pop_front(q_tiles);
    CircularBuffer(R::sum).pop_front(q_tiles);
    if (keep_max) {
        recipe_pack_to_unpack_barrier();  // the next merge reads the new maxima
    }
}

// BF16 recipes: the (merged) state, O and l at CB 9's and CB 13's bases, through the recipe's normalization into
// `out`; the maxima popped. `normalize` is the caller's copy of normalize_row_streaming.
template <uint32_t q_tiles, uint32_t vd_tiles, typename Normalize>
ALWI void normalize_bf16(const RecipeAccumulatorState& state, const Normalize& normalize) {
    constexpr uint32_t o_stride = sdpa_out_stride;
    const uint32_t n_out = state.cur.out, n_sum = state.cur.sum;
    CircularBuffer(n_out).reserve_back(q_tiles * vd_tiles * o_stride);
    CircularBuffer(n_sum).reserve_back(q_tiles);
    CircularBuffer(n_out).push_back(q_tiles * vd_tiles * o_stride);
    CircularBuffer(n_sum).push_back(q_tiles);
    sdpa_cb_pop_front_out_of_line(state.prev.max, q_tiles);
    normalize(n_sum, n_out);
}

// FP32 recipes: the own state after its last chunk (not normalized; the pushed prev banks) merged with the received
// state straight into the normalized output CB `out`: per Q row one first-column pass for the two rescales s_a, s_b
// (normalization and output_scale folded in), then one pass per O tile, O_a s_a + O_b s_b. The own state is popped.
template <
    uint32_t q_tiles,
    uint32_t vd_tiles,
    uint32_t scale_fp32,
    uint32_t output_scale_bits,
    uint32_t exp_max_diff,
    uint32_t out,
    typename R>
void merge_normalize_fp32(const RecipeAccumulatorState& state) {
    const uint32_t a_out = state.prev.out, a_sum = state.prev.sum, a_max = state.prev.max;
    PACK(sdpa_pack.format_cb = INVALID_CB; sdpa_pack.width = 0;)
    CircularBuffer(R::out).wait_front(q_tiles * vd_tiles);
    CircularBuffer(R::max).wait_front(q_tiles);
    CircularBuffer(R::sum).wait_front(q_tiles);
    // s_a into CB exp_max_diff, s_b into R::corr (FP32, first column).
    CircularBuffer(exp_max_diff).reserve_back(q_tiles);
    CircularBuffer(R::corr).reserve_back(q_tiles);
    PACK((llk_pack_reconfig_l1_acc(0)));
    pack_reconfig_data_format(exp_max_diff);
    configure_single_tile_pack(exp_max_diff);
    for (uint32_t r = 0; r < q_tiles; ++r) {
        tile_regs_acquire();
        copy_state_tile(a_max, r, 0);
        copy_state_tile(R::max, r, 1);
        copy_state_tile(a_sum, r, 2);
        copy_tile(R::sum, r, 3);  // FP32 to dest as l_a: the same unpack configuration
        tile_regs_commit();
        tile_regs_wait();
        PACK((SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_recipe_merge_scales,
            (scale_fp32, output_scale_bits),
            0,
            VectorMode::C)));
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        pack_tile<true>(0, exp_max_diff, r);
        pack_tile<true>(1, R::corr, r);
        tile_regs_release();
    }
    CircularBuffer(exp_max_diff).push_back(q_tiles);
    CircularBuffer(R::corr).push_back(q_tiles);
    CircularBuffer(exp_max_diff).wait_front(q_tiles);
    CircularBuffer(R::corr).wait_front(q_tiles);
    CircularBuffer(out).reserve_back(q_tiles * vd_tiles);
    pack_reconfig_data_format(out);
    configure_single_tile_pack(out);
    sdpa_stream_reconfig_srca(a_out);  // every state CB here is FP32 unpacked to dest
    for (uint32_t r = 0; r < q_tiles; ++r) {
        for (uint32_t j = 0; j < vd_tiles; ++j) {
            tile_regs_acquire();
            copy_init(a_out);
            copy_tile(a_out, r * vd_tiles + j, 0);
            copy_tile(R::out, r * vd_tiles + j, 1);
            unary_bcast_init<BroadcastType::COL>(exp_max_diff);
            unary_bcast<BroadcastType::COL>(exp_max_diff, r, 2);
            unary_bcast<BroadcastType::COL>(R::corr, r, 3);
            unary_bcast_uninit<BroadcastType::COL>(exp_max_diff);
            tile_regs_commit();
            tile_regs_wait();
            PACK((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_recipe_merge_normalize, 0, VectorMode::None)));
            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
            pack_tile<true>(0, out, r * vd_tiles + j);
            tile_regs_release();
        }
    }
    CircularBuffer(out).push_back(q_tiles * vd_tiles);
    CircularBuffer(exp_max_diff).pop_front(q_tiles);
    CircularBuffer(R::corr).pop_front(q_tiles);
    CircularBuffer(R::out).pop_front(q_tiles * vd_tiles);
    CircularBuffer(R::max).pop_front(q_tiles);
    CircularBuffer(R::sum).pop_front(q_tiles);
    sdpa_cb_pop_front_out_of_line(a_out, q_tiles * vd_tiles);
    sdpa_cb_pop_front_out_of_line(a_sum, q_tiles);
    sdpa_cb_pop_front_out_of_line(a_max, q_tiles);
    PACK(sdpa_pack.format_cb = INVALID_CB; sdpa_pack.width = 0;)
}
}  // namespace recipe_merge
