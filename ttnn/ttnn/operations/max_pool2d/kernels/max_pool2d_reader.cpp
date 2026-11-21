// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// max_pool2d reader: reader thread p of R. Two block operations per staging block (op_design.md):
//
//   stage_receptive_field   the block's receptive field, input sticks [s_lo, s_hi) x this core's channel run, is
//                           copied into the `staging` scratchpad once. The copy is split into R contiguous parts,
//                           one per thread; one barrier on trid 0, then sync_threads() so every thread sees all of
//                           it. Skipped on an in-place core: its own width shard already is the receptive field.
//   gather_window_taps      for each stream tile u a NEO folds, its K tap stream tiles are assembled in the
//                           `window_taps` ring by core-local copies out of the staging (or the own shard): tap tile
//                           (u, k) holds, for the positions tile u covers, the sticks tap k of their windows reads,
//                           `pitch` elements per position. Consecutive positions whose source sticks are
//                           contiguous are one read. A padded tap reads a real tap of the same window instead.
//
// Dealing: in block b, NEO t owns stream tiles [b*C*bt + t*bt_b, + bt_b); reader thread p feeds NEOs p, p+R, ...
// one window each in strict rotation (STRIDED ring). A NEO past the core's last real tile recomputes that tile.
//
// Explicit sync on `window_taps` (the host opts it out of implicit sync): an entry is assembled from many reads.
// Every NoC transfer carries the user transaction id 0 (target page, ring protocols).
//
// Raw dataflow calls throughout: the design's API mapping names no dataflow helper for the stage or the gather
// beyond the self-aimed (loopback) source, which the Scratchpad / LocalTensorAccessor endpoints provide.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t R = get_arg(args::reader_threads);
constexpr uint32_t C = get_arg(args::compute_threads);
constexpr uint32_t H = get_arg(args::H);
constexpr uint32_t W = get_arg(args::W);
constexpr uint32_t H_out = get_arg(args::H_out);
constexpr uint32_t W_out = get_arg(args::W_out);
constexpr uint32_t kh = get_arg(args::kh);
constexpr uint32_t kw = get_arg(args::kw);
constexpr uint32_t sh = get_arg(args::sh);
constexpr uint32_t sw = get_arg(args::sw);
constexpr uint32_t ph = get_arg(args::ph);
constexpr uint32_t pw = get_arg(args::pw);
constexpr uint32_t dh = get_arg(args::dh);
constexpr uint32_t dw = get_arg(args::dw);
constexpr uint32_t K = kh * kw;
constexpr uint32_t kh_eff = dh * (kh - 1) + 1;
constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);  // one stream tile = one ring entry
constexpr uint32_t esize = get_arg(args::esize);
constexpr uint32_t pitch_bytes = get_arg(args::pitch_bytes);  // bytes per position in the streams and the staging
constexpr uint32_t pitch = pitch_bytes / esize;
constexpr uint32_t tile_elems = tile_bytes / esize;
constexpr uint32_t taps_per_batch = get_arg(args::taps_per_batch);
constexpr bool tile_input = get_arg(args::tile_input) != 0;
constexpr uint32_t in_tile_bytes = get_arg(args::in_tile_bytes);
// Channel tiles of the run (tile layout only; 0 for row-major, read as 1 so the unused tile path compiles).
constexpr uint32_t channel_tiles = get_arg(args::channel_tiles) ? get_arg(args::channel_tiles) : 1;
constexpr uint32_t in_ct_total = get_arg(args::in_ct_total);
constexpr uint32_t in_page_stride = get_arg(args::in_page_stride);
constexpr bool input_sharded = get_arg(args::input_sharded) != 0;
constexpr bool input_width_sharded = get_arg(args::input_width_sharded) != 0;
constexpr uint32_t face_row_elems = 16;                        // a TILE page: four 16 x 16 faces
constexpr uint32_t face_row_bytes = face_row_elems * esize;
constexpr uint32_t face_bytes = 16 * face_row_bytes;
constexpr NocOptVals user_txn = {.trid = 0};

static_assert(C % R == 0, "R must divide C on a STRIDED ring");
static_assert(K % taps_per_batch == 0, "a tap batch must divide the window");
static_assert(!tile_input || pitch % 32 == 0, "a tile-layout channel run is whole 32-channel tiles");

struct Assignment {
    uint32_t pos_start;
    uint32_t pos_count;
    uint32_t num_tiles;
    uint32_t block_tiles;
    uint32_t in_page_col;
    uint32_t in_chan_offset;
    uint32_t in_ct0;
};

// The nearest in-image index of a window index `v` (a padded position) on the window's dilation lattice: a real
// tap of the same window, so the padding never wins and is never stored (rule R1).
FORCE_INLINE uint32_t in_image(int32_t v, int32_t extent, int32_t step) {
    if (v < 0) {
        v += ((-v + step - 1) / step) * step;
    } else if (v > extent - 1) {
        v -= ((v - (extent - 1) + step - 1) / step) * step;
    }
    return static_cast<uint32_t>(v);
}

// [s_lo, s_hi): input sticks output positions [q0, q1) read, as whole image rows (the host's
// Geometry.receptive_sticks).
FORCE_INLINE void receptive_field(uint32_t q0, uint32_t q1, uint32_t& s_lo, uint32_t& s_hi) {
    constexpr uint32_t plane = H_out * W_out;
    const uint32_t img0 = q0 / plane, i0 = (q0 % plane) / W_out;
    const uint32_t img1 = (q1 - 1) / plane, i1 = ((q1 - 1) % plane) / W_out;
    const int32_t r_lo = static_cast<int32_t>(i0 * sh) - static_cast<int32_t>(ph);
    const int32_t r_hi = static_cast<int32_t>(i1 * sh + kh_eff) - static_cast<int32_t>(ph) - 1;
    const uint32_t lo = r_lo < 0 ? 0u : static_cast<uint32_t>(r_lo);
    const uint32_t hi = r_hi > static_cast<int32_t>(H - 1) ? H - 1 : static_cast<uint32_t>(r_hi);
    s_lo = (img0 * H + lo) * W;
    s_hi = (img1 * H + hi + 1) * W;
}

// stage_receptive_field: this thread's contiguous part of the block's receptive field into the staging.
template <typename InAccessor, typename Staging>
FORCE_INLINE void stage_receptive_field(
    const Noc& noc, const InAccessor& in, const Staging& staging, const Assignment& a, uint32_t s_lo, uint32_t s_hi,
    uint32_t thread) {
    if constexpr (tile_input) {
        // Whole tile pages for tile rows [tr_lo, tr_hi] x the run's channel tiles, row-major in the staging.
        const uint32_t tr_lo = s_lo / 32;
        const uint32_t units = ((s_hi - 1) / 32 - tr_lo + 1) * channel_tiles;
        const uint32_t first = units * thread / R, last = units * (thread + 1) / R;
        for (uint32_t idx = first; idx < last;) {
            const uint32_t tr = tr_lo + idx / channel_tiles, ct = idx % channel_tiles;
            const uint32_t page = tr * in_ct_total + a.in_ct0 + ct;
            // Pages consecutive in the staging are consecutive page ids within a tile row, and across rows when
            // the run spans every channel tile; a sharded source holds such runs contiguously.
            uint32_t run = 1;
            if constexpr (input_sharded) {
                const uint32_t row_end = channel_tiles == in_ct_total ? last : idx - ct + channel_tiles;
                const uint32_t span = (row_end < last ? row_end : last) - idx;
                run = in.num_contiguous_pages(page, page + span);
            }
            noc.async_read<NocOptions::TXN_ID>(
                in, staging, run * in_tile_bytes, {.page_id = page, .offset_bytes = 0},
                {.offset_bytes = idx * in_tile_bytes}, user_txn);
            idx += run;
        }
    } else {
        // One stick of pitch_bytes per input row: channels [chan, chan + pitch) of row s.
        const uint32_t units = s_hi - s_lo;
        const uint32_t first = s_lo + units * thread / R, last = s_lo + units * (thread + 1) / R;
        for (uint32_t s = first; s < last;) {
            uint32_t run = 1;
            if constexpr (input_sharded && in_page_stride == 1) {
                // Height-sharded: whole rows, contiguous within a shard.
                run = in.num_contiguous_pages(s, last);
            }
            noc.async_read<NocOptions::TXN_ID>(
                in, staging, run * pitch_bytes,
                {.page_id = s * in_page_stride + a.in_page_col, .offset_bytes = a.in_chan_offset},
                {.offset_bytes = (s - s_lo) * pitch_bytes}, user_txn);
            s += run;
        }
    }
    noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
}

// One pending core-local copy; consecutive segments that continue it in source and destination extend it.
struct Run {
    uint32_t src = 0, dst = 0, len = 0;
};

// gather_window_taps for one entry: tap `tap` of stream tile `u`, into the reserved window at `dst_base`.
template <typename Source, typename Taps>
FORCE_INLINE void gather_tap_tile(
    const Noc& noc, const Source& src, Taps& taps, const Assignment& a, uint32_t u, uint32_t tap, uint32_t s_lo,
    uint32_t dst_base) {
    const uint32_t a_row = tap / kw, b_col = tap % kw;
    const uint32_t e0 = u * tile_elems;
    const uint32_t total = a.pos_count * pitch;
    const uint32_t e1 = e0 + tile_elems < total ? e0 + tile_elems : total;  // the last tile's tail is left unread

    uint32_t q = e0 / pitch;  // local position
    const uint32_t p = a.pos_start + q;
    constexpr uint32_t plane = H_out * W_out;
    uint32_t img = p / plane, i = (p % plane) / W_out, j = p % W_out;

    Run run;
    auto flush = [&]() __attribute__((always_inline)) {
        if (run.len) {
            noc.async_read<NocOptions::TXN_ID>(
                src, taps, run.len, {.offset_bytes = run.src}, {.offset_bytes = dst_base + run.dst}, user_txn);
        }
    };
    auto add = [&](uint32_t s_off, uint32_t d_off, uint32_t len) __attribute__((always_inline)) {
        if (run.len && s_off == run.src + run.len && d_off == run.dst + run.len) {
            run.len += len;
        } else {
            flush();
            run = {s_off, d_off, len};
        }
    };

    uint32_t dst = 0;
    for (uint32_t e = e0; e < e1; ++q) {
        const uint32_t stick_end = (q + 1) * pitch;
        const uint32_t seg_end = stick_end < e1 ? stick_end : e1;
        const uint32_t c_lo = e - q * pitch, c_hi = seg_end - q * pitch;  // elements within the stick
        const uint32_t row = in_image(
            static_cast<int32_t>(i * sh + a_row * dh) - static_cast<int32_t>(ph), static_cast<int32_t>(H),
            static_cast<int32_t>(dh));
        const uint32_t col = in_image(
            static_cast<int32_t>(j * sw + b_col * dw) - static_cast<int32_t>(pw), static_cast<int32_t>(W),
            static_cast<int32_t>(dw));
        const uint32_t s = (img * H + row) * W + col;
        if constexpr (tile_input) {
            // A stick's 32 channels of one tile are two 16-element face rows.
            const uint32_t tr = s / 32 - s_lo / 32, r = s % 32;
            for (uint32_t c = c_lo; c < c_hi; c += face_row_elems) {
                const uint32_t face = 2 * (r / 16) + (c / 16) % 2;
                const uint32_t off = (tr * channel_tiles + c / 32) * in_tile_bytes + face * face_bytes +
                                     (r % 16) * face_row_bytes;
                add(off, dst, face_row_bytes);
                dst += face_row_bytes;
            }
        } else {
            const uint32_t len = (c_hi - c_lo) * esize;
            add((s - s_lo) * pitch_bytes + c_lo * esize, dst, len);
            dst += len;
        }
        e = seg_end;
        if (++j == W_out) {
            j = 0;
            if (++i == H_out) {
                i = 0;
                ++img;
            }
        }
    }
    flush();
}

// The block schedule of one core: stage (unless in place), then gather every NEO's taps, block by block.
template <bool kInPlace, typename InAccessor, typename Source, typename Staging>
FORCE_INLINE void run_schedule(
    const Noc& noc, const InAccessor& in, const Source& src, const Staging& staging, const Assignment& a) {
    DataflowBuffer taps(dfb::window_taps);
    const uint32_t entry_stride = taps.get_stride_size();  // bytes between this thread's consecutive entries
    const uint32_t thread = get_my_thread_id();
    const uint32_t T = a.num_tiles;
    const uint32_t Q = (T + C - 1) / C;
    const uint32_t bt = a.block_tiles;
    const uint32_t num_blocks = (Q + bt - 1) / bt;

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const uint32_t bt_b = Q - b * bt < bt ? Q - b * bt : bt;
        const uint32_t u0 = b * C * bt;
        const uint32_t u1 = u0 + C * bt_b < T ? u0 + C * bt_b : T;
        uint32_t s_lo = 0, s_hi = 0;
        if constexpr (!kInPlace) {
            const uint32_t q0 = a.pos_start + (u0 * tile_elems) / pitch;
            const uint32_t e1 = u1 * tile_elems < a.pos_count * pitch ? u1 * tile_elems : a.pos_count * pitch;
            const uint32_t q1 = a.pos_start + (e1 + pitch - 1) / pitch;
            receptive_field(q0, q1, s_lo, s_hi);
            if (b > 0) {
                sync_threads();  // every thread's gathers of block b-1 are done before the staging is overwritten
            }
            stage_receptive_field(noc, in, staging, a, s_lo, s_hi, thread);
            sync_threads();  // the whole receptive field has landed
        }
        for (uint32_t i = 0; i < bt_b; ++i) {
            for (uint32_t batch = 0; batch < K / taps_per_batch; ++batch) {
                for (uint32_t g = 0; g < C / R; ++g) {
                    const uint32_t neo = thread + g * R;
                    uint32_t u = u0 + neo * bt_b + i;
                    if (u >= T) {
                        u = T - 1;  // padded round: recompute the core's last real tile
                    }
                    taps.reserve_back(taps_per_batch);
                    for (uint32_t k = 0; k < taps_per_batch; ++k) {
                        gather_tap_tile(noc, src, taps, a, u, batch * taps_per_batch + k, s_lo, k * entry_stride);
                    }
                    noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
                    taps.push_back(taps_per_batch);
                }
            }
        }
    }
}

// The in-place path names the input's local region; built only for a width-sharded input (an L1 tensor), so the
// token is a template argument: LocalTensorAccessor refuses a DRAM tensor at compile time.
template <typename Token, typename InAccessor, typename Staging>
FORCE_INLINE void run_in_place(const Noc& noc, Token token, const InAccessor& in, const Staging& staging,
                               const Assignment& a) {
    const LocalTensorAccessor<uint8_t> own_shard(token);
    run_schedule<true>(noc, in, own_shard, staging, a);
}

}  // namespace

void kernel_main() {
    const Assignment a{
        .pos_start = get_arg(args::pos_start),
        .pos_count = get_arg(args::pos_count),
        .num_tiles = get_arg(args::num_tiles),
        .block_tiles = get_arg(args::block_tiles),
        .in_page_col = get_arg(args::in_page_col),
        .in_chan_offset = get_arg(args::in_chan_offset),
        .in_ct0 = get_arg(args::in_ct0),
    };
    const bool in_place = get_arg(args::in_place) != 0;

    Noc noc;
    const auto in = TensorAccessor(tensor::input);
    const Scratchpad<uint8_t> staging(scratch::staging);

    bool done = false;
    if constexpr (input_width_sharded) {
        if (in_place) {
            run_in_place(noc, tensor::input, in, staging, a);
            done = true;
        }
    }
    if (!done) {
        run_schedule<false>(noc, in, staging, staging, a);
    }
    // Every transfer above carried the user id, so the hart's tag register already holds it; set it anyway, so a
    // program of tt-metal's own that follows with untagged transfers inherits no ring's id.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
