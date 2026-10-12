// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// grid_sample, nearest, sharded bf16 grid: the three compute RISCs (UNPACK, MATH, PACK) turn the core's
// grid shard of normalized (x, y) pairs into int16 (h, w) pixel indices in an L1 buffer, a third of the
// sticks each, so the two data-movement RISCs only read indices and issue the NoC reads (the same work as
// the precomputed-grid path). No math unit is used. Progress is published through the page counters of
// three CBs the firmware zeroes at launch, written as plain stream registers (no LLK push: those go
// through Tensix instructions that stall on packer/unpacker state this kernel never sets up).
// The firmware only initialises UNPACK's read-side and PACK's write-side CB interfaces; MATH reads the
// CB addresses straight from the CB config table of the launch message.

#include <cstdint>
#include "api/compute/common.h"
#include "internal/circular_buffer_interface.h"
#include "stream_io_map.h"
#include "internal/circular_buffer_init.h"
#define GRID_SAMPLE_NEAREST_NO_TABLE
#include "ttnn/cpp/ttnn/operations/pool/grid_sample/device/kernels/grid_sample_nearest_index.hpp"

// The index word for sample k of stick s. Without de-duplication: int16 h, int16 w (the precomputed-grid
// layout). With it: 0xFFFFFFFF outside the image; bit 31 | sample index of the first occurrence of the same
// stick handed to the same reader by this thread (served by a local copy); otherwise the stick index with
// the batch folded in (a NoC read). With two readers the samples alternate between them (reader 0 takes
// the even sample indices s * K + k, reader 1 the odd ones), so the table is chosen by the sample's parity.
template <bool dedup, bool split_reader, uint32_t H, uint32_t W, uint32_t grid_hw, uint32_t ENTRIES, uint32_t K>
inline __attribute__((always_inline)) uint32_t publish_index(
    uint32_t* const* tab, uint32_t grid_stick_start, uint32_t s, uint32_t k, uint32_t h, uint32_t w) {
    if constexpr (!dedup) {
        return h | (w << 16);
    } else {
        if (h == 0xFFFFu || w == 0xFFFFu) {
            return 0xFFFFFFFFu;
        }
        const uint32_t batch = (grid_stick_start + s) / grid_hw;
        const uint32_t stick = batch * (H * W) + h * W + w;
        const uint32_t sample = s * K + k;
        uint32_t* e = tab[split_reader ? (sample & 1) : 0] + (stick & (ENTRIES - 1)) * 2;
        if (e[0] == stick + 1) {
            return 0x80000000u | e[1];
        }
        e[0] = stick + 1;
        e[1] = sample;
        return stick;
    }
}

void kernel_main() {
    constexpr uint32_t grid_cb = get_compile_time_arg_val(0);
    constexpr uint32_t idx_cb = get_compile_time_arg_val(1);
    constexpr uint32_t prog_cb0 = get_compile_time_arg_val(2);
    constexpr uint32_t prog_cb1 = get_compile_time_arg_val(3);
    constexpr uint32_t prog_cb2 = get_compile_time_arg_val(4);
    constexpr uint32_t nsticks = get_compile_time_arg_val(5);
    constexpr uint32_t K = get_compile_time_arg_val(6);
    constexpr uint32_t input_height = get_compile_time_arg_val(7);
    constexpr uint32_t input_width = get_compile_time_arg_val(8);
    constexpr bool align_corners = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(10);
    constexpr bool gather_grid = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t grid_ready_cb = get_compile_time_arg_val(12);
    constexpr uint32_t grid_ready_cb1 = get_compile_time_arg_val(13);
    constexpr uint32_t stage_cb = get_compile_time_arg_val(14);
    constexpr uint32_t grid_page_aligned_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t num_grid_banks = get_compile_time_arg_val(16);
    constexpr uint32_t stage_run_bytes = get_compile_time_arg_val(17);
    // Read de-duplication: a grid that maps many cells to one image region sends most of the readers' NoC
    // reads to a few source cores, and the op is then bound by those cores' NoC ports. Each thread keeps,
    // per reader, a direct-mapped table of the sticks it has already handed out (stick index -> sample
    // index of the first occurrence); a repeat is published as "copy from that sample" and the reader
    // serves it with a local copy instead of a NoC read. The index word is then: 0xFFFFFFFF outside the
    // image, bit 31 set = copy from sample (low bits), otherwise the stick index (batch included).
    constexpr uint32_t dedup_cb = get_compile_time_arg_val(18);
    constexpr bool dedup = get_compile_time_arg_val(19) != 0;
    constexpr uint32_t grid_hw = get_compile_time_arg_val(20);
    constexpr bool split_reader = get_compile_time_arg_val(21) != 0;
    constexpr uint32_t DEDUP_ENTRIES = 256;  // per table, 8 B per entry; must match the program factory and the writer
    const uint32_t grid_stick_start = get_arg_val<uint32_t>(0);  // this core's first grid stick (gather mode)

    uint32_t t = 0;
    uint32_t grid_base = 0;
    uint32_t idx_base = 0;
    uint32_t stage_base = 0;
    constexpr uint32_t src_cb = gather_grid ? stage_cb : grid_cb;
    uint32_t dedup_base = 0;
    UNPACK({
        t = 0;
        grid_base = get_local_cb_interface(src_cb).fifo_rd_ptr << cb_addr_shift;
        idx_base = get_local_cb_interface(idx_cb).fifo_rd_ptr << cb_addr_shift;
        if constexpr (dedup) {
            dedup_base = get_local_cb_interface(dedup_cb).fifo_rd_ptr << cb_addr_shift;
        }
    });
    MATH({
        t = 1;
        tt_l1_ptr mailboxes_t* const mb = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
        launch_msg_t* lm = &(mb->launch[mb->launch_msg_rd_ptr]);
        const uint32_t kcb = lm->kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX];
        uint32_t tt_l1_ptr* cb_l1_base = (uint32_t tt_l1_ptr*)(kcb + lm->kernel_config.local_cb_offset);
        grid_base = cb_l1_base[src_cb * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG];
        idx_base = cb_l1_base[idx_cb * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG];
        if constexpr (dedup) {
            dedup_base = cb_l1_base[dedup_cb * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG];
        }
    });
    PACK({
        t = 2;
        grid_base = get_local_cb_interface(src_cb).fifo_wr_ptr << cb_addr_shift;
        idx_base = get_local_cb_interface(idx_cb).fifo_wr_ptr << cb_addr_shift;
        if constexpr (dedup) {
            dedup_base = get_local_cb_interface(dedup_cb).fifo_wr_ptr << cb_addr_shift;
        }
    });
    stage_base = grid_base;
    // this thread's two tables (one per reader) sit at tables [2t, 2t+1] of the six in the dedup CB
    uint32_t* dedup_tab[2] = {nullptr, nullptr};
    if constexpr (dedup) {
        for (uint32_t r = 0; r < 2; ++r) {
            dedup_tab[r] = reinterpret_cast<uint32_t*>(dedup_base + (2 * t + r) * DEDUP_ENTRIES * 8);
            for (uint32_t i = 0; i < DEDUP_ENTRIES * 2; i += 2) {
                dedup_tab[r][i] = 0;  // key 0 = empty
            }
        }
    }
    volatile uint32_t* prog = get_cb_tiles_received_ptr(t == 0 ? prog_cb0 : (t == 1 ? prog_cb1 : prog_cb2));
    // gather mode: the readers stage the bank runs in GATHER_CHUNKS slices of the in-bank index (same split as in
    // the writer kernel); wait for the slice a stick needs, from both readers, before reading its pages
    constexpr uint32_t GATHER_CHUNKS = 8;
    constexpr uint32_t npages_total = nsticks * K;
    constexpr uint32_t max_count = (npages_total + num_grid_banks - 1) / num_grid_banks;
    constexpr uint32_t ic = (max_count + GATHER_CHUNKS - 1) / GATHER_CHUNKS;
    volatile uint32_t* ready0 = get_cb_tiles_received_ptr(grid_ready_cb);
    volatile uint32_t* ready1 = get_cb_tiles_received_ptr(grid_ready_cb1);
    uint32_t have = 0;

    // Sticks are dealt in groups of 8 (UNPACK: slots 0-2, MATH: 3-4, PACK: 5-7) so the readers, which walk
    // the sticks in order, are fed by all three threads at once; MATH gets fewer because it runs this scalar
    // loop ~1.4x slower than UNPACK / PACK (measured). Letting the readers compute a share themselves was
    // slower (their loop is the NoC-issue critical path).
    constexpr uint32_t OFF[3] = {0, 3, 5};
    constexpr uint32_t CNT[3] = {3, 2, 3};
    uint32_t done = 0;
    for (uint32_t base = 0; base < nsticks; base += 8) {
      for (uint32_t s = base + OFF[t]; s < base + OFF[t] + CNT[t] && s < nsticks; ++s) {
        volatile uint32_t* o = reinterpret_cast<volatile uint32_t*>(idx_base + s * K * 4);
        if constexpr (gather_grid) {
            // page u = s*K + k of this core's range sits in bank (r0 + u) % NB at in-bank index u / NB of its staged run
            const uint32_t need = ((s * K + K - 1) / num_grid_banks) / ic + 1;
            if (need > have) {
                while (static_cast<int16_t>(static_cast<uint16_t>(*ready0) - static_cast<uint16_t>(need)) < 0) {
                }
                while (static_cast<int16_t>(static_cast<uint16_t>(*ready1) - static_cast<uint16_t>(need)) < 0) {
                }
                have = need;
            }
            const uint32_t r0 = (grid_stick_start * K) % num_grid_banks;
            const uint32_t u0 = s * K;
            uint32_t um = u0 % num_grid_banks;
            uint32_t ib = u0 / num_grid_banks;
            uint32_t b = r0 + um;
            if (b >= num_grid_banks) {
                b -= num_grid_banks;
            }
            for (uint32_t k = 0; k < K; ++k) {
                volatile uint32_t* pg = reinterpret_cast<volatile uint32_t*>(stage_base + b * stage_run_bytes + ib * grid_page_aligned_bytes);
                const uint32_t xy = *pg;  // x in the low half, y in the high half
                const uint32_t w = static_cast<uint16_t>(nearest_index_bf16<align_corners, input_width>(static_cast<uint16_t>(xy & 0xFFFF)));
                const uint32_t h = static_cast<uint16_t>(nearest_index_bf16<align_corners, input_height>(static_cast<uint16_t>(xy >> 16)));
                o[k] = publish_index<dedup, split_reader, input_height, input_width, grid_hw, DEDUP_ENTRIES, K>(dedup_tab, grid_stick_start, s, k, h, w);
                if (++um == num_grid_banks) {
                    um = 0;
                    ++ib;
                }
                if (++b == num_grid_banks) {
                    b = 0;
                }
            }
        } else {
            volatile uint16_t* g = reinterpret_cast<volatile uint16_t*>(grid_base + s * grid_stick_nbytes);
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t w = static_cast<uint16_t>(nearest_index_bf16<align_corners, input_width>(g[2 * k]));       // x
                const uint32_t h = static_cast<uint16_t>(nearest_index_bf16<align_corners, input_height>(g[2 * k + 1]));  // y
                o[k] = publish_index<dedup, split_reader, input_height, input_width, grid_hw, DEDUP_ENTRIES, K>(dedup_tab, grid_stick_start, s, k, h, w);
            }
        }
        *prog = static_cast<uint16_t>(++done);
      }
    }
}
