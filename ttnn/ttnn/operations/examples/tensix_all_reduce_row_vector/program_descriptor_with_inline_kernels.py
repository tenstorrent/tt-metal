# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-row-vector all-reduce: a row reduction split across K Tensix cores.

`tensix_all_reduce` sums WHOLE TILES elementwise. A very common special case only needs one value
per row: a row is WIDTH_SHARDED across K Tensix cores ("members"), b tile-rows and w tiles per
member per tile-row, bf16 in L1. Each member row-reduces its slice (REDUCE_ROW, float32 DST), which
leaves its per-row partial in COLUMN 0 of one float32 tile per tile-row; the K partials of a
tile-row are added by that tile-row's OWNER (one member per tile-row), and every member needs the
per-row totals back. The output is that column-0 tile at every member (float32, one tile per
tile-row, WIDTH_SHARDED like the input): column 0 = sum over the WHOLE row, the other 31 columns
don't-care. The result is never expanded across the tile; a consumer applies it with a column
broadcast (`mul_tiles_bcast_cols` and friends).

Two kernel-level choices vary; the row reduce and the owner placement are identical in every variant.

Concept 1, bytes per member on the wire, both directions.
  Gather (member -> owner), the variant prefix:
  - `tile_partial_*`:  the whole float32 partial tile, 4096 B per tile-row; the owner adds K tiles.
  - `faces_partial_*`: only faces 0 and 2, the two 16x16 faces that hold column 0, 2 x 1024 B =
                       2048 B per tile-row; the owner still adds K tiles (faces 1/3 are don't-care).
  - `row_partial_*`:   the member transposes the partial so the 32 values are row 0 and ships that row,
                       2 x 64 B = 128 B per tile-row, into row m % 32 of gather tile m // 32 at the
                       owner; the owner column-reduces ceil(K / 32) gather tiles instead of adding K,
                       and transposes the one result back to column 0.
  Return (owner -> members): the whole column-0 tile (4096 B), or with `_faces_return` only its
  faces 0/2 (2 x 1024 B).
Concept 2, the return path, the variant suffix:
  - `*_unicast`: every owner writes its result tile to all K members, then bumps each one's semaphore.
  - `*_mcast`:   owners write their result tile to one root (member 0); the root multicasts all b
                 result tiles to the members' bounding rectangle, then multicasts the semaphore.

Kernels: compute (row reduce, [transpose], owner combine), a gather kernel on RISCV_1 + NoC0
(scaler tile, partials to the owners, wait for the result), a return kernel on RISCV_0 + NoC1
(owner hand-off and the return).
"""

from dataclasses import dataclass

import ttnn

RV_PARTIALS = ("tile", "faces", "row")
RV_RETURNS = ("tile", "faces")
RV_PATHS = ("unicast", "mcast")


def row_vector_variant_name(partial, returned, path):
    return f"{partial}_partial{'_faces_return' if returned == 'faces' else ''}_{path}"


# Every combination builds; ROW_VECTOR_VARIANTS is the measured set, baseline first.
ALL_ROW_VECTOR_VARIANTS = tuple(
    row_vector_variant_name(p, r, path) for p in RV_PARTIALS for r in RV_RETURNS for path in RV_PATHS
)
ROW_VECTOR_VARIANTS = (
    "tile_partial_unicast",
    "faces_partial_unicast",
    "row_partial_unicast",
    "tile_partial_mcast",
    "faces_partial_mcast",
    "row_partial_mcast",
    "faces_partial_faces_return_unicast",
    "faces_partial_faces_return_mcast",
)

FP32_TILE_BYTES = 4096

RV_CB_IN = 0
RV_CB_SCALER = 1
RV_CB_PARTIAL = 2
RV_CB_TMP = 3
RV_CB_GATHER = 4
RV_CB_OWNED = 5

RV_SEM_GATHER = 0  # on an owner: +1 per member partial landed
RV_SEM_RESULT = 1  # on a member: result landed (+1 per owner for unicast, iteration stamp for mcast)
RV_SEM_COLLECT = 2  # on the root (mcast): +1 per owner result landed
RV_SEM_LOCAL = 3  # on the root (mcast): the word the semaphore multicast copies from

RV_NOT_OWNER = 0xFFFFFFFF


def parse_row_vector_variant(variant):
    """(partial, returned, path), e.g. faces_partial_faces_return_mcast -> (faces, faces, mcast)."""
    for p in RV_PARTIALS:
        for r in RV_RETURNS:
            for path in RV_PATHS:
                if row_vector_variant_name(p, r, path) == variant:
                    return p, r, path
    raise ValueError(f"variant must be one of {ALL_ROW_VECTOR_VARIANTS}, got {variant!r}")


@dataclass(frozen=True)
class MemberLayout:
    """K members packed row-major into a rows x cols rectangle at logical (0, 0)."""

    num_members: int
    rows: int
    cols: int
    members: tuple

    @property
    def member_ranges(self):
        ranges = []
        full_rows = self.num_members // self.cols
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.cols - 1, full_rows - 1)))
        tail = self.num_members % self.cols
        if tail:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(tail - 1, full_rows)))
        return ttnn.CoreRangeSet(ranges)

    @property
    def bbox_ranges(self):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.cols - 1, self.rows - 1))])

    @property
    def placement(self):
        tail = self.num_members % self.cols
        shape = f"{self.rows}x{self.cols}"
        return shape if tail == 0 else f"{shape} (last row {tail})"


def build_member_layout(device, num_members):
    """The widest exact rectangle that fits the worker grid; else whole grid rows plus a partial row."""
    grid = device.compute_with_storage_grid_size()
    if num_members < 2:
        raise ValueError("a row all-reduce needs at least two members")
    if num_members > grid.x * grid.y:
        raise ValueError(f"{num_members} members do not fit the {grid.x}x{grid.y} worker grid")
    rows = cols = None
    for c in range(min(grid.x, num_members), 0, -1):
        if num_members % c == 0 and num_members // c <= grid.y:
            rows, cols = num_members // c, c
            break
    if cols is None:
        # No exact rectangle: whole grid rows, then a partial row. The multicast return then also
        # reaches the bounding rectangle's few non-member Tensix cores (they hold only the result
        # semaphore; the output's L1 range is reserved on every Tensix core).
        cols = grid.x
        rows = -(-num_members // cols)
    members = tuple((i % cols, i // cols) for i in range(num_members))
    return MemberLayout(num_members, rows, cols, members)


def row_owner_indices(layout, tile_rows):
    """Member index that owns each tile-row. Owners sit on a diagonal of the members' rectangle
    (tile-row r -> grid column r * cols / b, grid row r * rows / b), so no two owners share a grid
    column when b <= cols: owners stacked in one column share that column's NoC links for every
    incoming partial. Falls back to an even spread over the member order."""
    k, rows, cols = layout.num_members, layout.rows, layout.cols
    diagonal = [((r * rows) // tile_rows) * cols + (r * cols) // tile_rows for r in range(tile_rows)]
    if all(i < k for i in diagonal) and len(set(diagonal)) == tile_rows:
        return diagonal
    return [(r * k) // tile_rows for r in range(tile_rows)]


def create_row_vector_memory_config(device, num_members, tile_rows, width_tiles):
    """WIDTH_SHARDED over the K members: (b x 32) x (w x 32) per member (w = 1 for the output)."""
    layout = build_member_layout(device, num_members)
    return ttnn.create_sharded_memory_config(
        shape=(tile_rows * ttnn.TILE_SIZE, width_tiles * ttnn.TILE_SIZE),
        core_grid=layout.member_ranges,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def num_gather_tiles(variant, num_members):
    """Tiles an owner reduces per tile-row: K partial tiles, or ceil(K / 32) gather tiles of rows."""
    return -(-num_members // ttnn.TILE_SIZE) if parse_row_vector_variant(variant)[0] == "row" else num_members


# Compute: per-member row reduce -> [transpose] -> owner combine. Differs only by ROW_PARTIAL
# (FACES_PARTIAL and FACES_RETURN change only what the dataflow kernels move).
# RT args: [tile_rows, width_tiles, is_owner, num_gather, kernel_iters]
_RV_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_partial = 2, cb_tmp = 3, cb_gather = 4, cb_owned = 5;

// Transpose n tiles of icb into ocb (row 0 <-> column 0), one tile per wait/pop: cb_tmp carries both
// the b member partials and the owner's single result, so its quantum must stay 1 tile.
ALWI void transpose_tiles(uint32_t icb, uint32_t ocb, uint32_t n) {
    reconfig_data_format_srca(icb);
    pack_reconfig_data_format(ocb);
    transpose_init(icb);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(icb, 1);
        cb_reserve_back(ocb, 1);
        tile_regs_acquire();
        transpose_tile(icb, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, ocb);
        tile_regs_release();
        cb_push_back(ocb, 1);
        cb_pop_front(icb, 1);
    }
}

void kernel_main() {
    const uint32_t tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t width_tiles = get_arg_val<uint32_t>(1);
    const uint32_t is_owner = get_arg_val<uint32_t>(2);
    const uint32_t num_gather = get_arg_val<uint32_t>(3);
    const uint32_t kernel_iters = get_arg_val<uint32_t>(4);

#ifdef ROW_PARTIAL
    constexpr uint32_t cb_reduced = cb_tmp;      // transposed before it ships
#else
    constexpr uint32_t cb_reduced = cb_partial;  // ships as is
#endif

    compute_kernel_hw_startup(cb_scaler, cb_in, cb_reduced);
    cb_wait_front(cb_in, tile_rows * width_tiles);  // resident shard, read by index, never popped
    cb_wait_front(cb_scaler, 1);

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        // (1) Member row reduce: one float32 partial tile per tile-row, the sums in column 0.
        reconfig_data_format(cb_scaler, cb_in);
        pack_reconfig_data_format(cb_reduced);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_reduced);
        for (uint32_t r = 0; r < tile_rows; ++r) {
            cb_reserve_back(cb_reduced, 1);
            tile_regs_acquire();
            for (uint32_t t = 0; t < width_tiles; ++t) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, r * width_tiles + t, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_reduced);
            tile_regs_release();
            cb_push_back(cb_reduced, 1);
        }
        reduce_uninit();
#ifdef ROW_PARTIAL
        // Column 0 -> row 0: the 32 values become two contiguous 64 B face rows.
        transpose_tiles(cb_tmp, cb_partial, tile_rows);
#endif

        // (2) Owner combine of its tile-row.
        if (is_owner) {
            cb_wait_front(cb_gather, num_gather);
#ifdef ROW_PARTIAL
            // Gather tile g holds member partials as rows; a column reduce adds them into row 0.
            reconfig_data_format(cb_gather, cb_scaler);
            pack_reconfig_data_format(cb_tmp);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gather, cb_scaler, cb_tmp);
            cb_reserve_back(cb_tmp, 1);
            tile_regs_acquire();
            for (uint32_t g = 0; g < num_gather; ++g) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_gather, cb_scaler, g, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);
            reduce_uninit();
            cb_pop_front(cb_gather, num_gather);
            transpose_tiles(cb_tmp, cb_owned, 1);  // back to column 0
#else
            // K partial tiles added in float32 DST, paired (seeded with a copy when K is odd). With
            // FACES_PARTIAL faces 1/3 hold stale bytes; faces add independently and nothing reads
            // column > 0 of the result, so they are don't-care.
            reconfig_data_format(cb_gather, cb_gather);
            pack_reconfig_data_format(cb_owned);
            cb_reserve_back(cb_owned, 1);
            tile_regs_acquire();
            uint32_t first = 0;
            if (num_gather & 1) {
                copy_init(cb_gather);
                copy_tile(cb_gather, 0, 0);
                first = 1;
            }
            if (num_gather > 1) {
                add_init(cb_gather, cb_gather, true);
                for (uint32_t k = first; k < num_gather; k += 2) {
                    add_tiles(cb_gather, cb_gather, k, k + 1, 0);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_owned);
            tile_regs_release();
            cb_push_back(cb_owned, 1);
            cb_pop_front(cb_gather, num_gather);
#endif
        }
    }
}
"""


# Gather (RISCV_1, NoC0): scaler tile, each tile-row's partial to its owner, then wait for this
# iteration's result (so the next iteration never overwrites a gather slot an owner still reads).
# RT args: [tile_rows, width_tiles, member_index, kernel_iters, (owner_x, owner_y) * tile_rows]
_RV_GATHER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_partial = 2, cb_gather = 4;
constexpr uint32_t tile_bytes = 4096;     // float32 32x32 tile
constexpr uint32_t face_row_bytes = 64;   // 16 float32 datums
constexpr uint32_t face_bytes = 1024;     // 16x16 float32 face
constexpr uint32_t sem_gather_id = 0, sem_result_id = 1;

void kernel_main() {
    const uint32_t tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t width_tiles = get_arg_val<uint32_t>(1);
    const uint32_t my_index = get_arg_val<uint32_t>(2);
    const uint32_t kernel_iters = get_arg_val<uint32_t>(3);
    constexpr uint32_t owners_base = 4;

    // The gather CB is one round deep everywhere, so its base is the same address on every owner.
    const uint32_t gather_base = get_write_ptr(cb_gather);
    const uint32_t gather_sem = get_semaphore(sem_gather_id);
    volatile tt_l1_ptr uint32_t* result_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_result_id));

    cb_reserve_back(cb_in, tile_rows * width_tiles);
    cb_push_back(cb_in, tile_rows * width_tiles);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

#ifdef ROW_PARTIAL
    // Member m owns row m % 32 of gather tile m / 32. Row j < 16 lives in faces 0/1, j >= 16 in faces 2/3.
    const uint32_t j = my_index % 32;
    const uint32_t slot =
        (my_index / 32) * tile_bytes + (j < 16 ? j * face_row_bytes : 2 * face_bytes + (j - 16) * face_row_bytes);
#else
    const uint32_t slot = my_index * tile_bytes;  // whole tile or faces 0/2 of it: slot m, tile layout kept
#endif

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        cb_wait_front(cb_partial, tile_rows);
        const uint32_t src = get_read_ptr(cb_partial);
        for (uint32_t r = 0; r < tile_rows; ++r) {
            const uint32_t ox = get_arg_val<uint32_t>(owners_base + 2 * r);
            const uint32_t oy = get_arg_val<uint32_t>(owners_base + 2 * r + 1);
            const uint32_t from = src + r * tile_bytes;
#ifdef ROW_PARTIAL
            // Row 0 of the transposed partial = face 0 row 0 + face 1 row 0: 2 x 64 B.
            noc_async_write(from, get_noc_addr(ox, oy, gather_base + slot), face_row_bytes);
            noc_async_write(from + face_bytes, get_noc_addr(ox, oy, gather_base + slot + face_bytes), face_row_bytes);
#elif defined(FACES_PARTIAL)
            // Column 0 lives in faces 0 and 2 (byte offsets 0 and 2048): 2 x 1024 B instead of 4096 B.
            noc_async_write(from, get_noc_addr(ox, oy, gather_base + slot), face_bytes);
            noc_async_write(
                from + 2 * face_bytes, get_noc_addr(ox, oy, gather_base + slot + 2 * face_bytes), face_bytes);
#else
            noc_async_write(from, get_noc_addr(ox, oy, gather_base + slot), tile_bytes);
#endif
        }
        noc_async_write_barrier();
        for (uint32_t r = 0; r < tile_rows; ++r) {
            const uint32_t ox = get_arg_val<uint32_t>(owners_base + 2 * r);
            const uint32_t oy = get_arg_val<uint32_t>(owners_base + 2 * r + 1);
            noc_semaphore_inc(get_noc_addr(ox, oy, gather_sem), 1);
        }
        cb_pop_front(cb_partial, tile_rows);

        // This member holds the all-reduced row vector once the result semaphore says so.
#ifdef MCAST_RETURN
        noc_semaphore_wait_min(result_sem, iter + 1);              // iteration stamp from the root
#else
        noc_semaphore_wait_min(result_sem, (iter + 1) * tile_rows);  // +1 per owner
#endif
    }
    noc_async_atomic_barrier();
}
"""


# Return (RISCV_0, NoC1): owner hand-off to compute, then the result to every member's output shard.
# RT args: [tile_rows, num_members, owned_row (or NOT_OWNER), num_gather, kernel_iters, output_addr,
#           MCAST:   is_root, root_x, root_y, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_dests
#           UNICAST: (member_x, member_y) * num_members]
_RV_RETURN_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_gather = 4, cb_owned = 5;
constexpr uint32_t tile_bytes = 4096;
constexpr uint32_t face_row_bytes = 64;
constexpr uint32_t face_bytes = 1024;
constexpr uint32_t sem_gather_id = 0, sem_result_id = 1, sem_collect_id = 2, sem_local_id = 3;
constexpr uint32_t NOT_OWNER = 0xFFFFFFFF;

#ifdef FACES_RETURN
// Only column 0 goes back: faces 0 and 2 of the result tile, 2 x 1024 B at byte offsets 0 and 2048.
constexpr uint32_t piece_bytes = face_bytes, num_pieces = 2, piece_stride = 2 * face_bytes;
#else
constexpr uint32_t piece_bytes = tile_bytes, num_pieces = 1, piece_stride = 0;
#endif

void kernel_main() {
    const uint32_t tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_members = get_arg_val<uint32_t>(1);
    const uint32_t owned_row = get_arg_val<uint32_t>(2);
    const uint32_t num_gather = get_arg_val<uint32_t>(3);
    const uint32_t kernel_iters = get_arg_val<uint32_t>(4);
    const uint32_t output_base = get_arg_val<uint32_t>(5);
    const bool is_owner = owned_row != NOT_OWNER;

    const uint32_t gather_base = get_write_ptr(cb_gather);
    volatile tt_l1_ptr uint32_t* gather_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather_id));
    const uint32_t result_sem_addr = get_semaphore(sem_result_id);

#ifdef MCAST_RETURN
    const bool is_root = get_arg_val<uint32_t>(6) != 0;
    const uint32_t root_x = get_arg_val<uint32_t>(7);
    const uint32_t root_y = get_arg_val<uint32_t>(8);
    const uint32_t mc_sx = get_arg_val<uint32_t>(9);
    const uint32_t mc_sy = get_arg_val<uint32_t>(10);
    const uint32_t mc_ex = get_arg_val<uint32_t>(11);
    const uint32_t mc_ey = get_arg_val<uint32_t>(12);
    const uint32_t num_dests = get_arg_val<uint32_t>(13);
    const uint32_t collect_sem_addr = get_semaphore(sem_collect_id);
    volatile tt_l1_ptr uint32_t* collect_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(collect_sem_addr);
    const uint32_t local_addr = get_semaphore(sem_local_id);
    volatile tt_l1_ptr uint32_t* local_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_addr);
    volatile tt_l1_ptr uint32_t* result_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_sem_addr);
#else
    constexpr uint32_t members_base = 6;
#endif

#ifdef ROW_PARTIAL
    // Rows no member writes (K % 32 .. 31 of the last gather tile) must read as zero. Members never
    // touch them, so clearing them cannot race with an incoming partial. In face order they are up to
    // three contiguous byte ranges; clear them with NoC reads of the hardware zeros region (RISC-V
    // stores would cost ~3 ns per word, more than the whole combine at small K).
    if (is_owner && (num_members % 32) != 0) {
        const uint32_t last = gather_base + (num_gather - 1) * tile_bytes;
        const uint32_t j0 = num_members % 32;
        uint32_t starts[3], ends[3], n = 0;
        if (j0 < 16) {
            starts[n] = last + j0 * face_row_bytes; ends[n++] = last + face_bytes;                    // face 0
            starts[n] = last + face_bytes + j0 * face_row_bytes; ends[n++] = last + 2 * face_bytes;   // face 1
            starts[n] = last + 2 * face_bytes; ends[n++] = last + tile_bytes;                         // faces 2, 3
        } else {
            starts[n] = last + 2 * face_bytes + (j0 - 16) * face_row_bytes; ends[n++] = last + 3 * face_bytes;
            starts[n] = last + 3 * face_bytes + (j0 - 16) * face_row_bytes; ends[n++] = last + tile_bytes;
        }
        const uint64_t zeros = get_noc_addr(my_x[noc_index], my_y[noc_index], MEM_ZEROS_BASE);
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t a = starts[i]; a < ends[i]; a += MEM_ZEROS_SIZE) {
                const uint32_t chunk = ends[i] - a < MEM_ZEROS_SIZE ? ends[i] - a : MEM_ZEROS_SIZE;
                noc_async_read(zeros, a, chunk);
            }
        }
        noc_async_read_barrier();
    }
#endif

    for (uint32_t iter = 0; iter < kernel_iters; ++iter) {
        if (is_owner) {
            cb_reserve_back(cb_gather, num_gather);
            noc_semaphore_wait_min(gather_sem, (iter + 1) * num_members);
            cb_push_back(cb_gather, num_gather);

            cb_wait_front(cb_owned, 1);
            const uint32_t src = get_read_ptr(cb_owned);
            const uint32_t dst = output_base + owned_row * tile_bytes;
#ifdef MCAST_RETURN
            for (uint32_t p = 0; p < num_pieces; ++p) {
                noc_async_write(src + p * piece_stride, get_noc_addr(root_x, root_y, dst + p * piece_stride), piece_bytes);
            }
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(root_x, root_y, collect_sem_addr), 1);
#else
            for (uint32_t m = 0; m < num_members; ++m) {
                const uint32_t mx = get_arg_val<uint32_t>(members_base + 2 * m);
                const uint32_t my = get_arg_val<uint32_t>(members_base + 2 * m + 1);
                for (uint32_t p = 0; p < num_pieces; ++p) {
                    noc_async_write(src + p * piece_stride, get_noc_addr(mx, my, dst + p * piece_stride), piece_bytes);
                }
            }
            noc_async_write_barrier();
            for (uint32_t m = 0; m < num_members; ++m) {
                const uint32_t mx = get_arg_val<uint32_t>(members_base + 2 * m);
                const uint32_t my = get_arg_val<uint32_t>(members_base + 2 * m + 1);
                noc_semaphore_inc(get_noc_addr(mx, my, result_sem_addr), 1);
            }
#endif
            cb_pop_front(cb_owned, 1);
        }

#ifdef MCAST_RETURN
        if (is_root) {
            noc_semaphore_wait_min(collect_sem, (iter + 1) * tile_rows);
            // Data, then the semaphore: same NoC and multicast VC, so they land in this order.
#ifdef FACES_RETURN
            for (uint32_t r = 0; r < tile_rows; ++r) {
                for (uint32_t p = 0; p < num_pieces; ++p) {
                    const uint32_t a = output_base + r * tile_bytes + p * piece_stride;
                    noc_async_write_multicast(a, get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, a), piece_bytes, num_dests);
                }
            }
#else
            noc_async_write_multicast(
                output_base,
                get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, output_base),
                tile_rows * tile_bytes,
                num_dests);
#endif
            *local_word = iter + 1;
            noc_semaphore_set_multicast(
                local_addr, get_noc_multicast_addr(mc_sx, mc_sy, mc_ex, mc_ey, result_sem_addr), num_dests);
            // The root is not a multicast destination: its own output is complete (the owners wrote it),
            // and it releases its own gather kernel once the multicast has read its source.
            noc_async_write_barrier();
            *result_sem = iter + 1;
        }
#endif
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
"""


def _normal_cb(cb_index, core_ranges, num_pages, page_bytes, dtype):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=cb_index,
                data_format=dtype,
                page_size=page_bytes,
            )
        ],
    )


def _fp32_cb(cb_index, core_ranges, num_tiles):
    """Circular buffer of `num_tiles` float32 tiles (one 4096 B page per tile)."""
    return _normal_cb(cb_index, core_ranges, num_tiles, FP32_TILE_BYTES, ttnn.float32)


def _inline_kernel(source, core_ranges, runtime_args, config, defines):
    """Kernel descriptor for inline kernel source; the row-vector kernels take no compile-time args."""
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_ranges,
        compile_time_args=[],
        defines=list(defines),
        runtime_args=runtime_args,
        config=config,
    )


def create_row_vector_program_descriptor(
    input_tensor, output_tensor, *, variant, num_members, tile_rows, width_tiles=1, kernel_iters=1
):
    partial, returned, path = parse_row_vector_variant(variant)
    if tile_rows < 1 or width_tiles < 1 or kernel_iters < 1:
        raise ValueError("tile_rows, width_tiles and kernel_iters must be positive")
    if tile_rows > num_members:
        raise ValueError(f"one owner per tile-row needs tile_rows <= members, got {tile_rows} > {num_members}")
    device = input_tensor.device()
    layout = build_member_layout(device, num_members)
    expected_in = [tile_rows * ttnn.TILE_SIZE, num_members * width_tiles * ttnn.TILE_SIZE]
    expected_out = [tile_rows * ttnn.TILE_SIZE, num_members * ttnn.TILE_SIZE]
    if list(input_tensor.shape) != expected_in or list(output_tensor.shape) != expected_out:
        raise ValueError(f"input shape must be {expected_in} and output shape {expected_out}")
    if input_tensor.dtype != ttnn.bfloat16 or output_tensor.dtype != ttnn.float32:
        raise ValueError("input must be bfloat16 and output float32")

    mcast = path == "mcast"
    defines = [("ROW_PARTIAL", "1")] if partial == "row" else []
    defines += [("FACES_PARTIAL", "1")] if partial == "faces" else []
    defines += [("FACES_RETURN", "1")] if returned == "faces" else []
    defines += [("MCAST_RETURN", "1")] if mcast else []
    n_gather = num_gather_tiles(variant, num_members)

    def virt(xy):
        core = device.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
        return core.x, core.y

    member_virtual = [virt(xy) for xy in layout.members]
    owner_of_row = row_owner_indices(layout, tile_rows)
    owned = {member: r for r, member in enumerate(owner_of_row)}
    root_x, root_y = member_virtual[owner_of_row[0]]  # member 0, the owner of tile-row 0
    # Multicast rectangle in NoC1 order: the start corner is the far (max x, max y) one.
    lo = virt((0, 0))
    hi = virt((layout.cols - 1, layout.rows - 1))
    mcast_rect = [hi[0], hi[1], lo[0], lo[1]]
    num_dests = layout.rows * layout.cols - 1
    output_addr = output_tensor.buffer_address()

    compute_rt = ttnn.RuntimeArgs()
    gather_rt = ttnn.RuntimeArgs()
    return_rt = ttnn.RuntimeArgs()
    flat_members = [v for xy in member_virtual for v in xy]
    flat_owners = [v for member in owner_of_row for v in member_virtual[member]]
    for index, (x, y) in enumerate(layout.members):
        owned_row = owned.get(index, RV_NOT_OWNER)
        is_owner = owned_row != RV_NOT_OWNER
        compute_rt[x][y] = [tile_rows, width_tiles, int(is_owner), n_gather, kernel_iters]
        gather_rt[x][y] = [tile_rows, width_tiles, index, kernel_iters] + flat_owners
        common = [tile_rows, num_members, owned_row, n_gather, kernel_iters, output_addr]
        if mcast:
            return_rt[x][y] = common + [int(index == 0), root_x, root_y] + mcast_rect + [num_dests]
        else:
            return_rt[x][y] = common + flat_members

    members = layout.member_ranges
    # The multicast also lands on the bounding rectangle's non-member Tensix cores, so the result
    # semaphore exists there too.
    result_ranges = layout.bbox_ranges if mcast else members
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(RV_CB_IN, input_tensor),
        _normal_cb(RV_CB_SCALER, members, 1, 2048, ttnn.bfloat16),
        _fp32_cb(RV_CB_PARTIAL, members, tile_rows),
        _fp32_cb(RV_CB_GATHER, members, n_gather),
        _fp32_cb(RV_CB_OWNED, members, 1),
    ]
    if partial == "row":
        cbs.append(_fp32_cb(RV_CB_TMP, members, tile_rows))
    semaphores = [
        ttnn.SemaphoreDescriptor(id=RV_SEM_GATHER, core_ranges=members, initial_value=0),
        ttnn.SemaphoreDescriptor(id=RV_SEM_RESULT, core_ranges=result_ranges, initial_value=0),
    ]
    if mcast:
        semaphores += [
            ttnn.SemaphoreDescriptor(id=RV_SEM_COLLECT, core_ranges=members, initial_value=0),
            ttnn.SemaphoreDescriptor(id=RV_SEM_LOCAL, core_ranges=members, initial_value=0),
        ]

    kernels = [
        _inline_kernel(
            _RV_COMPUTE_KERNEL,
            members,
            compute_rt,
            ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            defines,
        ),
        _inline_kernel(
            _RV_GATHER_KERNEL,
            members,
            gather_rt,
            ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            defines,
        ),
        _inline_kernel(
            _RV_RETURN_KERNEL,
            members,
            return_rt,
            ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            defines,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def row_sum_all_reduce(
    input_tensor, *, variant, num_members, tile_rows, width_tiles=1, kernel_iters=1, output_tensor=None
):
    """Every member ends with the sum over the WHOLE row as column 0 of a float32 tile per tile-row
    (output WIDTH_SHARDED over the same members, one tile column each; columns 1..31 don't-care)."""
    device = input_tensor.device()
    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([tile_rows * ttnn.TILE_SIZE, num_members * ttnn.TILE_SIZE]),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            device,
            create_row_vector_memory_config(device, num_members, tile_rows, 1),
        )
    descriptor = create_row_vector_program_descriptor(
        input_tensor,
        output_tensor,
        variant=variant,
        num_members=num_members,
        tile_rows=tile_rows,
        width_tiles=width_tiles,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
