// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Generate index tiles for TopK multicore local processing phase.
 *
 * This function creates index tiles that track the original positions of elements
 * within each width tile. The generated indices enable proper tracking of element
 * origins after sorting operations.
 *
 * Index Generation Pattern:
 * - Each tile contains indices for 32x32 = 1024 elements arranged in tile format
 * - For tile at position 'wt' along the width: elements have indices [wt*32, wt*32+31]
 * - Layout: First 32 elements get indices {wt*32, wt*32+1, ..., wt*32+31}
 *           Second 32 elements get indices {wt*32+32, wt*32+33, ..., wt*32+63}, etc.
 *
 * @param cb_id Circular buffer index to write the generated index tile
 * @param wt    Width tile position [0, Wt_local) identifying which tile position along width
 */
template <typename T = uint16_t>
FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt) {
    // Constants
    constexpr uint32_t one_tile = 1;

    // Reserve space
    cb_reserve_back(cb_id, one_tile);

    // Writer config
    const uint32_t writer_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr T* ptr = reinterpret_cast<volatile tt_l1_ptr T*>(writer_addr);
    const uint32_t w = wt << 5;  // wt * 2^(5)

    // Writer loop
    uint32_t count = 0;
    /*
    The 32x32 tile is subdivided into four 16x16 quadrants(faces): top-left, top-right, bottom-left, and bottom-right.
    These quadrants are stored contiguously in memory. Therefore, indices must be written in memory according
    to their respective quadrant, rather than sequentially from left to right across the entire tile.
    */
    constexpr uint32_t tile_faces = 2;
    constexpr uint32_t face_size = 16;
    for (uint32_t i = 0; i < tile_faces; ++i) {
        for (uint32_t j = 0; j < tile_faces; ++j) {
            for (uint32_t k = 0; k < face_size; ++k) {
                for (uint32_t l = 0; l < face_size; l++) {
                    const T value = l + face_size * j + w;
                    ptr[count] = value;
                    count++;
                }  // l loop
            }  // k loop
        }  // j loop
    }  // i loop

    // Push the tile
    cb_push_back(cb_id, one_tile);
}
