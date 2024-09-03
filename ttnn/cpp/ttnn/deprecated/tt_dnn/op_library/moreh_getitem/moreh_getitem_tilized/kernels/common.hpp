// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define TILE_HEIGHT 32
#define TILE_WIDTH 32
#define FACE_WIDTH 16

#define INDEX_TILE_SIZE (4096)

uint32_t get_noc_offset_in_tile(uint32_t stick_h, uint32_t stick_w, uint32_t tile_h, uint32_t element_size) {
    uint32_t noc_offset = 0;
    stick_h = stick_h - tile_h * TILE_HEIGHT;

    const bool is_even = (stick_w % 2 == 0);
    const bool is_odd = !is_even;

    const uint32_t stick_bytes = FACE_WIDTH * element_size;

    if (stick_h < FACE_WIDTH && is_even) noc_offset += stick_h * stick_bytes;
    else if (stick_h < FACE_WIDTH && is_odd) noc_offset += (16 + stick_h) * stick_bytes;
    else if (stick_h >= FACE_WIDTH && is_even) noc_offset += (16 + stick_h) * stick_bytes;
    else if (stick_h >= FACE_WIDTH && is_odd) noc_offset += (32 + stick_h) * stick_bytes;

    return noc_offset;
}

struct Idx4d
{
    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;
};

Idx4d get_stick_indices(uint32_t stick_idx, uint32_t size_c, uint32_t size_h, uint32_t num_stick_width) {
    Idx4d stick_index_4d;

    stick_index_4d.w = stick_idx % num_stick_width;
    uint32_t stick_nch = stick_idx / num_stick_width;

    stick_index_4d.h = stick_nch % size_h;
    uint32_t stick_nc = stick_nch / size_h;

    stick_index_4d.c = stick_nc % size_c;
    stick_index_4d.n = stick_nc / size_c;

    return stick_index_4d;
}

Idx4d get_tile_indices(Idx4d stick_index_4d) {
    Idx4d tile_index_4d;

    tile_index_4d.n = stick_index_4d.n;
    tile_index_4d.c = stick_index_4d.c;
    tile_index_4d.h = stick_index_4d.h / TILE_HEIGHT;
    tile_index_4d.w = stick_index_4d.w / 2;

    return tile_index_4d;
}

struct Idx5d
{
    uint32_t n;
    uint32_t c;
    uint32_t d;
    uint32_t h;
    uint32_t w;
};

Idx5d get_stick_indices(uint32_t stick_idx, uint32_t size_c, uint32_t size_d, uint32_t size_h, uint32_t num_stick_width) {
    Idx5d stick_index_5d;

    stick_index_5d.w = stick_idx % num_stick_width;
    uint32_t stick_ncdh = stick_idx / num_stick_width;

    stick_index_5d.h = stick_ncdh % size_h;
    uint32_t stick_ncd = stick_ncdh / size_h;

    stick_index_5d.d = stick_ncd % size_d;
    uint32_t stick_nc = stick_ncd / size_d;

    stick_index_5d.c = stick_nc % size_c;
    stick_index_5d.n = stick_nc / size_c;

    return stick_index_5d;
}

Idx5d get_tile_indices(Idx5d stick_index_5d) {
    Idx5d tile_index_5d;

    tile_index_5d.n = stick_index_5d.n;
    tile_index_5d.c = stick_index_5d.c;
    tile_index_5d.d = stick_index_5d.d;
    tile_index_5d.h = stick_index_5d.h / TILE_HEIGHT;
    tile_index_5d.w = stick_index_5d.w / 2;

    return tile_index_5d;
}
