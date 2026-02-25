#pragma once

struct InterRange {
    uint32_t n_tiles;
    uint32_t n_tiles_proc;
};

inline std::array<InterRange, 3> get_inter_range(
    uint32_t arg_num_tiles, uint32_t num_tiles_per_batch, uint32_t num_batches) {
    auto num_tail_tiles = arg_num_tiles % (num_batches * num_tiles_per_batch);
    auto _num_tiles = arg_num_tiles - num_tail_tiles;

    auto num_tiles2 = arg_num_tiles - _num_tiles;
    num_tail_tiles = num_tiles2 % num_tiles_per_batch;
    num_tiles2 = num_tiles2 - num_tail_tiles;

    auto num_tiles3 = arg_num_tiles - num_tiles2 - _num_tiles;

    return std::array<InterRange, 3>{
        {{_num_tiles, num_batches * num_tiles_per_batch}, {num_tiles2, num_tiles_per_batch}, {num_tiles3, num_tiles3}}};
}
