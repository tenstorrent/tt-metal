// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Contains utility functions for partitioning shards work between multiple cores.
//

#include "sharding_utilities.hpp"

namespace tt::tt_metal {

// Calculate the sharding specs for input sticks with padding (no halo)
NewShardingConfig get_shard_specs(int32_t start_stick, int32_t end_stick, const PoolConfig& pc, bool to_print) {
    int32_t nsticks_per_core = end_stick - start_stick;

    if (nsticks_per_core < 1) {
        // range is 0
        return NewShardingConfig{.first_partial_right_aligned_row_width = 0,
                                 .first_partial_image_num_rows = 0,
                                 .num_full_images = 0,
                                 .last_partial_image_num_rows = 0,
                                 .last_partial_left_aligned_row_width = 0,
                                 .skip_after_partial_right_aligned_row = 0,
                                 .skip_after_first_partial_image_row = 0,
                                 .skip_after_full_image = 0,
                                 .initial_skip = 0};
    }
    if (nsticks_per_core < pc.in_w) {
        int32_t in_w_i = start_stick % pc.in_w;
        int32_t in_h_i = (start_stick % (pc.in_w * pc.in_h)) / pc.in_w;
        int32_t last_in_w_i = (end_stick - 1) % pc.in_w;
        int32_t last_in_h_i = ((end_stick - 1) % (pc.in_w * pc.in_h)) / pc.in_w;
        int32_t initial_skip = 0;
        if ((start_stick % (pc.in_h * pc.in_w) == 0) || (start_stick % pc.stride_h != 0)) {
            int32_t halo_nsticks = (pc.in_w + 2 * pc.pad_w) * pc.pad_h + pc.window_w / 2;
            initial_skip = halo_nsticks;
        }
        switch (nsticks_per_core) {
            case 1:
                // only one stick in the range. need to figure out if there are padding to be attached to it, and in
                // which place
                if (in_w_i == pc.in_w - 1) {
                    // this is the last stick in the row
                    // there needs to be padding sticks attached
                    if (in_h_i % pc.in_h == pc.in_h - 1) {
                        // this is the last stick in the image
                        // there needs to be full padding rows
                        return NewShardingConfig{
                            .first_partial_right_aligned_row_width = 1,
                            .first_partial_image_num_rows = 0,
                            .num_full_images = 0,
                            .last_partial_image_num_rows = 0,
                            .last_partial_left_aligned_row_width = 0,
                            .skip_after_partial_right_aligned_row = (int32_t)(pc.pad_h * (pc.in_w + 2 * pc.pad_w)),
                            .skip_after_first_partial_image_row = 0,
                            .skip_after_full_image = 0,
                            .initial_skip = initial_skip};
                    } else {
                        // this is just the last stick in the row and not the image
                        // there are just width padding
                        return NewShardingConfig{.first_partial_right_aligned_row_width = 1,
                                                 .first_partial_image_num_rows = 0,
                                                 .num_full_images = 0,
                                                 .last_partial_image_num_rows = 0,
                                                 .last_partial_left_aligned_row_width = 0,
                                                 .skip_after_partial_right_aligned_row = 2 * (int32_t)pc.pad_w,
                                                 .skip_after_first_partial_image_row = 0,
                                                 .skip_after_full_image = 0,
                                                 .initial_skip = initial_skip};
                    }
                } else {
                    // this is just one stick without any padding
                    return NewShardingConfig{.first_partial_right_aligned_row_width = 1,
                                             .first_partial_image_num_rows = 0,
                                             .num_full_images = 0,
                                             .last_partial_image_num_rows = 0,
                                             .last_partial_left_aligned_row_width = 0,
                                             .skip_after_partial_right_aligned_row = 0,
                                             .skip_after_first_partial_image_row = 0,
                                             .skip_after_full_image = 0,
                                             .initial_skip = initial_skip};
                }
            case 2:
                // two sticks in the range. figure out if there are any padding attached
                if (in_w_i == pc.in_w - 2) {
                    // these are two last sticks of the row
                    // there needs to be padding after
                    if (in_h_i == pc.in_h - 1) {
                        // these are the last sticks in the last row of the image
                        // insert full padding rows
                        return NewShardingConfig{
                            .first_partial_right_aligned_row_width = 2,
                            .first_partial_image_num_rows = 0,
                            .num_full_images = 0,
                            .last_partial_image_num_rows = 0,
                            .last_partial_left_aligned_row_width = 0,
                            .skip_after_partial_right_aligned_row = (int32_t)(pc.pad_h * (pc.in_w + 2 * pc.pad_w)),
                            .skip_after_first_partial_image_row = 0,
                            .skip_after_full_image = 0,
                            .initial_skip = initial_skip};
                    } else {
                        // just need width padding
                        return NewShardingConfig{.first_partial_right_aligned_row_width = 2,
                                                 .first_partial_image_num_rows = 0,
                                                 .num_full_images = 0,
                                                 .last_partial_image_num_rows = 0,
                                                 .last_partial_left_aligned_row_width = 0,
                                                 .skip_after_partial_right_aligned_row = 2 * (int32_t)pc.pad_w,
                                                 .skip_after_first_partial_image_row = 0,
                                                 .skip_after_full_image = 0,
                                                 .initial_skip = initial_skip};
                    }
                } else if (in_w_i == pc.in_w - 1) {
                    // these are one last stick of the row and one first stick of next row
                    // there needs to be padding in between
                    if (in_h_i == pc.in_h - 1) {
                        // these two sticks belong to different images
                        // insert full padding rows between them
                        return NewShardingConfig{
                            .first_partial_right_aligned_row_width = 1,
                            .first_partial_image_num_rows = 0,
                            .num_full_images = 0,
                            .last_partial_image_num_rows = 0,
                            .last_partial_left_aligned_row_width = 1,
                            .skip_after_partial_right_aligned_row = (int32_t)(pc.pad_h * (pc.in_w + 2 * pc.pad_w)),
                            .skip_after_first_partial_image_row = 0,
                            .skip_after_full_image = 0,
                            .initial_skip = initial_skip};
                    } else {
                        // just width padding between then
                        return NewShardingConfig{.first_partial_right_aligned_row_width = 1,
                                                 .first_partial_image_num_rows = 0,
                                                 .num_full_images = 0,
                                                 .last_partial_image_num_rows = 0,
                                                 .last_partial_left_aligned_row_width = 1,
                                                 .skip_after_partial_right_aligned_row = (int32_t)(2 * pc.pad_w),
                                                 .skip_after_first_partial_image_row = 0,
                                                 .skip_after_full_image = 0,
                                                 .initial_skip = initial_skip};
                    }
                } else {
                    // no padding needs to be attached
                    return NewShardingConfig{.first_partial_right_aligned_row_width = 2,
                                             .first_partial_image_num_rows = 0,
                                             .num_full_images = 0,
                                             .last_partial_image_num_rows = 0,
                                             .last_partial_left_aligned_row_width = 0,
                                             .skip_after_partial_right_aligned_row = 0,
                                             .skip_after_first_partial_image_row = 0,
                                             .skip_after_full_image = 0,
                                             .initial_skip = initial_skip};
                }

            default:
                if (in_h_i == last_in_h_i) {
                    // all sticks belong to same row
                    if (last_in_w_i == pc.in_w - 1) {
                        // these sticks are the last in the row
                        // insert padding at end
                        if (in_h_i == pc.in_h - 1) {
                            // this is the last row
                            // need full padding rows
                            return NewShardingConfig{
                                .first_partial_right_aligned_row_width = nsticks_per_core,
                                .first_partial_image_num_rows = 0,
                                .num_full_images = 0,
                                .last_partial_image_num_rows = 0,
                                .last_partial_left_aligned_row_width = 0,
                                .skip_after_partial_right_aligned_row = (int32_t)(pc.pad_h * (pc.in_w + 2 * pc.pad_w)),
                                .skip_after_first_partial_image_row = 0,
                                .skip_after_full_image = 0,
                                .initial_skip = initial_skip};
                        } else {
                            // just width padding needed
                            return NewShardingConfig{.first_partial_right_aligned_row_width = nsticks_per_core,
                                                     .first_partial_image_num_rows = 0,
                                                     .num_full_images = 0,
                                                     .last_partial_image_num_rows = 0,
                                                     .last_partial_left_aligned_row_width = 0,
                                                     .skip_after_partial_right_aligned_row = (int32_t)(2 * pc.pad_w),
                                                     .skip_after_first_partial_image_row = 0,
                                                     .skip_after_full_image = 0,
                                                     .initial_skip = initial_skip};
                        }
                    } else {
                        // no padding is needed
                        return NewShardingConfig{.first_partial_right_aligned_row_width = 0,
                                                 .first_partial_image_num_rows = 0,
                                                 .num_full_images = 0,
                                                 .last_partial_image_num_rows = 0,
                                                 .last_partial_left_aligned_row_width = nsticks_per_core,
                                                 .skip_after_partial_right_aligned_row = 0,
                                                 .skip_after_first_partial_image_row = 0,
                                                 .skip_after_full_image = 0,
                                                 .initial_skip = initial_skip};
                    }
                } else {
                    // sticks span two different rows. figure out where does the padding go.
                    // padding will go at the end of the start row
                    // find the last stick in the start row
                    int32_t insert_padding_at = pc.in_w - in_w_i;
                    if (in_h_i == pc.in_h - 1) {
                        // sticks span across different images
                        // full padding rows need to be inserted
                        return NewShardingConfig{
                            .first_partial_right_aligned_row_width = insert_padding_at,
                            .first_partial_image_num_rows = 0,
                            .num_full_images = 0,
                            .last_partial_image_num_rows = 0,
                            .last_partial_left_aligned_row_width = nsticks_per_core - insert_padding_at,
                            .skip_after_partial_right_aligned_row = (int32_t)(pc.pad_h * (pc.in_w + 2 * pc.pad_w)),
                            .skip_after_first_partial_image_row = 0,
                            .skip_after_full_image = 0,
                            .initial_skip = initial_skip};
                    } else {
                        // sticks belong to same image
                        // only width padding needed
                        return NewShardingConfig{
                            .first_partial_right_aligned_row_width = insert_padding_at,
                            .first_partial_image_num_rows = 0,
                            .num_full_images = 0,
                            .last_partial_image_num_rows = 0,
                            .last_partial_left_aligned_row_width = nsticks_per_core - insert_padding_at,
                            .skip_after_partial_right_aligned_row = (int32_t)(2 * pc.pad_w),
                            .skip_after_first_partial_image_row = 0,
                            .skip_after_full_image = 0,
                            .initial_skip = initial_skip};
                    }
                }
        }
    }

    // First partial right-aligned row
    int32_t image_row_start_left_width = start_stick % pc.in_w;
    if (to_print)
        log_debug("image_row_start_left_width: {}", image_row_start_left_width);
    int32_t first_partial_right_aligned_row_width =
        image_row_start_left_width > 0 ? pc.in_w - image_row_start_left_width : 0;
    if (to_print)
        log_debug("first_partial_right_aligned_row_width: {}", first_partial_right_aligned_row_width);

    if (first_partial_right_aligned_row_width > nsticks_per_core) {
        return NewShardingConfig{.first_partial_right_aligned_row_width = nsticks_per_core,
                                 .first_partial_image_num_rows = 0,
                                 .num_full_images = 0,
                                 .last_partial_image_num_rows = 0,
                                 .last_partial_left_aligned_row_width = 0,
                                 .skip_after_partial_right_aligned_row = 0,
                                 .skip_after_first_partial_image_row = 0,
                                 .skip_after_full_image = 0};
    }

    // Last partial left-aligned row
    int32_t sticks_after_first_partial_row = nsticks_per_core - first_partial_right_aligned_row_width;
    if (to_print)
        log_debug("sticks_after_first_partial_row: {}", sticks_after_first_partial_row);
    int32_t last_partial_left_aligned_row_width = sticks_after_first_partial_row % pc.in_w;
    if (to_print)
        log_debug("last_partial_left_aligned_row_width: {}", last_partial_left_aligned_row_width);

    // Figure out how to allocate full image rows to first partial image, full images, or last partial image
    // This also affects skip after first_partial_right_aligned_row
    int32_t image_row_start_idx = start_stick / pc.in_w;
    int32_t image_row_start_idx_after_partial_right_aligned_row =
        (start_stick + first_partial_right_aligned_row_width) / pc.in_w;
    int32_t image_row_end_idx = (end_stick - 1) / pc.in_w;
    int32_t image_start_idx = image_row_start_idx / pc.in_h;
    int32_t image_start_idx_after_partial_right_aligned_row =
        image_row_start_idx_after_partial_right_aligned_row / pc.in_h;
    int32_t image_start_height_after_partial_right_aligned_row =
        image_row_start_idx_after_partial_right_aligned_row % pc.in_h;
    int32_t image_end_idx = image_row_end_idx / pc.in_h;

    // Default case: We don't have a partial right aligned row; so we start with either full images, last partial image,
    // or partial left aligned row
    int32_t skip_after_partial_right_aligned_row = 0;
    // Case: partial_right_aligned_row > 0 and completes an image
    if (first_partial_right_aligned_row_width > 0 && image_start_height_after_partial_right_aligned_row == 0) {
        skip_after_partial_right_aligned_row = pc.window_w - 1 + pc.pad_h * (pc.in_w + 2 * pc.pad_w);
        // Case: partial_right_aligned_row > 0 and doesn't complete an image
    } else if (first_partial_right_aligned_row_width > 0) {
        skip_after_partial_right_aligned_row = pc.window_w - 1;
    }

    int32_t first_partial_image_num_rows = 0;
    int32_t skip_after_first_partial_image_row = 0;
    // Only case where we have first_partial_image_rows: We have at at least 1 completed image and the starting image
    // row is in the middle of an image
    if (image_end_idx - image_start_idx_after_partial_right_aligned_row > 0 &&
        image_start_height_after_partial_right_aligned_row > 0) {
        first_partial_image_num_rows = pc.in_h - image_start_height_after_partial_right_aligned_row;
        skip_after_first_partial_image_row = pc.pad_h * (pc.in_w + 2 * pc.pad_w);
    }

    // Full images
    int32_t image_rows_after_first_partial_image =
        sticks_after_first_partial_row / pc.in_w - first_partial_image_num_rows;
    if (to_print)
        log_debug("image_rows_after_first_partial_image: {}", image_rows_after_first_partial_image);
    int32_t num_full_images = image_rows_after_first_partial_image / pc.in_h;
    int32_t skip_after_full_image = num_full_images > 0 ? pc.pad_h * (pc.in_w + 2 * pc.pad_w) : 0;

    // Last partial image rows
    int32_t last_partial_image_num_rows = image_rows_after_first_partial_image % pc.in_h;

    return NewShardingConfig{.first_partial_right_aligned_row_width = first_partial_right_aligned_row_width,
                             .first_partial_image_num_rows = first_partial_image_num_rows,
                             .num_full_images = num_full_images,
                             .last_partial_image_num_rows = last_partial_image_num_rows,
                             .last_partial_left_aligned_row_width = last_partial_left_aligned_row_width,
                             .skip_after_partial_right_aligned_row = skip_after_partial_right_aligned_row,
                             .skip_after_first_partial_image_row = skip_after_first_partial_image_row,
                             .skip_after_full_image = skip_after_full_image};
}

// Calculate the sharding config for input sticks with padding and halo data included
NewShardingConfig get_shard_specs_with_halo(int32_t start_stick,
                                            int32_t end_stick,
                                            const PoolConfig& pc,
                                            bool to_print) {
    int32_t nsticks = end_stick - start_stick;

    if (nsticks < 1) {
        // range is 0. set everything to 0
        return NewShardingConfig{.first_partial_right_aligned_row_width = 0,
                                 .first_partial_image_num_rows = 0,
                                 .num_full_images = 0,
                                 .last_partial_image_num_rows = 0,
                                 .last_partial_left_aligned_row_width = 0,
                                 .skip_after_partial_right_aligned_row = 0,
                                 .skip_after_first_partial_image_row = 0,
                                 .skip_after_full_image = 0,
                                 .initial_skip = 0,
                                 .start_stick = 0};
    }

    // NOTE: all cores have the exact same sized halo, including the left cores' left halo, which is initial_skip
    // (padding), and right cores' right halo
    int32_t halo_nsticks = (pc.in_w + 2 * pc.pad_w) * pc.pad_h + pc.window_w / 2;

    int32_t halo_nsticks_nopad = pc.in_w + pc.window_w / 2;
    int32_t halo_start_stick =
        start_stick - halo_nsticks_nopad;  // this will be -ve for initial cores where start_stick < halo_nsticks
    int32_t halo_end_stick = end_stick + halo_nsticks_nopad;
    int32_t nsticks_with_halo = halo_end_stick - halo_start_stick;

    // calculate in_start_id of my current batch
    int32_t batch_start = (start_stick / (pc.in_h * pc.in_w)) * (pc.in_h * pc.in_w);

    // pad sticks before the data sticks begin, add edge pads
    int32_t initial_skip = halo_start_stick < batch_start ? ((batch_start - halo_start_stick) + 2 * pc.pad_w) : 0;

    // start with the first valid stick incl. halo
    // start_stick = halo_start_stick < 0 ? 0 : halo_start_stick;
    start_stick = halo_start_stick < batch_start ? batch_start : halo_start_stick;
    end_stick = halo_end_stick;  // TODO: take care of max, which would be (in_h * in_w * nbatch + halo_nsticks)

    // log_debug(" -- range with halo: [{},{})", start_stick, end_stick);

    int32_t curr_stick = start_stick;

    int32_t batch_i = curr_stick / (pc.in_h * pc.in_w);
    int32_t curr_batch_stick = curr_stick % (pc.in_h * pc.in_w);
    int32_t in_h_i = curr_batch_stick / pc.in_w;
    int32_t in_w_i = curr_batch_stick % pc.in_w;

    int32_t last_in_w_i = (end_stick - 1) % pc.in_w;
    int32_t last_in_h_i = ((end_stick - 1) % (pc.in_w * pc.in_h)) / pc.in_w;

    int32_t pad_size = in_h_i * 2 * pc.pad_w;
    int32_t start_stick_padded = start_stick + pad_size;

    if (in_h_i == last_in_h_i) {
        // all sticks belong to the same row (neither first partial, nor last partial)
        return NewShardingConfig{.first_partial_right_aligned_row_width = 0,
                                 .first_partial_image_num_rows = 0,
                                 .num_full_images = 0,
                                 .last_partial_image_num_rows = 0,
                                 .last_partial_left_aligned_row_width = nsticks,
                                 .skip_after_partial_right_aligned_row = 0,
                                 .skip_after_first_partial_image_row = 0,
                                 .skip_after_full_image = 0,
                                 .initial_skip = initial_skip,
                                 .start_stick = start_stick_padded};
    }

    int32_t partial_first_row_nsticks = 0;
    int32_t skip_after_partial_first_row = 0;
    if (curr_stick % pc.in_w > 0) {
        partial_first_row_nsticks = pc.in_w - (curr_stick % pc.in_w);
        skip_after_partial_first_row = 2 * pc.pad_w;
    } else {
        // no partial first row
        partial_first_row_nsticks = 0;
        skip_after_partial_first_row = 0;
    }
    curr_stick += partial_first_row_nsticks;
    // NOTE: curr_stick is now at the leftmost edge
    TT_ASSERT(curr_stick % pc.in_w == 0);

    // full rows
    int32_t total_full_rows = (end_stick - curr_stick) / pc.in_w;

    // figure out how many of total full rows belong to partial top image, full images, and partial bottom image

    // partial top image rows
    int32_t partial_top_image_nrows = 0;
    int32_t skip_after_partial_top_image = 0;
    if (curr_batch_stick == 0) {
        // this is beginning of a new image, ie no partial top image
        partial_top_image_nrows = 0;
        skip_after_partial_top_image = 0;
        // if there was a partial top row, insert full padding rows
        if (partial_first_row_nsticks > 0) {
            skip_after_partial_first_row += (pc.in_w + 2 * pc.pad_w) * pc.pad_h;
        }
    } else {
        // we have partial top image with (in_h * in_w - curr_batch_stick) sticks
        // check if my range includes end of this image.
        if ((batch_i + 1) * (pc.in_h * pc.in_w) < end_stick) {
            // yes it does
            int32_t stick_counter = curr_stick;
            while (stick_counter / (pc.in_h * pc.in_w) == batch_i) {
                ++stick_counter;
            }
            partial_top_image_nrows = (stick_counter - curr_stick) / pc.in_w;
            curr_stick += partial_top_image_nrows * pc.in_w;
            // insert full padding rows
            skip_after_partial_top_image = (pc.in_w + 2 * pc.pad_w) * pc.pad_h;
        } else {
            // no it does not
            // count this image in the bottom partial image
            partial_top_image_nrows = 0;
            skip_after_partial_top_image = 0;
        }
    }

    // full images
    int32_t full_nimages = 0;
    int32_t skip_after_full_image = 0;
    // from curr_stick (which is at the left edge), to end_stick, numbewr of rows
    int32_t rem_full_rows = (end_stick - curr_stick) / pc.in_w;
    full_nimages = rem_full_rows / pc.in_h;
    curr_stick += full_nimages * (pc.in_h * pc.in_w);
    if (full_nimages > 0) {
        // insert full padding rows
        skip_after_full_image = (pc.in_w + 2 * pc.pad_w) * pc.pad_h;
    }
    // any remaining rows belong to the bottom partial image
    rem_full_rows = rem_full_rows % pc.in_h;

    // partial bottom image
    int32_t partial_bottom_image_nrows = rem_full_rows;
    curr_stick += rem_full_rows * pc.in_w;

    int32_t partial_last_row_nsticks = (end_stick - curr_stick) % pc.in_w;  // TODO: take care of padding and edges etc

    auto sc = NewShardingConfig{.first_partial_right_aligned_row_width = partial_first_row_nsticks,
                                .first_partial_image_num_rows = partial_top_image_nrows,
                                .num_full_images = full_nimages,
                                .last_partial_image_num_rows = partial_bottom_image_nrows,
                                .last_partial_left_aligned_row_width = partial_last_row_nsticks,
                                .skip_after_partial_right_aligned_row = skip_after_partial_first_row,
                                .skip_after_first_partial_image_row = skip_after_partial_top_image,
                                .skip_after_full_image = skip_after_full_image,
                                .initial_skip = initial_skip,
                                .start_stick = start_stick_padded};

    // log_debug(LogOp, "initial_skip: {}", sc.initial_skip);
    // log_debug(LogOp, "partial_first_row_nsticks: {}", sc.first_partial_right_aligned_row_width);
    // log_debug(LogOp, "partial_top_image_nrows: {}", sc.first_partial_image_num_rows);
    // log_debug(LogOp, "full_nimages: {}", sc.num_full_images);
    // log_debug(LogOp, "partial_bottom_image_nrows: {}", sc.last_partial_image_num_rows);
    // log_debug(LogOp, "partial_last_row_nsticks: {}", sc.last_partial_left_aligned_row_width);
    // log_debug(LogOp, "skip_after_partial_first_row: {}", sc.skip_after_partial_right_aligned_row);
    // log_debug(LogOp, "skip_after_partial_top_image: {}", sc.skip_after_first_partial_image_row);
    // log_debug(LogOp, "skip_after_full_image: {}", sc.skip_after_full_image);

    return sc;
}

// For given pool config output, calculate the out sticks sharding config and the corresponding input shard config
std::tuple<InOutShardingConfig, InOutShardingConfig> get_inout_shard_specs(int32_t start_stick,
                                                                           int32_t end_stick,
                                                                           const PoolConfig& pc,
                                                                           bool to_print) {
    InOutShardingConfig out_sc = InOutShardingConfig{
        .start_stick = 0,
        .first_partial_right_aligned_row_width = 0,
        .first_partial_image_num_rows = 0,
        .num_full_images = 0,
        .last_partial_image_num_rows = 0,
        .last_partial_left_aligned_row_width = 0,
        .initial_skip = 0,
        .skip_after_stick = 0,
        .skip_after_partial_right_aligned_row = 0,
        .skip_after_first_partial_image_row = 0,
        .skip_after_full_image = 0,
        .skip_after_each_full_row = 0,
        .skip_after_each_stick = 0,
    };
    InOutShardingConfig in_sc = out_sc;

    int32_t nsticks = end_stick - start_stick;
    if (nsticks < 1) {
        // range is 0. set everything to 0
        return std::make_tuple(out_sc, out_sc);
    }

    // calculate start_id of my current batch
    int32_t batch_start_stick = (start_stick / (pc.out_h * pc.out_w)) * (pc.out_h * pc.out_w);

    // calculate output sticks coords and corresponding input window's center stick coords:
    int32_t start_batch_i = start_stick / (pc.out_h * pc.out_w);
    int32_t start_batch_stick = start_stick % (pc.out_h * pc.out_w);
    int32_t start_out_h_i = start_batch_stick / pc.out_w;
    int32_t start_out_w_i = start_batch_stick % pc.out_w;
    int32_t start_in_h_i = start_out_h_i * pc.stride_h;
    int32_t start_in_w_i = start_out_w_i * pc.stride_w;
    int32_t start_batch_in_stick = start_in_h_i * (pc.in_w + 2 * pc.pad_w) + start_in_w_i;
    int32_t start_in_stick = start_batch_i * (pc.in_w + 2 * pc.pad_w) * (pc.in_h + pc.pad_h) + start_batch_in_stick;

    int32_t end_batch_i = (end_stick - 1) / (pc.out_h * pc.out_w);
    int32_t end_out_w_i = (end_stick - 1) % pc.out_w;
    int32_t end_out_h_i = ((end_stick - 1) % (pc.out_w * pc.out_h)) / pc.out_w;
    int32_t end_in_h_i = end_out_h_i * pc.stride_h;
    int32_t end_in_w_i = end_out_w_i * pc.stride_w;
    int32_t end_batch_in_stick = end_in_h_i * (pc.in_w + 2 * pc.pad_w) + end_in_w_i;
    int32_t end_in_stick = end_batch_i * (pc.in_w + 2 * pc.pad_w) * (pc.in_h + pc.pad_h) + end_batch_in_stick;

    int32_t curr_out_stick = start_stick;
    int32_t curr_in_stick = start_in_stick;

    // set the per row and per stick skips
    out_sc.skip_after_each_stick = 0;
    in_sc.skip_after_each_stick = pc.stride_w;
    out_sc.skip_after_each_full_row = 0;
    in_sc.skip_after_each_full_row = 2 * pc.pad_w + (pc.stride_h - 1) * (pc.in_w + 2 * pc.pad_w);

    // calculate the initial skip in input if starting stick is left-most and not the first in image
    if (start_out_w_i == 0 && start_out_h_i != 0) {
        in_sc.initial_skip = 2 * pc.pad_w;
    }

    out_sc.skip_after_stick = 0;
    in_sc.skip_after_stick = pc.stride_w;

    if (start_out_h_i == end_out_h_i && end_out_w_i != pc.out_w - 1) {
        // all out sticks belong to the same row
        // treating them as last partial with no skip
        out_sc.start_stick = curr_out_stick;
        out_sc.last_partial_left_aligned_row_width = nsticks;
        in_sc.start_stick = curr_in_stick;
        in_sc.last_partial_left_aligned_row_width = end_in_stick + 1 - start_in_stick;
        return std::make_tuple(in_sc, out_sc);
    }

    if (curr_out_stick % pc.out_w > 0) {
        // we have a partial row
        out_sc.first_partial_right_aligned_row_width = pc.out_w - (curr_out_stick % pc.out_w);
        out_sc.skip_after_partial_right_aligned_row = 0;
        in_sc.first_partial_right_aligned_row_width = pc.in_w - (curr_in_stick % pc.in_w);
        in_sc.skip_after_partial_right_aligned_row = 2 * pc.pad_w + (pc.stride_h - 1) * (pc.in_w + 2 * pc.pad_w);
    } else {
        // no partial row
        out_sc.first_partial_right_aligned_row_width = 0;
        out_sc.skip_after_partial_right_aligned_row = 0;
        in_sc.first_partial_right_aligned_row_width = 0;
        in_sc.skip_after_partial_right_aligned_row = 0;
    }
    curr_out_stick += out_sc.first_partial_right_aligned_row_width + out_sc.skip_after_partial_right_aligned_row;
    curr_in_stick += in_sc.first_partial_right_aligned_row_width + in_sc.skip_after_partial_right_aligned_row;

    // NOTE: curr_out_stick is now at the leftmost edge
    TT_ASSERT(curr_out_stick % pc.out_w == 0);

    // figure out how many of total full rows belong to partial top image, full images, and partial bottom image

    // partial top image rows
    if (start_batch_stick == 0) {
        // this is beginning of a new image, ie no partial top image
        out_sc.first_partial_image_num_rows = 0;
        out_sc.skip_after_first_partial_image_row = 0;
        in_sc.first_partial_image_num_rows = 0;
        in_sc.skip_after_first_partial_image_row = 0;
    } else {
        // we have partial top image with (out_h * out_w - curr_batch_stick) sticks
        // check if my range includes end of this image.
        if ((start_batch_i + 1) * (pc.out_h * pc.out_w) < end_stick) {
            // yes it does
            int32_t out_stick_counter = curr_out_stick;
            int32_t in_stick_counter = curr_in_stick;
            while (out_stick_counter / (pc.out_h * pc.out_w) == start_batch_i) {
                out_stick_counter += 1;
                in_stick_counter += pc.stride_w;
            }
            out_sc.first_partial_image_num_rows = (out_stick_counter - curr_out_stick) / pc.out_w;
            out_sc.skip_after_first_partial_image_row = 0;
            in_sc.first_partial_image_num_rows = (in_stick_counter - curr_in_stick) / pc.in_w;
            in_sc.skip_after_first_partial_image_row = pc.pad_h * (pc.in_w + 2 * pc.pad_w);
            curr_out_stick +=
                out_sc.first_partial_image_num_rows * pc.out_w + out_sc.skip_after_first_partial_image_row;
            curr_in_stick += in_sc.first_partial_image_num_rows * pc.in_w + in_sc.skip_after_first_partial_image_row;
        } else {
            // no it does not
            // count this image in the bottom partial image
            out_sc.first_partial_image_num_rows = 0;
            out_sc.skip_after_first_partial_image_row = 0;
            in_sc.first_partial_image_num_rows = 0;
            in_sc.skip_after_first_partial_image_row = 0;
        }
    }

    // full images
    // from curr_stick (which is at the left edge), to end_stick, number of rows
    int32_t rem_full_out_rows = (end_stick - curr_out_stick) / pc.out_w;
    int32_t rem_full_in_rows = (end_in_stick - curr_in_stick) / (pc.in_w + 2 * pc.pad_w);
    out_sc.num_full_images = rem_full_out_rows / pc.out_h;
    out_sc.skip_after_full_image = 0;
    in_sc.num_full_images = out_sc.num_full_images;  // should be the same
    in_sc.skip_after_full_image = pc.pad_h * (pc.in_w + 2 * pc.pad_w);
    curr_out_stick += out_sc.num_full_images * (pc.out_h * pc.out_w) + out_sc.skip_after_full_image;
    curr_in_stick += in_sc.num_full_images * (pc.in_h * (pc.in_w + 2 * pc.pad_w)) + in_sc.skip_after_full_image;

    // any remaining rows belong to the bottom partial image
    rem_full_out_rows = rem_full_out_rows % pc.out_h;
    rem_full_in_rows = rem_full_in_rows % pc.in_h;
    // partial bottom image
    out_sc.last_partial_image_num_rows = rem_full_out_rows;
    in_sc.last_partial_image_num_rows = rem_full_in_rows;
    curr_out_stick += out_sc.last_partial_image_num_rows * pc.out_w;
    curr_in_stick += in_sc.last_partial_image_num_rows * (pc.in_w + 2 * pc.pad_w);

    out_sc.last_partial_left_aligned_row_width = (end_stick - curr_out_stick) % pc.out_w;
    in_sc.last_partial_left_aligned_row_width = (end_in_stick - curr_in_stick) % (pc.in_w + 2 * pc.pad_w);

    return std::make_tuple(in_sc, out_sc);
}

ShardingConfig get_specs_for_sharding_partition(uint32_t start_stick,
                                                uint32_t end_stick,
                                                uint32_t in_h,
                                                uint32_t in_w,
                                                uint32_t window_w,
                                                uint32_t pad_h,
                                                uint32_t pad_w) {
    /* Logic to compute:
     * NOTE: This logic is wrong if stride !=1
     *   - Input:
     *     start_stick
     *     end_stick
     *     in_h
     *     in_w
     *     window_w
     *     pad_h
     *     pad_w
     *
     *   - Output:
     *     first_partial_right_aligned_row_width
     *     first_partial_image_num_rows
     *     num_full_images
     *     last_partial_image_num_rows
     *     last_partial_left_aligned_row_width
     *     skip_after_partial_right_aligned_row
     *     skip_after_first_partial_image_row
     *     skip_after_full_image
     */

    uint32_t nsticks_per_core = end_stick - start_stick;

    // First partial right-aligned row
    uint32_t image_row_start_left_width = start_stick % in_w;
    uint32_t first_partial_right_aligned_row_width =
        image_row_start_left_width > 0 ? in_w - image_row_start_left_width : 0;

    // Last partial left-aligned row
    uint32_t sticks_after_first_partial_row = nsticks_per_core - first_partial_right_aligned_row_width;
    uint32_t last_partial_left_aligned_row_width = sticks_after_first_partial_row % in_w;

    // Figure out how to allocate full image rows to first partial image, full images, or last partial image
    // This also affects skip after first_partial_right_aligned_row
    uint32_t image_row_start_idx = start_stick / in_w;
    uint32_t image_row_start_idx_after_partial_right_aligned_row =
        (start_stick + first_partial_right_aligned_row_width) / in_w;
    uint32_t image_row_end_idx = end_stick / in_w;
    uint32_t image_start_idx = image_row_start_idx / in_h;
    uint32_t image_start_idx_after_partial_right_aligned_row =
        image_row_start_idx_after_partial_right_aligned_row / in_h;
    uint32_t image_start_height_after_partial_right_aligned_row =
        image_row_start_idx_after_partial_right_aligned_row % in_h;
    uint32_t image_end_idx = image_row_end_idx / in_h;

    // Default case: We don't have a partial right aligned row; so we start with either full images, last partial image,
    // or partial left aligned row
    uint32_t skip_after_partial_right_aligned_row = 0;
    // Case: partial_right_aligned_row > 0 and completes an image
    if (first_partial_right_aligned_row_width > 0 and image_start_height_after_partial_right_aligned_row == 0) {
        skip_after_partial_right_aligned_row = window_w - 1 + pad_h * (in_w + 2 * pad_w);
        // Case: partial_right_aligned_row > 0 and doesn't complete an image
    } else if (first_partial_right_aligned_row_width > 0) {
        skip_after_partial_right_aligned_row = window_w - 1;
    }

    uint32_t first_partial_image_num_rows = 0;
    uint32_t skip_after_first_partial_image_row = 0;
    // Only case where we have first_partial_image_rows: We have at at least 1 completed image and the starting image
    // row is in the middle of an image
    if (image_end_idx - image_start_idx_after_partial_right_aligned_row > 0 and
        image_start_height_after_partial_right_aligned_row > 0) {
        first_partial_image_num_rows = in_h - image_start_height_after_partial_right_aligned_row;
        skip_after_first_partial_image_row = pad_h * (in_w + 2 * pad_w);
    }

    // Full images
    uint32_t image_rows_after_first_partial_image =
        sticks_after_first_partial_row / in_w - first_partial_image_num_rows;
    uint32_t num_full_images = image_rows_after_first_partial_image / in_h;
    uint32_t skip_after_full_image = num_full_images > 0 ? pad_h * (in_w + 2 * pad_w) : 0;

    // Last partial image rows
    uint32_t last_partial_image_num_rows = image_rows_after_first_partial_image % in_h;

    return ShardingConfig{.first_partial_right_aligned_row_width = first_partial_right_aligned_row_width,
                          .first_partial_image_num_rows = first_partial_image_num_rows,
                          .num_full_images = num_full_images,
                          .last_partial_image_num_rows = last_partial_image_num_rows,
                          .last_partial_left_aligned_row_width = last_partial_left_aligned_row_width,
                          .skip_after_partial_right_aligned_row = skip_after_partial_right_aligned_row,
                          .skip_after_first_partial_image_row = skip_after_first_partial_image_row,
                          .skip_after_full_image = skip_after_full_image};
}

}  // namespace tt::tt_metal
