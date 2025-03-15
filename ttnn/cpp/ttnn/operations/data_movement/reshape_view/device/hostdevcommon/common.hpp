#pragma once

namespace ttnn::operations::data_movement::reshape {

struct ReshapeMapping {
    typedef uint32_t value_type;

    // TODO could actually pre-compute and store page address in here
    value_type input_page_index;
    value_type input_tile_offset;
    value_type num_elements;
    value_type output_tile_offset;

    inline bool increment_contiguous(
        const uint32_t other_input_page_index,
        const uint32_t other_input_tile_offset,
        const uint32_t other_output_tile_offset) volatile {
        if (other_input_page_index == input_page_index && other_input_tile_offset == input_tile_offset + 1 &&
            other_output_tile_offset == output_tile_offset + 1) {
            ++num_elements;
            return true;
        } else {
            return false;
        }
    }
};

}  // namespace ttnn::operations::data_movement::reshape
