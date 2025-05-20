#include <tt-metalium/constants.hpp>

using namespace tt::constants;

/*
 * @brief Zero pads a part of a face in a tile, dictated by the number of elements to pad on the right and bottom edges
 * of the face.
 *
 * This function zero pads a part of a face in a tile. It handles both width and height padding
 * by setting the specified number of elements to zero on the right and bottom edges of the face.
 *
 * @tparam T The type of the elements in the tile
 * @param num_elements_padded_w Number of elements to pad on the right edge (width)
 * @param num_elements_padded_h Number of elements to pad on the bottom edge (height)
 * @param tile_ptr Pointer to the start of the face in the tile
 */
template <typename T>
void fill_pad_face(uint32_t num_elements_padded_w, uint32_t num_elements_padded_h, T* tile_ptr, T fill_value) {
    // Right padding (width padding)
    if (num_elements_padded_w > 0) {
        for (uint32_t row = 0; row < FACE_HEIGHT; ++row) {
            auto row_start = tile_ptr + row * FACE_WIDTH + (FACE_WIDTH - num_elements_padded_w);
            for (uint32_t col = 0; col < num_elements_padded_w; ++col) {
                row_start[col] = fill_value;
            }
        }
    }

    // Bottom padding (height padding)
    if (num_elements_padded_h > 0) {
        for (uint32_t row = FACE_HEIGHT - num_elements_padded_h; row < FACE_HEIGHT; ++row) {
            auto row_start = tile_ptr + row * FACE_WIDTH;
            // TODO: Use NOC for faster zero padding - can be done in a single write from MEM_ZERO_ADDR
            // noc_async_read(zeros_noc_addr, row_start, num_elements_padded_h * sizeof(T));
            for (uint32_t col = 0; col < FACE_WIDTH; ++col) {
                row_start[col] = fill_value;
            }
        }
    }
}

/**
 * @brief Zero pads a tile by handling padding for each face within the tile.
 *
 * This function handles zero padding for a tile by breaking it down into faces and applying
 * padding to each face individually. It calculates the appropriate padding for each face
 * based on the unpadded dimensions and applies the padding using fill_pad_face.
 *
 * @tparam T The type of the elements in the tile
 * @tparam num_faces_w Number of faces in the width dimension (default: TILE_WIDTH / FACE_WIDTH)
 * @tparam num_faces_h Number of faces in the height dimension (default: TILE_HEIGHT / FACE_HEIGHT)
 * @param num_elements_unpadded_w Number of unpadded elements in the width dimension
 * @param num_elements_unpadded_h Number of unpadded elements in the height dimension
 * @param l1_tile_ptr Pointer to the start of the tile
 */
template <typename T, uint32_t num_faces_w = TILE_WIDTH / FACE_WIDTH, uint32_t num_faces_h = TILE_HEIGHT / FACE_HEIGHT>
void fill_pad_tile(
    uint32_t num_elements_unpadded_w, uint32_t num_elements_unpadded_h, uint32_t l1_tile_ptr, uint32_t fill_value) {
    auto tile_ptr = reinterpret_cast<T*>(l1_tile_ptr);

    for (uint32_t face_h = 0; face_h < num_faces_h; ++face_h) {
        for (uint32_t face_w = 0; face_w < num_faces_w; ++face_w) {
            // Calculate face offset
            uint32_t face_offset = (face_h * num_faces_w + face_w) * FACE_HW;
            auto face_ptr = tile_ptr + face_offset;
            // Calculate padding for this specific face
            if (num_elements_unpadded_w > 0) {
                uint32_t face_pad_w = num_elements_unpadded_w < (face_w * FACE_WIDTH)
                                          ? FACE_WIDTH
                                          : (FACE_WIDTH - num_elements_unpadded_w % FACE_WIDTH);
                fill_pad_face<T>(face_pad_w, 0, face_ptr, fill_value);
            }

            if (num_elements_unpadded_h > 0) {
                uint32_t face_pad_h = num_elements_unpadded_h < (face_h * FACE_HEIGHT)
                                          ? FACE_HEIGHT
                                          : (FACE_HEIGHT - num_elements_unpadded_h % FACE_HEIGHT);
                fill_pad_face<T>(0, face_pad_h, face_ptr, fill_value);
            }
        }
    }
}
