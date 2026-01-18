tilize
======

.. doxygenfunction:: tilize_init(uint32_t icb, uint32_t block, uint32_t ocb)
.. doxygenfunction:: tilizeA_B_reduce_init(uint32_t icb0, uint32_t icb1_scaler, uint32_t block, uint32_t ocb, uint32_t num_faces = 4, uint32_t face_r_dim = 16)
.. doxygenfunction:: tilize_init_short_with_dt(uint32_t old_icb, uint32_t new_icb, uint32_t block, uint32_t ocb)
.. doxygenfunction:: tilize_block(uint32_t icb, uint32_t block, uint32_t ocb, uint32_t input_tile_index = 0, uint32_t output_tile_index = 0)
.. doxygenfunction:: unpack_tilizeA_B_block(uint32_t icb0, uint32_t icb1, uint32_t block, uint32_t tile_idx_b, uint32_t num_faces = 4, uint32_t srca_face_r_dim = 16)
.. doxygenfunction:: tilize_uninit(uint32_t icb, uint32_t ocb)
.. doxygenfunction:: tilize_uninit_with_dt(uint32_t old_icb, uint32_t new_icb, uint32_t ocb)
