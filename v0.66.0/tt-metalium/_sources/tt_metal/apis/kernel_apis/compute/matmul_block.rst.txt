matmul_block
============

.. doxygenfunction:: mm_block_init(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id, const uint32_t transpose=0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1)
.. doxygenfunction:: mm_block_init_short(uint32_t in0_cb_id, uint32_t in1_cb_id, const uint32_t transpose=0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1)
.. doxygenfunction:: mm_block_init_short_with_dt(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t old_in1_cb_id, const uint32_t transpose=0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1)
.. doxygenfunction:: matmul_block(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst, const uint32_t transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim)
