void kernel_main() {
  uint32_t num_tiles = get_arg_val<uint32_t>(0);
  uint32_t start_id = get_arg_val<uint32_t>(1);

  uint32_t end_id = start_id + num_tiles;
  for (uint32_t i = start_id; i < end_id; ++i) {
    cb_reserve_back(tt::CBIndex::c_0, 1);
    cb_push_back(tt::CBIndex::c_0, 1);

    cb_reserve_back(tt::CBIndex::c_1, 1);
    cb_push_back(tt::CBIndex::c_1, 1);

    cb_reserve_back(tt::CBIndex::c_2, 1);
    cb_push_back(tt::CBIndex::c_2, 1);

    cb_reserve_back(tt::CBIndex::c_3, 1);
    cb_push_back(tt::CBIndex::c_3, 1);
  }
}

