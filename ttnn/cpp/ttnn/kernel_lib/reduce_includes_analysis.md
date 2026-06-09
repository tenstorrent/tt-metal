# Reduce Helper Library - Include Cleanup Analysis

This document analyzes which includes can be safely removed from kernel files that use `reduce_helpers.hpp`.

**Important**: This analysis does NOT rely on transitive includes. An include is only marked as removable
if the file does not directly use any functionality from that header.

## Analysis Criteria

An include is **completely redundant** only if:
1. The functionality is available transitively through another include, AND
2. The file does NOT directly call any functions from that header

## Per-File Analysis

### 1. deepseek_grouped_gate.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce` from reduce_helpers.hpp)
- `common.h`: YES - only file that explicitly includes this, but used via other includes
- `tile_move_copy.h`: YES - uses `copy_tile_init`, `copy_tile`
- `reconfig_data_format.h`: YES - uses `reconfig_data_format`, `pack_reconfig_data_format`
- `pack.h`: YES - uses `pack_tile`

**Can remove:**
- `compute_kernel_api/reduce.h` - no direct reduce_init/reduce_tile/reduce_uninit calls
- `compute_kernel_api/common.h` - no unique functions used; common.h is included via other headers

---

### 2. ssm_1d_sum_reduce.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 3. fused_distributed_rmsnorm/rmsnorm_post_allgather.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 4. fused_distributed_rmsnorm/rmsnorm_pre_allgather.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 5. groupnorm.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)
- `tile_move_copy.h`: YES - uses `copy_tile_init`, `copy_tile`

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 6. groupnorm_sharded_v2.cpp
**Direct function usage analysis:**
- `reduce.h`: **YES** - uses `reduce_init`, `reduce_tile`, `reduce_uninit` directly (7 occurrences)
- `tile_move_copy.h`: YES - uses `copy_tile_init`, `copy_tile`

**Can remove:**
- None - all includes are used directly

---

### 7. layernorm_sharded_pre_allgather.cpp
**Direct function usage analysis:**
- `reduce.h`: **YES** - uses `reduce_init`, `reduce_tile`, `reduce_uninit` directly (6 occurrences)
- `tile_move_copy.h`: YES - uses `copy_tile_init`, `copy_tile`

**Can remove:**
- None - all includes are used directly

---

### 8. layernorm_distributed/layernorm_pre_allgather.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 9. layernorm_distributed/layernorm_pre_allgather_2d.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 10. rmsnorm_distributed/rmsnorm_post_allgather.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 11. rmsnorm_distributed/rmsnorm_pre_allgather.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 12. rmsnorm_distributed/rmsnorm_pre_allgather_2d.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 13. softmax/softmax.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)
- `tile_move_copy.h`: YES - uses `copy_tile_init`, `copy_tile`

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 14. softmax/softmax_large_tensor.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)
- `tile_move_copy.h`: YES - uses `copy_tile_init`, `copy_tile`

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 15. softmax/softmax_sharded.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)
- `tile_move_copy.h`: YES - uses `copy_tile_init`, `copy_tile`

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 16. reduce_h.cpp
**Status:** Only includes `reduce_helpers.hpp` - already minimal.

---

### 17. reduce_hw.cpp
**Status:** Only includes `reduce_helpers.hpp` - already minimal.

---

### 18. reduce_w.cpp
**Status:** Includes `reduce_helpers.hpp` and `matmul.h` - both needed.

---

### 19. moe.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)
- `tile_move_copy.h`: YES - uses `copy_tile_init`, `copy_tile`, `matmul_pack_tile`
- `reconfig_data_format.h`: YES - uses `reconfig_data_format`, `pack_reconfig_data_format`
- `pack.h`: YES - uses `pack_tile`

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 20. sampling.cpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)
- `tile_move_copy.h`: YES - uses `copy_tile_init`, `copy_tile`
- `reconfig_data_format.h`: YES - uses `reconfig_data_format`, `pack_reconfig_data_format`
- `pack.h`: YES - uses `pack_tile`

**Can remove:**
- `compute_kernel_api/reduce.h`

---

### 21. tilize_untilize_compute.cpp
**Direct function usage analysis:**
- `reduce.h`: **YES** - uses `reduce_init`, `reduce_tile` directly (2 occurrences)

**Can remove:**
- None - reduce.h is used directly

---

### 22. sdpa_decode/compute_common.hpp
**Direct function usage analysis:**
- `reduce.h`: NO direct usage (uses `compute_kernel_lib::reduce`)
- `tile_move_copy.h`: YES - uses `copy_tile_init`, `copy_tile`

**Can remove:**
- `compute_kernel_api/reduce.h`

---

## Summary Table

| File | Can Remove `reduce.h`? | Reason |
|------|------------------------|--------|
| deepseek_grouped_gate.cpp | ✅ Yes | Uses library wrapper only |
| ssm_1d_sum_reduce.cpp | ✅ Yes | Uses library wrapper only |
| fused_rmsnorm_post_allgather.cpp | ✅ Yes | Uses library wrapper only |
| fused_rmsnorm_pre_allgather.cpp | ✅ Yes | Uses library wrapper only |
| groupnorm.cpp | ✅ Yes | Uses library wrapper only |
| **groupnorm_sharded_v2.cpp** | ❌ No | Direct reduce_init/tile/uninit calls |
| **layernorm_sharded_pre_allgather.cpp** | ❌ No | Direct reduce_init/tile/uninit calls |
| layernorm_pre_allgather.cpp | ✅ Yes | Uses library wrapper only |
| layernorm_pre_allgather_2d.cpp | ✅ Yes | Uses library wrapper only |
| rmsnorm_post_allgather.cpp | ✅ Yes | Uses library wrapper only |
| rmsnorm_pre_allgather.cpp | ✅ Yes | Uses library wrapper only |
| rmsnorm_pre_allgather_2d.cpp | ✅ Yes | Uses library wrapper only |
| softmax.cpp | ✅ Yes | Uses library wrapper only |
| softmax_large_tensor.cpp | ✅ Yes | Uses library wrapper only |
| softmax_sharded.cpp | ✅ Yes | Uses library wrapper only |
| reduce_h.cpp | N/A | Not included |
| reduce_hw.cpp | N/A | Not included |
| reduce_w.cpp | N/A | Not included |
| moe.cpp | ✅ Yes | Uses library wrapper only |
| sampling.cpp | ✅ Yes | Uses library wrapper only |
| **tilize_untilize_compute.cpp** | ❌ No | Direct reduce_init/tile calls |
| compute_common.hpp | ✅ Yes | Uses library wrapper only |

## Conclusion

**Files where `reduce.h` can be safely removed: 19 out of 22**

Only 3 files must keep `reduce.h` because they directly call low-level reduce functions:
1. `groupnorm_sharded_v2.cpp`
2. `layernorm_sharded_pre_allgather.cpp`
3. `tilize_untilize_compute.cpp`

**All other includes** (`tile_move_copy.h`, `pack.h`, `reconfig_data_format.h`, etc.) should be kept
because they are used directly in the kernel files - NOT just available transitively.
