# Fused Matmul Helpers — Verification

**Commit**: a62a03c2181e083484fb6ba0496610b2d66c0ba7
**Branch**: wransom/fused2

## Summary

- **CONFIRMED**: 54 claims
- **INCORRECT**: 3 claims (C-049, C-056, C-059)
- **UNVERIFIABLE**: 0

## INCORRECT Claims — Corrections

### C-049 (fifo ptr manipulation count in D1)

**Claimed**: "~8 locations" of fifo_rd_ptr/fifo_wr_ptr manipulation
**Actual**: ~17+ distinct fifo pointer assignments across the kernel
**Impact on design**: LOW — the count was wrong but the cited line ranges are correct. The conv kernel has extensive CB pointer manipulation that the fused helper should NOT own.

### C-056 (D1 matmul core "structurally identical" to C1)

**Claimed**: D1 matmul core is structurally identical to C1
**Actual**: The tile_regs_acquire/reload ordering differs:
- C1 (line 283): `tile_regs_acquire()` BEFORE `if (enable_reload)` check
- D1 (line 396): `tile_regs_acquire()` INSIDE the `if (enable_reload)` branch, separate acquire at line 415 in the `else` branch

**Impact on design**: MEDIUM — the fused helper must handle this difference. The existing matmul_block helper uses C1's pattern (acquire before reload). D1's pattern acquires inside the reload path. For the fused helper, C1's pattern is preferred (simpler, already proven in existing helper). D1 migration would need to match whichever pattern the helper uses. Note C-057 already correctly documented this difference, making C-056 internally inconsistent.

### C-059 (D1 bias section "nearly identical" to C1)

**Claimed**: D1 bias section after K-loop is nearly identical to C1:425-484
**Actual**: Two structural differences:
1. D1 waits for `out_block_num_tiles` once before subblock loops (D1:556); C1 waits `out_subblock_num_tiles` per subblock (C1:445)
2. D1 pops partials per subblock during bias add (D1:578); C1 also pops per subblock (C1:459) but the wait granularity differs

**Impact on design**: MEDIUM — the bias_add helper already uses per-subblock waits (matching C1). D1's block-level wait is an optimization (single wait for all subblocks). The fused helper should use per-subblock waits (matching existing helper and C1) — D1's optimization can be deferred.

## Incomplete Claims (Confirmed but Notable Omissions)

- **C-008**: Omits that `reload_from_cb_to_dst` internally calls `wait_front`/`pop_front` on mm_partials. Important for encapsulation: the reload function owns its CB management.
- **C-032**: Only shows UNPACK mailbox_read; MATH and PACK threads each independently read from BriscThreadId.

## Design Implications

1. The fused helper should use C1's acquire-before-reload pattern (matches existing matmul_block helper)
2. The bias_add helper's per-subblock wait pattern is correct and should be preserved
3. D1's conv-specific patterns (fifo pointer manipulation, tilize, split reader) confirm these should stay outside the fused helper
4. The gathered variants (C2/C3) are correctly classified as Tier 3 (out of scope)
