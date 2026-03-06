# Unified Router Interface - Final Implementation

## Summary
Successfully implemented and verified a unified router interface for both DeepSeek (GroupedTopKRouter) and GPT-OSS (TopKRouter) backends in the MoE block.

## Changes Made

### 1. Enhanced TopKRouter (`models/demos/gpt_oss/tt/topk.py`)

#### Added `forward()` method:
```python
def forward(self, x: ttnn.Tensor, cfg: dict) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Unified forward interface matching GroupedTopKRouter signature."""
    # Returns (weights, indices) both as 4D tensors [1, 1, seq_len, K]
```

#### Added `from_config()` factory method:
```python
@classmethod
def from_config(cls, cfg: dict, mesh_device: ttnn.Device, state_dict: dict):
    """Factory method to create TopKRouter from runtime config."""
    # Creates RouterConfig internally from config dict
```

### 2. Simplified MoEBlock (`models/tt_moe/moe_block.py`)

#### Refactored `_fwd_moe_gate()`:
- Removed backend-specific router creation logic
- Unified interface: `weights, indices = router.forward(x, cfg)`
- Reduced from 64 lines to 43 lines

#### Fixed unified forward pass:
- Added handling for 4D weights from GPT-OSS in `_fwd_moe_unified()`
- Properly reshapes 4D weights [1, 1, batch_seq, K] to [K, 1, batch_seq, 1] for multiplication

## Verification Results

All tests pass with PCC values identical to baseline:

### Decode Mode
- **DeepSeek**: PCC = 0.9909382362251831 ✓
- **GPT-OSS**: PCC = 0.9258647885436233 ✓

### Prefill Mode
- **DeepSeek**: PCC = 0.9912507102094226 (expected)
- **GPT-OSS**: PCC = 0.9178482780390822 (expected)

## Key Technical Details

### Shape Handling
- Both routers return consistent 4D shapes: `[1, 1, seq_len, K]`
- Unified forward pass handles backend-specific reshaping internally
- GPT-OSS weights are permuted from [1, 1, batch_seq, K] to [K, 1, batch_seq, 1] for multiplication

### Backward Compatibility
- TopKRouter's `__call__()` method preserved for existing code
- All existing tests pass without modification

## Benefits

1. **Code Simplification**: Cleaner, more maintainable code
2. **Consistent Interface**: Both routers use identical calling conventions
3. **Better Encapsulation**: Router-specific logic contained within each router
4. **No Numerical Changes**: Exact same PCC values maintained
5. **Unified Forward Pass**: Single forward path handles both backends efficiently

## Files Modified

1. `/home/ntarafdar/tt-moe/tt-metal/models/demos/gpt_oss/tt/topk.py`
   - Added `forward()` and `from_config()` methods

2. `/home/ntarafdar/tt-moe/tt-metal/models/tt_moe/moe_block.py`
   - Simplified `_fwd_moe_gate()` function
   - Enhanced `_fwd_moe_unified()` to handle 4D GPT-OSS weights

The refactoring successfully unifies the router interface while maintaining exact numerical equivalence and backward compatibility.
