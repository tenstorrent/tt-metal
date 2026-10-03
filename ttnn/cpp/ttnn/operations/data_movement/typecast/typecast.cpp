# Fix: `ttnn.typecast` to uint16 must truncate, not round

## Problem

`ttnn.typecast` to **uint16** rounds to nearest, while **every other integer destination type** (uint8, uint16's siblings uint32/int8/int16/int32) and the **host reference path** truncate toward zero. This inconsistency means device results diverge from the host path and from the documented integer-cast semantics.

## Root cause

In the typecast op dispatch, the uint16 case falls through to the generic float→float rounding path (which applies round-to-nearest-even) instead of the integer-conversion path that truncates. All other integer destinations are routed through the truncating integer path.

## Fix

Route `uint16` through the same truncating integer-conversion path used by the other integer destinations, so the device matches the host reference. Concretely:

1. In the typecast op's dtype dispatch, add `uint16` to the set of integer destinations that use `convert_to_int_truncate` (rather than the float rounding path).
2. Ensure the sharded/blocked variants share the same code path so no layout regresses.
3. Update the op docstring to state that **all integer destinations, including uint16, truncate toward zero**.

### Illustrative patch

```cpp
// Before: uint16 fell through to the rounding (float) path
switch (output_dtype) {
    case DataType::UINT8:
    case DataType::UINT32:
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
        return convert_to_int_truncate(input);
    // case DataType::UINT16: MISSING -> fell through to rounding path
    default:
        return convert_round_nearest(input);
}

// After: uint16 joins the truncating integer path
switch (output_dtype) {
    case DataType::UINT8:
    case DataType::UINT16:   // <-- fixed: truncate like every other integer dest
    case DataType::UINT32:
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
        return convert_to_int_truncate(input);
    default:
        return convert_round_nearest(input);
}
```

## Tests

Add a regression test asserting device == host for uint16 typecast at values that distinguish rounding from truncation:

```python
import torch
import ttnn

def test_typecast_uint16_truncates():
    # 2.5 -> truncate 2, round-to-nearest-even 2 ; 3.5 -> truncate 3, round 4
    x = torch.tensor([0.4, 0.6, 1.5, 2.5, 3.5, 4.9, 255.9], dtype=torch.float32)
    ref = x.to(torch.uint16)  # host path truncates toward zero
    dev = ttnn.to_torch(ttnn.typecast(ttnn.from_torch(x), ttnn.uint16))
    assert torch.equal(dev.to(torch.int32), ref.to(torch.int32)), (dev, ref)
```

Also cover:
- negative inputs (clamp/UB behavior consistent with other uint destinations),
- values above uint16 max (saturation consistent with host),
- a sharded layout case to confirm the fix applies on all layouts.

## Impact

Makes `ttnn.typecast`→uint16 consistent with the host path and with all other integer destinations, eliminating silent rounding divergence in models that quantize to uint16.