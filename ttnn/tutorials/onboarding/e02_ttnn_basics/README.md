# E02: TTNN Basics

Learn to use ttnn ops and the pytest workflow.

## Goals

1. Implement `matmul_add(device, a, b, c)` that computes `a @ b + c` on a Tenstorrent device
2. Get familiar with the pytest framework used for testing at Tenstorrent

## Files

- `reference.py` - PyTorch reference implementation
- `exercise.py` - Your implementation (fill in the TODOs)
- `solution.py` - Complete implementation for comparison
- `test.py` - Pytest that compares your implementation against reference

## Workflow

1. Read `reference.py` to understand the expected behavior
2. Implement the TODOs in `exercise.py`
3. Test your implementation:

```bash
./run.sh "e02 and exercise"
```

4. Compare with solution:

```bash
./run.sh "e02 and solution"
```
