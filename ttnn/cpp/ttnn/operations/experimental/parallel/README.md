# ttnn.parallel

## Overview
This document describes `ttnn.parallel`, an experimental feature in tt-metal that provides a way of composing multiple ttnn operations to run in parallel on disjoint core grids and dispatch them as a single fused program.

Currently, dispatching ops to run in parallel would have to be done through subdevices, which dispatches a separate program to each subdevice. However, for ops that fit on a single device and run in parallel, it would be more flexible to construct and dispatch a single program that has the appropriate kernels built for each core. This would automatically sync the parallel ops upon completion.

`ttnn.parallel` builds a single composite `Program` object by merging the program factory logic for each of the parallel ops to run, where each factory is restricted to a disjoint `CoreRangeSet` on the device. Kernels, semaphores, and runtime arguments are built per-core into a single `Program`.

```python
```
