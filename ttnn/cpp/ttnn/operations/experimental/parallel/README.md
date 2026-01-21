# ttnn.parallel and ttnn.sequential

## Overview
This document describes two new experimental features in tt-metal: `ttnn.parallel` and `ttnn.sequential`. Together they provide a way of composing a directed graph of ttnn operations in C++ and/or Python, where adjacent branches are dispatched and run in parallel on disjoint core grids.

Currently, dispatching ops to run in parallel would have to be done through subdevices, which dispatches a separate program to each subdevice. However, for ops that fit on a single device and run in parallel, it would be more flexible to construct and dispatch a single program that has the appropriate kernels built for each core. This would automatically sync the parallel ops upon completion.

`ttnn.parallel` builds a single composite `Program` object by merging the program factory logic for each of the parallel ops to run, where each factory is restricted to a disjoint `CoreRangeSet` on the device. Kernels, semaphores, and runtime arguments are built per-core into a single `Program`.

`ttnn.sequential` is a simple wrapper around a sequence of operations that are dispatched and run one after the other. It is used only for graph expressiveness, and does not do anything "smart" like chain CBs or together in some optimized way or fuse kernels (though the opportunity is there).

Together, `ttnn.parallel` and `ttnn.sequential` can be used to generate and dispatch complex op graphs in Python:

```python
```
