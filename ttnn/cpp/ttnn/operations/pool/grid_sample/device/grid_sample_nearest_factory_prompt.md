#  Objective
I need you to work with me on implementing a new mode for the existing grid sample TTNN operation (ttnn/cpp/ttnn/operations/pool/grid_sample/grid_sample.cpp) that does nearest-neighbor sampling on 4D (N, H, W, C) tensors in row-major DRAM interleaved layout. We should start with designing the program factory (in cpp), as this is the place where data flow and tensor layout are agreed upon. DO NOT move forward with kernel and Python Binding implementations before we agree on how the factory should look like.

# Execution plan and patterns to look at
To better understand the pattern, I need you to look at the implementation of the existing grid sample operation program factory in ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_program_factory.cpp
You can consult the existing Torch documentation on grid sampling functionality
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
You should think deep and do the following, step by step

## Step 1
Read the provided operation example and gather all the knowledge around the underlying concepts of the current program factory implementation:

**Work unit definition**: What constitutes one unit of work? (row, tile, block?)
**Circular buffer management**: How are CBs sized and allocated?
**Data flow pattern**: How does data move through reader → CB → compute → writer?
**Index calculations**: How are tensor indices mapped to/from linear memory indices?
**Memory access patterns**: What order is data read/written?
**Layout-specific handling**: Differences between row-major, tiled, sharded layouts
**Core distribution**: How is work split across available cores?
**Compile-time vs runtime arguments**: What parameters are fixed vs dynamic?

Make sure you use Deep Wiki for gathering context around TTNN program factories and other concepts mentioned above.

Write your findings in a temporary file grid_sample_context.md in this TTNN operation's folder

## Step 2
Produce a plan on how to approach expanding the functionality with "nearest" sampling, which steps need to be taken and how it differs from "bilinear". Save this in grid_sample_nearest_factoryPlan.md in this TTNN operation's folder. DO NOT outline specific code changes in kernel or other files, this should NOT be a part of the plan. Focus ONLY on program factory for now, with a note that kernels should be modified/added later on (or optionally reused if applicable). It is OK for the new factory to point to non-existing kernels with a "ToDo:" comment. This is a one-shot generation of a program factory skeleton that we will iterate later on. It doesn't need to necessairly run or even compile, but you should do your best effort to make it as close to the (predicted) final version as possible.

## Step 3
Write a new program factory ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_nearest_program_factory.cpp
