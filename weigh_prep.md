I want to implement weight preparation for depthwise conv2d.

Terminology:
- STICK: data within the same kernel_h and kernel_w. These one should be contigous in memory. If channels=50, stick is size of 50, because in one stick we have data from all channels

Current state:
- Depthwise conv2d is part of conv2d (same API, but has its own implementation)
- It uses pool kernels (compute pool, reader pool), not ones from conv because it has different implementation
- But, pool kernel for now does not support weights, and we want weights handling because conv2d gets weights
- To test funcionality, I simulated weights as 1s. It is OK. You can check in reader kernel that weight CB is filled with 1s, and than used in compute kernel. Tests are written to have torch.ones as weight tensor
- Weights in depthwise are used to be eltwise multiplied with input, and put result into mul_cb
- The way how in_cb is populated :
    - each row of data is one input stick
    - these sticks are store contigously in memory, which means they are stored ROW_MAJOR logically
    - but in practice, they are consumed as they are TILED - mul_tiles expect data to be in TILED order
    - we unpack data, there is some reshufling (because tiled layout divide data into faces), but when packing is done, it inverse it so order is same
    - that is just a trick, because when doing eltwise multiply, data will be packed in mul_cb in the same way

Weight prep:

- See how weights are prepared in conv2d in conv2d_weight_preparation.md
- That means weights should be stored in the same way, stick by stick, per tile
- In depthwise, in_channels=out_channels=groups - that means weight tensor has size of kernel_h * kernel_w * in_channels
- We want each channel to be stored contigously in memory. If we have 3x3 kernel, in memory we should have first stick (channel), than secon stick, etc.
- Layout is TILED, but in some different way

#### **Target Storage Format: TILED Layout**
- **Tile Structure**: 32×32 elements per tile
- **Face Boundaries**: Ignore 16×16 face boundaries - treat each tile as contiguous 32×32 block
- **Contiguous Placement**: Weight data placed contiguously within tiles without face-level rearrangement

#### **Weight Organization Strategy**
- **Stick Definition**: Each stick (row) contains `in_channels` weight values (values with same indices for kernel_h and kernel_w)
- **Tile Filling**: Multiple sticks placed contiguously within 32×32 tiles
- **Effective Size**: Tensor sized to effective number of tiles needed for all weight data

Example:
```
kernel_h = 3
kernel_w = 3
channels = 96
Total elements per channel: 9 elements (kernel is 3x3x)
Total channels: 96
Total sticks: 9 (one stick per kernel_h kernel_w position)

Tile organization:
- Tile dimensions: 32×32 = 1024 elements
- Tiles needed: (kernel_h * kernel_w * channels) / 1024 = 1
- Rows in tile needed: 3 x 3 x 3 = 27 (32 values per row)
- Other 5 rows are unpopulated

Memory layout per tile:
Row 0: [ch0_k0, ch1_k0, ch2_k0, ch3_k0, ch4_k0, ch5_k0, ... , ch31_k0]
Row 1: [ch32_k0, ch33_k0, ch34_k0, ch35_k0, ch36_k0, ch37_k0, ... , ch63_k0]
Row 2: [ch64_k0, ch65_k0, ch66_k0, ch67_k0, ch68_k0, ch69_k0, ... , ch95_k0]
Row 3: [ch0_k1, ch1_k1, ch2_k1, ch3_k1, ch4_k1, ch5_k1, ... , ch31_k1]
Row 4: [ch32_k1, ch33_k1, ch34_k1, ch35_k1, ch36_k1, ch37_k1, ... , ch63_k1]
Row 5: [ch64_k1, ch65_k1, ch66_k1, ch67_k1, ch68_k1, ch69_k1, ... , ch95_k1]
...
Row 24: [ch0_k8, ch1_k8, ch2_k8, ch3_k8, ch4_k8, ch5_k8, ... , ch31_k8]
Row 25: [ch32_k8, ch33_k8, ch34_k8, ch35_k8, ch36_k8, ch37_k8, ... , ch63_k8]
Row 26: [ch64_k8, ch65_k8, ch66_k8, ch67_k8, ch68_k8, ch69_k8, ... , ch95_k8]
Rows 27-31: unused (empty rows in tile)
```
- That is how data should be stored in DRAM - we need to consume that data as well
- First core is supposed to read whole data, and store it in weight CB. So stick by stick in weight CB
- We need to send rt arg to first core that he is sender, and will load that data
- It should send data to other cores, and other readers will get same data throught mcast (we will recognize them with same rt arg)
- Each core should put data in weight cb
- Compute kernel will need to wait that data, and then use it in mul_tiles (already simulated with 1s)


Steps:
- building is done by ./build_metal.sh --release
- running test is done by source python_env/bin/activate && pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::test_groups_vs_pool2. You may need to include
- test may hang. so you need to run them with 30s timeout. If test hang, run tt-smi -r 0
- For debugging host code, you may run log_info(...)
- Fore debugging kernel, you may use DPRINT << ... << ENDL(); You can check for DPRINT in generated folder, if TT_METAL_DPRINT_ONE_FILE_PER_
RISC is enabled
- If you make changes in .py or kernel, you don't need to rebuild

I want from you to make some plan, and store it in new file how we will achieve this.
When makeing a plan, make some header with progress, commands, goal, and make sure you do hard stop after every substep. Also do not proceed to new step if current is not passing.
Some rough steps are:
- Make a test that is easy to debug (for example all values in same stick are same)
- Preparing weights on the host
    - Having path to prepare weights for depthwise conv2d
    - Reordering data, to have weights stick by stick
    - Checking if tensor is prepared correctly on host. You can have weight tensor that will have sticks with values 1, 2, ... (each stick will have same values). In pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py::test_groups_vs_pool2 you will notice that weight tensor is made of 1s, you need to change that
- Getting data from DRAM to first core, and populate weight CB
    - You should make sure that data is fetched well. You can use tt::data_movement::common::print_bf16_pages(get_read_ptr(weight_cb_id), values in channel, kernel_h * kernel_w), and in generated folder you can se BRISC output. You need run test with flags TT_METAL_CLEAR_L1=0 TT_METAL_DPRINT_ONE_FILE_PER_
RISC=1 TT_METAL_DPRINT_CORES="(0,0)"
- For now, there will be a test just for 1 core conv2d. It should not send data to other cores for now.
- Use that in compute kernel. Make sure that data is there by running UNPACK( tt::compute::common::print_full_tile(weight_cb_id, 0) ), and than checking TRISC0 in generated folder. You need to check this after waiting for weight cb.
- If everything is OK, in this stage test should pass


Make me some plan how we can achieve this. I want data from here there.
Plan should have a lot of small steps. For each step, we need to know what is goal of step, what we need to do to achieve that, and how it will be tested.
Before marking some step is done, you need to be sure test for that step passes. Do not go to another step if current step is not done. I want a lot of small steps.

DOING ONE SUBSTEP IN A TIME. WHEN FINISH SOME SUBSTEP, HARD STOP! CONTINUE JUST WHEN TRIGGERED.
WHEN FINISH SOMETHING (MAKE SURE TEST PASSES), MARK IT AS COMPLETED!
