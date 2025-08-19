# User Prompts from Upsample3D Planning Session

## Prompt 1:
How to add documentation and context to you

## Prompt 2:
I need to implement an upsample3d op in ttnn. I already have a file called upsample3d.md. That file is the plan. For now, I just want to edit the plan, so do not generate anything yet. Read that file and fully understand it. Also read the implementation of upsample2d

## Prompt 3:
Ok so here is the thing. I want every single test that is done along the way to be put into a python file!!! INTO A PYTHON FILE !!! MAKE SURE THOSE FILES EXIST AND ARE READABLE TO ME SO I CAN TRACK YOUR PROGRESS. ALSO DO NOT SKIP STEPS!!!! YOU CAN THINK OF WHAT TO IMPLEMENT DURING BUILD TIME, AND GATHER SOME CONTEXT FROM OTHER OPS, BUT TO NOT CHANGE STUFF. MAKE SURE THAT, WHEN I TELL YOU TO ACTUALLY GENERATE ALL OF THIS, YOU ACTUALLY LISTEN TO THIS ADVICE. BE MORE STRICT AND CLEAR THAN I AM RIGHT NOW. DO NOT SKIP STEPS !!!. Also, when you implement kernels, keep in mind that device might hang. SET A TIMEOUT FOR EACH TEST THAT IS reasonable. Build are done with ./build_metal.sh and python env source is source python_env/bin/activate. ALL OF THE TESTS ALONG THE WAY I WANT IN A SEPERATE FOLDER, THAT WILL NOT PUSH ON MAIN, AND I WILL EVENTUALLY DELETE IT (MYSELF) BUT I NEED TO LOOK AT THEM ALONG THE WAY. FINAL TEST, WHICH IS PCC COMPARISON AGAINST PYTORCH, HAS TO EXIST IN THE tests/ FOLDER OF THE REPO. EDIT THE PLAN TO FOLLOW ALL OF THESE

## Prompt 4:
OK BUT ARE YOU SURE ALL OF THESE TESTS MAKE SENSE. IF THEY DO NOT CALL STUFF FROM TTNN, THEY ARE MORE OR LESS USELESS. IF THEY DO NOT ACTUALLY CALL ttnn.upsample3d THEN THEY DO NOTHING. THINK ABOUT THE CALL STACK AND HOW YOU CAN, FROM PYTHON, TEST PARTS OF THE CALL STACK. MAYBE SOME DUMMY OUTPUTS. IF YOU DO NOT HAVE A MEANS OF ACCESSING THAT STUFF IN PYTHON, EITHER DO NOT TESTS THEM, OR ADD SOME PRINTS TO PROGRAM FACTORY TO HELP GUIDE YOU, BUT DELETE THEM LATER ON.

## Prompt 5:
LOWER THE TIMEOUT FOR HANGS. 120 SECONDS IS WAY TOO LONG, PUT IT TO LIKE 20 PER INDIVIDUAL TEST

## Prompt 6:
Can you output all of the prompts I gave you to a new file

## Prompt 7:
Let's make a plan for extending Upsample 3d to support height sharded input and output tensors.
You can find implementation  in this folder ttnn/cpp/ttnn/operations/pool/upsample3d.
Current op supports just row major interleaved tensors.
Now we want to do row major height sharded tensors.
You can take a look at upsample_multi_core_sharded in file ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_sharded.cpp.
You need to do sharding scheme and work parallelisation from the output tensor perspective.
You should ignore config tensor in upsample_multi_core_sharded implementation, instead we want to use
TensorAccessor to fetch height sharded bits of tensor.
You can learn about TensorAccessor in tech_reports/tensor_accessor/tensor_accessor.md
You should stick to strict test driven development and layout plan for tests.
Every test should be saved in in tests/ttnn/unit_tests/operations/test_upsample3d.py
Keep in mind that test can hang when run on device so run each test with tight timeout.
When iterating on the solution, stop as soon as first test starts to fail.
Don't do any work now just make a plan and save it to upsample3d_hs_plan.md

## Prompt 8:
For this we don't need a reader and writer kernel can you take a deeper look at how it's done in upsample
op? Checkout ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_multi_core_sharded.cpp
only this kernel is enough to implement height sharded upsample2d op. Keep ignoring config tensor since we
want to stick to TensorAcccessor.
Let's update the upsample3d_hs_plan.md with this.

## Prompt 9:
In writer kernel impl you need to it from output perspective.
You need calculate which input stick/page contributes to current output element.
Use TensorAccessor to obtain that stick/page.

## Prompt 10:
You should figure out how to use ttnn.create_sharded_memory_config function to prepare memory config
in order to produce valid input height sharded tensors for python tests
