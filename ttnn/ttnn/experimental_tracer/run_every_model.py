from sample_tracer import allowed_modes, get_parser, main

# Define a specific input shape
input_shape = [[1, 3, 2048, 2048]]  # Nested list to match the expected format in sample_tracer.py
input_dtype = ["float32"]  # Default input dtype

# make directory if it does not exist
import os

output_dir = "/home/salnahari/testing_dir/tt-metal/ttnn/ttnn/experimental_tracer/models_pytest/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through all modes and run the main function
for mode in allowed_modes:
    print(f"Running mode: {mode}")
    try:
        # Prepare arguments dictionary
        args_dict = {
            "model": mode,
            "input_shape": input_shape,
            "input_dtype": input_dtype,
            "disable_torch_summary": True,  # Set to True to disable torch summary
            "no_infer": True,
        }
        # Call the main function directly
        main(args_dict)

        # move generated files to the appropriate directory
        file_path = f"test.py"
        if os.path.exists(file_path):
            # move file to output directory
            os.rename(file_path, os.path.join(output_dir, f"{mode}.py"))
    except Exception as e:
        print(f"Error occurred while running mode {mode}: {e}")
