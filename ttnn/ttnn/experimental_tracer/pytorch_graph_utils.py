
def format_file_with_black(file_path: str):
    """Format the generated code using black."""
    try:
        import black
        with open(file_path, "r") as f:
            content = f.read()
        formatted_content = black.format_file_contents(content, fast=True, mode=black.FileMode())
        with open(file_path, "w") as f:
            f.write(formatted_content)
    except Exception as e:
        print(f"Skipped formatting {file_path} file with black: {e}")