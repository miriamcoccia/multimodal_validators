import os
from pathlib import Path
import time


def combine_python_files_fast(
    source_dir, output_file="project.txt", exclude_dirs=None, show_progress=True
):
    """
    Fast version that combines Python files with minimal overhead.
    Args:
        source_dir: Directory containing Python files
        output_file: Name of the output text file
        exclude_dirs: Set of directory names to exclude
        show_progress: Show progress while processing
    """
    start_time = time.time()
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"Error: Directory '{source_dir}' does not exist")
        return

    # Default exclusions (using set for O(1) lookup)
    if exclude_dirs is None:
        exclude_dirs = {
            "__pycache__",
            ".git",
            "venv",
            "env",
            ".venv",
            "node_modules",
            ".tox",
            "dist",
            "build",
            ".eggs",
        }
    else:
        exclude_dirs = set(exclude_dirs)

    # Collect files first (faster than filtering during iteration)
    python_files = []

    # Use os.walk for better performance than Path.rglob
    for root, dirs, files in os.walk(source_dir):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # Add Python files from current directory
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)

    if not python_files:
        print(f"No Python files found in '{source_dir}'")
        return

    # Sort files
    python_files.sort()

    if show_progress:
        print(f"Found {len(python_files)} Python files. Combining...")

    # Use larger buffer for better I/O performance
    with open(output_file, "w", encoding="utf-8", buffering=8192) as outfile:
        for i, py_file in enumerate(python_files):
            if show_progress and (i + 1) % 10 == 0:
                print(f"Processing: {i + 1}/{len(python_files)} files...")

            # Get relative path
            try:
                relative_path = py_file.relative_to(source_path)
            except ValueError:
                relative_path = py_file

            # Write header
            outfile.write("-" * 50 + "\n")
            outfile.write(f"{relative_path}\n")
            outfile.write("-" * 50 + "\n")

            # Read and write file contents
            try:
                with open(py_file, "r", encoding="utf-8", errors="replace") as infile:
                    # Read entire file at once (faster than line by line)
                    contents = infile.read()
                    outfile.write(contents)

                    if contents and not contents.endswith("\n"):
                        outfile.write("\n")

                    if i < len(python_files) - 1:
                        outfile.write("\n")

            except Exception as e:
                outfile.write(f"# Error reading file: {e}\n\n")

    elapsed = time.time() - start_time
    print(f"\n✓ Successfully combined {len(python_files)} files into '{output_file}'")
    print(f"  Time taken: {elapsed:.2f} seconds")

    # Show file size
    output_size = Path(output_file).stat().st_size
    if output_size > 1024 * 1024:
        print(f"  Output size: {output_size / (1024 * 1024):.2f} MB")
    else:
        print(f"  Output size: {output_size / 1024:.2f} KB")


def combine_specific_files(file_list, output_file="project.txt"):
    """
    Combine only specific Python files (useful for large projects).
    Args:
        file_list: List of specific file paths to include
        output_file: Name of the output text file
    """
    with open(output_file, "w", encoding="utf-8", buffering=8192) as outfile:
        for i, py_file in enumerate(file_list):
            py_file = Path(py_file)

            if not py_file.exists():
                print(f"Warning: {py_file} does not exist, skipping...")
                continue

            outfile.write("-" * 50 + "\n")
            outfile.write(f"{py_file.name}\n")
            outfile.write("-" * 50 + "\n")

            try:
                with open(py_file, "r", encoding="utf-8", errors="replace") as infile:
                    contents = infile.read()
                    outfile.write(contents)

                    if contents and not contents.endswith("\n"):
                        outfile.write("\n")

                if i < len(file_list) - 1:
                    outfile.write("\n")

            except Exception as e:
                outfile.write(f"# Error reading file: {e}\n\n")

    print(f"Combined {len(file_list)} files into '{output_file}'")


def combine_by_pattern(source_dir, patterns=None, output_file="project.txt"):
    """
    Combine only files matching specific patterns.
    Args:
        source_dir: Directory to search
        patterns: List of patterns like ['**/models/*.py', '**/views/*.py']
        output_file: Output file name
    """
    if patterns is None:
        patterns = ["**/*.py"]

    source_path = Path(source_dir)
    python_files = []

    for pattern in patterns:
        python_files.extend(source_path.glob(pattern))

    # Remove duplicates and sort
    python_files = sorted(set(python_files))

    if not python_files:
        print(f"No files found matching patterns: {patterns}")
        return

    with open(output_file, "w", encoding="utf-8", buffering=8192) as outfile:
        for i, py_file in enumerate(python_files):
            relative_path = py_file.relative_to(source_path)

            outfile.write("-" * 50 + "\n")
            outfile.write(f"{relative_path}\n")
            outfile.write("-" * 50 + "\n")

            try:
                with open(py_file, "r", encoding="utf-8", errors="replace") as infile:
                    contents = infile.read()
                    outfile.write(contents)

                    if contents and not contents.endswith("\n"):
                        outfile.write("\n")

                    if i < len(python_files) - 1:
                        outfile.write("\n")

            except Exception as e:
                outfile.write(f"# Error reading file: {e}\n\n")

    print(f"Combined {len(python_files)} files into '{output_file}'")


# Example usage
if __name__ == "__main__":
    # FAST: Basic usage
    combine_python_files_fast(".", "project.txt")

    # For very large projects, combine only specific directories
    # combine_by_pattern('.', patterns=['src/**/*.py', 'lib/**/*.py'], output_file='project.txt')

    # Or list specific files
    # files_to_combine = [
    #     'main.py',
    #     'models/user.py',
    #     'utils/helpers.py'
    # ]
    # combine_specific_files(files_to_combine, 'project.txt')
