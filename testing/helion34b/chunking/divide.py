from typing import Any
import json
from typing import Optional
from math import ceil
import argparse
from pathlib import Path

TASK_DIR = Path("tasks/")
TASK_DIR.mkdir(exist_ok=True)


def divide(args_list: list[tuple[Any]], chunk_size: Optional[int] = 1, chunk_count: Optional[int] = None
           ):
    """
    Divide the input data into chunks and save them to disk.

    :param args_list: The input data to divide into chunks. Is a list of arguments. Each entry represents one function call.
    :param chunk_size: The size of each chunk.
    :param chunk_count: The number of chunks to create. Mutually exclusive with chunk_size.
    :return: The results of running the function on each chunk.
    """
    if (chunk_size is None) == (chunk_count is None):
        raise ValueError("Exactly one of chunk_size and chunk_count must be specified.")
    if chunk_count is not None:
        chunk_size = ceil(len(args_list) / chunk_count)
    elif chunk_size is not None:
        chunk_count = ceil(len(args_list) / chunk_size)
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1.")

    idx = 0
    for i in range(0, len(args_list), chunk_size):
        chunk = args_list[i:i + chunk_size]
        with open(TASK_DIR / f"task_{idx}.json", "w") as f:
            json.dump(
                chunk, f
            )
        idx += 1
    print(f"Created {chunk_count} tasks.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Divide task into chunks')
    parser.add_argument('args_file', type=str, help='The file with the arguments for the task')
    parser.add_argument("--chunk-size", type=int, help="The size of each chunk")
    parser.add_argument("--chunk-count", type=int, help="The number of chunks to create")
    args = parser.parse_args()
    with open(args.args_file, "r") as f:
        args_list = json.load(f)
    divide(args_list, args.chunk_size, args.chunk_count)
