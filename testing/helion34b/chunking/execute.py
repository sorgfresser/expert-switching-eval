import argparse
from spider_from_args import generate_one
import json
from pathlib import Path


def execute_task(task_id, task_dir):
    print(f"Executing task {task_id} from {task_dir}")
    task_dir = Path(task_dir)
    task_file = task_dir / f"task_{task_id}.json"
    with task_file.open("r") as f:
        task = json.load(f)
    results = []
    for arguments in task:
        results.append(generate_one(*arguments))
    # Dump the results to a file
    result_file = task_dir / f"result_{task_id}.json"
    with result_file.open("w") as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute task chunk')
    parser.add_argument('task_id', type=int, help='The task ID to execute')
    parser.add_argument("task_dir", type=str, help="The directory where the tasks are stored")
    args = parser.parse_args()
    execute_task(args.task_id, args.task_dir)
