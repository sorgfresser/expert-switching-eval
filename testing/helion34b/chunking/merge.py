import argparse
from pathlib import Path
import json


def result_to_dict(result: list[str | int]) -> dict[str, str | int]:
    return {
        "sql": result[0],
        "fallbacks": result[1],
        "switches": result[2],
        "switches_wo_fallback": result[3],
    }


def merge(result_dir: str):
    """
    Merge results from result dir to one large json. Will only use result files, i.e. files with result_ prefix.

    :param result_dir: The directory where the results are stored.
    """
    result_dir = Path(result_dir)
    results = []
    result_files = list(result_dir.glob("result_*.json"))
    # Sort by task_id
    result_files.sort(key=lambda x: int(x.stem.split("_")[1]))
    for result_file in result_files:
        with result_file.open("r") as f:
            results.extend(json.load(f))
    for idx in range(len(results)):
        results[idx] = result_to_dict(results[idx])

    fallbacks = sum(result["fallbacks"] for result in results)
    switches = sum(result["switches"] for result in results)
    switches_wo_fallback = sum(result["switches_wo_fallback"] for result in results)
    print(f"Fallbacks: {fallbacks}\nSwitches: {switches}\nSwitches without fallback: {switches_wo_fallback}")
    with open("result.sql", "w") as f:
        for result in results:
            f.write(result["sql"] + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge task results')
    parser.add_argument('task_dir', type=str, help='The directory where the tasks are stored')
    args = parser.parse_args()
    merge(args.task_dir)
