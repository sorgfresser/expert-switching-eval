import csv
from dataclasses import dataclass


@dataclass
class SweepFallback:
    name: str
    fallbacks: int
    switches_wo_fallback: int
    switches: int
    top_k_experts: int = 0
    top_p_experts: float = 0.0
    top_k_tokens: int = 0
    top_p_tokens: float = 0.0


def read_fallbacks():
    with open("sweep_export.csv") as f:
        reader = csv.DictReader(f)
        fallbacks = []
        for row in reader:
            fallbacks.append(SweepFallback(row["Name"], int(row["fallbacks"]), int(row["switches_wo_fallback"]),
                                           int(row["switches"]), int(row["top_k_experts"]), float(row["top_p_experts"]),
                                             int(row["top_k_tokens"]), float(row["top_p_tokens"])
                                           ))
    return fallbacks


def switches_wo_ratio(fallbacks):
    return sum(fb.switches_wo_fallback for fb in fallbacks) / sum(fb.switches for fb in fallbacks)


def best_switch_wo_ratio(fallbacks):
    return max(fallbacks, key=lambda x: x.switches_wo_fallback / x.switches if x.switches > 0 else 0)

def best_k_switch_wo_ratio(fallbacks, k):
    return sorted(fallbacks, key=lambda x: x.switches_wo_fallback / x.switches if x.switches > 0 else 0, reverse=True)[:k]

def avg_switches_wo_ratio(fallbacks):
    return sum(fb.switches_wo_fallback / fb.switches if fb.switches > 0 else 0 for fb in fallbacks) / len(fallbacks)

def median_switches_wo_ratio(fallbacks):
    sorted_fallbacks = sorted(fallbacks, key=lambda x: x.switches_wo_fallback / x.switches if x.switches > 0 else 0)
    return sorted_fallbacks[len(sorted_fallbacks) // 2]

def runs_with_non_fallback(fallbacks):
    return sum(1 for fb in fallbacks if fb.switches_wo_fallback != 0)


if __name__ == '__main__':
    fallbacks = read_fallbacks()
    # Best switch_wo_ratio
    print(best_switch_wo_ratio(fallbacks))
    # Best k switch_wo_ratio
    best_k = best_k_switch_wo_ratio(fallbacks, 10)
    for fb in best_k:
        print(fb)

    # Average switch_wo_ratio
    print(avg_switches_wo_ratio(fallbacks))

    # Median switch_wo_ratio
    median_fb = median_switches_wo_ratio(fallbacks)
    print(median_fb)
    print(median_fb.switches_wo_fallback / median_fb.switches if median_fb.switches > 0 else 0)


    # Count of runs with switches without fallbacks
    print(runs_with_non_fallback(fallbacks))
