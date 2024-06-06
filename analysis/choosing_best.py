import csv
from dataclasses import dataclass


@dataclass
class Run:
    name: str
    execution_accuracy: float
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
            fallbacks.append(Run(row["Name"], float(row["execution_accuracy"]),
                                 int(row["fallbacks"]), int(row["switches_wo_fallback"]),
                                 int(row["switches"]), int(row["top_k_experts"]), float(row["top_p_experts"]),
                                 int(row["top_k_tokens"]), float(row["top_p_tokens"])
                                 ))
    return fallbacks


def only_working_switching(fallbacks):
    return list(filter(lambda x: x.switches_wo_fallback != 0, fallbacks))


def only_highest_execution_accuracy(fallbacks):
    highest_acc = max(fallbacks, key=lambda x: x.execution_accuracy).execution_accuracy
    return list(filter(lambda x: x.execution_accuracy == highest_acc, fallbacks))


def only_with_expert_stopping(fallbacks):
    return list(filter(lambda x: x.top_k_experts != 0 and x.top_p_experts != 0.0, fallbacks))


def only_with_token_limits(fallbacks):
    return list(filter(lambda x: x.top_k_tokens != 0 and x.top_p_tokens != 0.0, fallbacks))


def with_good_switching_ratio(fallbacks):
    return list(filter(lambda x: (x.switches_wo_fallback / x.switches if x.switches > 0 else 0) > 0.015, fallbacks))


def with_high_switches_count(fallbacks):
    return list(filter(lambda x: x.switches > 3000, fallbacks))

def max_switches_wo_ratio(fallbacks):
    max_val = max(fallbacks, key=lambda x: x.switches_wo_fallback / x.switches if x.switches > 0 else 0)
    ratio = max_val.switches_wo_fallback / max_val.switches if max_val.switches > 0 else 0
    return list(filter(lambda x: (x.switches_wo_fallback / x.switches if x.switches > 0 else 0) == ratio, fallbacks))

if __name__ == '__main__':
    fallbacks = read_fallbacks()
    fallbacks = only_working_switching(fallbacks)
    fallbacks = only_highest_execution_accuracy(fallbacks)
    fallbacks = only_with_expert_stopping(fallbacks)
    fallbacks = only_with_token_limits(fallbacks)
    fallbacks = with_good_switching_ratio(fallbacks)
    fallbacks = with_high_switches_count(fallbacks)

    # Select the max switch_wo_ratio
    fallbacks = max_switches_wo_ratio(fallbacks)
    # Select top_k tokens 20 to allow many tokens but not the max value of 40
    fallbacks = list(filter(lambda x: x.top_k_tokens == 20, fallbacks))
    print(fallbacks)
    