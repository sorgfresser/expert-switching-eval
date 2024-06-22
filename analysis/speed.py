import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class Run:
    name: str
    execution_accuracy: float
    fallbacks: int
    switches_wo_fallback: int
    switches: int
    gpu_type: str
    top_k_experts: int = 0
    top_p_experts: float = 0.0
    top_k_tokens: int = 0
    top_p_tokens: float = 0.0
    runtime_seconds: int = 0


def read_fallbacks():
    with open("sweep_export.csv") as f:
        reader = csv.DictReader(f)
        fallbacks = []
        for row in reader:
            fallbacks.append(Run(row["Name"], float(row["execution_accuracy"]),
                                 int(row["fallbacks"]), int(row["switches_wo_fallback"]),
                                 int(row["switches"]), row["GPU Type"],
                                 int(row["top_k_experts"]), float(row["top_p_experts"]),
                                 int(row["top_k_tokens"]), float(row["top_p_tokens"]), int(row["Runtime"])
                                 ))
    return fallbacks


if __name__ == '__main__':
    fallbacks = read_fallbacks()
    # Extract switches and runtime_seconds data
    switches = [run.switches for run in fallbacks]
    runtime_seconds = [run.runtime_seconds for run in fallbacks]
    # Find GPU types
    types = set(run.gpu_type for run in fallbacks)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    # Color based on gpu type
    for gpu_type in types:
        indices = [i for i, run in enumerate(fallbacks) if run.gpu_type == gpu_type]
        plt.scatter([runtime_seconds[i] for i in indices], [switches[i] for i in indices], label=gpu_type, alpha=0.6,
                    edgecolors='w', linewidth=0.5)
    # plt.scatter(runtime_seconds, switches, color='blue', alpha=0.6, edgecolors='w', linewidth=0.5)

    # Add titles and labels
    plt.title('Scatter Plot of Switches vs. Runtime Seconds')
    plt.ylabel('Number of Switches')
    plt.xlabel('Runtime in Seconds')
    plt.legend(loc="lower right", title="GPU Type")

    # Convert to DataFrame
    data = pd.DataFrame([vars(run) for run in fallbacks])

    # Do regression for each GPU type
    gpu_types = data["gpu_type"].unique()
    for gpu_type in gpu_types:
        # GPU type only one
        data_subset = data[data["gpu_type"] == gpu_type]
        X = data_subset["runtime_seconds"].values.reshape(-1, 1)
        y = data_subset["switches"]

        # Linear Regression
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        y_pred_linear = linear_model.predict(X)

        # Evaluate Linear Regression
        mse_linear = mean_squared_error(y, y_pred_linear)
        print(f"Mean Squared Error for Linear Regression {gpu_type}: {mse_linear}")
        r2_linear = r2_score(y, y_pred_linear)
        print(f"R^2 for Linear Regression {gpu_type}: {r2_linear}")
        # Get coefficients
        print(f"Coefficients for Linear Regression {gpu_type}: {linear_model.coef_}")
        plt.plot(data_subset["runtime_seconds"], y_pred_linear, color='red', label=f"Linear Regression {gpu_type}",
                 linewidth=1)

    # Store
    plt.savefig("scatter_plot.png")
