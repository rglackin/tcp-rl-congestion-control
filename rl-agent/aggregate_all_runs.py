import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from parse_all_flowmons import parse_run_directory

import seaborn as sns


METRICS = [
    ("throughput_mbps", "Throughput (Mbps)"),
    ("avg_delay_ms", "Average Delay (ms)"),
    ("loss_rate", "Loss Rate"),
    ("jain_fairness", "Jain Fairness"),
]

TRAINING_METRICS = [
    ("avg_reward", "Average Reward"),
    ("total_reward", "Total Reward"),
]

RUN_ID_RE = re.compile(r"r(?P<reward_id>\d+)_s(?P<seed>\d+)$")


def format_reward_label(reward_id):
    return "NewReno" if int(reward_id) == 0 else f"Reward {int(reward_id)}"


def discover_run_directories(dir_paths):
    run_dirs = set()
    for directory in dir_paths:
        if (directory / "logs").is_dir():
            run_dirs.add(directory.resolve())

        for child in directory.iterdir():
            if child.is_dir() and (child / "logs").is_dir():
                run_dirs.add(child.resolve())

    return sorted(run_dirs)


def resolve_input_csvs(inputs, auto_generate_metrics=True, destination_port=None):
    metric_csv_paths = []
    training_csv_paths = []
    dir_paths = []
    run_dirs_needing_metrics = set()

    def classify_and_add_csv(csv_path):
        name = csv_path.name
        if name.startswith("metrics_r"):
            metric_csv_paths.append(csv_path)
        elif name.startswith("q_r"):
            training_csv_paths.append(csv_path)
        else:
            training_csv_paths.append(csv_path)

    for raw_input in inputs:
        path = Path(raw_input).expanduser()

        if any(token in raw_input for token in "*?[]"):
            for csv_path in path.parent.glob(path.name):
                if csv_path.is_file() and csv_path.suffix == ".csv":
                    classify_and_add_csv(csv_path)
            continue

        if path.is_file() and path.suffix == ".csv":
            classify_and_add_csv(path)
            continue

        if path.is_dir():
            dir_paths.append(path.resolve())

            for csv_path in path.glob("train_data/metrics_r*_s*.csv"):
                classify_and_add_csv(csv_path)
            for csv_path in path.glob("metrics_r*_s*.csv"):
                classify_and_add_csv(csv_path)
            for csv_path in path.glob("**/metrics_r*_s*.csv"):
                classify_and_add_csv(csv_path)

            run_dirs_from_path = discover_run_directories([path.resolve()])
            if auto_generate_metrics:
                for run_dir in run_dirs_from_path:
                    has_metrics = any(run_dir.glob("train_data/metrics_r*_s*.csv")) or any(
                        run_dir.glob("metrics_r*_s*.csv")
                    )
                    if has_metrics:
                        continue
                    run_dirs_needing_metrics.add(run_dir)

            for csv_path in path.glob("train_data/q_r*.csv"):
                classify_and_add_csv(csv_path)
            for csv_path in path.glob("q_r*.csv"):
                classify_and_add_csv(csv_path)
            for csv_path in path.glob("**/q_r*.csv"):
                classify_and_add_csv(csv_path)
            continue

        raise ValueError(f"Input path does not exist or is unsupported: {raw_input}")

    unique_metric_csvs = sorted({csv_path.resolve() for csv_path in metric_csv_paths if csv_path.is_file()})
    unique_training_csvs = sorted({csv_path.resolve() for csv_path in training_csv_paths if csv_path.is_file()})

    if auto_generate_metrics and run_dirs_needing_metrics:
        generated_csvs = []
        for run_dir in sorted(run_dirs_needing_metrics):
            try:
                _, output_csv_path = parse_run_directory(
                    str(run_dir),
                    destination_port=destination_port,
                    output_csv=None,
                )
                generated_csvs.append(Path(output_csv_path).resolve())
            except Exception as error:
                print(f"Skipping auto-generation for {run_dir}: {error}")

        unique_metric_csvs = sorted(
            {csv_path for csv_path in unique_metric_csvs if csv_path.is_file()}
            | {csv_path for csv_path in generated_csvs if csv_path.is_file()}
        )

    if unique_metric_csvs:
        return unique_metric_csvs

    if unique_training_csvs:
        return unique_training_csvs

    raise ValueError(
        "No compatible CSV files found from the provided inputs. "
        "Expected `metrics_r*_s*.csv` or `q_r*.csv`, or run directories containing FlowMonitor XML files."
    )


def infer_result_type(combined):
    flow_columns = {"throughput_mbps", "avg_delay_ms", "loss_rate", "jain_fairness"}
    training_columns = {"run_id", "total_reward", "avg_reward"}

    if flow_columns.issubset(set(combined.columns)):
        return "flow"
    if training_columns.issubset(set(combined.columns)):
        return "training"

    raise ValueError(
        "Unsupported CSV schema. Expected FlowMonitor metrics columns "
        "or training reward columns (`run_id`, `total_reward`, `avg_reward`)."
    )


def ensure_reward_seed_columns(combined, result_type):
    if {"reward_id", "seed"}.issubset(set(combined.columns)):
        return combined

    if result_type != "training":
        missing_columns = {"reward_id", "seed", "episode"} - set(combined.columns)
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if "run_id" not in combined.columns:
        raise ValueError("Training CSVs must include `run_id` when `reward_id`/`seed` are absent")

    extracted = combined["run_id"].astype(str).str.extract(RUN_ID_RE)
    if extracted["reward_id"].isnull().any() or extracted["seed"].isnull().any():
        bad_values = combined.loc[extracted["reward_id"].isnull() | extracted["seed"].isnull(), "run_id"].unique()
        raise ValueError(f"Could not parse reward/seed from run_id values: {sorted(bad_values)}")

    combined["reward_id"] = extracted["reward_id"].astype(int)
    combined["seed"] = extracted["seed"].astype(int)
    return combined


def load_results(csv_paths):
    dataframes = []
    for csv_path in csv_paths:
        dataframe = pd.read_csv(csv_path)
        dataframe["source_csv"] = str(csv_path)
        dataframes.append(dataframe)

    combined = pd.concat(dataframes, ignore_index=True)
    result_type = infer_result_type(combined)
    combined = ensure_reward_seed_columns(combined, result_type)

    required_columns = {"reward_id", "seed", "episode"}
    missing_columns = required_columns - set(combined.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    sorted_data = combined.sort_values(["reward_id", "seed", "episode"]).reset_index(drop=True)
    return sorted_data, result_type


def select_final_window(all_data, final_episodes):
    if final_episodes <= 0:
        raise ValueError("final_episodes must be positive")

    return (
        all_data.groupby(["reward_id", "seed"], group_keys=False)
        .tail(final_episodes)
        .reset_index(drop=True)
    )


def build_summary(final_data, result_type):
    if result_type == "flow":
        summary = final_data.groupby("reward_id").agg(
            throughput_mbps_mean=("throughput_mbps", "mean"),
            throughput_mbps_std=("throughput_mbps", "std"),
            avg_delay_ms_mean=("avg_delay_ms", "mean"),
            avg_delay_ms_std=("avg_delay_ms", "std"),
            loss_rate_mean=("loss_rate", "mean"),
            loss_rate_std=("loss_rate", "std"),
            jain_fairness_mean=("jain_fairness", "mean"),
            jain_fairness_std=("jain_fairness", "std"),
            n_samples=("episode", "count"),
            n_seeds=("seed", "nunique"),
        )
        summary = summary.round(6).reset_index()
        summary.insert(1, "reward_label", summary["reward_id"].apply(format_reward_label))
        return summary

    summary = final_data.groupby("reward_id").agg(
        avg_reward_mean=("avg_reward", "mean"),
        avg_reward_std=("avg_reward", "std"),
        total_reward_mean=("total_reward", "mean"),
        total_reward_std=("total_reward", "std"),
        n_samples=("episode", "count"),
        n_seeds=("seed", "nunique"),
    )
    summary = summary.round(6).reset_index()
    summary.insert(1, "reward_label", summary["reward_id"].apply(format_reward_label))
    return summary


def plot_metric_with_fallback(ax, data, metric, title):
    reward_ids = sorted(data["reward_id"].dropna().unique())
    reward_labels = [format_reward_label(reward_id) for reward_id in reward_ids]

    if sns is not None:
        sns.boxplot(data=data, x="reward_id", y=metric, ax=ax)
        ax.set_xticks(range(len(reward_labels)))
        ax.set_xticklabels(reward_labels)
    else:
        grouped = [series[metric].to_numpy() for _, series in data.groupby("reward_id")]
        labels = [format_reward_label(reward_id) for reward_id, _ in data.groupby("reward_id")]
        ax.boxplot(grouped, labels=labels)

    ax.set_title(title)
    ax.set_xlabel("Controller")
    ax.set_ylabel(title)


def save_plots(final_data, output_dir, result_type):
    metric_list = METRICS if result_type == "flow" else TRAINING_METRICS
    n_metrics = len(metric_list)

    if n_metrics == 4:
        figure, axes = plt.subplots(2, 2, figsize=(12, 10))
        plot_axes = list(axes.flat)
    else:
        figure, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4.5))
        plot_axes = [axes] if n_metrics == 1 else list(axes)

    for axis, (metric, title) in zip(plot_axes, metric_list):
        plot_metric_with_fallback(axis, final_data, metric, title)

    for axis in plot_axes[n_metrics:]:
        axis.axis("off")

    figure.tight_layout()
    plot_path = output_dir / "reward_comparison.png"
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def aggregate_results(inputs, output_dir, final_episodes, auto_generate_metrics=True, destination_port=None):
    csv_paths = resolve_input_csvs(
        inputs,
        auto_generate_metrics=auto_generate_metrics,
        destination_port=destination_port,
    )
    all_data, result_type = load_results(csv_paths)
    final_data = select_final_window(all_data, final_episodes=final_episodes)
    summary = build_summary(final_data, result_type=result_type)

    output_dir.mkdir(parents=True, exist_ok=True)

    combined_csv_path = output_dir / "all_results_combined.csv"
    summary_csv_path = output_dir / "all_results_summary.csv"
    final_window_csv_path = output_dir / f"final_{final_episodes}_episodes.csv"

    all_data.to_csv(combined_csv_path, index=False)
    final_data.to_csv(final_window_csv_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
    plot_path = save_plots(final_data, output_dir, result_type=result_type)

    print(f"Loaded {len(csv_paths)} CSV files ({result_type} mode)")
    print(f"Saved combined data to {combined_csv_path}")
    print(f"Saved final-window data to {final_window_csv_path}")
    print(f"Saved summary table to {summary_csv_path}")
    print(f"Saved comparison plot to {plot_path}")
    print("\nSummary table:")
    print(summary.to_string(index=False))

    return {
        "csv_paths": csv_paths,
        "result_type": result_type,
        "all_data": all_data,
        "final_data": final_data,
        "summary": summary,
        "combined_csv_path": combined_csv_path,
        "final_window_csv_path": final_window_csv_path,
        "summary_csv_path": summary_csv_path,
        "plot_path": plot_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Metrics CSV files, glob patterns, run directories, or a runs root directory",
    )
    parser.add_argument(
        "--output-dir",
        default="flowmon-analysis",
        help="Directory where combined CSVs and plots will be written",
    )
    parser.add_argument(
        "--final-episodes",
        type=int,
        default=50,
        help="Number of final episodes per reward/seed run to include in the summary and plots",
    )
    parser.add_argument(
        "--destination-port",
        type=int,
        default=None,
        help="Destination port to pass through to FlowMonitor parsing when auto-generating metrics CSVs",
    )
    parser.add_argument(
        "--no-auto-generate-metrics",
        action="store_true",
        help="Disable auto-generation of metrics CSVs from FlowMonitor XML files",
    )
    args = parser.parse_args()

    aggregate_results(
        inputs=args.inputs,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        final_episodes=args.final_episodes,
        auto_generate_metrics=not args.no_auto_generate_metrics,
        destination_port=args.destination_port,
    )


if __name__ == "__main__":
    main()