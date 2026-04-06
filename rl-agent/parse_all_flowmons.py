import argparse
import csv
import re
from pathlib import Path

from parse_single_flowmon import parse_flowmon_xml


FLOWMON_FILENAME_RE = re.compile(r"flowmon_r(?P<reward_id>\d+)_s(?P<seed>\d+)_.+_ep(?P<episode>\d+)\.xml$")


def parse_flowmon_filename(file_path):
    match = FLOWMON_FILENAME_RE.fullmatch(file_path.name)
    if not match:
        raise ValueError(f"Unexpected FlowMonitor filename: {file_path.name}")

    return {
        "reward_id": int(match.group("reward_id")),
        "seed": int(match.group("seed")),
        "episode": int(match.group("episode")),
    }


def resolve_logs_dir(input_dir):
    directory = Path(input_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Run directory does not exist: {directory}")

    logs_dir = directory / "logs"
    if logs_dir.is_dir():
        return logs_dir

    return directory


def resolve_output_csv(logs_dir, output_csv, reward_id, seed):
    if output_csv is not None:
        return Path(output_csv).expanduser().resolve()

    run_dir = logs_dir.parent
    train_data_dir = run_dir / "train_data"
    if train_data_dir.is_dir():
        return train_data_dir / f"metrics_r{reward_id}_s{seed}.csv"

    return logs_dir / f"metrics_r{reward_id}_s{seed}.csv"


def list_flowmon_files(logs_dir):
    flowmon_files = []
    for file_path in logs_dir.iterdir():
        if not file_path.is_file() or not file_path.name.startswith("flowmon_"):
            continue
        metadata = parse_flowmon_filename(file_path)
        flowmon_files.append((file_path, metadata))

    if not flowmon_files:
        raise ValueError(f"No flowmon XML files found in: {logs_dir}")

    flowmon_files.sort(key=lambda item: item[1]["episode"])
    return flowmon_files


def parse_run_directory(run_dir, destination_port=None, output_csv=None):
    logs_dir = resolve_logs_dir(run_dir)
    flowmon_files = list_flowmon_files(logs_dir)

    reward_ids = {metadata["reward_id"] for _, metadata in flowmon_files}
    seeds = {metadata["seed"] for _, metadata in flowmon_files}
    if len(reward_ids) != 1 or len(seeds) != 1:
        raise ValueError(
            f"Expected one reward and one seed per run directory, found rewards={sorted(reward_ids)}, seeds={sorted(seeds)}"
        )

    reward_id = next(iter(reward_ids))
    seed = next(iter(seeds))
    output_path = resolve_output_csv(logs_dir, output_csv, reward_id, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for file_path, metadata in flowmon_files:
        metrics = parse_flowmon_xml(str(file_path), destination_port=destination_port)
        row = {
            "reward_id": metadata["reward_id"],
            "seed": metadata["seed"],
            "episode": metadata["episode"],
            "xml_file": file_path.name,
            **metrics,
        }
        rows.append(row)

    fieldnames = [
        "reward_id",
        "seed",
        "episode",
        "xml_file",
        "flow_id",
        "source_address",
        "destination_address",
        "source_port",
        "destination_port",
        "throughput_mbps",
        "avg_delay_ms",
        "loss_rate",
        "rx_packets",
        "tx_packets",
        "lost_packets",
        "rx_bytes",
        "flow_duration_sec",
        "jain_fairness",
        "n_flows",
    ]

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} episodes to {output_path}")
    return rows, output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", help="Run directory or logs directory containing flowmon XML files")
    parser.add_argument("--destination-port", type=int, default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    parse_run_directory(
        args.run_directory,
        destination_port=args.destination_port,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()