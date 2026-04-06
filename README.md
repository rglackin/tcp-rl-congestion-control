# TCP RL Congestion Control Project

## Overview

This repository contains the core source files for a Final Year Project investigating the question:

**What is the optimal reward function for reinforcement learning models handling TCP network congestion?**

The project studies whether a reinforcement learning agent can learn useful TCP congestion-control behaviour in an `ns-3` dumbbell topology, and how strongly that behaviour depends on the reward function used during training. The work is deliberately focused on **reward design** rather than on building the strongest possible RL controller.

## What This Repository Contains

This repository includes the following files:

### 1. `ns3-scripts/tcp_dumbbell_env.cc`

This is the `ns-3` scenario used as the reinforcement learning environment.

It defines:
- a dumbbell network topology with a shared bottleneck,
- the observation space exposed to the agent,
- the discrete action space for congestion-window adjustment,
- the reward functions evaluated in the study,
- the episode step logic and FlowMonitor export.

In practical terms, this file is the **network side of the experiment**. It decides what the agent can observe, what it can control, and how each action is scored.

### 2. `rl-agent/qlearn.py`

This is the main reinforcement learning driver.

It implements:
- tabular Q-learning,
- state discretisation,
- epsilon-greedy exploration,
- per-episode `ns-3` process launching,
- JSONL logging and run-directory creation,
- export of learned Q-tables and episode reward CSVs.

This file is the **learning side of the project**. It is where reward-driven policy learning happens.

The decision to use **tabular Q-learning** was intentional: it keeps the learning algorithm simple and inspectable so that differences in outcome can be attributed more directly to reward design, rather than being obscured by the additional representational power of deep neural networks.

### 3. `rl-agent/parse_single_flowmon.py`

This script parses a single FlowMonitor XML file produced by `ns-3` and extracts the metrics used in the study, including:
- throughput,
- average delay,
- loss rate,
- Jain fairness,
- flow metadata.

Its purpose is to convert raw simulator output into a form suitable for quantitative analysis.

### 4. `rl-agent/parse_all_flowmons.py`

This script applies the single-file parser across a full run directory.

It:
- locates FlowMonitor XML files,
- infers reward ID, seed, and episode number from filenames,
- parses each episode,
- writes a consolidated metrics CSV.

This file bridges the gap between **raw simulation logs** and **structured per-run datasets**.

### 5. `rl-agent/aggregate_all_runs.py`

This script aggregates results across multiple runs and seeds.

It supports:
- loading metrics CSVs or training reward CSVs,
- generating missing metrics files from FlowMonitor XML when needed,
- selecting the final evaluation window,
- building summary tables,
- producing plots across reward functions and seeds.

This is the **analysis and summarisation layer** of the project.

### 6. `requirements.txt`

This file records the Python packages used by the analysis and agent-side scripts.

## How the Pieces Fit Together

At a high level, the project works as follows:

1. `tcp_dumbbell_env.cc` defines the network environment and reward function.
2. `qlearn.py` launches the environment and trains a tabular Q-learning agent.
3. `ns-3` produces FlowMonitor XML outputs for each episode.
4. `parse_single_flowmon.py` and `parse_all_flowmons.py` convert those XML files into CSV metrics.
5. `aggregate_all_runs.py` combines results across runs and builds the summaries used for interpretation.

This separation is important to the project design: the simulator, learner, parser, and aggregator are kept distinct so that changes in reward function can be traced clearly through the pipeline.

## Research Scope Reflected in the Code

The code in this repository reflects a deliberately constrained experimental design:

- **Focus on reward functions:** multiple reward definitions are compared under the same learning setup.
- **Simple learner by design:** tabular Q-learning is used for interpretability and methodological clarity.
- **Controlled network scenario:** the dumbbell topology isolates congestion effects in a clean setting.
- **Post-run metric extraction:** FlowMonitor parsing is treated as a separate analysis step, not mixed into training logic.

This means the repository should be read primarily as a **research artifact**, not as a general-purpose TCP implementation or a production RL networking framework.