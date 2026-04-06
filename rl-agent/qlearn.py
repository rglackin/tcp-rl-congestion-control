#!/usr/bin/env python3
"""
Tabular Q-Learning for ns-3 Gym TCP dumbbell environment.
Usage:
    python qlearn.py --port=5555 --rewardId=1 --episodes=1000
"""

import argparse
import contextlib
import io
import csv
import json
import os
import subprocess
import threading
import time
import numpy as np
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from ns3gym import ns3env

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--port',      type=int,   default=5555)
parser.add_argument('--rewardId',  type=int,   default=1,         help='1-4')
parser.add_argument('--episodes',  type=int,   default=1000)
parser.add_argument('--simTime',   type=float, default=2.0)
parser.add_argument('--stepTime',  type=float, default=0.2)
parser.add_argument('--seed',      type=int,   default=1)
parser.add_argument('--nFlows',    type=int,   default=1,
                    help='Number of TCP flows to pass to ns-3 scenario')
parser.add_argument('--quiet',     action='store_true',
                    help='Suppress stdout JSON logs; file logging remains enabled')
parser.add_argument('--simScript', type=str, default='tcp_dumbbell_env',
                    help='ns-3 program name in scratch/, e.g. tcp_dumbbell_env')
parser.add_argument('--ns3Dir',    type=str, default='/home/ruairi/ns-3.40',
                    help='Path to ns-3 root directory containing ./ns3')
parser.add_argument('--mode',      type=str, choices=('rl', 'newreno'), default='rl',
                    help='Congestion-control mode to pass to ns-3 (rl|newreno)')
parser.add_argument('--forceAction', type=int, default=None,
                    help='Optional fixed action index to use for every step (e.g. 2)')
args = parser.parse_args()

# Resolve paths relative to the script's directory so the agent can be
# invoked from any working directory (e.g. the ns-3 root).
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
RUN_ID = f"r{args.rewardId}_s{args.seed}"
RUN_TS = time.strftime("%Y%m%d-%H%M%S", time.localtime())
FLOWMON_XML_STEM = f"flowmon_{RUN_ID}_{RUN_TS}"

_RUNS_DIR = os.path.join(_PROJECT_ROOT, 'runs')
_RUN_ROOT_DIR = os.path.join(_RUNS_DIR, f"run_{args.rewardId}_{RUN_TS}")
_RUN_LOG_DIR = os.path.join(_RUN_ROOT_DIR, 'logs')
_RUN_TRAIN_DIR = os.path.join(_RUN_ROOT_DIR, 'train_data')

os.makedirs(_RUNS_DIR, exist_ok=True)
os.makedirs(_RUN_ROOT_DIR, exist_ok=True)
os.makedirs(_RUN_LOG_DIR, exist_ok=True)
os.makedirs(_RUN_TRAIN_DIR, exist_ok=True)

args.out = os.path.join(_RUN_TRAIN_DIR, f'q_r{args.rewardId}.npy')
args.logFile = os.path.join(_RUN_LOG_DIR, f"qlearn_{RUN_ID}_{RUN_TS}.jsonl")

csv_path = args.out.replace('.npy', '.csv')

# Hyperparams
ALPHA   = 0.1
GAMMA   = 0.99
EPS_START = 1.0
EPS_END   = 0.05

# State discretisation
# thru      : [0, 10] Mbps   → 11 bins (0..10)
# norm_cwnd : [0, ~2]        → 11 bins (0..10) with 0.2 width
# util      : [0, 1]         →  6 bins (0..5) with 0.2 width
# rtt_ratio : [1.0, 2.0]     →  6 bins (0..5) with 0.2 width
# state = b_thru + b_cwnd * 11 + b_util * 11 * 11 + b_rtt * 11 * 11 * 6
#
# Total states : 11 * 11 * 6 * 6 = 4356

N_THRU = 11
N_RATE = 11
N_UTIL = 6
N_RTT  = 6
N_STATES  = N_THRU * N_RATE * N_UTIL * N_RTT   # 4356
N_ACTIONS = 5
LOG_FILE_HANDLE = None

if args.forceAction is not None and not (0 <= args.forceAction < N_ACTIONS):
    raise ValueError(f'--forceAction must be in [0, {N_ACTIONS - 1}], got {args.forceAction}')

log_dir = os.path.dirname(args.logFile)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)
LOG_FILE_HANDLE = open(args.logFile, 'a', buffering=1)


def emit_log(source, level, event, **fields):
    record = {
        "ts": time.time(),
        "source": source,
        "level": level,
        "event": event,
    }
    for key, value in fields.items():
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        record[key] = value
    log_line = json.dumps(record, sort_keys=True)
    if not args.quiet:
        print(log_line, flush=True)
    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.write(log_line + "\n")


KEEP_PATTERNS = ('flow', 'reward', 'thru', 'util', 'rate', 'step', 'done', 'obs', 'action')


def pump_subprocess_logs(process, episode):
    if process.stdout is None:
        return

    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue

        line_lower = line.lower()
        if not any(p in line_lower for p in KEEP_PATTERNS):
            continue

        emit_log(
            source="ns3",
            level="info",
            event="flow_reward",
            episode=episode,
            program=args.simScript,
            pid=process.pid,
            message=line,
        )


def discretize(obs):
    """Map obs [thru, norm_cwnd, util, rtt_ratio] → integer state index."""
    thru, norm_cwnd, util, rtt_ratio = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])

    b_thru = int(np.clip(np.floor(thru),          0, N_THRU - 1))
    b_cwnd = int(np.clip(np.floor(norm_cwnd / 0.2), 0, N_RATE - 1))
    b_util = int(np.clip(np.floor(util / 0.2),    0, N_UTIL - 1))
    b_rtt  = int(np.clip(np.floor((rtt_ratio - 1.0) / 0.2), 0, N_RTT - 1))

    return (
        b_thru
        + b_cwnd * N_THRU
        + b_util * N_THRU * N_RATE
        + b_rtt * N_THRU * N_RATE * N_UTIL
    )


# Connect to ns-3 (manual launch per episode with startSim=False)
env_kwargs = {
    "port": args.port,
    "stepTime": args.stepTime,
    "startSim": False,
    "simSeed": args.seed,
    "simArgs": {},
    "debug": False,
}

def launch_ns3_episode(episode: int) -> tuple[subprocess.Popen, str]:
    flowmon_xml_path = os.path.join(_RUN_LOG_DIR, f"{FLOWMON_XML_STEM}_ep{episode}.xml")
    ns3_bin = f"{args.ns3Dir}/ns3"
    run_args = (
        f"{args.simScript} "
        f"--openGymPort={args.port} "
        f"--simSeed={args.seed} "
        f"--simTime={args.simTime} "
        f"--envStepTime={args.stepTime} "
        f"--nFlows={args.nFlows} "
        f"--rewardId={args.rewardId} "
        f"--mode={args.mode} "
        f"--flowmonXml={flowmon_xml_path}"
    )
    process = subprocess.Popen(
        [ns3_bin, "run", run_args],
        cwd=args.ns3Dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return process, flowmon_xml_path


emit_log(
    source="qlearn",
    level="info",
    event="startup",
    run_id=RUN_ID,
    states=N_STATES,
    actions=N_ACTIONS,
    alpha=ALPHA,
    gamma=GAMMA,
    eps_start=EPS_START,
    eps_end=EPS_END,
    episodes=args.episodes,
    reward_id=args.rewardId,
    seed=args.seed,
    n_flows=args.nFlows,
    ns3_dir=args.ns3Dir,
    ns3_script=args.simScript,
    mode=args.mode,
    force_action=args.forceAction,
    log_file=args.logFile,
    quiet=args.quiet,
    q_table_out=args.out,
    rewards_csv=csv_path,
    run_dir=_RUN_ROOT_DIR,
    run_log_dir=_RUN_LOG_DIR,
    run_train_dir=_RUN_TRAIN_DIR,
    run_timestamp=RUN_TS,
    flowmon_xml_pattern=os.path.join(_RUN_LOG_DIR, f"{FLOWMON_XML_STEM}_ep<N>.xml"),
)

# ---------------------------------------------------------------------------
# Q-table
# ---------------------------------------------------------------------------
Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
episode_avg_rewards = []

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['run_id', 'episode', 'total_reward', 'avg_reward'])
    emit_log(source="qlearn", level="info", event="csv_opened", path=csv_path, run_id=RUN_ID)

    for ep in range(1, args.episodes + 1):
        ns3_proc, flowmon_xml_path = launch_ns3_episode(ep)
        ns3_log_thread = threading.Thread(
            target=pump_subprocess_logs,
            args=(ns3_proc, ep),
            daemon=True,
        )
        ns3_log_thread.start()
        emit_log(
            source="qlearn",
            level="info",
            event="episode_start",
            episode=ep,
            ns3_pid=ns3_proc.pid,
            flowmon_xml=flowmon_xml_path,
        )
        time.sleep(0.25)
        env = ns3env.Ns3Env(**env_kwargs)
        emit_log(
            source="qlearn",
            level="info",
            event="env_connected",
            episode=ep,
            port=args.port,
        )

        if ep == 1:
            emit_log(
                source="qlearn",
                level="info",
                event="spaces",
                episode=ep,
                observation_space=str(env.observation_space),
                action_space=str(env.action_space),
            )
            obs_shape = getattr(env.observation_space, "shape", None)
            action_n = getattr(env.action_space, "n", None)
            if obs_shape != (4,):
                raise RuntimeError(
                    f"Unexpected observation space shape: {obs_shape}; expected (4,)"
                )
            if action_n != N_ACTIONS:
                raise RuntimeError(
                    f"Unexpected action space size: {action_n}; expected {N_ACTIONS}"
                )

        # Linear ε decay
        eps = EPS_START - (EPS_START - EPS_END) * (ep - 1) / max(args.episodes - 1, 1)
        emit_log(source="qlearn", level="info", event="epsilon", episode=ep, value=eps)

        obs = env.reset()
        emit_log(source="qlearn", level="info", event="reset", episode=ep, observation=obs)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            state = discretize(obs)

            # ε-greedy action
            if args.forceAction is not None:
                action = int(args.forceAction)
            else:
                if np.random.random() < eps:
                    action = np.random.randint(N_ACTIONS)
                else:
                    action = int(np.argmax(Q[state]))

            obs2, reward, done, info = env.step(action)
            reward = float(reward)
            total_reward += reward
            steps += 1
            emit_log(
                source="qlearn",
                level="debug",
                event="step",
                episode=ep,
                step=steps,
                state=state,
                action=action,
                reward=reward,
                done=done,
                observation=obs2,
            )

            state2 = discretize(obs2)

            # Q-update
            Q[state, action] += ALPHA * (
                reward + GAMMA * np.max(Q[state2]) - Q[state, action]
            )

            obs = obs2

        avg_reward = (total_reward / steps) if steps > 0 else 0.0
        episode_avg_rewards.append(avg_reward)

        writer.writerow([RUN_ID, ep, f'{total_reward:.4f}', f'{avg_reward:.4f}'])
        emit_log(
            source="qlearn",
            level="info",
            event="episode_summary",
            run_id=RUN_ID,
            episode=ep,
            steps=steps,
            total_reward=total_reward,
            avg_reward=avg_reward,
            epsilon=eps,
            flowmon_xml=flowmon_xml_path,
        )

        env.close()
        emit_log(source="qlearn", level="info", event="env_closed", episode=ep)
        try:
            ns3_proc.wait(timeout=10)
            emit_log(
                source="qlearn",
                level="info",
                event="ns3_exit",
                episode=ep,
                returncode=ns3_proc.returncode,
                pid=ns3_proc.pid,
            )
        except subprocess.TimeoutExpired:
            ns3_proc.kill()
            ns3_proc.wait(timeout=5)
            emit_log(
                source="qlearn",
                level="warning",
                event="ns3_killed_after_timeout",
                episode=ep,
                returncode=ns3_proc.returncode,
                pid=ns3_proc.pid,
            )
        ns3_log_thread.join(timeout=1)

# ---------------------------------------------------------------------------
# Save Q-table
# ---------------------------------------------------------------------------
np.save(args.out, Q)
emit_log(source="qlearn", level="info", event="q_table_saved",
         path=os.path.abspath(args.out), shape=list(Q.shape), dtype=str(Q.dtype))
emit_log(source="qlearn", level="info", event="training_complete",
         run_id=RUN_ID,
         episodes=args.episodes,
         avg_reward=float(np.mean(episode_avg_rewards[-10:])) if episode_avg_rewards else 0.0,
         rewards_csv=csv_path,
         q_table=os.path.abspath(args.out),
         alpha=ALPHA, gamma=GAMMA,
         eps_start=EPS_START, eps_end=EPS_END,
         n_states=N_STATES)

if LOG_FILE_HANDLE is not None:
    LOG_FILE_HANDLE.close()
