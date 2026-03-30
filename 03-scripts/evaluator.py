import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config_loader import load_configs
from curriculum_builder import build_curriculum, get_target_spec
from agent import QLearningAgent
from trainer import make_env, evaluate_agent

seed = 42

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved policies from results folder without retraining."
    )
    parser.add_argument("--results-dir", required=True,
                        help="Path to experiment results dir")
    parser.add_argument("--env-config",   default="configs/environments-including-target-new.json")
    parser.add_argument("--learn-config", default="configs/learning-for-generated.json")
    parser.add_argument("--n-eval-episodes", type=int, default=3000)
    return parser.parse_args()


def load_agent(path: str, env, seed: int) -> QLearningAgent:
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        epsilon=0.0,
        seed=seed,
    )
    agent.load(path)
    return agent

def find_curriculum_stage_policies(stage_dir: str) -> list[tuple[int, str]]:
    results = []
    for env_dir in sorted(Path(stage_dir).iterdir()):
        parts = env_dir.name.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        idx = int(parts[1])
        npy_files = list(env_dir.glob("final_*.npy"))
        if npy_files:
            results.append((idx, str(npy_files[0])))
    return sorted(results, key=lambda x: x[0])

def find_baseline_policies(baseline_dir: str) -> list[tuple[str, str]]:
    results = []
    for b_dir in sorted(Path(baseline_dir).iterdir()):
        npy_files = list(b_dir.glob("final_*.npy"))
        if npy_files:
            results.append((b_dir.name, str(npy_files[0])))
    return results

# evaluate them for same episode, see their success.
def evaluate(seed_dir, target_spec, seed, n_eval_episodes):
    target_env = make_env(target_spec, seed=seed + 999)

    # curriculum
    curriculum_results = []
    for stage_idx, policy_path in find_curriculum_stage_policies(seed_dir):
        agent = load_agent(policy_path, target_env, seed)
        mean_reward, success_rate = evaluate_agent(agent, target_env, n_eval_episodes)
        curriculum_results.append({
            "stage": stage_idx,
            "label": f"E{stage_idx}",
            "mean_reward": mean_reward,
            "success_rate": success_rate,
        })
        print(f" Curriculum E{stage_idx}: reward={mean_reward:.3f}, success={success_rate:.3f}")

    # baselines
    baseline_results = []
    for label, policy_path in find_baseline_policies(seed_dir):
        agent = load_agent(policy_path, target_env, seed)
        mean_reward, success_rate = evaluate_agent(agent, target_env, n_eval_episodes)
        baseline_results.append({
            "label": label,
            "mean_reward": mean_reward,
            "success_rate": success_rate,
        })
        print(f" Baseline {label}: reward={mean_reward:.3f}, success={success_rate:.3f}")

    target_env.close()
    return curriculum_results, baseline_results


def make_bar_chart_success_rate(
    all_curriculum: dict[int, dict],
    all_baseline: list[dict],
    output_dir: str,
):
    cur_labels = [all_curriculum[s]["label"] for s in sorted(all_curriculum)]
    cur_ys = [all_curriculum[s]["success_rate"] for s in sorted(all_curriculum)]

    # if list as Curriculum (E5-E6) -> Curriculum (E4-E6)....
    # cur_labels = [all_curriculum[s]["label"] for s in sorted(all_curriculum, reverse=True)]
    # cur_ys = [all_curriculum[s]["success_rate"] for s in sorted(all_curriculum, reverse=True)]


    bas_labels = [r["label"] for r in all_baseline]
    bas_ys = [r["success_rate"] for r in all_baseline]

    x_labels = bas_labels + cur_labels
    y_vals = bas_ys + cur_ys
    colors = ["tab:orange"] * len(bas_labels) + ["tab:blue"] * len(cur_labels)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_labels, y_vals, color=colors)
    # ax.set_title("Success Rate on Target Environment", fontsize=14)
    # ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=10)
    # ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bar_success_rate.png"), dpi=300)
    plt.close(fig)



def make_bar_chart_mean_return(
    all_curriculum: dict[int, dict],
    all_baseline: list[dict],
    output_dir: str,
):
    cur_labels = [all_curriculum[s]["label"] for s in sorted(all_curriculum)]
    cur_ys = [all_curriculum[s]["mean_reward"] for s in sorted(all_curriculum)]

    # if list as Curriculum (E5-E6) -> Curriculum (E4-E6)....
    # cur_labels = [all_curriculum[s]["label"] for s in sorted(all_curriculum, reverse=True)]
    # cur_ys = [all_curriculum[s]["mean_reward"] for s in sorted(all_curriculum, reverse=True)]


    bas_labels = [r["label"] for r in all_baseline]
    bas_ys = [r["mean_reward"] for r in all_baseline]

    x_labels = bas_labels + cur_labels
    y_vals = bas_ys + cur_ys
    colors = ["tab:orange"] * len(bas_labels) + ["tab:blue"] * len(cur_labels)

    fig, ax = plt.subplots(figsize=(8, 4))
    bar_container = ax.bar(x_labels, y_vals, color=colors)
    ax.bar_label(bar_container, fmt='{:,.1f}')
    # ax.set_title("Average Episode Return on Target Environment", fontsize=14)
    # ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("Mean Cumulative Reward", fontsize=10)
    # ax.set_ylim(-2, 15)
    ax.tick_params(axis="x", rotation=15, labelsize=8)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bar_mean_reward.png"), dpi=300)
    plt.close(fig)


def evaluate_curricula(
    stage_dirs, target_env, seed, n_eval_episodes
) -> dict[int, dict]:
    all_curriculum = {}
    for stage_dir in stage_dirs:
        start_stage = int(stage_dir.name.split("_")[2])
        stage_policies = find_curriculum_stage_policies(str(stage_dir))
        if not stage_policies:
            continue

        last_idx, last_path = stage_policies[-1]
        n_stages_total = start_stage + len(stage_policies) - 1
        agent = load_agent(last_path, target_env, seed)
        mean_reward, success_rate = evaluate_agent(agent, target_env, n_eval_episodes)
        all_curriculum[start_stage] = {
            "label": f"Curriculum\n(E{start_stage}–E{n_stages_total})",
            "mean_reward": mean_reward,
            "success_rate": success_rate,
        }
        print(f" Curriculum E{start_stage}–E{n_stages_total}: reward={mean_reward:.3f}, success={success_rate:.3f}")

    return all_curriculum

def evaluate_baselines(
    results_dir, target_env, seed, n_eval_episodes
) -> list[dict]:
    all_baseline = []
    baseline_dir = str(Path(results_dir) / "baseline")
    for i, (_, policy_path) in enumerate(find_baseline_policies(baseline_dir), start=1):
        label = f"Baseline\n(E{i})"
        agent = load_agent(policy_path, target_env, seed)
        mean_reward, success_rate = evaluate_agent(agent, target_env, n_eval_episodes)
        all_baseline.append({
            "label": label,
            "mean_reward": mean_reward,
            "success_rate": success_rate,
        })
        print(f" Baseline {label}: reward={mean_reward:.3f}, success={success_rate:.3f}")
    return all_baseline




def main():
    args = parse_args()
    env_cfg, learn_cfg = load_configs(args.env_config, args.learn_config)

    curriculum_template = build_curriculum(env_cfg, learn_cfg)
    target_spec = get_target_spec(curriculum_template)

    stage_dirs = sorted(
        Path(args.results_dir).glob("start_stage_*"),
        key=lambda p: int(p.name.split("_")[2])
    )
    assert stage_dirs, f"No start_stage_* folders found in {args.results_dir}"

    target_env = make_env(target_spec, seed=seed + 999)

    all_curriculum = evaluate_curricula(stage_dirs, target_env, seed, args.n_eval_episodes)
    all_baseline = evaluate_baselines(args.results_dir, target_env, seed, args.n_eval_episodes)

    target_env.close()

    os.makedirs(args.results_dir, exist_ok=True)
    make_bar_chart_success_rate(all_curriculum, all_baseline, args.results_dir)
    make_bar_chart_mean_return (all_curriculum, all_baseline, args.results_dir)
    print("\nDone.")

if __name__ == "__main__":
    main()