import os
import numpy as np
import matplotlib.pyplot as plt

from trainer import TrainResult
#
CURRICULUM_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
BASELINE_COLORS   = ["tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "tab:blue"]

# CURRICULUM_COLORS = [
#     "tab:blue",      # 0
#     "tab:orange",    # 1
#     "tab:green",     # 2
#     "tab:red",       # 3
#     "tab:purple",    # 4
#     "tab:brown",     # 5
#     "tab:pink",      # 6
#     "tab:gray",      # 7
#     "tab:olive",     # 8
#     "tab:cyan"       # 9
# ]
#
# BASELINE_COLORS = [
#     "darkred",       # 0
#     "darkblue",      # 1
#     "darkgreen",     # 2
#     "gold",          # 3
#     "indigo",        # 4
#     "sienna",        # 5
#     "navy",          # 6
#     "khaki",         # 7
#     "salmon",        # 8
#     "teal"           # 9
# ]

def _curriculum_color(start_stage: int) -> str:
    return CURRICULUM_COLORS[(start_stage - 1) % len(CURRICULUM_COLORS)]

def _baseline_color(label: str, all_labels: list[str]) -> str:
    idx = all_labels.index(label)
    return BASELINE_COLORS[idx % len(BASELINE_COLORS)]

def make_training_cumulative_reward_plot_each(
    curriculum_results_by_stage: dict[int, list],
    baseline_results: dict[str, TrainResult],
    baseline_specs_by_stage: dict[int, dict],
    output_dir: str,
):
    all_baseline_labels = list(baseline_results.keys())

    for start_stage, cur_results in curriculum_results_by_stage.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        n_stages_total = start_stage + len(cur_results) - 1
        x_cur = np.array([ts for r in cur_results for ts in r.timesteps], dtype=float)
        y_cur = np.cumsum([rew for r in cur_results for rew in r.episode_rewards])
        ax.plot(x_cur, y_cur,
                label=f"Curriculum (E{start_stage}–E{n_stages_total})",
                color=_curriculum_color(start_stage),
                linewidth=1.5)

        x_max = x_cur[-1]
        for label in baseline_specs_by_stage[start_stage].keys():
            result = baseline_results[label]
            x_bas = np.array(result.timesteps, dtype=float)
            y_bas = np.cumsum(result.episode_rewards)
            mask = x_bas <= x_max
            ax.plot(x_bas[mask], y_bas[mask],
                    label=label,
                    color=_baseline_color(label, all_baseline_labels),
                    linewidth=1.5, linestyle="dashed")

        ax.set_xlim(0, x_max * 1.02)
        # ax.set_title(f"Cumulative Reward of Curriculum E{start_stage}–E{n_stages_total} v.s. Baselines", fontsize=11)
        ax.set_xlabel("Timesteps", fontsize=12)
        ax.set_ylabel("Cumulative Reward", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"cumulative_reward_start_{start_stage}.png"), dpi=300)
        plt.close(fig)


def make_training_cumulative_reward_plot_all(
    curriculum_results_by_stage: dict[int, list],
    baseline_results: dict[str, TrainResult],
    output_dir: str,
):
    all_baseline_labels = list(baseline_results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    x_max = 0

    for start_stage, cur_results in curriculum_results_by_stage.items():
        n_stages_total = start_stage + len(cur_results) - 1
        x = np.array([ts for r in cur_results for ts in r.timesteps], dtype=float)
        y = np.cumsum([rew for r in cur_results for rew in r.episode_rewards])
        ax.plot(x, y,
                label=f"Curriculum (E{start_stage}–E{n_stages_total})",
                color=_curriculum_color(start_stage),
                linewidth=1.5)
        x_max = max(x_max, x[-1])

    for label, result in baseline_results.items():
        x = np.array(result.timesteps, dtype=float)
        y = np.cumsum(result.episode_rewards)
        mask = x <= x_max
        ax.plot(x[mask], y[mask],
                label=label,
                color=_baseline_color(label, all_baseline_labels),
                linewidth=1.5, linestyle="dashed")

    # ax.set_title("Cumulative Reward of All Curricula v.s. Baselines", fontsize=11)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cumulative_reward_all.png"), dpi=300)
    plt.close(fig)




