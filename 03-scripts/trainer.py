"""
Training and evaluation pipeline for curriculum learning experiments.
"""

import os
import numpy as np
from dataclasses import dataclass, field

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import env
import gymnasium as gym
from config_loader import EnvSpec
from agent import QLearningAgent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    """Metrics collected during training on one environment."""
    env_id: str
    # Per-episode metrics (collected via callback)
    timesteps: list[int] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    # Periodic eval metrics (collected at checkpoint_frequency)
    eval_timesteps: list[int] = field(default_factory=list)
    eval_mean_rewards: list[float] = field(default_factory=list)
    eval_success_rates: list[float] = field(default_factory=list)
    # Saved agent paths
    checkpoint_paths: list[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Make environment
# ---------------------------------------------------------------------------
def make_env(spec: EnvSpec, seed: int | None = None):
    """Create a BurningForest."""
    env = gym.make(
        "BurningForest-v0",
        desc=spec.desc,
        reward_schedule=spec.reward_schedule,
        # is_slippery=False,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env

# ---------------------------------------------------------------------------
# Create DQN agent
# ---------------------------------------------------------------------------
def create_agent(rl_cfg, env, seed):
    return QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        lr=rl_cfg["learning_rate"],
        gamma=rl_cfg["discount_factor"],
        epsilon=rl_cfg["epsilon_start"],
        epsilon_decay=rl_cfg["epsilon_decay"],
        epsilon_min=rl_cfg["epsilon_min"],
        seed=seed,
    )

# ---------------------------------------------------------------------------
# Evaluate the trained agent
# ---------------------------------------------------------------------------
def evaluate_agent(agent, env, n_episodes: int = 100) -> tuple[float, float]:
    """
    Evaluate a trained agent on an environment.
    Returns:
        (mean_reward, success_rate)
    """
    total_rewards = []
    successes = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        if terminated and env.unwrapped.desc.flat[env.unwrapped.s] == b"G":
            successes += 1

    mean_reward = float(np.mean(total_rewards))
    success_rate = successes / n_episodes
    return mean_reward, success_rate


# ---------------------------------------------------------------------------
# Train on a single environment
# ---------------------------------------------------------------------------
def train_on_env(
    agent: QLearningAgent,
    env,
    eval_env,
    steps: int,
    env_id: str,
    checkpoint_dir: str,
    checkpoint_freq: int = 2000,
    eval_episodes: int = 20,
    global_step_offset: int = 0,
    verbose: int = 1,
) -> TrainResult:
    result = TrainResult(env_id=env_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0

    for step in range(1, steps + 1):
        action = agent.predict(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, next_obs, terminated)

        episode_reward += reward
        episode_length += 1
        done = terminated or truncated

        if done:
            global_ts = step + global_step_offset
            result.timesteps.append(global_ts)
            result.episode_rewards.append(episode_reward)
            result.episode_lengths.append(episode_length)
            agent.decay_epsilon()
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
        else:
            obs = next_obs

        # Periodic checkpoint + eval <- remove checkpoint
        if step % checkpoint_freq == 0:
            global_ts = step + global_step_offset

            # ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{global_ts}.npy")
            # agent.save(ckpt_path)
            # result.checkpoint_paths.append(ckpt_path)

            mean_reward, success_rate = evaluate_agent(agent, eval_env, eval_episodes)
            result.eval_timesteps.append(global_ts)
            result.eval_mean_rewards.append(mean_reward)
            result.eval_success_rates.append(success_rate)

            if verbose:
                print(f"[step {global_ts}] eval: mean_reward={mean_reward:.3f}, "
                      f"success_rate={success_rate:.3f}, epsilon={agent.epsilon:.3f}")

    # Save final
    final_path = os.path.join(checkpoint_dir, f"final_{env_id}.npy")
    agent.save(final_path)
    result.checkpoint_paths.append(final_path)

    return result



# ---------------------------------------------------------------------------
# Curriculum training: loop over envs
# ---------------------------------------------------------------------------
def run_curriculum(curriculum, eval_spec, learn_cfg, output_dir):
    rl_cfg = learn_cfg["rl"]
    seed = learn_cfg["curriculum"].get("seed", 42)
    checkpoint_freq = learn_cfg["curriculum"].get("checkpoint_frequency", 2000)

    first_env = make_env(curriculum[0]["spec"], seed=seed)
    agent = create_agent(rl_cfg, first_env, seed=seed)

    all_results = []
    global_steps = 0

    for i, environment in enumerate(curriculum):
        spec = environment["spec"]
        steps = environment["steps"]
        print(f"\n=== Curriculum Environment {i+1}/{len(curriculum)}: {spec.env_id} "
              f"(complexity={spec.complexity}, steps={steps}) ===")

        # Reset exploration for new environment
        agent.epsilon = rl_cfg.get("epsilon_start", 1.0)

        env = make_env(spec, seed=seed + i)
        eval_env = make_env(eval_spec, seed=seed + 1000)

        result = train_on_env(
            agent=agent, env=env, eval_env=env,
            steps=steps, env_id=spec.env_id,
            checkpoint_dir=os.path.join(output_dir, spec.env_id),
            checkpoint_freq=checkpoint_freq,
            global_step_offset=global_steps,
        )

        all_results.append(result)
        global_steps += steps

    return agent, all_results


# ---------------------------------------------------------------------------
# Baseline training: single env, same total steps as CL
# ---------------------------------------------------------------------------
def run_baseline(
    target_spec: EnvSpec,
    total_steps: int,
    learn_cfg: dict,
    output_dir: str,
):
    """Train baseline: single environment, same total steps as curriculum."""
    rl_cfg = learn_cfg["rl"]
    seed = learn_cfg["curriculum"].get("seed", 42)
    checkpoint_freq = learn_cfg["curriculum"].get("checkpoint_frequency", 2000)

    env = make_env(target_spec, seed=seed)
    eval_env = make_env(target_spec, seed=seed + 1000)
    agent = create_agent(rl_cfg, env, seed=seed)

    print(f"\n=== Baseline Training: {target_spec.env_id} "
          f"(steps={total_steps}) ===")

    result = train_on_env(
        agent=agent,
        env=env,
        eval_env=eval_env,
        steps=total_steps,
        env_id=target_spec.env_id,
        checkpoint_dir=os.path.join(output_dir),
        checkpoint_freq=checkpoint_freq,
    )

    return agent, result




# ---------------------------------------------------------------------------
# Final evaluation: curriculum vs baseline on target env
# ---------------------------------------------------------------------------
def run_final_evaluation(
    curriculum_agent: DQN,
    baseline_agent: DQN | None,
    eval_spec: EnvSpec,
    n_episodes: int,
    seed: int = 123,
) -> dict:
    """Evaluate curriculum and baseline agents on the target environment."""
    eval_env = make_env(eval_spec, seed=seed)

    print(f"\n=== Final Evaluation ({n_episodes} episodes) ===")
    results = {}

    curriculum_reward, curriculum_success = evaluate_agent(curriculum_agent, eval_env, n_episodes)
    results["curriculum"] = {
        "mean_reward": curriculum_reward,
        "success_rate": curriculum_success,
    }
    print(f"Curriculum: mean_reward={curriculum_reward:.3f}, success_rate={curriculum_success:.3f}")

    if baseline_agent is not None:
        base_reward, base_success = evaluate_agent(baseline_agent, eval_env, n_episodes)
        results["baseline"] = {
            "mean_reward": base_reward,
            "success_rate": base_success,
        }
        print(f"Baseline: mean_reward={base_reward:.3f}, success_rate={base_success:.3f}")

    return results