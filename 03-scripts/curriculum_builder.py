from config_loader import EnvSpec
def build_curriculum(env_cfg: dict, learn_cfg: dict) -> list[dict]:
    """
    Build curriculum artifact from configs.
    Returns a list of dicts ordered by complexity from easy to hard.
        [
            {"spec": EnvSpec, "steps": int},
            ...
        ]
    The last one (highest complexity) will be the target & baseline environment.
    """
    steps_per_environment = learn_cfg["curriculum"]["steps_per_environment"]
    specs = []
    for env in env_cfg["environments"]:
        spec = EnvSpec(
            env_id=env["env_id"],
            desc=env["desc"],
            reward_schedule=tuple(env["reward_schedule"]),
            complexity=env["complexity"],
        )
        specs.append(spec)

    # Sort by complexity from easy to hard
    specs.sort(key=lambda s: s.complexity)

    # TODO: curriculum builder, based on these ordered population, try to sampling the expected size of environments

    # Validate actual size matches config size
    expected_size = learn_cfg["curriculum"]["size"]
    if len(specs) != expected_size:
        raise ValueError(
            f"curriculum.size={expected_size} but found {len(specs)} environments"
        )

    # Build curriculum artifact
    curriculum = [{"spec": spec, "steps": steps_per_environment} for spec in specs]
    return curriculum


def get_target_spec(curriculum: list[dict]) -> EnvSpec:
    return curriculum[-1]["spec"]


def resolve_baseline_specs(
    curriculum: list[dict],
    env_indices: list[int] | str | None = None,
) -> dict[str, EnvSpec]:
    if env_indices is None:
        target = curriculum[-1]["spec"]
        return {"Baseline (target)": target}

    indices = range(len(curriculum)) if env_indices == "all" else env_indices

    return {
        f"Baseline (complexity={curriculum[i]['spec'].complexity:.2f})": curriculum[i]["spec"]
        for i in indices
    }
