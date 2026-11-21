"""Recollect states and rerun sampled physical test branches from saved seeds."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .world_model_probes import collect_states, load_model
from .tdmpc2_retention import SavedState, _true_responses
from .tdmpc2_sequential_rules import sign_maps

ROOT = Path("experiments/tdmpc2_sequential_20261116")


def main():
    import gymnasium as gym
    import torch
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale
    torch.set_num_threads(2)
    if not hasattr(gym.logger, "set_level"):
        gym.logger.set_level = lambda level: None
    protocol = json.loads((ROOT / "protocol.json").read_text())
    checks = []
    for task in protocol["tasks"]:
        for seed in protocol["confirmation_seeds"]:
            path = ROOT / "results" / "confirmation" / f"{task}_{seed}"
            meta = json.loads(path.with_suffix(".json").read_text())
            env = action_scale.Wrapper(suite.load(*task.split("-", 1), task_kwargs={"random": meta["environment_seed"]}), minimum=-1, maximum=1)
            model, _ = load_model(Path("external/tdmpc2"),
                                  Path(f"external/tdmpc2/checkpoints/revision_single_task/{task}-{seed}.pt"), task, env, "cuda:0")
            rng = np.random.default_rng(meta["acquisition_seed"])
            states, reward = collect_states(model, env, sum(protocol["state_counts"].values()), rng, "cuda:0")
            rng.shuffle(states)
            observed_hashes = [hashlib.sha256(s.physics_state.tobytes()).hexdigest() for s in states]
            expected_hashes = [x for name in protocol["state_counts"] for x in meta["physical_state_hashes"][name]]
            assert observed_hashes == expected_hashes
            with np.load(path.with_suffix(".npz")) as data:
                indices = np.linspace(0, protocol["test_queries"] - 1, 6, dtype=int)
                queries = [{"state": SavedState(data["test__observations"][i], data["test__physics_states"][i]),
                            "direction": data["test__directions"][i], "horizon": int(data["test__horizons"][i])} for i in indices]
                maps = sign_maps(meta["action_dim"])
                map_ids = np.unique(np.linspace(0, len(maps) - 1, 4, dtype=int))
                maximum_error = 0.
                for code in map_ids:
                    actual = _true_responses(env, queries, protocol["pulse"], maps[code])
                    expected = data["true_test_bank"][code, indices]
                    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
                    maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
            checks.append({"task": task, "seed": seed, "physical_state_hashes_reproduced": len(states),
                           "test_pairs_reexecuted": len(map_ids) * len(indices), "maximum_absolute_response_difference": maximum_error})
            print(json.dumps(checks[-1]), flush=True)
            env.close()
            del model
            torch.cuda.empty_cache()
    (ROOT / "physics_verification.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
