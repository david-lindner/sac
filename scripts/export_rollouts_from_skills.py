import os
import joblib
import pickle
import argparse

import numpy as np
import tensorflow as tf

from sac.policies.hierarchical_policy import FixedOptionPolicy
from sac.misc.sampler import rollouts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the snapshot file.")
    parser.add_argument("--max-path-length", "-l", type=int, default=100)
    parser.add_argument(
        "--deterministic", "-d", dest="deterministic", action="store_true"
    )
    parser.add_argument(
        "--no-deterministic", "-nd", dest="deterministic", action="store_false"
    )
    parser.add_argument("--n-paths", "-np", type=int, default=1)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()
    rollouts_filename = worst_filename = (
        os.path.splitext(args.file)[0] + "_rollouts.pkl"
    )

    observations = []
    actions = []

    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data["policy"]
        env = data["env"]
        num_skills = (
            data["policy"].observation_space.flat_dim
            - data["env"].spec.observation_space.flat_dim
        )

        with policy.deterministic(args.deterministic):
            for z in range(num_skills):
                fixed_z_policy = FixedOptionPolicy(policy, num_skills, z)
                new_paths = rollouts(
                    env,
                    fixed_z_policy,
                    args.max_path_length,
                    n_paths=args.n_paths,
                    render=True,
                    render_mode="rgb_array",
                )
                for path in new_paths:
                    observations.append(path["observations"])
                    actions.append(path["actions"])

        with open(rollouts_filename, "wb") as f:
            data = {"observations": observations, "actions": actions}
            pickle.dump(data, f)
