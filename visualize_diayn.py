"""Visualise a saved skill-conditioned policy checkpoint."""

import argparse
import os
import random
import sys

import joblib
import tensorflow.compat.v1 as tf

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import NormalizedEnv, normalize
from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout
from sac.policies.hierarchical_policy import FixedOptionPolicy

tf.disable_v2_behavior()


def _rebuild_render_env(env):
    """Return an equivalent GymEnv with human rendering when possible."""

    is_normalized = isinstance(env, NormalizedEnv)
    inner_env = env._wrapped_env if is_normalized else env
    if not isinstance(inner_env, GymEnv):
        return env, False

    env_name = inner_env.env.spec.id
    render_env = GymEnv(
        env_name,
        record_video=False,
        record_log=False,
        render_mode="human",
    )
    return normalize(render_env) if is_normalized else render_env, True


def _infer_num_skills(policy, env):
    env_obs_dim = env.spec.observation_space.flat_dim
    if hasattr(policy, "_Ds"):
        policy_obs_dim = policy._Ds
    else:
        policy_obs_dim = policy.env_spec.observation_space.flat_dim
    return max(0, policy_obs_dim - env_obs_dim)


def _choose_skill(requested_skill, num_skills):
    if requested_skill == "random":
        return random.randint(0, num_skills - 1)
    if requested_skill is not None:
        skill = int(requested_skill)
        if not 0 <= skill < num_skills:
            raise ValueError("skill must be between 0 and %d" % (num_skills - 1))
        return skill

    while True:
        raw_skill = input(
            "Enter skill ID (0-%d), 'r' for random, or 'q' to quit: "
            % (num_skills - 1)
        ).strip()
        if raw_skill.lower() == "q":
            return None
        if raw_skill.lower() == "r":
            return random.randint(0, num_skills - 1)
        try:
            skill = int(raw_skill)
        except ValueError:
            print("Invalid skill.")
            continue
        if 0 <= skill < num_skills:
            return skill
        print("Skill must be between 0 and %d." % (num_skills - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="Path to an itr_*.pkl checkpoint.")
    parser.add_argument(
        "--skill",
        default=None,
        help="Skill id to visualise, 'random', or omit for interactive mode.",
    )
    parser.add_argument("--max-path-length", type=int, default=1000)
    parser.add_argument("--speedup", type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists(args.snapshot):
        print("Snapshot file not found: %s" % args.snapshot, file=sys.stderr)
        return 1

    print("Loading policy from %s" % args.snapshot)
    with tf.Session():
        data = joblib.load(args.snapshot)
        policy = data["policy"]
        env = data["env"]

        env, rebuilt = _rebuild_render_env(env)
        if not rebuilt:
            print("Warning: could not rebuild a GymEnv with render_mode='human'.")

        num_skills = _infer_num_skills(policy, env)
        if num_skills:
            print("Detected skill-conditioned policy with %d skills." % num_skills)

        while True:
            current_policy = policy
            if num_skills:
                skill = _choose_skill(args.skill, num_skills)
                if skill is None:
                    break
                print("Visualising skill %d." % skill)
                current_policy = FixedOptionPolicy(policy, num_skills, skill)

            rollout(
                env,
                current_policy,
                max_path_length=args.max_path_length,
                animated=True,
                speedup=args.speedup,
            )

            if args.skill is not None or not query_yes_no("Continue simulation?"):
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
