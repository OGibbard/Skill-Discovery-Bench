import gymnasium as gym
from types import SimpleNamespace
import inspect
from rllab.misc.ext import AttrDict
from rllab.viskit.core import flatten
import os
from rllab import config
import base64
import pickle
import datetime
import dateutil.tz
import os.path as osp
import re
import subprocess
import sys
import random
import numpy as np
from rllab.misc.instrument import StubBase, VariantGenerator, run_experiment_lite, variant
from rllab.misc.ext import set_seed
from rllab.misc.console import colorize
from rllab.misc import logger
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from rllab.envs.env_spec import EnvSpec
from rllab.envs.observation_corruption_env import (
    ObservationCorruptionEnv,
    build_observation_corruption_config,
    save_robustness_manifest,
)
from rllab import spaces
from sac.replay_buffers.simple_replay_buffer import SimpleReplayBuffer
from sac.value_functions.value_function import NNQFunction, NNVFunction, NNDiscriminatorFunction
from sac.policies.gmm import GMMPolicy
from sac.algos import SkillDiscoverySAC

DEFAULT_LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "logs",
)

SHARED_PARAMS = {
    "seed": [1, 2, 3],
    "lr": 3E-4,
    "discount": 0.99,
    "tau": 0.01,
    "K": 4,
    "layer_size": 128,
    "batch_size": 128,
    "max_pool_size": 1E5,
    "n_train_repeat": 1,
    "epoch_length": 100,
    "snapshot_mode": 'gap',
    "snapshot_gap": 100,
    "sync_pkl": True,
    'num_skills': 50,
    'scale_entropy': 0.1,
    'include_actions': False,
    'learn_p_z': False,
    'add_p_z': True,
}


ENV_PARAMS = {
    'swimmer': { # 2 DoF
        'prefix': 'swimmer',
        'env_name': 'Swimmer-v4',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'scale_reward': 100,
    },
    'hopper': { # 3 DoF
        'prefix': 'hopper',
        'env_name': 'Hopper-v4',
        'max_path_length': 1000,
        'n_epochs': 3000,
        'scale_reward': 1,
    },
    'half-cheetah': { # 6 DoF
        'prefix': 'half-cheetah',
        'env_name': 'HalfCheetah-v4',
        'max_path_length': 1000,
        'n_epochs': 1000,
        'scale_reward': 1,
    },
    'walker': { # 6 DoF
        'prefix': 'walker',
        'env_name': 'Walker2d-v4',
        'max_path_length': 1000,
        'n_epochs': 5000,
        'scale_reward': 3,
    },
    'ant': { # 8 DoF
        'prefix': 'ant',
        'env_name': 'Ant-v4',
        'max_path_length': 1000,
        'n_epochs': 10000,
        'scale_reward': 3,
    },
    'humanoid': { # 21 DoF
        'prefix': 'humanoid',
        'env_name': 'humanoid-rllab',
        'max_path_length': 1000,
        'n_epochs': 20000,
        'scale_reward': 3,
    },
}

def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = dict(SHARED_PARAMS)
    params.update(env_params)
    params.update({
        'seed': args.seeds,
        'num_skills': args.num_skills,
        'skill_objective': args.skill_objective,
        'dads_reward_std': args.dads_reward_std,
        'dads_reward_scale': args.dads_reward_scale,
        'lsd_reward_scale': args.lsd_reward_scale,
        'lsd_norm_penalty': args.lsd_norm_penalty,
    })
    params.update({
        'obs_noise_std': args.obs_noise_std,
        'obs_delay_steps': args.obs_delay_steps,
        'obs_encoder_type': args.obs_encoder_type,
        'obs_encoder_dim': args.obs_encoder_dim,
        'obs_encoder_hidden_dim': args.obs_encoder_hidden_dim,
        'obs_corruption_seed': args.obs_corruption_seed,
    })

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg

def run_experiment(variant):
    set_seed(variant['seed'])
    env = GymEnv(variant['env_name'], record_video=False, record_log=False)
    corruption_seed = variant['obs_corruption_seed']
    if corruption_seed is None:
        corruption_seed = variant['seed']
    corruption_config = build_observation_corruption_config(
        noise_std=variant['obs_noise_std'],
        delay_steps=variant['obs_delay_steps'],
        encoder_type=variant['obs_encoder_type'],
        encoder_dim=variant['obs_encoder_dim'],
        encoder_hidden_dim=variant['obs_encoder_hidden_dim'],
        seed=corruption_seed,
    )
    if corruption_config.enabled():
        env = ObservationCorruptionEnv(env, corruption_config)
    env = normalize(env)

    snapshot_dir = logger.get_snapshot_dir()
    if snapshot_dir is not None:
        save_robustness_manifest(
            log_dir=snapshot_dir,
            method=variant['skill_objective'],
            environment=variant['env_name'],
            seed=variant['seed'],
            config=corruption_config,
            extra={
                'num_skills': variant['num_skills'],
                'exp_prefix': variant['prefix'],
            },
        )

    obs_space = env.spec.observation_space
    assert isinstance(obs_space, spaces.Box)
    # Augment observation space with skill vector
    low = np.hstack([obs_space.low, np.full(variant['num_skills'], 0)])
    high = np.hstack([obs_space.high, np.full(variant['num_skills'], 1)])
    aug_obs_space = spaces.Box(low=low, high=high)
    aug_env_spec = EnvSpec(aug_obs_space, env.spec.action_space)

    pool = SimpleReplayBuffer(
        env_spec=aug_env_spec,
        max_replay_buffer_size=variant['max_pool_size'],
    )

    base_kwargs = dict(
        min_pool_size=variant['max_path_length'],
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = variant['layer_size']
    qf = NNQFunction(
        env_spec=aug_env_spec,
        hidden_layer_sizes=[M, M],
    )

    vf = NNVFunction(
        env_spec=aug_env_spec,
        hidden_layer_sizes=[M, M],
    )

    policy = GMMPolicy(
        env_spec=aug_env_spec,
        K=variant['K'],
        hidden_layer_sizes=[M, M],
        qf=qf,
        reg=0.001,
    )

    discriminator = NNDiscriminatorFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
        num_skills=variant['num_skills'],
    )

    algorithm = SkillDiscoverySAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        discriminator=discriminator,
        pool=pool,
        qf=qf,
        vf=vf,

        lr=variant['lr'],
        scale_entropy=variant['scale_entropy'],
        discount=variant['discount'],
        tau=variant['tau'],
        num_skills=variant['num_skills'],
        save_full_state=False,
        include_actions=variant['include_actions'],
        learn_p_z=variant['learn_p_z'],
        add_p_z=variant['add_p_z'],
        skill_objective=variant['skill_objective'],
        aux_hidden_layer_sizes=[M, M],
        dads_reward_std=variant['dads_reward_std'],
        dads_reward_scale=variant['dads_reward_scale'],
        lsd_reward_scale=variant['lsd_reward_scale'],
        lsd_norm_penalty=variant['lsd_norm_penalty'],
        find_best_skill_interval=100,
        best_skill_n_rollouts=3,
    )

    algorithm.train()

def launch_experiments(variant_generator):
    variants = variant_generator.variants()
    print(variants)

    for i, variant in enumerate(variants):
        print('Launching {} experiments.'.format(len(variants)))
        objective_prefix = variant['prefix'] + '/' + variant['skill_objective']
        objective_name = (
            variant['prefix']
            + '-'
            + variant['skill_objective']
            + '-'
            + args.exp_name
            + '-'
            + str(i).zfill(2)
        )
        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix=objective_prefix + '/' + args.exp_name,
            exp_name=objective_name,
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=variant['sync_pkl'],
        )

def run_sac_experiment(main, mode, include_folders=None, log_dir=None,
                       exp_prefix="experiment", exp_name=None, **kwargs):
    if exp_name is None:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        exp_name = timestamp

    if log_dir is None:
        log_dir = os.path.join(
            DEFAULT_LOG_DIR,
            "local",
            exp_prefix.replace("_", "-"),
            exp_name)

    if include_folders is None:
        include_folders = list()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.abspath(os.path.join(project_dir, os.pardir))
    pythonpath = os.pathsep.join(
        [project_dir, code_dir]
        + include_folders
        + ([os.environ["PYTHONPATH"]] if "PYTHONPATH" in os.environ else [])
    )

    run_experiment_lite(
        stub_method_call=main,
        mode=mode,
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        log_dir=log_dir,
        python_command=sys.executable,
        env={"PYTHONPATH": pythonpath},
        **kwargs,
    )

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='half-cheetah')
    parser.add_argument('--exp_name', type=str, default='scaled_run')
    parser.add_argument(
        '--skill_objective',
        type=str,
        default='diayn',
        choices=['diayn', 'dads', 'lsd_delta'],
        help='Intrinsic objective to run on the shared skill-discovery SAC backbone.',
    )
    parser.add_argument('--num_skills', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--n_epochs_override', type=int, default=None)
    parser.add_argument('--max_path_length_override', type=int, default=None)
    parser.add_argument('--epoch_length_override', type=int, default=None)
    parser.add_argument('--dads_reward_std', type=float, default=1.0)
    parser.add_argument('--dads_reward_scale', type=float, default=1.0)
    parser.add_argument('--lsd_reward_scale', type=float, default=1.0)
    parser.add_argument('--lsd_norm_penalty', type=float, default=0.0)
    parser.add_argument('--obs_noise_std', type=float, default=0.0)
    parser.add_argument('--obs_delay_steps', type=int, default=0)
    parser.add_argument('--obs_encoder_type', type=str, default='identity', choices=['identity', 'linear', 'mlp'])
    parser.add_argument('--obs_encoder_dim', type=int, default=0)
    parser.add_argument('--obs_encoder_hidden_dim', type=int, default=0)
    parser.add_argument('--obs_corruption_seed', type=int, default=None)
    args = parser.parse_args()
    
    # Scale down hardcoded params for cluster
    SHARED_PARAMS['num_skills'] = args.num_skills
    SHARED_PARAMS['epoch_length'] = args.epoch_length_override or 100
    for env_key in ENV_PARAMS:
        ENV_PARAMS[env_key]['n_epochs'] = args.n_epochs_override or 500
        ENV_PARAMS[env_key]['max_path_length'] = args.max_path_length_override or 500

    args.mode = 'local'
    args.log_dir = None
    variant_generator = get_variants(args)
    launch_experiments(variant_generator)
