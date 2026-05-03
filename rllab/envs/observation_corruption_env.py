"""Observation corruption wrapper for rllab environments."""

from __future__ import annotations

import os
import sys

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc.overrides import overrides
from rllab.spaces.box import Box


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from analysis.robustness_core import (  # noqa: E402
    ObservationCorruptor,
    ObservationCorruptionConfig,
    build_observation_corruption_config,
    save_robustness_manifest,
)


BIG = 1e6


class ObservationCorruptionEnv(ProxyEnv, Serializable):
    """Applies a shared perceived-observation pipeline to a rllab env."""

    def __init__(self, env, config: ObservationCorruptionConfig):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._config = config

        input_dim = env.observation_space.flat_dim
        self._corruptor = ObservationCorruptor(input_dim=input_dim, config=config)
        self._observation_space = Box(
            low=-BIG,
            high=BIG,
            shape=(self._corruptor.output_dim,),
        )

    @property
    def config(self):
        return self._config

    @property
    @overrides
    def observation_space(self):
        return self._observation_space

    @overrides
    def reset(self):
        obs = self._wrapped_env.reset()
        return self._corruptor.reset(obs)

    @overrides
    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        corrupted_obs = self._corruptor.step(next_obs)
        info = dict(info)
        info["robustness_clean_observation"] = np.asarray(next_obs)
        info["robustness_corrupted_observation"] = np.asarray(corrupted_obs)
        return Step(corrupted_obs, reward, done, **info)
