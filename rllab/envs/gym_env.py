import gymnasium as gym
import gymnasium.wrappers
import gymnasium.envs
import gymnasium.spaces
import traceback
import logging

# Monitor is gone in gymnasium, replaced by RecordVideo, but we need to check if we can easily map it.
# For now, let's just use gym modules.


import os
import os.path as osp
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.misc import logger


def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(x) for x in space.spaces])
    elif isinstance(space, gym.spaces.Dict):
         # Gymnasium has Dict spaces
        raise NotImplementedError("Dict spaces not fully supported in rllab conversion yet")
    else:
        raise NotImplementedError


class CappedCubicVideoSchedule(object):
    # Copied from gym, since this method is frequently moved around
    def __call__(self, count):
        if count < 1000:
            return int(round(count ** (1. / 3))) ** 3 == count
        else:
            return count % 1000 == 0


class FixedIntervalVideoSchedule(object):
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, count):
        return count % self.interval == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


class GymEnv(Env, Serializable):
    def __init__(self, env_name, record_video=True, video_schedule=None, log_dir=None, record_log=True,
                 force_reset=False, render_mode="rgb_array"):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        env = gym.make(env_name, render_mode=render_mode)
        self.env = env
        self.env_id = env.spec.id

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            
            # Gymnasium RecordVideo uses episode_trigger
            self.env = gym.wrappers.RecordVideo(self.env, log_dir, episode_trigger=video_schedule)
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))
        self._horizon = env.spec.max_episode_steps
        self._log_dir = log_dir
        self._force_reset = force_reset

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        if self._force_reset and self.monitoring:
            # Stats recorder is not directly accessible in the same way in RecordVideo
            pass
        obs, info = self.env.reset()
        return obs

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()

    def terminate(self):
        if self.monitoring:
            self.env.close()
            if self._log_dir is not None:
                print("""
    ***************************

    Training finished! You can upload results to OpenAI Gym by running the following command:

    python scripts/submit_gym.py %s

    ***************************
                """ % self._log_dir)
