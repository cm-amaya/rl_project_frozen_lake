import os
import gym
import pickle
from gym.envs.toy_text.frozen_lake import generate_random_map


class Enviroment:
    def __init__(self,
                 path: str = None,
                 size: int = 4,
                 render_mode: str = "human"):
        self._size = size
        self._render_mode = render_mode
        if path:
            self._map_path = path
        else:
            map_folder = 'saves/maps'
            if not os.path.exists(map_folder):
                os.mkdir(map_folder)
            self._map_path = os.path.join(map_folder, 'base.pkl')
        if os.path.exists(self._map_path):
            self.load_map()
        else:
            self.create_map()

    @property
    def env(self):
        return self._env

    @property
    def checkpoint(self):
        return self._checkpoint

    def save_map(self):
        with open(self._map_path, 'wb') as handle:
            pickle.dump(self._env, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_map(self):
        with open(self._map_path, 'rb') as handle:
            self._env = pickle.load(handle)

    def create_map(self):
        self._env = gym.make("FrozenLake-v1",
                             desc=generate_random_map(size=self._size),
                             render_mode=self._render_mode)
        self.save_map()
