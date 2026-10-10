import numpy as np
import pickle
import ratinabox as riab
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset
from typing import Callable
from data import transform
__all__ = ["RatInABox_TwoSample",
           "RatInABox_Independence"]


class RatInABox_TwoSample(Dataset):
    r"""simulate the firing rates of place cells in two different environments (with and without walls)."""

    def __init__(self,
                 size: int = 5000,
                 n_cells: int = 10,
                 wall_height: float = 0.3,
                 wall_geometry: str = "geodesic",
                 cache_data: bool = True,
                 transform: Callable = transform.Identity(),
                 seed: int = 0,):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.transform = transform
        self.order_a = self.rng.permutation(size)
        self.order_b = self.rng.permutation(size)

        # use cached data if exists
        FNAME = f"twosample_n={size}_d={n_cells}_height={wall_height}_geometry={wall_geometry}.pkl"
        FPATH = Path("data/cache/riab") / FNAME
        if FPATH.is_file():
            with open(FPATH, 'rb') as f:
                self.history = pickle.load(f)
            return None

        NP_RAND_STATE = np.random.get_state()   # ratinabox library uses the global numpy random state, so we need to save and restore it
        np.random.seed(123456789)               # fix the global numpy random state for dataset reproducibility
        env_a = riab.Environment()  # default [0,1]^2 box
        env_b = riab.Environment()  # box with wall
        if wall_height > 0.0:
            env_b.add_wall(np.array([[0.5, 0], [0.5, wall_height/2.]]))
            env_b.add_wall(np.array([[0.5, 1], [0.5, 1-wall_height/2.]]))
        cell_centres = env_a.sample_positions(n=n_cells, method="uniform_jitter")

        # simulate one trajectory per sample
        history_a = []
        history_b = []
        for i in tqdm(range(size)):
            # initialize agents in environment A and B
            agent_a = riab.Agent(env_a, params={'dt': 0.1,})
            agent_b = riab.Agent(env_b, params={'dt': 0.1,})
            # place cells
            defaults = {'n': n_cells,
                        'widths': 0.2,
                        'place_cell_centres': cell_centres,
                        'wall_geometry': wall_geometry,
                        'noise_std': 0.0,
                        'noise_coherence_time': 0.5,}
            pcells_a = riab.PlaceCells(agent_a, defaults)
            pcells_b = riab.PlaceCells(agent_b, defaults)
            # simulate 2s of trajectory
            for t in range(20):
                agent_a.update()
                agent_b.update()
                pcells_a.update()
                pcells_b.update()
            history_a.append(pcells_a.history["firingrate"][-1])
            history_b.append(pcells_b.history["firingrate"][-1])

        self.history = dict()
        self.history["pcells_a"] = np.array(history_a)  # (size, n_cells)
        self.history["pcells_b"] = np.array(history_b)  # (size, n_cells)
        np.random.set_state(NP_RAND_STATE)  # restore the global numpy random state

        if cache_data:
            FPATH.parent.mkdir(parents=True, exist_ok=True)
            with open(FPATH, 'wb') as f:
                pickle.dump(self.history, f)


    def __len__(self):
        return len(self.history["pcells_a"])

    def __getitem__(self, idx):
        i = self.order_a[idx]
        j = self.order_b[idx]
        x = self.history["pcells_a"][i]
        y = self.history["pcells_b"][j]
        return self.transform((x, y))



class RatInABox_Independence(Dataset):

    def __init__(self,
                 size: int = 5000,
                 n_pcells: int = 10,
                 n_bvcells: int = 10,
                 wall_height: float = 0.3,
                 wall_geometry: str = "geodesic",
                 cache_data: bool = True,
                 transform: Callable = transform.Identity(),
                 seed: int = 0,):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.transform = transform
        self.order = self.rng.permutation(size)

        env = riab.Environment()  # default [0,1]^2 box
        if wall_height > 0.0:
            env.add_wall(np.array([[0.5, 0], [0.5, wall_height/2.]]))
            env.add_wall(np.array([[0.5, 1], [0.5, 1-wall_height/2.]]))

        # use cached data if exists
        FNAME = f"independence_n={size}_dx={n_pcells}_dy={n_bvcells}_height={wall_height}_geometry={wall_geometry}.pkl"
        FPATH = Path("data/cache/riab") / FNAME
        if FPATH.is_file():
            with open(FPATH, 'rb') as f:
                self.history = pickle.load(f)
            return None

        NP_RAND_STATE = np.random.get_state()   # ratinabox library uses the global numpy random state, so we need to save and restore it
        np.random.seed(123456789)               # fix the global numpy random state for dataset reproducibility (max global seed is 2**32 - 1)

        # simulate one trajectory per sample
        self.history = []
        for i in tqdm(range(size)):
            agent = riab.Agent(env, params={'dt': 0.1,})
            pc_params = {'n': n_pcells,
                        'widths': 0.2,
                        'wall_geometry': wall_geometry,
                        'noise_std': 0.0,
                        'noise_coherence_time': 0.5,}
            bvc_params = {'n': n_bvcells,
                          'dtheta': 2,}
            pcells = riab.PlaceCells(agent, pc_params)
            bvcells = riab.BoundaryVectorCells(agent, bvc_params)

            # simulate 2s of trajectory
            for t in range(20):
                agent.update()
                pcells.update()
                bvcells.update()
            x = np.array(pcells.history["firingrate"][-1])
            y = np.array(bvcells.history["firingrate"][-1])
            self.history.append((x,y))

        np.random.set_state(NP_RAND_STATE)  # restore the global numpy random state
        if cache_data:
            FPATH.parent.mkdir(parents=True, exist_ok=True)
            with open(FPATH, 'wb') as f:
                pickle.dump(self.history, f)

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        i = self.order[idx]
        x, y = self.history[i]
        return self.transform((x, y))





def main():
    RatInABox_TwoSample(size=2000, wall_geometry='line_of_sight')
    RatInABox_Independence(size=5000, wall_geometry='line_of_sight')

if __name__ == "__main__":
    main()






