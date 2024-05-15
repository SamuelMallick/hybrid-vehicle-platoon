import pickle

import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcpwa.mpc.mpc_mld_cent_decup import MpcMldCentDecup
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
from scipy.linalg import block_diag

from env import PlatoonEnv
from fleet_cent_mld import MpcNonlinearGearCent
from misc.common_controller_params import Params, Sim
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Platoon
from mpcs.cent_mld import MpcMldCent
from mpcs.mpc_gear import MpcGear, MpcNonlinearGear

# from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet

np.random.seed(0)

sim = Sim()
n = sim.n  # num cars
N = sim.N  # controller horizon
ep_len = sim.ep_len  # length of episode (sim len)
ts = Params.ts
masses = sim.masses
spacing_policy = sim.spacing_policy
leader_trajectory = sim.leader_trajectory
leader_x = leader_trajectory.get_leader_trajectory()

platoon = Platoon(n, vehicle_type='nonlinear', masses=masses)
systems = platoon.get_vehicle_system_dicts(ts)

# env
env = MonitorEpisodes(
    TimeLimit(
        PlatoonEnv(
            n=n,
            platoon=platoon,
            leader_trajectory=leader_trajectory,
            spacing_policy=spacing_policy,
            start_from_platoon=sim.start_from_platoon,
            real_vehicle_as_reference=sim.real_vehicle_as_reference,
            ep_len=sim.ep_len,
            leader_index=0,
        ),
        max_episode_steps=ep_len,
    )
)

mpc = MpcNonlinearGearCent(
        n,
        N,
        systems,
        spacing_policy=spacing_policy,
        leader_index=0,
        thread_limit=1,
        real_vehicle_as_reference=sim.real_vehicle_as_reference,
    )

env.reset()
x0 = env.unwrapped.unwrapped.get_state()
u0, info = mpc.solve_mpc(x0)
x_pred = info['x']
u = np.vstack((info['u'], mpc.gears_pred))

x_real = np.zeros(x_pred.shape)
x_real[:, [0]] = x0
for k in range(u.shape[1]):
    x_real[:, [k]] = env.unwrapped.unwrapped.get_state()
    env.step(u[:, [k]])
x_real[:, [-1]] = env.unwrapped.unwrapped.get_state()
x_diff = x_real - x_pred
pass