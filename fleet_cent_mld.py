import pickle
import sys

import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcpwa.mpc.mpc_mld_cent_decup import MpcMldCentDecup
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
from scipy.linalg import block_diag

from env import PlatoonEnv
from mpcs.cent_mld import MpcMldCent
# from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet
from misc.leader_trajectory import ConstantVelocityLeaderTrajectory
from models import Platoon

from misc.common_controller_params import Params

np.random.seed(2)

PLOT = False
SAVE = True

n = 3  # num cars
N = 5  # controller horizon
ep_len = 10  # length of episode (sim len)
ts = Params.ts

COST_2_NORM = True
random_ICs = False

leader_trajectory = ConstantVelocityLeaderTrajectory(
    p=1000, v=20, trajectory_len=ep_len + 50, ts=1
)
leader_x = leader_trajectory.get_leader_trajectory()

# class MpcGearCent(MPCMldCent, MpcMldCentDecup, MpcGear):
#     def __init__(self, systems: list[dict], n: int, N: int) -> None:
#         self.n = n
#         MpcMldCentDecup.__init__(self, systems, n, N)  # use the MpcMld constructor
#         F = block_diag(
#             *([systems[0]["F"]] * n)
#         )  # here we are assuming that the F and G are the same for all systems
#         G = np.vstack([systems[0]["G"]] * n)
#         self.setup_gears(N, acc, F, G)
#         self.setup_cost_and_constraints(self.u_g, acc, COST_2_NORM)


class TrackingCentralizedAgent(MldAgent):
    def __init__(self, mpc: MpcMld) -> None:
        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))
        self.bin_var_counts = np.zeros((ep_len, 1))
        super().__init__(mpc)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.mpc.set_leader_traj(leader_x[:, timestep : (timestep + N + 1)])
        self.solve_times[env.step_counter - 1, :] = self.run_time
        self.node_counts[env.step_counter - 1, :] = self.node_count
        self.bin_var_counts[env.step_counter - 1, :] = self.num_bin_vars
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int, state) -> None:
        self.mpc.set_leader_traj(leader_x[:, 0 : N + 1])
        return super().on_episode_start(env, episode, state)


# vehicles
platoon = Platoon(n, vehicle_type="pwa_gear")
systems = platoon.get_vehicle_system_dicts(ts)

# env
env = MonitorEpisodes(
    TimeLimit(
        PlatoonEnv(n=n, platoon=platoon, leader_trajectory=leader_trajectory),
        max_episode_steps=ep_len,
    )
)

# mpcs
mpc = MpcMldCent(n, N, systems)

# agent
agent = TrackingCentralizedAgent(mpc)

# if DISCRETE_GEARS:
#     if HOMOGENOUS:  # by not passing the index all systems are the same
#         systems = [acc.get_friction_pwa_system() for i in range(n)]
#     else:
#         systems = [acc.get_friction_pwa_system(i) for i in range(n)]
#     mpc = MpcGearCent(systems, n, N)
#     mpc.set_leader_traj(leader_state[:, 0 : N + 1])
#     agent = TrackingMldAgent(mpc)
# else:
#     # mld mpc
#     if HOMOGENOUS:  # by not passing the index all systems are the same
#         systems = [acc.get_pwa_system() for i in range(n)]
#     else:
#         systems = [acc.get_pwa_system(i) for i in range(n)]
#     mld_mpc = MPCMldCent(systems, acc, COST_2_NORM, n, N)
#     # initialise the cost with the first tracking point
#     mld_mpc.set_leader_traj(leader_state[:, 0 : N + 1])
#     agent = TrackingMldAgent(mld_mpc)

agent.evaluate(env=env, episodes=1, seed=1)

if len(env.observations) > 0:
    X = env.observations[0].squeeze()
    U = env.actions[0].squeeze()
    R = env.rewards[0]
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

print(f"Return = {sum(R.squeeze())}")
print(f"Violations = {env.unwrapped.viol_counter}")
print(f"Run_times_sum: {sum(agent.solve_times)}")
print(f"average_bin_vars: {sum(agent.bin_var_counts)/len(agent.bin_var_counts)}")

if PLOT:
    plot_fleet(n, X, U, R, leader_state, violations=env.unwrapped.viol_counter[0])

if SAVE:
    with open(
        f"cent_n_{n}_N_{N}_Q_{COST_2_NORM}_DG_{DISCRETE_GEARS}_HOM_{HOMOGENOUS}_LT_{LEADER_TRAJ}"
        # + datetime.datetime.now().strftime("%d%H%M%S%f")
        + ".pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
        pickle.dump(agent.solve_times, file)
        pickle.dump(agent.node_counts, file)
        pickle.dump(env.unwrapped.viol_counter[0], file)
        pickle.dump(leader_state, file)
