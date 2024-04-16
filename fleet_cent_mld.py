import pickle

import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes

from env import PlatoonEnv
from misc.common_controller_params import Params, Sim
from misc.leader_trajectory import ConstantVelocityLeaderTrajectory
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Platoon
from mpcs.cent_mld import MpcMldCent
from dmpcpwa.mpc.mpc_mld_cent_decup import MpcMldCentDecup
from mpcs.mpc_gear import MpcGear
from scipy.linalg import block_diag

# from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet

np.random.seed(2)

PLOT = True
SAVE = False

n = 3  # num cars
N = 5  # controller horizon
ep_len = 50  # length of episode (sim len)
ts = Params.ts


spacing_policy = ConstantSpacingPolicy(50)
leader_trajectory = ConstantVelocityLeaderTrajectory(
    p=3000, v=20, trajectory_len=ep_len + 50, ts=ts
)
leader_x = leader_trajectory.get_leader_trajectory()


class MpcGearCent(MpcMldCent, MpcMldCentDecup, MpcGear):
    def __init__(
        self,
        n: int,
        N: int,
        systems: list[dict],
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        quadratic_cost: bool = True,
    ) -> None:
        self.n = n
        MpcMldCentDecup.__init__(self, systems, n, N)  # use the MpcMld constructor
        F = block_diag(*[systems[i]["F"] for i in range(n)])
        G = np.vstack([systems[i]["G"] for i in range(n)])
        self.setup_gears(N, F, G)
        self.setup_cost_and_constraints(self.u_g, spacing_policy, quadratic_cost)


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
platoon = Platoon(n, vehicle_type=Sim.vehicle_model_type)
systems = platoon.get_vehicle_system_dicts(ts)

# env
env = MonitorEpisodes(
    TimeLimit(
        PlatoonEnv(
            n=n,
            platoon=platoon,
            leader_trajectory=leader_trajectory,
            spacing_policy=spacing_policy,
            start_from_platoon=Sim.start_from_platoon,
        ),
        max_episode_steps=ep_len,
    )
)

# mpcs
if Sim.vehicle_model_type == "pwa_gear":
    mpc = MpcMldCent(n, N, systems, spacing_policy=spacing_policy)
elif Sim.vehicle_model_type == "pwa_friction":
    mpc = MpcGearCent(n, N, systems, spacing_policy=spacing_policy)
elif Sim.vehicle_model_type == "nonlinear":
    raise NotImplementedError()
else:
    raise ValueError(f"{Sim.vehicle_model_type} is not a valid vehicle model type.")


# agent
agent = TrackingCentralizedAgent(mpc)

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
    plot_fleet(n, X, U, R, leader_x, violations=env.unwrapped.viol_counter[0])

if SAVE:
    with open(
        f"cent_n_{n}_N_{N}_Q_{COST_2_NORM}_DG_{DISCRETE_GEARS}_HOM_{HOMOGENOUS}_LT_{LEADER_TRAJ}"
        + ".pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(R, file)
        pickle.dump(agent.solve_times, file)
        pickle.dump(agent.node_counts, file)
        pickle.dump(env.unwrapped.viol_counter[0], file)
        pickle.dump(leader_x, file)
