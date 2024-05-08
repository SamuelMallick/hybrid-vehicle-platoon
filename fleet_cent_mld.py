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
from misc.common_controller_params import Params, Sim
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Platoon
from mpcs.cent_mld import MpcMldCent
from mpcs.mpc_gear import MpcGear, MpcNonlinearGear

# from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet

np.random.seed(2)


class MpcGearCent(MpcMldCent, MpcMldCentDecup, MpcGear):
    def __init__(
        self,
        n: int,
        N: int,
        systems: list[dict],
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        leader_index: int = 0,
        quadratic_cost: bool = True,
        thread_limit: int | None = None,
        accel_cnstr_tightening: float = 0.0,
        real_vehicle_as_reference: bool = False,
    ) -> None:
        self.n = n
        MpcMldCentDecup.__init__(
            self, systems, n, N, thread_limit=thread_limit, constrain_first_state=False
        )  # use the MpcMld constructor
        F = block_diag(*[systems[i]["F"] for i in range(n)])
        G = np.vstack([systems[i]["G"] for i in range(n)])
        self.setup_gears(N, F, G)
        self.setup_cost_and_constraints(
            self.u_g,
            spacing_policy,
            leader_index,
            quadratic_cost,
            accel_cnstr_tightening,
            real_vehicle_as_reference,
        )


class MpcNonlinearGearCent(MpcMldCent, MpcNonlinearGear):
    def __init__(
        self,
        n: int,
        N: int,
        nl_systems: list[dict],
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        leader_index: int = 0,
        quadratic_cost: bool = True,
        thread_limit: int | None = None,
        real_vehicle_as_reference: bool = False,
    ) -> None:
        MpcNonlinearGear.__init__(self, nl_systems, N, thread_limit=thread_limit)
        F = block_diag(*[nl_systems[i]["F"] for i in range(n)])
        G = np.vstack([nl_systems[i]["G"] for i in range(n)])
        self.setup_gears(N, F, G)
        self.setup_cost_and_constraints(
            self.u_g,
            spacing_policy,
            leader_index,
            quadratic_cost,
            real_vehicle_as_reference,
        )


class TrackingCentralizedAgent(MldAgent):
    def __init__(self, mpc: MpcMld, ep_len: int, N: int, leader_x: np.ndarray) -> None:
        self.ep_len = ep_len
        self.N = N
        self.leader_x = leader_x

        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))
        self.bin_var_counts = np.zeros((ep_len, 1))
        super().__init__(mpc)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.mpc.set_leader_traj(self.leader_x[:, timestep : (timestep + self.N + 1)])
        self.solve_times[env.step_counter - 1, :] = self.run_time
        self.node_counts[env.step_counter - 1, :] = self.node_count
        self.bin_var_counts[env.step_counter - 1, :] = self.num_bin_vars
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int, state) -> None:
        self.mpc.set_leader_traj(self.leader_x[:, 0 : self.N + 1])
        return super().on_episode_start(env, episode, state)


def simulate(
    sim: Sim,
    save: bool = False,
    plot: bool = True,
    seed: int = 1,
    thread_limit: int | None = None,
    leader_index=0,
):
    n = sim.n  # num cars
    N = sim.N  # controller horizon
    ep_len = sim.ep_len  # length of episode (sim len)
    ts = Params.ts
    masses = sim.masses

    spacing_policy = sim.spacing_policy
    leader_trajectory = sim.leader_trajectory
    leader_x = leader_trajectory.get_leader_trajectory()
    # vehicles
    platoon = Platoon(n, vehicle_type=sim.vehicle_model_type, masses=masses)
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
                leader_index=leader_index,
            ),
            max_episode_steps=ep_len,
        )
    )

    # mpcs
    if sim.vehicle_model_type == "pwa_gear":
        mpc = MpcMldCent(
            n,
            N,
            systems,
            spacing_policy=spacing_policy,
            leader_index=leader_index,
            thread_limit=thread_limit,
            real_vehicle_as_reference=sim.real_vehicle_as_reference,
        )
    elif sim.vehicle_model_type == "pwa_friction":
        mpc = MpcGearCent(
            n,
            N,
            systems,
            spacing_policy=spacing_policy,
            leader_index=leader_index,
            thread_limit=thread_limit,
            real_vehicle_as_reference=sim.real_vehicle_as_reference,
        )
    elif sim.vehicle_model_type == "nonlinear":
        mpc = MpcNonlinearGearCent(
            n,
            N,
            systems,
            spacing_policy=spacing_policy,
            leader_index=leader_index,
            thread_limit=thread_limit,
            real_vehicle_as_reference=sim.real_vehicle_as_reference,
        )
    else:
        raise ValueError(f"{sim.vehicle_model_type} is not a valid vehicle model type.")

    # agent
    agent = TrackingCentralizedAgent(mpc, ep_len, N, leader_x)

    agent.evaluate(env=env, episodes=1, seed=seed)

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

    if plot:
        plot_fleet(n, X, U, R, leader_x, violations=env.unwrapped.viol_counter[0])

    if save:
        with open(
            f"cent_{sim.id}_seed_{seed}" + ".pkl",
            "wb",
        ) as file:
            pickle.dump(X, file)
            pickle.dump(U, file)
            pickle.dump(R, file)
            pickle.dump(agent.solve_times, file)
            pickle.dump(agent.node_counts, file)
            pickle.dump(env.unwrapped.viol_counter[0], file)
            pickle.dump(leader_x, file)


if __name__ == "__main__":
    simulate(Sim(), save=False, seed=1, leader_index=0)
