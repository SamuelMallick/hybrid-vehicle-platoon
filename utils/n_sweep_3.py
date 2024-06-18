import os
import sys

sys.path.append(os.getcwd())

from fleet_cent_mld import simulate as sim_cent
from fleet_decent_mld import simulate as sim_decent
from fleet_event_based import simulate as sim_event
from fleet_naive_admm import simulate as sim_admm
from fleet_seq_mld import simulate as sim_seq
from misc.common_controller_params import Sim_n_task_2

thread_limit = 5
seed_range = [i for i in range(1)]

for n in range(3, 11):
    lead_range = [i for i in range(2, n)] if n == 3 else [i for i in range(n)]
    for seed in seed_range:
        for leader_index in lead_range:
            sim = Sim_n_task_2(n, seed=seed, leader_index=leader_index)
            try:
                sim_admm(
                    sim,
                    admm_iters=20,
                    save=True,
                    plot=False,
                    seed=seed,
                    thread_limit=thread_limit,
                    leader_index=leader_index,
                )
            except:
                pass
            try:
                sim_cent(
                    sim,
                    save=True,
                    plot=False,
                    seed=seed,
                    thread_limit=thread_limit,
                    leader_index=leader_index,
                )
            except:
                pass
            try:
                sim_seq(
                    sim,
                    save=True,
                    plot=False,
                    seed=seed,
                    thread_limit=thread_limit,
                    leader_index=leader_index,
                )
            except:
                pass
            try:
                sim_decent(
                    sim,
                    save=True,
                    plot=False,
                    seed=seed,
                    thread_limit=thread_limit,
                    velocity_estimator=False,
                    leader_index=leader_index,
                )
            except:
                pass
            try:
                sim_decent(
                    sim,
                    save=True,
                    plot=False,
                    seed=seed,
                    thread_limit=thread_limit,
                    velocity_estimator=True,
                    leader_index=leader_index,
                )
            except:
                pass
            # try:
            #     sim_admm(
            #         sim,
            #         admm_iters=5,
            #         save=True,
            #         plot=False,
            #         seed=seed,
            #         thread_limit=thread_limit,
            #         leader_index=leader_index
            #     )
            # except:
            #     pass

            # try:
            #     sim_admm(
            #         sim,
            #         admm_iters=50,
            #         save=True,
            #         plot=False,
            #         seed=seed,
            #         thread_limit=thread_limit,
            #         leader_index=leader_index
            #     )
            # except:
            #     pass
            # try:
            #     sim_event(
            #         sim,
            #         event_iters=2,
            #         save=True,
            #         plot=False,
            #         seed=seed,
            #         thread_limit=thread_limit,
            #         leader_index=leader_index
            #     )
            # except:
            #     pass
            try:
                sim_event(
                    sim,
                    event_iters=5,
                    save=True,
                    plot=False,
                    seed=seed,
                    thread_limit=thread_limit,
                    leader_index=leader_index,
                )
            except:
                pass
            # try:
            #     sim_event(
            #         sim,
            #         event_iters=10,
            #         save=True,
            #         plot=False,
            #         seed=seed,
            #         thread_limit=thread_limit,
            #         leader_index=leader_index
            #     )
            # except:
            #     pass
