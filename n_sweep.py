from fleet_cent_mld import simulate as sim_cent
from fleet_decent_mld import simulate as sim_dec
from fleet_event_based import simulate as sim_event
from fleet_naive_admm import simulate as sim_admm
from fleet_seq_mld import simulate as sim_seq
from misc.common_controller_params import Sim_n_task_1, Sim_n_task_2

task_1 = False
thread_limit = 5
seed_range = [i for i in range(150)]

for seed in seed_range:
    for n in range(2, 9):
        if task_1:
            sim = Sim_n_task_1(n)
        else:
            sim = Sim_n_task_2(n, seed=seed)
        # try:
        #     sim_cent(sim, save=True, plot=False, seed=seed, thread_limit=thread_limit)
        # except:
        #     pass
        # try:
        #     sim_seq(sim, save=True, plot=False, seed=seed, thread_limit=thread_limit)
        # except:
        #     pass
        # try:
        #     sim_dec(
        #         sim,
        #         save=True,
        #         plot=False,
        #         seed=seed,
        #         thread_limit=thread_limit,
        #         velocity_estimator=False,
        #     )
        # except:
        #     pass
        # try:
        #     sim_dec(
        #         sim,
        #         save=True,
        #         plot=False,
        #         seed=seed,
        #         thread_limit=thread_limit,
        #         velocity_estimator=True,
        #     )
        # except:
        #     pass
        try:
            sim_event(
                sim, 2, save=True, plot=False, seed=seed, thread_limit=thread_limit
            )
        except:
            pass
        try:
            sim_admm(
                sim, 5, save=True, plot=False, seed=seed, thread_limit=thread_limit
            )
        except:
            pass
        try:
            sim_event(
                sim, 10, save=True, plot=False, seed=seed, thread_limit=thread_limit
            )
        except:
            pass
        try:
            sim_admm(
                sim, 50, save=True, plot=False, seed=seed, thread_limit=thread_limit
            )
        except:
            pass
