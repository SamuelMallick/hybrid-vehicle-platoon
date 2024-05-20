from fleet_cent_mld import simulate as sim_cent
from fleet_decent_mld import simulate as sim_decent
from fleet_event_based import simulate as sim_event
from fleet_naive_admm import simulate as sim_admm
from fleet_seq_mld import simulate as sim_seq
from misc.common_controller_params import Sim_n_task_1, Sim_n_task_2

thread_limit = 5
seed_range = [i for i in range(10)]
for N in range(2, 6):
    for seed in seed_range:
        sim = Sim_n_task_2(n=6, N=N, seed=seed)
        try:
            sim_event(
                sim,
                event_iters=3,
                save=True,
                plot=False,
                seed=seed,
                thread_limit=thread_limit,
            )
        except:
            pass
