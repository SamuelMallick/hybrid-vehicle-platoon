from fleet_cent_mld import simulate as sim_cent
from fleet_decent_mld import simulate as sim_dec
from fleet_event_based import simulate as sim_event
from fleet_naive_admm import simulate as sim_admm
from fleet_seq_mld import simulate as sim_seq
from misc.common_controller_params import Sim_n_task_1, Sim_n_task_2

task_1 = False
thread_limit = False
seed_range = [i for i in range(50)]

for seed in seed_range:
    for n in range(2, 6):
        if task_1:
            sim = Sim_n_task_1(n)
        else:
            sim = Sim_n_task_2(n, seed=seed)
        try:
            sim_admm(sim, admm_iters=20, save=True, plot=False, seed=seed, thread_limit=thread_limit, leader_index=min(n-1,2))
        except:
            pass 
        # try:
        #     sim_seq(sim, save=True, plot=False, seed=seed, thread_limit=thread_limit, leader_index=0)
        # except:
        #     pass   
        # try:
        #     sim_dec(sim, save=True, plot=False, seed=seed, thread_limit=thread_limit, leader_index=0, velocity_estimator=False)
        # except:
        #     pass
        # try:
        #     sim_dec(sim, save=True, plot=False, seed=seed, thread_limit=thread_limit, leader_index=0, velocity_estimator=True)
        # except:
        #     pass
