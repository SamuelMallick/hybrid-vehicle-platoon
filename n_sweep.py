from fleet_cent_mld import simulate as sim_cent
from fleet_decent_mld import simulate as sim_dec
from fleet_event_based import simulate as sim_event
from fleet_naive_admm import simulate as sim_admm
from fleet_seq_mld import simulate as sim_seq
from fleet_g_admm import simulate as sim_gadmm
from misc.common_controller_params import Sim_n_task_1, Sim_n_task_2

task_1 = True
thread_limit = None
seed = 2

for n in [5, 10, 15]:
    if task_1:
        sim = Sim_n_task_1(n)
    else:
        sim = Sim_n_task_2(n)
    sim_cent(sim, save=True, plot=False, seed=seed, thread_limit=thread_limit)
    sim_seq(sim, save=True, plot=False, seed=seed, thread_limit=thread_limit)
    sim_admm(sim, 100, save=True, plot=False, seed=seed, thread_limit=thread_limit)
    sim_gadmm(sim, save=True, plot=False, seed=seed)
    

