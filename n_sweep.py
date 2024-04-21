from fleet_cent_mld import simulate as sim_cent
from fleet_decent_mld import simulate as sim_dec
from fleet_event_based import simulate as sim_event
from fleet_naive_admm import simulate as sim_admm
from fleet_seq_mld import simulate as sim_seq
from misc.common_controller_params import Sim_n_task_2

for n in range(2, 10):
    sim = Sim_n_task_2(n)
    sim_cent(sim, save=True, plot=False)
    sim_seq(sim, save=True, plot=False)
    sim_dec(sim, save=True, plot=False)
    sim_event(sim, 4, save=True, plot=False)
    sim_event(sim, 6, save=True, plot=False)
    sim_event(sim, 10, save=True, plot=False)
    sim_admm(sim, 20, save=True, plot=False)
    sim_admm(sim, 50, save=True, plot=False)
