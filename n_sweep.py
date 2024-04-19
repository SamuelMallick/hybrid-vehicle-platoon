from fleet_cent_mld import simulate as sim_cent
from fleet_decent_mld import simulate as sim_dec
from fleet_seq_mld import simulate as sim_seq
from misc.common_controller_params import Sim_n_task_2

for n in range(2, 10):
    sim = Sim_n_task_2(n)
    sim_cent(sim, save=True, plot=False)
    sim_seq(sim, save=True, plot=False)
    sim_dec(sim, save=True, plot=False)
