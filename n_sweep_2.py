from fleet_cent_mld import simulate as sim_cent
from fleet_decent_mld import simulate as sim_decent
from fleet_event_based import simulate as sim_event
from fleet_naive_admm import simulate as sim_admm
from fleet_seq_mld import simulate as sim_seq
from misc.common_controller_params import Sim

thread_limit = 5

sim = Sim()
sim_cent(sim, save=True, plot=False, thread_limit=thread_limit)
sim_seq(sim, save=True, plot=False, thread_limit=thread_limit)
sim_decent(sim, save=True, plot=False, thread_limit=thread_limit)
sim_event(sim, event_iters=5, save=True, plot=False, thread_limit=thread_limit)
sim_admm(
    sim,
    admm_iters=20,
    save=True,
    plot=False,
    thread_limit=thread_limit,
)
