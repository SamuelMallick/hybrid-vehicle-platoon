import pickle

nx_l = 2
n = 2
N = 3
plot_len = 100
types = ["cent", "decent_vest_none", "seq", "admm_20"]

steps = 1

for name in types:
    with open(
        f"data/{name}_default_n_{n}_N_{N}_nonlinear_steps_{steps}_seed_1.pkl",
        "rb",
    ) as file:
        X = pickle.load(file)
        U = pickle.load(file)
        R = pickle.load(file)
        solve_times = pickle.load(file)
        node_counts = pickle.load(file)
        violations = pickle.load(file)
        leader_state = pickle.load(file)

    nl_cost = sum(R)
    nl_time = sum(solve_times) / len(solve_times)

    with open(
        f"data/{name}_default_n_{n}_N_{N}_disc_steps_{steps}_seed_1.pkl",
        "rb",
    ) as file:
        X = pickle.load(file)
        U = pickle.load(file)
        R = pickle.load(file)
        solve_times = pickle.load(file)
        node_counts = pickle.load(file)
        violations = pickle.load(file)
        leader_state = pickle.load(file)

    dg_cost = sum(R)
    dg_time = sum(solve_times) / len(solve_times)

    with open(
        f"data/{name}_default_n_{n}_N_{N}_pwa_steps_{steps}_seed_1.pkl",
        "rb",
    ) as file:
        X = pickle.load(file)
        U = pickle.load(file)
        R = pickle.load(file)
        solve_times = pickle.load(file)
        node_counts = pickle.load(file)
        violations = pickle.load(file)
        leader_state = pickle.load(file)

    pwa_cost = sum(R)
    pwa_time = sum(solve_times) / len(solve_times)

    print(name)
    print(f"disc J percent inc {100*(dg_cost-pwa_cost)/pwa_cost}")
    print(f"disc t percent inc {100*(dg_time-pwa_time)/pwa_time}")
    print(f"nl J percent inc {100*(nl_cost-pwa_cost)/pwa_cost}")
    print(f"nl t percent inc {100*(nl_time-pwa_time)/pwa_time}")
