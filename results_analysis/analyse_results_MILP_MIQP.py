import pickle

Q1_min = []
Q1_max = []
Q1_mean = []
Q0_min = []
Q0_max = []
Q0_mean = []
for i in range(5, 11):
    for j in range(2):
        with open(
            f"examples/ACC_fleet/data/MILP-MIQP/N{i}Q{j}.pkl",
            "rb",
        ) as file:
            X = pickle.load(file)
            U = pickle.load(file)
            R = pickle.load(file)
            run_times = pickle.load(file)

        m = [min(run_times)]
        M = [max(run_times)]
        a = [sum(run_times) / len(run_times)]

        with open(
            f"examples/ACC_fleet/data/MILP-MIQP/N{i}Q{j}_2.pkl",
            "rb",
        ) as file:
            X = pickle.load(file)
            U = pickle.load(file)
            R = pickle.load(file)
            run_times = pickle.load(file)

        m.append(min(run_times))
        M.append(max(run_times))
        a.append(sum(run_times) / len(run_times))

        with open(
            f"examples/ACC_fleet/data/MILP-MIQP/N{i}Q{j}_3.pkl",
            "rb",
        ) as file:
            X = pickle.load(file)
            U = pickle.load(file)
            R = pickle.load(file)
            run_times = pickle.load(file)

        m.append(min(run_times))
        M.append(max(run_times))
        a.append(sum(run_times) / len(run_times))

        if j == 0:
            Q0_min.append(sum(m) / len(m))
            Q0_max.append(sum(M) / len(M))
            Q0_mean.append(sum(a) / len(a))
        elif j == 1:
            Q1_min.append(sum(m) / len(m))
            Q1_max.append(sum(M) / len(M))
            Q1_mean.append(sum(a) / len(a))
print(Q1_min)
print(Q1_max)
print(Q1_mean)
print(Q0_min)
print(Q0_max)
print(Q0_mean)
