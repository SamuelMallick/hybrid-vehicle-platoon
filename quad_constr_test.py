import gurobipy as gp

m = gp.Model("model")

x = m.addMVar((3, 1), name="x")
m.addConstr(x[0, 0] == x[1, 0] * x[1, 0], name="quadratic constraint")
m.addConstr(x[2, 0] >= 3 + x[0, 0])
m.addConstr(x[0, 0] + x[1, 0] >= 1)
m.setObjective(x.T @ x, gp.GRB.MINIMIZE)

m.optimize()
x = x.X
