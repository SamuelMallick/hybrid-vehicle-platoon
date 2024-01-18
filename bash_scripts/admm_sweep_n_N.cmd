for /l %%n in (2, 1, 5) do (
   for /l %%N in (3, 2, 7) do (
      python examples/ACC_fleet/fleet_naive_admm.py %%n %%N 0 0 1 1
   )
)

for /l %%n in (2, 1, 5) do (
   for /l %%N in (3, 2, 7) do (
      python examples/ACC_fleet/fleet_naive_admm.py %%n %%N 1 0 1 1
   )
)
