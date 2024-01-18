for /l %%n in (2, 1, 5) do (
   for /l %%N in (3, 2, 7) do (
      python examples/ACC_fleet/fleet_seq_mld.py %%n %%N 1 0 1 1
   )
)
