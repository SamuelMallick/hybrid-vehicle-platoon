SET LT=1
SET HOM=1
for /l %%n in (2, 1, 8) do (
   python examples/ACC_fleet/fleet_event_based.py %%n 7 0 0 %HOM% %LT%
)