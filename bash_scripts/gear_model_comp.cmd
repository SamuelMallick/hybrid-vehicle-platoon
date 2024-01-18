SET nn=3
SET N=5
SET Q=1
SET HOM=1
SET LT=1
python examples/ACC_fleet/fleet_cent_mld.py %nn% %N% %Q% 1 %HOM% %LT%
python examples/ACC_fleet/fleet_decent_mld.py %nn% %N% %Q% 1 %HOM% %LT%
python examples/ACC_fleet/fleet_seq_mld.py %nn% %N% %Q% 1 %HOM% %LT%
python examples/ACC_fleet/fleet_event_based.py %nn% %N% %Q% 1 %HOM% %LT% 4
python examples/ACC_fleet/fleet_naive_admm.py %nn% %N% %Q% 1 %HOM% %LT% 20