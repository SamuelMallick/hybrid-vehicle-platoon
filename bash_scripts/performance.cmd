SET LT=1
SET HOM=1
SET N=2
SET n=2
SET Q=0
for /l %%N in (5, 1, 4) do (
	for /l %%n in (10, 1, 9) do (
		for /l %%Q in (0, 1, 1) do (	
   			python examples/ACC_fleet/fleet_cent_mld.py %%n %%N %%Q 0 %HOM% %LT%
   			python examples/ACC_fleet/fleet_decent_mld.py %%n %%N %%Q 0 %HOM% %LT%
   			python examples/ACC_fleet/fleet_seq_mld.py %%n %%N %%Q 0 %HOM% %LT%
			python examples/ACC_fleet/fleet_event_based.py %%n %%N %%Q 0 %HOM% %LT% 2
   			python examples/ACC_fleet/fleet_event_based.py %%n %%N %%Q 0 %HOM% %LT% 4
			python examples/ACC_fleet/fleet_event_based.py %%n %%N %%Q 0 %HOM% %LT% 6
			python examples/ACC_fleet/fleet_event_based.py %%n %%N %%Q 0 %HOM% %LT% 8
			python examples/ACC_fleet/fleet_event_based.py %%n %%N %%Q 0 %HOM% %LT% 10
 			python examples/ACC_fleet/fleet_naive_admm.py %%n %%N %%Q 0 %HOM% %LT% 10
			python examples/ACC_fleet/fleet_naive_admm.py %%n %%N %%Q 0 %HOM% %LT% 20
			python examples/ACC_fleet/fleet_naive_admm.py %%n %%N %%Q 0 %HOM% %LT% 50	
		)
	)
)
SET Q=1
SET nn=3
SET N=5
::python examples/ACC_fleet/fleet_cent_mld.py %nn% %N% %Q% 1 %HOM% %LT%
::python examples/ACC_fleet/fleet_decent_mld.py %nn% %N% %Q% 1 %HOM% %LT%
::python examples/ACC_fleet/fleet_seq_mld.py %nn% %N% %Q% 1 %HOM% %LT%
::python examples/ACC_fleet/fleet_event_based.py %nn% %N% %Q% 1 %HOM% %LT% 4
::python examples/ACC_fleet/fleet_naive_admm.py %nn% %N% %Q% 1 %HOM% %LT% 20

SET nn=10
SET N=8
python examples/ACC_fleet/fleet_event_based.py %nn% %N% %Q% 0 %HOM% %LT% 8
python examples/ACC_fleet/fleet_event_based.py %nn% %N% %Q% 0 %HOM% %LT% 10
python examples/ACC_fleet/fleet_naive_admm.py %nn% %N% %Q% 0 %HOM% %LT% 10
python examples/ACC_fleet/fleet_naive_admm.py %nn% %N% %Q% 0 %HOM% %LT% 20
python examples/ACC_fleet/fleet_naive_admm.py %nn% %N% %Q% 0 %HOM% %LT% 50
for /l %%N in (9, 1, 10) do (
	for /l %%n in (8, 1, 10) do (
		python examples/ACC_fleet/fleet_cent_mld.py %%n %%N %Q% 0 %HOM% %LT%
   		python examples/ACC_fleet/fleet_decent_mld.py %%n %%N %Q% 0 %HOM% %LT%
   		python examples/ACC_fleet/fleet_seq_mld.py %%n %%N %Q% 0 %HOM% %LT%
		python examples/ACC_fleet/fleet_event_based.py %%n %%N %Q% 0 %HOM% %LT% 2
   		python examples/ACC_fleet/fleet_event_based.py %%n %%N %Q% 0 %HOM% %LT% 4
		python examples/ACC_fleet/fleet_event_based.py %%n %%N %Q% 0 %HOM% %LT% 6
		python examples/ACC_fleet/fleet_event_based.py %%n %%N %Q% 0 %HOM% %LT% 8
		python examples/ACC_fleet/fleet_event_based.py %%n %%N %Q% 0 %HOM% %LT% 10
   		python examples/ACC_fleet/fleet_naive_admm.py %%n %%N %Q% 0 %HOM% %LT% 10
   		python examples/ACC_fleet/fleet_naive_admm.py %%n %%N %Q% 0 %HOM% %LT% 20
   		python examples/ACC_fleet/fleet_naive_admm.py %%n %%N %Q% 0 %HOM% %LT% 50
	)
)
