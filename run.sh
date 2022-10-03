#!/bin/bash

start=$(date +%s.%N)

numberOfImage=39000
python3 master.py $numberOfImage & sleep 3
python3 worker1.py $numberOfImage & sleep 3
python3 worker2.py $numberOfImage & sleep 3
python3 worker3.py $numberOfImage

duration=$(echo "$(date +%s.%N) - $start" | bc)
execution_time=`printf "%.2f seconds" $duration`

echo "Script Execution Time: $execution_time"
