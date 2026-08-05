#!/bin/bash

python inference.py --model msnn &
python inference.py --model regression &
python recurrent_blackbox.py &

wait

echo "All processes finished"

python plot_results.py

echo "Plotting finished"