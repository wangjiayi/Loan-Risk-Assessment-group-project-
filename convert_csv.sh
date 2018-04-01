#!/bin/bash
#
# Convert .npy to .csv
#

python convert_csv.py ecs171train.npy 0 -h
python convert_csv.py ecs171test.npy 0
