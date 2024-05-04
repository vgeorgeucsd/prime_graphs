#!/bin/bash

python3 make_prime_graphs.py
cd ../SANA
./sana -t 2 -tolerance 0 -fg1 ../g1_SANA.el -fg2 ../g2_SANA.el -ics 1 -ec 1 -s3 1 -o output 
