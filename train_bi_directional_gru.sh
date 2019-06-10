#!/usr/bin/env bash

nohup python3 ./train_bi_directional_gru.py > logs/run.log 2>&1 &
tail -f logs/run.log