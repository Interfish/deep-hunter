#!/usr/bin/env bash

nohup python3 $@ > ./logs/run.log 2>&1 &
echo 'trainning log is in ./logs/run.log, now automatically watching (by tail -f )...'
tail -f ./logs/run.log