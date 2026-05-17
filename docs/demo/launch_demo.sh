#!/bin/bash
tmux kill-session -t demo 2>/dev/null
tmux new-session -d -s demo -x 220 -y 45
tmux send-keys -t demo "python3 naive_demo.py" C-m
tmux split-window -h -t demo
tmux send-keys -t demo "sleep 1 && python3 flash_demo.py" C-m
tmux attach -t demo
