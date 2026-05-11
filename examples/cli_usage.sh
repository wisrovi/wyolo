#!/bin/bash

# Sample commands to use wyolo via CLI

# 1. Basic training using a config file
wyolo-train --config_path examples/config_template.yaml --fitness map50 --trial_number 1

# 2. Overriding fitness metric to evaluate
wyolo-train --config_path examples/config_template.yaml --fitness fitness/mAP50-95 --trial_number 2

# 3. Running inside Docker (example)
# docker run --gpus all -v $(pwd):/workspace wyolo:latest \
#    wyolo-train --config_path /workspace/examples/config_template.yaml
