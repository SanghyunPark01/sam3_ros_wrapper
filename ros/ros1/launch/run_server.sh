#!/usr/bin/env bash
exec python3.12 "$(rospack find sam3_ros_wrapper)/scripts/sam3_api_server.py"
