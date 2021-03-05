#!/usr/bin/env bash

echo "Total paid in bonuses: $(jq '.BonusAmount | tonumber' "$1" | awk '{s+=$1} END {print s}')"
echo "Thank you bonuses: $(grep -c 'thank' "$1")"
