#!/bin/bash

features_por_conjuto=(14 28 56)

for f in "${features_por_conjuto[@]}"; do
  echo "Executando: features_por_conjuto=$f"
  python wrapper.py --features_por_conjuto "$f" > "logs/log_wrapper_${f}.txt"
done