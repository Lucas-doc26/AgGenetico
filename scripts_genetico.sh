#!/bin/bash

geracoes=(2 3)
pop=(4 6 8 )
mutacao=(0.01 0.1 0.2)

mkdir logs

for g in "${geracoes[@]}"; do
  for p in "${pop[@]}"; do
    for m in "${mutacao[@]}"; do
      echo "Executando: geracoes=$g, pop=$p, mutacao=$m"
      python genetico.py --geracoes "$g" --pop "$p" --mutacao "$m" > "logs/log_genetico_${g}_${p}_${m}.txt"
    done
  done
done
