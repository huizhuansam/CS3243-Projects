#!/bin/zsh

for i in ./Public\ Testcases/*.txt; do
  echo "$i"
  python3 ./Local.py "$i"
  echo -e "\n"
done