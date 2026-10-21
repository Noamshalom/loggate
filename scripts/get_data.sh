#!/usr/bin/env bash
# Download the HDFS sessions in DeepLog format (~30 MB total) into data/.
# Source: https://github.com/wuyifan18/DeepLog (data preprocessed by the DeepLog authors
# from the Loghub HDFS_v1 dataset, https://github.com/logpai/loghub).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
BASE="https://raw.githubusercontent.com/wuyifan18/DeepLog/master/data"
for f in hdfs_train hdfs_test_normal hdfs_test_abnormal; do
  if [ -s "data/$f" ]; then echo "data/$f already present"; continue; fi
  echo "downloading $f ..."
  curl -sSL -o "data/$f" "$BASE/$f"
done
wc -l data/hdfs_train data/hdfs_test_normal data/hdfs_test_abnormal
