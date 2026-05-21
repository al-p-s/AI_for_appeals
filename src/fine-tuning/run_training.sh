mkdir -p logs
for level in 2 3 4; do
    python train_keryx_multi.py --level $level 2>&1 | tee -a logs/train_multi.log
done
