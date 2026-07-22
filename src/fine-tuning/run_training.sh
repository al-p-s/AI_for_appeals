mkdir -p logs
python train_keryx_multi_flatL3.py 2>&1 | tee -a logs/train_m_1340_flatL3.log

#for level in 2 3 4; do
#   python train_keryx_multi.py --level $level 2>&1 | tee -a logs/train_m_1340_G.log
#done
