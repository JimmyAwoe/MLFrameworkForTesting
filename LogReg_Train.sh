# 请在bash中输入
# bash LogReg_Train.sh
# 启动训练
python3 logistic_reg.py \
        --output output.txt\
        --bthsz 1024\
        --epoch 300\
        --datapath 'spotify_songs.csv'\
        --lr 0.01\
        --optimizer Vanilla_SGD\
        --model VanillaLogReg

