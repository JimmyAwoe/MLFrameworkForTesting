# 请在bash中输入
# bash LogReg_Plot.sh
# 启动绘图

python3 logistic_reg_plot.py \
        --output_file Store_Plot\
        --batch_list 512,128,256,1024,2048\
        --epoch 30 \
        --optimizer_list AdaDelta_SGD,Vanilla_SGD,Momentum_SGD,Nesterov_SGD,AdaGrad_SGD,RMSProp_SGD,Adam_SGD\
        --datapath spotify_songs.csv \
        --lr_list 0.001,0.01,0.05,0.10\
        --model_list VanillaLogReg,VanillaLogRegNorm\
        --norm_weight 1 \