nO=1
nD=100
k_size=20
nI=50


lrD=0.0004
lrG=0.0004
batchsize=60
epochs=100

trainingC=10
gp_w=5
de_w=0.6
gp_w_ded=5

degree=4
block_size=100
strides=100

ts_perc=0.1667
data=
data_bad="AR_Anomaly_1.txt"
dataset="PJM"

python main.py -data_path "PJM_LMP_temperature.txt" -dataset "PJM" -output_dim 1 -hidden_dim 100 -seq_len 50 -num_feature 2 -filter_size 20 -lrD 0.0001 \
-lrG 0.0001 -batch_size 60 -epochs 100 -num_critic 10 -gp_coef_inn 5 -gp_coef_recons 5 -coef_recons 0.6 -test_perc 0.1667 -seed 200
