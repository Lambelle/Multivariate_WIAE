output_dim=1
hidden_dim=100
filter_size=40
seq_len=120
num_feature=2
batchsize=60
epochs=100

num_critic=10


#data="data/NYISO_Jul_spread_load.csv"
#dataset="NYISO_spread"

data="data/NYISO_Jul_RTDA.csv"
dataset="NYISO_RT"

for lrG in $(seq 0.00001 0.00001 1)
do
  for gp_coef_inn in $(seq 0.1 0.1 5)
  do
    for gp_coef_recons in $(seq 0.1 0.1 5)
    do
      for coef_recons in $(seq 0.1 0.1 5)
      do
        for seed in $(seq 1 20 200)
        do
          python main.py -data_path $data -dataset $dataset -output_dim $output_dim -hidden_dim $hidden_dim -seq_len $seq_len \
          -num_feature $num_feature -filter_size $filter_size -lrD $lrG -lrG $lrG -batch_size $batchsize -epochs $epochs\
           -num_critic $num_critic -gp_coef_inn $gp_coef_inn -gp_coef_recons $gp_coef_recons -coef_recons $coef_recons\
          -seed $seed -pred_step 24
        done
      done
    done
  done
done
