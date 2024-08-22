python3 -u main.py --alg fedavg_nonpara --dataset fourdataset --batch 64 --device cuda:0 --comms 120 --lr 1e-4 --n_clients 80 --model_type L2P --alpha 0.1
# python3 -u main.py --alg fedavg_nonpara --dataset fourdataset --batch 16 --device cuda:1 --comms 120 --lr 1e-4 --n_clients 80 --model_type L2P --alpha 0.1 --instance_label 1
# echo "fedavg_nonpara fourdataset comms=120 lr=1e-4 alpha=0.1 done! (1)"

# export CUDA_HOME=/usr/local/cuda-11 CUDA_VISIBLE_DEVICES=1 python -u main.py --alg fedavg_nonpara --dataset fourdataset --batch 256 --device cuda --comms 120 --lr 1e-4 --n_clients 80 --model_type L2P --alpha 0.1
