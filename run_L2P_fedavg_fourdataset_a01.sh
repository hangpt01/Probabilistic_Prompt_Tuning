python3 -u main.py --alg fedavg --dataset fourdataset --batch 128 --device cuda:0 --comms 12 --lr 1e-4 --n_clients 80 --model_type L2P --alpha 0.1 --instance_label 1
echo "fedavg fourdataset comms=12 lr=1e-4 alpha=0.1 done! (1)"
# python3 -u main.py --alg fedavg --dataset fourdataset --batch 16 --device cuda:0 --comms 120 --lr 1e-4 --n_clients 80 --model_type L2P --alpha 0.1 --instance_label 2
# echo "fedavg fourdataset comms=120 lr=1e-4 alpha=0.1 done! (2)"
# python3 -u main.py --alg fedavg --dataset fourdataset --batch 16 --device cuda:0 --comms 120 --lr 1e-4 --n_clients 80 --model_type L2P --alpha 0.1 --instance_label 3
# echo "fedavg fourdataset comms=120 lr=1e-4 alpha=0.1 done! (3)"