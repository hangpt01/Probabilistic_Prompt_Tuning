python3 -u main.py --alg fedavg --dataset tinyimagenet --batch 16 --device cuda:0 --comms 120 --lr 5e-4 --model_type L2P --alpha 0.5 --reduce_sim_scalar 0.002 --instance_label 1
echo "fedavg tinyimagenet comms=120 lr=5e-4 alpha=0.5 done! (1)"
python3 -u main.py --alg fedavg --dataset tinyimagenet --batch 16 --device cuda:0 --comms 120 --lr 5e-4 --model_type L2P --alpha 0.5 --reduce_sim_scalar 0.002 --instance_label 2
echo "fedavg tinyimagenet comms=120 lr=5e-4 alpha=0.5 done! (2)"
python3 -u main.py --alg fedavg --dataset tinyimagenet --batch 16 --device cuda:0 --comms 120 --lr 5e-4 --model_type L2P --alpha 0.5 --reduce_sim_scalar 0.002 --instance_label 3
echo "fedavg tinyimagenet comms=120 lr=5e-4 alpha=0.5 done! (3)"