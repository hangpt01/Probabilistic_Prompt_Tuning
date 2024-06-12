python3 -u main.py --alg pfedpg --dataset fourdataset --batch 16 --device cuda:$1 --comms 1200 --lr 1e-4 --n_clients 80 --data_distribution manual_extreme_heterogeneity --n_dominated_class 1
echo "pfedpg fourdataset comms=1200 lr=1e-4 manual_extreme_heterogeneity done! (1)"
python3 -u main.py --alg pfedpg --dataset fourdataset --batch 16 --device cuda:$1 --comms 1200 --lr 1e-4 --n_clients 80 --data_distribution manual_extreme_heterogeneity --n_dominated_class 1
echo "pfedpg fourdataset comms=1200 lr=1e-4 manual_extreme_heterogeneity done! (2)"
python3 -u main.py --alg pfedpg --dataset fourdataset --batch 16 --device cuda:$1 --comms 1200 --lr 1e-4 --n_clients 80 --data_distribution manual_extreme_heterogeneity --n_dominated_class 1
echo "pfedpg fourdataset comms=1200 lr=1e-4 manual_extreme_heterogeneity done! (3)"