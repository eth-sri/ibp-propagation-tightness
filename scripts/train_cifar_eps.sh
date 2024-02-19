#!/bin/bash

cmds=()
for seed in 1 12 123
do
        for eps in 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1
        do
            c2="python mix_train.py \
                --dataset cifar10 --init default --use-adv-training --random-seed $seed \
                --net cnn_3layer \
                --cert-weight 1 \
                --lr 0.0005 --L2-reg 0 --lr-milestones 120 140 \
                --train-eps $eps --test-eps $eps  --train-steps 20 --test-steps 20  \
                --train-batch 128 --test-batch 128  \
                --pgd-weight-start 1 --pgd-weight-end 1 \
                --grad-clip 10 \
                --n-epochs 160 --start-epoch-eps 0 --end-epoch-eps 0 \
                --min-eps-pgd 0 \
                --soft-thre 0.5 \
                --alpha-box 5  \
                --save-dir ./eps_models/seed_$seed \
                --block-sizes 17 4"
            cmds+=("$c2")
        done
done

gpu_list=(0)
gpu_prefix="CUDA_VISIBLE_DEVICES"
num_gpus=${#gpu_list[@]}

num_cmds=${#cmds[@]}
echo "Commands in total will run: $num_cmds"
echo "Number of parallel tasks for each GPU?"
read max_parallel_task
echo "Estimated time for each process in secs?"
read estimation
secs=$(($estimation * $num_cmds / $max_parallel_task / $num_gpus))
echo "Your estimated execution time:"

printf '%02dh:%02dm:%02fs\n' $(echo -e "$secs/3600\n$secs%3600/60\n$secs%60"| bc)

echo "Running in a screen is highly recommended. Proceed? y/n: "
read decision

if [ $decision != "y" ]
then
    exit
else
    echo "Your job will start now. Good luck!"
    sleep 1
fi




for ((i = 0; i < ${#cmds[@]}; i++))
do
    gpu_index="$(($i % ($num_gpus * $max_parallel_task) ))"
    c="$gpu_prefix=${gpu_list[$(($gpu_index / $max_parallel_task))]} ${cmds[$i]}"
    if [ "$(( $(($i + 1)) % $(($num_gpus * $max_parallel_task)) ))" == "0" ] || [ "$(($i + 1))" == $num_cmds ]
    then
        true
    else
        c="$c &"
    fi
    eval " $c"
done