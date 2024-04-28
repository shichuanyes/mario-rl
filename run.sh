#!/bin/bash

# Define all possible values for each argument
AGENTS=('DQN' 'PPO' 'A2C')
CNNS=('Baseline' 'AlexNet' 'ResNet' 'VGG' 'Inception')
SKIP_FRAME_NUMS=(0 2 5 8 10 15 20)
STACK_FRAME_NUMS=(0 2 5 8 10 15 20)
RESIZES=(42 84 128 160)
GRAY_VALUES=('False' 'True')
TOTAL_TIMESTEPS=(10000 25000 50000 75000 100000 150000 2000000 300000)

# Loop through all combinations
for agent in "${AGENTS[@]}"; do
    for cnn in "${CNNS[@]}"; do
        for skip_frame_num in "${SKIP_FRAME_NUMS[@]}"; do
            for stack_frame_num in "${STACK_FRAME_NUMS[@]}"; do
                for resize in "${RESIZES[@]}"; do
                    for gray in "${GRAY_VALUES[@]}"; do
                        for total_timesteps in "${TOTAL_TIMESTEPS[@]}"; do
                            # Generate unique model save path based on arguments
                            model_save_path="./model_file/experiment_${agent}_${cnn}_skip${skip_frame_num}_stack${stack_frame_num}_resize${resize}_gray${gray}_timesteps${total_timesteps}.zip"
                            
                            echo "Running with arguments:"
                            echo "--agent $agent --cnn $cnn --skip_frame_num $skip_frame_num --stack_frame_num $stack_frame_num --resize $resize --gray $gray --total_timesteps $total_timesteps"
                            python main.py --agent "$agent" --cnn "$cnn" --skip_frame_num "$skip_frame_num" --stack_frame_num "$stack_frame_num" --resize "$resize" --gray "$gray" --total_timesteps "$total_timesteps" --model_save_path "$model_save_path"
                        done
                    done
                done
            done
        done
    done
done
