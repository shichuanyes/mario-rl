#!/bin/bash

# # Define all possible values for each argument
# AGENTS=('DQN' 'PPO' 'A2C')
# CNNS=('Baseline' 'ResNet' 'VGG' 'Inception')
# # SKIP_FRAME_NUMS=(4)
# # RESIZES=(84)
# # GRAY_VALUES=('True')
# TOTAL_TIMESTEPS=(10000 100000 1000000 2000000 5000000 10000000)

# # Loop through all combinations
# for agent in "${AGENTS[@]}"; do
#     for cnn in "${CNNS[@]}"; do
#         for total_timesteps in "${TOTAL_TIMESTEPS[@]}"; do
#             # Generate unique model save path based on arguments
#             model_save_path="./model_file/experiment_${agent}_${cnn}_skip${skip_frame_num}_stack${stack_frame_num}_resize${resize}_gray${gray}_timesteps${total_timesteps}.zip"
            
#             echo "Running with arguments:"
#             echo "--agent $agent --cnn $cnn --skip_frame_num $skip_frame_num --resize $resize --gray $gray --total_timesteps $total_timesteps"
            
#             # Suppress warnings by redirecting stderr to /dev/null
#             python main.py --agent "$agent" --cnn "$cnn" --skip_frame_nums "$skip_frame_num"  --resize "$resize" --gray "$gray" --total_timesteps "$total_timesteps" --model_save_path "$model_save_path" 
#         done

#     done
# done


# Define all possible values for each argument
AGENTS=('DQN' 'PPO' 'A2C')
CNNS=('Baseline' 'ResNet' 'VGG')
# TOTAL_TIMESTEPS=(10000 100000 1000000)
TOTAL_TIMESTEPS=(1000000)


# Loop through all combinations
for total_timesteps in "${TOTAL_TIMESTEPS[@]}"; do
    for agent in "${AGENTS[@]}"; do
        for cnn in "${CNNS[@]}"; do
            # Generate unique model save path based on arguments
            model_save_path="./model_file/experiment_${agent}_${cnn}_timesteps${total_timesteps}.zip"
            
            echo "Running with arguments:"
            echo "--agent $agent --cnn $cnn --total_timesteps $total_timesteps"
            
            # Suppress warnings by redirecting stderr to /dev/null
            python main.py --agent "$agent" --cnn "$cnn" --total_timesteps "$total_timesteps" --model_save_path "$model_save_path" 
        done
    done
done
