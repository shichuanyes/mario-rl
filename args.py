import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Mario Benchmark")

    parser.add_argument('--agent', type=str, default='DQN', choices=['DQN', 'PPO', 'A2C'], help='RL Agent')
    
    parser.add_argument('--cnn', type=str, default='Baseline', choices=['Baseline', 'AlexNet', 'ResNet', 'VGG', 'Inception'], help='CNN Type')
    
    parser.add_argument('--skip_frame_num', type=int, default=0, help='Number of frames skipped')
    
    parser.add_argument('--stack_frame_num', type=int, default=0, help='Number of frames stacked')
    
    parser.add_argument('--resize', type=int, default=84, help='New size of picture')
    
    parser.add_argument('--gray', type=bool, default=False, help='Whether to do gray scale')

    parser.add_argument('--model_save_path', type=str, default=None, help='Path to the file where the RL agent should be saved')

    parser.add_argument('--total_timesteps', type=int, default=25000, help='Total Number of samples (env steps) to train on')
    
    return parser.parse_args()
