import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Mario Benchmark")

    parser.add_argument('--agent', type=str, default='DQN', choices=['DQN', 'DDQN', 'TRPO', 'PPO', 'DDPG', 'MCTS'], help='RL Agent')
    
    parser.add_argument('--cnn', type=str, default='AlexNet', choices=['AlexNet', 'ResNet', 'VGG', 'Inception'], help='CNN Type')
    
    parser.add_argument('--skip_frame_num', type=int, default=0, help='Number of frames skipped')
    
    parser.add_argument('--stack_frame_num', type=int, default=0, help='Number of frames stacked')
    
    parser.add_argument('--resize', type=int, default=84, help='New size of picture')
    
    parser.add_argument('--gray', type=bool, default=False, help='Whether to do gray scale')

    return args