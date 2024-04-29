import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the directory containing the CSV files
directory = './data/'

# List all CSV files in the directory
csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
print(csv_files)

# Define dictionaries to store the data
dataframes = {}

# Read the data from the CSV files
for file in csv_files:
    dataframes[file] = {}
    ep_rew_mean_values = []
    time_elapsed_values = []
    total_timesteps_values = []
    with open(file, mode="r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            ep_rew_mean = float(row["ep_rew_mean"]) if row["ep_rew_mean"] else None
            time_elapsed = float(row["time_elapsed"]) if row["time_elapsed"] else None
            total_timesteps = int(row["total_timesteps"]) if row["total_timesteps"] else None
            
            if ep_rew_mean is not None:
                ep_rew_mean_values.append(ep_rew_mean)
            if time_elapsed is not None:
                time_elapsed_values.append(time_elapsed)
            if total_timesteps is not None:
                total_timesteps_values.append(total_timesteps)
    dataframes[file]["ep_rew_mean"] = ep_rew_mean_values
    dataframes[file]["time_elapsed"] = time_elapsed_values
    dataframes[file]["total_timesteps"] = total_timesteps_values
    
import matplotlib.pyplot as plt

# Filter dataframes for each model
dqn_dataframes = {file: dataframe for file, dataframe in dataframes.items() if 'DQN' in file}
a2c_dataframes = {file: dataframe for file, dataframe in dataframes.items() if 'A2C' in file}
ppo_dataframes = {file: dataframe for file, dataframe in dataframes.items() if 'PPO' in file}

# Plot for DQN models
plt.figure()
plt.title('DQN Models')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
for file, dataframe in dqn_dataframes.items():
    label = file.split('/')[-1][:-4]  # Extract only the model name without the path and extension
    plt.plot(dataframe["total_timesteps"], dataframe["ep_rew_mean"], label=label)
plt.legend()
plt.savefig('./plots/dqn_models_plot.png')

# Plot for A2C models
plt.figure()
plt.title('A2C Models')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
for file, dataframe in a2c_dataframes.items():
    label = file.split('/')[-1][:-4]  # Extract only the model name without the path and extension
    plt.plot(dataframe["total_timesteps"], dataframe["ep_rew_mean"], label=label)
plt.legend()
plt.savefig('./plots/a2c_models_plot.png')

# Plot for PPO models
plt.figure()
plt.title('PPO Models')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
for file, dataframe in ppo_dataframes.items():
    label = file.split('/')[-1][:-4]  # Extract only the model name without the path and extension
    plt.plot(dataframe["total_timesteps"], dataframe["ep_rew_mean"], label=label)
plt.legend()
plt.savefig('./plots/ppo_models_plot.png')

baseline_dataframes = {file: dataframe for file, dataframe in dataframes.items() if 'Baseline' in file}
resnet_dataframes = {file: dataframe for file, dataframe in dataframes.items() if 'ResNet' in file}
vgg_dataframes = {file: dataframe for file, dataframe in dataframes.items() if 'VGG' in file}

# Plot for Baseline CNN
plt.figure()
plt.title('Baseline CNN')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
for file, dataframe in baseline_dataframes.items():
    label = file.split('/')[-1][:-4]  # Extract only the model name without the path and extension
    plt.plot(dataframe["total_timesteps"], dataframe["ep_rew_mean"], label=label)
plt.legend()
plt.savefig('./plots/baseline_cnn_plot.png')  # Save the plot

# Plot for ResNet CNN
plt.figure()
plt.title('ResNet CNN')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
for file, dataframe in resnet_dataframes.items():
    label = file.split('/')[-1][:-4]  # Extract only the model name without the path and extension
    plt.plot(dataframe["total_timesteps"], dataframe["ep_rew_mean"], label=label)
plt.legend()
plt.savefig('./plots/resnet_cnn_plot.png')  # Save the plot

# Plot for VGG CNN
plt.figure()
plt.title('VGG CNN')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
for file, dataframe in vgg_dataframes.items():
    label = file.split('/')[-1][:-4]  # Extract only the model name without the path and extension
    plt.plot(dataframe["total_timesteps"], dataframe["ep_rew_mean"], label=label)
plt.legend()
plt.savefig('./plots/vgg_cnn_plot.png')  # Save the plot

# Plot for total_timesteps vs time_elapsed for all files
plt.figure()
plt.title('Total Timesteps vs Time Elapsed')
plt.xlabel('Total Timesteps')
plt.ylabel('Time Elapsed')
for file, dataframe in dataframes.items():
    label = file.split('/')[-1][:-4]  # Extract only the model name without the path and extension
    plt.plot(dataframe["total_timesteps"], dataframe["time_elapsed"], label=label)
plt.legend()
plt.savefig('./plots/total_timesteps_vs_time_elapsed_plot.png')  # Save the plot