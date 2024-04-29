import csv

# Define the filename to read the data from
output_filename = "output_data.csv"

# Define arrays to store the data
ep_rew_mean_values = []
time_elapsed_values = []
total_timesteps_values = []

# Read the data from the CSV file
with open(output_filename, mode="r") as file:
    reader = csv.DictReader(file)
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

print("ep_rew_mean_values:", ep_rew_mean_values)
print("time_elapsed_values:", time_elapsed_values)
print("total_timesteps_values:", total_timesteps_values)