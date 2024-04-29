import csv

# Define the filename to read the data from and save the data to
input_filename = "log.txt"
output_filename = "output_data.csv"

# Define the data structure to store the values
output_data = []

# Read the text from the input file
with open(input_filename, mode="r") as file:
    output_text = file.read()

# Split the output text into individual sections
output_sections = output_text.split("----------------------------------\n")

# Iterate over each section to extract the desired values
for section in output_sections[:-1]:  # Skip the last empty section
    values = {}
    lines = section.split("\n")
    for line in lines:
        parts = line.split("|")
        if len(parts) > 2:
            key = parts[1].strip()
            value = parts[2].strip()
            values[key] = value
    output_data.append(values)

# Write the data to a CSV file
with open(output_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["ep_rew_mean", "time_elapsed", "total_timesteps"])
    writer.writeheader()
    for data in output_data:
        writer.writerow({
            "ep_rew_mean": data.get("ep_rew_mean", ""),
            "time_elapsed": data.get("time_elapsed", ""),
            "total_timesteps": data.get("total_timesteps", "")
        })

print(f"Data saved to {output_filename}")