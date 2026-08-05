import pandas as pd
import numpy as np  

# Load the CSV files
for i in range(1):
    df = pd.read_csv(f"dataset_sin/train/mpc_sim_data_{i}.csv")
    if i == 0:
        combined_df = df
    else:
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
# Save the combined DataFrame to a new CSV file
combined_df.to_csv("dataset_sin/train/all.csv", index=False)