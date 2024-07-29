import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the DataFrame from the CSV file
df = pd.read_csv('svd_results.csv')

# Ensure that 'k', 'l', and 'm' are treated as categorical variables
df['v'] = df['v'].astype(int)
df['m'] = df['m'].astype(int)
df['n'] = df['n'].astype(int)

# Set up the plotting environment
sns.set(style="whitegrid")

# Create the folder if it doesn't exist
output_folder = "ToShowJenny"
os.makedirs(output_folder, exist_ok=True)

# Function to calculate the order of magnitude
def calculate_order_of_magnitude(values) -> int:
    max_val = values.max().max()
    order_of_magnitude = np.floor(np.log10(max_val))
    return order_of_magnitude

# Function to format annotations
def format_annotations(values, order_of_magnitude):
    return values / (10 ** order_of_magnitude)

# Iterate over each unique value of k
for k_value in df['v'].unique():
    # Filter the DataFrame for the current k value
    df_k = df[df['v'] == k_value]
    
    # Pivot the DataFrame to get the data in a suitable format for heatmap
    pivot_max_singular_value = df_k.pivot_table(values="max_singular_value", index="m", columns="n").iloc[::-1]
    pivot_exponential_decay = df_k.pivot_table(values="exponential_decay", index="m", columns="n").iloc[::-1]
    pivot_terms_to_one_percent = df_k.pivot_table(values="terms_to_one_percent", index="m", columns="n").iloc[::-1]
    
    # Calculate the order of magnitude for max_singular_value and exponential_decay
    order_max_singular_value = calculate_order_of_magnitude(pivot_max_singular_value)
    order_exponential_decay = calculate_order_of_magnitude(pivot_exponential_decay)
    
    # Scale the values for better readability
    scaled_max_singular_value = format_annotations(pivot_max_singular_value, order_max_singular_value)
    scaled_exponential_decay = format_annotations(pivot_exponential_decay, order_exponential_decay)
    
    # Ensure terms_to_one_percent values are integers
    pivot_terms_to_one_percent = pivot_terms_to_one_percent.round().astype(int)
    
    # Create a new figure for the current k value
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))  # Adjusted size to accommodate three subplots
    
    # Plot the heatmap for max_singular_value
    sns.heatmap(scaled_max_singular_value, annot=True, fmt=".2f", cmap="viridis", ax=axes[0])#, cbar_kws={'label': f'Max Singular Value ($10^{{int(order_max_singular_value)}}$)'})
    axes[0].set_title(f'Max Singular Value ($10^{{{int(order_max_singular_value)}}}$) for $\\nu$={k_value}', fontsize=16)
    axes[0].set_xlabel('n', fontsize=16)
    axes[0].set_ylabel('m', fontsize=16, rotation=0, labelpad=30)  # Ensure the label is vertical    
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].yaxis.set_label_coords(-0.1, 0.5)  # Adjust label position if needed
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=00)
    for spine in axes[0].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # Plot the heatmap for exponential_decay
    sns.heatmap(scaled_exponential_decay, annot=True, fmt=".2f", cmap="viridis", ax=axes[1])#, cbar_kws={'label': f'Exponential Decay ($10^{{int(order_exponential_decay)}}$)'})
    axes[1].set_title(f'Exponential Decay ($10^{{{int(order_exponential_decay)}}}$) for $\\nu$={k_value}', fontsize=16)
    axes[1].set_xlabel('n', fontsize=16)
    axes[1].set_ylabel('m', fontsize=16, rotation=0, labelpad=30)  # Ensure the label is vertical    
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    axes[1].yaxis.set_label_coords(-0.1, 0.5)  # Adjust label position if needed
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
    for spine in axes[1].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # Plot the heatmap for terms_to_one_percent
    sns.heatmap(pivot_terms_to_one_percent, annot=True, fmt="d", cmap="viridis", ax=axes[2])
    axes[2].set_title(f'Terms to 0.1% of Max Singular Value for $\\nu$={k_value}', fontsize=16)
    axes[2].set_xlabel('n', fontsize=16)
    axes[2].set_ylabel('m', fontsize=16, rotation=0, labelpad=30)  # Ensure the label is vertical    
    axes[2].tick_params(axis='both', which='major', labelsize=12)
    axes[2].yaxis.set_label_coords(-0.1, 0.5)  # Adjust label position if needed
    axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation=0)
    for spine in axes[2].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{output_folder}/heatmap_k_{k_value}.svg")
    plt.close()
