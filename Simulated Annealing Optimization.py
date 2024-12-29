import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Mention the Dataset name
x = 'GSE18842.xlsx'

# Load the Dataset using pandas
df = pd.read_excel(x)
# print(df.head(5))

# Define the keys to specify 'Healthy' and 'Disease' samples
Healthy = 'Control'
Disease = 'Tumor'

# Preprocessing
df = df.drop(['!Sample_title'], axis=1)
df = df.dropna(subset=['Gene Symbol'])

# Handle the Duplicate rows based on Gene name
duplicates = df[df.duplicated(subset='Gene Symbol', keep=False)]
avg_duplicates = duplicates.groupby('Gene Symbol').mean().reset_index()
filtered_df = df[~df.duplicated(subset='Gene Symbol', keep=False)]
df = pd.concat([filtered_df, avg_duplicates])

# Normalize the matrix
scaler = MinMaxScaler()
normalized = scaler.fit_transform(df.iloc[:, 1:])
normalized_df = pd.DataFrame(normalized, columns=df.columns[1:]).reset_index(drop=True)
normalized_df['Gene Symbol'] = df['Gene Symbol'].reset_index(drop=True)
normalized_df.set_index('Gene Symbol', inplace=True)

# Transpose the matrix
transposed_df = normalized_df.transpose()

# Define seed for reproducibility 
np.random.seed(43)

# Calculating fitness Value
def calculate_fitness(selected_genes):
    sample_names = ['Control', 'Tumor']
    selected_rows = transposed_df.index.str.contains('|'.join(sample_names))
    selected_samples_df = transposed_df[selected_rows]

    control_rows = selected_samples_df.index.str.contains('Control')
    tumor_rows = selected_samples_df.index.str.contains('Tumor')
    mean_differences = abs(selected_samples_df[control_rows].mean() - selected_samples_df[tumor_rows].mean())
    mean_differences_selected = mean_differences[selected_genes]
    fitness_value = mean_differences_selected.sum()
    return fitness_value

# Annealing 
def next_iteration(selected_genes, temperature):

    new_gene = np.random.choice(transposed_df.columns)
    while new_gene in selected_genes:
        new_gene = np.random.choice(transposed_df.columns)

    new_selected_genes = selected_genes.copy()
    index_to_replace = np.random.randint(len(selected_genes))
    new_selected_genes[index_to_replace] = new_gene

    current_fitness = calculate_fitness(selected_genes)
    new_fitness = calculate_fitness(new_selected_genes)

    if new_fitness > current_fitness:
        return new_selected_genes, new_fitness
    else:
        delta_fitness = new_fitness - current_fitness
        scaled_delta_fitness = delta_fitness / max(current_temperature, 1e-10)
        probability = np.exp(scaled_delta_fitness)

        if np.random.rand() < probability:
            return new_selected_genes, new_fitness
        else:
            return selected_genes, current_fitness

 
selected_genes = np.random.choice(transposed_df.columns, size=40, replace=False)

# Define initial temperature, cooling rate and iteration
initial_temperature = 1000000000.0    # 1000000000.0 Initial temperature
cooling_rate = 0.95    # 0.95 cooling rate
iteration = 250000    # 250000 iteration

# Main simulated annealing loop
current_temperature = initial_temperature
fitness_values = []
for i in range(iteration):  
    selected_genes, current_fitness = next_iteration(selected_genes, current_temperature)
    current_temperature *= cooling_rate
    fitness_values.append(current_fitness)

print("Selected genes in the final iteration:\n", selected_genes.tolist())
print("\nFitness value in the final iteration:", current_fitness)

# Plotting 
plt.figure(figsize=(14,7))
plt.plot(fitness_values)
plt.xlabel('Iteration', fontsize = 24)
plt.ylabel('Fitness Value', fontsize = 24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
# plt.title('Simulated Annealing')
plt.grid(True)

# # Save the figure as a PNG file
# plt.savefig('GSE18842__Simulated_Annealing_Optimization 02.png', dpi=500)  # Adjust dpi for higher resolution if needed

plt.show()

