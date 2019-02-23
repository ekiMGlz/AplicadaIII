import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/wdbc.data")

# Normalizar la matriz de datos
for col in df.columns[2:]:
    mu = np.mean(df[col])
    sigma = np.std(df[col])
    df[col] = df[col].apply(lambda x: (x-mu)/sigma)

# Scatter Matrix por nucleo de cada variable
sns.pairplot(df, hue="Diagnosis", vars=list(df)[2:12])
plt.savefig("graphs/scatter_matrix_n1.png")

sns.pairplot(df, hue="Diagnosis", vars=list(df)[12:22])
plt.savefig("graphs/scatter_matrix_n2.png")

sns.pairplot(df, hue="Diagnosis", vars=list(df)[22:32])
plt.savefig("graphs/scatter_matrix_n3.png")

# Scatter Matrix por variable de cada nucleo
for i in range(2, 12):
    sns.pairplot(df, hue="Diagnosis", vars=list(df)[i::10])
    plt.savefig("graphs/scatter_matrix_v" + str(i - 1) + ".png")
