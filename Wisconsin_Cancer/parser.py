import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/wdbc.data")
A = df.iloc[:, 2:32].values

# Normalizar la matriz de datos
for col in A.T:
    mu = np.mean(col)
    sigma = np.std(col)
    col -= mu
    col /= sigma

sns.pairplot(df, hue="Diagnosis", vars=list(df)[2:32:10])
plt.show()
