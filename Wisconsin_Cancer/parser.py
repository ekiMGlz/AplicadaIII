import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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

# Analisis en componentes principales
pca = PCA()
X_transform = pca.fit_transform(df.iloc[:, 2:].values)
Var_C = pca.explained_variance_ratio_
C = pca.components_

# Grafica departicipacion de variables en componentes
plt.title("Participacion de Variables en Componentes")
cmap = sns.diverging_palette(10, 240, n=40, as_cmap=True)
sns.heatmap(C, cmap=cmap, vmin=-1, vmax=1)
plt.xlabel("Variables")
plt.ylabel("Componentes")
plt.savefig("graphs/participacion_vars_cmpt.png")

# Grafica de varianza acumulada
plt.title("Varianza por Componente")
plt.xlabel("Componente")
plt.ylabel("Varianza")
plt.ylim([0, 0.5])
plt.plot(range(30), Var_C, "-o")
plt.vlines(range(30), 0, Var_C, "tab:orange")
plt.savefig("graphs/varianza_cmpt.png")

# Creacion de DataFrame con dos componentes y score a 1 y 2 componentes
norms = np.array([np.linalg.norm(row) for row in df.iloc[:, 2:].values])
score_1cmpt = (X_transform[:, 0]/norms)**2
norms2 = np.apply_along_axis(np.linalg.norm, 1, X_transform[:, :2])
score_2cmpt = (norms2 / norms) ** 2

df_transform = df.iloc[:, :2].copy()
df_transform["Componente 0"] = X_transform[:, 0]
df_transform["Componente 1"] = X_transform[:, 1]
df_transform["Score 1"] = score_1cmpt
df_transform["Score 2"] = score_2cmpt

# Grafica Scatter de Componente 0 contra Score 1
plt.title("Componente 0 con Calidad de Aproximacion")
plt.axhline(0, color="tab:grey", alpha=0.5)
plt.axvline(0, color="tab:grey", alpha=0.5)
sns.scatterplot(x="Componente 0", y="Score 1", data=df_transform,
                hue="Diagnosis")
plt.savefig("graphs/cmpt0_score1")

# Grafica a 2 Componentes
plt.title("Componentes 0 y 1 con Calidad de Aproximacion")
plt.axhline(0, color="tab:grey", alpha=0.5)
plt.axvline(0, color="tab:grey", alpha=0.5)
sns.scatterplot(x="Componente 0", y="Componente 1",
                data=df_transform, size="Score 2",
                sizes=(10, 100), alpha=0.6,
                hue="Diagnosis", legend=False)
plt.savefig("graphs/cmpt0_cmpt1")
