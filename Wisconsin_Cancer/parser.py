import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
plt.clf()

sns.pairplot(df, hue="Diagnosis", vars=list(df)[12:22])
plt.savefig("graphs/scatter_matrix_n2.png")
plt.clf()

sns.pairplot(df, hue="Diagnosis", vars=list(df)[22:32])
plt.savefig("graphs/scatter_matrix_n3.png")
plt.clf()

# Scatter Matrix por variable de cada nucleo
for i in range(2, 12):
    sns.pairplot(df, hue="Diagnosis", vars=list(df)[i::10])
    plt.savefig("graphs/scatter_matrix_v" + str(i - 1) + ".png")
    plt.clf()

# Analisis en componentes principales
pca = PCA()
X_transform = pca.fit_transform(df.iloc[:, 2:].values)
Var_C = pca.explained_variance_ratio_
C = pca.components_

# Grafica departicipacion de variables en componentes
plt.title("Participación de Variables en Componentes")
cmap = sns.diverging_palette(10, 240, n=40, as_cmap=True)
sns.heatmap(C, cmap=cmap, vmin=-1, vmax=1,
            linewidths=.5, cbar_kws={"shrink": .5},
            xticklabels=5, yticklabels=5)
plt.xlabel("Variables")
plt.ylabel("Componentes")
plt.savefig("graphs/participacion_vars_cmpt.png", dpi=900)
plt.clf()

# Grafica de varianza acumulada
plt.title("Varianza por Componente")
plt.xlabel("Componente")
plt.ylabel("Varianza")
plt.ylim([0, 0.5])
plt.plot(range(30), Var_C, "-o")
plt.vlines(range(30), 0, Var_C, "tab:orange")
plt.savefig("graphs/varianza_cmpt.png")
plt.clf()

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
df_transform["Diagnosis con CA"] = np.where(df_transform["Diagnosis"] == "M",
                                            -df_transform["Score 2"],
                                            df_transform["Score 2"])

# Grafica Scatter de Componente 0 contra Score 1
plt.title("Componente 0 con Calidad de Aproximacion")
plt.axhline(0, color="tab:grey", alpha=0.5)
plt.axvline(0, color="tab:grey", alpha=0.5)
sns.scatterplot(x="Componente 0", y="Score 1", data=df_transform,
                hue="Diagnosis", alpha=0.5)
plt.savefig("graphs/cmpt0_score1")
plt.clf()

# Grafica a 2 Componentes
palette = sns.diverging_palette(220, 20, as_cmap=True)
plt.title("Componentes 0 y 1 con Calidad de Aproximacion")
plt.axhline(0, color="tab:grey", alpha=0.5)
plt.axvline(0, color="tab:grey", alpha=0.5)
sns.scatterplot(x="Componente 0", y="Componente 1",
                data=df_transform, alpha=0.9,
                palette=palette, hue="Diagnosis con CA",
                legend=False)
plt.savefig("graphs/cmpt0_cmpt1")
plt.clf()


cov = df.iloc[:, 2:].corr()

mask = np.zeros_like(cov, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(10, 220, as_cmap=True)

sns.heatmap(cov, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0,
            linewidths=.5, cbar_kws={"shrink": .5},
            xticklabels=10, yticklabels=10)
plt.savefig("graphs/cov", dpi=900)
plt.clf()

# Plot de vectores
colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
plt.quiver(0, 0, C[0, :], C[1, :],
           angles="xy", scale=1,
           scale_units="xy", color=colors)

legends = []
for s, color in zip(df.columns[2:12], colors):
    legends.append(mlines.Line2D([], [], color=color, label=s[3:]))

plt.legend(handles=legends)

# for s, coors in zip(df.columns[2:], C[:2, :].T):
#     plt.annotate(s, coors)

plt.xlim(0, 0.5)
plt.ylim(-0.4, 0.4)
plt.show()
