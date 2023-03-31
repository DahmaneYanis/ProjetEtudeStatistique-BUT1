import pandas as pd

# Récupération du fichier csv -> DataFrame
df = pd.read_csv("donnee/student-mat.csv", sep=";")

# Quantité de donnée
## Quel type de donnée ?

print(df.dtypes)

## Taille des données
print("Colonne : ", len(df.columns))
print("Ligne : ", len(df))
