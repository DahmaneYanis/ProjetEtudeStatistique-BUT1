import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression

# Récupération du fichier csv -> DataFrame
df = pd.read_csv("donnee/student-mat.csv", sep=";")

# Quantité de donnée
## Quel type de donnée ?

print(df.dtypes)

## Taille des données
colonnes = len(df.columns)
ligne = len(df)

# Nettoyage des données*

# Analyse des données et études statistiques.
# Variables
SommeGP = [0, 0, 0]
SommeMS = [0, 0, 0]

nbGP = 0
nbMS = 0

## Moyenne des notes en fonction des écoles
### Toutes les notes dans un tablea
tabGP1 = np.array([]) 
tabGP2 = np.array([])
tabGP3 = np.array([])

tabMS1 = np.array([])
tabMS2 = np.array([])
tabMS3 = np.array([])


def ListeNotes():
    global tabGP1, tabGP2, tabGP3, tabMS1, tabMS2, tabMS3
    for i in range(len(df)):
        if (df.iloc[i, 0] == 'GP'):
            tabGP1 = np.append(tabGP1, np.array(df.iloc[i, 30]))
            tabGP2 = np.append(tabGP2, np.array(df.iloc[i, 31]))
            tabGP3 = np.append(tabGP3, np.array(df.iloc[i, 32]))
        elif (df.iloc[i, 0] == 'MS'):
            tabMS1 = np.append(tabMS1, np.array(df.iloc[i, 30]))
            tabMS2 = np.append(tabMS2, np.array(df.iloc[i, 31]))
            tabMS3 = np.append(tabMS3, np.array(df.iloc[i, 32]))

ListeNotes()
tabNotes= [tabGP1, tabGP2, tabGP3, tabMS1, tabMS2, tabMS3] #Tableau 2D des notes
moytabGP = 0
moytabMS = 0
### Les moyennes
def moyenne():
    global tabGP1, tabGP2, tabGP3, tabMS1, tabMS2, tabMS3, moytabGP, moytabMS
    moytabGP = np.mean(np.array([tabGP1.mean(), tabGP2.mean(), tabGP3.mean()])) # GP
    moytabMS = np.mean(np.array([tabMS1.mean(), tabMS2.mean(), tabMS3.mean()])) # MS

    print(moytabGP, moytabMS)
    

### Calcul de la médiane
medianeGP = np.median(np.array([tabGP1.mean(), tabGP2.mean(), tabGP3.mean()])) # GP
medianeMS = np.median(np.array([tabMS1.mean(), tabMS2.mean(), tabMS3.mean()])) # MS

varGP1, varGP2, varGP3, varMS1, varMS2, varMS3 = 0, 0, 0, 0, 0, 0

### Variance
def Var(affiche : bool = True):
    global tabGP1, tabGP2, tabGP3, tabMS1, tabMS2, tabMS3, varGP1, varGP2, varGP3, varMS1, varMS2, varMS3

    varGP1 = tabGP1.var()
    varGP2 = tabGP2.var()
    varGP3 = tabGP3.var()

    varMS1 = tabMS1.var()
    varMS2 = tabMS2.var()
    varMS3 = tabMS3.var()

    if (affiche) :
        print("var : " ,  varGP1, varGP2, varGP3)
        print("var : " ,  varMS1, varMS2, varMS3)

### Ecart-type
def EcartType():
    global devGP1, devGP2, devGP3, devMS1, devMS2, devMS3, varGP1, varGP2, varGP3, varMS1, varMS2, varMS3

    Var(False)

    devGP1 = math.sqrt(varGP1)
    devGP2 = math.sqrt(varGP2)
    devGP3 = math.sqrt(varGP3)

    devMS1 = math.sqrt(varMS1)
    devMS2 = math.sqrt(varMS2)
    devMS3 = math.sqrt(varMS3)

    print("Ecart-type : \n----------------")
    print("GP : " ,  devGP1, devGP2, devGP3)
    print("MS : " ,  devMS1, devMS2, devMS3)

### Quantilles
# https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
listeQuantille = [] #liste des quantilles (0, 25, 75, 100) pour chaque semestre de chaque classe


def Quantille():
    global tabNotes, listeQuantille
    for i in range(4):
        lTEMPO = []
        for j in range(2):
            lTEMPO.append([0, 0, 0])
        listeQuantille.append(lTEMPO)

    listeQuantille = np.array(listeQuantille)

    valQuant = [0, 0.25, 0.75, 1]
    for i in range(4):
        z = 0
        for j in range(2):
            for k in range(3):
                listeQuantille[i, j, k] = np.quantile(tabNotes[z], valQuant[i])
                z+=1

    print(listeQuantille)

#listeQuantille[0][0][0] = np.quantile(tabGP1, 0.25) #quantile 25, premiere école, premier semestre

## Histogram
### Nombre d'élève ayant eu cette note

def HistoNotes():
    titre = ["Semestre 1 - Gabriel Pereira", "Semestre 2 - Gabriel Pereira", "Semestre 3 - Gabriel Pereira", "Semestre 1 - Mousinho da Silveira", "Semestre 2 - Mousinho da Silveira", "Semestre 3 - Mousinho da Silveira"]

    for i in range(6):
        plt.figure()

        plt.hist(tabNotes[i], bins=21, range=(0, 21))

        plt.xticks(np.arange(0, 21, 1)) # De -5 à 5 avec un intervalle de 1
        plt.xlabel("Note" )
        plt.ylabel("Nombre d'élèves ayant eu cette note")
        plt.title(titre[i])
    plt.show()
# We can set the number of bins with the *bins* keyword argument.

## Regression linéaire
def Linear1():
    global df
    temps = df["studytime"]
    note = df["G1"]
    temps = temps[:, np.newaxis]
    model = LinearRegression(fit_intercept=True)
    model.fit(temps, note)
    x = np.linspace(1, 4, 4)
    x = x[:, np.newaxis]
    y = model.predict(x)

    plt.figure()
    plt.scatter(temps, note, color="b")
    plt.plot(x, y, color="r")
    
    plt.xlabel("Temps de travail")
    plt.ylabel("Notes")
    plt.title("Régression linéaire des notes en fonction du temps de travail")
    
    plt.show()

    # On peut voir ici que plus le temps de travail augmente, plus les notes ont tendances à augmenter

def Linear2():
    global df
    temps = df["freetime"]
    note = df["G1"]
    temps = temps[:, np.newaxis]
    model = LinearRegression(fit_intercept=True)
    model.fit(temps, note)
    x = np.linspace(1, 5, 5)
    x = x[:, np.newaxis]
    y = model.predict(x)

    plt.figure()
    plt.scatter(temps, note, color="b")
    plt.plot(x, y, color="r")
    
    plt.xlabel("Temps libre")
    plt.ylabel("Notes")
    plt.title("Régression linéaire des notes en fonction du temps libre")
    
    plt.show()

    ### Conclusion:
    # LE TEMPS LIBRE N'INFLU CASIMENT PAS SUR LES NOTES

def Linear3():
    global df
    temps = df["Dalc"]
    note = df["G1"]
    temps = temps[:, np.newaxis]
    model = LinearRegression(fit_intercept=True)
    model.fit(temps, note)
    x = np.linspace(1, 5, 5)
    x = x[:, np.newaxis]
    y = model.predict(x)

    plt.figure()
    plt.scatter(temps, note, color="b")
    plt.plot(x, y, color="r")
    
    plt.xlabel("Consommation d'alcool durant les cours")
    plt.ylabel("Notes")
    plt.title("Régression linéaire des notes en fonction de la consommation d'alcool")
    
    plt.show()

    ### Conclusion:
    # LA CONSOMMATION D'ALCOOL SEMBLE INFLUER SUR LES NOTES -> Plus on consomme de l'alcool dans la semaine plus la moyenne des notes semble baisser

def Linear4():
    global df
    temps = df["health"]
    note = df["G1"]
    temps = temps[:, np.newaxis]
    model = LinearRegression(fit_intercept=True)
    model.fit(temps, note)
    x = np.linspace(1, 5, 5)
    x = x[:, np.newaxis]
    y = model.predict(x)

    plt.figure()
    plt.scatter(temps, note, color="b")
    plt.plot(x, y, color="r")
    
    plt.xlabel("Niveau de santé")
    plt.ylabel("Notes")
    plt.title("Régression linéaire des notes en fonction du niveau de santé")
    
    plt.show()

    ### Conclusion:
    # LA SANTé SEMBLE INFLUER SUR LES NOTES -> Plus la santé est basse plus la moyenne des notes semble baisser




# Ajouter des labels pour les axes x et y

if __name__ == "__main__":
    Linear3()