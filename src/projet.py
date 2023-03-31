import func as f
import tkinter as tk

if __name__ == "__main__":
    # Créer une nouvelle fenêtre
    fenetre = tk.Tk()
    fenetre.geometry("500x700")
    fenetre.resizable(False,False)

    # Ajouter du texte à la fenêtre
    label = tk.Label(fenetre, text="Bienvenue sur le logiciel de parcours de donnée.")
    label.pack()
    label2 = tk.Label(fenetre, text="Choisissez ce que vous voulez voir :\n")
    label2.pack()

    # Ajouter un bouton
    # Histogrammes
    labelHIST = tk.Label(fenetre, text="Histogrammes :\n")
    labelHIST.pack(pady=3)
    bouton1 = tk.Button(fenetre, text="Histogrammes des moyennes pour chaque semestre de chaque classe", command=f.HistoNotes)
    bouton1.pack(pady=5)

    #Régression linéaire
    labelreg = tk.Label(fenetre, text="Régressions linéaires : \n")
    labelreg.pack(pady=3)
    boutonLab1 = tk.Button(fenetre, text="Regression linéraire : Notes en fonction du temps de travail", command=f.Linear1)
    boutonLab1.pack(pady=5)
    boutonLab2 = tk.Button(fenetre, text="Regression linéraire : Notes en fonction du temps libre", command=f.Linear2)
    boutonLab2.pack(pady=5)
    boutonLab3 = tk.Button(fenetre, text="Regression linéraire : Notes en fonction de la consommation d'alcool", command=f.Linear3)
    boutonLab3.pack(pady=5)
    boutonLab4 = tk.Button(fenetre, text="Regression linéraire : Notes en fonction du niveau de santé", command=f.Linear4)
    boutonLab4.pack(pady=5)

    #Autres
    labelautre = tk.Label(fenetre, text="Autres : \n")
    labelautre.pack(pady=5)
    bouton3 = tk.Button(fenetre, text="Variances", command=f.Var)
    bouton3.pack(pady=5)
    bouton3 = tk.Button(fenetre, text="Moyenne", command=f.moyenne)
    bouton3.pack(pady=5)
    bouton4 = tk.Button(fenetre, text="Ecart-type", command=f.EcartType)
    bouton4.pack(pady=5)
    bouton2 = tk.Button(fenetre, text="Quitter", command=fenetre.destroy)
    bouton2.pack(pady=100)

    # Lancer la boucle principale de la fenêtre
    fenetre.mainloop()