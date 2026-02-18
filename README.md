# Indications préliminaires 

- Pour avoir un aperçu des résultats à l'échelle nationale : National_Scale/Analyse_globale.
- Pour tracer le réseau de chaleur d'une commune d'un département : Local_Scale/State_of_the_art.ipynb 

Base de données disponibles sur GitHub : tout ce qui concerne le département 60

# Structure générale du code 

Ce code comporte 5 dossiers principaux : 

- data_basis
- Local_Scale
- Departemental_Scale
- National_Scale
- Evolution_Modeles

Quelques précisions générales : 

### data_basis 

Contient les bases de données nécessaires au traitement des communes à toutes les échelles, avec données des bâtiments et des routes. Contient également les fichiers de données utiles aux études annexes comme l'évolution de l'énergie thermique demandée par un bâtiment pour différents profils types

### Local_Scale

Contient le programme le plus important dans le notebook State_of_the_art.ipynb, permet de tracer le réseau de chaleur de n'importe quelle commune de n'importe quel département, de jouer sur les hypothèses, de comparer éventuellement avec le réseau réel s'il existe...

### Departemental_Scale 

Contient l'ensemble des fonctions décrites dans State_of_the_art.ipynb condensées dans le fichier fonctions_utiles.ipynb afin d'avoir un notebook court et clair lorsque l'on veut lancer l'étude à l'échelle du département dans Departement.ipynb

### National_Scale

Contient également un fichier fonctions_utiles.py pour lancer le programme à l'échelle nationale dans National.py, un fichier storage.py qui gère la gestion du stockage des communes et un dossier Storage_results où sont stockés les résultats par commune pour toute la France.
Le notebook Analyse_Globale.ipynb présente une analyse statistique des résultats à l'échelle nationale.
Le dossier Storage_results contient les résultats avec 
 - results.ipynb permet de lire les résultats de la commune de votre choix
 - fonctions_utiles.py
 - depart_XX : avec meta.json : résumé des résultats, small_communes.csv : petits réseaux et leurs caractéristiques, rejected_communes.csv : communes rejetées et leurs caractéristiques, selected_communes.csv : communes sélectionnées et leurs caractéristiques. Communes et Small_networks contiennent respectivement les résultats des grands et petits réseaux au format pkl (à lire avec results.py)

### Evolution_Modeles

Lien_conso_reseau.ipynb a servi pour l'étude de quel bâtiment prendre pour construire le réseau.
conso_plant contient des images des performances des réseaux construits par communes en partant de tous les bâtiments possibles


Le fichier preparation_fichier.ipynb contient le code permettant de préparer les bases de données de référence pour utiliser dans State_of_the_art.ipynb


