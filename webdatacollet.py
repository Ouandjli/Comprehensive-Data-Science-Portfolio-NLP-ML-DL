import requests
from bs4 import BeautifulSoup

# URL de la page Wikipédia
wiki_url = 'https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches'

# Effectuer une requête GET à la page Wikipédia
response = requests.get(wiki_url)

# Vérifier si la requête a réussi
if response.status_code == 200:
    # Convertir la réponse en BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Trouver tous les tableaux contenant les lancements
    tables = soup.find_all('table', {'class': 'wikitable'})

    # Variables pour compter les lancements
    falcon9_launches = 0
    falcon1_launches = 0

    # Parcourir chaque tableau
    for table in tables:
        rows = table.find_all('tr')
        for row in rows[1:]:  # Ignore l'en-tête
            columns = row.find_all('td')
            # Vérifier le nombre de colonnes avant d'accéder aux données
            if len(columns) > 1:  # Assurez-vous qu'il y a au moins deux colonnes
                rocket_type = columns[1].get_text(strip=True)
                # Déboguer l'extraction des colonnes
                print(f"Colonnes trouvées : {[col.get_text(strip=True) for col in columns]}")
                if 'Falcon 9' in rocket_type:
                    falcon9_launches += 1
                elif 'Falcon 1' in rocket_type:
                    falcon1_launches += 1

    # Calculer les lancements de Falcon 9 après avoir retiré ceux de Falcon 1
    net_falcon9_launches = falcon9_launches - falcon1_launches

    print(f'Nombre total de lancements de Falcon 9 : {falcon9_launches}')
    print(f'Nombre total de lancements de Falcon 1 : {falcon1_launches}')
    print(f'Nombre de lancements de Falcon 9 après avoir retiré les lancements de Falcon 1 : {net_falcon9_launches}')

else:
    print(f"Erreur lors de la requête : {response.status_code}")