import heapq
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

from collections import defaultdict
from matplotlib.collections import LineCollection
from pyproj import Transformer
from shapely import STRtree
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, Point
from shapely.ops import transform

from dataclasses import dataclass




E_MAX=58.3507 #MWh/an

def selection_fichiers_et_communes(departement) :
    buildings_file = f"/Users/louisedeferran/Documents/Louise/Mines/effynersys/reseau_chaleur/DHN_Potential_France_EFFINERSYS/bases_de_donnees/depart_{departement}/conso_coords_{departement}.csv"
    roads_file = f"/Users/louisedeferran/Documents/Louise/Mines/effynersys/reseau_chaleur/DHN_Potential_France_EFFINERSYS/bases_de_donnees/depart_{departement}/final_routes_{departement}.csv"



    # Charger avec le séparateur correct
    df_buildings= pd.read_csv(buildings_file, sep=',', dtype=str, on_bad_lines='skip', encoding='utf-8')


    roads_df = pd.read_csv(roads_file, sep=',', dtype=str, on_bad_lines='skip', encoding='utf-8')

    df_buildings_copie=df_buildings[["COM_INSEE", "CHAUF_MWH", "ECS_MWH", "geom_group"]].copy()
    df_buildings_copie.rename(columns={"CHAUF_MWH": "conso_ref_chauf_MWh", "ECS_MWH": "conso_ref_ecs_MWh", "COM_INSEE": "code_commune_insee", "geom_group" : "geom_groupe"}, inplace=True)

    df_buildings_copie["conso_ref_tot_MWh"] = df_buildings_copie["conso_ref_chauf_MWh"].astype(float) + df_buildings_copie["conso_ref_ecs_MWh"].astype(float)
    # Filtrage global (une seule fois)
    df_buildings_filtered = df_buildings_copie[
        df_buildings_copie["conso_ref_tot_MWh"].notna()
        & df_buildings_copie["geom_groupe"].notna()
    ]

    # Groupement par commune
    buildings_by_commune = {
        commune: df.reset_index(drop=True)
        for commune, df in df_buildings_filtered.groupby("code_commune_insee")
    }
    return df_buildings_filtered, roads_df, buildings_by_commune

def extraction_routes(df_roads, verbose=False) :
    """
    Extrait les routes d'un département
    
    Args:
        df_roads (pd.Dataframe) : contient les informations des routes d'un département
    
    Returns :
        pd.DataFrame : les routes adéquates (bien définies)
    """
    roads = []  # bâtiments qui seront considérés pour le RCU
    n = len(df_roads)

    for road in range(n):
        properties = df_roads.iloc[road].to_dict()

        if pd.notna(properties["geom_groupe"]):
            roads.append(road)

    if verbose : 
        print(len(roads), "routes ont été retenues.")

    df_selected_roads = df_roads.iloc[roads]
    return df_selected_roads


def extraction_batiments(df_buildings, code_commune, verbose=False):
    """
    Extrait les bâtiments d'une commune spécifique à partir d'un DataFrame de bâtiments.

    Args:
        df_buildings (pd.DataFrame): DataFrame contenant les données des bâtiments.
        code_commune (str): Code INSEE de la commune à extraire.

    Returns:
        pd.DataFrame: DataFrame contenant uniquement les bâtiments de la commune spécifiée.
    """
    buildings = []  # bâtiments qui seront considérés pour le RCU
    n = len(df_buildings)

    for building in range(n):
        properties = df_buildings.iloc[building].to_dict()

        # Vérification des conditions
        if (
            properties["code_commune_insee"] == code_commune and
            pd.notna(properties["conso_ref_tot_MWh"]) and
            pd.notna(properties["geom_groupe"])
        ):
            buildings.append(building)

    if verbose:
        print(len(buildings), "bâtiments ont été retenus.")
    df_selected_buildings = df_buildings.iloc[buildings].reset_index(drop=True)
    return df_selected_buildings
  
    
def transformer_coordonnees(df_buildings):
    """
    Transforme les coordonnées géométriques d'un système de référence à un autre.

    Args:
        df_buildings (pd.DataFrame): DataFrame contenant les données des bâtiments.
        from_crs (str): Système de référence source (par défaut "EPSG:2154").
        to_crs (str): Système de référence cible (par défaut "EPSG:4326").

    Returns:
        pd.DataFrame: DataFrame avec les géométries transformées.
    """
    transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:2154",
    always_xy=True
    )

    df_buildings_copie=df_buildings.copy()
    def reproject_wkt(wkt_geom, transformer):
        try:
            geom = wkt.loads(wkt_geom)
            return transform(transformer.transform, geom)
        except Exception:
            return None
        
    df_buildings_copie["geom_proj"] = df_buildings_copie["geom_groupe"].apply(

    lambda g: reproject_wkt(g, transformer)
    )
    return df_buildings_copie



def sort_and_cut_buildings_by_consumption(df_buildings, verbose=False):

    """
    Trie les bâtiments par consommation énergétique décroissante.

    Args:
        df_buildings (pd.DataFrame): DataFrame contenant les données des bâtiments.

    Returns:
        pd.DataFrame: DataFrame triée par consommation énergétique décroissante.
    """
    df_buildings_copie = df_buildings[df_buildings["conso_ref_tot_MWh"] >= E_MAX].copy()
    df_sorted = df_buildings_copie.sort_values(by="conso_ref_tot_MWh", ascending=False).reset_index(drop=True)
    if verbose: 
        print(f"Le nombre final de bâtiments retenus est de {len(df_sorted["conso_ref_tot_MWh"])}.")
        print(f"La demande totale en chaleur est de {np.sum(df_sorted["conso_ref_tot_MWh"])/1e3:.2f} GWh/an.")
        print(f"Le plus gros bâtiment a une demande de {df_sorted['conso_ref_tot_MWh'].iloc[0]:.2f} MWh/an.")

        print(df_sorted["conso_ref_tot_MWh"].describe())
    return df_sorted


def get_centroid_geom(geom):
    if geom is None:
        return None
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda p: p.area)
    if isinstance(geom, Polygon):
        c = geom.centroid
        return c.x, c.y
    return None

# Fonction pour transformer LineString / MultiLineString en segments XY
def extract_lines_xy(wkt_geom):
    try:
        geom = wkt.loads(wkt_geom)
    except Exception:
        return []  # géométrie invalide

    segments = []

    if isinstance(geom, LineString):
        coords_2d = [(c[0], c[1]) for c in geom.coords]
        segments.append(coords_2d)

    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            coords_2d = [(c[0], c[1]) for c in line.coords]
            segments.append(coords_2d)

    return segments

# Filtrer les segments de routes à l'intérieur du bounding box
def filter_segments(segment, min_x, max_x, min_y, max_y):
    x, y = zip(*segment)  # segment est une liste de tuples (x, y)
    x = np.array(x)
    y = np.array(y)
    return ((x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)).any()


def selection_geom_batiments_routes(df_buildings, df_roads) :
    """
    Sélectionne les géométries des bâtiments et des routes.
    Args:
        df_buildings (pd.DataFrame): DataFrame contenant les données des bâtiments.
        df_roads (pd.DataFrame): DataFrame contenant les données des routes.
    Returns:
        tuple: Listes des coordonnées x et y des centroids des bâtiments, et des segments des routes filtrées  
    """

    all_segments = []
    for geom_wkt in df_roads["geom_groupe"]:
        all_segments.extend(extract_lines_xy(geom_wkt))

    # Extraire les centroids des bâtiments
    centroids = df_buildings["geom_proj"].apply(get_centroid_geom).dropna()
    df_buildings["centroids"]=centroids
    centroids = list(centroids)
    if centroids:
        centroids_x, centroids_y = zip(*centroids)
    else:
        centroids_x, centroids_y = [], []
    



    # Définir la zone des bâtiments (bounding box) car toutes les routes ne sont pas nécessaires
    if centroids_x:
        buffer = 50
        min_x, max_x = min(centroids_x) - buffer, max(centroids_x) + buffer
        min_y, max_y = min(centroids_y) - buffer, max(centroids_y) + buffer
        filtered_lines = [seg for seg in all_segments if filter_segments(seg, min_x, max_x, min_y, max_y)]
    else:
        filtered_lines = all_segments
    
    return df_buildings,filtered_lines



def normalize_coordinates(coords):
    """
    Normalise les coordonnées en liste de tuples [(x, y)]
    """
    if coords is None:
        return []
    # Cas : (x, y)
    if isinstance(coords, (tuple, list)) and len(coords) == 2:
        if all(isinstance(c, (int, float)) for c in coords):
            return [tuple(coords)]

    # Cas : [(x, y), ...]
    if isinstance(coords, list):
        return [(c[0], c[1]) for c in coords if len(c) >= 2]
    

    raise ValueError(f"Format de coordonnées invalide : {coords}")

class Node:
    def __init__(self, index: int = None, coordinates : list[tuple] = None, road_idx1 : int=None ):
        self._index = index
        self._coordinates = normalize_coordinates(coordinates) #equivalent de self.point
        self._center=None
        if self._coordinates != [] : 
            x, y = zip(*self._coordinates)
            self._center = (np.mean(x), np.mean(y))

        self.road_idx1=road_idx1
        self.pipe=0.0
    

    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, value):
        self._index = value

    @property
    def road_idx1(self):
        return self._road_idx1
    
    @road_idx1.setter
    def road_idx1(self, value):
        self._road_idx1 = value
    

    @property
    def center(self):
        return self._center
    
    @center.setter
    def center(self, new_center):
        self._center = new_center

    @property
    def coordinates(self):
        return self._coordinates
    
    
    @coordinates.setter
    def coordinates(self, new_coords):
        self._coordinates = normalize_coordinates(new_coords)
        x, y = zip(*self._coordinates)
        self.center = (np.mean(x), np.mean(y))
    
    @property
    def pipe(self) :
        return self._pipe
    
    @pipe.setter
    def pipe(self, new_pipe) :
        self._pipe=float(new_pipe)
    
    def calcul_distance(self, other) :
        return np.linalg.norm(np.array(self.center)-np.array(other.center))
    


class NodeBuilding(Node):
    def __init__(self, building_coordinates, heat_demand, buildings_idx, index : int = None,  coordinates : list[tuple] = None, road_idx1 : int=None) :
   
        self.building_coordinates = normalize_coordinates(building_coordinates)
        self.building_center = (np.mean([c[0] for c in self.building_coordinates]), np.mean([c[1] for c in self.building_coordinates]))
        self.buildings_idx=buildings_idx
        
        super().__init__(
            index=index,
            coordinates=coordinates, #coordonnées projetées
            road_idx1=road_idx1,
        )

        self.heat_demand = heat_demand  # MWh/an
        



    @property
    def pipe(self) :
        return self._pipe
    
    
    @pipe.setter
    def pipe(self, new_pipe) :
        self._pipe=float(new_pipe)
    
    
    #Redéfinition de la fonction égalité pour comparer deux NodesBuilding
    def __eq__(self, other):
        if not isinstance(other, NodeBuilding):
            return False
        return self.buildings_idx==other.buildings_idx and self.building_coordinates==other.building_coordinates

    
    # Permet l'utilisation comme clé de dictionnaire
    def __hash__(self):
        return hash((tuple(self.building_center), tuple(self.center), self.buildings_idx, tuple(self.coordinates),self.index ))
    def __repr__(self):
        return f"Building Node (index = {self.index}, index_building={self.buildings_idx}, center_building={self.building_center},center={self.center})"
    


class NodeRoad(Node):
    def __init__(self, index, coordinates, road_idx1, road_idx2) :
        super().__init__(
            index=index,
            coordinates=coordinates,
            road_idx1=road_idx1
        )

        self.road_idx2=road_idx2
    
        #Redéfinition de la fonction égalité pour comparer deux NodesRoad
    def __eq__(self, other):
        if not isinstance(other, NodeRoad):
            return False
        return self.road_idx1==other.road_idx1 and self.road_idx2==other.road_idx2 and self.coordinates==other.coordinates
    
    # Permet l'utilisation comme clé de dictionnaire
    def __hash__(self):
        return hash((self.road_idx1, self.road_idx2, self.index, tuple(self.coordinates), self.center))
    def __repr__(self):
        return f"Road_node (index={self.index}, center={self.center}, road_idx1={self.road_idx1}, road_idx2={self.road_idx2})"


class Road():
    def __init__(self, index : int, segments : list[list[tuple]]):
        self.coordinates = normalize_coordinates(segments[index])
        self.index = index
    

        if not self.coordinates:
            raise ValueError(f"Aucune coordonnée valide pour la route {index}")

        self.coords_array = np.array(self.coordinates, dtype=np.float32)
        self.linestring = LineString(self.coordinates)
        self.length = self.linestring.length

def create_liste_nodes_buildings(df_buildings):
    liste_nodes_buildings = []

    for prov_index in range(len(df_buildings)):
        row = df_buildings.iloc[prov_index]
        building_coordinates = row["centroids"]
        heat_demand = row["conso_ref_tot_MWh"]
        building_idx=len(liste_nodes_buildings)
        liste_nodes_buildings.append(
            NodeBuilding(building_coordinates, heat_demand, building_idx)
        )
    return liste_nodes_buildings

def create_roads(filtered_lines):
    liste_roads = []
    for i in range(len(filtered_lines)):
        liste_roads.append(Road(index=i, segments=filtered_lines))
    return liste_roads

PRECISION = 1

def normalize_point(pt):
    return (
        round(pt[0], PRECISION),
        round(pt[1], PRECISION)
    )

def intersection_points_and_roads(liste_roads, liste_nodes_buildings, verbose=False) :

    """
    Docstring pour intersection_points
    
    :param liste_roads: Description
    :param liste_nodes_buildings: Description

    renvoit road_intersections : dictionnaire avec pour clé les indexs des routes dans liste_roads
    et pour valeur la liste des index des points avec qui il y a une intersection

    renvoit aussi liste_intersection_points : liste de tous les points d'intersection (NodeBuilding ou NodeRoad)
    """


    lines = [r.linestring for r in liste_roads]
    tree = STRtree(lines)

    liste_intersection_points=[]
    road_intersections = {r.index: [] for r in liste_roads} #dictionnaire de la liste des intersections entre les routes

    for road in liste_roads:
        line = road.linestring
        idx = road.index

        candidate_indices = tree.query(line) #XXX : normalement ce sont les mêmes indices que dans lines
        for other_idx in candidate_indices:
            if other_idx <= idx:
                continue  # évite doublons

            other_line = lines[other_idx]

            if not line.intersects(other_line):
                continue

            inter = line.intersection(other_line)

            if inter.is_empty:
                continue

            if inter.geom_type == "Point":
                #pt = normalize_point((inter.x, inter.y))
                pt = (inter.x, inter.y)
                road_point = NodeRoad(
                    index=len(liste_intersection_points),
                    coordinates=[pt],
                    road_idx1=idx,
                    road_idx2=other_idx
                )
                liste_intersection_points.append(road_point)
                road_intersections[idx].append(road_point.index)
                road_intersections[other_idx].append(road_point.index)

    nb_roads_intersections=len(liste_intersection_points)

    if verbose : 
        print(f"Le nombre de routes dans road_intersections: {len(road_intersections)}")
        print(f"Le nombre de points d'intersection avec les routes: {len(liste_intersection_points)}")



    # Projection des bâtiments sur les routes
    for building in liste_nodes_buildings:

        building_point = Point(building.building_center)

        nearest_idx = tree.nearest(building_point)
        nearest_line = lines[nearest_idx]

        # Projection orthogonale
        projected_point = nearest_line.interpolate(
            nearest_line.project(building_point)
        )

        pt = normalize_point((projected_point.x, projected_point.y))
        #pt=(projected_point.x, projected_point.y)
        #modifier très légèrement la coordonnée pour éviter les doublons
        if len(liste_intersection_points)>0 :
            if pt in [b.center for b in liste_intersection_points[:nb_roads_intersections]]:
                pt = (pt[0] + 10**-1, pt[1] + 10**-1)
        building.coordinates=[pt] #coordonnées projetées
        pipe=building_point.distance(projected_point)
        building.pipe=pipe

        building.road_idx1=nearest_idx
        building.index=len(liste_intersection_points)


        liste_intersection_points.append(building)
        road_intersections[nearest_idx].append(building.index)



        if verbose:
            print("Intersections calculées entre routes et bâtiments.")
            print(f"Le nombre total de noeuds (bâtiments + intersections routes) est de {len(liste_intersection_points)}.")

    return road_intersections, liste_intersection_points




def sort_road_intersections(road_intersections, liste_intersection_points, liste_roads) : 
    for road_idx, intersections in road_intersections.items():
        line = liste_roads[road_idx].linestring
        intersections.sort(
            key=lambda ip: line.project(Point(liste_intersection_points[ip].center))
        )
        road_intersections[road_idx] = intersections  
    return road_intersections

def find_index_maps(road_intersections) : 

    return( {
        road_idx: {ip_idx: i for i, ip_idx in enumerate(intersections)}
        for road_idx, intersections in road_intersections.items()
    })


def find_voisins(road_intersections, liste_intersection_points, index_maps) :

    #Donne un dictionnaire avec pour clé l'index d'une route et pour valeur un dictionnaire
    #avec pour clé l'index d'un point d'intersection et pour valeur l'indice de ce point 
    #dans la liste triée des intersections de cette route


    def add_voisin(ip_idx, voisin_idx):
        if voisin_idx != ip_idx:
            voisins[ip_idx].add(voisin_idx)
            voisins[voisin_idx].add(ip_idx)

    #Calcul des voisins 
    voisins={ip.index : set() for ip in liste_intersection_points} # dictionnaire des voisins pour chaque point d'intersection, on évite les doublons avec le set

    for i, intersection_point in enumerate(liste_intersection_points):
        road_idx1= intersection_point.road_idx1

        if isinstance(intersection_point, NodeRoad) :
            road_idx2=intersection_point.road_idx2
        else :
            road_idx2 = None


        voisins_pot1 = road_intersections[road_idx1] #liste des index des ip triés dans l'ordre
        voisins_pot2 = road_intersections[road_idx2] if road_idx2 is not None else []

        indice1 = index_maps[road_idx1][intersection_point.index] #place du voisin potentiel dans la liste triée de road_idx1
        indice2 = index_maps[road_idx2][intersection_point.index] if road_idx2 is not None else None

    

        if indice1>0 :
            add_voisin(intersection_point.index, voisins_pot1[indice1-1])
            if indice1<len(voisins_pot1)-1 :
                add_voisin(intersection_point.index, voisins_pot1[indice1+1])
        if indice2 is not None :
            if indice2>0 :
                add_voisin(intersection_point.index, voisins_pot2[indice2-1])
            if indice2<len(voisins_pot2)-1 :
                add_voisin(intersection_point.index, voisins_pot2[indice2+1])

    voisins = {ip: list(vs) for ip, vs in voisins.items()} #conversion des sets en listes d'index d'intersection points, 
                                            #résultat = dictionnaire avec pour clé IntersectionPoint et pour valeur la liste des index de ses voisins IntersectionPoint
    
    return voisins




class Graph:

    def __init__(self, liste_nodes_buildings,road_intersections, liste_intersection_points, voisins, idx_ip_plant):
        self.liste_nodes_buildings = liste_nodes_buildings
        self.road_intersections = road_intersections
        self.liste_intersection_points=liste_intersection_points
        self.voisins=voisins

        self.ip_plant=liste_nodes_buildings[idx_ip_plant] #le powerplant, par defaut le premier elt de liste_nodes_buildings(le plus gros consommateur)
    

        self.distances_to_plant={self.ip_plant.index : 0.0} #dictionnaire des distances entre le powerplant et chaque point d'intersection

        self.nb_buildings=len(liste_nodes_buildings) #on compte le powerplant

        for i in range(len(self.liste_intersection_points)): #on initialise les distances à l'infini sauf pour le powerplant
            if i != self.ip_plant.index :
                self.distances_to_plant[i]=np.inf
        
        self.predecessors=defaultdict(set) #dictionnaire des prédécesseurs pour chaque point d'intersection







    @property
    def ip_plant(self) :
        return self._ip_plant
    
    @ip_plant.setter
    def ip_plant(self, new_ip) :
        self._ip_plant=new_ip
    
    

    def dijkstra(self):
        pq = [(0.0, self.ip_plant.index)]
        self.distances_to_plant[self.ip_plant.index] = 0.0

        while pq:
            current_dist, ip_index = heapq.heappop(pq)
            ip = self.liste_intersection_points[ip_index]

            if (current_dist + ip.pipe > self.distances_to_plant[ip.index] and ip.index != self.ip_plant.index):
                continue

            if (current_dist + ip.pipe == self.distances_to_plant[ip.index] and len(self.voisins[ip.index]) == 1 ):
                continue  #eviter les cycles dû à la gestion du pipe
            
            for v_idx in self.voisins[ip.index]:

                if len(self.voisins[v_idx]) == 0 : #certaines routes sont isolées, on les ignore
                    continue
                
                v=self.liste_intersection_points[v_idx]

                d = current_dist + v.pipe + v.calcul_distance(ip)

                if d < self.distances_to_plant[v.index] and self.predecessors[v.index] != {ip.index}:#éviter les cycles infinis
                    self.distances_to_plant[v.index] = d
                    self.predecessors[v.index] = {ip.index}
                    heapq.heappush(pq, (d - v.pipe, v.index))

    

    

    def distances_b_to_plant(self):
        return {
            ip.index: self.distances_to_plant[ip.index]
            for ip in self.liste_intersection_points
            if isinstance(ip, NodeBuilding)
        }

    
    def building_ip(self, idx):
        for ip in self.liste_intersection_points:
            if isinstance(ip, NodeBuilding) and ip.index == idx:
                return ip
        return None


    def distance_dijkstra_to_network(self, idx_building, network_nodes_idx):
        """
        Longueur réelle du chemin Dijkstra à construire
        entre un bâtiment et le réseau existant.
        """

        length = 0.0
        current = idx_building
        visited=set()

        while current not in network_nodes_idx:
            
            if current in visited:
                print(f"Cycle détecté pour {idx_building}")
                print()
                return np.inf
            
            visited.add(current)

            preds = self.predecessors.get(current)
            if not preds:
                # bâtiment non atteignable depuis le réseau
                return np.inf

            # arbre Dijkstra → un seul prédécesseur effectif
            p = next(iter(preds))

            ip_current = self.liste_intersection_points[current]
            ip_pred = self.liste_intersection_points[p]

            length += ip_current.calcul_distance(ip_pred)
            current = p

        return length


    
    
    def connected_buildings(self, verbose=False):
        """
        Connexion basée sur la distance géométrique
        au nœud du réseau déjà construit le plus proche,
        avec mise à jour incrémentale du réseau.
        """

        connected = [self.ip_plant.index]
        #a_parcourir = [b for b in distances_buildings.keys() if b != self.ip_plant.index ]

        a_parcourir=[building.index for building in self.liste_nodes_buildings if building.index != self.ip_plant.index]

        heat_demand_total = self.ip_plant.heat_demand
        network_length = self.ip_plant.pipe  # longueur initiale (connexion du plant)
        added = True

        # Réseau utilisé (ensemble d'indices)
        network_nodes_idx = {self.ip_plant.index}
                            # Mise à jour incrémentale du réseau
        stack = [self.ip_plant.index]
        while stack:
            idx = stack.pop()
            if idx in network_nodes_idx:
                continue
            network_nodes_idx.add(idx)
            for p in self.predecessors.get(idx, []):
                stack.append(p)

        while added:
            added = False

            for idx_building in list(a_parcourir):

                building = self.building_ip(idx_building)

                d_to_network = self.distance_dijkstra_to_network(
                    idx_building, network_nodes_idx
                )
                if verbose :
                    print(f"Distance Dijkstra bâtiment {idx_building} au réseau : {d_to_network:.2f} m")
                if not np.isfinite(d_to_network):
                    continue



                ratio = (
                    heat_demand_total + building.heat_demand
                ) / (
                    network_length + d_to_network
                )

                if verbose:
                    print(
                        f"Bâtiment {idx_building} | "
                        f"Predecesseurs = {self.predecessors.get(idx_building, set())} | "
                        f"d_reseau = {d_to_network:.2f} m | "
                        f"ratio = {ratio:.2f}"
                    )

                if ratio > 1.5: #MWh/ml/an
                    # Connexion acceptée
                    connected.append(idx_building)
                    a_parcourir.remove(idx_building)

                    heat_demand_total += building.heat_demand
                    network_length += d_to_network
                    added = True

                    # Mise à jour incrémentale du réseau
                    stack = [idx_building]
                    while stack:
                        idx = stack.pop()
                        if idx in network_nodes_idx:
                            continue
                        network_nodes_idx.add(idx)
                        for p in self.predecessors.get(idx, []):
                            stack.append(p)

                    if verbose:
                        print(f"→ Bâtiment {idx_building} connecté.")

        return connected, network_length, network_nodes_idx, heat_demand_total

def tracer_graphe_optimal(commune, result) :
    #Tracé le graphe optimal trouvé
    G, network_nodes_idx, connected, liste_nodes_buildings, filtered_lines = result["graph"],result["network_nodes"], result["connected_buildings"], result["liste_nodes_buildings"], result["filtered_lines"]

    network_nodes= [ip for ip in G.liste_intersection_points if ip.index in network_nodes_idx]
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    # Tracé des routes
    lc = LineCollection(filtered_lines, colors='lightgrey', linewidths=0.8)
    ax.add_collection(lc)   
    # Tracé des nœuds du réseau utilisé
    for ip in network_nodes:
        x, y = ip.center

        if isinstance(ip, NodeRoad):
            plt.scatter(x, y, color='green', s=10)
        elif isinstance(ip, NodeBuilding) and ip.index in connected:
            plt.scatter(x, y, color='orange', s=10)

    #relier optimalement les noeuds utilisés entre eux 
    for ip in network_nodes:
        for pred_idx in G.predecessors.get(ip.index, []):
            pred_ip = next((n for n in network_nodes if n.index == pred_idx), None)
            if pred_ip is not None:
                x_values = [ip.center[0], pred_ip.center[0]]
                y_values = [ip.center[1], pred_ip.center[1]]
                plt.plot(x_values, y_values, color='red', linewidth=1)


    # Tracé des bâtiments
    centroids_x = [b.coordinates[0][0] for b in liste_nodes_buildings]
    centroids_y = [b.coordinates[0][1] for b in liste_nodes_buildings]
    plt.scatter(centroids_x, centroids_y, color='blue', s=5, label='Bâtiments')

    #afficher le batiment d'index 2 en purple
    building_2 = liste_nodes_buildings[2].coordinates[0]
    plt.scatter(building_2[0], building_2[1], color='purple', s=40, label='Bâtiment d\'index 2')

    #afficher le batiment d'index 3 en cyan
    building_3 = liste_nodes_buildings[3].coordinates[0]
    plt.scatter(building_3[0], building_3[1], color='cyan', s=40, label='Bâtiment d\'index 3')

    #afficher le batiment d'index 4 en magenta
    building_4 = liste_nodes_buildings[4].coordinates[0]
    plt.scatter(building_4[0], building_4[1], color='magenta', s=40, label='Bâtiment d\'index 4')

    #afficher le batiment d'index 5 en yellow
    building_5 = liste_nodes_buildings[5].coordinates[0]
    plt.scatter(building_5[0], building_5[1], color='yellow', s=40, label='Bâtiment d\'index 5')

    #afficher le batiment d'index 6 en orange
    building_6 = liste_nodes_buildings[6].coordinates[0]
    plt.scatter(building_6[0], building_6[1], color='orange', s=40, label='Bâtiment d\'index 6')

    #afficher le batiment d'index 7 en brown
    building_7 = liste_nodes_buildings[7].coordinates[0]
    plt.scatter(building_7[0], building_7[1], color='brown', s=40, label='Bâtiment d\'index 7')

    #afficher le batiment d'index 8 en pink
    building_8 = liste_nodes_buildings[8].coordinates[0]
    plt.scatter(building_8[0], building_8[1], color='pink', s=40, label='Bâtiment d\'index 8')

    #Plant
    plant = liste_nodes_buildings[0].coordinates[0]
    plt.scatter(plant[0], plant[1], color='red', s=40, label='Plant')

    #Limites strictes sur la zone des bâtiments
    all_x = centroids_x
    all_y = centroids_y 
    if all_x:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    plt.legend()
    plt.title(f"Réseau optimal trouvé pour la commune {commune}")
    plt.axis("equal")
    plt.show()

def tracer_graphe_optimal_zoom(commune, result) : 
    G, network_nodes_idx, connected, liste_nodes_buildings, filtered_lines = result["graph"], \
    result["network_nodes"], \
    result["connected_buildings"], \
    result["liste_nodes_buildings"], \
    result["filtered_lines"]

    network_nodes= [ip for ip in G.liste_intersection_points if ip.index in network_nodes_idx]
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    red_segments = []

    for ip in network_nodes:
        for pred_idx in G.predecessors.get(ip.index, []):
            pred_ip = next((n for n in network_nodes if n.index == pred_idx), None)
            if pred_ip is not None:
                red_segments.append([
                    (ip.center[0], ip.center[1]),
                    (pred_ip.center[0], pred_ip.center[1])
                ])


    if not red_segments:
        raise ValueError("Aucun segment rouge trouvé")

    all_x = [pt[0] for seg in red_segments for pt in seg]
    all_y = [pt[1] for seg in red_segments for pt in seg]

    margin = 5  # unités CRS
    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin

    #bloquer autoscale
    ax.set_autoscale_on(False)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    lc = LineCollection(filtered_lines, colors="lightgrey", linewidths=0.8, label = "Réseau routier")
    ax.add_collection(lc)

    red_lc = LineCollection(red_segments, colors="red", linewidths=2, label = "Réseau optimal")
    ax.add_collection(red_lc)

    for ip in network_nodes:
        x, y = ip.center
        if isinstance(ip, NodeRoad):
            ax.scatter(x, y, color="green", s=10, zorder=3, label='Noeud route')
        elif isinstance(ip, NodeBuilding) and ip.index in connected:
            ax.scatter(x, y, color="orange", s=10, zorder=3, label='Building projeté connecté')


    plant = liste_nodes_buildings[0].building_coordinates[0]
    ax.scatter(plant[0], plant[1], color="red", s=40, zorder=4, label="Plant")

    #tracé des batiments
    centroids_x = [b.building_coordinates[0][0] for b in liste_nodes_buildings]
    centroids_y = [b.building_coordinates[0][1] for b in liste_nodes_buildings]
    ax.scatter(centroids_x, centroids_y, color="blue", s=10, zorder=2, label="Bâtiments")

    #gestion des légendes sans doublons
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())   

    ax.set_aspect("equal")
    ax.set_title(f"Réseau optimal pour la commune {commune} – zoom strict sur le réseau utilisé")
    plt.show()



def total_length (G, network_nodes, liste_intersection_points, verbose=False) :
    #Calcul de la longueur totale du réseau
    total_length = 0.0
    for ip in network_nodes:
        for pred_idx in G.predecessors.get(ip.index, []):
            pred_ip = next((n for n in liste_intersection_points if n.index == pred_idx), None)
            if pred_ip is not None:
                total_length += ip.calcul_distance(pred_ip)

    
    if verbose:
        print(f"Longueur totale du réseau : {total_length:.2f} m.")
    return total_length

def total_heating_coverage(G, connected, liste_nodes_buildings, total_length, verbose=False) :
    #Calcul de la demande totale connectée
    total_demand = 0.0
    for idx in connected:
        building_node = G.building_ip(idx)
        if building_node is not None:
            total_demand += building_node.heat_demand

    if verbose:
        print(f"Demande énergétique totale connectée : {total_demand:.2f} MWh/an.")

    #Calcul du ratio demande/longueur
    if total_length > 0:
        ratio = total_demand / total_length  # MWh/m/an
        if verbose:
            print(f"Ratio demande/longueur : {ratio:.4f} MWh/m/an.")

    #Calcul du pourcentage de la demande énergétique couverte
    total_possible_demand = sum(nb.heat_demand for nb in liste_nodes_buildings)
    if total_possible_demand > 0:
        coverage_percentage = (total_demand / total_possible_demand) * 100
        if verbose:
            print(f"Pourcentage de la demande énergétique couverte : {coverage_percentage:.2f}%.")
    
    return total_demand

@dataclass
class CommuneOutcome:
    status: str  # "selected" | "rejected"
    commune: str
    stade: str
    reason: str | None
    nb_buildings: int
    nb_buildings_total : int
    result: dict | None

def process_commune(
    buildings_by_commune,
    commune,
    df_selected_roads,
    NB_MIN=2, #Au moins deux bâtiments pour faire un réseau
    NB_SEP=10, #Seuil pour différencier petit et grand réseau
    NB_TESTS=4 #Nombre de bâtiments testés comme emplacement de plant
) -> CommuneOutcome:

    df_selected_buildings_interm = buildings_by_commune.get(
        commune, pd.DataFrame()
    )

    if len(df_selected_buildings_interm) < NB_SEP:
        print(f"Commune {commune} rejetée au stade 1 : {len(df_selected_buildings_interm)} bâtiments")
        return CommuneOutcome(
            "rejected", commune, "1",
            f"Stade 1 : {len(df_selected_buildings_interm)} bâtiments",
            len(df_selected_buildings_interm), len(df_selected_buildings_interm), None
        )

    df_selected_buildings = transformer_coordonnees(
        df_selected_buildings_interm
    )
    df_buildings_sorted = sort_and_cut_buildings_by_consumption(
        df_selected_buildings
    )

    if len(df_buildings_sorted) < NB_SEP: #On enlève ici les toutes petites communes avec moins de NB_SEP bâtiments
        print(f"Commune {commune} rejetée au stade 2 : {len(df_buildings_sorted)} bâtiments")
        return CommuneOutcome(
            "rejected", commune, "2",
            f"Stade 2 : {len(df_buildings_sorted)} bâtiments",
            len(df_buildings_sorted), len(df_buildings_sorted), None
        )

    df_buildings_sorted_final, filtered_lines = selection_geom_batiments_routes(
        df_buildings_sorted, df_selected_roads
    )

    liste_nodes_buildings = create_liste_nodes_buildings(
        df_buildings_sorted_final
    )
    liste_roads = create_roads(filtered_lines)

    road_intersections, liste_intersection_points = (
        intersection_points_and_roads(
            liste_roads, liste_nodes_buildings
        )
    )

    road_intersections = sort_road_intersections(
        road_intersections,
        liste_intersection_points,
        liste_roads
    )

    index_maps = find_index_maps(road_intersections)
    voisins = find_voisins(
        road_intersections,
        liste_intersection_points,
        index_maps
    )


    #best_nb_buildings_connected=0
    best_idx_plant=0
    best_head_demand=0.0

    max_tests = min(NB_TESTS, len(liste_nodes_buildings))

    for idx_plant in range(max_tests) :
        G = Graph(
            liste_nodes_buildings,
            road_intersections,
            liste_intersection_points,
            voisins,
            idx_ip_plant=idx_plant
        )
        G.dijkstra()

        connected, network_length, network_nodes, heat_demand = (
            G.connected_buildings(verbose=False)
        )

        if heat_demand > best_head_demand : #on prend le bâtiment suivant que si le réseau est strictement meilleur en demande couverte 
            best_idx_plant=idx_plant
            best_head_demand=heat_demand
        
    G = Graph(
            liste_nodes_buildings,
            road_intersections,
            liste_intersection_points,
            voisins,
            idx_ip_plant=best_idx_plant
        )
    G.dijkstra()    
    connected, network_length, network_nodes, heat_demand = (
            G.connected_buildings(verbose=False)
        )


    if len(connected)<NB_MIN :
        print(f"Commune {commune} rejetée au stade 3 : {len(connected)} bâtiment connecté avec meilleur idx plant {best_idx_plant}")
        return CommuneOutcome(
        "rejected", commune, "3",
        f"Stade 3: {len(connected)} bâtiment connecté avec meilleur idx_plant {best_idx_plant}",
        len(connected), len(liste_nodes_buildings), None
    )
    result = {
        "graph": G,
        "connected_buildings": connected,
        "network_nodes": network_nodes,
        "filtered_lines": filtered_lines,
        "liste_nodes_buildings": liste_nodes_buildings,
        "network_length_km": network_length / 1000,
        "total_heating_coverage_MWh_per_year": heat_demand,
        "density_MWh_per_m_per_year": heat_demand / network_length,
        "idx_plant": G.ip_plant.buildings_idx,
        "heat_coverage_pct": (
            heat_demand
            / df_buildings_sorted_final["conso_ref_tot_MWh"].sum()
        ) * 100,
    }
    if len(connected) < NB_SEP:
        print(f"Commune {commune} avec un très petit réseau  : {len(connected)} bâtiments connectés avec meilleur idx plant {best_idx_plant}")
        return CommuneOutcome(
        "small_network", commune, "4",
        f"Stade 4 : {len(connected)} bâtiments connectés avec meilleur idx_plant {best_idx_plant}",
        len(connected), len(liste_nodes_buildings), result
    )   
    print(f"Commune {commune} sélectionnée : {len(connected)} bâtiments connectés")
    return CommuneOutcome(
        "big_network", commune, "5",
        None, len(connected), len(liste_nodes_buildings), result
    )
