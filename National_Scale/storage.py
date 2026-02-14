import os
import json
import pickle
import pandas as pd
from datetime import datetime   


def init_department_storage(base_dir, departement):
    dep_dir = os.path.join(base_dir, f"dep_{departement}")
    os.makedirs(dep_dir, exist_ok=True)
    os.makedirs(os.path.join(dep_dir, "communes"), exist_ok=True)
    os.makedirs(os.path.join(dep_dir, "small_networks"), exist_ok=True)

    rejected_csv = os.path.join(dep_dir, "rejected_communes.csv")
    selected_csv = os.path.join(dep_dir, "selected_communes.csv")
    small_networks_csv = os.path.join(dep_dir, "small_networks.csv")
    meta_json = os.path.join(dep_dir, "meta.json")

    if not os.path.exists(rejected_csv):
        pd.DataFrame(columns=[
            "departement", "commune", "stade",
            "reason", "nb_buildings", "nb_buildings_total"
        ]).to_csv(rejected_csv, index=False)

    if not os.path.exists(selected_csv):
        pd.DataFrame(columns=[
            "departement", "commune",
            "nb_connected",
            "network_length_km",
            "heat_demand_MWh",
            "heat_coverage_pct",
            "density_MWh_per_m_per_year",
            "idx_plant"
        ]).to_csv(selected_csv, index=False)
    
    if not os.path.exists(small_networks_csv):
        pd.DataFrame(columns=[
            "departement", "commune",
            "nb_connected",
            "network_length_km",
            "heat_demand_MWh",
            "heat_coverage_pct",
            "density_MWh_per_m_per_year",
            "idx_plant"
        ]).to_csv(small_networks_csv, index=False)

    if not os.path.exists(meta_json):
        meta = {
            "departement": departement,
            "date_run": datetime.now().isoformat(),
            "nb_communes_total": 0,
            "nb_selected_final": 0,
            "nb_small_networks": 0,
            "nb_rejected": 0
        }
        with open(meta_json, "w") as f:
            json.dump(meta, f, indent=2)

    return dep_dir

def save_rejected_commune(dep_dir, departement, commune, stade, reason, nb_buildings, nb_buildings_total):
    path = os.path.join(dep_dir, "rejected_communes.csv")
    df = pd.DataFrame([{
        "departement": departement,
        "commune": commune,
        "stade": stade,
        "reason": reason,
        "nb_buildings": nb_buildings,
        "nb_buildings_total": nb_buildings_total    
    }])
    df.to_csv(path, mode="a", header=False, index=False)

def save_selected_commune(dep_dir, departement, commune, result):
    # CSV résumé
    path = os.path.join(dep_dir, "selected_communes.csv")

    row = {
        "departement": departement,
        "commune": commune,
        "nb_connected": len(result["connected_buildings"]),
        "network_length_km": result["network_length_km"],
        "heat_demand_MWh": result["total_heating_coverage_MWh_per_year"],
        "heat_coverage_pct": result["heat_coverage_pct"],
        "density_MWh_per_m_per_year": result["density_MWh_per_m_per_year"],
        "idx_plant": result["idx_plant"]
    }

    pd.DataFrame([row]).to_csv(
        path, mode="a", header=False, index=False
    )

    # Pickle complet
    pickle_path = os.path.join(
        dep_dir, "communes", f"{commune}.pkl"
    )
    with open(pickle_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_small_network_commune(dep_dir, departement, commune, result):  
    # CSV résumé
    path = os.path.join(dep_dir, "small_networks.csv")

    row = {
        "departement": departement,
        "commune": commune,
        "nb_connected": len(result["connected_buildings"]),
        "network_length_km": result["network_length_km"],
        "heat_demand_MWh": result["total_heating_coverage_MWh_per_year"],
        "heat_coverage_pct": result["heat_coverage_pct"],
        "density_MWh_per_m_per_year": result["density_MWh_per_m_per_year"],
        "idx_plant": result["idx_plant"]
    }

    pd.DataFrame([row]).to_csv(
        path, mode="a", header=False, index=False
    )
    
    pickle_path = os.path.join(
        dep_dir, "small_networks", f"{commune}.pkl"
    )
    with open(pickle_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

def update_meta(dep_dir, status):
    meta_path = os.path.join(dep_dir, "meta.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    meta["nb_communes_total"] += 1

    if status == "big_network":
        meta["nb_selected_final"] += 1

    elif status == "small_network":
        meta["nb_small_networks"] += 1
        meta["nb_selected_final"] += 1  

    elif status == "rejected":
        meta["nb_rejected"] += 1

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

