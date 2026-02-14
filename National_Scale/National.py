import fonctions_utiles as fu
import time

from storage import (
    init_department_storage,
    save_rejected_commune,
    save_selected_commune,
    save_small_network_commune,
    update_meta
)

BASE_DIR ="/Users/louisedeferran/Documents/Louise/Mines/effynersys/reseau_chaleur/DHN_Potential_France_EFFINERSYS/National_Scale/Storage_results/"

departements=["2A","2B"]


for departement in departements:
    start_time = time.time()
    dep_dir = init_department_storage(BASE_DIR, departement)

    df_buildings_filtered, roads_df, buildings_by_commune = (
        fu.selection_fichiers_et_communes(departement)
    )

    df_selected_roads = fu.extraction_routes(roads_df)

    communes = sorted(
        df_buildings_filtered["code_commune_insee"].astype(str).unique(),
        key=lambda x: int(x)
    )
    #Pour départements 2A et 2B : pas de tri 
    #communes=df_buildings_filtered["code_commune_insee"].astype(str).unique()

    for commune in communes:
        try:
            outcome = fu.process_commune(
                buildings_by_commune,
                commune,
                df_selected_roads
            )

            if outcome.status == "small_network" or outcome.status == "big_network":
                if outcome.stade == "4":
                    # Réseau trop petit, enregistré dans small_networks.csv
                    save_small_network_commune(
                        dep_dir, departement, commune,
                        outcome.result
                    )
                    update_meta(dep_dir, status=outcome.status)
                else:
                    save_selected_commune(
                            dep_dir, departement, commune,
                            outcome.result
                        )
                    update_meta(dep_dir, status=outcome.status  )

            else:
                save_rejected_commune(
                    dep_dir, departement, commune,
                    outcome.stade,
                    outcome.reason,
                    outcome.nb_buildings,
                    outcome.nb_buildings_total
                )
                update_meta(
                    dep_dir,
                    status=outcome.status
                )

        except Exception as e:
            print(f"Erreur commune {commune}: {e}")
    end_time = time.time()
    print(f"Temps total de traitement du departement {departement}  : {(end_time - start_time)/60:.2f} minutes")






