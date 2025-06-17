import os
import shutil
import subprocess
def kopiere_und_umbenennen(src_pfad, ziel_ordner, neuer_name):
    os.makedirs(ziel_ordner, exist_ok=True)
    ziel_pfad = os.path.join(ziel_ordner, neuer_name)
    shutil.copy2(src_pfad, ziel_pfad)
    print(f"{src_pfad} -> {ziel_pfad}")

def ordner_kopieren(quellordner, zielordner):
    if os.path.exists(zielordner):
        shutil.rmtree(zielordner)
        print(f"Zielordner '{zielordner}' bereits vorhanden → wurde ersetzt.")
    shutil.copytree(quellordner, zielordner)
    print(f"Ordner kopiert: {quellordner} → {zielordner}")


kopiere_und_umbenennen("utility_scripts/environment_full.yaml","megapose6d/conda" ,"environment_full.yaml")
kopiere_und_umbenennen("utility_scripts/run_inference_on_example.py","megapose6d/src/megapose/scripts" ,"run_inference_on_example.py")

ordner_kopieren("data/morobot", "megapose6d/local_data/examples/morobot")
    