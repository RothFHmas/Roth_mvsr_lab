import os
import shutil
import subprocess
# === Definitionen ===
def kopiere_und_umbenennen(src_pfad, ziel_ordner, neuer_name):
    os.makedirs(ziel_ordner, exist_ok=True)
    ziel_pfad = os.path.join(ziel_ordner, neuer_name)
    shutil.copy2(src_pfad, ziel_pfad)
    print(f"{src_pfad} -> {ziel_pfad}")

def befehl_ausfuehren(befehl):
    print(f"Starte: {befehl}")
    try:
        subprocess.run(befehl, shell=True, check=True)
        print("Befehl abgeschlossen.")
    except subprocess.CalledProcessError as e:
        print(f"Fehler bei Befehl: {befehl}")
        print(e)

def ordner_kopieren(quellordner, zielordner):
    if os.path.exists(zielordner):
        shutil.rmtree(zielordner)
        print(f"Zielordner '{zielordner}' bereits vorhanden → wurde ersetzt.")
    shutil.copytree(quellordner, zielordner)
    print(f"Ordner kopiert: {quellordner} → {zielordner}")

# === Programm Anfang ===
# === YOLO-Bounding Boxes generieren ===
befehl_ausfuehren("python generate_bboxes_with_yolo.py")

# === Hauptschleife für Bilder 0 bis 9 ===
for i in range(10):
    print(f"\n Verarbeitung Bild {i}\n" + "="*30)

    nummer = str(i)

    # === Schritt 1: Daten kopieren ===
    kopiere_und_umbenennen(f"data/rgb/{nummer}.png", "megapose6d/local_data/examples/morobot", "image_rgb.png")
    kopiere_und_umbenennen(f"data/depth/{nummer}.png", "megapose6d/local_data/examples/morobot", "image_depth.png")
    kopiere_und_umbenennen(f"data/yolo_detections/{nummer}.json",  "megapose6d/local_data/examples/morobot/inputs", "object_data.json")
    kopiere_und_umbenennen("data/camera_data.json",  "megapose6d/local_data/examples/morobot", "camera_data.json")

    ordner_kopieren("data/morobot/morobot/meshes", "megapose6d/local_data/examples/morobot/meshes")

    # === Schritt 2: Megapose-Befehle ===
    befehle = [
        "python -m megapose.scripts.run_inference_on_example morobot --vis-detections",
        "python -m megapose.scripts.run_inference_on_example morobot --run-inference",
        "python -m megapose.scripts.run_inference_on_example morobot --vis-outputs",
        "python draw_pose.py"
    ]
    for befehl in befehle:
        befehl_ausfuehren(befehl)

    # === Schritt 3: Ergebnisse speichern ===
    kopiere_und_umbenennen("megapose6d/local_data/examples/morobot/visualizations/all_poses.png", "results/visualizations", f"{nummer}_poses.png")
    kopiere_und_umbenennen("megapose6d/local_data/examples/morobot/visualizations/all_results.png", "results/visualizations/megapose", f"{nummer}_all_results.png")
    kopiere_und_umbenennen("megapose6d/local_data/examples/morobot/visualizations/detections.png", "results/visualizations/detections/megapose", f"{nummer}_detections_megapose.png")
    kopiere_und_umbenennen("megapose6d/local_data/examples/morobot/outputs/object_data.json", "results/pose", f"{nummer}_poses.json")
    kopiere_und_umbenennen(f"data/yolo_detections/{nummer}_detected.png", "results/visualizations/detections/yolo", f"{nummer}_detections_yolo.png")
