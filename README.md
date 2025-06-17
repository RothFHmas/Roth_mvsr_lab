# mvsr_lab_Roth

In diesem Projekt wird mithilfe von **YOLOv11-s** und **Megapose6D** eine **6D-Pose-Estimation** durchgeführt.

Zunächst werden mithilfe des YOLO-Modells Objekte in den zu untersuchenden Bildern detektiert.  
Anhand dieser Informationen, den RGBD-Bildern sowie den CAD-Modellen der zu erkennenden Objekte bestimmt **Megapose6D** anschließend die 6D-Posen der Objekte im Raum.

Am Ende des Prozesses werden die berechneten Posen visualisiert, z. B. durch Koordinatenachsen und Bounding-Boxen im Bild.

Dieses Projekt wurde durchgeführt auf einem system mit:<br>
OS:  Ubuntu 20.04.6 LTS x86_64 (nativ installiert)<br>
GPU: NVIDIA GeForce RTX 3050  (CUDA Version `CUDA Version: 12.8` (entnommen aus `nvidia-smi`))<br>
CPU: AMD Ryzen 5 5600X (12) @ 3.700GHz<br>

Ich verwende hier miniconda (conda 25.5.1)
Das system ist komplett neu aufgesetzt, wenn irgendwelche requirements fehlen weiß ich nicht warum da ich hier paralel mitgeschrieben habe was ich gemacht habe. unter `utility_scrips` ist eine `Beispiel_erfolgreiche_ausfürhung_in_console.txt` dort wurde einal die komplette konsolen ausgabe hin kopiert für die erfolgreiche ausführung von `main.py`.

## 1. Git-Repository klonen
Zu erst mein Git-Repository klonen.
```bash
git clone https://github.com/RothFHmas/mvsr_lab_roth.git
```
Ich habe es nicht geschafft, wegen mangelnder Git-Erfahrung das Megapose-Submodule richtig einzurichten.
Deswegen bitte Megapose noch einmal extra clonen:
```bash
cd mvsr_lab_roth/
git clone https://github.com/megapose6d/megapose6d.git
```

## 2. Conda Environment erstellen
Ich habe zunächst das Conda-Environment erstellt, in dem das Pose-Estimation-Programm läuft.  
Hierfür habe ich der Anleitung des Megapose6D-Git-Repositories gefolgt. Doch zuvor das Setup ausführen:
```bash
python setup.py
```
Das setup ersetzt im megapose Git-Repository die benötigten datein mit bearbeiteten versionen aus dem ordner utility_scrips. In der enviroment_full.yaml wir nur die notebook version auf 6.4.12 gesetzt, und in der run on exampel wird nur das -icp ergänzt um rgbd daten zu verwenden. Ich verwende hier miniconda (conda 25.5.1).
```bash
cd mvsr_lab_roth
cd megapose6d
conda env create -f conda/environment_full.yaml
conda activate megapose
cd megapose6d
pip install -e .
cd ..
```

## 2.1 Requirements
Ich musste anschließend die für mich passende Torch-Version installieren.
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Außerdem habe ich vorher grafik treiber updates gemacht weiß nciht ob das einen einflauss hat.

Außerdem muss dises bop_toolkit installiert werden falls der error erschint.
```bash
pip install git+https://github.com/thodan/bop_toolkit.git
```


## 2.2 Ultralytics installieren
Als Nächstes habe ich Ultralytics installiert, um das trainierte YOLO-Modell zu verwenden  
(darauf wird später eingegangen).
```bash
pip install ultralytics
```

## 3. Beispieldaten herunterladen und setup ausführen
Ich habe das vortrainierte Modell von Megapose6D heruntergeladen:
```bash
cd megapose6d
python -m megapose.scripts.download --megapose_models
cd ..
```
Und hier die Example-Daten von Megapose6D (optional):
```bash
cd megapose6d
python -m megapose.scripts.download --example_data
cd ..
```


## 4. Hauptprogramm starten
Starten des Hauptprogramm
```bash
python main.py
```
Im `main.py` wird zuallererst dieses Programm aufgerufen:
```bash
python generate_bboxes_with_yolo.py
```
In diesem Programm habe ich das von mir trainierte YOLO-Modell (`yolo11-s`) auf die Bilder im Ordner `data/rgb` angewendet. Das Ergebnis davon sind die Bilder und Bounding-Box-Daten im Ordner `data/yolo_detections`.

Anschließend habe ich die benötigten Bilder (RGB und Depth) sowie die Kamera-Kalibrierungsdaten und Modellmeshes im `.ply`-Format in den Ordner `megapose6d/local_data/examples/morobot` kopiert.  
Das habe ich getan, um die in Megapose6D vorhandenen Example-Programme zu verwenden. Diese Entscheidung habe ich getroffen, da ich den Megapose6D-Code so wenig wie möglich bearbeiten wollte.  
Nachdem alle Daten dort waren, wo sie sein sollten, habe ich diese Befehle ausgeführt:

```bash
python -m megapose.scripts.run_inference_on_example morobot --vis-detections
python -m megapose.scripts.run_inference_on_example morobot --run-inference
python -m megapose.scripts.run_inference_on_example morobot --vis-outputs
python draw_pose.py
```
Die ersten drei Befehle sind die Example-Programme aus Megapose6D, um  
1. die Bounding-Boxen erneut zu visualisieren (nur zu Fehlerüberprüfungszwecken),  
2. Inferenz auf die RGB- und Tiefenbilder anzuwenden (hierbei habe ich im Megapose-Code angepasst, dass das Modell `megapose-1.0-RGB-multi-hypothesis-icp` verwendet wird anstatt `megapose-1.0-RGB-multi-hypothesis`. Der Unterschied ist, dass das von mir gewählte Modell auch Tiefendaten mit verarbeitet),  
3. die Ergebnisse, die Megapose6D erzeugt, zu visualisieren (Konturen und Segmentierungsmaske),  
4. Der letzte Code ist dafür da, ein Bild zu erstellen, in welchem die Posen in Form eines Koordinatenkreuzes und einer Bounding-Box dargestellt werden. Dieser Code ist stark inspiriert von diesem Issues-Post: https://github.com/megapose6d/megapose6d/issues/52, wurde aber angepasst für meine Zwecke.

Danach habe ich im Hauptprogramm alle Ergebnisse in den Ordner `results/` kopiert, entsprechend umbenannt und in Ordner einsortiert.  
Sobald dies geschehen war, war der Durchlauf für das erste Bild abgeschlossen (`0.png`) und dieser Ablauf wiederholte sich in einem `for`-Loop für alle Bilder (`0.png` bis `9.png`) aus der MOODLE-Angabe.

# 5. YOLO-Modell trainieren

Das YOLO-Modell habe ich nach der Anleitung aus diesem YouTube-Video trainiert: https://youtu.be/r0RspiLG260  
In diesem wird sehr gut das Labeln von Daten erklärt und wie das Dataset zum Trainieren auszusehen hat.

Die Labeldaten für das Dataset habe ich mit Roboflow erstellt: https://roboflow.com/  
Hierbei habe ich lediglich Bounding-Boxen mit Klassenlabels auf den Bildern eingefügt.  
Dabei möchte ich erwähnen, dass ich für das Training der Daten dankenswerterweise etwa 50 ungelabelte Bilder von Kollegen Karim El-Harery zur Verfügung gestellt bekommen habe.  
Diese gelabelten Bilder habe ich dann anschließend in Roboflow noch augmentiert, um zusammen mit den in MOODLE zur Verfügung gestellten Bildern 183 Bilder für das Training des YOLO-Modells zu erhalten.

Dieses Google Colab habe ich dann verwendet, um YOLO zu trainieren:  
https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb  
Ich habe mich für ein `yolo11-s` entschieden, da es nach mehrfachen Tests bessere Ergebnisse als `yolo11-n` und `yolo11-m` erzielt hat.

Um die Bounding-Boxen nun auf den gewünschten Bildern zu erstellen, habe ich Code-Snippets von https://docs.ultralytics.com verwendet und erweitert durch Funktionen, um die Bounding-Box-Daten im `.json`-Format abzuspeichern und richtig vorzubereiten für Megapose.  
Dies geschieht in diesem Code:

```bash
python generate_bboxes_with_yolo.py
```
# 6. Utility Scrips

Im ordner `utility_scripts` vefinden sich 2 python scripte<br> das 1. ist dazu da aus den .ply datein der CAD-Modelle die größe der benötigten Bounding Box zu bestimmen.<br> das 2. ist dazu da gewesen testwiese eine pose und Bounding Box für die beispieldaten aus Megapose6d zu erstellen.

Diese Codes sind nicht wichtig für den betreib des programmes sind aber in diesem Git-Repository der vollständigkeit wegen.

# 7. Ergebnisse

Die Ergebnisse die ich erziehlt habe mit dem betrieb dieses von mir erstellten Git-Repository sind im ordner `results` die fertig visualisierten Bilder )Pose mit Bounding Box) sind in `results/pose`.

