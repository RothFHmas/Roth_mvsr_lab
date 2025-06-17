from ultralytics import YOLO
import cv2
import json
from pathlib import Path

def run_yolo_and_save_bboxes():
    model = YOLO("yolo_model/my_model.pt")

    image_dir = Path("data/rgb")
    output_dir = Path("data/yolo_detections")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Klassen-Namen direkt aus dem Modell auslesen
    id2label = model.names  # Liste oder Dict mit Index -> Label

    for i in range(10):
        img_path = image_dir / f"{i}.png"
        if not img_path.exists():
            continue

        results = model(img_path)

        # Optional: Bild mit Bounding Boxes speichern
        annotated = results[0].plot()
        cv2.imwrite(str(output_dir / f"{i}_detected.png"), annotated)

        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        classes = results[0].boxes.cls.cpu().numpy().tolist()
        confidences = results[0].boxes.conf.cpu().numpy().tolist()

        bboxes_for_image = []

        for box, cls, conf in zip(boxes, classes, confidences):
            if conf < 0.4:
                continue

            xmin, ymin, xmax, ymax = map(int, box)
            label = id2label[int(cls)] if int(cls) in id2label else "unknown"

            bboxes_for_image.append({
                "label": label,
                "bbox_modal": [xmin, ymin, xmax, ymax]
            })

        # Speichere pro Bild eine JSON-Datei
        output_file = output_dir / f"{i}.json"
        with open(output_file, "w") as f:
            json.dump(bboxes_for_image, f, indent=2)

    print("[INFO] Bounding boxes im gewünschten Format gespeichert.")

if __name__ == "__main__":
    run_yolo_and_save_bboxes()