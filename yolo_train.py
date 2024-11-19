from ultralytics import YOLO
import yaml

model = YOLO('yolov8n.pt')  

# Percorso del tuo file YAML
yaml_path = '/work/grana_pbl/Detection_Glomeruli/yolo.yaml'

# Carica il file YAML
with open(yaml_path, 'r') as file:
    yolo_yaml = yaml.safe_load(file)

# Stampa il dizionario
print(yolo_yaml)

model.train(data='/work/grana_pbl/Detection_Glomeruli/yolo.yaml', epochs=80, imgsz=1024)



