from ultralytics import YOLO
import cv2
import pandas as pd
from PIL import Image
import os
import json
import numpy as np

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2
    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    iou = inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
    return iou

# Funzione per disegnare le bounding box
def draw_bounding_boxes(image, boxes, labels, confidences, color=(0, 255, 0)):
    for box, label, confidence in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{label}: {confidence:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def test_yolo (model_path, data_yaml):

    model = YOLO(model_path)

    # 3. Esegui il predict per salvare le immagini annotate
    model.val(
        data=data_yaml,   # Usa il set di test
        save=True,      # Salva le immagini con le predizioni
        save_txt=True   # Salva le predizioni in formato txt
    )

    # 4. Calcola le metriche sul set di test
    # metrics = model.val(
    #     data=data_yaml,
    #     split='test',   # Usa il set di test
    #     save=True,      # Salva le immagini con le predizioni
    #     save_txt=True   # Salva le predizioni in formato txt
    # )

   # return metrics  # Ritorna le metriche per l'analisi

# YoloV5 ha già implementato questo metodo mentre YoloV8 no
def get_pandas(results):
  # translate boxes data from a Tensor to the List of boxes info lists
  boxes_list = results[0].boxes.data.tolist()
  columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id']

  # iterate through the list of boxes info and make some formatting
  for i in boxes_list:
    # round float xyxy coordinates:
    i[:4] = [round(i, 1) for i in i[:4]]
    # translate float class_id to an integer
    i[5] = int(i[5])
    # add a class name as a last element
    i.append(results[0].names[i[5]])

  # create the result dataframe
  columns.append('class_name')
  result_df = pd.DataFrame(boxes_list, columns=columns)

  return result_df


def save_predicted_patches(model_path, folder_path, labels_path, save_dir, saved_results_json_file):
        
        results_txt_path = os.path.join(save_dir, 'results.txt')
        # Creare un insieme per raccogliere le ground truth uniche
        unique_total_gt_boxes = set()
        # Inizializza il conteggio delle bounding boxes valide
        total_valid_boxes_03 = 0
        total_valid_boxes_05 = 0
        # Numero totale di bounding box di ground_truth
        total_ground_truth_boxes = 0

        all_results = []

        model = YOLO(model_path)
        # Itera sulle immagini della cartella di validazione
        for image_file in sorted(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, image_file)
            label_file = os.path.join(labels_path, image_file.replace(".png", ".txt"))
            
            # Apri l'immagine
            image = Image.open(image_path)
            
            # Carica le predizioni del modello
            results = model.predict(image)

            dataframe = get_pandas(results)

            bounding_boxes = dataframe.to_dict(orient="records")

            all_results.append({

            "nome_wsi" : image_file,
            "bounding_boxes" : bounding_boxes,

            })

            # Carica le annotazioni originali (ground truth)
            gt_boxes = []
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        _, x_center, y_center, width, height = map(float, line.strip().split())
                        img_width, img_height = image.size
                        x1 = int((x_center - width / 2) * img_width)
                        y1 = int((y_center - height / 2) * img_height)
                        x2 = int((x_center + width / 2) * img_width)
                        y2 = int((y_center + height / 2) * img_height)
                        gt_boxes.append([x1, y1, x2, y2])

            pred_boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []
            confidences = results[0].boxes.conf.cpu().numpy() if len(results) > 0 else []
            
            annotated_image = np.array(image)
            if len(pred_boxes) > 0:
                annotated_image = draw_bounding_boxes(annotated_image, pred_boxes, ['Pred'] * len(pred_boxes), confidences)
            if len(gt_boxes) > 0:
                annotated_image = draw_bounding_boxes(annotated_image, gt_boxes, ['GT'] * len(gt_boxes), [1.0] * len(gt_boxes), color=(0, 0, 255))  # Rosso per GT

            # Salva l'immagine annotata se ci sono predizioni

            if len(pred_boxes) > 0:
                save_path = os.path.join(save_dir, f'annotated_{image_file}')
                # Decommenta se vuoi salvare le immagini con le bounding boxes rilevate
                # Image.fromarray(annotated_image).save(save_path)

            # Filtra le predizioni in base alla confidenza
            filtered_pred_boxes = pred_boxes[confidences > 0.3]

            unique_img_boxes = set()
            # Aggiungi solo le bounding box uniche di ground truth all'insieme
            for box in gt_boxes:
                box_tuple = tuple(box)  # Converti in tupla per usare nel set
                unique_total_gt_boxes.add(box_tuple)
                unique_img_boxes.add(box_tuple)

            # Calcola il numero di bounding box valide uniche
            for pred_box in filtered_pred_boxes:
                # Usa le predizioni filtrate
                for gt_box in unique_img_boxes:
                    # Incremento il numero di bounding box di ground truth in totale 
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > 0.3:  # Modificato a 0.5 se vuoi una soglia di 0.5
                        total_valid_boxes_03 += 1
                    if iou > 0.5:
                        total_valid_boxes_05 += 1

                        print('Questo è il numero parziale di bounding box a soglia 0.3 valide al quale sono arrivato : ', total_valid_boxes_03)
                        print('Questo è il numero parziale di bounding box a soglia 0.5 valide al quale sono arrivato : ', total_valid_boxes_05)
                        break  # Esci dal ciclo, una volta trovato un match

        # Scrivi i risultati nel file di testo
        with open(results_txt_path, 'a') as f:
            f.write(f'Totale bounding boxes valide con confidenza > 0.3 e IoU con il ground truth > 0.3: {total_valid_boxes_03}\n')
            f.write(f'Totale bounding boxes valide con confidenza > 0.5 e IoU con il ground truth > 0.5: {total_valid_boxes_05}\n')
            f.write(f'Totali bounding box uniche di ground_truth: {len(unique_total_gt_boxes)}\n')
            print('Numero di bb GT: ', len(unique_total_gt_boxes))
          
        # Salva tutti i risultati delle bounding box predette in un file JSON
        with open(saved_results_json_file, 'w') as f:
            json.dump(all_results, f, indent=4)

        print('END!')

if __name__ == '__main__':

    # Percorsi dei file
    folder_path = "/work/grana_pbl/Detection_Glomeruli/Yolo_dataset_Lv1_Overlapping05/images/val"
    labels_path = "/work/grana_pbl/Detection_Glomeruli/Yolo_dataset_Lv1_Overlapping05/labels/val"
    save_dir = '/work/grana_pbl/Detection_Glomeruli/Yolo_results/Yolo_results_Lv1_Overlapping05/annotated_patches'
    # Load weights
    model_path = '/work/grana_pbl/Detection_Glomeruli/runs/detect/train_overlapping_05/weights/best.pt'
    data_yaml = '/work/grana_pbl/Detection_Glomeruli/yolo.yaml'
    saved_results_json_file = "/work/grana_pbl/Detection_Glomeruli/Yolo_results/Yolo_results_Lv1_Overlapping05/all_predictions.json"
    
    save_predicted_patches(model_path, folder_path, labels_path, save_dir, saved_results_json_file)
    
    #test_yolo(model_path, data_yaml)



