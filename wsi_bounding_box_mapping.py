import os
import pandas as pd
import numpy as np
import json
import re
import yaml
import re
from collections import defaultdict
from shapely.geometry import Polygon, box
from openslide import open_slide
import yaml
from shapely.geometry import Polygon
import cv2

def merge_bounding_boxes(aggregated_bounding_boxes_path, output_final_complete_aggregated_wsi_path):

    """ 
    Se ho già generato tramite YOLO il file JSON 
    a partire da quel file JSON posso in teoria già controllare le unioni e intersezioni tra le bounding boxes
    salvandole in un nuovo file JSON delle unioni, fatto questo posso riportarle sulla WSI generale. 

    """

    # Prendo per esempio 'R22-117 C1q'
    with open(aggregated_bounding_boxes_path, 'r') as f:
        bb_data = json.load(f)

    # Core del merge ---------------
    final_boxes = {}

    for slide_name, bounding_boxes in bb_data.items():
        merged = False
        while True:  # Continua finché non ci sono più sovrapposizioni
            merged = False
            new_bounding_boxes = []  # Lista temporanea per le bounding box aggiornate
            skip_indices = set()  # Indici delle bounding box già unite

            for i in range(len(bounding_boxes)):
                if i in skip_indices:
                    continue
                bb1 = bounding_boxes[i]
                merged_with_another = False

                for j in range(i + 1, len(bounding_boxes)):
                    if j in skip_indices:
                        continue
                    bb2 = bounding_boxes[j]

                    if intersection_area(bb1, bb2) > 0:
                        new_bb_coords = unite_bounding_boxes(bb1, bb2)
                        new_bb = {
                            'x_min': int(new_bb_coords[0]),
                            'y_min': int(new_bb_coords[1]),
                            'x_max': int(new_bb_coords[2]),
                            'y_max': int(new_bb_coords[3]),
                            'confidence': None,
                            'class_id': 0,
                            'class_name': 'Glomerulo'
                        }
                        new_bounding_boxes.append(new_bb)
                        skip_indices.update({i, j})  # Salta bb1 e bb2 nelle iterazioni future
                        merged = True
                        merged_with_another = True
                        break  # Esci dal ciclo `j`

                if not merged_with_another:  # bb1 non è stata unita con altre
                    new_bounding_boxes.append(bb1)

            # Se nessuna unione è stata fatta, usciamo dal ciclo
            if not merged:
                break

            # Aggiorna la lista delle bounding box
            bounding_boxes = new_bounding_boxes

        # Salva le bounding box finali per questa slide
        final_boxes[slide_name] = bounding_boxes

    #-------------------------------
    # Vecchia versione che non mi funziona 
#     final_boxes = {}

#    # Scorri le bounding boxes per ciascuna slide
#     for slide_name, bounding_boxes in bb_data.items():
#         i = 0
#         while i < (len(bounding_boxes)):  # Continuiamo finché ci sono modifiche da fare
#             bb1 = bounding_boxes[i]
#             # Usa un ciclo for per scorrere tutte le bounding boxes successive
#             for j in range(i + 1, len(bounding_boxes)):
#                 bb2 = bounding_boxes[j]
#                 # Verifica se le bounding boxes si intersecano
#                 if intersection_area(bb1, bb2) > 0:
#                     # Unisci le bounding boxes
#                     new_bb = unite_bounding_boxes(bb1, bb2)
#                     # Rimuovi le bounding boxes originali dalla lista
#                     bounding_boxes.remove(bb1)
#                     bounding_boxes.remove(bb2)
#                     # Aggiungi la bounding box unita
#                     new_bb  = {
#                         'x_min': int(new_bb[0]),
#                         'y_min' : int(new_bb[1]),
#                         'x_max' : int(new_bb[2]),
#                         'y_max' : int(new_bb[3]),
#                         'confidence' : None,
#                         'class_id' : 0, 
#                         'class_name' : 'Glomerulo'
#                     }
#                     bounding_boxes.append(new_bb)
#                     # Riavvia l'iterazione per controllare altre intersezioni
#                     i = 0  # Riavvia l'indice principale
#                     break  # Uscita dal ciclo for interno
#             i += 1  # Incrementa l'indice per proseguire
#         final_boxes[slide_name] = bounding_boxes

    # Prendo per esempio 'R22-117 C1q'
    with open(output_final_complete_aggregated_wsi_path, 'a') as f:
        json.dump(final_boxes, f, indent=4)

def unite_bounding_boxes(box1, box2):
  
    x1_1, y1_1, x2_1, y2_1, confidence, class_id, class_name = box1['x_min'], box1['y_min'], box1['x_max'], box1['y_max'], box1['confidence'], box1['class_id'], box1['class_name']
    x1_2, y1_2, x2_2, y2_2, confidence, class_id, class_name = box2['x_min'], box2['y_min'], box2['x_max'], box2['y_max'], box2['confidence'], box2['class_id'], box2['class_name']

    # Calcola i limiti della bounding box unita
    x1_union = min(x1_1, x1_2)
    y1_union = min(y1_1, y1_2)
    x2_union = max(x2_1, x2_2)
    y2_union = max(y2_1, y2_2)

    return (x1_union, y1_union, x2_union, y2_union)

def intersection_area(box1, box2):

    x1_1, y1_1, x2_1, y2_1, confidence, class_id, class_name = box1['x_min'], box1['y_min'], box1['x_max'], box1['y_max'], box1['confidence'], box1['class_id'], box1['class_name']
    x1_2, y1_2, x2_2, y2_2, confidence, class_id, class_name = box2['x_min'], box2['y_min'], box2['x_max'], box2['y_max'], box2['confidence'], box2['class_id'], box2['class_name']

    # Calcola i limiti dell'intersezione
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # Se l'area di intersezione è positiva, calcola l'area
    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)

    intersection_area = intersection_width * intersection_height
    return intersection_area

def get_slide_element_at_level(slide, lvl):
        num_levels = slide.level_count
        # Controlla se il livello richiesto è valido
        while lvl >= 0:
            if lvl < num_levels:  # Se il livello esiste
                return slide.level_dimensions[lvl], lvl
            lvl -= 1  # Scendi al livello successivo
        return None  # Nessun livello valido trovato

def remap(wsi_path, wsi_info_file, wsi_level, aggregated_bounding_boxes_path, output_path, ground_truth_path, ground_truth_flag):

    """
    Con questa funzione voglio rimappare il file JSON finale sulla WSI.

    """
    # Prendo per esempio 'R22-117 C1q'
    with open(aggregated_bounding_boxes_path, 'r') as f:
        bb_data = json.load(f)

    slide_img = open_slide(wsi_path)
    slide = wsi_path.split('/')[-1]
    slide_name = os.path.splitext(slide)[0]
    slide_dims = slide_img.dimensions[wsi_level]

    # Scelgo il livello 
    try:
        slide_level_dims, found_level = get_slide_element_at_level(slide_img, wsi_level)
    except:
        if wsi_level > 0:
            slide_level_dims, found_level = get_slide_element_at_level(slide_img, wsi_level - 1)

    region = slide_img.read_region((0, 0), found_level, slide_level_dims) 
    region_np = np.array(region)
    
    # TODO FARE IN MODO DI SCORRERE SOLO SULLE IMMAGINI DI TEST PER EVITARE DI SALVARE SOLO SE CI SONO DELLE PREDICTION
    
    # Salvo l'immagine solo se ho delle predizioni su di essa
    found_prediction = False

    # Primo ciclo per disegnare le prediciton
    if slide_name in bb_data:
        print('Slide name found!')
        count = 0
        for bb in bb_data[slide_name]:
            x_min = int(bb['x_min'])
            x_max = int(bb['x_max'])
            y_min = int(bb['y_min'])
            y_max = int(bb['y_max'])
            count += 1
            print('Drawing bb number : {}'.format(count))
            color = (0, 255, 0)  # Verde
            thickness = 2
            cv2.rectangle(region_np, (x_min, y_min), (x_max, y_max), color, thickness)
            found_prediction = True
    
    if ground_truth_flag:

        with open(ground_truth_path, 'r') as gt_file:
            gt_boxes = json.load(gt_file)

        with open(wsi_info_file, 'r') as info:
            slide_info=yaml.safe_load(info)

        slide_name_extended = slide_name + '.svs'
        slide_name_extended_2 = slide_name +'.ndpi'

        if slide_name_extended in slide_info.keys(): 
            lv0_dims = slide_info[slide_name_extended]["Slide_LV0_dims"]
            lv1_dims = slide_info[slide_name_extended]["Slide_LV1_dims"]
            print(f"Dimensioni del livello 1 per {slide_name_extended}: {lv1_dims}")
        else:
            print(f"La WSI {slide_name} non è presente nel file.")

        if slide_name_extended_2 in slide_info.keys():
            lv0_dims = slide_info[slide_name_extended_2]["Slide_LV0_dims"]
            lv1_dims = slide_info[slide_name_extended_2]["Slide_LV1_dims"]
            print(f"Dimensioni del livello 1 per {slide_name_extended_2}: {lv1_dims}")
        else:
            print(f"La WSI {slide_name} non è presente nel file.")

        scale_x = int(lv1_dims[0]) / int(lv0_dims[0])
        scale_y = int(lv1_dims[1]) / int(lv0_dims[1])

        gt_count = 0
        for feature in gt_boxes['features']:
                coordinates = feature['geometry']['coordinates'][0]
                x_min_gt = min(c[0] for c in coordinates) * scale_x
                y_min_gt = min(c[1] for c in coordinates) * scale_x
                x_max_gt = max(c[0] for c in coordinates) * scale_y
                y_max_gt = max(c[1] for c in coordinates) * scale_y
                gt_count += 1
                print('Drawing gt_bb number : {}'.format(gt_count))
                color = (0, 0, 255)  # Blu
                thickness = 2
                cv2.rectangle(region_np, (int(x_min_gt), int(y_min_gt)), (int(x_max_gt), int(y_max_gt)), color, thickness)
        
    # Dopo aver disegnato le bounding box
    output_file_path = os.path.join(output_path, f"{slide_name}_mapped.jpg")

    # Specifica la qualità per il salvataggio
    quality = 70  # Scegli una qualità tra 0 (bassa) e 100 (alta)

    # Salva l'immagine con la qualità specificata
    if found_prediction:
        cv2.imwrite(output_file_path, cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        print(f"Immagine salvata con qualità ridotta ({quality}%) in: {output_file_path}")


    # cv2.imwrite(output_file_path, cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR))
    # print(f"Immagine salvata (senza ridimensionamento) in: {output_file_path}")

    # Ridimensiona solo se necessario
    # scale_percent = 100
    # width = int(region_np.shape[1] * scale_percent / 100)
    # height = int(region_np.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # region_resized = cv2.resize(region_np, dim, interpolation=cv2.INTER_AREA)

    # Salva la versione ridimensionata
    # output_file_resized = os.path.join(output_path, f"{slide_name}_resized.jpg")
    # cv2.imwrite(output_file_resized, cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR))
    # print(f"Immagine salvata (ridimensionata) in: {output_file_resized}")


def extract_patch_coordinates(patch_name):
    """Estrae le coordinate y e x dal nome della patch."""
    match = re.search(r"_y(\d+)_x(\d+)", patch_name)
    if match:
        y_offset = int(match.group(1))
        x_offset = int(match.group(2))
        return y_offset, x_offset
    else:
        raise ValueError(f"Nome patch non valido: {patch_name}")

def convert_bounding_boxes(data_path):
    """Converte le coordinate delle bounding box dal sistema della patch al sistema della WSI globale."""
    global_boxes = []
    
    with open(data_path, 'r') as f:
        data = json.load(f)

    for item in data:
        patch_name = item["nome_wsi"]
        bounding_boxes = item["bounding_boxes"]
        
        # Estrai le coordinate della patch
        y_offset, x_offset = extract_patch_coordinates(patch_name)
        
        # Converti ogni bounding box
        for box in bounding_boxes:
            global_box = {
                "x_min": box["x_min"] + x_offset,
                "y_min": box["y_min"] + y_offset,
                "x_max": box["x_max"] + x_offset,
                "y_max": box["y_max"] + y_offset,
                "confidence": box["confidence"],
                "class_id": box["class_id"],
                "class_name": box["class_name"]
            }
            global_boxes.append({"nome_wsi": patch_name, "global_bounding_box": global_box})
    
    return global_boxes

def save_to_json(data, destination_path):
    with open(destination_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"File salvato in {destination_path}")

def aggregate_bounding_boxes(mapped_boxes_path, aggregated_bounding_boxes_path):
   
    wsi_bounding_boxes = defaultdict(list)
    
    with open(mapped_boxes_path, 'r') as f:
        input_data = json.load(f)
    
    for item in input_data:
        # Estrai il nome della WSI rimuovendo le coordinate del patch
        match = re.match(r"(.*?)(?=_y\d+_x\d+)", item["nome_wsi"])
        if match:
            wsi_name = match.group(1)
        else:
            print(f"Formato nome WSI non valido: {item['nome_wsi']}")
            continue

      
        wsi_bounding_boxes[wsi_name].append(item["global_bounding_box"])

    output_data = dict(wsi_bounding_boxes)
    save_to_json(output_data, aggregated_bounding_boxes_path)



if __name__ == '__main__':

    wsi_info_file = "/work/grana_pbl/Detection_Glomeruli/INFO_wsi_file_dictionary_ALL.yaml"
    bounding_box_file = "/work/grana_pbl/Detection_Glomeruli/Yolo_results/Yolo_results_Lv1_Overlapping05/all_predictions.json"
    mapped_wsi_dest_path = "/work/grana_pbl/Detection_Glomeruli/WSI_bb_final_merged_coordinates.json"
    mapped_boxes_path = "/work/grana_pbl/Detection_Glomeruli/Yolo_results/Yolo_results_Lv1_Overlapping05/global_bounding_boxes.json"
    aggregated_bounding_boxes_path = "/work/grana_pbl/Detection_Glomeruli/Yolo_results/Yolo_results_Lv1_Overlapping05/aggregated_bounding_boxes.json"
    complete_wsi_boxed_path = "/work/grana_pbl/Detection_Glomeruli/Yolo_results/Yolo_results_Lv1_Overlapping05/final_boxed_complete_wsi_only_predicted"
    #res = merge(wsi_info_file, mapped_wsi_dest_path, bounding_box_file)
    # Converti le bounding box rispetto alla WSI globale
    #global_boxes = convert_bounding_boxes(bounding_box_file)
    #final_bounding_boxes_wsi_file = aggregate_bounding_boxes(mapped_boxes_path, aggregated_bounding_boxes_path)

    # Salva il risultato in un file JSON
    # with open(mapped_boxes_path, "w") as f:
    #     json.dump(global_boxes, f, indent=4)

    # TODO
    # METTERE A POSTO IL FILE INFO FILE, LA STORIA DELLE ESTENSIONI
    # GENERARE UNA CATENA DI JSON PIU' BREVE, PERCHE' PER ORA PRIMA LI TRASFORMO DA YOLO IN GLOBALE, POI AGGREGO I PATCHES E LI UNISCO PER WSI, POI INTERSECO
    # METTERE A POSTO LA SOGLIA DI INTERSEZIONE
    # GENERALE SOLO SE HO UNA PREDIZIONE

    # Prova
    wsi_level = 1
    wsi_path = '/work/grana_pbl/Detection_Glomeruli/HAMAMATSU'
    output_final_complete_aggregated_wsi_path = '/work/grana_pbl/Detection_Glomeruli/Yolo_results/Yolo_results_Lv1_Overlapping05/final_boxed_complete_wsi_only_predicted/mapping_finale.json'
    ground_truth_path = '/work/grana_pbl/Detection_Glomeruli/Coordinate_HAMAMATSU'
    # This flag is setted 'True' if you want to draw ground_truth to WSI.
    ground_truth_flag = True
    #merge_bounding_boxes(aggregated_bounding_boxes_path, output_final_complete_aggregated_wsi_path)
    
    # Ottieni i file di WSI e ground truth come dizionari basati sul nome base
    wsi_files = {os.path.splitext(f)[0]: os.path.join(wsi_path, f) for f in os.listdir(wsi_path)}
    gt_files = {os.path.splitext(f)[0]: os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)}

    common_keys = set(wsi_files.keys()) & set(gt_files.keys())

    if not common_keys:
        raise ValueError("Nessuna corrispondenza trovata tra WSI e ground truth!")

    for key in common_keys:
        current_wsi_path = wsi_files[key]
        current_gt_path = gt_files[key]
        
        print(f"Processando WSI: {current_wsi_path} con GT: {current_gt_path}")

        remap(
            current_wsi_path,
            wsi_info_file,
            wsi_level,
            output_final_complete_aggregated_wsi_path,
            complete_wsi_boxed_path,
            current_gt_path,
            ground_truth_flag
        )
    print('END !!!!!')
