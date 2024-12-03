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

def merge_bounding_boxes(aggregated_bounding_boxes_path, output_final_complete_aggregated_wsi_path, intersection_over_smaller_bounding_box):

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

                    # Verifica se l'overlap è superiore al tot %
                    if calculate_overlap_ratio(bb1, bb2) > intersection_over_smaller_bounding_box:
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


    
def remap(wsi_path, wsi_info_file, wsi_level, aggregated_bounding_boxes_path, output_path, ground_truth_path, ground_truth_flag, area_min_flag,  aspect_ratio_filter_flag, aspect_ratio_filter_value):

    """
    Con questa funzione voglio rimappare il file JSON finale sulla WSI. E' solo la funzione di stampa delle bounding boxes sulla WSI.

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
    
    # Ciclo per disegnare le bounding box di ground_truth
    if area_min_flag or ground_truth_flag:

        area_min = float('inf')
        gt_count = 0

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

        for feature in gt_boxes['features']:
                coordinates = feature['geometry']['coordinates'][0]
                x_min_gt = min(c[0] for c in coordinates) * scale_x
                y_min_gt = min(c[1] for c in coordinates) * scale_y
                x_max_gt = max(c[0] for c in coordinates) * scale_x
                y_max_gt = max(c[1] for c in coordinates) * scale_y
                area_current_box = (y_max_gt - y_min_gt)*(x_max_gt - x_min_gt)
                if area_current_box < area_min:
                    area_min = area_current_box
                gt_count += 1
                print('Drawing gt_bb number : {}'.format(gt_count))
                color = (0, 0, 255)  # Blu
                thickness = 2
                if ground_truth_flag:
                    cv2.rectangle(region_np, (int(x_min_gt), int(y_min_gt)), (int(x_max_gt), int(y_max_gt)), color, thickness)
    
    # Ciclo per disegnare le prediciton di YOLO
    if slide_name in bb_data:
        
        print('Slide name found!')
        number_of_predicted_bb = 0
        
        for bb in bb_data[slide_name]:
            x_min = int(bb['x_min'])
            x_max = int(bb['x_max'])
            y_min = int(bb['y_min'])
            y_max = int(bb['y_max'])
            height = y_max - y_min
            width = x_max - x_min

            # Qui applico uno dei filtri visuali che ho pensato, se la mia bounding_box ha un aspect_ratio strano non la disegno
            if aspect_ratio_filter_flag:
                aspect_ratio = max(height, width) / min(height, width)
                if aspect_ratio > aspect_ratio_filter_value:
                    print('Questo aspect ratio non va bene, non credo sia un glomerulo!')
                    continue
   
            area_predicted_bb = (height * width)

            # Filtro sull'area minima
            if not area_min_flag or area_predicted_bb >= area_min:
                cv2.rectangle(region_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                print(f'Drawing bb number: { number_of_predicted_bb + 1}')
            
            number_of_predicted_bb += 1
           
    # Dopo aver disegnato le bounding box
    output_file_path = os.path.join(output_path, f"{slide_name}_mapped.jpg")

    # Specifica la qualità per il salvataggio
    quality = 70  # Scegli una qualità tra 0 (bassa) e 100 (alta)

    # Salva l'immagine con la qualità specificata SOLO se ho trovato almeno una predizione
    if number_of_predicted_bb > 0:
        cv2.imwrite(output_file_path, cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        print(f"Immagine salvata con qualità ridotta ({quality}%) in: {output_file_path}")


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

def calculate_overlap_ratio(box1, box2):

    """
       Calcoliamoci il rapporto di sovrapposizione rispetto alle aree delle due bounding box, 
       non so se può andare bene ma proviamo.

    """
    intersection = intersection_area(box1, box2)

    area_box1 = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    area_box2 = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    
    # Rapporto rispetto all'area minima tra le due
    overlap_ratio = intersection / min(area_box1, area_box2)
    print('Questo è l\'overlap ratio:', overlap_ratio)

    percorso_tmp = '/work/grana_pbl/Detection_Glomeruli/Files_di_esempio_per_fare_prove/intersection.json'
    with open(percorso_tmp, 'a') as p_tmp:
        print('sto scrivendo l\'overlap ratio !')
        json.dump(overlap_ratio, p_tmp, indent=4)
    
    return overlap_ratio

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
    complete_wsi_boxed_path = "/work/grana_pbl/Detection_Glomeruli/Yolo_results/Yolo_results_Lv1_Overlapping05/final_boxed_complete_wsi_only_predicted_unified_boxes_with_intersection_threshold"
    #res = merge_bounding_boxes(wsi_info_file, mapped_wsi_dest_path, bounding_box_file)
    # Converti le bounding box rispetto alla WSI globale
    #global_boxes = convert_bounding_boxes(bounding_box_file)
    #final_bounding_boxes_wsi_file = aggregate_bounding_boxes(mapped_boxes_path, aggregated_bounding_boxes_path)

    # Salva il risultato in un file JSON
    # with open(mapped_boxes_path, "w") as f:
    #     json.dump(global_boxes, f, indent=4)


    # TODO
    # FRAE UN FILE DI CONFIGURAZIONE PER METTERE TUTTI I PERCORSI DENTRO UN UNICO FILE SEPARATO
    # METTERE A POSTO IL FILE INFO FILE, LA STORIA DELLE ESTENSIONI
    # GENERARE UNA CATENA DI JSON PIU' BREVE, PERCHE' PER ORA PRIMA LI TRASFORMO DA YOLO IN GLOBALE, POI AGGREGO I PATCHES E LI UNISCO PER WSI, POI INTERSECO
    # METTERE A POSTO LA SOGLIA DI INTERSEZIONE - Done
    # GENERALE SOLO SE HO UNA PREDIZIONE, QUESTO NON VA BENE, DEVO LAVORARE SOLO CON LE WSI DI TEST, DI CUI PERO' DEVO TENERE TRACCIA
    
    # METTERE A POSTO L'AGGREGAZIONE FACENDO IN MODO CHE: 
    # POST-PROCESSING
    # Per ridurre i falsi positivi
    # 1. LE BOUNDING BOX TROPPO PICCOLE NON VENGANO CONSIDERATE 
    # 2. NON CONSIDERARE QUELLE CHE HANNO MENO DI 2 DETECTION
    # 3. FILTRAGGIO BASATO SULLA FORMA DELLA BOUNDING BOX - Done


    # wsi_level = 1
    # wsi_path = '/work/grana_pbl/Detection_Glomeruli/HAMAMATSU'
    # output_final_complete_aggregated_wsi_path = '/work/grana_pbl/Detection_Glomeruli/Yolo_results/Yolo_results_Lv1_Overlapping05/final_boxed_complete_wsi_only_predicted_unified_boxes_with_intersection_threshold/mapping_finale.json'
    # ground_truth_path = '/work/grana_pbl/Detection_Glomeruli/Coordinate_HAMAMATSU'
    # # This flag is setted 'True' if you want to draw ground_truth to WSI.
    # ground_truth_flag = True
    # aspect_ratio_filter_flag = True
    # intersection_over_smaller_bounding_box = 0.3
    # aspect_ratio_filter_value = 2
    # #merge_bounding_boxes(aggregated_bounding_boxes_path, output_final_complete_aggregated_wsi_path)
    
    # # Ottieni i file di WSI e ground truth come dizionari basati sul nome base
    # wsi_files = {os.path.splitext(f)[0]: os.path.join(wsi_path, f) for f in os.listdir(wsi_path)}
    # gt_files = {os.path.splitext(f)[0]: os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)}

    # common_keys = set(wsi_files.keys()) & set(gt_files.keys())

    # if not common_keys:
    #     raise ValueError("Nessuna corrispondenza trovata tra WSI e ground truth!")

    # for key in common_keys:
    #     current_wsi_path = wsi_files[key]
    #     current_gt_path = gt_files[key]
        
    #     print(f"Processando WSI: {current_wsi_path} con GT: {current_gt_path}")

    #     remap(
    #         current_wsi_path,
    #         wsi_info_file,
    #         wsi_level,
    #         output_final_complete_aggregated_wsi_path,
    #         complete_wsi_boxed_path,
    #         current_gt_path,
    #         ground_truth_flag,
    #         aspect_ratio_filter_flag,
    #         aspect_ratio_filter_value,
    #         intersection_over_smaller_bounding_box
    #     )
    # print('END !!!!!')
    #----------------------------

    # Prova con una sola WSI che ho scelto
    wsi_path = '/work/grana_pbl/Detection_Glomeruli/Files_di_esempio_per_fare_prove/WSI_PATH'
    ground_truth_path = '/work/grana_pbl/Detection_Glomeruli/Files_di_esempio_per_fare_prove/COORDINATE'
    aggregated_bb_path = '/work/grana_pbl/Detection_Glomeruli/Files_di_esempio_per_fare_prove/mapping_aggregato.json'
    prova_wsi_test_path = '/work/grana_pbl/Detection_Glomeruli/Files_di_esempio_per_fare_prove/COMPLETE_WSI_BOXED_PATH'
    output_final_complete_aggregated_wsi_path = '/work/grana_pbl/Detection_Glomeruli/Files_di_esempio_per_fare_prove/mapping_finale.json'

    # Choose wsi-zoom-level
    wsi_level = 1
    # This flag is setted 'True' if you want to draw ground_truth to WSI.
    ground_truth_flag = True
    # Do you want to filter respect aspect ratio ?
    aspect_ratio_filter_flag = True
    # Set the value of the aspect ratio filter
    aspect_ratio_filter_value = 2
    # Set the falg True if you want to apply the filter over the area (if the predicted bb area is bigger than the minimum gt_bounding_box area)
    area_min_flag = True
    intersection_over_smaller_bounding_box = 0.3
    # merge_bounding_boxes(aggregated_bb_path, output_final_complete_aggregated_wsi_path, intersection_over_smaller_bounding_box)
    
    # # Ottieni i file di WSI e ground truth come dizionari basati sul nome base
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
            prova_wsi_test_path,
            current_gt_path,
            ground_truth_flag,
            area_min_flag,
            aspect_ratio_filter_flag,
            aspect_ratio_filter_value,
        )

    print('END !!!!!')
