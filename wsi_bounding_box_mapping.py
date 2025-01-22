import os
import numpy as np
import json
import re
import yaml
import re
from collections import defaultdict
from openslide import open_slide
import yaml
import cv2
import math

class Merger:
    def __init__ (self, bounding_box_file, mapped_boxes_path, aggregated_b_boxes_path, output_final_complete_aggregated_wsi_path, **merger_config):
        
        self.bounding_box_file = bounding_box_file
        self.mapped_boxes_path = mapped_boxes_path
        self.aggregated_b_boxes_path = aggregated_b_boxes_path
        self.output_final_complete_wsi_path = output_final_complete_aggregated_wsi_path  
        self.intersection_over_smaller_bounding_box = merger_config["intersection_over_smaller_bounding_box"]
        self.merging_technique = merger_config["merging_technique"]
        self.tolerance = merger_config["tolerance"]
        self.wsi_level = merger_config["wsi_level"]
        # Salvo l'output del metodo
        self.result = None
    
    def get_result(self):
        return self.result

    def convert_bounding_boxes(self):

        """Converte le coordinate delle bounding box dal sistema della patch al sistema della WSI globale."""

        print('Inizio funzione convert_bounding_boxes : Converto le coordinate ...')

        global_boxes = []
        
        with open(self.bounding_box_file, 'r') as f:
            data = json.load(f)

        for item in data:
            patch_name = item["nome_wsi"]
            bounding_boxes = item["bounding_boxes"]
            
            # Estrai le coordinate della patch
            y_offset, x_offset = self._extract_patch_coordinates(patch_name)
            
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
        

        # # Salva il risultato in un file JSON
        with open(self.mapped_boxes_path, "a") as f:
            json.dump(global_boxes, f, indent=4)

        self.result = global_boxes

        return self

    def aggregate_bounding_boxes(self):

        "Aggrego tutte le box che appartengono alla stessa WSI"
        
        print('Inizio funzione aggregate_bounding_boxes : Aggrego tutte le WSI ...')

        wsi_bounding_boxes = defaultdict(list)
    
        for item in self.result:
            # Estrai il nome della WSI rimuovendo le coordinate del patch
            match = re.match(r"(.*?)(?=_y\d+_x\d+)", item["nome_wsi"])
            if match:
                wsi_name = match.group(1)
            else:
                print(f"Formato nome WSI non valido: {item['nome_wsi']}")
                continue
        
            wsi_bounding_boxes[wsi_name].append(item["global_bounding_box"])

        output_data = dict(wsi_bounding_boxes)
        self._save_to_json(output_data, self.aggregated_b_boxes_path)

        self.result = output_data

        return self

    def merge_bounding_boxes(self):

        """Unisce le bounding box in base alla tecnica di merging specificata."""

        print('Completo il merging con la funzione merge_bounding_boxes: Fondo le bounding box ...')

        final_boxes = {}

        for slide_name, bounding_boxes in self.result.items():
            merged = False
            while True:
                merged = False
                new_bounding_boxes = []  # Lista temporanea per le bounding box aggiornate
                skip_indices = set()  # Indici delle bounding box già unite

                # Gestione delle unioni a seconda della tecnica scelta
                if self.merging_technique == 'overlapping':
                    new_bounding_boxes, merged = self._merge_overlapping_bounding_boxes(
                        bounding_boxes, skip_indices, new_bounding_boxes
                    )
                else:
                    new_bounding_boxes, merged = self._merge_non_overlapping_bounding_boxes(
                        bounding_boxes, skip_indices, new_bounding_boxes
                    )

                # Se nessuna unione è stata fatta, usciamo dal ciclo
                if not merged:
                    break

                # Aggiorna la lista delle bounding box
                bounding_boxes = new_bounding_boxes

            # Salva le bounding box finali per questa slide
            final_boxes[slide_name] = bounding_boxes

        # Salva il risultato finale
        with open(self.output_final_complete_wsi_path, 'a') as f:
            json.dump(final_boxes, f, indent=4)

        print('End of mapping!')

    def _merge_overlapping_bounding_boxes(self, bounding_boxes, skip_indices, new_bounding_boxes):
        """Unisce le bounding box che si sovrappongono."""
        merged = False
        for i in range(len(bounding_boxes)):
            if i in skip_indices:
                continue
            bb1 = bounding_boxes[i]
            merged_with_another = False

            for j in range(i + 1, len(bounding_boxes)):
                if j in skip_indices:
                    continue
                bb2 = bounding_boxes[j]

                # Verifica se l'overlap è superiore alla soglia
                if self._calculate_overlap_ratio(bb1, bb2) > self.intersection_over_smaller_bounding_box:
                    new_bb_coords = self._unite_bounding_boxes(bb1, bb2)
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

        return new_bounding_boxes, merged

    def _merge_non_overlapping_bounding_boxes(self, bounding_boxes, skip_indices, new_bounding_boxes):
        """Unisce le bounding box che si toccano, con tolleranza aggiuntiva."""
        merged = False
        for i in range(len(bounding_boxes)):
            if i in skip_indices:
                continue
            bb1 = self._expand_bounding_box(bounding_boxes[i], self.tolerance)
            merged_with_another = False

            for j in range(i + 1, len(bounding_boxes)):
                if j in skip_indices:
                    continue
                bb2 = self._expand_bounding_box(bounding_boxes[j], self.tolerance)

                # Verifica se l'overlap tra le bounding box espanse è significativo
                if self._intersection_area(bb1, bb2) > 0 and self._intersection_area(bb1, bb2) <= self._min_tolerable_area(bb1, bb2, self.tolerance):
                    new_bb_coords = self._unite_bounding_boxes(bb1, bb2)
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

        return new_bounding_boxes, merged

    def _extract_patch_coordinates(self, patch_name):
    
        """Estrae le coordinate y e x dal nome della patch."""

        match = re.search(r"_y(\d+)_x(\d+)", patch_name)
        if match:
            y_offset = int(match.group(1))
            x_offset = int(match.group(2))
            return y_offset, x_offset
        else:
            raise ValueError(f"Nome patch non valido: {patch_name}")

    def _save_to_json(self, data, destination_path):
        with open(destination_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"File salvato in {destination_path}")

    def _calculate_overlap_ratio(self, box1, box2):

        """
        Calcoliamoci il rapporto di sovrapposizione rispetto alle aree delle due bounding box, 
        non so se può andare bene ma proviamo.

        """
        intersection = self._intersection_area(box1, box2)

        area_box1 = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
        area_box2 = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
        
        # Rapporto rispetto all'area minima tra le due
        overlap_ratio = intersection / min(area_box1, area_box2)
        #print('Questo è l\'overlap ratio:', overlap_ratio)

        percorso_tmp = '/work/grana_pbl/Detection_Glomeruli/Files_di_esempio_per_fare_prove/intersection.json'
        with open(percorso_tmp, 'a') as p_tmp:
            #print('sto scrivendo l\'overlap ratio !')
            json.dump(overlap_ratio, p_tmp, indent=4)
        
        return overlap_ratio

    def _intersection_area(self, box1, box2):

        x1_1, y1_1, x2_1, y2_1 = box1['x_min'], box1['y_min'], box1['x_max'], box1['y_max']
        x1_2, y1_2, x2_2, y2_2 = box2['x_min'], box2['y_min'], box2['x_max'], box2['y_max']

        # Calcola i limiti dell'intersezione
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        intersection_width = max(0, x_right - x_left)
        intersection_height = max(0, y_bottom - y_top)

        intersection_area = intersection_width * intersection_height
        return intersection_area

    def _unite_bounding_boxes(self, box1, box2):
  
        x1_1, y1_1, x2_1, y2_1  = box1['x_min'], box1['y_min'], box1['x_max'], box1['y_max']
        x1_2, y1_2, x2_2, y2_2  = box2['x_min'], box2['y_min'], box2['x_max'], box2['y_max']

        # Calcola i limiti della bounding box unita
        x1_union = min(x1_1, x1_2) # Coordinate piu a sinistra (x)
        y1_union = min(y1_1, y1_2) # Coordinate piu in alto (y)
        x2_union = max(x2_1, x2_2) # Coordinate piu a destra (x)
        y2_union = max(y2_1, y2_2) # Coordinate piu u

        return (x1_union, y1_union, x2_union, y2_union)

    def _expand_bounding_box(self, bb, tolerance):
        """
        Espande una bounding box di una certa tolleranza. Mi serve nel caso non overlapping, quando una volta rimappate le bounding box sulla wsi, voglio esapnderle per vedere quelle che sono adiacenti.

        :param bb: Bounding box originale.
        :param tolerance: Valore di espansione (in pixel).
        :return: Bounding box espansa.
        """
        return {
            'x_min': bb['x_min'] - tolerance,
            'y_min': bb['y_min'] - tolerance,
            'x_max': bb['x_max'] + tolerance,
            'y_max': bb['y_max'] + tolerance
        }

    def _min_tolerable_area(self, box1, box2, tolerance):
        #  Calcola l'area minima tollerabile di intersezione tra due bounding box espanse.
        x1_1, y1_1, x2_1, y2_1 = box1['x_min'], box1['y_min'], box1['x_max'], box1['y_max']
        x1_2, y1_2, x2_2, y2_2 = box2['x_min'], box2['y_min'], box2['x_max'], box2['y_max']

        # Voglio calcolare il lato di massima lunghezza tra le due bounding box
        # Calcolo dei lati delle bounding box
        width1 = x2_1 - x1_1
        height1 = y2_1 - y1_1
        width2 = x2_2 - x1_2
        height2 = y2_2 - y1_2
        # Lato massimo tra le due bounding box
        max_side = max(width1, height1, width2, height2)
        # Qui ho messo il valore 3 facendo qualche prova, è empirico, andrebbe sistemato
        return 3*tolerance*max_side

class Mapper:
    def __init__(self, wsi_path, ground_truth_path, output_final_complete_aggregated_wsi_path, wsi_info_file, **mapper_config):
        self.wsi_path = wsi_path
        self.ground_truth_path = ground_truth_path
        self.wsi_info_file = wsi_info_file
        self.output_final_complete_aggregated_wsi_path = output_final_complete_aggregated_wsi_path
        self.ground_truth_flag = mapper_config["ground_truth_flag"]
        self.aspect_ratio_filter_flag = mapper_config["aspect_ratio_filter_flag"]
        self.aspect_ratio_filter_value = mapper_config["aspect_ratio_filter_value"]
        # Salviamo la wsi con una data qualità rispetto a 100
        self.wsi_level = mapper_config["wsi_level"]
        self.quality = mapper_config["quality"]
        self.min_area = mapper_config["min_area"]
        self.save_wsi_flag = mapper_config["save_wsi_flag"]

    def remap(self, current_wsi_path, current_gt_path, output_wsi_mapped_folder_path):
        
        """
            Con questa funzione voglio rimappare il file JSON finale dove ho le bounding box ggregate dal merge sulla WSI. 
            E' solo la funzione di stampa delle bounding boxes sulla WSI.

         """
        
        # Prendo per esempio 'R22-117 C1q'
        with open(self.output_final_complete_aggregated_wsi_path, 'r') as f:
            bb_data = json.load(f)

        slide = current_wsi_path.split('/')[-1]
        slide_name = os.path.splitext(slide)[0]

        if self.save_wsi_flag == True:
        
            slide_img = open_slide(current_wsi_path)
            print(f"Slide - {slide_name} - opened successfully")
            # Scelgo il livello 
            try:
                slide_level_dims, found_level = self._get_slide_element_at_level(slide_img, self.wsi_level)
            except:
                if self.wsi_level > 0:
                    slide_level_dims, found_level = self._get_slide_element_at_level(slide_img, self.wsi_level - 1)

            region = slide_img.read_region((0, 0), found_level, slide_level_dims) 
            region_np = np.array(region)
    

        # TODO FARE IN MODO DI SCORRERE SOLO SULLE IMMAGINI DI TEST PER EVITARE DI SALVARE SOLO SE CI SONO DELLE PREDICTION
        
        # Ciclo per disegnare le bounding box di ground_truth
        ground_truth_boxes = []

        if self.min_area or self.ground_truth_flag:

            gt_count = 0

            with open(current_gt_path, 'r') as gt_file:
                gt_boxes = json.load(gt_file)

            with open(self.wsi_info_file, 'r') as info:
                slide_info=yaml.safe_load(info)

            slide_name_extended = slide_name + '.svs'
            slide_name_extended_2 = slide_name +'.ndpi'

            if slide_name_extended in slide_info.keys(): 
                lv0_dims = slide_info[slide_name_extended]["Slide_LV0_dims"]
                lv1_dims = slide_info[slide_name_extended]["Slide_LV1_dims"]
                print(f"Dimensioni del livello 1 per {slide_name_extended}: {lv1_dims}")
            else:
                pass

            if slide_name_extended_2 in slide_info.keys():
                lv0_dims = slide_info[slide_name_extended_2]["Slide_LV0_dims"]
                lv1_dims = slide_info[slide_name_extended_2]["Slide_LV1_dims"]
                print(f"Dimensioni del livello 1 per {slide_name_extended_2}: {lv1_dims}")
            else:
                pass

            scale_x = int(lv1_dims[0]) / int(lv0_dims[0])
            scale_y = int(lv1_dims[1]) / int(lv0_dims[1])

            for feature in gt_boxes['features']:
                    coordinates = feature['geometry']['coordinates'][0]
                    x_min_gt = min(c[0] for c in coordinates) * scale_x
                    y_min_gt = min(c[1] for c in coordinates) * scale_y
                    x_max_gt = max(c[0] for c in coordinates) * scale_x
                    y_max_gt = max(c[1] for c in coordinates) * scale_y
                    gt_count += 1
                    # Calcolo centroide ed area
                    centroid = ((x_min_gt + x_max_gt)/2, (y_min_gt + y_max_gt)/2)
                    
                    width = x_max_gt - x_min_gt
                    height = y_max_gt - y_min_gt
                    area = height * width
                    # Costurisco la lista delle bounding box di ground truth
                    box = {'x_min' : x_min_gt, 'y_min' : y_min_gt, 'x_max' : x_max_gt, 'y_max' : y_max_gt, 'x_centroid': centroid[0], 'y_centroid' : centroid[1], 'area':area}
                    ground_truth_boxes.append(box)

                    print('Drawing gt_bb number : {}'.format(gt_count))
                    color = (0, 0, 255)  # Blu
                    thickness = 2
                    if self.ground_truth_flag and self.save_wsi_flag == True:
                        cv2.rectangle(region_np, (int(x_min_gt), int(y_min_gt)), (int(x_max_gt), int(y_max_gt)), color, thickness)
                        # Disegno anche il centroide
                        cv2.circle(region_np,  (int(centroid[0]), int(centroid[1])), 6, (255,0,0), 2)
        
        # Ciclo per disegnare le prediciton di YOLO
        number_of_predicted_bb = 0

        predicted_boxes = []

        if slide_name in bb_data:
            
            print('Slide name {} found!'.format(slide_name))
            
            for bb in bb_data[slide_name]:
                x_min = int(bb['x_min'])
                x_max = int(bb['x_max'])
                y_min = int(bb['y_min'])
                y_max = int(bb['y_max'])
        
                # Calcolo centroide ed area
                centroid = ((x_min + x_max)/2, (y_min + y_max)/2)

                width = x_max - x_min
                height = y_max - y_min
                area = height * width

                box = {'x_min' : x_min, 'x_max' : x_max, 'y_min' : y_min, 'y_max' : y_max, 'x_centroid': centroid[0], 'y_centroid' : centroid[1], 'area': area}
                

                # Aggiungi la tolleranza per stampare le bounding boxes espanse e controllare
                # tolerance = 10  # Sostituisci con il valore desiderato
                # x_min = x_min - tolerance
                # y_min = y_min - tolerance
                # x_max = x_max + tolerance
                # y_max = y_max + tolerance

                # Qui applico uno dei filtri visuali che ho pensato, se la mia bounding_box ha un aspect_ratio strano non la disegno
                if self.aspect_ratio_filter_flag:
                    aspect_ratio = max(height, width) / min(height, width)
                    if aspect_ratio > self.aspect_ratio_filter_value:
                        print('Questo aspect ratio non va bene, non credo sia un glomerulo!')
                        continue
                    else:
                        predicted_boxes.append(box)
                else:
                    predicted_boxes.append(box)

                area_predicted_bb = (height * width)

                # Filtro sull'area
                if self.save_wsi_flag == True:
                    if not self.min_area or (area_predicted_bb >= self.min_area):
                        cv2.rectangle(region_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        # Disegno anche il centroide
                        cv2.circle(region_np,  (int(centroid[0]), int(centroid[1])), 6, (255,0,0), 2)
                        print(f'Drawing predicted bb number: { number_of_predicted_bb + 1}')
                
                number_of_predicted_bb += 1
            
        # Dopo aver disegnato le bounding box
        output_file_path = os.path.join(output_wsi_mapped_folder_path, f"{slide_name}_mapped.jpg")

        # Salva l'immagine con la qualità specificata SOLO se ho trovato almeno una predizione e ho il flag a True
        if number_of_predicted_bb > 0 and self.save_wsi_flag == True:
            cv2.imwrite(output_file_path, cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            print(f"Immagine salvata con qualità ridotta ({self.quality}%) in: {output_file_path}")
        else:
            print('Non ci sono bounding box per la WSI : ', slide_name)

        return predicted_boxes, ground_truth_boxes

    def _get_slide_element_at_level(self, slide, lvl):
            num_levels = slide.level_count
            # Controlla se il livello richiesto è valido
            while lvl >= 0:
                if lvl < num_levels: 
                    return slide.level_dimensions[lvl], lvl
                lvl -= 1  
            return None 

class BoxEvaluator:
    def __init__(self, threshold):
        
        self.threshold = threshold

    @staticmethod
    def calculate_iou(box1, box2):
        """Calcola l'Intersection over Union (IoU) tra due bounding box."""
        area_intersection = BoxEvaluator.intersection_area(box1, box2)
        area_box1 = box1['area']
        area_box2 = box2['area']
        area_union = area_box1 + area_box2 - area_intersection

        if area_union == 0:
            return 0.0
        return area_intersection / area_union

    @staticmethod
    def intersection_area(box1, box2):
        """Calcola l'area di intersezione tra due bounding box."""
        x_min = max(box1['x_min'], box2['x_min'])
        y_min = max(box1['y_min'], box2['y_min'])
        x_max = min(box1['x_max'], box2['x_max'])
        y_max = min(box1['y_max'], box2['y_max'])

        if x_min < x_max and y_min < y_max:
            return (x_max - x_min) * (y_max - y_min)
        return 0.0

    def compute_final_metric(self, predictions_boxes, ground_truth_boxes, slide_name):

        """Calcola la metrica finale basata sull'IoU."""

        paired_boxes = []
        p_boxes = []
        gt_boxes = []

        print(f"Computing final metrics")

        for box_g in ground_truth_boxes:
            min_centroid_distance = float('inf')  
            tmp_box = None
            
            for box_p in predictions_boxes:
                x_value = pow(box_g['x_centroid'] - box_p['x_centroid'], 2)
                y_value = pow(box_g['y_centroid'] - box_p['y_centroid'], 2)
                actual_centroid_distance = math.sqrt(x_value + y_value)

                if actual_centroid_distance <= min_centroid_distance:
                    min_centroid_distance = actual_centroid_distance
                    tmp_box = box_p
                    
            if tmp_box:
                p_boxes.append(tmp_box)
                gt_boxes.append(box_g)
                paired_boxes.append({'predicted_box': tmp_box, 'gt_box': box_g})
        
        # Salva i paired_boxes in un file YAML
        with open('_paired_boxes.yaml', 'a') as f:
            yaml.dump(paired_boxes, f, default_flow_style=False)
        
        # Calcolo della IoU
        total_ious = []
        for bb_pred, bb_gt in zip(p_boxes, gt_boxes):
            iou = self.calculate_iou(bb_pred, bb_gt)
            if iou > 0:
                total_ious.append(iou)

        mean_iou_per_slide = sum(total_ious) / len(total_ious) if total_ious else 0
        print(f'La IoU per la slide {slide_name} è {mean_iou_per_slide:.4f}')
        return mean_iou_per_slide

    def compute_fp_fn_tp(self, predictions, ground_truth):
        """Calcola False Positives (FP), False Negatives (FN) e True Positives (TP)."""
        fn = 0
        fp = 0
        tp = 0
        matched_ground_truth = set()

        for pred in predictions:
            used_prediction = False
            for idx, gt in enumerate(ground_truth):
                iou = self.calculate_iou(pred, gt)
                if iou >= self.threshold and idx not in matched_ground_truth:
                    tp += 1
                    matched_ground_truth.add(idx)
                    used_prediction = True
                    break
            if not used_prediction:
                fp += 1

        fn = len(ground_truth) - len(matched_ground_truth)
        return len(ground_truth), fp, fn, tp
        
if __name__ == '__main__':
    
    wsi_info_file = "/work/grana_pbl/Detection_Glomeruli/INFO_wsi_file_dictionary_ALL.yaml"
    bounding_box_file = "/work/grana_pbl/Detection_Glomeruli/Yolo_results/Yolo_results_LV1_Non_Overlapping_SEED_42/all_predictions.json"
    mapped_boxes_path = "/work/grana_pbl/Detection_Glomeruli/prova_nuovo_script/global_bounding_boxes.json"
    aggregated_b_boxes_path = "/work/grana_pbl/Detection_Glomeruli/prova_nuovo_script/aggregated_bounding_boxes.json"
    output_wsi_mapped_folder_path = "/work/grana_pbl/Detection_Glomeruli/prova_nuovo_script/Final_boxed_wsi"
    output_final_complete_aggregated_wsi_path = '/work/grana_pbl/Detection_Glomeruli/prova_nuovo_script/mapping_finale.json'
    splitted_wsi_names = '/work/grana_pbl/Detection_Glomeruli/Yolo_results/Yolo_results_LV1_Overlapping_05_SEED_42/train_test_indices.yaml'
    metrics_path = "/work/grana_pbl/Detection_Glomeruli/prova_nuovo_script/Metrics.txt"
    wsi_path = '/work/grana_pbl/Detection_Glomeruli/HAMAMATSU'
    ground_truth_path = '/work/grana_pbl/Detection_Glomeruli/Coordinate_HAMAMATSU'

    merger_config = {
    "merging_technique": 'overlapping',  # 'overlapping' o 'non-overlapping'
    "tolerance": 10,
    "wsi_level": 1,
    # If the value is set to 0.5, means that the intersection area is at least 50% of the area of the smallest bounding box
    "intersection_over_smaller_bounding_box": 0.3
    }

    merger = Merger(bounding_box_file, mapped_boxes_path, aggregated_b_boxes_path, output_final_complete_aggregated_wsi_path, **merger_config)
    # merger.convert_bounding_boxes()
    # merger.aggregate_bounding_boxes()
    # merger.merge_bounding_boxes()  
    
    mapper_config = {

        # This flag is setted 'True' if you want to draw ground_truth to WSI.
        "ground_truth_flag" : True,
        "aspect_ratio_filter_flag" : True,
        "aspect_ratio_filter_value" : 2,
        # Salviamo la wsi con una data qualità rispetto a 100
        "wsi_level" : 1, 
        "quality" : 70,
        "min_area": 500,
        "save_wsi_flag": False
    }
  
    # Ottieni i file di WSI e ground truth come dizionari basati sul nome base

    mapper = Mapper(wsi_path, ground_truth_path, output_final_complete_aggregated_wsi_path, wsi_info_file,**mapper_config)

    evaluator = BoxEvaluator(threshold=0.5)

    wsi_files = {os.path.splitext(f)[0]: os.path.join(wsi_path, f) for f in os.listdir(wsi_path)}
    gt_files = {os.path.splitext(f)[0]: os.path.join(ground_truth_path, f) for f in os.listdir(ground_truth_path)}

    with open(splitted_wsi_names, 'r') as file:
        wsi = yaml.safe_load(file)

    # Seleziono solo le WSI che ho nello split di test  
    wsi_files_test = {key: value for key, value in wsi_files.items() if key in wsi['test']}

    common_keys = set(wsi_files_test.keys()) & set(gt_files.keys())

    if not common_keys:
        raise ValueError("Nessuna corrispondenza trovata tra WSI e ground truth!")

    slides_mean_iou = [] 
    total_gt, total_fp, total_fn, total_tp = 0, 0, 0, 0

    for key in common_keys:

        current_wsi_path = wsi_files_test[key]
        current_gt_path = gt_files[key]

        slide_name = current_wsi_path.split('/')[-1]
        slide_name = os.path.splitext(slide_name)[0]
        
        print("\n")
        print(f"Processando WSI: {slide_name} con GT: {current_gt_path}")
    
        predicted_boxes, ground_truth_boxes = mapper.remap(
            current_wsi_path,
            current_gt_path,
            output_wsi_mapped_folder_path,
        )

        if predicted_boxes :
            single_slide_mean_iou = evaluator.compute_final_metric(predicted_boxes, ground_truth_boxes, slide_name)
        else:
            # Non ho predizioni per questa WSI di test
            single_slide_mean_iou = None

        if single_slide_mean_iou is not None:
            truncated_value = math.floor(single_slide_mean_iou * 100) / 100
            slides_mean_iou.append(truncated_value)
        else:
            slides_mean_iou.append(single_slide_mean_iou)

        gt, fp, fn, tp = evaluator.compute_fp_fn_tp(predicted_boxes, ground_truth_boxes)

        total_gt += gt
        total_fp += fp
        total_fn += fn
        total_tp += tp

    print(f"Ecco il vettore di Intersection over Union finale:  {slides_mean_iou}\n")

    slides_iou_filtered = [value for value in slides_mean_iou if value is not None and value!= 0]
    print(f"Questo è il valore di mean iou {sum(slides_iou_filtered)/len(slides_iou_filtered)}")

    # Escludiamo i None o 0 e calcoliamo la media
    valid_iou_values = [value for value in slides_mean_iou if value is not None and value != 0]
    mean_iou = sum(valid_iou_values) / len(valid_iou_values)

    with open(metrics_path, 'a') as metric:
        metric.write(f"WSI Folder: {wsi_path}\n")
        metric.write(f"Merging technique: {merger.merging_technique}\n")
        metric.write(f"Mean Intersection over Union: {mean_iou}\n")
        metric.write(f"Total_GT: {total_gt}\n")
        metric.write(f"Total_TP: {total_tp}\n")
        metric.write(f"Total_FN: {total_fn}\n")
        metric.write(f"Total_FP: {total_fp}\n")

    print(f"Mean Intersection over Union: {mean_iou}\n"
        f"Total_GT: {total_gt}\n"
        f"Total_TP: {total_tp}\n"
        f"Total_FN: {total_fn}\n"
        f"Total_FP: {total_fp}\n")

    

