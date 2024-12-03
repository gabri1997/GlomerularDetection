import os
import shutil
import random
import yaml


def split_wsi_and_create_folders_from_yaml(SEED, yaml_path, train_percentage):

    with open(yaml_path, 'r') as names:
        data = yaml.safe_load(names)

    all_ids = list(data['WSI'].keys())
    
    random.seed(SEED)
    random.shuffle(all_ids)
    split_idx = train_percentage*(len(all_ids))
    train_wsi_ids = all_ids[:int(split_idx)]
    test_wsi_ids = all_ids[int(split_idx):]

    train_wsi = [data['WSI'][id] for id in train_wsi_ids]
    test_wsi = [data['WSI'][id] for id in test_wsi_ids]    

    # Aggiungo l'operazione di flattening per avere una lista e non una lista di liste 
    flat_train = [item for sublist in train_wsi for item in sublist]
    flat_test = [item for sublist in test_wsi for item in sublist]

    return flat_train, flat_test

# Funzione per copiare annotazioni e immagini
def copy_files_by_prefisso(annotazioni_folder, wsi_list, dest_img_folder, dest_annot_folder):

    # Ottenere la lista delle annotazioni
    annotazioni_subfolders = sorted(os.listdir(annotazioni_folder))

    for wsi in wsi_list:
        for annotazione in annotazioni_subfolders:
            if wsi in annotazione:

                path = os.path.join(annotazioni_folder, annotazione)
                
                for annotazione_txt in sorted(os.listdir(path)):
                    # Nome file senza estensione
                    
                        annotazione_name = os.path.splitext(annotazione_txt)[0]
                        annotazione_src = os.path.join(path, annotazione_txt)
                        shutil.copy(annotazione_src, dest_annot_folder)
                        print('Ho copiato la label {}'.format(annotazione_txt))

                        # Scorrere le cartelle immagini per cercare una corrispondenza
                        # Iterare su entrambi i contenuti combinati con la distinzione del percorso
                        for cartella_img in sorted(os.listdir(immagini_base_folder_3D)) + sorted(os.listdir(immagini_base_folder_HAMA)):
                            # Verifica se la cartella è in immagini_base_folder_3D o immagini_base_folder_HAMA
                            if os.path.exists(os.path.join(immagini_base_folder_3D, cartella_img)):
                                tiles_folder = os.path.join(immagini_base_folder_3D, cartella_img, 'tiles')
                            else:
                                tiles_folder = os.path.join(immagini_base_folder_HAMA, cartella_img, 'tiles')
                            
                            if not os.path.exists(tiles_folder):
                                continue

                            for file_img in sorted(os.listdir(tiles_folder)):
                                if file_img.startswith(annotazione_name):
                                    image_src = os.path.join(tiles_folder, file_img)
                                    image_dest = os.path.join(dest_img_folder, file_img)
                                    shutil.copy(image_src, image_dest)
                                    print('Ho copiato img : {}'.format(file_img))
                                    break

    # Copia i file di train e val basati sui prefissi
    print("Copia completata!")

if __name__ == '__main__':

    # Definire i percorsi delle cartelle
    annotazioni_folder = '/work/grana_pbl/Detection_Glomeruli/Final_yolo_annotations_wsi_folder/Final_patches_yolo_annotations_folder_Lv1_scale_05_Overlapped'    # Cartella con le annotazioni
    immagini_base_folder_3D = '/work/grana_pbl/Detection_Glomeruli/Output_patchified_files_folder/Output_patchified_files_Lv1_3D_Overlapping05'          # Cartella dei patches già estratti
    immagini_base_folder_HAMA = '/work/grana_pbl/Detection_Glomeruli/Output_patchified_files_folder/Output_patchified_files_Lv1_HAMA_Overlapping05'      # Cartella che contiene le WSI con la sottocartella 'tiles'
    dest_train_immagini_folder = '/work/grana_pbl/Detection_Glomeruli/Yolo_dataset_Lv1_Overlapping05_final/images/train'        # Cartella di destinazione per immagini di train
    dest_train_annotazioni_folder = '/work/grana_pbl/Detection_Glomeruli/Yolo_dataset_Lv1_Overlapping05_final/labels/train'     # Cartella di destinazione per annotazioni di train
    dest_val_immagini_folder = '/work/grana_pbl/Detection_Glomeruli/Yolo_dataset_Lv1_Overlapping05_final/images/val'            # Cartella di destinazione per immagini di val
    dest_val_annotazioni_folder = '/work/grana_pbl/Detection_Glomeruli/Yolo_dataset_Lv1_Overlapping05_final/labels/val'  
    yaml_path = '/work/grana_pbl/Detection_Glomeruli/wsi_file_for_split.yaml'
    SEED = 42       # Cartella di destinazione per annotazioni di val
    train_percentage = 0.7

    # Creazione delle cartelle di destinazione se non esistono
    os.makedirs(dest_train_annotazioni_folder, exist_ok=True)
    os.makedirs(dest_train_immagini_folder, exist_ok=True)
    os.makedirs(dest_val_annotazioni_folder, exist_ok=True)
    os.makedirs(dest_val_immagini_folder, exist_ok=True)
    
    wsi_train, wsi_test = split_wsi_and_create_folders_from_yaml(SEED, yaml_path, train_percentage)
    print('Train prefissi : ', sorted(wsi_train))
    print('Test prefissi : ', sorted(wsi_test))

    #copy_files_by_prefisso(annotazioni_folder, wsi_train, dest_train_immagini_folder, dest_train_annotazioni_folder)
    #copy_files_by_prefisso(annotazioni_folder, wsi_test, dest_val_immagini_folder, dest_val_annotazioni_folder)
    print('END')