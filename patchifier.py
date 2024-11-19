import json
import os
from collections import defaultdict
import cv2
import argparse
import numpy as np
from shapely.geometry import Polygon, box
from openslide import open_slide
import yaml
from shapely.geometry import Polygon

class Patchifier:
    def __init__(self, wsi_base_path, output_patches_path, patchifier_patch_size, max_patch_background, annotations_base_path, iterations, lower_bound_pixel_value, upper_bound_pixel_value, patchification_level):
        
        self.wsi_base_path = wsi_base_path 
        self.output_patches_path = output_patches_path
        self.patchifier_patch_size = patchifier_patch_size
        self.max_patch_background = max_patch_background
        self.annotations_base_path = annotations_base_path
        self.label_min_area_intersection = 0.05
        self.reduction_ratio = 15
        self.use_contour_approx = True
        self.save_processing_images = True
        self.save_mask_patches = True
        self.save_annot_patches = True
        self.iterations = iterations
        self.lower_bound_pixel_value = lower_bound_pixel_value
        self.upper_bound_pixel_value = upper_bound_pixel_value
        self.patchification_level = patchification_level

    @staticmethod
    def _find_wsis_with_annotations(wsi_base_path, annotations_base_path=None):
        # Function to list files without extensions
        def list_files_without_extensions(directory):
            return {os.path.splitext(file)[0] for file in os.listdir(directory)}

        # List all files in each directory without their extensions
        wsi_imgs_names = list_files_without_extensions(wsi_base_path)
        if annotations_base_path:
            wsi_annotations_names = list_files_without_extensions(annotations_base_path)

            # Find intersection of both sets to get a list of matching filenames (without extensions)
            wsi_imgs_names = wsi_imgs_names.intersection(wsi_annotations_names)

            annotation_path_list = [
                annotation
                for annotation in os.listdir(annotations_base_path)
                if os.path.splitext(annotation)[0] in wsi_imgs_names
            ]
            annotation_path_list.sort()
            # pprint.pp(annotation_path_list)
        else:
            annotation_path_list = None

        annotaded_wsi_path_list = [
            wsi
            for wsi in os.listdir(wsi_base_path)
            if os.path.splitext(wsi)[0] in wsi_imgs_names
        ]
        annotaded_wsi_path_list.sort()
        # pprint.pp(annotaded_wsi_path_list)

        if annotations_base_path:
            assert len(annotaded_wsi_path_list) == len(
                annotation_path_list
            ), "Number of WSI images and annotations don't match!"

        return annotaded_wsi_path_list, annotation_path_list

    @staticmethod
    def _flatten_geojson_coordinates(coordinates):
        """
        Recursively flattens a list of GeoJSON coordinates to a simple list of [x, y] points.
        """
        # The flattened list of coordinates to return
        flattened = []

        # Helper function to determine if an item is a coordinate pair
        def is_coordinate_pair(item):
            return (
                isinstance(item, list)
                and len(item) == 2
                and all(isinstance(x, (int, float)) for x in item)
            )

        # Recursive function to process the elements
        def flatten(items):
            for item in items:
                if is_coordinate_pair(item):
                    # If it's a coordinate pair, append to the flattened list
                    flattened.append(item)
                elif isinstance(item, list):
                    # If it's a list, recurse into it
                    flatten(item)
                else:
                    # Not a coordinate pair or a list, ignore or raise an error as needed
                    raise ValueError("Invalid coordinate structure")

        # Start flattening process
        flatten(coordinates)
        return flattened

    @staticmethod
    def _validate_and_correct(geometry):
        if not geometry.is_valid:
            # print("Invalid Geometry:", explain_validity(geometry))
            # A common trick to fix invalid geometries
            geometry = geometry.buffer(0)
            if not geometry.is_valid:
                pass
                # print("Failed to fix geometry:", explain_validity(geometry))
        return geometry

    @staticmethod
    def _accumulate_values(data):
        result = defaultdict(int)

        for item in data:
            for label, value in item.items():
                result[label] += value

        return dict(result)
   

    def _segment_wsi(self, wsi):

        # Convert the image to grayscale
        gray = cv2.cvtColor(wsi, cv2.COLOR_BGR2GRAY)

        # Crea una maschera per i pixel tra lower e upper
        final_mask = cv2.inRange(gray, self.lower_bound_pixel_value, self.upper_bound_pixel_value)

        # Specify a kernel for dilation
        kernel = np.ones((10, 10), np.uint8)  # Puoi regolare la dimensione del kernel

        # Apply dilation to the mask
        dilated_mask = cv2.dilate(final_mask, kernel, iterations=self.iterations)

        # Composite image for visualization
        composite_image = np.concatenate((gray, final_mask, dilated_mask), axis=1)
        height, width = composite_image.shape
        composite_resized = cv2.resize(
            composite_image,
            (width // self.reduction_ratio, height // self.reduction_ratio),
            interpolation=cv2.INTER_NEAREST,
        )

        return dilated_mask, composite_resized


    def _create_annotated_masks(self, wsi, annotation):

        print("Creating annotated wsi mask image...")
        
        annotation_mask = np.zeros(wsi.size, dtype=np.uint8)
        wsi_masked = wsi.copy()

        # create a annotation_mask for annotations from the geojson files
        for feature in annotation["features"]:

            object_type_str = (
                "objectType" if "objectType" in feature["properties"] else "object_type"
            )

            if feature["properties"][object_type_str] == "annotation":
                coordinates = feature["geometry"]["coordinates"]
                poly_points = self._flatten_geojson_coordinates(coordinates)
                polygon = np.array(poly_points, dtype=np.int32)

                if (
                    "classification" in feature["properties"]
                    and "color" in feature["properties"]["classification"]
                ):
                    color = tuple(feature["properties"]["classification"]["color"])
                    cv2.fillPoly(
                        annotation_mask, np.int32([polygon]), color=color[::-1]
                    )
                    cv2.polylines(
                        wsi_masked,
                        np.int32([polygon]),
                        isClosed=True,
                        color=color[::-1],
                        thickness=20,
                    )
                else:
                    cv2.fillPoly(
                        annotation_mask, np.int32([polygon]), color=(255, 0, 0)
                    )
                    cv2.polylines(
                        wsi_masked,
                        np.int32([polygon]),
                        isClosed=True,
                        color=(255, 0, 0),
                        thickness=20,
                    )

        return wsi_masked, annotation_mask

    # Pad dell'immagine con il colore nero
    def _pad_img(self, img, pad_color):

        height, width = img.shape[:2]
        patch_size = (self.patchifier_patch_size[0], self.patchifier_patch_size[0])
        # pad image and mask and tissue mask to be divisible by the patch size dimensions
        new_height = height + (patch_size[0] - height % patch_size[0])
        new_width = width + (patch_size[1] - width % patch_size[1])
        img = cv2.copyMakeBorder(
            img,
            0,
            new_height - height,
            0,
            new_width - width,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )

        if height < patch_size[0]:
            padding = patch_size[0] - height
            img = cv2.copyMakeBorder(
                img, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=pad_color
            )

        if width < patch_size[1]:
            padding = patch_size[1] - width
            img = cv2.copyMakeBorder(
                img, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=pad_color
            )
       

        # Ridimensiona l'immagine a 800x600
        #resized_img = cv2.resize(img, (800, 600))

        # Visualizza l'immagine in una finestra
        # cv2.imshow("Padded Image", resized_img)
        # cv2.waitKey(0)  # Attende un tasto per chiudere la finestra
        # cv2.destroyAllWindows()


        return img

    def _label_patch(
        self,
        pos_y,
        pos_x,
        patch_size: tuple,
        annotation_file,
        intersection_threshold=0.1,
    ):

        # Define the rectangular polygon for the WSI patch
        patch_polygon = box(pos_x, pos_y, pos_x + patch_size[0], pos_y + patch_size[1])
        patch_area = patch_polygon.area  # get patch area

        # Initialize a list to hold labels associated with this patch
        labels = []
        colors = []
        intersections = []

        for feature in annotation_file["features"]:
            object_type_str = (
                "objectType" if "objectType" in feature["properties"] else "object_type"
            )

            if feature["properties"][object_type_str] == "annotation":
                coordinates = feature["geometry"]["coordinates"]
                poly_points = self._flatten_geojson_coordinates(coordinates)
                annotation_polygon = Polygon(poly_points)
                annotation_polygon = self._validate_and_correct(annotation_polygon)
                if patch_polygon.intersects(annotation_polygon):
                    intersection_polygon = patch_polygon.intersection(
                        annotation_polygon
                    )
                    intersection_polygon = self._validate_and_correct(
                        intersection_polygon
                    )
                    intersection_area = (
                        intersection_polygon.area
                    )  # get intersection area
                    intersection_perc = np.round(intersection_area / patch_area, 3)
                    # self.logger.info("Intersection Percentage:", intersection_perc)

                    # if the amount of intersected annotated area in the patch surpasses a certain percentage, associate the label and color to the patch
                    if intersection_perc >= intersection_threshold or (intersection_polygon.contains(patch_polygon) and intersection_area <= 0.3 * patch_area):
                        # QUESTO LO FACCIO PER CONSIDERARE SE LA MIA BOUNDING BOX E' COMPLETAMENTE CONTENUTA ALL'INTERNO DEL MIO PATCH 
                        if "classification" in feature["properties"]:
                            if "name" in feature["properties"].get(
                                "classification", {}
                            ):
                                label = feature["properties"]["classification"]["name"]
                                labels.append(label)
                                # save intersection only if label is present
                                intersections.append({label: intersection_perc})
                            if "color" in feature["properties"]["classification"]:
                                colors.append(
                                    tuple(
                                        feature["properties"]["classification"]["color"]
                                    )[::-1]
                                )

                    else:
                        pass
                        # self.logger.info("skipping label for patch")

        labels = labels if labels else ["none"]
        labels = list(set(labels))  # remove duplicate labels
        colors = colors if colors else None
        intersections = (
            self._accumulate_values(data=intersections) if intersections else None
        )
        if colors:
            colors = list(set(colors))  # remove duplicate colors

        # Return labels if any intersections are found, else return ["none"]
        return labels, colors, intersections

    def _color_patch(self, img, patch_size, r, c, new_r, new_c, colors=None):
        default_color = (152, 152, 152)  # if no colors, default to this one
        if colors:  # If there are colors specified
            num_colors = len(colors)
            # Calculate height for each stripe
            delta = patch_size[1] // num_colors

            # Loop to draw each colored stripe
            for i, color in enumerate(colors):
                start_y = r + i * delta
                # Check if it's the last stripe; it should extend to the end of the patch
                if i == num_colors - 1:
                    end_y = new_r
                else:
                    end_y = start_y + delta

                # Draw the rectangle with the specified color
                cv2.rectangle(
                    img,
                    (c, start_y),
                    (new_c, end_y),
                    color,
                    -1,
                )
        # if no annotation or label color is specified, use the default one
        else:
            cv2.rectangle(
                img,
                (c, r),
                (new_c, new_r),
                default_color,
                -1,
            )

    def _slidunique_labels(self, annotation_data):
        labels_list = []
        for feature in annotation_data["features"]:
            object_type_str = (
                "objectType" if "objectType" in feature["properties"] else "object_type"
            )

            if feature["properties"][object_type_str] == "annotation":
                if "classification" in feature["properties"]:
                    if "name" in feature["properties"].get("classification", {}):
                        label = feature["properties"]["classification"]["name"]
                        labels_list.append(label)

        labels_list = labels_list if labels_list else ["none"]

        return list(set(labels_list))  # remove duplicate labels

    def get_slide_element_at_level(self, slide, lvl):
        num_levels = slide.level_count
        # Controlla se il livello richiesto è valido
        while lvl >= 0:
            if lvl < num_levels:  # Se il livello esiste
                return slide.level_dimensions[lvl], lvl
            lvl -= 1  # Scendi al livello successivo
        return None  # Nessun livello valido trovato

    
    def patchify_wsi(self, wsi_dict_path, micrometer_target, save_mask_patches=None, save_annot_patches=None, overlapped=None):
        
        # Devo patchificare ad una dimensione fissa


        with open(wsi_dict_path, 'r') as yaml_file:
            wsi_dict = yaml.safe_load(yaml_file)  # Carica il dizionario dal file YAML
        
        for slide_name, _ in wsi_dict.items():

            if os.path.splitext(slide_name)[0] not in os.listdir(self.output_patches_path):


                print('Slide base path', self.wsi_base_path)
                single_wsi_path = os.path.join(self.wsi_base_path, slide_name) 
                print('Slide final path', single_wsi_path)
                slide_img = open_slide(single_wsi_path) 
                slide_dims = slide_img.dimensions
                slide_name = os.path.splitext(slide_name)[0]
                
                # Scelgo il livello 
                try:
                    slide_dims, found_level = self.get_slide_element_at_level(slide_img, self.patchification_level)
                except:
                    print ('Sto riducendo il livello e passando al max resolution')
                    slide_dims, found_level = self.get_slide_element_at_level(slide_img, self.patchification_level - 1)

                print("Queste sono le dimensioni della slide : ", slide_dims)
                print("Questo è il livello selezionato : ", found_level)

                _, _, micrometer_lv0, micrometer_lv1 = get_dimensions_for_key(wsi_dict_path, slide_name)
                print('Micrometer lv0:', micrometer_lv0)
                print('Micrometer lv1:', micrometer_lv1)

                #TODO
                # Qui ho i micrometri per entrambi i livelli, lv0 e lv1
                # Ho le dimensioni della mia immagine, devo definire un valore target di micrometri e fare un resize in modo che la mia wsi abbia sempre gli stessi micrometri
                # Devo definire lo 'scale_factor' = 'target_mpp'/'level_mpp'
                if micrometer_target != None:
                    micrometer_scale_factor = micrometer_target / micrometer_lv1  

                slide_stats = {}
                # initialize optional annotation JSON dict to None
                annotation = None
                # default values for supplementaty image previews
                patch_border_color = (0, 0, 0)  
                patch_border_width = 15  # default to thick enough border
                alpha = 0.6  # Transparency factor to super-impose WSI with patch grid

                # vars to store info on the average of tissue % for the patches in all the WSI
                selected_patches = 0
                tot_percentage_patches = 0
                tot_percentage_annot_patches = 0

                # extract the wsi image name
                print(f"Segmenting image {slide_name}...")
        
                dir_path = os.path.join(self.output_patches_path, slide_name)

                # create a sub-directory with the image name inside the specified output path
                os.makedirs(dir_path, exist_ok=True)

                # create an additional sub-directory within the image name folder to contain patches/tiles
                patches_path = os.path.join(dir_path, "tiles")
                os.makedirs(patches_path, exist_ok=True)

                # create an additional sub-directory within the image name folder to contain patches/tiles of the obtained segmentation
                if save_mask_patches:
                    seg_patches_path = os.path.join(dir_path, "mask_tiles")
                    os.makedirs(seg_patches_path, exist_ok=True)

                # LOAD THE WSI IMAGE
                # self.logger.info("Loading the WSI...")
                # wsi = cv2.imread(wsi_path)

                # Estrarre l'immagine intera o una regione (x, y, w, h) della WSI
                region = slide_img.read_region((0, 0), found_level, slide_dims)  # Ottieni l'immagine intera al livello scelto

                # Convertire in un formato che OpenCV può usare (NumPy array, BGR)
                region_np = np.array(region)  # Converti la regione in array NumPy
                region_np = cv2.cvtColor(region_np, cv2.COLOR_RGBA2BGR)  # Converti da RGBA a BGR per OpenCV

                # Creare una finestra a schermo intero
                # cv2.namedWindow("Region Image", cv2.WND_PROP_FULLSCREEN)
                # cv2.setWindowProperty("Region Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                # Mostrare l'immagine estratta
                # cv2.imshow("Region Image", region_np)

                # cv2.waitKey(0)  # Attendere un input da tastiera
                # cv2.destroyAllWindows()  # Chiudere tutte le finestre apert
                                
                # SEGMENT THE WSI, obtaining a final segmentation mask for the tissue present in the image
                print("Segmenting WSI tissue...")
                #final_mask, composite_resized = self._segment_wsi(wsi=region_np)
                if micrometer_target != None:
                    resized_image = cv2.resize(region_np, (0, 0), fx=micrometer_scale_factor, fy=micrometer_scale_factor, interpolation=cv2.INTER_CUBIC)
                    final_mask, composite_resized = self._segment_wsi(wsi=resized_image)
                else:
                    final_mask, composite_resized = self._segment_wsi(wsi=region_np)

                annot_patches_path = None
                wsi_masked = None
                tissue_mask = final_mask.copy()

                # PAD THE IMGS TO BE PATCHIFIED
                pad_color = (0, 0, 0)

                if micrometer_target:
                    wsi = self._pad_img(resized_image, pad_color)
                else:
                    wsi = self._pad_img(region_np, pad_color)

                tissue_mask = self._pad_img(tissue_mask, pad_color=pad_color)

                # cv2.namedWindow("Tissue mask", cv2.WINDOW_NORMAL)
                # # Ridimensionare la finestra a 800x600 pixel (dimensioni intermedie)
                # cv2.resizeWindow("Tissue mask", 800, 600)
                # # Mostrare l'immagine nella finestra
                # cv2.imshow("Tissue mask", tissue_mask)
                # # Attendere un tasto per chiudere la finestra
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                if annotation:
                    annotation_mask = self._pad_img(annotation_mask, pad_color=(0, 0, 0))

                height, width, _ = (
                    wsi.shape
                )  # extract the new height and width of the padded wsi

                if self.save_processing_images:
                    wsi_show_patches = wsi.copy()  # image to visualize selected patches
                    # image to merge with wsi_show_patches to visualize selected patches
                    # here we will draw the patch grid and color label
                    wsi_drawn_patches = np.zeros_like(wsi, np.uint8)
                    # image to merge with the tissue mask (and annotations if present) to show the selected patches
                    wsi_mask_patches = cv2.cvtColor(tissue_mask.copy(), cv2.COLOR_GRAY2BGR)

                print("Patching WSI...")
                # ITERATE ALL THE PATCHES IN THE WSI
                patch_size = (self.patchifier_patch_size[0], self.patchifier_patch_size[0])
                patch_area = patch_size[0] * patch_size[1]
                
                # Sovrapposizione indicata come parametro 
                if overlapped :
                    # Se ho ad esempio 0.9, con (1 - overlap) avrò una sovrapposizione del 90%
                    # Perchè lo stride diventa 0.1 e quindi mi sto spostando di una porzione molto piccola a destra e sinistra. 
                    stride = (int(patch_size[0] * (1 - overlapped)), int(patch_size[1] * (1 - overlapped)))
                else:
                    stride = (int(patch_size[0] * (1 - overlapped)), int(patch_size[1] * (1 - overlapped)))

                for r in range(0, height, stride[0]):
                    for c in range(0, width, stride[0]):
                        new_r = min(r + patch_size[0], height)
                        new_c = min(c + patch_size[1], width)

                        wsi_patch = wsi[r:new_r, c:new_c]

                        mask_patch = tissue_mask[r:new_r, c:new_c]

                        # Visualizza la mask_patch
                        # Visualizza la mask_patch con OpenCV
                        # cv2.imshow('Mask Patch', mask_patch)
                        # cv2.waitKey(0)  # Attende un input da tastiera per proseguire
                        # cv2.destroyAllWindows()

                        if save_annot_patches and annotation:
                            annot_patch = annotation_mask[r:new_r, c:new_c]

                        # Drawing patches based condition, non zero-pixels in mask 
                        patch_qty = np.count_nonzero(mask_patch)
                        # print('Patch quantity : ', patch_qty)

                        print(f"Patch coordinates: ({r}, {c}), Non-zero pixels in mask: {patch_qty}")

                        mask_percentage = np.round((patch_qty / patch_area), 3)
                        #print('Mask percentage : ', mask_percentage)
                        count = 0
                        #-----------------FILTER PATCHES-----------------------:
                        background_percentage = 1 - mask_percentage
                        # Only select patches that have less or equal of the maximum background percentage
                        if background_percentage > self.max_patch_background:
                            with open('background_percentage.txt', 'a') as bkp:
                                bkp.write(f'{background_percentage}\n')
                                count +=1
                                print(count)
                            continue  # skip to the next patch if the current patch have more the allowed maximum
                        #-------------------------------------------------------:

                        # Save patch images
                        patch_name = f"{slide_name}_y{r}_x{c}.png"

                        # if background_percentage <= self.max_patch_background:
                        cv2.imwrite(os.path.join(patches_path, patch_name), wsi_patch)
                        #print('Salvo il patch avendo una percentuale di tessuto/background superiore alla soglia')

                        if save_mask_patches:  # save mask patches only if specified
                            mask_patch_name = f"{slide_name}_y{r}_x{c}.png"
                            cv2.imwrite(
                                os.path.join(seg_patches_path, mask_patch_name), mask_patch
                            )
                        if (
                            save_annot_patches and annotation
                        ):  # save annot patches only if annot present and specified
                            annot_patch_name = f"{slide_name}_y{r}_x{c}.png"
                            cv2.imwrite(
                                os.path.join(annot_patches_path, annot_patch_name), annot_patch
                            )

                        # Update slide statistics
                        slide_stats[patch_name] = {
                            "pos_y": r,
                            "pos_x": c,
                            "tissue_percentage": mask_percentage,
                        }
                        selected_patches += 1
                        tot_percentage_patches += mask_percentage

                        # EXTRACT THE PATCH LABELS AND COLORS if an annotation geojson file is present
                        if annotation:
                            patch_labels, colors, intersections = self._label_patch(
                                pos_y=r,
                                pos_x=c,
                                patch_size=patch_size,
                                annotation_file=annotation,
                                intersection_threshold=self.label_min_area_intersection,
                            )
                            # only add the patch label if the annotations are present
                            slide_stats[patch_name]["label"] = patch_labels
                            slide_stats[patch_name]["annot_intersections"] = intersections

                            if intersections:
                                labels_inters_values = np.array(list(intersections.values()))
                                labels_inters_mean_values = np.round(
                                    np.mean(labels_inters_values), 3
                                )
                                slide_stats[patch_name][
                                    "avg_annot_percentage_patch"
                                ] = labels_inters_mean_values
                                tot_percentage_annot_patches += labels_inters_mean_values

                            if self.save_processing_images:
                                # color the patch using the colors list associated with the labels
                                self._color_patch(
                                    img=wsi_drawn_patches,
                                    patch_size=patch_size,
                                    r=r,
                                    c=c,
                                    new_r=new_r,
                                    new_c=new_c,
                                    colors=colors,
                                )

                        elif self.save_processing_images:
                            # if no annotations were passed, color the patch with the default color
                            self._color_patch(
                                img=wsi_drawn_patches,
                                patch_size=patch_size,
                                r=r,
                                c=c,
                                new_r=new_r,
                                new_c=new_c,
                            )

                        if self.save_processing_images:
                            # Draw border around the entire patch for better visualization
                            cv2.rectangle(
                                wsi_drawn_patches,
                                (c, r),
                                (new_c, new_r),
                                patch_border_color,
                                patch_border_width,
                            )  # Thick white border

                # compute the avg tissue percentage of the selected patches
                avg_tissue_perc = (
                    np.round(tot_percentage_patches / selected_patches, 3)
                    if selected_patches > 0
                    else 0
                )
                slide_stats["avg_tissue_percentage"] = avg_tissue_perc

                if annotation:
                    avg_annot_percentage = (
                        np.round(tot_percentage_annot_patches / selected_patches, 3)
                        if selected_patches > 0
                        else 0
                    )
                    slide_stats["avg_annot_percentage"] = avg_annot_percentage

                slide_stats["wsi_dims"] = {
                    "height": height,
                    "width": width,
                }

                print(
                    f"WSI {slide_name} selected average tissue percentage = {avg_tissue_perc}"
                )

                # save the json slide stats for the wsi
                try:
                    with open(
                        os.path.join(self.output_patches_path, slide_name, "wsi_metadata.json"),
                        "w",
                    ) as f:
                        json.dump(slide_stats, f, indent=4)
                except Exception as e:
                    print(f"An unexpected error occurred: {e}", exc_info=True)

                if self.save_processing_images:
                    print(f"Saving processing images for WSI {slide_name} ...")
                    # fuse the wsi tissue image with the drawn grid of selected patches with label color
                    cv2.addWeighted(
                        wsi_show_patches,
                        1 - alpha,
                        wsi_drawn_patches,
                        alpha,
                        0,
                        wsi_show_patches,
                    )

                    # fuse the wsi tissue mask image and the colored annotation mask image with the drawn grid of selected patches with label color
                    if annotation:
                        wsi_mask_patches = np.where(
                            np.any(annotation_mask > 0, axis=-1, keepdims=True),
                            annotation_mask,
                            wsi_mask_patches,
                        )
                    cv2.addWeighted(
                        wsi_mask_patches,
                        1 - alpha,
                        wsi_drawn_patches,
                        alpha,
                        0,
                        wsi_mask_patches,
                    )

                    # Save the resulting superimposed patches and original + mask images
                    wsi_show_patches = cv2.resize(
                        wsi_show_patches,
                        (width // self.reduction_ratio, height // self.reduction_ratio),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    wsi_mask_patches = cv2.resize(
                        wsi_mask_patches,
                        (width // self.reduction_ratio, height // self.reduction_ratio),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    wsi_with_patches_path = os.path.join(
                        dir_path, slide_name + "_patches.jpeg"
                    )
                    cv2.imwrite(wsi_with_patches_path, wsi_show_patches)
                    print(
                        f"WSI with patches preview saved at {wsi_with_patches_path}"
                    )

                    mask_with_patches_path = os.path.join(
                        dir_path, slide_name + "_mask_patches.jpeg"
                    )
                    cv2.imwrite(
                        mask_with_patches_path,
                        wsi_mask_patches,
                    )
                    print(
                        f"WSI MASK with patches preview saved at {mask_with_patches_path}"
                    )

                    # Save the resized composite image
                    save_image_path = os.path.join(dir_path, slide_name + "_preview.jpeg")
                    cv2.imwrite(save_image_path, composite_resized)
                    print(f"Composite image saved at {save_image_path}")

                    # save full resolution mask of the wsi tissue
                    save_mask_path = os.path.join(dir_path, slide_name + "_mask.png")
                    cv2.imwrite(save_mask_path, final_mask)
                    print(f"Full resolution mask image saved at {save_mask_path}")

                    # check if the annotation were present and the annotated wsi and annotation masks were created
                    if annotation:

                        # save full resolution wsi annotation annotation_mask image
                        save_annotated_mask_path = os.path.join(
                            dir_path, slide_name + "_annotated_mask.png"
                        )
                        cv2.imwrite(save_annotated_mask_path, annotation_mask)
                        print(
                            f"Full res. annotation mask saved at {save_annotated_mask_path}"
                        )

                        # save reduced resolution original wsi image with annotations
                        save_annotated_wsi = os.path.join(
                            dir_path, slide_name + "_annotated.jpeg"
                        )
                        wsi_masked_resized = cv2.resize(
                            wsi_masked,
                            (width // self.reduction_ratio, height // self.reduction_ratio),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        cv2.imwrite(save_annotated_wsi, wsi_masked_resized)
                        print(f"Annotated WSI image saved at {save_annotated_wsi}")

        #return slide_stats
        

    def create_wsi_info_dictionary(self, wsi_zoom_level : int, output_dictionary_path : str):

        wsi_info_dictionary = {}

        for slide in sorted(os.listdir(self.wsi_base_path)):
            
                print('Inserisco le info per la slide : ', slide)
                single_slide_path = os.path.join(self.wsi_base_path, slide)
                # Controlla se è un file e non una cartella
                if os.path.isfile(single_slide_path):
                    slide_img = open_slide(single_slide_path)
                    slide_name = os.path.splitext(slide)[0]
                    slide_dims = slide_img.dimensions

                    # Scelgo il livello 
                    try:
                        slide_level_dims, _ = self.get_slide_element_at_level(slide_img, wsi_zoom_level)
                    except:
                        if wsi_zoom_level > 0:
                            slide_level_dims, _ = self.get_slide_element_at_level(slide_img, wsi_zoom_level - 1)

                    # Estrai i micron per pixel in x 
                    mpp_x = float(slide_img.properties['openslide.mpp-x'])
                    
                    # La funzione downsamples restituisce il fattore di downsampling che c'è tra un livello e l'altro, questo valore viene poi moltiplicato per il valore di mpp calcolato al livello 0
                    mpp_x_level = slide_img.level_downsamples[wsi_zoom_level] * mpp_x  # Calcola il micron per pixel per il livello desiderato
                    
                    # Ricorda che prima di assegnare valori ad una certa chiave del dizionario bisonga inizializzare il dizionario per quella chiave 
                    if slide not in wsi_info_dictionary:
                        wsi_info_dictionary[slide] = {}

                    # Dopo aver inizializzato il dizionario per quella chiave posso accedere ai singoli valori della slide
                    wsi_info_dictionary[slide] = {
                        'Name' : slide_name, 
                        'Slide_LV0_dims' : slide_dims,
                        'Slide_LV1_dims' : slide_level_dims, 
                        'Micron_per_pixel_x_LV0' : mpp_x,
                        'Micron_per_pixel_x_LV1' : mpp_x_level,
                    }
                    # Salva il dizionario completo in JSON
                with open(output_dictionary_path, 'w') as f:
                    json.dump(wsi_info_dictionary, f, indent=2)


    def compute_bounding_boxes(self, source_qpath_annotations_folder, computed_patches_folder : str, final_yolo_annotations_folder : str, boxed_patches_path : str, threshold_patch_bb_intersection : float, wsi_dimensions_file : str, target_micrometer_value : float ):
            
            # Scrittura delle coordinate nel formato YOLO
            initial_final_yolo_annotations_folder = final_yolo_annotations_folder
            # Per evitare che si concateni ongi volta slide_names
            initial_patches_path = boxed_patches_path

            # ∀ ANNOTAZIONE GENERATA CON QPATH
            for annotated_slide in sorted(os.listdir(source_qpath_annotations_folder)):

                # COSTRUISCO IL NOME DEL FILE CHE CONTIENE L'ANNOTAZIONE 
                file_path = os.path.join(source_qpath_annotations_folder, annotated_slide)
                # PRENDO IL NOME DELLA WSI CORRENTE SENZA ESTENSIONE, MI SERVE PER REPERIRE LE ALTRE INFO (DIMENSIONI E PATCHES)
                slide_name = os.path.splitext(annotated_slide)[0]
                # QUA MI SALVERO' LE COORDINARE DELLE BOUNDING BOX CHE LEGGO
                bounding_box_coordinates = {}
                # APRO IL FILE DELLE ANNOTAZIONI
                with open(file_path, 'r',  encoding='utf-8') as f:
                    data = json.load(f)

                if slide_name not in bounding_box_coordinates:
                    bounding_box_coordinates[slide_name] = []  

                # LEGGO LE ANNOTAZIONI 
                for dict in data['features']:

                    points = dict['geometry']['coordinates']
                    x_coordinates = [value[0] for value in points[0]]
                    y_coordinates = [value[1] for value in points[0]]
                    minimun_x_value = min(x_coordinates)
                    maximum_x_value = max(x_coordinates)
                    minimun_y_value = min(y_coordinates)
                    maximum_y_value = max(y_coordinates)
                   
                    bounding_box_coordinates[slide_name].append(
                    { 
                        'min_x_coord' : minimun_x_value,
                        'max_x_coord' : maximum_x_value,
                        'min_y_coord' : minimun_y_value,
                        'max_y_coord' : maximum_y_value })
                    
                # GIUSTO PER CONTROLLARE CHE IL NUMERO DI BOUNDING BOX SIA CORRETTO
                # with open('saved_dictionary.pkl', 'wb') as f:
                #     pickle.dump(bounding_box_coordinates, f)
                # # Apri il file pickle in modalità lettura binaria
                # with open('saved_dictionary.pkl', 'rb') as f:
                #     # Carica l'oggetto dal file
                #     data = pickle.load(f)

                # total_bounding_boxes = 0
                # # Stampa il contenuto dell'oggetto
                # for slide_name, bounding_box_coordinates in data.items():
                #     total_bounding_boxes += len(bounding_box_coordinates)
                # print ("Il numero totale di bounding box trovate è : ", total_bounding_boxes)
    
                # PRENDO LE DIMENSIONI DELLA IMMAGINE
                # with open(images_info_dictionary_file, 'r',  encoding='utf-8') as d:
                #     dictionary = json.load(d)
               
                # ∀ PATCH CHE HO NELLA MIA CARTELLA DEI PATCHES GENERATI 
                # METTIAMO CHE LA GERARCHIA DELLE CARTELLE SIA LA SEGUENTE
                    # -- output_files
                    # ----------- wsi_name
                    # ----------------- tiles
                    # ------------------------ patch1_name_x_coord_y_coord
                    # ------------------------ patch2_name_x_coord_y_coord
              
                patches_path = os.path.join(computed_patches_folder, slide_name , 'tiles')
                # Verifica se la cartella delle patch esiste
                if os.path.exists(patches_path) and os.path.isdir(patches_path):
                    for patch in os.listdir(patches_path):
                        p_name = os.path.splitext(patch)[0]
                        # IL PUNTO IN ALTO A SINISTRA VIENE INDICATO COME X_MIN E Y_MIN
                        x_min_patch_coordinate = int(p_name.split('_')[-1].replace('x',''))
                        y_min_patch_coordinate = int(p_name.split('_')[-2].replace('y',''))

                        #  CONTROLLO SE IL SINGOLO PATCH INTERSECA UNA BOUNDING BOX
                        #  QUINDI PRENDO IL MIO DIZIONARIO DOVE HO SALVATO LE COORDINATE DELLE BOUNDING BOX DELLA MIA INTERA WSI
                        #  E CONTROLLO CHE NON INTERSECHI CON LE COORDINATE DEL MIO PATCH
                        
                        #  QUESTA PARTE SERVE PERCHE' VOGLIO LO SCALE FACTOR TRA IL LIVELLO 0 E IL LIVELLO SPECIFICATO (1 NEL MIO CASO)
                        lv0_width, lv1_width, micrometer_lv0, micrometer_lv1 =  get_dimensions_for_key(wsi_dimensions_file, annotated_slide)
                        scale_factor =  lv0_width/lv1_width
                        #print('Questo è il valore micron a livello 0 : ', micrometer_lv0)
                        #print('Questo è il valore micron a livello 1 : ', micrometer_lv1)
                        #print('Questo è il fattore di scale : ', scale_factor)

                        for bbox in bounding_box_coordinates[slide_name]:
                            
                            # Estrai i valori dal dizionario
                            bb_min_x = int(bbox['min_x_coord'] / scale_factor)
                            bb_max_x = int(bbox['max_x_coord'] / scale_factor)
                            bb_min_y = int(bbox['min_y_coord'] / scale_factor)
                            bb_max_y = int(bbox['max_y_coord'] / scale_factor)
                            if target_micrometer_value:

                                micrometer_scale_factor = target_micrometer_value / micrometer_lv1  
                                # Ridimensiona le coordinate delle bounding box in base al micrometer_scale_factor
                                bb_min_x = int(bb_min_x * micrometer_scale_factor)
                                bb_max_x = int(bb_max_x * micrometer_scale_factor)
                                bb_min_y = int(bb_min_y * micrometer_scale_factor)
                                bb_max_y = int(bb_max_y * micrometer_scale_factor)
                                        
                            # AVENDO IL PUNTO IN ALTO A SINISTRA DEL MIO PATCH DEVO CALCOLARE IL PUNTO IN BASSO A DESTRA    
                            patch_width = self.patchifier_patch_size[0]
                            patch_height = self.patchifier_patch_size[0]
                            
                            x_max_patch_coordinate = x_min_patch_coordinate + patch_width
                            y_max_patch_coordinate = y_min_patch_coordinate + patch_height

                            if (bb_min_x >= x_min_patch_coordinate and bb_max_x <= x_max_patch_coordinate and
                                bb_min_y >= y_min_patch_coordinate and bb_max_y <= y_max_patch_coordinate):
                                print(f'Bounding box completamente contenuta nel patch {patch}.')
                                inter_area = (bb_max_x - bb_min_x) * (bb_max_y - bb_min_y)
                                print(f'Area di intersezione per bounding box contenuta: {inter_area}')

                            # VERIFICO SE C'E' INTERSEZIONE TRA I DUE RETTANGOLI, IL PATCH E LA BOUNDING BOX
                            if (x_min_patch_coordinate < bb_max_x and x_max_patch_coordinate > bb_min_x and y_min_patch_coordinate < bb_max_y and y_max_patch_coordinate > bb_min_y):

                                    inter_x_min = max(x_min_patch_coordinate,  bb_min_x)
                                    inter_y_min = max(y_min_patch_coordinate, bb_min_y)
                                    inter_x_max = min(x_max_patch_coordinate, bb_max_x)
                                    inter_y_max = min(y_max_patch_coordinate, bb_max_y)

                                    # CALCOLO ORA L'AREA DI INTERSEZIONE 
                                    inter_width = inter_x_max - inter_x_min
                                    inter_height = inter_y_max - inter_y_min

                                    if inter_width > 0 and inter_height > 0:
                                        inter_area = inter_width * inter_height
                                    else:
                                        inter_area = 0  # Non c'è intersezione
                    
                                    # QUA BISOGNA FILTRARE PER L'AREA DI INTERSEZIONE, CIOE' SE HO UN' AREA DI INTERSEZIONE INFERIORE A UN TOT DEVO SCARTARE IL PATCH E NON GENERARE L'ANNOTAZIONE
                                    # SCARTO LE INTERSEZIONI CHE CORRISPONDONO A MENO DEL x PERCENTO DEL PATCH

                                    patch_area_ths = patch_height * patch_width * threshold_patch_bb_intersection

                                
                                    # if inter_area > patch_area_ths or (intersection_polygon.contains(patch_polygon) and intersection_area <= 0.3 * patch_area):
                                    if inter_area >= patch_area_ths:
                                        print('Intersection Area of patch {}'.format(patch), inter_area)

                                        # ORA DEVO TRASFORMARE QUESTE COORDINATE IN COORDINATE ADATTE A YOLO
                                        # Se esce dai bordi la riporto nei bordi del mio patch
                                        adapted_bb_min_x = max(bb_min_x, x_min_patch_coordinate)
                                        adapted_bb_max_x = min(bb_max_x, x_max_patch_coordinate)
                                        adapted_bb_min_y = max(bb_min_y, y_min_patch_coordinate)
                                        adapted_bb_max_y = min(bb_max_y, y_max_patch_coordinate)

                                        # FACCIAMO LA PROVA DISEGNANDOLE SUL PATCH PER VEDERE SE VANNO BENE
                                        
                                        # PERCHE' RECTANGLE LE DISEGNA RELATIVE AL PATCH E QUINDI BISOGNA RIADEGUARLE
                                        # Clipping delle coordinate

                                        relative_bb_min_x = adapted_bb_min_x - x_min_patch_coordinate
                                        relative_bb_max_x = adapted_bb_max_x - x_min_patch_coordinate
                                        relative_bb_min_y = adapted_bb_min_y - y_min_patch_coordinate
                                        relative_bb_max_y = adapted_bb_max_y - y_min_patch_coordinate

                                        # QUESTA COSA NON SERVE A MOLTO PER LA VISUALIZZAZIONE, TANTO SONO GIA' CERTO CHE NON ESCA DALLE DIMENSIONI DEL PATCH
                                        relative_bb_min_x = max(0, relative_bb_min_x)
                                        relative_bb_max_x = min(patch_width, relative_bb_max_x)
                                        relative_bb_min_y = max(0, relative_bb_min_y)
                                        relative_bb_max_y = min(patch_height, relative_bb_max_y)
                                        
                                        print('I valori della bb adattata sono : ', relative_bb_min_x, relative_bb_max_x, relative_bb_min_y, relative_bb_max_y, ' il patch è : ', patch)
                                        full_patch_path = os.path.join(patches_path, patch)

                                        save_path = os.path.join(initial_patches_path, slide_name)
                                        # Controlla se esiste già un boxed patch salvato
                                        boxed_patch_path = '{}/{}.png'.format(save_path, p_name)
                                        if os.path.exists(boxed_patch_path):
                                            # Carica l'immagine con le bounding box già disegnate
                                            image_with_boxes = cv2.imread(boxed_patch_path)
                                            print('Carico un patch boxato già esistente')
                                        else:
                                            # Se non esiste, carica il patch originale
                                            image_with_boxes = cv2.imread(full_patch_path)

                                        # Disegna la nuova bounding box sull'immagine caricata
                                        image_with_boxes = cv2.rectangle(image_with_boxes, 
                                                                        (int(relative_bb_min_x), int(relative_bb_min_y)), 
                                                                        (int(relative_bb_max_x), int(relative_bb_max_y)), 
                                                                        (0, 0, 255), 3)

                                        print("Creo in 'Boxed_patches' le cartelle dove si visualizzano i patches che contengono i glomeruli con la bounding box")
                                        boxed_patches_path = os.path.join(initial_patches_path, slide_name)
                                        os.makedirs(boxed_patches_path, exist_ok=True)

                                        print("Salvo i boxed patches")
                                        cv2.imwrite(boxed_patch_path, image_with_boxes)

                                        # ORA DEVO CONVERTIRE QUESTA BOUNDING BOX IN YOLO COORDINATES E GENERARE IL TXT CON LO STESSO NOME      
                                        # ORA DEVO CONVERTIRE QUESTE COORDINATE IN YOLO COORDINATES E SALVARE IN UN FILE TXT CHE HA LO STESSO NOME 
                                        # calcolo il centro della mia bb
                                        x_c = (relative_bb_min_x + relative_bb_max_x) / 2
                                        y_c = (relative_bb_min_y + relative_bb_max_y) / 2
                                        # proporziono le coordinate 
                                        bb_width = relative_bb_max_x - relative_bb_min_x
                                        bb_height = relative_bb_max_y - relative_bb_min_y
                                        x_normalized_center = x_c / patch_width
                                        y_normalized_center = y_c / patch_height
                                        bb_width_normalized = bb_width / patch_width
                                        bb_height_normalized = bb_height / patch_height

                                        # Scrivo nella cartella le annotazioni di yolo
                                        os.makedirs(initial_final_yolo_annotations_folder, exist_ok=True)
                                        folder_slide_path = os.path.join(initial_final_yolo_annotations_folder, slide_name)
                                        os.makedirs(folder_slide_path, exist_ok=True)
                                        file_annotation_folder = os.path.join(initial_final_yolo_annotations_folder, slide_name, '{}.txt'.format(p_name)) 
                                        with open(file_annotation_folder, 'a') as f:
                                            f.write(f'0 {x_normalized_center} {y_normalized_center} {bb_width_normalized} {bb_height_normalized}\n')     
                        else:
                          
                            # Scrittura delle coordinate nel formato YOLO senza annotazione
                            folder_slide_path = os.path.join(initial_final_yolo_annotations_folder, slide_name)
                            os.makedirs(folder_slide_path, exist_ok=True)
                            file_annotation_folder = os.path.join(initial_final_yolo_annotations_folder, slide_name, '{}.txt'.format(p_name)) 
                            with open(file_annotation_folder, 'a') as f:
                                # semplicemente non scrivo nulla
                                pass
                else: 
                    print('Questa annotazione non ha patches corrispondenti', annotated_slide)


def save_to_yaml(data, output_path):
    with open(output_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

def get_dimensions_for_key(json_file, key):
    
    with open(json_file, 'r') as f:
        data = json.load(f)

    key = os.path.splitext(key)[0]
    for k in data.keys():
        if os.path.splitext(k)[0] == key:
            found_key = os.path.splitext(k)[0]  
            key_1 = found_key + '.ndpi'
            key_2 = found_key + '.svs'  

            try:
                lv_0_dims = data[key_1].get("Slide_LV0_dims", None)[0]  
                lv_1_dims = data[key_1].get("Slide_LV1_dims", None)[0]
                micrometer_lv0 = data[key_1].get("Micron_per_pixel_x_LV0", None)
                micrometer_lv1 = data[key_1].get("Micron_per_pixel_x_LV1", None)
            except KeyError:
                print(f"Chiave mancante in JSON per {key_1}")
                lv_0_dims = data[key_2].get("Slide_LV0_dims", None)[0]  
                lv_1_dims = data[key_2].get("Slide_LV1_dims", None)[0]
                micrometer_lv0 = data[key_2].get("Micron_per_pixel_x_LV0", None)
                micrometer_lv1 = data[key_2].get("Micron_per_pixel_x_LV1", None)
                return lv_0_dims, lv_1_dims, micrometer_lv0, micrometer_lv1

            return lv_0_dims, lv_1_dims, micrometer_lv0, micrometer_lv1
    
def main():

    # I PARAMETRI DI INPUT POSSONO ESSERE PASSATI O DA CONFIG FILE OPPURE DIRETTAMENTE NEL MAIN, QUANDO SARA' RIFINITO BENE LASCER0' SOLO IL CONFIG FILE 
    
    # ------------------------------------CONFIG FILE----------------------------------------------------------------
    # parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    # parser.add_argument('--WSI_FOLDER', type=str, help="wsi input folder")
    # parser.add_argument('--WSI_ANNOTATIONS_FOLDER', type=str, help="wsi annotations folder")
    # parser.add_argument('--PATCH_SIZE', type=str, help="patch size")
    # parser.add_argument('--MAX_PATCH_BACKGROUND', type=str, help="maximum allowed background to keep a patch")
    # parser.add_argument('--THRESHOLD_PATCH_BB_INTERSECTION', type=str, help="intersection value to keep a patch")
    # parser.add_argument('--PATCHES_FOLDER', type=str, help="folder containing the patches")
    # parser.add_argument('--OUTPUT_WSI_YAML_PATH', type=str, help="output wsi yaml path")
    # parser.add_argument('--BOXED_PATCHES_PATH', type=str, help="boxed patches path")
    # parser.add_argument('--FINAL_YOLO_ANNOTATIONS_FOLDER', type=str, help="Wsi_input_folder")
    # parser.add_argument('--WSI_DIMENSIONS_FILE', type=str, help="wsi dimensions for each level")
  
    # # Parsing dei parametri nel file config.conf
    # args = parser.parse_args(['@detection_config.conf'])
    # print(args)
    #------------------------------------------------------------------------------------------------------------------
    
    #---------------------------------------ESEMPIO DI CONFIG FILE ----------------------------------------------------
    # INPUT
    WSI_FOLDER = "/work/grana_pbl/Detection_Glomeruli/3DHISTECH"
    WSI_ANNOTATIONS_FOLDER = "/work/grana_pbl/Detection_Glomeruli/Coordinate_3DHISTECH"
    # WSI_ANNOTATIONS_FOLDER = "/work/grana_pbl/Detection_Glomeruli/prova_calcolo_annotazione"
    PATCH_SIZE = 1024,
    MAX_PATCH_BACKGROUND = 0.8
    THRESHOLD_PATCH_BB_INTERSECTION = 0
    # Questa serve per la dilation
    ITERATIONS = 20
    # Questi sono i valori di luminanza dei pixels minima e massima
    LOWER_BOUND_PIXEL_VALUE = 30
    UPPER_BOUND_PIXEL_VALUE = 200
    # OUTPUT
    PATCHES_FOLDER  = "/work/grana_pbl/Detection_Glomeruli/Output_patchified_files_folder/Output_patchified_files_Lv1_3D_Overlapping05"
    # PATCHES_FOLDER = "/work/grana_pbl/Detection_Glomeruli/prova_single_patch"
    BOXED_PATCHES_PATH = "/work/grana_pbl/Detection_Glomeruli/Boxed_patches_folder/Boxed_patches_Lv1_Scale_05_Overlapped"
    #BOXED_PATCHES_PATH = "/work/grana_pbl/Detection_Glomeruli/Boxed_patches_prova"
    FINAL_YOLO_ANNOTATIONS_FOLDER = "/work/grana_pbl/Detection_Glomeruli/Final_yolo_annotations_wsi_folder/Final_patches_yolo_annotations_folder_Lv1_scale_05_Overlapped"
    #WSI_DIMENSIONS_OUTPUT_FILE = "/work/grana_pbl/Detection_Glomeruli/dimensions_dictionary.json"
    WSI_INFORMATION_FILE = "/work/grana_pbl/Detection_Glomeruli/INFO_wsi_file_dictionary_3D.yaml"
    MICROMETER_TARGET = None # Settalo a 'None', se non vuoi avere le wsi con gli stessi valori di Micron per pixel.
    # Livello di definizione della mia wsi
    ZOOM_LEVEL = 1
    OVERLAPPING = 0.5
    #------------------------------------------------------------------------------------------------------------------
   
    patchifier = Patchifier(
        wsi_base_path = WSI_FOLDER,
        output_patches_path = PATCHES_FOLDER,
        patchifier_patch_size = PATCH_SIZE,
        max_patch_background = MAX_PATCH_BACKGROUND,
        annotations_base_path = WSI_ANNOTATIONS_FOLDER,
        iterations = ITERATIONS,
        lower_bound_pixel_value = LOWER_BOUND_PIXEL_VALUE,
        upper_bound_pixel_value = UPPER_BOUND_PIXEL_VALUE,
        patchification_level = ZOOM_LEVEL,
    )

    # patchifier.create_wsi_info_dictionary(
    #     wsi_zoom_level = 1,
    #     output_dictionary_path = WSI_INFORMATION_FILE,
    # )

    #patchifier.patchify_wsi(wsi_dict_path = WSI_INFORMATION_FILE, micrometer_target = MICROMETER_TARGET, overlapped = OVERLAPPING)

    patchifier.compute_bounding_boxes(
        source_qpath_annotations_folder = WSI_ANNOTATIONS_FOLDER,
        computed_patches_folder = PATCHES_FOLDER,
        boxed_patches_path = BOXED_PATCHES_PATH,
        final_yolo_annotations_folder = FINAL_YOLO_ANNOTATIONS_FOLDER,
        threshold_patch_bb_intersection = THRESHOLD_PATCH_BB_INTERSECTION,
        wsi_dimensions_file =  WSI_INFORMATION_FILE,
        target_micrometer_value = MICROMETER_TARGET,
    )

    print("\nTerminating...")

if __name__ == "__main__":
    main()