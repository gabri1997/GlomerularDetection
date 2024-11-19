from openslide import open_slide
import numpy as np
import matplotlib.pyplot as plt

slide = open_slide('/work/grana_pbl/Detection_Glomeruli/3DHISTECH/R22-90 C3.svs')
#slide = open_slide('/work/grana_pbl/Detection_Glomeruli/HAMAMATSU/R23 209_2A1_C3-FITC.ndpi')
slide_props = slide.properties

print("Vendor is:", slide_props['openslide.vendor'])
print("Pixel size of X is", slide_props["openslide.mpp-x"])
print("Pixel size of Y is", slide_props["openslide.mpp-y"])

# objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
# # Essendo objective power di 20 vuol dire che l'ingrandimento è a 20x quindi abbiamo che un pixel corrispondono a 0.5 Micron
# print("The objective power is:", objective)

slide_dims = slide.dimensions
print(slide_dims)

slide_thumb_600 = slide.get_thumbnail(size=(600, 600))
#slide_thumb_600.show()

dims = slide.level_dimensions

num_levels = len(dims)
print("Number of levels in this imag are:", num_levels)
print("Dimensions of various levels in this image are", dims)

# Copio una immagine di un livello a scelta 
level1_dim = dims[2]
# Ricorda che l'output sarà una immagine RGBA 
level1_img = slide.read_region((0,0), 2, level1_dim)

# Converti l'immagine a RGB
level1_img_RGB = level1_img.convert('RGB')
#level1_img_RGB.show()

# Convert the image into a numpy array for preprocessing 
level1_img_np = np.array(level1_img_RGB)
plt.imshow(level1_img_np)
plt.show()
plt.savefig('slide_svs.png')