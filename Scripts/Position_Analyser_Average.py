import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Chemin vers le dossier contenant les images
image_folder_path = './100_Particules_Model_Standard'  # Remplacez par le chemin correct

# Taille de la grille de densité (heatmap)
grid_size = (1000, 1000)  # Ajustez selon la résolution des images

# Initialiser une grille pour cumuler les positions de particules
density_grid = np.zeros(grid_size)

# Fonction pour obtenir les positions des particules dans une image
def get_particle_positions(image_path):
    image = Image.open(image_path).convert('L')  # Convertir en niveau de gris
    data = np.array(image)
    threshold = data < 128  # Particules sombres
    positions = np.argwhere(threshold)  # Obtenir les positions
    return positions

# Parcourir chaque image et cumuler les positions des particules dans la grille de densité
for image_name in os.listdir(image_folder_path):
    if image_name.endswith('.jpg'):
        image_path = os.path.join(image_folder_path, image_name)
        positions = get_particle_positions(image_path)
        
        # Mettre à jour la grille de densité
        for pos in positions:
            if 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]:  # Vérifier les limites
                density_grid[pos[0], pos[1]] += 1

# Visualiser la heatmap des positions
plt.figure(figsize=(10, 10))
plt.imshow(density_grid, cmap='hot', interpolation='nearest')
plt.colorbar(label='Nombre de particules')
plt.title("Carte de densité des positions des particules à travers les simulations")
plt.xlabel("Position X")
plt.ylabel("Position Y")
plt.show()
