import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Chemin vers le dossier contenant les images
image_folder_path = './simulations_output'  # Remplacez par le chemin correct pour le modèle Janus

# Taille de la grille de densité (heatmap)
grid_size = (500, 500)  # Ajustez selon la résolution des images pour correspondre aux particules visibles

# Initialiser deux grilles pour cumuler les positions de particules positives et négatives
density_grid_positive = np.zeros(grid_size)
density_grid_negative = np.zeros(grid_size)

# Fonction pour obtenir les positions et types des particules (positives ou négatives) dans une image
def get_particle_positions_and_types(image_path, crop_box=None):
    image = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
    if crop_box:
        image = image.crop(crop_box)  # Recadrer l'image pour éviter les bordures

    data = np.array(image)
    threshold = data < 128  # Particules sombres
    positions = np.argwhere(threshold)  # Obtenir les positions des particules sombres

    # Détermination du type de particule basé sur un seuil de luminosité
    particle_types = (data[threshold] < 64).astype(int) * 2 - 1  # -1 pour négatif, +1 pour positif
    return positions, particle_types

# Définir la zone de recadrage pour ignorer les axes et titres
crop_box = (100, 100, 900, 900)  # Ajustez selon la zone de particules

# Parcourir chaque image et cumuler les positions des particules dans les grilles de densité
for image_name in os.listdir(image_folder_path):
    if image_name.endswith('.jpg'):
        image_path = os.path.join(image_folder_path, image_name)
        positions, particle_types = get_particle_positions_and_types(image_path, crop_box)
        
        # Mettre à jour les grilles de densité
        for pos, p_type in zip(positions, particle_types):
            if 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]:  # Vérifier les limites
                if p_type > 0:
                    density_grid_positive[pos[0], pos[1]] += 1
                else:
                    density_grid_negative[pos[0], pos[1]] += 1

# Superposer les cartes de densité pour chaque type de particule
plt.figure(figsize=(10, 10))

# Affichage de la densité des masses positives en rouge
plt.imshow(density_grid_positive, cmap='Reds', interpolation='nearest', alpha=0.6, vmin=0, vmax=2)

# Affichage de la densité des masses négatives en bleu
plt.imshow(density_grid_negative, cmap='Blues', interpolation='nearest', alpha=0.4, vmin=0, vmax=2)

# Ajouter une barre de couleur personnalisée
plt.colorbar(label="Nombre de particules (positives en rouge, négatives en bleu)", fraction=0.046, pad=0.04)
# plt.colorbar(label="Nombre de particules (positives en rouge, négatives en bleu)")
plt.title("Carte de densité des positions des particules (Modèle Janus)")
plt.xlabel("Position X")
plt.ylabel("Position Y")
plt.show()
