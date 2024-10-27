import numpy as np
import matplotlib.pyplot as plt

# Constantes de la simulation
G = 6.67430e-11
num_particles = 500
time_step = 1e2  # Pas de temps en secondes
num_steps = 500  # Nombre d'étapes de simulation
epsilon = 1e8  # Pour éviter les divisions par zéro
expansion_rate = 1.0001  # Facteur d'expansion par étape

# Définir le nombre de simulations à exécuter
num_simulations = 50  # Par exemple, 50 simulations

# Fonction pour calculer la force gravitationnelle
def gravitational_force(pos1, pos2, m1, m2):
    distance = np.linalg.norm(pos1 - pos2) + epsilon
    force_magnitude = G * np.abs(m1 * m2) / distance**2
    direction = (pos2 - pos1) / distance
    return force_magnitude * direction

# Exécuter plusieurs simulations
for sim_num in range(1, num_simulations + 1):  # De 1 à num_simulations inclus
    # Initialisation aléatoire des particules pour chaque simulation
    np.random.seed(sim_num)  # Changer la graine pour chaque simulation pour des résultats différents
    positions = np.random.randn(num_particles, 2) * 1e18  # Positions aléatoires
    velocities = np.random.randn(num_particles, 2) * 1e3  # Vitesses aléatoires
    masses = np.random.randn(num_particles) * 1e30  # Masses positives ou négatives

    # Ajout d'une rotation initiale autour d'un centre pour simuler un disque galactique
    center = np.mean(positions, axis=0)
    for i in range(num_particles):
        distance_to_center = positions[i] - center
        tangential_velocity = np.array([-distance_to_center[1], distance_to_center[0]])  # Vitesse tangentielle
        velocities[i] += tangential_velocity * 1e-4  # Ajuster ce facteur selon la rotation souhaitée

    # Boucle de simulation pour chaque étape
    plt.figure(figsize=(8, 8))
    for step in range(num_steps):
        forces = np.zeros((num_particles, 2))

        # Calcul des forces gravitationnelles
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                force = gravitational_force(positions[i], positions[j], masses[i], masses[j])
                forces[i] += force
                forces[j] -= force

        # Mise à jour des vitesses et positions
        velocities += forces / masses[:, np.newaxis] * time_step
        positions += velocities * time_step
        positions *= expansion_rate  # Expansion de l'univers

        # Calcul du temps écoulé en secondes
        time_seconds = step * time_step

    # Sauvegarde du rendu final de chaque simulation
    plt.clf()
    colors = ['blue' if m < 0 else 'red' for m in masses]  # Bleu pour masses négatives, rouge pour positives
    plt.scatter(positions[:, 0], positions[:, 1], s=1, c=colors)
    plt.xlim(-5e18, 5e18)
    plt.ylim(-5e18, 5e18)
    plt.title(f"Simulation {sim_num} | Dernière étape | Temps écoulé : {time_seconds:.0f} secondes")
    plt.savefig(f"Simulation_N_{sim_num}.jpg", format='jpg')  # Sauvegarde de l'image
    plt.close()  # Fermer la figure pour libérer de la mémoire
