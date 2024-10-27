import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Manager, Queue, Value, Process
import os
import signal
import sys

# Constantes de la simulation
G = 6.67430e-11
time_step = 1e2  # Pas de temps en secondes
epsilon = 1e8  # Pour éviter les divisions par zéro
expansion_rate = 1.0001  # Facteur d'expansion par étape

# Demander à l'utilisateur le nombre de simulations, d'étapes et de particules
try:
    num_simulations = int(input("Entrez le nombre de simulations à exécuter : "))
    num_steps = int(input("Entrez le nombre d'étapes par simulation : "))
    num_particles = int(input("Entrez le nombre de particules que vous souhaitez générer : "))
except ValueError:
    print("Veuillez entrer des nombres entiers valides pour les paramètres.")
    sys.exit(1)

# Créer un répertoire pour les images de simulation
os.makedirs("simulations_output", exist_ok=True)

# Initialisation d'un drapeau d'arrêt partagé entre processus
stop_flag = Value('b', False)

# Nombre maximum de processus simultanés
max_processes = 5

# Fonction pour gérer l'interruption par l'utilisateur
def signal_handler(sig, frame):
    print("\nInterruption détectée ! Arrêt des simulations en cours...")
    stop_flag.value = True

# Enregistrer le gestionnaire de signal
signal.signal(signal.SIGINT, signal_handler)

# Fonction pour calculer la force gravitationnelle avec le modèle Janus
def gravitational_force(pos1, pos2, m1, m2):
    distance = np.linalg.norm(pos1 - pos2) + epsilon
    force_magnitude = G * np.abs(m1 * m2) / distance**2
    direction = (pos2 - pos1) / distance
    return force_magnitude * direction if m1 * m2 > 0 else -force_magnitude * direction

# Fonction pour exécuter une seule simulation
def run_simulation(sim_num, queue, stop_flag):
    start_time = time.time()
    
    # Initialisation aléatoire des particules
    np.random.seed(sim_num)
    positions = np.random.randn(num_particles, 2) * 1e18
    velocities = np.random.randn(num_particles, 2) * 1e3
    masses = np.random.randn(num_particles) * 1e30  # Masses positives et négatives pour le modèle Janus
    
    # Rotation initiale pour simuler un disque galactique
    center = np.mean(positions, axis=0)
    for i in range(num_particles):
        distance_to_center = positions[i] - center
        tangential_velocity = np.array([-distance_to_center[1], distance_to_center[0]])
        velocities[i] += tangential_velocity * 1e-4
    
    # Boucle de simulation pour chaque étape
    for step in range(num_steps):
        if stop_flag.value:  # Vérifie si une interruption est demandée
            print(f"Simulation {sim_num} interrompue à l'étape {step}.")
            return
        
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

        # Envoi de la progression à la file
        queue.put((sim_num, step + 1, num_steps, time.time() - start_time))
    
    # Sauvegarder le rendu final de la simulation
    plt.figure(figsize=(8, 8))
    colors = ['blue' if m < 0 else 'red' for m in masses]  # Bleu pour masses négatives, rouge pour positives
    plt.scatter(positions[:, 0], positions[:, 1], s=1, c=colors)
    plt.xlim(-5e18, 5e18)
    plt.ylim(-5e18, 5e18)
    plt.title(f"Simulation Model Janus {sim_num} | Dernière étape")
    plt.savefig(f"simulations_output/Simulation_N_{sim_num}.jpg", format='jpg')
    plt.close()
    
    # Temps total de la simulation
    total_time = time.time() - start_time
    queue.put((sim_num, num_steps, num_steps, total_time, "done"))

# Fonction pour afficher la progression en continu
def display_progress(queue, num_simulations):
    progress = {}
    completed_simulations = 0
    
    while completed_simulations < num_simulations:
        while not queue.empty():
            data = queue.get()
            if len(data) == 4:
                sim_num, step, num_steps, elapsed_time = data
                progress[sim_num] = (step, num_steps, elapsed_time)
            elif len(data) == 5 and data[4] == "done":
                sim_num, step, num_steps, total_time, _ = data
                print(f"\nSimulation {sim_num}/{num_simulations} terminée en {total_time:.2f} secondes.")
                completed_simulations += 1
                if sim_num in progress:
                    del progress[sim_num]
        
        # Effacer l'écran et afficher la progression sans accumulation
        print("\033c", end="")  # Code pour effacer l'écran
        print("=== Progression des simulations ===")
        for sim_num, (step, num_steps, elapsed_time) in progress.items():
            avg_time_per_step = elapsed_time / step
            time_remaining = avg_time_per_step * (num_steps - step)
            print(f"Simulation {sim_num}/{num_simulations} | Étape {step}/{num_steps} "
                  f"| Temps écoulé : {elapsed_time:.2f}s | Temps restant estimé : {time_remaining:.2f}s")
        
        time.sleep(1)  # Rafraîchir chaque seconde

    print("\nToutes les simulations sont terminées.")

# Lancement des processus pour les simulations
if __name__ == "__main__":
    with Manager() as manager:
        queue = manager.Queue()
        
        # Lancer l'affichage de la progression en parallèle
        progress_display = Process(target=display_progress, args=(queue, num_simulations))
        progress_display.start()
        
        # Lancer les simulations en limitant à `max_processes` simultanés
        processes = []
        active_processes = 0
        for sim_num in range(1, num_simulations + 1):
            process = Process(target=run_simulation, args=(sim_num, queue, stop_flag))
            processes.append(process)
            process.start()
            active_processes += 1
            
            # Limiter le nombre de processus simultanés à `max_processes`
            if active_processes >= max_processes:
                # Attendre que l'un des processus en cours se termine
                for p in processes:
                    p.join()
                    active_processes -= 1  # Décrémenter quand un processus se termine
                    processes.remove(p)  # Retirer le processus terminé de la liste
                    break  # Sortir pour lancer le prochain processus dès qu'un slot est libre
        
        # Attendre la fin des processus restants
        for process in processes:
            process.join()
        
        # Attendre la fin de l'affichage de la progression
        progress_display.join()
