import numpy as np
import random
from iaLib import agent


class agentGenetic(agent.Agent):
    def __init__(self, population_size=150, generations=100, mutation_rate=0.1, seed=42):
        super().__init__(long_memoria=0)
        # Tamany de la població
        self.population_size = population_size
        # Quantes generacions evolucionarem
        self.generations = generations
        # Probabilitat que un gen (accio) muta
        self.mutation_rate = mutation_rate
        # La millor política trobada
        self.policy = None
        # Fixar semilla per a la reproducibilitat
        np.random.seed(seed)
        random.seed(seed)

    # Avaluar la política
    def evaluate_policy(self, policy, env):

        # Executar 15 episodis i retorna recompensa mitjana
        total_reward = 0
        n_episodes = 15 
        
        for _ in range(n_episodes):
            # Resetear entorn
            state, _ = env.reset()
            done = False
            steps = 0
            
            # Jugar un episodi
            while not done and steps < 100:
                # Seleccionar acció segons la política
                action = policy[state]
                # Executar acció
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
        
        # Retornar la mitjana
        return total_reward / n_episodes

    # Mètode que combina dos pares per crear un fill
    def crossover(self, parent1, parent2):

        # Triar punt de tall aleatori i mesclar
        cut_point = random.randint(1, len(parent1) - 1)
        # Crear fill: primera part del pare1 + segona part del pare2
        child = np.concatenate([parent1[:cut_point], parent2[cut_point:]])
        return child

    # Mètode de mutació
    def mutate(self, policy, n_actions):

        # muta aleatòriament alguns gens (ACCIONS) de la política
        for i in range(len(policy)):
            # Amb probabilitat mutation_rate, canviar l'acció
            if random.random() < self.mutation_rate:
                policy[i] = random.randint(0, n_actions - 1)
        return policy

    def train(self, episodis, env, epsilon):

        self.env = env
        n_states = env.observation_space.n  # Nombre d'estats (16 en FrozenLake)
        n_actions = env.action_space.n      # Nombre d'accions (4: amunt, dreta, avall, esquerra)
        
        # Pas 1: Crear població inicial aleatòria
        # Cada individu és una llista de 16 accions (una per estat)
        population = []
        for _ in range(self.population_size):
            individual = np.random.randint(0, n_actions, size=n_states)
            population.append(individual)
        
        best_fitness = 0
        best_policy = None
        
        # Pas 2: Evolucionar per diverses generacions
        for generation in range(self.generations):
            
            # Avaluar fitness de cada individu
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_policy(individual, env)
                fitness_scores.append(fitness)
            
            # Trobar el millor d'aquesta generació
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_policy = population[best_idx].copy()
            
            # Imprimir progrés cada 20 generacions
            if (generation + 1) % 20 == 0:
                print(f"Generació {generation + 1}/{self.generations}, Millor fitness: {best_fitness:.2f}")

            # Pas 3: Crear nova generació
            new_population = []
            
            # Mantenir el millor (elitisme)
            new_population.append(best_policy.copy())
            
            # Crear la resta d'individus
            while len(new_population) < self.population_size:
                # Seleccionar dos pares: els que tinguin millor fitness tenen més oportunitat
                parent1 = self.select_parent(population, fitness_scores)
                parent2 = self.select_parent(population, fitness_scores)
                
                # Creuar pares per crear fill
                child = self.crossover(parent1, parent2)
                
                # Mutar el fill
                child = self.mutate(child, n_actions)
                
                new_population.append(child)
            
            # La nova generació reemplaça a l'antiga
            population = new_population
        
        # Guardar la millor política trobada
        self.policy = best_policy
        print(f"Entrenament completat. Millor fitness: {best_fitness:.2f}")
        
        # Crear taula Q (per compatibilitat amb altres agents)
        self.q = np.zeros((n_states, n_actions))
        for s in range(n_states):
            self.q[s, self.policy[s]] = 1.0
        
        return self.q

    def select_parent(self, population, fitness_scores):
        """
        Selecciona un padre usando selección por torneo.
        Elige 3 individuos al azar y devuelve el mejor de los 3.
        """
        # Escollir 3 índexs aleatoris
        candidates = random.sample(range(len(population)), 3)
        # Veure quin té millor fitness
        best_candidate = candidates[0]
        for candidate in candidates:
            if fitness_scores[candidate] > fitness_scores[best_candidate]:
                best_candidate = candidate
        # Devolver aquest individu
        return population[best_candidate]

    def actua(self, estat):
        if self.policy is None:
            raise RuntimeError("Agent not trained yet.")
        return int(self.policy[estat])
