from practica2.agentSARSA import agentSARSA
from practica2.agentGenetic import agentGenetic
from practica2.agentMonteCarlo import agentMonteCarlo
from practica2.agentQLearning import agentQLearning
import gymnasium as gym
import numpy as np
import time

# Mostra la política apresa
def print_policy(Q, n_states, goal, frozen, start, hole):
    arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    # Crear graella buida
    n_rows = 4
    n_cols = 4
    grid = [['' for _ in range(n_cols)] for __ in range(n_rows)]

    for idx in range(n_states):
        r = idx // n_cols
        c = idx % n_cols
        # Afegir G a la meta, H als forats
        if (r, c) == goal:
            grid[r][c] = 'G'
        elif (r, c) in hole:
            grid[r][c] = 'H'
        else:
            # Afegir les fletxes segons la millor acció
            a = np.argmax(Q[idx])
            grid[r][c] = arrows[a]

    for row in grid:
        print(' '.join(f'{cell:>5}' for cell in row))
    print()

# Funció principal
def main():

    # Iniciar comptador de temps total
    start_time_total = time.time()

    #  Paràmetres
    episodis = 20000
    epsilon = 0.5
    n_iteraciones = 10
    n_test_episodes = 10000
    alpha = 0.1
    gamma = 0.99
    
    # Llistes per emmagatzemar estadístiques
    success_rates = []
    avg_steps_list = []
    iteration_times = []
    
    print(f"\n{'='*60}")
    print(f"Entrenant agent - {n_iteraciones} iteracions")
    print(f"{'='*60}")
    
    # INICIAR ITERACIONS
    for i in range(n_iteraciones):
        iteration_start = time.time()
        
        # Crear entorn i agent amb seed diferent
        env = gym.make("FrozenLake-v1", is_slippery=True)

        # AGENTS:
        agent = agentSARSA(alpha=alpha, gamma=gamma, seed=i)
        #agent = agentQLearning(alpha=alpha, gamma=gamma, seed=i)
        #agent = agentMonteCarlo(alpha=alpha, gamma=gamma, seed=i)
        #agent = agentGenetic()

        agent.env = env
        # Entrenar
        agent.train(episodis=episodis, env=env, epsilon=epsilon)
        
        # Avaluar
        env_test = gym.make("FrozenLake-v1", is_slippery=True)
        wins = 0
        total_steps = 0
        
        # FER ELS DISTINTS EPISODIS DE TEST
        for _ in range(n_test_episodes):
            state, _ = env_test.reset()
            done = False
            steps = 0
            
            while not done:
                action = agent.actua(state)
                state, reward, terminated, truncated, _ = env_test.step(action)
                done = terminated or truncated
                steps += 1
                
                if done and reward == 1.0:
                    wins += 1
                    total_steps += steps
        # Calcular estadístiques
        success_rate = (wins / n_test_episodes) * 100
        avg_steps = total_steps / wins if wins > 0 else 0
        iteration_time = time.time() - iteration_start
        
        success_rates.append(success_rate)
        avg_steps_list.append(avg_steps)
        iteration_times.append(iteration_time)
        
        print(f"Iteració {i+1:2d}: {success_rate:6.2f}% encerts, {avg_steps:6.2f} passos promig, {iteration_time:6.2f}s")
    
    # Calcular temps total
    total_time = time.time() - start_time_total
    
    # MOSTRAR RESULTATS
    print(f"\n{'='*60}")
    print("RESULTATS AGREGATS:")
    print(f"{'='*60}")
    print(f"Percentatge d'encerts:")
    print(f"  Mitjana:             {np.mean(success_rates):6.2f}%")
    print(f"  Desviació estàndard: {np.std(success_rates):6.2f}%")
    print(f"  Mínim:               {np.min(success_rates):6.2f}%")
    print(f"  Màxim:               {np.max(success_rates):6.2f}%")
    print(f"\nPassos promig:")
    print(f"  Mitjana:             {np.mean(avg_steps_list):6.2f}")
    print(f"  Desviació estàndard: {np.std(avg_steps_list):6.2f}")
    print(f"\nTemps per iteració:")
    print(f"  Mitjana:             {np.mean(iteration_times):6.2f}s")
    print(f"  Desviació estàndard: {np.std(iteration_times):6.2f}s")
    print(f"\nTemps total d'execució: {total_time:.2f} segons ({total_time/60:.2f} minuts)")
    print(f"{'='*60}\n")
    
    # Visualitzar la darrera iteració
    print("Política apresa:")
    print_policy(
        agent.q, env.observation_space.n,
        goal=(3, 3),
        frozen = {
                (0,1), (0,2), (0,3),
                (1,0), (1,2),
                (2,0), (2,1), (2,2),
                (3,1), (3,2)
            },
        start=(0, 0),
        hole={(1, 1),(1,3),(2,3),(3,0)},
    )

    # Visualitzar l'agent jugant un episodi
    # Està comentat per evitar que s'obri una finestra cada vegada que s'executa l'script
    '''
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
      
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.actua(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    '''

if __name__ == '__main__':
    main()