import numpy as np
import random
from iaLib import agent

class agentMonteCarlo(agent.Agent):

    def __init__(self, alpha, gamma, seed=0):
        super().__init__(long_memoria=0)

        self.__alpha = alpha      # Taxa d'aprenentatge
        self.__gamma = gamma      # Factor de descompte
        self.q = None             # Taula Q

        np.random.seed(seed)
        random.seed(seed)


    # Política epsilon-greedy
    def epsilon_greedy(self, estat, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()  # Explora
        else:
            return np.argmax(self.q[estat]) # Explota

    # Genera un episodi seguint la política epsilon-greedy
    def generar_episodio(self, env, epsilon):
        episodio = []
        estado, _ = env.reset()
        done = False

        while not done:
            accion = self.epsilon_greedy(estado, epsilon)
            siguiente_estado, recompensa, terminated, truncated, _ = env.step(accion)
            done = terminated or truncated
            episodio.append((estado, accion, recompensa))
            estado = siguiente_estado

            if done:
                break

        return episodio

    # Entrenament utilitzant Monte Carlo
    def train(self, episodis, env, epsilon):

        n_states = env.observation_space.n
        n_actions = env.action_space.n
        # Inicialitzem Q(s,a)
        self.q = np.zeros((n_states, n_actions))
        for episodi in range(episodis):
            ep = self.generar_episodio(env, epsilon)
            # Llista inversa per acumular retorns cap enrere
            G = 0
            visited_state_actions = set()
            # Recorrem l'episodi des del final
            for t in reversed(range(len(ep))):
                state, action, reward = ep[t]
                G = self.__gamma * G + reward
                # Actualització First-Visit MC, només la primera vegada que es visita (estat, acció)
                if (state, action) not in visited_state_actions:
                    visited_state_actions.add((state, action))
                    self.q[state, action] += self.__alpha * (G - self.q[state, action])


    def actua(self, estat):
        # Selecciona l'acció amb el valor Q més alt
        return np.argmax(self.q[estat])