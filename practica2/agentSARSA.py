import numpy as np
import random

from iaLib import agent

class agentSARSA(agent.Agent):
    def __init__(self, alpha, gamma, seed=0):
        super().__init__(long_memoria=0)

        self.__alpha = alpha     # Taxa d'aprenentatge
        self.__gamma = gamma     # Factor de descompte
        self.q = None       # Taula Q

        np.random.seed(seed)
        random.seed(seed)

    # Política epsilon-greedy
    def epsilon_greedy(self, estat, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()  # Explora
        else:
            return np.argmax(self.q[estat])  # Explota  


    # Entrenament utilitzant SARSA
    def train(self, episodis, env, epsilon):

        n_estats = env.observation_space.n
        n_actions = env.action_space.n
        # Inicialitzam la Q-taula
        self.q = np.zeros((n_estats, n_actions))
        for episodi in range(episodis):
            estat, _ = env.reset(seed=42)
            done = False
            while not done:
                accio= self.epsilon_greedy(estat, epsilon)
                estat_seguent, recompensa, terminated, truncated, _ = env.step(accio)
                done = terminated or truncated
                #print(estat)
                # Cercam l'accio següent
                accio_seguent = self.epsilon_greedy(estat_seguent, epsilon)
                # Actualitzam la Q-taula, Q(s,a) = Q(s,a) + α * ( r + γ * Q(s',a') - Q(s,a) )
                self.q[estat][accio] += self.__alpha * (
                recompensa + self.__gamma * self.q[estat_seguent][accio_seguent] - self.q[estat][accio]
                )
                estat = estat_seguent

    def actua(self, estat):
        return np.argmax(self.q[estat]) # Selecciona l'acció amb el valor Q més alt