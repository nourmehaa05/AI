from practica import joc, agent_profunditat,agent_A_estrella, agent_minimax_alpha_beta

# Main per a provar els diferents agents.
def main():
    mida = (10, 10)

    agents = [
        agent_profunditat.Viatger(),
        #agent_A_estrella.Viatger(),
        #agent_minimax_alpha_beta.Viatger(),
        #agent_minimax_alpha_beta.Viatger()

    ]

    lab = joc.Laberint(agents, mida_taulell=mida)
    lab.comencar()


if __name__ == "__main__":
    main()