import copy

# Classe que representa un estat del joc.
class Estat:
    def __init__(self, pos_agent, parets, mida_taulell, desti, accions_previes=None):
        self.pos_agent = pos_agent        # (x, y)
        self.desti = desti                # (x, y)
        self.parets = parets              # set((x, y), ...)
        self.mida_taulell = mida_taulell  # (columnes, files)
        self.accions_previes = accions_previes or []
        self.pes = 0                      # Cost acumulat

    # Comparació d'estats basada en la posició de l'agent.
    def __eq__(self, other):
        return isinstance(other, Estat) and self.pos_agent == other.pos_agent

    # Hash basat en la posició de l'agent.
    def __hash__(self):
        return hash(self.pos_agent)


    # Mètode que permet comparar estats segons la seva heurística.
    def __lt__(self, other):
        return self.calc_heuristica() < other.calc_heuristica()

    # Mètode que comprova si l'agent pot moure's.
    def __pot_moure(self):
        x, y = self.pos_agent
        columnes, files = self.mida_taulell
        direccions = [(-1,0),(0,1),(1,0),(0,-1)]  # O, S, E, N
        for i,j in direccions:
            nova_posicio  = (x+i , y+j)
            if not (0 <= nova_posicio[0] < columnes and 0 <= nova_posicio[1] < files):
                continue
            if nova_posicio not in self.parets:
                return True
        return False

    # Mètode que comprova si l'agent pot botar.
    def __pot_botar(self):
        x, y = self.pos_agent
        columnes, files = self.mida_taulell
        direccions = [(-2,0),(0,2),(2,0),(0,-2)]
        for i,j in direccions:
            nova_posicio  = (x+i , y+j)
            if not (0 <= nova_posicio[0] < columnes and 0 <= nova_posicio[1] < files):
                continue
            if nova_posicio not in self.parets:
                return True
        return False

    # Mètode que comprova si l'agent ha arribat a la posició meta.
    def guanyador(self):
        return self.pos_agent == self.desti

    # Mètode que comprova si l'estat és meta.
    def es_meta(self) -> bool:
        guanyador = self.guanyador()
        moviment = self.__pot_moure()
        if not moviment:
            moviment = self.__pot_botar()
        # O bé hem guanyat o bé no podem moure'ns més.
        return guanyador or not moviment

    # Mètode que genera un nou estat a partir d'una acció i direcció.
    def __transicio(self, accio, direccio):
        # Diccionari de moviments.
        moviments = {
            "O": (-1, 0),
            "S": (0, 1),
            "E": (1, 0),
            "N": (0, -1)
        }
        i, j = moviments[direccio]
        mov = 1 if accio == "MOURE" else 2
        nova_posicio = (self.pos_agent[0] + i * mov, self.pos_agent[1] + j * mov)

        x, y = nova_posicio
        cols, files = self.mida_taulell

        if x < 0 or x >= cols or y < 0 or y >= files:
            return None
        if nova_posicio in self.parets:
            return None

        nou_estat = Estat(
            pos_agent=nova_posicio,
            parets=set(self.parets),
            mida_taulell=self.mida_taulell,
            desti=self.desti,
            accions_previes=self.accions_previes + [(accio, direccio)]
        )
        # El cost d'una acció és 1.
        nou_estat.pes = self.pes + 1
        return nou_estat

    # Mètode que genera tots els fills possibles de l'estat actual.
    def genera_fills(self):
        fills = []
        for accio in ("MOURE", "BOTAR"):
            for direccio in ("O", "S", "E", "N"):
                nou_estat = self.__transicio(accio, direccio)
                if nou_estat is not None:
                    fills.append(nou_estat)
        return fills

    # Mètode que calcula la distància de Manhattan fins a la destinació.
    def __pos_lliure(self):
        x1, y1 = self.pos_agent
        x2, y2 = self.desti
        return abs(x1 - x2) + abs(y1 - y2)

    # Mètode que calcula l'heurística de l'estat.
    def calc_heuristica(self):
        # f(n) = g(n) + h(n)
        return self.pes + self.__pos_lliure()
