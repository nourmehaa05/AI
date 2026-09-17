from practica import joc
from practica.estat import Estat

# Accions: "MOURE", "BOTAR", "ESPERAR"
# Direccions = "O", "S", "E", "N" (antihorari)

# Agent que implementa l'algorisme Alfa-Beta per al joc.

class Viatger(joc.Viatger):
    # Inicialitzador.
    def __init__(self, *args, **kwargs):
        super(Viatger, self).__init__(*args, **kwargs)
        self.__cami_exit = None # Pila camí final
        self.__visitats = set()

    # Mètode principal de l'agent.
    def actua(self, percepcio: dict):
        # Construïm l'estat actual.
        estat_actual = Estat(
            percepcio["AGENTS"][self.nom],
            percepcio["PARETS"],
            percepcio["MIDA"],
            percepcio["DESTI"]
        )

        # Comprovam si és el nostre torn.
        if percepcio["TORN"] != self.nom:
            return ("ESPERAR", None)

        # Reiniciam visitats.
        self.__visitats.clear()

        # Primera planificació.
        res = self.cerca(estat_actual, alpha=-float('inf'), beta=float('inf'))

        # Si hem trobat una solució, la guardam.
        if isinstance(res, tuple) and res[0].accions_previes is not None and len(res[0].accions_previes) > 0:
            solucio, _ = res
            self.__cami_exit = [solucio.accions_previes[i] for i in range(len(solucio.accions_previes)) if (i % 2 == 0)]
            # Si hi ha camí, retornam l'acció. 
            if self.__cami_exit:
                # Hem de mirar si la posició següent està lliure.
                accio, direccio = self.__cami_exit[0]
                nova_pos = self.__nova_posicio(estat_actual.pos_agent, estat_actual.mida_taulell, accio, direccio)

                # Comprovam si la posició està lliure.
                if self.__posicio_lliure(nova_pos, percepcio):
                    return accio, direccio
                else:
                    # Si l'agent vol anar a una posició ocupada, afegim una "paret temporal" i tornem a calcular.
                    estat_actual.parets.add(nova_pos)
                    self.__visitats.clear()
                    res = self.cerca(estat_actual, alpha=-float('inf'), beta=float('inf'))
                    if isinstance(res, tuple) and res[0].accions_previes is not None and len(res[0].accions_previes) > 0:
                        solucio, _ = res
                        self.__cami_exit = [solucio.accions_previes[i] for i in range(len(solucio.accions_previes)) if (i % 2 == 0)]
                        if self.__cami_exit:
                            return self.__cami_exit[0]
        # Si no hi ha camí o no es pot moure, esperar
        return ("ESPERAR", None)

    
    # Mètode de cerca amb poda alfa-beta.
    def cerca(self,estat: Estat, alpha, beta, torn_max=True, iter = 0):

        # Comprovam si l'estat ja ha estat visitat.
        if estat in self.__visitats:
            return estat, 0  # Estat ja visitat, retornam 0.
        self.__visitats.add(estat)

        # Comprovam si l'estat és una meta.
        if estat.es_meta():
            res = 0
            # Si hem guanyat guardam el resultat.
            if estat.guanyador():
                res = (1 if not torn_max else -1)
            return estat, res

        # Si no és meta, continuem la cerca.
        # Generam el arbre de fills.
        puntuacio_fills = []
        fills = estat.genera_fills()

        
        for fill in fills:

            # Crida recursivament la funció 'cerca' (algorisme minimax amb poda alfa-beta).
            punt_fill = self.cerca(fill, alpha, beta, not torn_max, iter + 1)

            # Si és el torn del jugador MAX, actualitza alpha amb el valor màxim sino amb el mínim.
            if torn_max:
                alpha = max(alpha, punt_fill[1])
            else:
                beta = min(beta, punt_fill[1])

            # Desa el resultat (jugada + puntuació) dins la llista de puntuacions dels fills.
            puntuacio_fills.append(punt_fill)


            # si alpha major que beta, poda alfa-beta.
            if alpha >= beta:
                break

        # Obté l’índex del millor fill segons el torn (màxim o mínim).
        idx = self.arg_max(puntuacio_fills, not torn_max)
        # Retorna el millor fill amb la seva puntuació corresponent.
        return puntuacio_fills[idx]
    
    # Mètode estàtic per trobar l'índex de l'estat amb la millor puntuació.
    @staticmethod
    def arg_max(estats, reverse = False):
        """ Troba l'índex de l'estat amb la puntuació més alta, o més baixa si reverse és True.

        Args:
            estats: Llista de tuples (Estat, puntuació).
            reverse: Si és True, busca la puntuació més baixa.
        Returns:
            Enter, índex de l'estat amb la puntuació més alta o més baixa.
        """

        major_idx = 0
        # Guardam la puntuació del primer estat com a referència inicial.
        major_puntuacio = estats[0][1] 


        # Si cercam la puntuació més baixa canviam el signe per invertir la comparació.
        if reverse:
            major_puntuacio *= -1


        # Recorrem tots els estats de la llista.
        for i, estat in enumerate(estats):
            # Extreiem la puntuació de l’estat actual. 
            puntuacio_estat = estat[1] 

            # Si estam cercant el mínim, invertim la puntuació.
            if reverse:
                puntuacio_estat *= -1

            # Si la puntuació actual és millor que la millor trobada fins ara, actualitzam l'índex i la millor puntuació.
            if puntuacio_estat > major_puntuacio:
                major_idx = i
                major_puntuacio = puntuacio_estat

        # Retorna l'índex del millor estat trobat.
        return major_idx


    # Mètode per comprovar si una posició està lliure d'altres agents.
    def __posicio_lliure(self, pos, percepcio):
        # Recorrem tots els noms d'agents 
        for nom_ag in percepcio["AGENTS"]:
            # Obtenim la posició de cada agent
            pos_ag = percepcio["AGENTS"][nom_ag]
            
            # Si hi ha un altre agent en la mateixa posició, la posició no està lliure
            if nom_ag != self.nom and pos_ag == pos:
                return False

        # Si cap agent ocupa aquesta posició, la posició està lliure
        return True
    
    # Mètode per calcular la nova posició donada una acció i direcció.
    def __nova_posicio(self, pos, mida, accio, direccio):
        # Definim els moviments possibles.
        moviments = {"O": (-1, 0), "S": (0, 1), "E": (1, 0), "N": (0, -1)}
        # Calculam la nova posició segons l'acció i direcció.
        dx, dy = moviments[direccio]
        # Determinam la distància a moure.
        dist = 1 if accio == "MOURE" else 2
        # Retornam la nova posició.
        return (pos[0] + dx * dist, pos[1] + dy * dist)