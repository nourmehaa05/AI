from practica import joc
from practica.estat import Estat
from queue import PriorityQueue

# Accions: "MOURE", "BOTAR", "ESPERAR"
# Direccions = "O", "S", "E", "N" (antihorari)

# Agent que implementa l'algorisme de cerca A* per al joc.

class Viatger(joc.Viatger):
    def __init__(self, *args, **kwargs):
        # Inicialitzador.
        super(Viatger, self).__init__(*args, **kwargs)
        self.__cami = None
        self.__visitats = None
        self.__frontera = None

    # Mètode principal de l'agent.
    def actua(self, percepcio: dict):
        # Inicialment, no tenim el camí cercat llavors ho generam.
        if self.__cami is None:
            # Construïm l'estat inicial.
            estat_inicial = Estat(
                percepcio["AGENTS"][self.nom],
                percepcio["PARETS"],
                percepcio["MIDA"],
                percepcio["DESTI"]
            )

            # Cercam el camí utilitzant A*.
            self.cerca(estat_inicial)

        # Si tenim un camí, retornam la següent acció.
        if self.__cami:
            accio, direccio = self.__cami.pop(0)
            return accio, direccio

        # Si no hi ha camí, esperam.
        return "ESPERAR", None

    # Mètode que implementa l'algorisme de cerca A*.
    def cerca(self, estat_inicial):
        # Inicialitzam la frontera, el conjunt de visitats i el camí.
        self.__frontera = PriorityQueue()
        self.__visitats = set()
        self.__cami = []

        # Insertam l'estat inicial.
        self.__frontera.put((estat_inicial.calc_heuristica(), estat_inicial))

        # M'entre hi hagi estats per explorar...
        while not self.__frontera.empty():
            # Extreiem l'estat amb frontera i agafam el primer 
            # element (menor heurística) de la cua de prioritat.
            _, estat_actual = self.__frontera.get()

            # Si ja hem visitat aquest estat, passam al següent.
            if estat_actual in self.__visitats:
                continue
            # Afegim l'estat actual als visitats.
            self.__visitats.add(estat_actual)
            # Comprovam si és un estat meta.
            if estat_actual.es_meta():
                # Guardam el camí i retornam True.
                self.__cami = estat_actual.accions_previes
                return True
            # Generam els estats fills i els afegim a la frontera si no
            # han estat visitats.
            estats_fills = estat_actual.genera_fills()
            for fill in estats_fills:
                if fill not in self.__visitats:
                    self.__frontera.put((fill.calc_heuristica(), fill))

        return False
