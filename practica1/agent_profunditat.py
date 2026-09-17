from practica import joc
from practica.estat import Estat

# Accions: "MOURE", "BOTAR", "ESPERAR"
# Direccions = "O", "S", "E", "N" (antihorari)

# Agent que implementa l'algorisme profunditat (DFS) per al joc.

class Viatger(joc.Viatger):
    def __init__(self, *args, **kwargs):
        super(Viatger, self).__init__(*args, **kwargs)
        self.__cami = None # Pila camí final.
        self.__visitats = None # Conjunt d'estats visitats.
        self.__frontera = None # Pila estats per explorar.

    def actua(self, percepcio: dict):
        # Inicialment, no tenim el camí cercat llavors ho generam.
        if self.__cami is None: 
            # Guardam paràmetres del joc.
            estat_inicial = Estat(
                percepcio["AGENTS"][self.nom],
                percepcio["PARETS"],
                percepcio["MIDA"],
                percepcio["DESTI"]
            )

            # Si inicialment l'agent apareix a la posició meta,
            # no cal cercar camí.
            if estat_inicial.es_meta():
                return "ESPERAR", None
            
            self.generar_cami(estat_inicial) # Generam camí

        # Si ja tenim camí, el retornam.
        if self.__cami:
            accio, direccio = self.__cami.pop(0)
            return accio,direccio

        # Si no hi ha camí, esperam.
        return "ESPERAR", None



    # Mètode que genera el camí des de l'estat inicial fins a l'estat meta.
    def generar_cami(self, estat_inicial):
        self.__cami = [] # Pila camí final.
        self.__visitats = set() # Conjunt d'estats visitats.
        self.__frontera = [] # Pila d'estats per explorar.
        
        # Afegim l'estat inicial a la frontera
        self.__frontera.append(estat_inicial)

        # M'entre encara hi hagi estats per visitar...
        while self.__frontera:
            # Treim l'últim estat de la frontera (LIFO).
            estat_actual = self.__frontera.pop(-1)
            pos_actual = estat_actual.pos_agent
            # Evitam visitar 2 pics la mateixa posició.
            if pos_actual in self.__visitats:
                continue

            # Afegim la posició actual als visitats.
            self.__visitats.add(pos_actual)

            # Si ja som a la meta, guardam el camí.
            if estat_actual.es_meta():
                self.__cami = estat_actual.accions_previes
                return True
            
            # Generam els fills de l'estat actual i els afegim a la frontera.
            for fill in estat_actual.genera_fills():
                self.__frontera.append(fill)
                
        return False
    


    