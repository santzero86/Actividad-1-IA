# Inteligencia Artificial - Grupo 80
# Actividad 2: Laberinto basado en grafos / Niño rata / Algoritmo A* h1, h2, h3
# Integrantes:
# - Santiago David Guerrero Jaramillo - 2419030
# - Hugart Andres Estrada Velasco - 2610126

import heapq
import math


# Matriz 9x9 (0=libre, 1=muro, 2=penalización, 3=premio, 9=meta)
laberinto_matriz = [
    [0, 0, 0, 0, 0, 0, 3, 1, 0],
    [3, 1, 1, 1, 2, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 2, 1, 3, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 2, 0, 0, 0, 1, 0, 0, 3],
    [0, 1, 1, 1, 0, 0, 3, 1, 0],
    [2, 0, 0, 0, 0, 1, 0, 2, 9]
]

COSTO_PASO_MINIMO = 1.0

def obtener_costo_celda(valor):
    if valor == 0: 
        return 1.0       # Normal
    if valor == 2: 
        return 3.0       # Penalización: -3 coins
    if valor == 3: 
        return 0.25      # Premio: +1/4 coins
    if valor == 9: 
        return 0.0       # Meta
    return float('inf')  # Muro

def construir_grafo_estados(matriz):
    """
    Convierte la matriz 9x9 en una lista de adyacencia, que simula un grafo de estados.
    Clave: Nodo Origen (x, y) -> Valor: Diccionario {Nodo Destino: Peso/Costo}
    """
    filas = len(matriz)
    cols = len(matriz[0])
    grafo = {}

    # Transiciones de estado (Arriba, Abajo, Izquierda, Derecha)
    movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(filas):
        for c in range(cols):
            if matriz[r][c] == 1:
                continue  # Muro, se ignora

            nodo_actual = (r, c)
            grafo[nodo_actual] = {}

            for dr, dc in movimientos: # Recorrer movimientos en x,y
                nr, nc = r + dr, c + dc # Calcular posiciones vencinas
                if 0 <= nr < filas and 0 <= nc < cols: # Comprueba que filas y columnas son válidas
                    if matriz[nr][nc] != 1:  # Si la posición destino no es muro
                        costo = obtener_costo_celda(matriz[nr][nc]) # Validar costo de celda destino
                        grafo[nodo_actual][(nr, nc)] = costo # Agregar transición al grafo

    return grafo

# HEURÍSTICA 1: Chebyshev (h1)
# Calcula la distancia tomando el mayor valor entre las diferencias de filas y columnas.
# Es admisible y monótona, pero resulta menos efectiva porque estima costos bajos y puede hacer que A* explore más nodos.

def heuristica_1_chebyshev(nodo_actual, meta):
    dr = abs(nodo_actual[0] - meta[0])
    dc = abs(nodo_actual[1] - meta[1])
    distancia = max(dr, dc)
    return distancia * COSTO_PASO_MINIMO

# HEURÍSTICA 2: Euclidiana (h2)
# Calcula la distancia en línea recta entre el nodo actual y la meta mediante el teorema de Pitágoras.
# Es admisible y monótona, y tiene una efectividad intermedia al ofrecer una estimación más precisa que Chebyshev.

def heuristica_2_euclidiana(nodo_actual, meta):
    dr = nodo_actual[0] - meta[0]
    dc = nodo_actual[1] - meta[1]
    distancia = math.sqrt(dr ** 2 + dc ** 2)
    return distancia * COSTO_PASO_MINIMO

# HEURÍSTICA 3: Manhattan (h3)
# Suma las diferencias entre filas y columnas, considerando únicamente movimientos horizontales y verticales.
# Es admisible y monótona, y suele guiar mejor a A* en cuadrículas de cuatro direcciones al dar estimaciones más ajustadas.

def heuristica_3_manhattan(nodo_actual, meta):
    dr = abs(nodo_actual[0] - meta[0])
    dc = abs(nodo_actual[1] - meta[1])
    distancia = dr + dc
    return distancia * COSTO_PASO_MINIMO

def a_estrella_camino(grafo, inicio, objetivo, funcion_heuristica):
    """
    Implementación de A* basada en las diapositivas de clase (f = g + h).
    Utiliza una cola de prioridad basada en el valor f(n).
    Retorna el camino óptimo, el costo total y el número de nodos expandidos.
    """
    # cola_prioridad guarda tuplas: (f_score, g_cost, nodo)
    cola_prioridad = []
    g_costos = {inicio: 0.0}
    f_inicial = 0.0 + funcion_heuristica(inicio, objetivo)
    heapq.heappush(cola_prioridad, (f_inicial, 0.0, inicio))
    
    predecesor = {inicio: None}
    cerrados = set()
    nodos_expandidos = 0

    while cola_prioridad:
        f_actual, g_actual, nodo = heapq.heappop(cola_prioridad)

        if nodo in cerrados:
            continue
        cerrados.add(nodo)
        nodos_expandidos += 1

        # Criterio de parada: cuando se extrae la meta
        if nodo == objetivo:
            camino = []
            curr = nodo
            while curr is not None:
                camino.append(curr)
                curr = predecesor[curr]
            return list(reversed(camino)), g_actual, nodos_expandidos

        for vecino, costo in grafo[nodo].items():
            if vecino in cerrados:
                continue

            nuevo_g = g_actual + costo
            if vecino not in g_costos or nuevo_g < g_costos[vecino]:
                g_costos[vecino] = nuevo_g
                predecesor[vecino] = nodo
                h_vecino = funcion_heuristica(vecino, objetivo)
                nuevo_f = nuevo_g + h_vecino
                heapq.heappush(cola_prioridad, (nuevo_f, nuevo_g, vecino))

    return None, float('inf'), nodos_expandidos

def imprimir_tablero(matriz, pos_actual, coins):
    print(f"\n🪙 Monedas actuales: {coins}")
    for i in range(len(matriz)):
        fila_str = ""
        for j in range(len(matriz[0])):
            if (i, j) == pos_actual:
                fila_str += " 🐭 "
            elif matriz[i][j] == 1:
                fila_str += " 🧱 "
            elif matriz[i][j] == 2:
                fila_str += " ⚠️  "
            elif matriz[i][j] == 3:
                fila_str += " 🎁 "
            elif matriz[i][j] == 9:
                fila_str += " 🏁 "
            else:
                fila_str += " ⬜ "
        print(fila_str)
    print()
    
def bfs_camino(grafo, inicio, objetivo):
    cola = [inicio]
    visitados = {inicio}
    predecesor = {inicio: None}
    while cola:
        nodo = cola.pop(0)
        if nodo == objetivo:
            # reconstrucción del camino
            camino = []
            while nodo is not None:
                camino.append(nodo)
                nodo = predecesor[nodo]
            return list(reversed(camino))
        
        for vecino in grafo[nodo]:
            if vecino not in visitados:
                visitados.add(vecino)
                predecesor[vecino] = nodo
                cola.append(vecino) 
    return None

def dfs_camino(grafo, inicio, objetivo):
    pila = [inicio]
    visitados = {inicio}
    predecesor = {inicio: None}
    while pila:
        nodo = pila.pop(-1)
        if nodo == objetivo:
            # reconstrucción del camino
            camino = []
            while nodo is not None:
                camino.append(nodo)
                nodo = predecesor[nodo]
            return list(reversed(camino))
        
        for vecino in grafo[nodo]:
            if vecino not in visitados:
                visitados.add(vecino)
                predecesor[vecino] = nodo
                pila.append(vecino) 
    return None

def bcu_camino(grafo, inicio, objetivo):
    cola_prioridad = [(0, inicio)]
    costos = {inicio: 0}
    predecesor = {inicio: None}

    while cola_prioridad:
        costo_actual, nodo = heapq.heappop(cola_prioridad)

        # Ignorar caminos que ya fueron superados por uno más barato
        if costo_actual > costos[nodo]:
            continue

        # Si llegamos a la meta, reconstruimos el camino
        if nodo == objetivo:
            camino = []
            while nodo is not None:
                camino.append(nodo)
                nodo = predecesor[nodo]

            return list(reversed(camino)), costo_actual

        for vecino, costo in grafo[nodo].items():
            nuevo_costo = costo_actual + costo

            if vecino not in costos or nuevo_costo < costos[vecino]:
                costos[vecino] = nuevo_costo
                predecesor[vecino] = nodo
                heapq.heappush(
                    cola_prioridad,
                    (nuevo_costo, vecino)
                )

    return None, float('inf')

def calcular_costo_camino(grafo, camino):
    costo_total = 0

    for i in range(len(camino) - 1):
        origen = camino[i]
        destino = camino[i + 1]
        costo_total += grafo[origen][destino]

    return costo_total

def imprimir_grafo_coordenadas(grafo):
    for nodo, vecinos in sorted(grafo.items()):
        print(f"{nodo} = {list(vecinos.keys())}")


def iniciar_juego():
    # Construcción del modelo de estados
    grafo = construir_grafo_estados(laberinto_matriz)
    
    imprimir_grafo_coordenadas(grafo)  # Mostrar el grafo de estados al inicio del juego
    
    # Configuración inicial
    pos_actual = (0, 0)
    meta = (8, 8)
    coins = 15.0  # Monedas iniciales ajustadas para permitir ganar
    
    camino_bfs = bfs_camino(grafo, pos_actual, meta)
    camino_dfs = dfs_camino(grafo, pos_actual, meta)
    
    costo_bfs = calcular_costo_camino(grafo, camino_bfs)
    costo_dfs = calcular_costo_camino(grafo, camino_dfs)

    print(f"----------------------------------------------------------------------------------------------------------------------------------")
    print(f"Camino BFS: {camino_bfs}")
    print(f"Movimientos BFS: {len(camino_bfs) - 1}")
    print(f"Costo BFS: {costo_bfs}")
    print(f"Coins finales BFS: {coins - costo_bfs}")
    
    print(f"----------------------------------------------------------------------------------------------------------------------------------")
    print(f"Camino DFS: {camino_dfs}")
    print(f"Movimientos DFS: {len(camino_dfs) - 1}")
    print(f"Costo DFS: {costo_dfs}")
    print(f"Coins finales DFS: {coins - costo_dfs}")    

    camino_bcu, costo_bcu = bcu_camino(grafo, pos_actual, meta)
    print(f"----------------------------------------------------------------------------------------------------------------------------------")
    print(f"Camino BCU: {camino_bcu}")
    print(f"Costo total del camino BCU: {costo_bcu}")
    print(f"Coins iniciales: {coins}")
    print(f"Coins finales: {coins - costo_bcu}")
    print(f"Movimientos del camino BCU: {len(camino_bcu) - 1}")
    
    camino_a1, costo_a1, exp_a1 = a_estrella_camino(grafo, pos_actual, meta, heuristica_1_chebyshev)
    camino_a2, costo_a2, exp_a2 = a_estrella_camino(grafo, pos_actual, meta, heuristica_2_euclidiana)
    camino_a3, costo_a3, exp_a3 = a_estrella_camino(grafo, pos_actual, meta, heuristica_3_manhattan)
    
    print(f"----------------------------------------------------------------------------------------------------------------------------------")
    print(f"A* con h1 (Chebyshev - Menos efectiva):")
    print(f"Costo: {costo_a1} ") 
    print(f"Movimientos: {len(camino_a1) - 1}") 
    print(f"Nodos expandidos: {exp_a1}")
    print(f"Coins finales: {coins - costo_a1}")
    
    print(f"----------------------------------------------------------------------------------------------------------------------------------")
    print(f"A* con h2 (Euclidiana):")
    print(f"Costo: {costo_a2}")
    print(f"Movimientos: {len(camino_a2) - 1}")
    print(f"Nodos expandidos: {exp_a2}")
    print(f"Coins finales: {coins - costo_a2}")

    print(f"----------------------------------------------------------------------------------------------------------------------------------")
    print(f"A* con h3 (Manhattan - MÁS DOMINANTE):")
    print(f"Costo: {costo_a3}")
    print(f"Movimientos: {len(camino_a3) - 1}")
    print(f"Nodos expandidos: {exp_a3}")
    print(f"Coins finales: {coins - costo_a3}")

    print(f"----------------------------------------------------------------------------------------------------------------------------------")
    print("🐭 ¡BIENVENIDO AL LABERINTO BASADO EN GRAFOS! 🎮")
    print("Controles: w=arriba, s=abajo, a=izquierda, d=derecha")
    print("Leyenda: ⬜ Libre (1.0) | ⚠️ Penalización (3.0) | 🎁 Premio (0.25) | 🧱 Muro")

    while True:
        imprimir_tablero(laberinto_matriz, pos_actual, coins)

        # Validación de derrota (Requisito 6)
        if coins <= 0:
            print("❌ Te quedaste sin coins. ¡Has perdido el juego!")
            break

        # Validación de victoria (Requisito 7)
        if pos_actual == meta:
            print("🎉 ¡Has llegado a la meta!")
            print(f"✅ Tu puntuación final es de {coins} coins restantes.")
            break

        mov = input("Ingresa movimiento (w/a/s/d): ").strip().lower()
        direcciones = {'w': (-1, 0), 's': (1, 0), 'a': (0, -1), 'd': (0, 1)}

        if mov not in direcciones:
            print("Movimiento inválido ❌")
            continue

        dr, dc = direcciones[mov]
        pos_siguiente = (pos_actual[0] + dr, pos_actual[1] + dc)

        # Consulta de transición en el Grafo
        if pos_siguiente in grafo.get(pos_actual, {}):
            costo = grafo[pos_actual][pos_siguiente]
            coins -= costo
            pos_actual = pos_siguiente
        else:
            print("Choque o movimiento fuera del mapa (transición no válida en el mapa) 🚧")

if __name__ == "__main__":
    iniciar_juego()