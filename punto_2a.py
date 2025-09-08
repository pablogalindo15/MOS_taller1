from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np

model = ConcreteModel()

T_list = [1, 2, 3, 4, 5]       
E_list = [1, 2, 3]              

G = {1: 50, 2: 60, 3: 40, 4: 70, 5: 30}   # Ganancias G_t
h = {1: 4,  2: 5,  3: 3,  4: 6,  5: 2}    # Horas requeridas h_t
Te = {1: 8,  2: 10, 3: 6}                 # Horas disponibles por trabajador T_e

# Conjuntos
model.T = Set(initialize=list(T_list))
model.E = Set(initialize=list(E_list))

# Variable
model.x = Var(model.E, model.T, domain=Binary)

# Parámetros
model.G  = Param(model.T, initialize=G, within=Reals)               # ganancia G_t
model.h  = Param(model.T, initialize=h, within=NonNegativeReals)    # horas h_t
model.Te = Param(model.E, initialize=Te, within=NonNegativeReals)   # horas disponibles T_e

# Función Objetivo
model.obj = Objective(
        expr=sum(model.G[t] * model.x[e, t] for e in model.E for t in model.T),
        sense=maximize )

# Restricción 1: Disponibilidad horaria por trabajador
def Availability_rule(m, e):
    return sum(m.h[t] * m.x[e, t] for t in m.T) <= m.Te[e]
model.Availability = Constraint(model.E, rule=Availability_rule)

# Restricción 2: A lo sumo un trabajador por tarea
def OneWorkerPerTask_rule(m, t):
    return sum(m.x[e, t] for e in m.E) <= 1
model.OneWorkerPerTask = Constraint(model.T, rule=OneWorkerPerTask_rule)


if __name__ == "__main__":

    solver = SolverFactory('glpk')
    result = solver.solve(model, tee=False)

    # Estado y objetivo
    print("Solver status:", result.solver.status)
    print("Solver termination condition:", result.solver.termination_condition)
    print("Objetivo (ganancia total):", value(model.obj))

    # Imprimir asignaciones y calcular métricas por trabajador
    print("\nAsignaciones (x[e,t] = 1):")
    tasks = list(model.T)
    workers = list(model.E)

    horas_usadas = {e: 0 for e in workers}
    ganancia_por_dev = {e: 0 for e in workers}
    asignaciones = {e: [] for e in workers}

    for e in model.E:
        asignados = [t for t in model.T if value(model.x[e, t]) > 0.5]
        if asignados:
            horas = sum(value(model.h[t]) for t in asignados)
            ganancia = sum(value(model.G[t]) for t in asignados)
            horas_usadas[e] = horas
            ganancia_por_dev[e] = ganancia
            asignaciones[e] = asignados
            print(f"  Trabajador {e}: trabajos {asignados} | horas usadas = {horas} | ganancia aportada = {ganancia}")
        else:
            print(f"  Trabajador {e}: ningún trabajo asignado")

    # -------------------------
    # 3. VISUALIZACIÓN
    # -------------------------
    # 3.1 Heatmap: matriz T x E (1 si asignada, 0 si no)
    data = np.zeros((len(tasks), len(workers)))
    for i, t in enumerate(tasks):
        for j, e in enumerate(workers):
            data[i, j] = 1.0 if value(model.x[e, t]) > 0.5 else 0.0

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    cax = ax1.matshow(data, aspect='auto')
    plt.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks(range(len(workers)))
    ax1.set_xticklabels([f"Dev {e}" for e in workers])
    ax1.set_yticks(range(len(tasks)))
    ax1.set_yticklabels([f"T{t}" for t in tasks])
    ax1.set_xlabel("Desarrolladores")
    ax1.set_ylabel("Tareas")
    ax1.set_title("Asignación (matriz Tareas x Desarrolladores)")
    for (i, j), val in np.ndenumerate(data):
        if val > 0.5:
            ax1.text(j, i, "✓", ha='center', va='center', color='white', fontsize=12)
    fig1.tight_layout()
    fig1.savefig("asignacion_heatmap.png", dpi=200)

    # 3.2 Barras: Horas usadas por trabajador vs capacidad
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    idx = np.arange(len(workers))
    horas_vals = [horas_usadas[e] for e in workers]
    cap_vals = [value(model.Te[e]) for e in workers]
    bar_h = ax2.bar(idx - 0.15, horas_vals, width=0.3, label="Horas usadas")
    bar_c = ax2.bar(idx + 0.15, cap_vals, width=0.3, label="Capacidad (Te)")
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"Dev {e}" for e in workers])
    ax2.set_ylabel("Horas")
    ax2.set_title("Horas usadas por desarrollador vs Capacidad")
    ax2.legend()
    # Anotar valores encima de las barras (separado para cada grupo)
    for rect in bar_h:
        height = rect.get_height()
        ax2.annotate(f'{height:.0f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    for rect in bar_c:
        height = rect.get_height()
        ax2.annotate(f'{height:.0f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    fig2.tight_layout()
    fig2.savefig("horas_por_dev.png", dpi=200)

    # 3.3 Barras: Ganancia aportada por trabajador
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    gan_vals = [ganancia_por_dev[e] for e in workers]
    bars = ax3.bar(range(len(workers)), gan_vals)
    ax3.set_xticks(range(len(workers)))
    ax3.set_xticklabels([f"Dev {e}" for e in workers])
    ax3.set_ylabel("Ganancia")
    ax3.set_title("Ganancia aportada por desarrollador")
    for rect in bars:
        height = rect.get_height()
        ax3.annotate(f'{height:.0f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    fig3.tight_layout()
    fig3.savefig("ganancia_por_dev.png", dpi=200)

    # Mostrar todas las figuras
    plt.show()
