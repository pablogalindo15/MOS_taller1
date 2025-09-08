# ==========================================================
# PROBLEMA 1: Planificación de Sprint Ágil en Pyomo
# ==========================================================
from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------
# 1. DATOS DEL PROBLEMA
# ----------------------------------------------------------
# Puntos por tarea
p = {
    1:5,  2:8,  3:13, 4:1,  5:21, 
    6:2,  7:8,  8:5,  9:8, 10:13, 11:21
}

# Prioridad por tarea (1=menor, 7=mayor)
w = {
    1:7,  2:5,  3:6,  4:3,  5:1, 
    6:4,  7:6,  8:6,  9:2, 10:7, 11:6
}

# Capacidad total del equipo
C = 52
# Capacidad individual de cada desarrollador (Parte B)
K = {1:15, 2:15, 3:15, 4:15}

# ==========================================================
# PARTE A: MODELO BÁSICO (Capacidad global)
# ==========================================================
modelA = ConcreteModel()

# Conjuntos
modelA.T = RangeSet(11)

# Parámetros
modelA.p = Param(modelA.T, initialize=p)
modelA.w = Param(modelA.T, initialize=w)

# Variables: x[t] = 1 si se selecciona la tarea t
modelA.x = Var(modelA.T, within=Binary)

# Función objetivo
def objA_rule(model):
    return sum(model.w[t] * model.x[t] for t in model.T)
modelA.obj = Objective(rule=objA_rule, sense=maximize)

# Restricción: capacidad total ≤ 52
def capA_rule(model):
    return sum(model.p[t] * model.x[t] for t in model.T) <= C
modelA.cap = Constraint(rule=capA_rule)

# ==========================================================
# PARTE B: MODELO CON ASIGNACIÓN INDIVIDUAL
# ==========================================================
modelB = ConcreteModel()

# Conjuntos
modelB.T = RangeSet(11)
modelB.D = RangeSet(4)

# Parámetros
modelB.p = Param(modelB.T, initialize=p)
modelB.w = Param(modelB.T, initialize=w)
modelB.K = Param(modelB.D, initialize=K)

# Variables: y[t,d] = 1 si la tarea t se asigna al dev d
modelB.y = Var(modelB.T, modelB.D, within=Binary)

# Función objetivo
def objB_rule(model):
    return sum(model.w[t] * model.y[t,d] for t in model.T for d in model.D)
modelB.obj = Objective(rule=objB_rule, sense=maximize)

# Restricción: capacidad global ≤ 52
def cap_global_rule(model):
    return sum(model.p[t] * model.y[t,d] for t in model.T for d in model.D) <= C
modelB.cap_global = Constraint(rule=cap_global_rule)

# Restricción: capacidad individual por desarrollador ≤ 15
def cap_dev_rule(model, d):
    return sum(model.p[t] * model.y[t,d] for t in model.T) <= model.K[d]
modelB.cap_dev = Constraint(modelB.D, rule=cap_dev_rule)

# Restricción: solo un dev por tarea
def unique_dev_rule(model, t):
    return sum(model.y[t,d] for d in model.D) <= 1
modelB.unique_dev = Constraint(modelB.T, rule=unique_dev_rule)

# ==========================================================
# 2. RESOLVER MODELOS
# ==========================================================
from pyomo.opt import SolverFactory
solver = SolverFactory('glpk')

print("=====================================")
print(" SOLUCIÓN PROBLEMA 1: SPRINT ÁGIL ")
print("=====================================")

# --- Resolver Parte A ---
solver.solve(modelA)
print("\n--- PARTE A ---")
print(f"Valor óptimo (Parte A): {modelA.obj():.2f}")
print("Tareas seleccionadas:")
for t in modelA.T:
    if modelA.x[t].value > 0.5:
        print(f" - Tarea {t} (Puntos={modelA.p[t]}, Prioridad={modelA.w[t]})")

# --- Resolver Parte B ---
solver.solve(modelB)
print("\n--- PARTE B ---")
print(f"Valor óptimo (Parte B): {modelB.obj():.2f}")
print("Asignación de tareas a desarrolladores:")
for t in modelB.T:
    for d in modelB.D:
        if modelB.y[t,d].value > 0.5:
            print(f" - Tarea {t} → Dev {d} (Puntos={modelB.p[t]}, Prioridad={modelB.w[t]})")

# ==========================================================
# 3. VISUALIZACIÓN RESULTADOS (PARTE B)
# ==========================================================
# Crear matriz tareas x devs
data = np.zeros((len(modelB.T), len(modelB.D)))
for t in modelB.T:
    for d in modelB.D:
        data[t-1, d-1] = modelB.y[t,d].value

fig, ax = plt.subplots()
cax = ax.matshow(data, cmap="Greens")
plt.colorbar(cax)

ax.set_xticks(range(len(modelB.D)))
ax.set_yticks(range(len(modelB.T)))
ax.set_xticklabels([f"Dev {d}" for d in modelB.D])
ax.set_yticklabels([f"Tarea {t}" for t in modelB.T])

plt.title("Asignación de tareas a desarrolladores (Parte B)")
plt.show()
