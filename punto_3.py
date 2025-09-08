# ==========================================================
# PROBLEMA 3: Logística de Misión Humanitaria en Pyomo
# ==========================================================
# Objetivo: Maximizar el valor de recursos transportados 
# en 3 aviones, respetando restricciones de peso, volumen 
# y disponibilidad de recursos.
# Parte B incluye restricciones adicionales de seguridad
# e incompatibilidad.
# ==========================================================
 
from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------
# 1. CREACIÓN DEL MODELO
# ----------------------------------------------------------
model = ConcreteModel()

# ----------------------------------------------------------
# 2. CONJUNTOS
# ----------------------------------------------------------
# Recursos: 1=Alimentos, 2=Medicinas, 3=Equipos Médicos,
#           4=Agua Potable, 5=Mantas
# Aviones: 1, 2, 3
model.R = RangeSet(5)
model.A = RangeSet(3)

# ----------------------------------------------------------
# 3. PARÁMETROS
# ----------------------------------------------------------
# Valor de cada recurso por tonelada
v = {1:50, 2:100, 3:120, 4:60, 5:40}
# Stock disponible (toneladas)
s = {1:15, 2:5, 3:20, 4:18, 5:10}
# Volumen por tonelada (m3)
u = {1:8, 2:2, 3:10, 4:12, 5:6}
# Capacidad de peso por avión (toneladas)
W = {1:30, 2:40, 3:50}
# Capacidad de volumen por avión (m3)
U = {1:25, 2:30, 3:35}

model.v = Param(model.R, initialize=v)
model.s = Param(model.R, initialize=s)
model.u = Param(model.R, initialize=u)
model.W = Param(model.A, initialize=W)
model.U = Param(model.A, initialize=U)

# ----------------------------------------------------------
# 4. VARIABLES DE DECISIÓN
# ----------------------------------------------------------
# x[r,a] = toneladas del recurso r transportadas en avión a
model.x = Var(model.R, model.A, within=NonNegativeReals)

# ----------------------------------------------------------
# 5. FUNCIÓN OBJETIVO
# ----------------------------------------------------------
def obj_rule(model):
    return sum(model.v[r] * model.x[r,a] for r in model.R for a in model.A)
model.obj = Objective(rule=obj_rule, sense=maximize)

# ----------------------------------------------------------
# 6. RESTRICCIONES
# ----------------------------------------------------------

# 6.1 Restricción de stock (disponibilidad de recursos)
def stock_rule(model, r):
    return sum(model.x[r,a] for a in model.A) <= model.s[r]
model.stock = Constraint(model.R, rule=stock_rule)

# 6.2 Restricción de peso (capacidad de cada avión)
def weight_rule(model, a):
    return sum(model.x[r,a] for r in model.R) <= model.W[a]
model.weight = Constraint(model.A, rule=weight_rule)

# 6.3 Restricción de volumen (capacidad de cada avión)
def volume_rule(model, a):
    return sum(model.u[r] * model.x[r,a] for r in model.R) <= model.U[a]
model.volume = Constraint(model.A, rule=volume_rule)

# ----------------------------------------------------------
# 7. RESTRICCIONES ADICIONALES (PARTE B)
# ----------------------------------------------------------

# 7.1 Seguridad: Medicinas (r=2) no pueden ir en Avión 1
def security_rule(model):
    return model.x[2,1] == 0
model.security = Constraint(rule=security_rule)

# 7.2 Incompatibilidad: Equipos Médicos (r=3) y Agua (r=4)
# no pueden ir en el mismo avión.
# Usamos una restricción lógica con Big-M.
M = 1000  # constante grande
model.y_med = Var(model.A, within=Binary)   # binaria: hay equipos médicos en avión a
model.y_agua = Var(model.A, within=Binary)  # binaria: hay agua en avión a

# Restricción Big-M para equipos médicos
def med_bigM(model, a):
    return model.x[3,a] <= M * model.y_med[a]
model.med_bigM = Constraint(model.A, rule=med_bigM)

# Restricción Big-M para agua potable
def agua_bigM(model, a):
    return model.x[4,a] <= M * model.y_agua[a]
model.agua_bigM = Constraint(model.A, rule=agua_bigM)


# Restricción de incompatibilidad (no ambos en el mismo avión)
def incompat_rule(model, a):
    return model.y_med[a] + model.y_agua[a] <= 1
model.incompat = Constraint(model.A, rule=incompat_rule)

# ----------------------------------------------------------
# 8. RESOLVER EL MODELO
# ----------------------------------------------------------
solver = SolverFactory('glpk')
solver.solve(model)

# ----------------------------------------------------------
# 9. RESULTADOS
# ----------------------------------------------------------
print("=====================================")
print(" SOLUCIÓN DEL PROBLEMA DE LOGÍSTICA ")
print("=====================================")
print(f"Valor óptimo transportado: {model.obj():.2f}\n")

for r in model.R:
    for a in model.A:
        if model.x[r,a].value > 0:
            print(f"Recurso {r} en Avión {a}: {model.x[r,a].value:.2f} ton")

# ----------------------------------------------------------
# 10. VISUALIZACIÓN DE RESULTADOS
# ----------------------------------------------------------
# Crear matriz recurso x avión
data = np.zeros((len(model.R), len(model.A)))
for r in model.R:
    for a in model.A:
        data[r-1, a-1] = model.x[r,a].value

fig, ax = plt.subplots()
cax = ax.matshow(data, cmap="Blues")
plt.colorbar(cax)

ax.set_xticks(range(len(model.A)))
ax.set_yticks(range(len(model.R)))
ax.set_xticklabels([f"Avión {a}" for a in model.A])
ax.set_yticklabels(["Alimentos","Medicinas","Equipos Médicos","Agua","Mantas"])

plt.title("Asignación de recursos a aviones")
plt.show()
