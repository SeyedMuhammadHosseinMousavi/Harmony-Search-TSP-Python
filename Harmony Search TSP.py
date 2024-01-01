import numpy as np
import matplotlib.pyplot as plt

# Cost Function
def CostF(s, model):
    d = model['d']
    tour = np.argsort(s)
    sol = {'tour': tour}
    n = len(tour)
    tour = np.append(tour, tour[-1])
    L = -1
    for i in range(n):
        L += d[tour[i], tour[i+1]]
    sol['L'] = L
    z = L
    return z, sol

# Model Function
def MakeModel3():
    #x = np.array([13, 25, 91, 86, 66, 87, 50])
    #y = np.array([19, 28, 37, 100, 10, 32, 56])
    x = np.array([13, 25, 91, 86, 66, 87, 50])
    y = np.array([19, 28, 37, 100, 10, 32, 56])
    n = len(x)
    d = np.zeros((n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            d[i, j] = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            d[j, i] = d[i, j]
    model = {'n': n, 'x': x, 'y': y, 'd': d}
    return model

# Plot Function
def Plotfig(tour, model):
    tour = np.append(tour, tour[-1])
    x = model['x']
    y = model['y']
    plt.plot(x[tour], y[tour], '-hr', linewidth=3, markersize=15, markerfacecolor=[0.1, 0.9, 0.2], markeredgecolor='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('TSP Tour')
    plt.grid(True)
    plt.show()

# Main Code
np.random.seed(0)

# Problem
model = MakeModel3()
CostFunction = lambda s: CostF(s, model)  # Cost Function
nVar = model['n']             # Number of Decision Variables
VarSize = (1, nVar)           # Decision Variables Matrix Size
VarMin = 0                    # Lower Bound of Variables
VarMax = 1                    # Upper Bound of Variables

# Harmony Search Parameters
MaxIt = 500                  # Maximum Number of Iterations
HMS = 7000                      # Harmony Memory Size
nNew = 100                     # Number of New Harmonies
HMCR = 0.9                   # Harmony Memory Consideration Rate
PAR = 0.1                   # Pitch Adjustment Rate
FW = 0.02*(VarMax-VarMin)     # Fret Width (Bandwidth)
FW_damp = 0.995              # Fret Width Damp Ratio

# Initialization
empty_harmony = {'Position': None, 'Cost': None, 'Sol': None}
HM = [empty_harmony.copy() for _ in range(HMS)]

# Create Initial Harmonies
for i in range(HMS):
    HM[i]['Position'] = np.random.uniform(VarMin, VarMax, VarSize)
    HM[i]['Cost'], HM[i]['Sol'] = CostFunction(HM[i]['Position'])

# Sort Harmony Memory
HM.sort(key=lambda x: x['Cost'])
BestSol = HM[-1]
BestCost = []

# Harmony Search Main Loop
for it in range(MaxIt):
    NEW = [empty_harmony.copy() for _ in range(nNew)]
    for k in range(nNew):
        NEW[k]['Position'] = np.random.uniform(VarMin, VarMax, VarSize)
        for j in range(nVar):
            if np.random.rand() <= HMCR:
                i = np.random.randint(-1, HMS)
                NEW[k]['Position'][-1][j] = HM[i]['Position'][-1][j]
            if np.random.rand() <= PAR:
                DELTA = FW * np.random.randn()
                NEW[k]['Position'][-1][j] += DELTA
        NEW[k]['Position'] = np.maximum(NEW[k]['Position'], VarMin)
        NEW[k]['Position'] = np.minimum(NEW[k]['Position'], VarMax)
        NEW[k]['Cost'], NEW[k]['Sol'] = CostFunction(NEW[k]['Position'])
    HM.extend(NEW)
    HM.sort(key=lambda x: x['Cost'])
    HM = HM[:HMS]
    BestSol = HM[-1]
    BestCost.append(BestSol['Cost'])
    print('Iteration {}: Best Cost = {}'.format(it+1, BestCost[it]))
    FW *= FW_damp

# Plot Results
Plotfig(BestSol['Sol']['tour'], model)
plt.figure()
plt.plot(BestCost, 'k', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost Value')
plt.title('Best Cost')
plt.grid(True)
plt.show()


# Print the Best Solution's Cost and Tour
print("Harmony Search Best Cost:", BestSol['Cost'])
print("TSP Best Tour:", BestSol['Sol']['tour'])

