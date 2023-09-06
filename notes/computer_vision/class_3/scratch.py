from statistics import stdev as sample_stdev
from statistics import pstdev as pop_stdev


pressures = []
for i in range(0,3):
    pressures.append(20)

for i in range(0, 4):
    pressures.append(22)

for i in range(0, 2):
    pressures.append(25)

print(sample_stdev(pressures))

def f(x):
    return (5/3)*x**3

print(f(2) - f(1))

