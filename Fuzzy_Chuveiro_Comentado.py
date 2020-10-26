# Importando as bibliotecas
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Preparando os dados para fazer a fuzzificação
temp_agua = ctrl.Antecedent(np.arange(5, 46, 1), 'temp_agua')
clima = ctrl.Antecedent(np.arange(5, 36, 1), 'clima')
potencia = ctrl.Consequent(np.arange(0, 5, 1), 'potencia')

# Fuzzificando
temp_agua['frio'] = fuzz.trimf(temp_agua.universe, [5, 5, 25])
temp_agua['morno'] = fuzz.trimf(temp_agua.universe, [5, 25, 45])
temp_agua['quente'] = fuzz.trimf(temp_agua.universe, [25, 45, 45])

clima['frio'] = fuzz.trimf(clima.universe, [5, 5, 20])
clima['fresco'] = fuzz.trimf(clima.universe, [5, 20, 35])
clima['calor'] = fuzz.trimf(clima.universe, [20, 35, 35])

potencia['muito baixo'] = fuzz.trimf(potencia.universe, [0, 0, 1])
potencia['baixo'] = fuzz.trimf(potencia.universe, [0, 1, 2])
potencia['medio'] = fuzz.trimf(potencia.universe, [1, 2, 3])
potencia['alto'] = fuzz.trimf(potencia.universe, [2, 3, 4])
potencia['muito alto'] = fuzz.trimf(potencia.universe, [3, 4, 4])

# Plotando os gráficos das fuzzificações
temp_agua.view()
clima.view()
potencia.view()

# Definindo as regras para treinar o algoritmo
rule1 = ctrl.Rule(temp_agua['frio'] & clima['frio'], potencia['muito alto'])
rule2 = ctrl.Rule(temp_agua['frio'] & clima['fresco'], potencia['alto'])
rule3 = ctrl.Rule(temp_agua['frio'] & clima['calor'], potencia['medio'])
rule4 = ctrl.Rule(temp_agua['morno'] & clima['frio'], potencia['alto'])
rule5 = ctrl.Rule(temp_agua['morno'] & clima['fresco'], potencia['medio'])
rule6 = ctrl.Rule(temp_agua['morno'] & clima['calor'], potencia['baixo'])
rule7 = ctrl.Rule(temp_agua['quente'] & clima['frio'], potencia['medio'])
rule8 = ctrl.Rule(temp_agua['quente'] & clima['fresco'], potencia['baixo'])
rule9 = ctrl.Rule(temp_agua['quente'] & clima['calor'], potencia['muito baixo'])

# Treinando o algoritmo
potenciaping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
potenciaping = ctrl.ControlSystemSimulation(potenciaping_ctrl)

# Fazendo a defuzzificação e plotando o resultado
potenciaping.input['temp_agua'] = 42
potenciaping.input['clima'] = 33
potenciaping.compute()
print(potenciaping.output['potencia'])
potencia.view(sim=potenciaping)