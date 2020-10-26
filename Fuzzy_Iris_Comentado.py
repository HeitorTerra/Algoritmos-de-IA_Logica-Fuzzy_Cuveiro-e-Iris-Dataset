# Importando todas as bibliotecas
import numpy as np
import skfuzzy as fuzz
import pandas as pd 
from skfuzzy import control as ctrl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Carregando o dataset
iris = load_iris()

# Preparando e dividindo o dataset
df_iris = pd.DataFrame(np.column_stack((iris.data, iris.target)), 
columns = iris.feature_names + ['target'])

X=iris.data[:,(2,3)]
y=range(0,150)

X_iris, X_test, y_iris, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Preparando os dados para fazer a fuzzificação
petal_length = ctrl.Antecedent(np.arange(min(X_iris[:,0]),max(X_iris[:,0]),0.1), 'petal_length')
petal_width = ctrl.Antecedent(np.arange(min(X_iris[:,1]),max(X_iris[:,1]),0.1), 'petal_width')
flor = ctrl.Consequent(np.arange(0, 4.5, 0.5), 'flor')

# Fuzzificando
petal_length.automf(5)
petal_width.automf(3)
flor['Iris Setosa'] = fuzz.trimf(flor.universe, [0, 1, 2])
flor['Iris Versicolour'] = fuzz.trimf(flor.universe, [1, 2, 3])
flor['Iris Virginica'] = fuzz.trimf(flor.universe, [2, 3, 4])

# Plotando os gráficos das fuzzificações
petal_width.view()
petal_length.view()
flor.view()

# Definindo as regras para treinar o algoritmo
rule1 = ctrl.Rule(petal_width['poor'] | petal_length['poor'] , flor['Iris Setosa'])
rule2 = ctrl.Rule(petal_width['good'] | petal_length['mediocre'] | petal_length['average'] , flor['Iris Versicolour'])
rule3 = ctrl.Rule(petal_width['average'] | petal_length['good'] | petal_length['decent'], flor['Iris Virginica'])

# Treinando o algoritmo
florping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
florping = ctrl.ControlSystemSimulation(florping_ctrl)

# Fazendo a defuzzificação do dataset de treino e printando os resultados
for i in range(len(X_iris[:,0])):
    florping.input['petal_length'] = X_iris[i,0]
    florping.input['petal_width'] = X_iris[i,1]
    florping.compute()
    print(florping.output['flor'])
flor.view(sim=florping)

#Fazendo a defuzzificação so dataset de teste e printando os resultados
for i in range(len(X_test[:, 0])):
    florping.input['petal_length'] = X_test[i,0]
    florping.input['petal_width'] = X_test[i,1]
    florping.compute()
    print(florping.output['flor'])
flor.view(sim=florping)

