import pandas as pd
import numpy as np  
from sklearn import tree
from sklearn.metrics import accuracy_score
import time

def treino(x_train, y_train):
    clf = tree.DecisionTreeClassifier(random_state=1)
    inicio = time.time()
    clf = clf.fit(x_train, y_train)
    fim = time.time()
    print("Tempo de treinamento:", round(fim - inicio, 2), "segundos")

    y_pred = clf.predict(x_train)
    acuracia = accuracy_score(y_train, y_pred)
    print("Acurácia:", acuracia)

    return acuracia

def forward_selection(x_train, y_train, valores, conjuntos_selecionados):
    melhor_acuracia = 0
    melhor_conjunto = (None, None)
    
    for i in range(len(valores)-1):
        comeco = valores[i]
        fim = valores[i+1]

        # evita testar blocos já escolhidos
        if (comeco, fim) in conjuntos_selecionados:
            continue

        print(f"Testando com features de {comeco} a {fim}")

        # concatena blocos já escolhidos + o novo
        x_train_subset = pd.concat(
            [x_train.iloc[:, comeco:fim]] +
            [x_train.iloc[:, c[0]:c[1]] for c in conjuntos_selecionados],
            axis=1
        )

        acuracia = treino(x_train_subset, y_train)

        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            melhor_conjunto = (comeco, fim)

    return melhor_acuracia, melhor_conjunto

if __name__ == "__main__":
    clf = tree.DecisionTreeClassifier(random_state=1)
    n_features = 784
    n_features_por_conjunto = 7
    valores = np.arange(0, n_features + 1, n_features_por_conjunto)

    train = pd.read_csv('mnist_train.csv')
    y_train = train['label'].values 
    x_train = train.drop('label', axis=1)
    
    test = pd.read_csv('mnist_test.csv')
    y_test = test['label'].values 
    x_test = test.drop('label', axis=1)
    
    conjuntos_selecionados = []
    historico = []

    inicio_escolha_features = time.time()

    #limitando por iterações 
    for etapa in range(40):
        print("----------------------------------------")
        melhor_acuracia, melhor_conjunto = forward_selection(
            x_train, y_train,  valores, conjuntos_selecionados
        )

        conjuntos_selecionados.append(melhor_conjunto)
        historico.append({
            'etapa': etapa + 1,
            'melhor_conjunto': melhor_conjunto,
            'acuracia': melhor_acuracia,
            'conjuntos_atuais': conjuntos_selecionados.copy()
        })

        print(f"Conjunto selecionado: {melhor_conjunto} com acurácia {melhor_acuracia:.4f}")
        print(f"Conjuntos atuais: {conjuntos_selecionados}")

    final_escolha_features = time.time()

    for registro in historico:
        print(f"Etapa {registro['etapa']}: Acurácia = {registro['acuracia']:.4f}, "
              f"Bloco adicionado = {registro['melhor_conjunto']}, "
              f"Conjuntos usados = {registro['conjuntos_atuais']}")
        
    x_train_subset = pd.concat(
        [x_train.iloc[:, c[0]:c[1]] for c in conjuntos_selecionados],
        axis=1
    )

    x_test_subset = pd.concat(
        [x_test.iloc[:, c[0]:c[1]] for c in conjuntos_selecionados],
        axis=1
    )


    inicio_treino = time.time()
    clf_final = tree.DecisionTreeClassifier(random_state=1)
    clf_final.fit(x_train_subset, y_train)
    fim_treino = time.time()


    # Avalia no conjunto de teste
    y_test_pred = clf_final.predict(x_test_subset)
    acuracia_teste = accuracy_score(y_test, y_test_pred)                                                                                     

    print("\n")
    print("Métricas finais:")
    print(f"Tempo total usado para seleção das features: {round(final_escolha_features - inicio_escolha_features, 2)} segundos")
    print(f"Tempo de treinamento final: {round(fim_treino - inicio_treino, 2)} segundos")
    print(f"Conjuntos selecionados: {conjuntos_selecionados}")
    print(f"Acurácia no conjunto de teste: {acuracia_teste}")
