import pandas as pd
import numpy as np  
from sklearn import tree
from sklearn.metrics import accuracy_score
import time

def individuo(n_genes):
    return np.array(np.random.choice([0, 1], size=n_genes))

def cruzar(mae, pai):
    ponto_de_corte = len(mae) // 2
    #pega exatamente metade de cada um
    filho1 = np.concatenate((mae[:ponto_de_corte], pai[ponto_de_corte:]))
    filho2 = np.concatenate((pai[:ponto_de_corte], mae[ponto_de_corte:]))
    return filho1, filho2

def mutar(individuo, taxa_de_mutacao=0.05):
    for i in range(len(individuo)):
        #muta com uma certa probabilidade
        if np.random.rand() < taxa_de_mutacao:
            individuo[i] = 1 - individuo[i]
    return individuo

def funcao_aptidao(individuo, x_train, y_train):

    # evita indivíduos sem features selecionadas
    if np.sum(individuo) == 0:
        return 0  
    

    x_filtrado = x_train[:, individuo == 1] #pega todas as colunas onde o valor é 1
    
    #faz o treinamento e calcula a acurácia no treino msm
    clf = tree.DecisionTreeClassifier(random_state=1)
    inicio = time.time()
    clf.fit(x_filtrado, y_train)
    fim = time.time()
    print("Tempo de treinamento (GA):", round(fim - inicio, 2), "segundos")
    y_pred = clf.predict(x_filtrado)
    
    acuracia = accuracy_score(y_train, y_pred)
    porc_features = np.sum(individuo) / len(individuo)
    
    #penaliza muitas features e recompensa acurácias mais altas
    apitdao = 0.9 * acuracia - 0.1 * porc_features
    return apitdao

def roleta(populacao, aptidoes):
    soma_aptidoes = sum(aptidoes)
    probabilidades = [aptidao / soma_aptidoes for aptidao in aptidoes]
    return populacao[np.random.choice(len(populacao), p=probabilidades)]

import numpy as np

def ga_selecao_features(x_train, y_train, n_geracoes=20, n_pop=20, taxa_mut=0.05, n_elite=1):
    n_features = x_train.shape[1]
    populacao = [individuo(n_features) for _ in range(n_pop)]
    
    for g in range(n_geracoes):
        aptidoes = [funcao_aptidao(ind, x_train, y_train) for ind in populacao]
        
        elite_idx = np.argsort(aptidoes)[-n_elite:]
        elite = [populacao[i].copy() for i in elite_idx]

        nova_pop = []
        # Geração de nova população
        while len(nova_pop) < (n_pop - n_elite):
            mae = roleta(populacao, aptidoes)
            pai = roleta(populacao, aptidoes)
            f1, f2 = cruzar(mae, pai)
            f1 = mutar(f1, taxa_mut)
            f2 = mutar(f2, taxa_mut)
            nova_pop += [f1, f2]

        # Garante tamanho correto
        nova_pop = nova_pop[:n_pop - n_elite]
        # Adiciona elite
        populacao = nova_pop + elite

        # Atualiza aptidões após gerar nova população
        aptidoes = [funcao_aptidao(ind, x_train, y_train) for ind in populacao]
        melhor_fit = max(aptidoes)
        melhor = populacao[np.argmax(aptidoes)]
        print(f"Geração {g+1}: melhor fitness = {melhor_fit:.4f}")

    # Retorna melhor da última geração
    melhor = populacao[np.argmax(aptidoes)]
    return melhor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--geracoes', type=int, default=2)
    parser.add_argument('--pop', type=int, default=4)
    parser.add_argument('--mutacao', type=float, default=0.3)
    args = parser.parse_args()

    train = pd.read_csv('mnist_train.csv')
    y_train = train['label'].values 
    x_train = train.drop('label', axis=1)
    
    test = pd.read_csv('mnist_test.csv')
    y_test = test['label'].values 
    x_test = test.drop('label', axis=1)
    
    tempo_inicio = time.time()
    melhor_individuo = ga_selecao_features(x_train.values, y_train, n_geracoes=args.geracoes, n_pop=args.pop, taxa_mut=args.mutacao)
    tempo_fim = time.time()

    features_selecionadas = np.where(melhor_individuo == 1)[0]
    
    #teste no conjunto de teste
    inicio_treino = time.time()
    clf = tree.DecisionTreeClassifier(random_state=1)
    clf.fit(x_train.iloc[:, features_selecionadas], y_train)                                
    fim_treino = time.time()
    y_pred_teste = clf.predict(x_test.iloc[:, features_selecionadas])
    acuracia_teste = accuracy_score(y_test, y_pred_teste)

    print("Tempo para decidir as features:", round(tempo_fim - tempo_inicio, 2), "segundos")
    print("Tempo de treinamento (modelo final):", round(fim_treino - inicio_treino, 2), "segundos")
    print(f"Acurácia no conjunto de teste com features selecionadas: {acuracia_teste:.4f}")
    print(f"Número de features selecionadas: {len(features_selecionadas)}")