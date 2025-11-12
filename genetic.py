import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
import copy


def inicializar_populacao(tamanho_populacao, num_features):
    # Gera 'tamanho_populacao' cromossomos binários de tamanho 'num_features'
    populacao = np.random.randint(0, 2, size=(tamanho_populacao, num_features))
    return populacao

def calcular_fitness(cromossomo, x_train_full, y_train_full):
    # Obtém um array 1-D com os índices das features selecionadas.
    # np.flatnonzero retorna um array com os índices onde a condição é True.
    indices = np.flatnonzero(cromossomo == 1)

    # Se um cromossomo não tiver features, seu fitness é 0
    if indices.size == 0:
        return 0.0

    # 2. Filtra o X_train para usar apenas as features selecionadas
    x_train_subset = x_train_full.iloc[:, indices]
    
    # 3. Divide os dados de treino em sub-treino e sub-validação
    #    Isso é crucial para seguir a regra do TDE.
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train_subset, y_train_full, test_size=0.2, random_state=1, stratify=y_train_full
    )

    # 4. Treina e avalia o modelo
    clf = tree.DecisionTreeClassifier(random_state=1)
    clf.fit(x_fit, y_fit)
    acuracia = accuracy_score(y_val, clf.predict(x_val))
    
    # 5. Penaliza pelo número de features 
    #    Fitness = acuracia - (um pouco * % de features usadas)
    # usa o número total de features do dataset (mais robusto que 784 hard-coded)
    penalidade = 0.01 * (indices.size / x_train_full.shape[1]) 
    
    # Garante que o fitness não seja negativo (para a Roleta)
    return max(0, acuracia - penalidade)

def selecao_roleta(populacao, fitness_scores):
    """
    Seleciona dois pais usando a Seleção por Roleta.
    """
    fitness_total = np.sum(fitness_scores)
    
    # Se todos os fitness forem 0, seleciona aleatoriamente
    if fitness_total == 0:
        indices = np.random.choice(len(populacao), 2, replace=False)
        return populacao[indices], populacao[indices[1]]

    # Calcula as probabilidades de seleção
    probabilidades = fitness_scores / fitness_total
    
    # Seleciona 2 pais (com reposição, pois a roleta permite)
    indices = np.random.choice(len(populacao), 2, replace=True, p=probabilidades)
    
    return populacao[indices], populacao[indices[1]]

def crossover_ponto_unico(pai1, pai2, taxa_crossover):
    """
    Realiza o crossover de ponto único.
    """
    filho1, filho2 = pai1.copy(), pai2.copy()
    
    if np.random.rand() < taxa_crossover:
        # Escolhe um ponto de corte (não nas extremidades)
        ponto = np.random.randint(1, len(pai1) - 1)
        
        # Troca os "rabos" dos cromossomos
        filho1 = np.concatenate((pai1[:ponto], pai2[ponto:]))
        filho2 = np.concatenate((pai2[:ponto], pai1[ponto:]))
        
    return filho1, filho2

def mutacao_bit_flip(cromossomo, taxa_mutacao):
    """
    Aplica a mutação (bit-flip) em cada gene.
    """
    for i in range(len(cromossomo)):
        if np.random.rand() < taxa_mutacao:
            # Inverte o bit (0 -> 1, 1 -> 0)
            cromossomo[i] = 1 - cromossomo[i]
    return cromossomo

# --- Script Principal (main) ---

if __name__ == "__main__":
    
    # --- 1. Definição de Hiperparâmetros do GA ---
    # (Estes são os parâmetros que você deve justificar no TDE) 
    TAMANHO_POPULACAO = 20  # Nro de indivíduos por geração
    TAXA_CROSSOVER = 0.8    # 80% de chance de cruzar
    TAXA_MUTACAO = 0.01     # 1% de chance de mutação por gene
    NUM_GERACOES = 30       # Critério de parada 
    ELITISMO = 1            # Nro de melhores indivíduos mantidos (Elitismo) 
    
    N_FEATURES = 784        # Fixo para o MNIST

    # --- 2. Carregar Dados ---
    print("Carregando dados...")
    train = pd.read_csv('mnist_train.csv')
    y_train = train['label'].values 
    x_train = train.drop('label', axis=1)
    
    test = pd.read_csv('mnist_test.csv')
    y_test = test['label'].values 
    x_test = test.drop('label', axis=1)
    
    # Normalização (boa prática, embora a árvore não precise)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print("Iniciando Algoritmo Genético...")
    inicio_busca = time.time()

    # --- 3. Inicializar População ---
    populacao = inicializar_populacao(TAMANHO_POPULACAO, N_FEATURES)
    
    melhor_cromossomo_global = None
    melhor_fitness_global = -1.0

    # --- 4. Loop Evolutivo ---
    for geracao in range(NUM_GERACOES):
        
        # a. Calcular fitness de toda a população
        fitness_scores = np.array([
            calcular_fitness(c, x_train, y_train) for c in populacao
        ])
        
        # b. Elitismo: Salvar o melhor indivíduo
        melhor_idx_geracao = np.argmax(fitness_scores)
        melhor_fitness_geracao = fitness_scores[melhor_idx_geracao]
        
        if melhor_fitness_geracao > melhor_fitness_global:
            melhor_fitness_global = melhor_fitness_geracao
            #.copy() é crucial para não alterar o melhor salvo
            melhor_cromossomo_global = populacao[melhor_idx_geracao].copy()

        nova_populacao = []
        
        # Adiciona os 'N' melhores (Elitismo)
        indices_elite = np.argsort(fitness_scores)
        for idx in indices_elite:
            nova_populacao.append(populacao[idx].copy())

        # c. Gerar o resto da população (Seleção, Crossover, Mutação)
        while len(nova_populacao) < TAMANHO_POPULACAO:
            # Seleção
            pai1, pai2 = selecao_roleta(populacao, fitness_scores)
            
            # Crossover
            filho1, filho2 = crossover_ponto_unico(pai1, pai2, TAXA_CROSSOVER)
            
            # Mutação
            filho1 = mutacao_bit_flip(filho1, TAXA_MUTACAO)
            filho2 = mutacao_bit_flip(filho2, TAXA_MUTACAO)
            
            # Adiciona os novos filhos
            nova_populacao.append(filho1)
            if len(nova_populacao) < TAMANHO_POPULACAO:
                nova_populacao.append(filho2)
        
        populacao = np.array(nova_populacao)
        
        print(f"Geração {geracao + 1}/{NUM_GERACOES} | "
              f"Melhor Fitness: {melhor_fitness_global:.4f} | "
              f"Features: {np.sum(melhor_cromossomo_global)}")

    fim_busca = time.time()
    
    # --- 5. Resultados da Busca ---
    tempo_busca_features = fim_busca - inicio_busca
    
    print("\nBusca GA concluída.")
    
    # --- 6. Avaliação Final (para Tabela 1) ---
    
    # Pega os índices de features do melhor cromossomo (array 1-D)
    indices_finais = np.flatnonzero(melhor_cromossomo_global == 1)

    print(f"Total de features selecionadas: {indices_finais.size}")

    # Filtra os dados de treino e teste
    x_train_subset = x_train.iloc[:, indices_finais]
    x_test_subset = x_test.iloc[:, indices_finais]

    # Treina o modelo final (conforme TDE)
    clf_final = tree.DecisionTreeClassifier(random_state=1)
    
    inicio_treino = time.time()
    clf_final.fit(x_train_subset, y_train)
    fim_treino = time.time()
    
    tempo_treinamento_final = fim_treino - inicio_treino
    
    # Avalia no conjunto de TESTE 
    y_test_pred = clf_final.predict(x_test_subset)
    acuracia_teste_final = accuracy_score(y_test, y_test_pred)
    
    # --- 7. Imprimir Métricas da Tabela 1 ---
    print("\n")
    print("--- Métricas Finais (Algoritmo Genético) ---")
    print(f"Acurácia (no teste): {acuracia_teste_final * 100:.2f}%")
    print(f"Porcentagem de features: {(indices_finais.size / N_FEATURES) * 100:.2f}%")
    print(f"Tempo de treinamento: {tempo_treinamento_final:.2f} segundos")
    print(f"Tempo para busca das features: {tempo_busca_features:.2f} segundos")