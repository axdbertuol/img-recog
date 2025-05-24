import numpy as np
import scipy.io

def decode_label(label):
    if label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(label + 55)
    else:
        return chr(label + 61)

def carregar_dados():
    mat = scipy.io.loadmat('emnist-byclass.mat')

    X_test = mat['dataset'][0][0][1][0][0][0].astype(np.float32)
    y_test = mat['dataset'][0][0][1][0][0][1].flatten().astype(int)

    X_train = mat['dataset'][0][0][0][0][0][0].astype(np.float32)[:500]
    y_train = mat['dataset'][0][0][0][0][0][1].flatten().astype(int)[:500]

    return X_train, y_train, X_test, y_test

def distancia_euclidiana(a, b, mostrar_passos=False, max_passos=100):
    """
    Calcula a distância euclidiana entre dois vetores de números reais.

    Parameters
    ----------
    a : array-like
        Primeiro vetor de números reais.
    b : array-like
        Segundo vetor de números reais.
    mostrar_passos : bool, optional
        Se True, retorna uma lista com os detalhes de cada passo do cálculo da distância
        euclidiana. Se False, retorna apenas a distância calculada.
    max_passos : int, optional
        Número máximo de passos a serem mostrados se `mostrar_passos` for True.

    Returns
    -------
    distancia : float
        Distância euclidiana entre os vetores `a` e `b`.
    detalhes : list, optional
        Lista com os detalhes de cada passo do cálculo da distância euclidiana, se
        `mostrar_passos` for True.
    """
    soma_quadrados = 0
    detalhes = []

    for i in range(len(a)):
        diferenca = a[i] - b[i]
        quadrado = diferenca ** 2
        soma_quadrados += quadrado

        if mostrar_passos and i < max_passos:
            detalhes.append({
                "pixel": i + 1,
                "valor_a": round(a[i], 4),
                "valor_b": round(b[i], 4),
                "diferenca": round(diferenca, 4),
                "quadrado": round(quadrado, 6)
            })

    distancia = np.sqrt(soma_quadrados)
    return distancia, detalhes


def prever_numero(nova_imagem, X_train, y_train, mostrar_passos=False):
    distancias = []
    detalhes_lista = []
    imagens_comparadas = []

    for img, rotulo in zip(X_train, y_train):
        d, detalhes = distancia_euclidiana(nova_imagem, img, mostrar_passos=mostrar_passos)
        distancias.append((d, rotulo, img))  # Também armazena a imagem comparada
        if mostrar_passos:
            detalhes_lista.append(detalhes)
            imagens_comparadas.append(img)

    distancias.sort(key=lambda x: x[0])
    melhor_distancia, melhor_rotulo, melhor_imagem = distancias[0]
    return melhor_rotulo, melhor_distancia, (detalhes_lista[0] if mostrar_passos else None), melhor_imagem