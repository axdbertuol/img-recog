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
    data = np.load('emnist_data.npz')  # ✅ Carrega corretamente o arquivo .npz

    X_train = data['X_train'].astype(np.float32)
    y_train = data['y_train'].astype(int)
    X_test = data['X_test'].astype(np.float32)
    y_test = data['y_test'].astype(int)

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
    """
    Calcula a distância euclidiana entre uma imagem e todas as imagens treinadas e retorna
    o rótulo da imagem mais semelhante.

    Parameters
    ----------
    nova_imagem : array-like
        Vetor de números reais representando a imagem a ser reconhecida.
    X_train : array-like
        Vetor de vetores de números reais representando as imagens treinadas.
    y_train : array-like
        Vetor de rótulos (inteiros) das imagens treinadas.
    mostrar_passos : bool, optional
        Se True, retorna os detalhes de cada passo do cálculo da distância euclidiana
        para a imagem mais semelhante. Se False, retorna apenas a distância calculada.

    Returns
    -------
    melhor_rotulo : int
        Rótulo da imagem mais semelhante.
    melhor_distancia : float
        Distância euclidiana entre a imagem a ser reconhecida e a imagem mais semelhante.
    detalhes : list, optional
        Lista com os detalhes de cada passo do cálculo da distância euclidiana, se
        `mostrar_passos` for True.
    melhor_imagem : array-like
        Vetor de números reais representando a imagem mais semelhante.
    """
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