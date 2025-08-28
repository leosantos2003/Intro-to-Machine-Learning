import matplotlib.pyplot as plt
import numpy as np

def plotar_comparacao_barras(previsoes, valores_reais, titulo, nome_arquivo):
    """
    Cria e salva um gráfico de barras agrupadas comparando previsões e valores reais.

    Args:
        previsoes (list): Lista com os valores previstos pelo modelo.
        valores_reais (list): Lista com os valores reais (alvo).
        titulo (str): Título do gráfico.
        nome_arquivo (str): Nome do arquivo para salvar a imagem (ex: 'meu_grafico.png').
    """
    n_items = len(previsoes)
    labels = [f'Casa {i+1}' for i in range(n_items)]
    x = np.arange(len(labels))  # Posições dos rótulos
    width = 0.35  # Largura das barras

    # Cria a figura e os eixos
    fig, ax = plt.subplots(figsize=(12, 7))

    # Cria as barras de previsão e valores reais
    rects_pred = ax.bar(x - width/2, previsoes, width, label='Previsão', color='#3498db')
    rects_actual = ax.bar(x + width/2, valores_reais, width, label='Valor Real', color='#2ecc71')

    # Adiciona textos, título e rótulos
    ax.set_ylabel('Preço de Venda ($)', fontsize=12)
    ax.set_title(titulo, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Formata os valores em cima das barras para ficarem mais legíveis
    ax.bar_label(rects_pred, padding=3, fmt='${:,.0f}', rotation=45)
    ax.bar_label(rects_actual, padding=3, fmt='${:,.0f}', rotation=45)

    fig.tight_layout()
    plt.savefig(nome_arquivo)
    print(f"Gráfico '{nome_arquivo}' salvo com sucesso.")