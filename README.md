# Gerador de Sinais Avançado com FFT, Modulação e Análise Interativa

Este aplicativo Python fornece um ambiente gráfico completo para geração, visualização e análise de sinais no domínio do tempo e frequência. Com recursos profissionais como interpolação adaptativa, marcadores interativos, modulação AM/FM e exportação de dados, o sistema é ideal para engenheiros, pesquisadores e estudantes de processamento de sinais.

---

## 🎯 Funcionalidades Principais

### 🖥️ Interface Moderna

- Construída com `CustomTkinter` (modo escuro responsivo)
- Layout dividido em:
  - Painel de parâmetros (esquerda)
  - Área de gráficos (`tempo` e `FFT`)
  - Painel lateral com marcadores
  - Barra de status no rodapé

### 📈 Visualização e Análise

- Geração de sinais básicos: Senoidal, Quadrado, Triangular, Impulso, Passo, Aleatório
- Modulação AM/FM com parâmetros configuráveis (ganho, desvio)
- Transformada de Fourier (FFT) com janelamento e normalização
- Interpolação automática por `CubicSpline` ao aplicar zoom (melhora a suavidade)
- Ajuste dinâmico de frequência com sliders

### 📌 Marcadores Interativos

- Até 2 marcadores verticais e 2 horizontais por gráfico
- Arrastáveis com o mouse
- Valores de tempo/frequência/diferença atualizados em tempo real
- Respeitam os limites do gráfico visível
- Painel lateral com Δt, Δf, Δy, Δ|Y| organizados

### 💾 Entrada e Exportação de Dados

- Importa sinais de arquivos `.wav`
- Exporta dados de tempo (`t`, `y`) e frequência (`f`, `|Y|`) em `.csv` e `.json`
- Normalização e detecção automática de taxa de amostragem

### 💬 Barra de Status

- Mensagens claras para o usuário
- Exibe estados como: marcador adicionado, erro de limite, exportação concluída etc.

---

## 📁 Estrutura do Projeto

```
📂 FFT Signal Generator
├── V8.py               # Arquivo principal
├── README.md           # Este documento
├── requirements.txt    # Dependências do projeto
├── examples/           # Exemplos de sinais ou .wav
└── export/             # Arquivos exportados (.csv, .json)
```

---

## ⚙️ Requisitos

- Python 3.9 ou superior
- Bibliotecas:
  - `customtkinter`
  - `matplotlib`
  - `numpy`
  - `scipy`
  - `soundfile` (para leitura de WAV)
  - `json`, `csv`, `tkinter` (nativos)

Instalação recomendada:

```bash
pip install -r requirements.txt
```

---

## 🚀 Execução

```bash
python V8.py
```

---

## 🔍 Testes Recomendados

- Modificar tipo de sinal e observar a FFT
- Ativar modulação AM ou FM com sliders
- Adicionar dois marcadores verticais e horizontais
- Arrastar marcadores e verificar valores no painel lateral
- Exportar dados via botão “Exportar CSV”

---

## 🧑‍💻 Autor

**Geraldo César Simão**  
Engenheiro em Telecomunicações, especialista em RF, processamento de sinais e eletrônica aplicada.

---

## 📜 Licença

Uso livre para fins acadêmicos e pessoais. Para uso comercial ou institucional, entre em contato com o autor.