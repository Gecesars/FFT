# 🎛️ FFT - Analisador Avançado de Sinais

Este repositório contém um analisador gráfico interativo de sinais com suporte completo a FFT (Transformada Rápida de Fourier), geração de formas de onda, exportação de dados e manipulação com marcadores.

Desenvolvido em Python com interface `CustomTkinter`, o projeto é ideal para engenheiros, pesquisadores, estudantes e entusiastas que desejam uma ferramenta prática para análise no domínio do tempo e da frequência.

---

## 🧠 Destaques do Projeto

- ✅ Geração de sinais: Senoidal, Quadrada, Pulso, Dente de Serra, Ruído Branco, Chirp
- ✅ Modulação AM e FM com ajuste de portadora
- ✅ FFT com janelas: Hanning, Hamming, Blackman, Gaussian, entre outras
- ✅ Interpolação cúbica para sinais de baixa resolução
- ✅ Marcadores interativos com leitura de Δx, Δy, Δf, Δ|Y|
- ✅ Visualização simultânea nos domínios do tempo e da frequência
- ✅ Exportação de dados em `.csv` e `.json`
- ✅ Interface com múltiplos painéis (barras, status, controles)
- ✅ Suporte a arquivos `.wav` e análise de dados amostrados
- ✅ Zoom adaptativo com ajuste de resolução espectral
- ✅ Detecção automática de frequência principal (pico do espectro)
- ✅ Ambiente de código limpo, modular e escalável

---

## 🚀 Arquivo Principal

O ponto de entrada da aplicação é:

```bash
python main.py
```

Esse arquivo centraliza a inicialização da interface, definição dos eventos e criação das threads de análise e exportação.

---

## 🗂️ Estrutura de Diretórios

```
FFT/
├── main.py                 # Arquivo principal da aplicação (GUI + núcleo lógico)
├── versao15.py             # Versão completa com mais de 721 linhas (base de desenvolvimento)
├── utils/                  # Módulos auxiliares e lógicas específicas
├── assets/                 # Arquivos exportados, temporários ou amostrados
├── README.md               # Documento descritivo
├── requirements.txt        # Lista de dependências
└── .gitignore              # Exclusões de controle de versão
```

---

## 🔧 Requisitos

- Python 3.9+

### 📦 Instale as dependências

```bash
pip install -r requirements.txt
```

Ou manualmente:

```bash
pip install customtkinter matplotlib numpy scipy pillow
```

---

## 📊 Exemplo de uso

1. Selecione "Gerar Sinal Senoidal" com 10 kHz
2. Escolha "FFT com janela Hamming"
3. Marque o pico principal e visualize Δf
4. Exporte os dados para `resultados.csv`

---

## 💡 Aplicações

- Análise espectral de sinais laboratoriais
- Processamento digital de sinais (DSP)
- Verificação de formas de onda simuladas
- Ensino e demonstrações didáticas
- Comparação entre sinais reais e teóricos

---

## 📞 Contato do Autor

- **Nome:** Geraldo César Simão
- **Telefone / WhatsApp:** +55 35 91017-3582
- **E-mails:**
  - gecesars@gmail.com
  - geraldo_cesar_si@hotmail.com

---

## 📄 Licença

Distribuído sob a Licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.

---

**Este projeto está em constante evolução. Contribuições, testes e sugestões são bem-vindos!**