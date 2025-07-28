import json
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

# ----- Escalas pré-definidas FNIRSI -----
voltList = [[5.0, "V", 1], [2.5, "V", 1], [1.0, "V", 1], [500, "mV", 0.001],
            [200, "mV", 0.001], [100, "mV", 0.001], [50, "mV", 0.001]]
timeList = [[50, "S", 1], [20, "S", 1], [10, "S", 1], [5, "S", 1], [2, "S", 1], [1, "S", 1],
            [500, "mS", 0.001], [200, "mS", 0.001], [100, "mS", 0.001], [50, "mS", 0.001],
            [20, "mS", 0.001], [10, "mS", 0.001], [5, "mS", 0.001], [2, "mS", 0.001], [1, "mS", 0.001],
            [500, "uS", 1e-6], [200, "uS", 1e-6], [100, "uS", 1e-6], [50, "uS", 1e-6], [20, "uS", 1e-6],
            [10, "uS", 1e-6], [5, "uS", 1e-6], [2, "uS", 1e-6], [1, "uS", 1e-6],
            [500, "nS", 1e-9], [200, "nS", 1e-9], [100, "nS", 1e-9], [50, "nS", 1e-9], [20, "nS", 1e-9],
            [10, "nS", 1e-9]]

# ----- Buffers de dados -----
header = [bytes(208)]
dataBuff = [bytes(3000), bytes(3000)]

# ----- Objeto JSON de saída -----
jsObj = {
    "voltage": {
        "volts": [0, 0],
        "units": ["", ""],
        "multiplier": [0.001, 0.001],
        "probe": [1, 1],
        "coupling": ["DC", "DC"]
    },
    "timebase": {
        "time": 0,
        "units": "",
        "multiplier": 1e-6
    },
    "dataBuffer": [
        {"channel": "CH1", "units": "mV", "values": [0] * 1500},
        {"channel": "CH2", "units": "mV", "values": [0] * 1500}
    ]
}

# ----- Leitura do arquivo WAV -----
def getBinaryData(filename):
    with open(filename, "rb") as f:
        f.seek(0)
        header[0] = f.read(208)
        f.seek(1000)
        dataBuff[0] = f.read(3000)
        f.seek(4000)
        dataBuff[1] = f.read(3000)

# ----- Decodificação do cabeçalho -----
def parseHeader():
    voltScale = []
    for x in range(2):
        scale = voltList[header[0][4 + x * 10]]
        jsObj["voltage"]["volts"][x] = scale[0]
        jsObj["voltage"]["units"][x] = scale[1]
        jsObj["voltage"]["multiplier"][x] = scale[2]
        jsObj["voltage"]["probe"][x] = [1, 10, 100][header[0][10 + x * 10]]
        jsObj["voltage"]["coupling"][x] = ["DC", "AC"][header[0][8 + x * 10]]
        voltScale.append(scale)

    ts = timeList[header[0][22]]
    jsObj["timebase"]["time"] = ts[0]
    jsObj["timebase"]["units"] = ts[1]
    jsObj["timebase"]["multiplier"] = ts[2]
    return voltScale, ts[2]  # retorna também o dt para gráfico

# ----- Decodificação dos dados -----
def parseData(voltScale):
    for x in range(2):
        for y in range(1500):
            val = (dataBuff[x][y * 2] + 256 * dataBuff[x][y * 2 + 1] - 200) * voltScale[x][0] / 50
            jsObj["dataBuffer"][x]["values"][y] = val
        jsObj["dataBuffer"][x]["units"] = voltScale[x][1]

# ----- Salvamento em JSON -----
def saveJson(path):
    with open(path, "w") as f:
        json.dump(jsObj, f, indent=1)

# ----- Exibição do gráfico -----
def plotar():
    tempo = jsObj["timebase"]
    dt = tempo["multiplier"]
    t = [i * dt for i in range(1500)]
    for canal in jsObj["dataBuffer"]:
        plt.plot(t, canal["values"], label=canal["channel"] + f" ({canal['units']})")

    plt.title("FNIRSI 1014D - Forma de Onda Decodificada")
    plt.xlabel(f"Tempo [{tempo['units']}]")
    plt.ylabel("Tensão")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----- Interface para selecionar o arquivo -----
def selecionar_arquivo():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Selecione o arquivo .wav exportado do FNIRSI",
        filetypes=[("Arquivos WAV", "*.wav")]
    )

# ========== EXECUÇÃO ==========
wav_path = selecionar_arquivo()
if not wav_path:
    print("Nenhum arquivo selecionado.")
    exit()

getBinaryData(wav_path)
voltScale, dt = parseHeader()
parseData(voltScale)

json_path = os.path.splitext(wav_path)[0] + ".json"
saveJson(json_path)

print(f"JSON salvo em: {json_path}")
plotar()
