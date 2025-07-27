import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import square, sawtooth, unit_impulse
from scipy.fft import fft, fftfreq, fftshift
import csv
import json
from concurrent.futures import ThreadPoolExecutor

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class SignalGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gerador e Analisador de Sinais")
        self.geometry("1200x950")

        self.executor = ThreadPoolExecutor(max_workers=20)

        self.func_types = [
            "Seno", "Cosseno", "Quadrada", "Triangular",
            "Dente de Serra", "Pulso", "Ruído Branco", "Seno Modulado",
            "Exp Decaimento", "Seno Hiperbólico", "Tangente", "Tangente Hiperbólica",
            "Passo (step)", "Rampa", "Parábola", "Impulso Infinito"
        ]

        self.sidebar = ctk.CTkFrame(self, width=250)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.label_points = ctk.CTkLabel(self.sidebar, text="Pontos:")
        self.label_points.pack(pady=(5,0))
        self.entry_points = ctk.CTkEntry(self.sidebar)
        self.entry_points.insert(0, "256")
        self.entry_points.pack(pady=(0,5))

        self.label_freq = ctk.CTkLabel(self.sidebar, text="Frequência:")
        self.label_freq.pack(pady=(5,0))
        self.entry_freq = ctk.CTkEntry(self.sidebar)
        self.entry_freq.insert(0, "5")
        self.entry_freq.pack(pady=(0,5))

        self.label_fs = ctk.CTkLabel(self.sidebar, text="Taxa de Amostragem (Hz):")
        self.label_fs.pack(pady=(5,0))
        self.entry_fs = ctk.CTkEntry(self.sidebar)
        self.entry_fs.insert(0, "1000")
        self.entry_fs.pack(pady=(0,5))

        self.label_func = ctk.CTkLabel(self.sidebar, text="Função:")
        self.label_func.pack(pady=(5,0))
        self.combo_func = ctk.CTkComboBox(self.sidebar, values=self.func_types)
        self.combo_func.set("Seno")
        self.combo_func.pack(pady=(0,10))

        # Modulação AM
        self.mod_am = tk.BooleanVar(value=False)
        self.am_checkbox = ctk.CTkCheckBox(
            self.sidebar, text="Modulação AM", variable=self.mod_am, command=self.plot_signal
        )
        self.am_checkbox.pack(pady=2)
        self.am_depth = ctk.CTkSlider(
            self.sidebar, from_=0, to=1, number_of_steps=100, command=self.plot_signal
        )
        self.am_depth.set(0.5)
        self.am_depth.pack(pady=(0,5))
        self.am_depth_label = ctk.CTkLabel(self.sidebar, text="Profundidade AM")
        self.am_depth_label.pack(pady=(0,10))

        # Modulação FM
        self.mod_fm = tk.BooleanVar(value=False)
        self.fm_checkbox = ctk.CTkCheckBox(
            self.sidebar, text="Modulação FM", variable=self.mod_fm, command=self.plot_signal
        )
        self.fm_checkbox.pack(pady=2)
        self.fm_deviation = ctk.CTkSlider(
            self.sidebar, from_=0, to=1, number_of_steps=100, command=self.plot_signal
        )
        self.fm_deviation.set(50)
        self.fm_deviation.pack(pady=(0,5))
        self.fm_deviation_label = ctk.CTkLabel(self.sidebar, text="Desvio FM")
        self.fm_deviation_label.pack(pady=(0,10))

        # Botões principais
        self.button_plot = ctk.CTkButton(
            self.sidebar, text="Gerar Sinal", command=self.plot_signal
        )
        self.button_plot.pack(pady=10)

        self.button_export = ctk.CTkButton(
            self.sidebar, text="Exportar CSV/JSON", command=self.export_data
        )
        self.button_export.pack(pady=10)

        self.button_reset = ctk.CTkButton(
            self.sidebar, text="Resetar Zoom", command=self.reset_zoom
        )
        self.button_reset.pack(pady=10)

        # Área de plotagem
        self.plot_area = ctk.CTkFrame(self)
        self.plot_area.pack(side="right", fill="both", expand=True)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Zoom sliders
        self.zoom_time = ctk.CTkSlider(
            self.plot_area, from_=0.01, to=1.0, number_of_steps=100,
            command=self.update_time_zoom
        )
        self.zoom_time.set(1.0)
        self.zoom_time.pack(pady=5)

        self.zoom_freq = ctk.CTkSlider(
            self.plot_area, from_=0.01, to=1.0, number_of_steps=100,
            command=self.update_freq_zoom
        )
        self.zoom_freq.set(1.0)
        self.zoom_freq.pack(pady=5)

        # Marcadores
        self.marks = []
        self.canvas.mpl_connect('button_press_event', self._on_right_click)

        # Menu de contexto
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Marcar Tempo", command=self.add_time_marker)
        self.menu.add_command(label="Marcar Frequência", command=self.add_freq_marker)

    def plot_signal(self, *_):
        try:
            N = int(self.entry_points.get())
            freq = float(self.entry_freq.get())
            Fs = float(self.entry_fs.get())
        except ValueError:
            messagebox.showerror("Erro", "Insira valores numéricos válidos.")
            return

        # Validações
        if N < 128 or N % 2 != 0:
            messagebox.showerror("Erro", "Pontos ≥128 e múltiplo de 2")
            return
        if freq >= Fs / 2:
            messagebox.showerror("Erro de Nyquist", "Freq ≥ Fs/2")
            return

        # Ajusta desvio FM
        self.fm_deviation.configure(to=Fs/2)
        if self.mod_fm.get() and (freq + self.fm_deviation.get()) >= Fs/2:
            messagebox.showerror("Erro de Nyquist", "Freq+Δf ≥ Fs/2")
            return

        t = np.arange(N) / Fs
        func = self.combo_func.get()

        # Geração de forma de onda
        if func == "Seno":
            y = np.sin(2 * np.pi * freq * t)
        elif func == "Cosseno":
            y = np.cos(2 * np.pi * freq * t)
        elif func == "Quadrada":
            y = square(2 * np.pi * freq * t)
        elif func == "Triangular":
            y = sawtooth(2 * np.pi * freq * t, 0.5)
        elif func == "Dente de Serra":
            y = sawtooth(2 * np.pi * freq * t)
        elif func == "Pulso":
            y = np.where((t % (1/freq)) < (1/(2*freq)), 1, 0)
        elif func == "Ruído Branco":
            y = np.random.normal(0, 1, N)
        elif func == "Seno Modulado":
            y = np.sin(2 * np.pi * freq * t) * np.sin(2 * np.pi * freq/10 * t)
        elif func == "Exp Decaimento":
            y = np.exp(-5 * t) * np.sin(2 * np.pi * freq * t)
        elif func == "Seno Hiperbólico":
            y = np.sinh(freq * t)
        elif func == "Tangente":
            y = np.tan(2 * np.pi * freq * t)
        elif func == "Tangente Hiperbólica":
            y = np.tanh(freq * t)
        elif func == "Passo (step)":
            y = np.heaviside(t - 0.5, 1)
        elif func == "Rampa":
            y = t
        elif func == "Parábola":
            y = t**2
        elif func == "Impulso Infinito":
            y = unit_impulse(N)
        else:
            y = np.zeros_like(t)

        # AM e FM
        if self.mod_am.get():
            depth = self.am_depth.get()
            y *= (1 + depth * np.sin(2 * np.pi * freq/4 * t))
        if self.mod_fm.get():
            dev = self.fm_deviation.get()
            y = np.sin(2 * np.pi * freq * t + dev * np.sin(2 * np.pi * freq/4 * t))

        # FFT
        Y = fftshift(fft(y))
        f = fftshift(fftfreq(N, 1 / Fs))

        # Guarda
        self.last_t, self.last_y, self.last_f, self.last_Y = t, y, f, np.abs(Y)

        # Plots
        self.ax1.clear()
        self.ax1.plot(t, y, color='cyan')
        self.ax1.set_title('Domínio do Tempo')
        self.ax1.grid(True)

        self.ax2.clear()
        self.ax2.plot(f, np.abs(Y), color='orange')
        self.ax2.set_title('Domínio da Frequência')
        self.ax2.grid(True)

        self.fig.tight_layout()
        self.canvas.draw()
        self.reset_zoom()

    def reset_zoom(self):
        if self.last_t is not None:
            self.ax1.set_xlim(self.last_t[0], self.last_t[-1])
        if self.last_f is not None:
            self.ax2.set_xlim(self.last_f[0], self.last_f[-1])
        self.zoom_time.set(1.0)
        self.zoom_freq.set(1.0)
        self.canvas.draw()

    def update_time_zoom(self, val):
        frac = float(val)
        t = self.last_t
        N = len(t)
        if frac >= 0.999:
            lo, hi = 0, N
        else:
            show = int(N * frac)
            mid = N // 2
            lo = max(0, mid - show // 2)
            hi = min(N, mid + show // 2)
        self.ax1.set_xlim(t[lo], t[hi-1])
        self.canvas.draw()

    def update_freq_zoom(self, val):
        frac = float(val)
        f = self.last_f
        N = len(f)
        if frac >= 0.999:
            lo, hi = 0, N
        else:
            show = int(N * frac)
            mid = N // 2
            lo = max(0, mid - show // 2)
            hi = min(N, mid + show // 2)
        self.ax2.set_xlim(f[lo], f[hi-1])
        self.canvas.draw()

    def export_data(self):
        if self.last_t is None:
            messagebox.showerror("Erro", "Gere um sinal primeiro")
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON','*.json'),('CSV','*.csv')]
        )
        if not path:
            return
        if path.endswith('.json'):
            with open(path,'w') as f:
                json.dump({
                    'tempo':self.last_t.tolist(),
                    'sinal':self.last_y.tolist(),
                    'freq':self.last_f.tolist(),
                    'fft':self.last_Y.tolist()
                },f,indent=2)
        else:
            with open(path,'w',newline='') as f:
                w = csv.writer(f)
                w.writerow(['t','y','f','|Y|'])
                for ti,yi,fi,Ymi in zip(self.last_t,self.last_y,self.last_f,self.last_Y):
                    w.writerow([ti,yi,fi,Ymi])

    def _on_right_click(self, event):
        if event.button == 3:
            self.menu.post(event.x_root, event.y_root)

    def add_time_marker(self):
        if self.last_t is None:
            return
        x = self.last_t[len(self.last_t)//2]
        self.ax1.axvline(x, color='red', linestyle='--')
        self.canvas.draw()

    def add_freq_marker(self):
        if self.last_f is None:
            return
        x = self.last_f[len(self.last_f)//2]
        self.ax2.axvline(x, color='green', linestyle='--')
        self.canvas.draw()

if __name__ == '__main__':
    app = SignalGeneratorApp()
    app.mainloop()
