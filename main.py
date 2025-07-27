import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import square, sawtooth, unit_impulse
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import CubicSpline
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
        self.label_points.pack()
        self.entry_points = ctk.CTkEntry(self.sidebar)
        self.entry_points.insert(0, "256")
        self.entry_points.pack()

        self.label_freq = ctk.CTkLabel(self.sidebar, text="Frequência:")
        self.label_freq.pack()
        self.entry_freq = ctk.CTkEntry(self.sidebar)
        self.entry_freq.insert(0, "5")
        self.entry_freq.pack()

        self.label_fs = ctk.CTkLabel(self.sidebar, text="Taxa de Amostragem (Hz):")
        self.label_fs.pack()
        self.entry_fs = ctk.CTkEntry(self.sidebar)
        self.entry_fs.insert(0, "1000")
        self.entry_fs.pack()

        self.label_func = ctk.CTkLabel(self.sidebar, text="Função:")
        self.label_func.pack()
        self.combo_func = ctk.CTkComboBox(self.sidebar, values=self.func_types)
        self.combo_func.set("Seno")
        self.combo_func.pack()

        self.am_checkbox = ctk.CTkCheckBox(self.sidebar, text="Modulação AM", command=self.toggle_am)
        self.am_checkbox.pack(pady=2)
        self.am_depth = ctk.CTkSlider(self.sidebar, from_=0, to=1, number_of_steps=100, command=self.plot_signal)
        self.am_depth.set(0.5)
        self.am_depth.pack()
        self.am_depth_label = ctk.CTkLabel(self.sidebar, text="Profundidade AM")
        self.am_depth_label.pack()

        self.fm_checkbox = ctk.CTkCheckBox(self.sidebar, text="Modulação FM", command=self.toggle_fm)
        self.fm_checkbox.pack(pady=2)
        self.fm_deviation = ctk.CTkSlider(self.sidebar, from_=0, to=Fs/2, number_of_steps=100, command=self.plot_signal)
        self.fm_deviation.set(50)
        self.fm_deviation.pack()
        self.fm_deviation_label = ctk.CTkLabel(self.sidebar, text="Desvio FM")
        self.fm_deviation_label.pack()

        self.button_plot = ctk.CTkButton(self.sidebar, text="Gerar Sinal", command=self.plot_signal)
        self.button_plot.pack(pady=10)

        self.button_export = ctk.CTkButton(self.sidebar, text="Exportar CSV/JSON", command=self.export_data)
        self.button_export.pack(pady=10)

        self.button_reset = ctk.CTkButton(self.sidebar, text="Resetar Zoom", command=self.reset_zoom)
        self.button_reset.pack(pady=10)

        self.plot_area = ctk.CTkFrame(self)
        self.plot_area.pack(side="right", fill="both", expand=True)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.zoom_var = tk.DoubleVar(value=1.0)
        self.slider_zoom = ctk.CTkSlider(self.plot_area, from_=0.01, to=1.0, number_of_steps=100,
                                         variable=self.zoom_var, command=self.update_zoom)
        self.slider_zoom.pack(pady=5)

        self.fft_zoom_var = tk.DoubleVar(value=1.0)
        self.slider_fft_zoom = ctk.CTkSlider(self.plot_area, from_=0.01, to=1.0, number_of_steps=100,
                                             variable=self.fft_zoom_var, command=self.update_fft_zoom)
        self.slider_fft_zoom.pack(pady=5)

        self.dragging = None
        self.marks = {"tempo": [], "frequencia": []}

        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Adicionar Marca no Tempo", command=self.add_time_marker)
        self.menu.add_command(label="Adicionar Marca na Frequência", command=self.add_freq_marker)

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('button_press_event', self.on_right_click)

        self.enable_am = False
        self.enable_fm = False

    def toggle_am(self):
        self.enable_am = self.am_checkbox.get()
        self.plot_signal()

    def toggle_fm(self):
        self.enable_fm = self.fm_checkbox.get()
        self.plot_signal()

    def reset_zoom(self):
        self.zoom_var.set(1.0)
        self.fft_zoom_var.set(1.0)
        self.update_zoom(1.0)
        self.update_fft_zoom(1.0)

    def export_data(self):
        if hasattr(self, 'last_time') and hasattr(self, 'last_signal'):
            with open("dados_sinal.csv", "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Tempo", "Sinal"])
                for t, y in zip(self.last_time, self.last_signal):
                    writer.writerow([t, y])

            with open("dados_sinal.json", "w") as jsonfile:
                json.dump({"tempo": self.last_time.tolist(), "sinal": self.last_signal.tolist()}, jsonfile)
            messagebox.showinfo("Exportado", "Dados exportados com sucesso!")

    def update_zoom(self, val):
        if hasattr(self, 'last_time'):
            zoom = float(val)
            total = len(self.last_time)
            show = int(total * zoom)
            self.ax1.set_xlim(self.last_time[0], self.last_time[show - 1])
            self.canvas.draw()

    def update_fft_zoom(self, val):
        if hasattr(self, 'last_freq'):
            zoom = float(val)
            total = len(self.last_freq)
            show = int(total * zoom)
            self.ax2.set_xlim(self.last_freq[total//2 - show//2], self.last_freq[total//2 + show//2])
            self.canvas.draw()

    def add_time_marker(self):
        if hasattr(self, 'last_time'):
            x = self.last_time[len(self.last_time)//2]
            line = self.ax1.axvline(x=x, color='red')
            self.marks["tempo"].append(line)
            self.canvas.draw()

    def add_freq_marker(self):
        if hasattr(self, 'last_freq'):
            x = self.last_freq[len(self.last_freq)//2]
            line = self.ax2.axvline(x=x, color='red')
            self.marks["frequencia"].append(line)
            self.canvas.draw()

    def on_press(self, event):
        for mark in self.marks["tempo"] + self.marks["frequencia"]:
            contains, _ = mark.contains(event)
            if contains:
                self.dragging = mark
                break

    def on_motion(self, event):
        if self.dragging and event.xdata:
            self.dragging.set_xdata([event.xdata])
            self.canvas.draw()

    def on_release(self, event):
        self.dragging = None

    def on_right_click(self, event):
        if event.button == 3:
            self.menu.tk_popup(int(event.x_root), int(event.y_root))

    def plot_signal(self, *_):
        self.executor.submit(self._compute_and_plot)

    def _compute_and_plot(self):
        try:
            N = int(self.entry_points.get())
            freq = float(self.entry_freq.get())
            Fs = float(self.entry_fs.get())
        except ValueError:
            messagebox.showerror("Erro", "Insira valores numéricos válidos para Pontos, Frequência e Fs.")
            return

        if freq > Fs / 2:
            messagebox.showerror("Erro de Nyquist", f"Para {N} pontos e Fs={Fs}Hz, a frequência máxima permitida é {Fs/2}Hz.")
            return

        t = np.arange(N) / Fs
        func = self.combo_func.get()

        try:
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
                y = np.random.normal(0, 1, size=N)
            elif func == "Seno Modulado":
                y = np.sin(2 * np.pi * freq * t) * np.sin(2 * np.pi * freq/10 * t)
            elif func == "Exp Decaimento":
                y = np.exp(-5 * t) * np.sin(2 * np.pi * freq * t)
            elif func == "Seno Hiperbólico":
                y = np.sinh(t * freq)
            elif func == "Tangente":
                y = np.tan(2 * np.pi * freq * t)
            elif func == "Tangente Hiperbólica":
                y = np.tanh(t * freq)
            elif func == "Passo (step)":
                y = np.heaviside(t - 0.5, 1)
            elif func == "Rampa":
                y = t
            elif func == "Parábola":
                y = t ** 2
            elif func == "Impulso Infinito":
                y = unit_impulse(N)
            else:
                y = np.zeros_like(t)

            if self.enable_am:
                depth = self.am_depth.get()
                y *= 1 + depth * np.sin(2 * np.pi * freq/4 * t)

            if self.enable_fm:
                dev = self.fm_deviation.get()
                y = np.sin(2 * np.pi * freq * t + dev * np.sin(2 * np.pi * freq/4 * t))

        except Exception as e:
            messagebox.showerror("Erro ao gerar sinal", str(e))
            return

        try:
            yf = fftshift(fft(y))
            xf = fftshift(fftfreq(N, 1 / Fs))
        except Exception as e:
            messagebox.showerror("Erro na FFT", str(e))
            return

        self.last_time = t
        self.last_signal = y
        self.last_freq = xf
        self.last_fft = np.abs(yf)

        self.ax1.clear()
        self.ax1.plot(t, y, label="Sinal")
        self.ax1.set_title("Domínio do Tempo")
        self.ax1.grid(True)

        self.ax2.clear()
        self.ax2.plot(xf, np.abs(yf), label="FFT", color='orange')
        self.ax2.set_title("Domínio da Frequência (com negativas)")
        self.ax2.grid(True)

        self.fig.tight_layout()
        self.canvas.draw()

        self.slider_zoom.configure(from_=0.01, to=1.0)
        self.slider_zoom.set(1.0)
        self.slider_fft_zoom.configure(from_=0.01, to=1.0)
        self.slider_fft_zoom.set(1.0)

if __name__ == "__main__":
    app = SignalGeneratorApp()
    app.mainloop()
