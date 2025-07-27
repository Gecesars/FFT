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

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SignalGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gerador e Analisador de Sinais")
        self.geometry("1400x900")
        self.resizable(False, False)

        # Armazenam o último sinal gerado
        self.last_t = None
        self.last_y = None
        self.last_f = None
        self.last_Y = None

        # Flags de modulação
        self.mod_am = tk.BooleanVar(value=False)
        self.mod_fm = tk.BooleanVar(value=False)

        self._build_sidebar()
        self._build_plot_area()
        self._build_context_menu()

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=300, corner_radius=8)
        sidebar.pack(side="left", fill="y", padx=10, pady=10)

        def section(title, color):
            frm = ctk.CTkFrame(sidebar, fg_color=color, corner_radius=6)
            frm.pack(fill="x", pady=5)
            ctk.CTkLabel(
                frm, text=title, font=("Arial", 12, "bold")
            ).pack(anchor="w", padx=5, pady=(5, 2))
            return frm

        # --- Geração de Sinal ---
        frm_gen = section("Geração de Sinal", "#444444")
        self.entry_points = self._add_entry(
            frm_gen, "Pontos (≥128 e par):", "1024"
        )
        self.entry_fc = self._add_entry(frm_gen, "Fc (Hz):", "100")
        self.entry_fs = self._add_entry(frm_gen, "Fs (Hz):", "1000")
        ctk.CTkLabel(frm_gen, text="Forma de Onda:").pack(
            anchor="w", padx=5, pady=(5, 0)
        )
        self.waveform = ctk.CTkOptionMenu(
            frm_gen,
            values=[
                "Seno",
                "Cosseno",
                "Quadrada",
                "Triangular",
                "Dente de Serra",
                "Pulso",
                "Ruído Branco",
                "Exp Decaimento",
                "Seno Hiperbólico",
                "Tangente",
                "Tangente Hiperbólica",
                "Passo (step)",
                "Rampa",
                "Parábola",
                "Impulso Infinito",
            ],
        )
        self.waveform.set("Seno")
        self.waveform.pack(fill="x", padx=10, pady=5)

        # --- Modulação AM ---
        frm_am = section("Modulação AM", "#2d3e50")
        ctk.CTkCheckBox(
            frm_am,
            text="Ativar AM",
            variable=self.mod_am,
            command=self._on_am_fm_toggle,
        ).pack(anchor="w", padx=5, pady=(0, 5))
        self.slider_am = ctk.CTkSlider(frm_am, from_=0, to=1, number_of_steps=100)
        self.slider_am.set(0.5)
        self.slider_am.configure(state="disabled")
        self.slider_am.pack(fill="x", padx=10, pady=(0, 10))

        # --- Modulação FM ---
        frm_fm = section("Modulação FM", "#2d5032")
        ctk.CTkCheckBox(
            frm_fm,
            text="Ativar FM",
            variable=self.mod_fm,
            command=self._on_am_fm_toggle,
        ).pack(anchor="w", padx=5, pady=(0, 5))
        # Inicialmente de 0 a 1: só no plot_signal ajustamos para Fs/2
        self.slider_fm = ctk.CTkSlider(frm_fm, from_=0, to=1, number_of_steps=100)
        self.slider_fm.set(50)
        self.slider_fm.configure(state="disabled")
        self.slider_fm.pack(fill="x", padx=10, pady=(0, 10))

        # --- Modulação Digital (em breve) ---
        frm_dig = section("Modulação Digital (em breve)", "#544b3d")
        ctk.CTkLabel(frm_dig, text="ASK, FSK, PSK, QAM").pack(
            anchor="w", padx=5, pady=5
        )

        # --- Comandos ---
        frm_cmd = section("Comandos", "#333333")
        ctk.CTkButton(
            frm_cmd, text="Gerar Sinal", command=self.plot_signal
        ).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(
            frm_cmd, text="Exportar CSV/JSON", command=self.export_data
        ).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(
            frm_cmd, text="Resetar Zoom", command=self.reset_zoom
        ).pack(fill="x", padx=10, pady=5)

    def _add_entry(self, parent, label, default):
        ctk.CTkLabel(parent, text=label).pack(
            anchor="w", padx=5, pady=(5, 0)
        )
        entry = ctk.CTkEntry(parent)
        entry.insert(0, default)
        entry.pack(fill="x", padx=10, pady=(0, 5))
        return entry

    def _build_plot_area(self):
        frm = ctk.CTkFrame(self)
        frm.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig, (self.ax_time, self.ax_freq) = plt.subplots(2, 1, figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frm)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        ctrl = ctk.CTkFrame(frm)
        ctrl.pack(fill="x", pady=5)
        ctk.CTkLabel(ctrl, text="Zoom Tempo:").pack(side="left", padx=5)
        self.zoom_time = ctk.CTkSlider(
            ctrl,
            from_=0.05,
            to=1.0,
            number_of_steps=95,
            command=self.update_time_zoom,
        )
        self.zoom_time.set(1.0)
        self.zoom_time.pack(side="left", fill="x", expand=True, padx=5)

        ctk.CTkLabel(ctrl, text="Zoom Freq:").pack(side="left", padx=5)
        self.zoom_freq = ctk.CTkSlider(
            ctrl,
            from_=0.05,
            to=1.0,
            number_of_steps=95,
            command=self.update_freq_zoom,
        )
        self.zoom_freq.set(1.0)
        self.zoom_freq.pack(side="left", fill="x", expand=True, padx=5)

        self.canvas.mpl_connect("button_press_event", self._on_right_click)

    def _build_context_menu(self):
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Marcar Tempo", command=self.add_time_marker)
        self.menu.add_command(
            label="Marcar Frequência", command=self.add_freq_marker
        )

    def _on_am_fm_toggle(self):
        # apenas um tipo de modulação por vez
        if self.mod_am.get():
            self.mod_fm.set(False)
            self.slider_fm.configure(state="disabled")
            self.slider_am.configure(state="normal")
        elif self.mod_fm.get():
            self.mod_am.set(False)
            self.slider_am.configure(state="disabled")
            self.slider_fm.configure(state="normal")
        else:
            self.slider_am.configure(state="disabled")
            self.slider_fm.configure(state="disabled")

    def plot_signal(self):
        # Leitura e validações
        try:
            N = int(self.entry_points.get())
            Fc = float(self.entry_fc.get())
            Fs = float(self.entry_fs.get())
        except ValueError:
            messagebox.showerror("Erro", "Pontos, Fc e Fs devem ser numéricos")
            return

        if N < 128 or N % 2 != 0:
            messagebox.showerror("Erro", "Pontos ≥128 e múltiplo de 2")
            return
        if Fc >= Fs / 2:
            messagebox.showerror("Nyquist", "Fc ≥ Fs/2")
            return
        if self.mod_fm.get() and (Fc + self.slider_fm.get()) >= Fs / 2:
            messagebox.showerror("Nyquist", "Fc+Δf ≥ Fs/2")
            return

        # **Reconfigura o slider FM** agora que Fs está conhecido
        self.slider_fm.configure(to=Fs / 2)

        t = np.arange(N) / Fs
        wf = self.waveform.get()

        # --- Formas de Onda ---
        if wf == "Seno":
            y = np.sin(2 * np.pi * Fc * t)
        elif wf == "Cosseno":
            y = np.cos(2 * np.pi * Fc * t)
        elif wf == "Quadrada":
            y = square(2 * np.pi * Fc * t)  # ±1 perfeitos
        elif wf == "Triangular":
            y = sawtooth(2 * np.pi * Fc * t, 0.5)
        elif wf == "Dente de Serra":
            y = sawtooth(2 * np.pi * Fc * t, 1.0)
        elif wf == "Pulso":
            y = square(2 * np.pi * Fc * t, duty=0.1)
        elif wf == "Ruído Branco":
            y = np.random.normal(0, 1, N)
        elif wf == "Exp Decaimento":
            y = np.exp(-t) * np.sin(2 * np.pi * Fc * t)
        elif wf == "Seno Hiperbólico":
            y = np.sinh(np.sin(2 * np.pi * Fc * t))
        elif wf == "Tangente":
            y = np.tan(2 * np.pi * Fc * t)
        elif wf == "Tangente Hiperbólica":
            y = np.tanh(np.sin(2 * np.pi * Fc * t))
        elif wf == "Passo (step)":
            y = np.heaviside(t - t[N // 2], 1.0)
        elif wf == "Rampa":
            y = t
        elif wf == "Parábola":
            y = t**2
        elif wf == "Impulso Infinito":
            y = unit_impulse(N)
        else:
            y = np.zeros_like(t)

        # --- Modulações ---
        if self.mod_am.get():
            m = self.slider_am.get()
            y *= (1 + m * np.sin(2 * np.pi * (Fc / 4) * t))
        if self.mod_fm.get():
            d = self.slider_fm.get()
            y = np.sin(
                2 * np.pi * Fc * t + d * np.sin(2 * np.pi * (Fc / 4) * t)
            )

        # FFT
        Y = fftshift(fft(y))
        f = fftshift(fftfreq(N, 1 / Fs))

        # Guarda para zoom/export
        self.last_t, self.last_y, self.last_f, self.last_Y = t, y, f, np.abs(Y)

        # Plota
        self.ax_time.clear()
        self.ax_time.plot(t, y, color="cyan")
        self.ax_time.set_title("Domínio do Tempo")
        self.ax_time.grid(True)

        self.ax_freq.clear()
        self.ax_freq.plot(f, np.abs(Y), color="orange")
        self.ax_freq.set_title("Domínio da Frequência")
        self.ax_freq.grid(True)

        self.canvas.draw()
        self.reset_zoom()

    def reset_zoom(self):
        if self.last_t is not None:
            self.ax_time.set_xlim(self.last_t[0], self.last_t[-1])
        if self.last_f is not None:
            self.ax_freq.set_xlim(self.last_f[0], self.last_f[-1])
        self.zoom_time.set(1.0)
        self.zoom_freq.set(1.0)
        self.canvas.draw()

    def update_time_zoom(self, frac):
        if not self.last_t:
            return
        if frac >= 0.999:
            lo, hi = 0, len(self.last_t)
        else:
            N = len(self.last_t)
            show = int(N * frac)
            mid = N // 2
            lo = max(0, mid - show // 2)
            hi = min(N, mid + show // 2)
        self.ax_time.set_xlim(self.last_t[lo], self.last_t[hi - 1])
        self.canvas.draw()

    def update_freq_zoom(self, frac):
        if not self.last_f:
            return
        if frac >= 0.999:
            lo, hi = 0, len(self.last_f)
        else:
            N = len(self.last_f)
            show = int(N * frac)
            mid = N // 2
            lo = max(0, mid - show // 2)
            hi = min(N, mid + show // 2)
        self.ax_freq.set_xlim(self.last_f[lo], self.last_f[hi - 1])
        self.canvas.draw()

    def export_data(self):
        if self.last_t is None:
            messagebox.showerror("Erro", "Gere um sinal primeiro")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv")],
        )
        if not path:
            return
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(
                    {
                        "tempo": self.last_t.tolist(),
                        "sinal": self.last_y.tolist(),
                        "freq": self.last_f.tolist(),
                        "fft": self.last_Y.tolist(),
                    },
                    f,
                    indent=2,
                )
        else:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "y", "f", "|Y|"])
                for ti, yi, fi, Ymi in zip(
                    self.last_t, self.last_y, self.last_f, self.last_Y
                ):
                    w.writerow([ti, yi, fi, Ymi])

    def _on_right_click(self, event):
        if event.button == 3:
            self.menu.post(event.x_root, event.y_root)

    def add_time_marker(self):
        if not self.last_t:
            return
        x = self.last_t[len(self.last_t) // 2]
        self.ax_time.axvline(x, color="red", linestyle="--")
        self.canvas.draw()

    def add_freq_marker(self):
        if not self.last_f:
            return
        x = self.last_f[len(self.last_f) // 2]
        self.ax_freq.axvline(x, color="green", linestyle="--")
        self.canvas.draw()


if __name__ == "__main__":
    app = SignalGeneratorApp()
    app.mainloop()
