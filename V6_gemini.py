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
ctk.set_default_color_theme("blue")


class SignalGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gerador e Analisador de Sinais Definitivo")
        self.geometry("1400x950")

        # Executor para rodar cálculos em outra thread e não travar a UI
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Armazenam o último sinal gerado e marcadores
        self.last_data = {}
        self.markers = {'time': [], 'freq': []}
        self.dragging_marker = None
        self.last_click_event = None

        # Flags de modulação
        self.mod_am = tk.BooleanVar(value=False)
        self.mod_fm = tk.BooleanVar(value=False)

        self._build_sidebar()
        self._build_plot_area()
        self._build_context_menu()
        self._build_status_bar()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=8)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        def section(title, color):
            frm = ctk.CTkFrame(self.sidebar, fg_color=color, corner_radius=6)
            frm.pack(fill="x", pady=5, padx=5)
            ctk.CTkLabel(frm, text=title, font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(5, 2))
            return frm

        # --- Geração de Sinal ---
        frm_gen = section("Geração de Sinal", "#444444")
        self.entry_points = self._add_entry(frm_gen, "Pontos (≥128 e par):", "1024")
        self.entry_fc = self._add_entry(frm_gen, "Frequência da Portadora (Fc):", "100")
        self.entry_fs = self._add_entry(frm_gen, "Taxa de Amostragem (Fs):", "2000")
        self.waveform = self._add_option_menu(frm_gen, "Forma de Onda:",
                                              ["Seno", "Cosseno", "Quadrada", "Triangular", "Dente de Serra", "Pulso",
                                               "Ruído Branco", "Exp Decaimento", "Passo (step)", "Rampa", "Parábola",
                                               "Impulso", "Tangente"])

        # --- Modulação ---
        frm_mod = section("Modulação", "#2d3e50")
        self.entry_fm = self._add_entry(frm_mod, "Frequência da Moduladora (Fm):", "10")
        # AM
        ctk.CTkCheckBox(frm_mod, text="Ativar AM", variable=self.mod_am, command=self._on_am_fm_toggle).pack(anchor="w",
                                                                                                             padx=10,
                                                                                                             pady=5)
        self.slider_am = ctk.CTkSlider(frm_mod, from_=0, to=2, number_of_steps=200)
        self.slider_am.set(0.5)
        self.slider_am.configure(state="disabled")
        self.slider_am.pack(fill="x", padx=10, pady=(0, 10))
        # FM
        ctk.CTkCheckBox(frm_mod, text="Ativar FM", variable=self.mod_fm, command=self._on_am_fm_toggle).pack(anchor="w",
                                                                                                             padx=10,
                                                                                                             pady=5)
        self.slider_fm_dev = ctk.CTkSlider(frm_mod, from_=0, to=1, number_of_steps=100)
        self.slider_fm_dev.set(50)
        self.slider_fm_dev.configure(state="disabled")
        self.slider_fm_dev.pack(fill="x", padx=10, pady=(0, 10))

        # --- Comandos ---
        frm_cmd = section("Comandos", "#333333")
        self.btn_generate = ctk.CTkButton(frm_cmd, text="Gerar Sinal", command=self.submit_plot_task)
        self.btn_generate.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(frm_cmd, text="Exportar Dados", command=self.export_data).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(frm_cmd, text="Resetar Zoom", command=self.reset_zoom).pack(fill="x", padx=10, pady=5)

    def _add_entry(self, parent, label, default):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(5, 0))
        entry = ctk.CTkEntry(parent)
        entry.insert(0, default)
        entry.pack(fill="x", padx=10, pady=(0, 5))
        return entry

    def _add_option_menu(self, parent, label, values):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(5, 0))
        menu = ctk.CTkOptionMenu(parent, values=values)
        menu.set(values[0])
        menu.pack(fill="x", padx=10, pady=5)
        return menu

    def _build_plot_area(self):
        frm = ctk.CTkFrame(self)
        frm.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.fig, (self.ax_time, self.ax_freq) = plt.subplots(2, 1, facecolor="#2B2B2B")
        self.ax_time.set_facecolor("#3C3C3C")
        self.ax_freq.set_facecolor("#3C3C3C")
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frm)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        ctrl = ctk.CTkFrame(frm)
        ctrl.pack(fill="x", pady=5)
        ctk.CTkLabel(ctrl, text="Zoom Tempo:").pack(side="left", padx=5)
        self.zoom_time = ctk.CTkSlider(ctrl, from_=0.01, to=1.0, command=self.update_time_zoom)
        self.zoom_time.set(1.0)
        self.zoom_time.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkLabel(ctrl, text="Zoom Freq:").pack(side="left", padx=5)
        self.zoom_freq = ctk.CTkSlider(ctrl, from_=0.01, to=1.0, command=self.update_freq_zoom)
        self.zoom_freq.set(1.0)
        self.zoom_freq.pack(side="left", fill="x", expand=True, padx=5)

        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)

    def _build_context_menu(self):
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Adicionar Marcador de Tempo", command=lambda: self.add_marker('time'))
        self.menu.add_command(label="Adicionar Marcador de Frequência", command=lambda: self.add_marker('freq'))
        self.menu.add_separator()
        self.menu.add_command(label="Limpar Marcadores de Tempo", command=lambda: self.clear_markers('time'))
        self.menu.add_command(label="Limpar Marcadores de Frequência", command=lambda: self.clear_markers('freq'))

    def _build_status_bar(self):
        self.status_bar = ctk.CTkLabel(self, text="Pronto", anchor="w")
        self.status_bar.pack(side="bottom", fill="x", padx=10, pady=5)

    def _on_am_fm_toggle(self):
        self.slider_am.configure(state="normal" if self.mod_am.get() else "disabled")
        self.slider_fm_dev.configure(state="normal" if self.mod_fm.get() else "disabled")

    def submit_plot_task(self):
        self.btn_generate.configure(state="disabled", text="Gerando...")
        self.status_bar.configure(text="Iniciando geração de sinal...")
        self.executor.submit(self._compute_and_plot_task)

    def _compute_and_plot_task(self):
        try:
            # 1. Leitura e Validação de Parâmetros
            params = self._validate_inputs()
            if not params: return

            # 2. Geração da Onda Principal
            t = np.arange(params['N']) / params['Fs']
            y = self._generate_waveform(params, t)

            # 3. Aplicação de Modulação (se houver)
            y = self._apply_modulation(params, y, t)

            # 4. Cálculo da FFT
            Y = fftshift(fft(y))
            f = fftshift(fftfreq(params['N'], 1 / params['Fs']))

            # 5. Armazenar dados para uso posterior
            self.last_data = {'t': t, 'y': y, 'f': f, 'Y': np.abs(Y)}

            # 6. Agendar a atualização do plot na thread principal da UI
            self.after(0, self._update_plots)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Erro de Cálculo", str(e)))
        finally:
            # Reativar o botão na thread principal
            self.after(0, lambda: self.btn_generate.configure(state="normal", text="Gerar Sinal"))
            self.after(0, lambda: self.status_bar.configure(text="Pronto"))

    def _validate_inputs(self):
        try:
            params = {
                'N': int(self.entry_points.get()),
                'Fc': float(self.entry_fc.get()),
                'Fs': float(self.entry_fs.get()),
                'Fm': float(self.entry_fm.get()),
                'waveform': self.waveform.get(),
                'am_on': self.mod_am.get(), 'am_depth': self.slider_am.get(),
                'fm_on': self.mod_fm.get(), 'fm_dev': self.slider_fm_dev.get()
            }
            if params['N'] < 128 or params['N'] % 2 != 0: raise ValueError("Pontos deve ser ≥128 e par.")
            if params['Fc'] <= 0 or params['Fs'] <= 0 or params['Fm'] <= 0: raise ValueError(
                "Frequências e Fs devem ser > 0.")
            if params['Fc'] >= params['Fs'] / 2: raise ValueError(
                f"Critério de Nyquist violado! Fc ({params['Fc']}Hz) deve ser < Fs/2 ({params['Fs'] / 2}Hz).")

            # Atualizar slider de desvio FM com base na Fs atual
            self.after(0, lambda: self.slider_fm_dev.configure(to=params['Fs'] / 2 - params['Fc']))

            if params['fm_on'] and (params['Fc'] + params['fm_dev']) >= params['Fs'] / 2:
                raise ValueError(
                    f"Nyquist violado em FM! Fc+Δf ({params['Fc'] + params['fm_dev']}Hz) deve ser < Fs/2 ({params['Fs'] / 2}Hz).")
            return params
        except ValueError as e:
            self.after(0, lambda: messagebox.showerror("Erro de Entrada", str(e)))
            return None

    def _generate_waveform(self, p, t):
        wf, Fc = p['waveform'], p['Fc']
        if wf == "Seno":
            return np.sin(2 * np.pi * Fc * t)
        elif wf == "Cosseno":
            return np.cos(2 * np.pi * Fc * t)
        elif wf == "Quadrada":
            return square(2 * np.pi * Fc * t)
        elif wf == "Triangular":
            return sawtooth(2 * np.pi * Fc * t, 0.5)
        elif wf == "Dente de Serra":
            return sawtooth(2 * np.pi * Fc * t)
        elif wf == "Pulso":
            return square(2 * np.pi * Fc * t, duty=0.2)
        elif wf == "Ruído Branco":
            return np.random.normal(0, 0.5, p['N'])
        elif wf == "Exp Decaimento":
            return np.exp(-t * 5) * np.sin(2 * np.pi * Fc * t)
        elif wf == "Passo (step)":
            return np.heaviside(t - t[p['N'] // 4], 1.0)
        elif wf == "Rampa":
            return t / t[-1]
        elif wf == "Parábola":
            return (t / t[-1]) ** 2
        elif wf == "Impulso":
            return unit_impulse(p['N'], 'mid')
        elif wf == "Tangente":
            y = np.tan(np.pi * Fc * t); return np.clip(y, -5, 5)
        else:
            return np.zeros_like(t)

    def _apply_modulation(self, p, y, t):
        if p['am_on']:
            modulator = np.sin(2 * np.pi * p['Fm'] * t)
            y *= (1 + p['am_depth'] * modulator)
        elif p['fm_on']:
            # A onda 'y' original agora é o sinal modulador m(t)
            modulating_signal = y
            # Integração da fase: phi(t) = 2*pi*Fc*t + 2*pi*delta_f * integral(m(t) dt)
            integrated_modulator = np.cumsum(modulating_signal) * (1 / p['Fs'])
            y = np.sin(2 * np.pi * p['Fc'] * t + 2 * np.pi * p['fm_dev'] * integrated_modulator)
        return y

    def _update_plots(self):
        self.ax_time.clear()
        self.ax_time.plot(self.last_data['t'], self.last_data['y'], color="cyan")
        self.ax_time.set_title("Domínio do Tempo", color='white')
        self.ax_time.grid(True, linestyle='--', alpha=0.5)
        self.ax_time.tick_params(colors='white')

        self.ax_freq.clear()
        self.ax_freq.plot(self.last_data['f'], self.last_data['Y'], color="orange")
        self.ax_freq.set_title("Domínio da Frequência (FFT)", color='white')
        self.ax_freq.set_ylabel("|Y(f)|", color='white')
        self.ax_freq.grid(True, linestyle='--', alpha=0.5)
        self.ax_freq.tick_params(colors='white')

        self.fig.tight_layout(pad=3.0)
        self.clear_markers('all')
        self.reset_zoom()
        self.canvas.draw()
        self.status_bar.configure(text="Gráficos atualizados com sucesso!")

    def reset_zoom(self):
        if not self.last_data: return
        self.ax_time.set_xlim(self.last_data['t'][0], self.last_data['t'][-1])
        self.ax_freq.set_xlim(self.last_data['f'][0], self.last_data['f'][-1])
        self.zoom_time.set(1.0)
        self.zoom_freq.set(1.0)
        self.canvas.draw()

    def update_time_zoom(self, val):
        if not self.last_data: return
        center = self.ax_time.get_xlim()[0] + (self.ax_time.get_xlim()[1] - self.ax_time.get_xlim()[0]) / 2
        total_width = self.last_data['t'][-1] - self.last_data['t'][0]
        new_width = total_width * float(val)
        self.ax_time.set_xlim(center - new_width / 2, center + new_width / 2)
        self.canvas.draw()

    def update_freq_zoom(self, val):
        if not self.last_data: return
        center = self.ax_freq.get_xlim()[0] + (self.ax_freq.get_xlim()[1] - self.ax_freq.get_xlim()[0]) / 2
        total_width = self.last_data['f'][-1] - self.last_data['f'][0]
        new_width = total_width * float(val)
        self.ax_freq.set_xlim(center - new_width / 2, center + new_width / 2)
        self.canvas.draw()

    def export_data(self):
        if not self.last_data:
            messagebox.showerror("Erro", "Gere um sinal primeiro.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json"), ("CSV", "*.csv")])
        if not path: return
        try:
            if path.endswith(".json"):
                with open(path, "w") as f:
                    data_to_save = {k: v.tolist() for k, v in self.last_data.items()}
                    json.dump(data_to_save, f, indent=2)
            else:
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.last_data.keys())
                    writer.writerows(zip(*self.last_data.values()))
            self.status_bar.configure(text=f"Dados exportados para {path}")
        except Exception as e:
            messagebox.showerror("Erro de Exportação", str(e))

    def _on_mouse_press(self, event):
        if event.button == 3:  # Botão direito
            self.last_click_event = event
            self.menu.post(event.x_root, event.y_root)
            return

        if event.inaxes:
            ax = event.inaxes
            marker_list = self.markers['time'] if ax == self.ax_time else self.markers['freq']
            for marker in marker_list:
                contains, _ = marker.contains(event)
                if contains:
                    self.dragging_marker = marker
                    break

    def _on_mouse_release(self, event):
        self.dragging_marker = None

    def _on_mouse_motion(self, event):
        if self.dragging_marker and event.inaxes and event.xdata:
            self.dragging_marker.set_xdata([event.xdata])
            self.canvas.draw_idle()

    def add_marker(self, plot_type):
        if not self.last_click_event or not self.last_click_event.inaxes: return

        ax = self.ax_time if plot_type == 'time' else self.ax_freq
        if self.last_click_event.inaxes != ax: return

        x_pos = self.last_click_event.xdata
        color = 'red' if plot_type == 'time' else 'lime'
        line = ax.axvline(x=x_pos, color=color, linestyle='--', picker=5)
        self.markers[plot_type].append(line)
        self.canvas.draw()

    def clear_markers(self, plot_type):
        if plot_type == 'all':
            self.clear_markers('time')
            self.clear_markers('freq')
            return

        for marker in self.markers[plot_type]:
            marker.remove()
        self.markers[plot_type].clear()
        self.canvas.draw()


if __name__ == "__main__":
    app = SignalGeneratorApp()
    app.mainloop()