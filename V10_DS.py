import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FuncFormatter
from scipy.signal import square, sawtooth, unit_impulse
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import CubicSpline
import csv
import json
from concurrent.futures import ThreadPoolExecutor

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Constantes para clareza
INTERP_THRESHOLD = 100  # Abaixo deste nº de pontos na tela, a interpolação é ativada
INTERP_SAMPLES = 500  # Nº de pontos a gerar para a curva suave interpolada
UNIT_MULTIPLIERS = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


class SignalGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gerador de Sinais Avançado (Zoom Adaptativo)")
        self.geometry("1400x950")

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.last_data = {}
        self.markers = {
            'time_v': [],  # verticais no plot de tempo
            'time_h': [],  # horizontais no plot de tempo
            'freq_v': [],  # verticais no plot de frequência
            'freq_h': []  # horizontais no plot de frequência
        }
        self.dragging_marker = None
        self.last_click_event = None
        self.time_plot_line = None

        self.mod_am = tk.BooleanVar(value=False)
        self.mod_fm = tk.BooleanVar(value=False)

        # Frame principal para organizar layout
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_sidebar()
        self._build_plot_area()
        self._build_context_menu()
        self._build_marker_panel()
        self._build_status_bar()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self.main_frame, width=350, corner_radius=8)
        self.sidebar.pack(side="left", fill="y", padx=(0, 10), pady=10)

        def section(title, color):
            frm = ctk.CTkFrame(self.sidebar, fg_color=color, corner_radius=6)
            frm.pack(fill="x", pady=5, padx=5)
            ctk.CTkLabel(frm, text=title, font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(5, 2))
            return frm

        # --- Geração de Sinal ---
        frm_gen = section("Geração de Sinal", "#444444")
        self.entry_duration = self._add_entry(frm_gen, "Duração (s):", "0.1")
        self.entry_fc, self.units_fc = self._add_frequency_entry(frm_gen, "Portadora (Fc):", "100")
        self.entry_fs, self.units_fs = self._add_frequency_entry(frm_gen, "Amostragem (Fs):", "20", "kHz")
        self.waveform = self._add_option_menu(frm_gen, "Forma de Onda:",
                                              ["Seno", "Cosseno", "Quadrada", "Triangular", "Dente de Serra", "Pulso",
                                               "Ruído Branco", "Exp Decaimento", "Passo (step)", "Rampa", "Parábola",
                                               "Impulso", "Tangente"])

        # --- Modulação ---
        frm_mod = section("Modulação", "#2d3e50")
        self.entry_fm, self.units_fm = self._add_frequency_entry(frm_mod, "Moduladora (Fm):", "10")
        ctk.CTkCheckBox(frm_mod, text="Ativar AM", variable=self.mod_am, command=self._on_am_fm_toggle).pack(anchor="w",
                                                                                                             padx=10,
                                                                                                             pady=5)
        self.slider_am = ctk.CTkSlider(frm_mod, from_=0, to=2, number_of_steps=200);
        self.slider_am.set(0.5)
        self.slider_am.configure(state="disabled");
        self.slider_am.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkCheckBox(frm_mod, text="Ativar FM", variable=self.mod_fm, command=self._on_am_fm_toggle).pack(anchor="w",
                                                                                                             padx=10,
                                                                                                             pady=5)
        self.slider_fm_dev = ctk.CTkSlider(frm_mod, from_=0, to=1, number_of_steps=100);
        self.slider_fm_dev.set(0.5)
        self.slider_fm_dev.configure(state="disabled");
        self.slider_fm_dev.pack(fill="x", padx=10, pady=(0, 10))

        # --- Comandos ---
        frm_cmd = section("Comandos", "#333333")
        self.btn_generate = ctk.CTkButton(frm_cmd, text="Gerar Sinal", command=self.submit_plot_task)
        self.btn_generate.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(frm_cmd, text="Exportar Dados", command=self.export_data).pack(fill="x", padx=10, pady=5)

    def _add_entry(self, parent, label, default):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(5, 0))
        entry = ctk.CTkEntry(parent);
        entry.insert(0, default)
        entry.pack(fill="x", padx=10, pady=(0, 5))
        return entry

    def _add_frequency_entry(self, parent, label, default_val, default_unit="Hz"):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(5, 0))
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=(0, 5))
        entry = ctk.CTkEntry(frame);
        entry.insert(0, default_val)
        entry.pack(side="left", fill="x", expand=True)
        units = ctk.CTkOptionMenu(frame, values=list(UNIT_MULTIPLIERS.keys()), width=75)
        units.set(default_unit)
        units.pack(side="left", padx=(5, 0))
        return entry, units

    def _add_option_menu(self, parent, label, values):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(5, 0))
        menu = ctk.CTkOptionMenu(parent, values=values);
        menu.set(values[0])
        menu.pack(fill="x", padx=10, pady=5)
        return menu

    def _build_plot_area(self):
        # Frame para área de plotagem e controles
        plot_frame = ctk.CTkFrame(self.main_frame)
        plot_frame.pack(side="left", fill="both", expand=True, padx=0, pady=0)

        # Frame para gráficos
        graph_frame = ctk.CTkFrame(plot_frame)
        graph_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # Frame para controles de zoom
        ctrl_frame = ctk.CTkFrame(plot_frame, height=40)
        ctrl_frame.pack(fill="x", pady=(0, 5))

        # Criação dos gráficos
        self.fig, (self.ax_time, self.ax_freq) = plt.subplots(2, 1, facecolor="#2B2B2B")
        self.ax_time.set_facecolor("#3C3C3C")
        self.ax_freq.set_facecolor("#3C3C3C")
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Controles de zoom
        ctk.CTkLabel(ctrl_frame, text="Zoom Tempo:").pack(side="left", padx=(10, 5))
        self.zoom_time = ctk.CTkSlider(ctrl_frame, from_=0.001, to=1.0, command=self.update_time_zoom)
        self.zoom_time.set(1.0)
        self.zoom_time.pack(side="left", fill="x", expand=True, padx=5)

        ctk.CTkLabel(ctrl_frame, text="Zoom Freq:").pack(side="left", padx=(10, 5))
        self.zoom_freq = ctk.CTkSlider(ctrl_frame, from_=0.001, to=1.0, command=self.update_freq_zoom)
        self.zoom_freq.set(1.0)
        self.zoom_freq.pack(side="left", fill="x", expand=True, padx=5)

        ctk.CTkButton(ctrl_frame, text="Reset Zoom", width=100, command=self.reset_zoom).pack(side="right", padx=10)

        # Eventos do mouse
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)

    def _build_context_menu(self):
        self.menu = tk.Menu(self, tearoff=0)
        # tempo vertical / horizontal
        self.menu.add_command(label="Marcar Tempo (vertical)", command=lambda: self.add_marker('time_v'))
        self.menu.add_command(label="Marcar Tempo (horizontal)", command=lambda: self.add_marker('time_h'))
        self.menu.add_separator()
        # freq vertical / horizontal
        self.menu.add_command(label="Marcar Frequência (vertical)", command=lambda: self.add_marker('freq_v'))
        self.menu.add_command(label="Marcar Frequência (horizontal)", command=lambda: self.add_marker('freq_h'))
        self.menu.add_separator()
        # limpar
        self.menu.add_command(label="Limpar Todos Marcadores", command=lambda: self.clear_markers('all'))

    def _build_marker_panel(self):
        # Container à direita da área dos gráficos
        self.side_panel = ctk.CTkFrame(self.main_frame, width=250, corner_radius=6)
        self.side_panel.pack(side="right", fill="y", padx=(0, 0), pady=10)

        # --- Marcadores de Tempo
        frm_time = ctk.CTkFrame(self.side_panel, fg_color="#1E1E1E", corner_radius=6)
        frm_time.pack(fill="x", pady=(0, 10), padx=5)
        ctk.CTkLabel(frm_time, text="Marcadores de Tempo", text_color="cyan", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=2, padx=8)

        self.lbl_x1 = ctk.CTkLabel(frm_time, text="X1: ---")
        self.lbl_x1.pack(anchor="w", padx=8)
        self.lbl_x2 = ctk.CTkLabel(frm_time, text="X2: ---")
        self.lbl_x2.pack(anchor="w", padx=8)
        self.lbl_dx = ctk.CTkLabel(frm_time, text="ΔX: ---")
        self.lbl_dx.pack(anchor="w", padx=8, pady=(0, 5))
        self.lbl_y1 = ctk.CTkLabel(frm_time, text="Y1: ---")
        self.lbl_y1.pack(anchor="w", padx=8)
        self.lbl_y2 = ctk.CTkLabel(frm_time, text="Y2: ---")
        self.lbl_y2.pack(anchor="w", padx=8)
        self.lbl_dy = ctk.CTkLabel(frm_time, text="ΔY: ---")
        self.lbl_dy.pack(anchor="w", padx=8)

        # --- Marcadores de Frequência
        frm_freq = ctk.CTkFrame(self.side_panel, fg_color="#1E1E1E", corner_radius=6)
        frm_freq.pack(fill="x", pady=(0, 10), padx=5)
        ctk.CTkLabel(frm_freq, text="Marcadores de Frequência", text_color="orange", font=("Arial", 12, "bold")).pack(
            anchor="w", pady=2, padx=8)

        self.lbl_f1 = ctk.CTkLabel(frm_freq, text="F1: ---")
        self.lbl_f1.pack(anchor="w", padx=8)
        self.lbl_f2 = ctk.CTkLabel(frm_freq, text="F2: ---")
        self.lbl_f2.pack(anchor="w", padx=8)
        self.lbl_df = ctk.CTkLabel(frm_freq, text="ΔF: ---")
        self.lbl_df.pack(anchor="w", padx=8, pady=(0, 5))
        self.lbl_m1 = ctk.CTkLabel(frm_freq, text="|Y1|: ---")
        self.lbl_m1.pack(anchor="w", padx=8)
        self.lbl_m2 = ctk.CTkLabel(frm_freq, text="|Y2|: ---")
        self.lbl_m2.pack(anchor="w", padx=8)
        self.lbl_dm = ctk.CTkLabel(frm_freq, text="Δ|Y|: ---")
        self.lbl_dm.pack(anchor="w", padx=8)

    def _build_status_bar(self):
        self.status_bar = ctk.CTkLabel(
            self,
            text="Pronto",
            anchor="w",
            height=25,
            fg_color="#222222",
            text_color="white",
            font=("Arial", 11, "italic"),
            corner_radius=0
        )
        self.status_bar.pack(side="bottom", fill="x", padx=0, pady=0)

    def set_status(self, msg, color="white"):
        """Atualiza a barra de status no rodapé."""
        self.status_bar.configure(text=msg, text_color=color)

    def _on_am_fm_toggle(self):
        self.slider_am.configure(state="normal" if self.mod_am.get() else "disabled")
        self.slider_fm_dev.configure(state="normal" if self.mod_fm.get() else "disabled")

    def submit_plot_task(self):
        self.btn_generate.configure(state="disabled", text="Gerando...")
        self.set_status("⏳ Iniciando geração de sinal...", "yellow")
        self.executor.submit(self._compute_and_plot_task)

    def _compute_and_plot_task(self):
        try:
            params = self._validate_inputs()
            if not params:
                self.set_status("❌ Erro ao gerar o sinal", "red")
                return

            # gera eixo tempo e sinal
            t = np.arange(params['N']) / params['Fs']
            y = self._generate_waveform(params, t)
            y = self._apply_modulation(params, y, t)

            # FFT
            Y = fftshift(fft(y))
            f = fftshift(fftfreq(params['N'], 1 / params['Fs']))

            # guarda os dados para plot
            self.last_data = {'t': t, 'y': y, 'f': f, 'Y': np.abs(Y)}

            # executa atualização de plot na thread principal
            self.after(0, self._update_plots)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Erro de Cálculo", str(e)))
            self.set_status(f"❌ Erro: {str(e)}", "red")

        finally:
            # reativa o botão
            self.after(0, lambda: self.btn_generate.configure(state="normal", text="Gerar Sinal"))

    def _validate_inputs(self):
        try:
            p = {
                'duration': float(self.entry_duration.get()),
                'Fc': float(self.entry_fc.get()) * UNIT_MULTIPLIERS[self.units_fc.get()],
                'Fs': float(self.entry_fs.get()) * UNIT_MULTIPLIERS[self.units_fs.get()],
                'Fm': float(self.entry_fm.get()) * UNIT_MULTIPLIERS[self.units_fm.get()],
                'waveform': self.waveform.get(),
                'am_on': self.mod_am.get(),
                'am_depth': self.slider_am.get(),
                'fm_on': self.mod_fm.get(),
                'fm_dev': self.slider_fm_dev.get()
            }

            # Validações básicas
            if p['duration'] <= 0:
                raise ValueError("Duração deve ser maior que zero.")
            if p['Fs'] <= 0:
                raise ValueError("Taxa de amostragem deve ser maior que zero.")

            # Número de pontos
            p['N'] = int(p['duration'] * p['Fs'])
            if p['N'] < 16:
                raise ValueError("Combinação de duração e Fs resulta em poucos pontos (< 16).")
            if p['N'] % 2 != 0:
                p['N'] += 1  # garante N par

            # Critério de Nyquist (para Fc e Fm)
            max_freq = max(p['Fc'], p['Fm'])
            if max_freq >= p['Fs'] / 2:
                raise ValueError(
                    f"Nyquist violado! Máxima frequência ({max_freq}Hz) deve ser < Fs/2 ({p['Fs'] / 2}Hz)."
                )

            # Ajusta dinamicamente o alcance do slider de desvio FM
            self.after(0, lambda: self.slider_fm_dev.configure(
                to=max(0.1, p['Fs'] / 2 - p['Fc'])
            ))

            return p

        except ValueError as e:
            self.after(0, lambda: messagebox.showerror("Erro de Entrada", str(e)))
            self.set_status(f"❌ Erro: {str(e)}", "red")
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
            return np.exp(-t / (p['duration'] / 5)) * np.sin(2 * np.pi * Fc * t)
        elif wf == "Passo (step)":
            return np.heaviside(t - t[p['N'] // 4], 1.0)
        elif wf == "Rampa":
            return t / t[-1]
        elif wf == "Parábola":
            return (t / t[-1]) ** 2
        elif wf == "Impulso":
            return unit_impulse(p['N'], 'mid')
        elif wf == "Tangente":
            y = np.tan(np.pi * Fc * t);
            return np.clip(y, -10, 10)
        else:
            return np.zeros_like(t)

    def _apply_modulation(self, p, y, t):
        if p['am_on']:
            modulator = np.sin(2 * np.pi * p['Fm'] * t)
            y *= (1 + p['am_depth'] * modulator)
        if p['fm_on']:
            modulating_signal = y
            integrated_modulator = np.cumsum(modulating_signal) * (1 / p['Fs'])
            y = np.sin(2 * np.pi * p['Fc'] * t + 2 * np.pi * p['fm_dev'] * integrated_modulator)
        return y

    def _update_plots(self):
        if not self.last_data:
            return

        self.ax_time.clear()
        self.ax_freq.clear()

        # Cria a linha principal do gráfico do tempo
        self.time_plot_line, = self.ax_time.plot(self.last_data['t'], self.last_data['y'], color="cyan", zorder=5)
        self.ax_time.set_title("Domínio do Tempo", color='white')
        self.ax_time.grid(True, linestyle='--', alpha=0.5)
        self.ax_time.tick_params(colors='white')
        self.ax_time.xaxis.set_major_formatter(FuncFormatter(self._format_time_axis))

        self.ax_freq.plot(self.last_data['f'], self.last_data['Y'], color="orange")
        self.ax_freq.set_title("Domínio da Frequência (FFT)", color='white')
        self.ax_freq.set_ylabel("|Y(f)|", color='white')
        self.ax_freq.grid(True, linestyle='--', alpha=0.5)
        self.ax_freq.tick_params(colors='white')
        self.ax_freq.xaxis.set_major_formatter(FuncFormatter(self._format_freq_axis))

        self.fig.tight_layout(pad=3.0)
        self.clear_markers('all')
        self.reset_zoom()
        self.canvas.draw()
        self.set_status("✅ Gráficos atualizados com sucesso!", "lightgreen")

    def reset_zoom(self):
        if not self.last_data:
            return

        self.ax_time.set_xlim(self.last_data['t'][0], self.last_data['t'][-1])
        self.ax_freq.set_xlim(self.last_data['f'][0], self.last_data['f'][-1])
        self.zoom_time.set(1.0)
        self.zoom_freq.set(1.0)
        self.update_time_zoom(1.0)  # Força a atualização do plot
        self.canvas.draw()

    def update_time_zoom(self, val):
        if not self.last_data or self.time_plot_line is None:
            return

        # Define os novos limites do eixo X
        center = self.ax_time.get_xlim()[0] + (self.ax_time.get_xlim()[1] - self.ax_time.get_xlim()[0]) / 2
        if val is not None:  # Se o slider foi movido
            total_width = self.last_data['t'][-1] - self.last_data['t'][0]
            new_width = total_width * float(val) if val > 0.001 else total_width * 0.001
            self.ax_time.set_xlim(center - new_width / 2, center + new_width / 2)

        # Lógica de Interpolação Dinâmica
        x_lim = self.ax_time.get_xlim()

        # Encontra os índices dos dados originais que estão na tela
        visible_indices = np.where((self.last_data['t'] >= x_lim[0]) & (self.last_data['t'] <= x_lim[1]))[0]

        if 4 < len(visible_indices) < INTERP_THRESHOLD:
            # POUCOS PONTOS: Interpola para criar uma curva suave
            t_visible = self.last_data['t'][visible_indices]
            y_visible = self.last_data['y'][visible_indices]

            cs = CubicSpline(t_visible, y_visible)
            t_interp = np.linspace(t_visible[0], t_visible[-1], INTERP_SAMPLES)
            y_interp = cs(t_interp)

            self.time_plot_line.set_data(t_interp, y_interp)
        else:
            # MUITOS PONTOS: Apenas exibe os pontos originais (eficiente)
            self.time_plot_line.set_data(self.last_data['t'], self.last_data['y'])

        self.canvas.draw_idle()

    def update_freq_zoom(self, val):
        if not self.last_data:
            return

        # Define os novos limites do eixo X da frequência
        center = (self.ax_freq.get_xlim()[0] + self.ax_freq.get_xlim()[1]) / 2
        total_width = self.last_data['f'][-1] - self.last_data['f'][0]

        new_width = total_width * float(val) if val > 0.001 else total_width * 0.001
        self.ax_freq.set_xlim(center - new_width / 2, center + new_width / 2)
        self.canvas.draw_idle()

    def _on_mouse_press(self, event):
        self.last_click_event = event

        if event.button == 3:  # Botão direito
            self.menu.post(event.guiEvent.x_root, event.guiEvent.y_root)
            return

        if event.inaxes:
            key = 'time' if event.inaxes is self.ax_time else 'freq'
            for mk in self.markers[f'{key}_v'] + self.markers[f'{key}_h']:
                contains, _ = mk.contains(event)
                if contains:
                    self.dragging_marker = mk
                    break

    def _on_mouse_release(self, event):
        self.dragging_marker = None

    def _on_mouse_motion(self, event):
        if not self.dragging_marker or not event.inaxes:
            return

        # Move marcador vertical, se aplicável
        if hasattr(self.dragging_marker, 'get_xdata') and event.xdata is not None:
            self.dragging_marker.set_xdata([event.xdata])

        # Move marcador horizontal, se aplicável
        if hasattr(self.dragging_marker, 'get_ydata') and event.ydata is not None:
            self.dragging_marker.set_ydata([event.ydata])

        # Atualiza os rótulos laterais
        self.update_marker_panel()
        self.canvas.draw_idle()
        self.set_status("🔧 Marcador movido", "cyan")

    def add_marker(self, kind):
        ev = self.last_click_event
        if not ev or not ev.inaxes:
            return

        # Escolhe o eixo
        ax = self.ax_time if kind.startswith('time') else self.ax_freq

        # Limite de 2 marcadores
        if len(self.markers[kind]) >= 2:
            self.set_status("⚠️ Máximo de 2 marcadores por tipo atingido", "orange")
            return

        # Desenha a linha
        if kind.endswith('_v'):
            line = ax.axvline(x=ev.xdata, linestyle='--',
                              color=('red' if kind.startswith('time') else 'lime'),
                              picker=5)
        else:
            line = ax.axhline(y=ev.ydata, linestyle='--',
                              color=('red' if kind.startswith('time') else 'lime'),
                              picker=5)

        # Armazena e redesenha
        self.markers[kind].append(line)
        self.update_marker_panel()
        self.canvas.draw_idle()
        self.set_status("✅ Marcador adicionado", "lightgreen")

    def clear_markers(self, which):
        keys = list(self.markers.keys()) if which == 'all' else [which]
        for k in keys:
            for mk in self.markers[k]:
                mk.remove()
            self.markers[k].clear()
        self.update_marker_panel()
        self.canvas.draw_idle()

    def update_marker_panel(self):
        # --- Tempo
        tv = sorted([m.get_xdata()[0] for m in self.markers['time_v'] if
                     self.ax_time.get_xlim()[0] <= m.get_xdata()[0] <= self.ax_time.get_xlim()[1]])[:2]
        if len(tv) == 2:
            self.lbl_x1.configure(text=f"X1: {self._format_time(tv[0])}")
            self.lbl_x2.configure(text=f"X2: {self._format_time(tv[1])}")
            self.lbl_dx.configure(text=f"ΔX: {self._format_time(tv[1] - tv[0])}")
        else:
            self.lbl_x1.configure(text="X1: ---")
            self.lbl_x2.configure(text="X2: ---")
            self.lbl_dx.configure(text="ΔX: ---")

        th = sorted([m.get_ydata()[0] for m in self.markers['time_h'] if
                     self.ax_time.get_ylim()[0] <= m.get_ydata()[0] <= self.ax_time.get_ylim()[1]])[:2]
        if len(th) == 2:
            self.lbl_y1.configure(text=f"Y1: {th[0]:.3f}")
            self.lbl_y2.configure(text=f"Y2: {th[1]:.3f}")
            self.lbl_dy.configure(text=f"ΔY: {(th[1] - th[0]):.3f}")
        else:
            self.lbl_y1.configure(text="Y1: ---")
            self.lbl_y2.configure(text="Y2: ---")
            self.lbl_dy.configure(text="ΔY: ---")

        # --- Frequência
        fv = sorted([m.get_xdata()[0] for m in self.markers['freq_v'] if
                     self.ax_freq.get_xlim()[0] <= m.get_xdata()[0] <= self.ax_freq.get_xlim()[1]])[:2]
        if len(fv) == 2:
            self.lbl_f1.configure(text=f"F1: {self._format_freq(fv[0])}")
            self.lbl_f2.configure(text=f"F2: {self._format_freq(fv[1])}")
            self.lbl_df.configure(text=f"ΔF: {self._format_freq(fv[1] - fv[0])}")
        else:
            self.lbl_f1.configure(text="F1: ---")
            self.lbl_f2.configure(text="F2: ---")
            self.lbl_df.configure(text="ΔF: ---")

        fh = sorted([m.get_ydata()[0] for m in self.markers['freq_h'] if
                     self.ax_freq.get_ylim()[0] <= m.get_ydata()[0] <= self.ax_freq.get_ylim()[1]])[:2]
        if len(fh) == 2:
            self.lbl_m1.configure(text=f"|Y1|: {fh[0]:.2f}")
            self.lbl_m2.configure(text=f"|Y2|: {fh[1]:.2f}")
            self.lbl_dm.configure(text=f"Δ|Y|: {(fh[1] - fh[0]):.2f}")
        else:
            self.lbl_m1.configure(text="|Y1|: ---")
            self.lbl_m2.configure(text="|Y2|: ---")
            self.lbl_dm.configure(text="Δ|Y|: ---")

    def _format_time(self, s):
        if s < 1e-6:
            return f"{s * 1e9:.2f} ns"
        elif s < 1e-3:
            return f"{s * 1e6:.2f} µs"
        elif s < 1:
            return f"{s * 1e3:.2f} ms"
        else:
            return f"{s:.2f} s"

    def _format_freq(self, f):
        if f < 1e3:
            return f"{f:.1f} Hz"
        elif f < 1e6:
            return f"{f / 1e3:.2f} kHz"
        elif f < 1e9:
            return f"{f / 1e6:.2f} MHz"
        else:
            return f"{f / 1e9:.2f} GHz"

    def _format_time_axis(self, x, pos):
        return self._format_time(x)

    def _format_freq_axis(self, f, pos):
        return self._format_freq(f)

    def export_data(self):
        if not self.last_data:
            messagebox.showerror("Erro", "Gere um sinal primeiro.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json"), ("CSV", "*.csv")])
        if not path:
            return

        try:
            if path.endswith(".json"):
                with open(path, "w") as f:
                    json.dump({k: v.tolist() for k, v in self.last_data.items()}, f, indent=2)
            else:
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.last_data.keys())
                    writer.writerows(zip(*self.last_data.values()))

            self.set_status(f"📁 Dados exportados para {path}", "lightblue")
        except Exception as e:
            messagebox.showerror("Erro de Exportação", str(e))
            self.set_status(f"❌ Erro ao exportar: {str(e)}", "red")


if __name__ == "__main__":
    app = SignalGeneratorApp()
    app.mainloop()