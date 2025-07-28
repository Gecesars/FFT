import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FuncFormatter
from scipy.signal import square, sawtooth, unit_impulse, gausspulse, chirp
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import CubicSpline
from scipy.special import jv
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
import csv
import json
from concurrent.futures import ThreadPoolExecutor
import struct
import os
import math
import array

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Constantes
INTERP_THRESHOLD = 100
INTERP_SAMPLES = 500
UNIT_MULTIPLIERS = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
WAV_TOTAL_SIZE = 15360  # Tamanho total do arquivo para compatibilidade

# Escalas pré-definidas FNIRSI
VOLT_LIST = [[5.0, "V", 1], [2.5, "V", 1], [1.0, "V", 1], [500, "mV", 0.001],
             [200, "mV", 0.001], [100, "mV", 0.001], [50, "mV", 0.001]]

TIME_LIST = [[50, "S", 1], [20, "S", 1], [10, "S", 1], [5, "S", 1], [2, "S", 1], [1, "S", 1],
             [500, "mS", 0.001], [200, "mS", 0.001], [100, "mS", 0.001], [50, "mS", 0.001],
             [20, "mS", 0.001], [10, "mS", 0.001], [5, "mS", 0.001], [2, "mS", 0.001], [1, "mS", 0.001],
             [500, "uS", 1e-6], [200, "uS", 1e-6], [100, "uS", 1e-6], [50, "uS", 1e-6], [20, "uS", 1e-6],
             [10, "uS", 1e-6], [5, "uS", 1e-6], [2, "uS", 1e-6], [1, "uS", 1e-6],
             [500, "nS", 1e-9], [200, "nS", 1e-9], [100, "nS", 1e-9], [50, "nS", 1e-9], [20, "nS", 1e-9],
             [10, "nS", 1e-9]]


class SignalGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gerador de Sinais Avançado")
        self.geometry("1600x1000")

        # Configurar layout principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.executor = ThreadPoolExecutor(max_workers=20)
        self.last_data = {}
        self.markers = {
            'time_v': [],  # verticais no plot de tempo
            'time_h': [],  # horizontais no plot de tempo
            'freq_v': [],  # verticais no plot de frequência
            'freq_h': []  # horizontais no plot de frequência
        }
        self.dragging_marker = None
        self.dragging_type = None
        self.original_position = None
        self.time_plot_line = None
        self.freq_plot_line = None
        self.after_ids = []
        self.y_scale = 1.0  # Escala de amplitude
        self.zoom_factor = 1.0  # Fator de zoom inicial

        self.mod_am = tk.BooleanVar(value=False)
        self.mod_fm = tk.BooleanVar(value=False)

        self._build_sidebar()
        self._build_plot_area()
        self._build_context_menu()
        self._build_marker_panel()
        self._build_analysis_panels()
        self._build_status_bar()

        # Configurar tratamento de fechamento
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        """Cancelar callbacks pendentes ao fechar a janela"""
        for after_id in self.after_ids:
            self.after_cancel(after_id)
        self.destroy()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=8)
        self.sidebar.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.sidebar.grid_propagate(False)

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
        self.entry_vpp = self._add_entry(frm_gen, "Amplitude (Vpp):", "1.0")  # Amplitude pico a pico

        # Lista de formas de onda
        waveforms = [
            "Seno", "Cosseno", "Quadrada", "Triangular", "Dente de Serra", "Pulso",
            "Ruído Branco", "Exp Decaimento", "Passo (step)", "Rampa", "Parábola",
            "Impulso", "Tangente", "Sinc", "Gaussiana", "Chirp Linear", "Chirp Quadrático",
            "Onda AM", "Onda FM", "Batimento", "Lorentziana", "Hiperbólica", "Bessel",
            "Sinc Modulado", "Pulso Gaussiano", "Dente de Serra Modificado", "Onda AM-DSB",
            "Onda FM Estéreo", "Onda Quadrada Modulada", "Onda Triangular Modulada"
        ]
        self.waveform = self._add_option_menu(frm_gen, "Forma de Onda:", waveforms)

        # --- Modulação ---
        frm_mod = section("Modulação", "#2d3e50")
        self.entry_fm, self.units_fm = self._add_frequency_entry(frm_mod, "Moduladora (Fm):", "10")
        ctk.CTkCheckBox(frm_mod, text="Ativar AM", variable=self.mod_am, command=self._on_am_fm_toggle).pack(anchor="w",
                                                                                                             padx=10,
                                                                                                             pady=5)
        self.slider_am = ctk.CTkSlider(frm_mod, from_=0, to=2, number_of_steps=200)
        self.slider_am.set(0.5)
        self.slider_am.configure(state="disabled")
        self.slider_am.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkCheckBox(frm_mod, text="Ativar FM", variable=self.mod_fm, command=self._on_am_fm_toggle).pack(anchor="w",
                                                                                                             padx=10,
                                                                                                             pady=5)
        self.slider_fm_dev = ctk.CTkSlider(frm_mod, from_=0, to=1, number_of_steps=100)
        self.slider_fm_dev.set(0.5)
        self.slider_fm_dev.configure(state="disabled")
        self.slider_fm_dev.pack(fill="x", padx=10, pady=(0, 10))

        # --- Comandos ---
        frm_cmd = section("Comandos", "#333333")
        self.btn_generate = ctk.CTkButton(frm_cmd, text="Gerar Sinal", command=self.submit_plot_task)
        self.btn_generate.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(frm_cmd, text="Exportar Dados", command=self.export_data).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(frm_cmd, text="Importar Forma de Onda (WAV)", command=self.import_wav).pack(fill="x", padx=10,
                                                                                                  pady=5)
        ctk.CTkButton(frm_cmd, text="Exportar WAV", command=self.export_wav).pack(fill="x", padx=10, pady=5)

    def _add_entry(self, parent, label, default):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(5, 0))
        entry = ctk.CTkEntry(parent)
        entry.insert(0, default)
        entry.pack(fill="x", padx=10, pady=(0, 5))
        return entry

    def _add_frequency_entry(self, parent, label, default_val, default_unit="Hz"):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(5, 0))
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=(0, 5))
        entry = ctk.CTkEntry(frame)
        entry.insert(0, default_val)
        entry.pack(side="left", fill="x", expand=True)
        units = ctk.CTkOptionMenu(frame, values=list(UNIT_MULTIPLIERS.keys()), width=75)
        units.set(default_unit)
        units.pack(side="left", padx=(5, 0))
        return entry, units

    def _add_option_menu(self, parent, label, values):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(5, 0))
        menu = ctk.CTkOptionMenu(parent, values=values)
        menu.set(values[0])
        menu.pack(fill="x", padx=10, pady=5)
        return menu

    def _build_plot_area(self):
        plot_frame = ctk.CTkFrame(self)
        plot_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        plot_frame.grid_propagate(False)
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=1)

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

        # Frame para sliders de escala Y (agora à esquerda)
        y_scale_frame = ctk.CTkFrame(plot_frame, width=40)
        y_scale_frame.pack(side="left", fill="y", padx=(5, 0), pady=5)

        # Sliders de escala Y (agora à esquerda)
        ctk.CTkLabel(y_scale_frame, text="Escala Tempo").pack(pady=(5, 0))
        self.scale_time = ctk.CTkSlider(y_scale_frame, from_=0.1, to=5.0,
                                        orientation="vertical", command=self.update_time_scale)
        self.scale_time.set(1.0)
        self.scale_time.pack(fill="y", expand=True, pady=5, padx=5)

        ctk.CTkLabel(y_scale_frame, text="Escala Freq").pack(pady=(5, 0))
        self.scale_freq = ctk.CTkSlider(y_scale_frame, from_=0.1, to=5.0,
                                        orientation="vertical", command=self.update_freq_scale)
        self.scale_freq.set(1.0)
        self.scale_freq.pack(fill="y", expand=True, pady=5, padx=5)

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
        self.canvas.mpl_connect('pick_event', self._on_pick_event)

    def update_time_scale(self, value):
        """Atualiza a escala do eixo Y no gráfico de tempo"""
        if not self.last_data:
            return

        try:
            vpp = float(self.entry_vpp.get())
            self.ax_time.set_ylim(-vpp / 2 * float(value), vpp / 2 * float(value))
            self.canvas.draw_idle()
        except:
            pass

    def update_freq_scale(self, value):
        """Atualiza a escala do eixo Y no gráfico de frequência"""
        if not self.last_data:
            return

        try:
            scale_factor = float(value)
            max_val = np.max(self.last_data['Y']) * 1.1 * scale_factor
            self.ax_freq.set_ylim(0, max_val)
            self.canvas.draw_idle()
        except:
            pass

    def _build_context_menu(self):
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Marcar Tempo (vertical)", command=lambda: self.add_marker('time_v'))
        self.menu.add_command(label="Marcar Tempo (horizontal)", command=lambda: self.add_marker('time_h'))
        self.menu.add_separator()
        self.menu.add_command(label="Marcar Frequência (vertical)", command=lambda: self.add_marker('freq_v'))
        self.menu.add_command(label="Marcar Frequência (horizontal)", command=lambda: self.add_marker('freq_h'))
        self.menu.add_separator()
        self.menu.add_command(label="Limpar Todos Marcadores", command=lambda: self.clear_markers('all'))

    def _build_marker_panel(self):
        # Container à direita da área dos gráficos
        self.side_panel = ctk.CTkFrame(self, width=350, corner_radius=6)
        self.side_panel.grid(row=0, column=2, padx=(0, 10), pady=10, sticky="nsew")
        self.side_panel.grid_propagate(False)
        self.side_panel.grid_rowconfigure(1, weight=1)  # Dá espaço para o painel de análise

        # Frame para marcadores
        marker_frame = ctk.CTkFrame(self.side_panel)
        marker_frame.pack(fill="x", pady=(0, 10), padx=5)

        # --- Marcadores de Tempo
        frm_time = ctk.CTkFrame(marker_frame, fg_color="#1E1E1E", corner_radius=6)
        frm_time.pack(fill="x", pady=(0, 5), padx=5)
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
        frm_freq = ctk.CTkFrame(marker_frame, fg_color="#1E1E1E", corner_radius=6)
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

    def _build_analysis_panels(self):
        # Container para abas de análise (abaixo do painel de marcadores)
        self.analysis_notebook = ctk.CTkTabview(self.side_panel, height=300)
        self.analysis_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Aba de análise de tempo
        self.time_analysis_tab = self.analysis_notebook.add("Domínio do Tempo")
        self.freq_analysis_tab = self.analysis_notebook.add("Domínio da Frequência")

        # --- Painel de Análise de Tempo ---
        time_frame = ctk.CTkScrollableFrame(self.time_analysis_tab)
        time_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Título
        ctk.CTkLabel(time_frame, text="Análise do Sinal no Tempo",
                     font=("Arial", 14, "bold"), text_color="cyan").pack(anchor="w", pady=(0, 10))

        # Variáveis para os resultados
        self.time_analysis_results = {
            'vpp': ctk.StringVar(value="---"),
            'rms': ctk.StringVar(value="---"),
            'mean': ctk.StringVar(value="---"),
            'crest_factor': ctk.StringVar(value="---"),
            'zero_crossing': ctk.StringVar(value="---"),
            'frequency': ctk.StringVar(value="---"),
            'duty_cycle': ctk.StringVar(value="---"),
            'peak_to_rms': ctk.StringVar(value="---"),
            'kurtosis': ctk.StringVar(value="---"),
            'skewness': ctk.StringVar(value="---")
        }

        # Criar labels para cada métrica
        metrics = [
            ("Tensão Pico a Pico (Vpp):", self.time_analysis_results['vpp']),
            ("Tensão RMS:", self.time_analysis_results['rms']),
            ("Tensão Média (DC):", self.time_analysis_results['mean']),
            ("Fator de Crista:", self.time_analysis_results['crest_factor']),
            ("Taxa de Cruzamento por Zero:", self.time_analysis_results['zero_crossing']),
            ("Frequência Estimada:", self.time_analysis_results['frequency']),
            ("Ciclo de Trabalho (Duty Cycle):", self.time_analysis_results['duty_cycle']),
            ("Relação Pico/RMS:", self.time_analysis_results['peak_to_rms']),
            ("Curtose:", self.time_analysis_results['kurtosis']),
            ("Assimetria (Skewness):", self.time_analysis_results['skewness'])
        ]

        for label_text, var in metrics:
            frame = ctk.CTkFrame(time_frame, fg_color="transparent")
            frame.pack(fill="x", padx=5, pady=2)
            lbl = ctk.CTkLabel(frame, text=label_text, width=220, anchor="w")
            lbl.pack(side="left", anchor="w")
            val = ctk.CTkLabel(frame, textvariable=var, width=100, anchor="e")
            val.pack(side="right", anchor="e")

        # --- Painel de Análise de Frequência ---
        freq_frame = ctk.CTkScrollableFrame(self.freq_analysis_tab)
        freq_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Título
        ctk.CTkLabel(freq_frame, text="Análise do Sinal na Frequência",
                     font=("Arial", 14, "bold"), text_color="orange").pack(anchor="w", pady=(0, 10))

        # Variáveis para os resultados
        self.freq_analysis_results = {
            'fundamental': ctk.StringVar(value="---"),
            'fund_amp': ctk.StringVar(value="---"),
            'thd': ctk.StringVar(value="---"),
            'snr': ctk.StringVar(value="---"),
            'sfdr': ctk.StringVar(value="---"),
            'bandwidth': ctk.StringVar(value="---"),
            'mod_index': ctk.StringVar(value="---"),
            'harmonics': ctk.StringVar(value="---"),
            'noise_floor': ctk.StringVar(value="---"),
            'peak_freq': ctk.StringVar(value="---")
        }

        # Criar labels para cada métrica
        metrics = [
            ("Frequência Fundamental:", self.freq_analysis_results['fundamental']),
            ("Amplitude Fundamental:", self.freq_analysis_results['fund_amp']),
            ("THD (Distorção Harmônica):", self.freq_analysis_results['thd']),
            ("SNR (Relação Sinal-Ruído):", self.freq_analysis_results['snr']),
            ("SFDR (Faixa Dinâmica):", self.freq_analysis_results['sfdr']),
            ("Largura de Banda:", self.freq_analysis_results['bandwidth']),
            ("Índice de Modulação:", self.freq_analysis_results['mod_index']),
            ("Nível de Harmônicos:", self.freq_analysis_results['harmonics']),
            ("Piso de Ruído:", self.freq_analysis_results['noise_floor']),
            ("Frequência de Pico:", self.freq_analysis_results['peak_freq'])
        ]

        for label_text, var in metrics:
            frame = ctk.CTkFrame(freq_frame, fg_color="transparent")
            frame.pack(fill="x", padx=5, pady=2)
            lbl = ctk.CTkLabel(frame, text=label_text, width=220, anchor="w")
            lbl.pack(side="left", anchor="w")
            val = ctk.CTkLabel(frame, textvariable=var, width=100, anchor="e")
            val.pack(side="right", anchor="e")

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
        self.status_bar.grid(row=1, column=0, columnspan=3, sticky="ew", padx=0, pady=0)

    def set_status(self, msg, color="white"):
        self.status_bar.configure(text=msg, text_color=color)

    def _on_am_fm_toggle(self):
        self.slider_am.configure(state="normal" if self.mod_am.get() else "disabled")
        self.slider_fm_dev.configure(state="normal" if self.mod_fm.get() else "disabled")

    def submit_plot_task(self):
        self.btn_generate.configure(state="disabled", text="Gerando...")
        self.set_status("⏳ Iniciando geração de sinal...", "yellow")
        future = self.executor.submit(self._compute_and_plot_task)
        future.add_done_callback(
            lambda future: self.after(0, lambda: self.btn_generate.configure(state="normal", text="Gerar Sinal")))

    def _compute_and_plot_task(self):
        try:
            params = self._validate_inputs()
            if not params:
                self.set_status("❌ Erro ao gerar o sinal", "red")
                return

            # gera eixo tempo e sinal
            t = np.arange(params['N']) / params['Fs']
            y = self._generate_waveform(params, t)

            # Aplica amplitude
            try:
                vpp = float(self.entry_vpp.get())
                y = y * (vpp / 2)  # Normaliza para Vpp
            except ValueError:
                y = y  # Mantém amplitude padrão se entrada inválida

            # Aplica modulações apenas se não forem formas de onda pré-moduladas
            if not params['waveform'].startswith("Onda AM") and not params['waveform'].startswith("Onda FM"):
                y = self._apply_modulation(params, y, t)

            # FFT
            Y = fftshift(fft(y))
            f = fftshift(fftfreq(params['N'], 1 / params['Fs']))

            # guarda os dados para plot
            self.last_data = {'t': t, 'y': y, 'f': f, 'Y': np.abs(Y)}

            # executa atualização de plot na thread principal
            self.after(0, self._update_plots)
            self.after(0, self.update_analysis_panels)

        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Erro de Cálculo", error_msg))
            self.set_status(f"❌ Erro: {error_msg}", "red")

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

            # Critério de Nyquist
            max_freq = max(p['Fc'], p['Fm'])
            if max_freq >= p['Fs'] / 2:
                raise ValueError(
                    f"Nyquist violado! Máxima frequência ({max_freq}Hz) deve ser < Fs/2 ({p['Fs'] / 2}Hz)."
                )

            # Ajusta o alcance do slider de desvio FM
            self.after(0, lambda: self.slider_fm_dev.configure(
                to=max(0.1, p['Fs'] / 2 - p['Fc'])
            ))

            return p

        except ValueError as e:
            self.after(0, lambda: messagebox.showerror("Erro de Entrada", str(e)))
            self.set_status(f"❌ Erro: {str(e)}", "red")
            return None

    def _generate_waveform(self, p, t):
        wf, Fc, Fm = p['waveform'], p['Fc'], p['Fm']
        duration = p['duration']

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
            return np.exp(-t / (duration / 5)) * np.sin(2 * np.pi * Fc * t)
        elif wf == "Passo (step)":
            return np.heaviside(t - t[p['N'] // 4], 1.0)
        elif wf == "Rampa":
            return t / t[-1]
        elif wf == "Parábola":
            return (t / t[-1]) ** 2
        elif wf == "Impulso":
            return unit_impulse(p['N'], 'mid')
        elif wf == "Tangente":
            y = np.tan(np.pi * Fc * t)
            return np.clip(y, -10, 10)

        # Formas de onda adicionais
        elif wf == "Sinc":
            return np.sinc(2 * Fc * t)
        elif wf == "Gaussiana":
            return np.exp(-(t - t.mean()) ** 2 / (0.1 * duration) ** 2) * np.sin(2 * np.pi * Fc * t)
        elif wf == "Chirp Linear":
            return chirp(t, f0=Fc, f1=5 * Fc, t1=t[-1], method='linear')
        elif wf == "Chirp Quadrático":
            return chirp(t, f0=Fc, f1=10 * Fc, t1=t[-1], method='quadratic')
        elif wf == "Onda AM":
            return (1 + 0.5 * np.sin(2 * np.pi * Fm * t)) * np.sin(2 * np.pi * Fc * t)
        elif wf == "Onda FM":
            return np.sin(2 * np.pi * Fc * t + 5 * np.sin(2 * np.pi * Fm * t))
        elif wf == "Batimento":
            return np.sin(2 * np.pi * Fc * t) + np.sin(2 * np.pi * (Fc + Fm) * t)
        elif wf == "Lorentziana":
            return 1 / (1 + (2 * np.pi * Fc * t) ** 2)
        elif wf == "Hiperbólica":
            return np.sinh(2 * np.pi * Fc * t)
        elif wf == "Bessel":
            return jv(0, 2 * np.pi * Fc * t)  # Função de Bessel de ordem 0

        # Mais formas de onda
        elif wf == "Sinc Modulado":
            return np.sinc(2 * Fc * t) * np.sin(2 * np.pi * Fm * t)
        elif wf == "Pulso Gaussiano":
            return gausspulse(t, fc=Fc, bw=0.5)
        elif wf == "Dente de Serra Modificado":
            return sawtooth(2 * np.pi * Fc * t, width=0.3)
        elif wf == "Onda AM-DSB":
            return np.sin(2 * np.pi * Fc * t) * np.sin(2 * np.pi * Fm * t)
        elif wf == "Onda FM Estéreo":
            left = np.sin(2 * np.pi * Fc * t + 3 * np.sin(2 * np.pi * Fm * t))
            right = np.sin(2 * np.pi * (Fc + 1000) * t + 3 * np.sin(2 * np.pi * Fm * t))
            return 0.5 * (left + right)
        elif wf == "Onda Quadrada Modulada":
            return square(2 * np.pi * Fc * t) * (1 + 0.5 * np.sin(2 * np.pi * Fm * t))
        elif wf == "Onda Triangular Modulada":
            return sawtooth(2 * np.pi * Fc * t, 0.5) * (1 + 0.3 * np.sin(2 * np.pi * Fm * t))
        elif wf == "Pulso Exponencial":
            return np.exp(-5 * t / duration) * np.sin(2 * np.pi * Fc * t)
        elif wf == "Onda Comb":
            return np.sign(np.sin(2 * np.pi * Fc * t) + 0.7)
        elif wf == "Onda Harmônica":
            return np.sin(2 * np.pi * Fc * t) + 0.5 * np.sin(2 * np.pi * 2 * Fc * t)

        else:
            return np.zeros_like(t)

    def _apply_modulation(self, p, y, t):
        # Aplicação correta de AM e FM
        if p['am_on']:
            modulator = np.sin(2 * np.pi * p['Fm'] * t)
            y = y * (1 + p['am_depth'] * modulator)

        if p['fm_on']:
            # Para FM, usamos a integral do sinal modulador
            phase = 2 * np.pi * p['Fc'] * t + 2 * np.pi * p['fm_dev'] * np.cumsum(np.sin(2 * np.pi * p['Fm'] * t)) * (
                    1 / p['Fs'])
            y = np.sin(phase)

        return y

    def _read_wav_file(self, filepath):
        """Lê o arquivo WAV e retorna o cabeçalho e buffers de dados"""
        with open(filepath, 'rb') as f:
            header = f.read(208)  # Cabeçalho de 208 bytes

            # Posiciona e lê os dados do canal 1
            f.seek(1000)
            data_ch1 = f.read(3000)

            # Posiciona e lê os dados do canal 2
            f.seek(4000)
            data_ch2 = f.read(3000)

        return header, [data_ch1, data_ch2]

    def _parse_header(self, header_bytes):
        """Decodifica o cabeçalho do WAV do Fnirsi"""
        volt_scale = []
        for x in range(2):  # Para cada canal
            # Índice da escala de tensão (bytes 4 e 14)
            scale_idx = header_bytes[4 + x * 10]
            if scale_idx < len(VOLT_LIST):
                scale = VOLT_LIST[scale_idx]
            else:
                scale = VOLT_LIST[0]  # Default se índice inválido
            volt_scale.append(scale)

        # Índice da escala de tempo (byte 22)
        time_idx = header_bytes[22]
        if time_idx < len(TIME_LIST):
            ts = TIME_LIST[time_idx]
        else:
            ts = TIME_LIST[0]  # Default se índice inválido

        return volt_scale, ts[2]  # Retorna escalas de tensão e multiplicador de tempo

    def _parse_channel_data(self, data_bytes, scale):
        """Decodifica os dados de um canal específico"""
        values = []
        for i in range(1500):  # 1500 amostras por canal
            # Cada amostra são 2 bytes (little-endian)
            byte1 = data_bytes[i * 2]
            byte2 = data_bytes[i * 2 + 1]

            # Converte para valor numérico
            val = (byte1 + 256 * byte2 - 200) * scale[0] / 50.0
            values.append(val)

        return values

    def _show_wav_preview(self, t, y, filename):
        """Mostra uma prévia do sinal WAV em uma janela modal"""
        preview = ctk.CTkToplevel(self)
        preview.title(f"Visualização do Sinal: {filename}")
        preview.geometry("800x600")
        preview.transient(self)
        preview.grab_set()

        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(t, y, 'b-')
        ax.set_title(f"Forma de Onda: {filename}")
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=preview)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_frame = ctk.CTkFrame(preview)
        btn_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(btn_frame, text="Fechar", command=preview.destroy).pack(side="right")

    def import_wav(self):
        """Importa e decodifica arquivo WAV do Fnirsi"""
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not filepath:
            return

        try:
            # Lê o arquivo WAV
            header, data_buffers = self._read_wav_file(filepath)

            # Decodifica o cabeçalho
            volt_scale, time_multiplier = self._parse_header(header)

            # Decodifica os dados dos canais
            ch1_data = self._parse_channel_data(data_buffers[0], volt_scale[0])
            ch2_data = self._parse_channel_data(data_buffers[1], volt_scale[1])

            # Mostra prévia do sinal em janela modal
            t = np.arange(len(ch1_data)) * time_multiplier
            self._show_wav_preview(t, ch1_data, os.path.basename(filepath))

            # Calcula parâmetros do sinal
            N = len(ch1_data)
            t = np.arange(N) * time_multiplier  # Vetor de tempo
            Fs = 1 / time_multiplier  # Frequência de amostragem
            vpp = max(ch1_data) - min(ch1_data)  # Tensão pico a pico

            # FFT
            Y = fftshift(fft(ch1_data))
            f = fftshift(fftfreq(N, 1 / Fs))

            # Salva os dados
            self.last_data = {'t': t, 'y': ch1_data, 'f': f, 'Y': np.abs(Y)}

            # Atualiza campos de entrada
            self.after(0, lambda: self.entry_duration.delete(0, tk.END))
            self.after(0, lambda: self.entry_duration.insert(0, f"{t[-1]:.4f}"))
            self.after(0, lambda: self.entry_fc.delete(0, tk.END))
            self.after(0, lambda: self.entry_fc.insert(0, "0"))  # Não sabemos Fc
            self.after(0, lambda: self.entry_fs.delete(0, tk.END))
            self.after(0, lambda: self.entry_fs.insert(0, str(Fs)))
            self.after(0, lambda: self.units_fs.set("Hz"))
            self.after(0, lambda: self.entry_vpp.delete(0, tk.END))
            self.after(0, lambda: self.entry_vpp.insert(0, f"{vpp:.2f}"))

            # Atualiza gráficos e análises
            self.after(0, self._update_plots)
            self.after(0, self.update_analysis_panels)
            self.set_status(f"✅ Sinal importado: {os.path.basename(filepath)}", "lightgreen")

            # Salva como JSON
            json_path = os.path.splitext(filepath)[0] + ".json"
            with open(json_path, 'w') as f:
                json.dump({
                    'time_multiplier': time_multiplier,
                    'voltage_scale': volt_scale[0][0],
                    'voltage_unit': volt_scale[0][1],
                    'time_values': t.tolist(),
                    'signal_values': ch1_data
                }, f, indent=2)

            self.set_status(f"📊 Dados salvos em: {json_path}", "lightblue")

        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Erro ao importar WAV", error_msg))
            self.set_status(f"❌ Erro ao importar: {error_msg}", "red")

    def export_wav(self):
        """Exporta o sinal atual em formato WAV compatível com Fnirsi 1013D"""
        if not self.last_data:
            messagebox.showerror("Erro", "Gere ou importe um sinal primeiro.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")]
        )
        if not filepath:
            return

        try:
            # Obtém dados normalizados para int16
            y = self.last_data['y']

            # Redimensiona para 1500 pontos (tamanho esperado pelo osciloscópio)
            if len(y) != 1500:
                t_original = np.linspace(0, len(y) / len(y), len(y))
                t_new = np.linspace(0, len(y) / len(y), 1500)
                y = np.interp(t_new, t_original, y)

            y_int16 = np.int16(y * 32767)

            # Taxa de amostragem
            fs_value = int(float(self.entry_fs.get()) * UNIT_MULTIPLIERS[self.units_fs.get()])

            # Cria cabeçalho padrão de 208 bytes
            header = bytearray(208)

            # Preenche o arquivo com zeros até o tamanho total
            file_data = bytearray(WAV_TOTAL_SIZE)
            file_data[0:208] = header

            # Adiciona os dados (1500 amostras de 16 bits)
            data_bytes = y_int16.tobytes()
            file_data[1000:1000 + len(data_bytes)] = data_bytes  # Canal 1

            # Escreve no arquivo
            with open(filepath, 'wb') as f:
                f.write(file_data)

            self.set_status(f"📁 Sinal exportado como WAV: {os.path.basename(filepath)}", "lightblue")

        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Erro ao exportar WAV", error_msg)
            self.set_status(f"❌ Erro ao exportar: {error_msg}", "red")

    def _update_plots(self):
        if not self.last_data:
            return

        # Salva os marcadores existentes antes de limpar
        saved_markers = self._save_markers_state()

        # Limpa apenas o conteúdo dos eixos, preservando os marcadores
        self.ax_time.clear()
        self.ax_freq.clear()

        # Plota os novos dados
        self.time_plot_line, = self.ax_time.plot(self.last_data['t'], self.last_data['y'], color="cyan", zorder=5)
        self.ax_time.set_title("Domínio do Tempo", color='white')
        self.ax_time.grid(True, linestyle='--', alpha=0.5)
        self.ax_time.tick_params(colors='white')
        self.ax_time.xaxis.set_major_formatter(FuncFormatter(self._format_time_axis))

        # Configura escala fixa para eixo Y do tempo
        try:
            vpp = float(self.entry_vpp.get())
            self.ax_time.set_ylim(-vpp / 2, vpp / 2)
        except:
            # Usa escala automática se amplitude inválida
            self.ax_time.autoscale(axis='y')

        self.freq_plot_line, = self.ax_freq.plot(self.last_data['f'], self.last_data['Y'], color="orange")
        self.ax_freq.set_title("Domínio da Frequência (FFT)", color='white')
        self.ax_freq.set_ylabel("|Y(f)|", color='white')
        self.ax_freq.grid(True, linestyle='--', alpha=0.5)
        self.ax_freq.tick_params(colors='white')
        self.ax_freq.xaxis.set_major_formatter(FuncFormatter(self._format_freq_axis))

        # Configura escala para eixo Y da frequência
        self.ax_freq.set_ylim(0, np.max(self.last_data['Y']) * 1.1)

        # Ajusta a visualização inicial para mostrar ~8 ciclos
        self._adjust_initial_view()

        # Restaura os marcadores
        self._restore_markers_state(saved_markers)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
        self.set_status("✅ Gráficos atualizados com sucesso!", "lightgreen")

    def _adjust_initial_view(self):
        """Ajusta a visualização inicial para mostrar aproximadamente 8 ciclos completos"""
        if not self.last_data:
            return

        try:
            # Para o domínio do tempo
            t = self.last_data['t']
            y = self.last_data['y']

            # Estima a frequência fundamental
            zero_crossings = np.where(np.diff(np.sign(y)))[0]
            if len(zero_crossings) > 1:
                avg_period = np.mean(np.diff(t[zero_crossings])) * 2
                freq_est = 1 / avg_period if avg_period > 0 else 1

                # Calcula o tempo para mostrar 8 ciclos
                display_time = 8 / freq_est

                # Limita ao tempo total disponível
                display_time = min(display_time, t[-1] - t[0])

                # Centraliza no meio do sinal
                center = (t[0] + t[-1]) / 2
                self.ax_time.set_xlim(center - display_time / 2, center + display_time / 2)

                # Ajusta o slider de zoom para mostrar aproximadamente metade do sinal
                self.zoom_factor = display_time / (t[-1] - t[0])
                self.zoom_time.set(self.zoom_factor)
            else:
                # Se não detectar cruzamentos por zero, mostra todo o sinal
                self.ax_time.set_xlim(t[0], t[-1])
                self.zoom_time.set(1.0)

            # Para o domínio da frequência
            f = self.last_data['f']
            Y = self.last_data['Y']

            # Encontra a frequência fundamental
            fundamental_idx = np.argmax(Y)
            fundamental_freq = abs(f[fundamental_idx])

            # Ajusta para mostrar do zero até 5x a frequência fundamental
            max_freq = max(5 * fundamental_freq, 100)  # Mínimo de 100Hz
            self.ax_freq.set_xlim(0, max_freq)

            # Ajusta o slider de zoom
            self.zoom_freq.set(min(1.0, max_freq / (f[-1] - f[0])))

        except Exception as e:
            print(f"Erro ao ajustar visualização: {e}")
            # Se ocorrer erro, mostra todo o sinal
            self.ax_time.set_xlim(self.last_data['t'][0], self.last_data['t'][-1])
            self.ax_freq.set_xlim(self.last_data['f'][0], self.last_data['f'][-1])
            self.zoom_time.set(1.0)
            self.zoom_freq.set(1.0)

    def _save_markers_state(self):
        """Salva o estado atual dos marcadores"""
        state = {}
        for marker_type in self.markers:
            state[marker_type] = []
            for marker in self.markers[marker_type]:
                if marker_type.endswith('_v'):
                    state[marker_type].append(marker.get_xdata()[0])
                else:
                    state[marker_type].append(marker.get_ydata()[0])
        return state

    def _restore_markers_state(self, state):
        """Restaura os marcadores a partir do estado salvo"""
        for marker_type in state:
            self.markers[marker_type] = []
            for value in state[marker_type]:
                ax = self.ax_time if marker_type.startswith('time') else self.ax_freq
                color = 'red' if marker_type.startswith('time') else 'lime'

                if marker_type.endswith('_v'):
                    marker = ax.axvline(x=value, linestyle='--', color=color, picker=5)
                else:
                    marker = ax.axhline(y=value, linestyle='--', color=color, picker=5)

                self.markers[marker_type].append(marker)

    def reset_zoom(self):
        if not self.last_data:
            return

        self._adjust_initial_view()
        self.canvas.draw()

    def update_time_zoom(self, val):
        if not self.last_data or self.time_plot_line is None:
            return

        # Define os novos limites do eixo X
        center = self.ax_time.get_xlim()[0] + (self.ax_time.get_xlim()[1] - self.ax_time.get_xlim()[0]) / 2
        if val is not None:
            total_width = self.last_data['t'][-1] - self.last_data['t'][0]
            new_width = total_width * float(val) if val > 0.001 else total_width * 0.001
            self.ax_time.set_xlim(center - new_width / 2, center + new_width / 2)

        # Interpolação Dinâmica
        x_lim = self.ax_time.get_xlim()
        visible_indices = np.where((self.last_data['t'] >= x_lim[0]) & (self.last_data['t'] <= x_lim[1]))[0]

        if 4 < len(visible_indices) < INTERP_THRESHOLD:
            t_visible = self.last_data['t'][visible_indices]
            y_visible = self.last_data['y'][visible_indices]

            # Usa spline cúbica para interpolação suave
            cs = CubicSpline(t_visible, y_visible)
            t_interp = np.linspace(t_visible[0], t_visible[-1], INTERP_SAMPLES)
            y_interp = cs(t_interp)

            self.time_plot_line.set_data(t_interp, y_interp)
        else:
            self.time_plot_line.set_data(self.last_data['t'], self.last_data['y'])

        self.canvas.draw_idle()

    def update_freq_zoom(self, val):
        if not self.last_data:
            return

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

    def _on_pick_event(self, event):
        """Manipula a seleção de marcadores"""
        if event.mouseevent.button != 1:  # Apenas botão esquerdo
            return

        if event.artist in self.markers['time_v'] + self.markers['time_h']:
            self.dragging_type = 'time'
        elif event.artist in self.markers['freq_v'] + self.markers['freq_h']:
            self.dragging_type = 'freq'
        else:
            return

        self.dragging_marker = event.artist
        self.original_position = (event.mouseevent.xdata, event.mouseevent.ydata)

    def _on_mouse_release(self, event):
        self.dragging_marker = None
        self.dragging_type = None
        self.original_position = None

    def _on_mouse_motion(self, event):
        if not self.dragging_marker or not event.inaxes or not self.original_position:
            return

        # Atualiza a posição do marcador
        if self.dragging_marker in self.markers['time_v'] or self.dragging_marker in self.markers['freq_v']:
            if event.xdata is not None:
                self.dragging_marker.set_xdata([event.xdata])
        else:
            if event.ydata is not None:
                self.dragging_marker.set_ydata([event.ydata])

        # Atualiza o painel de informações
        self.update_marker_panel()
        self.canvas.draw_idle()
        self.set_status("🔧 Marcador movido", "cyan")

    def add_marker(self, kind):
        ev = self.last_click_event
        if not ev or not ev.inaxes:
            return

        # Escolhe o eixo correto
        ax = self.ax_time if kind.startswith('time') else self.ax_freq
        color = 'red' if kind.startswith('time') else 'lime'

        # Limite de 2 marcadores por tipo
        if len(self.markers[kind]) >= 2:
            self.set_status("⚠️ Máximo de 2 marcadores por tipo atingido", "orange")
            return

        # Cria o marcador
        if kind.endswith('_v'):
            line = ax.axvline(x=ev.xdata, linestyle='--', color=color, picker=5)
        else:
            line = ax.axhline(y=ev.ydata, linestyle='--', color=color, picker=5)

        # Armazena o marcador
        self.markers[kind].append(line)
        self.update_marker_panel()
        self.canvas.draw_idle()
        self.set_status(f"✅ Marcador {kind} adicionado", "lightgreen")

    def clear_markers(self, which):
        keys = list(self.markers.keys()) if which == 'all' else [which]
        for k in keys:
            for mk in list(self.markers[k]):
                try:
                    # Remove fisicamente a linha do gráfico
                    if mk in self.ax_time.lines:
                        self.ax_time.lines.remove(mk)
                    elif mk in self.ax_freq.lines:
                        self.ax_freq.lines.remove(mk)
                except ValueError:
                    pass
            self.markers[k] = []

        self.update_marker_panel()
        self.canvas.draw_idle()
        self.set_status("🧹 Marcadores limpos", "yellow")

    def update_marker_panel(self):
        # --- Tempo
        tv = sorted([m.get_xdata()[0] for m in self.markers['time_v']])[:2]
        if len(tv) == 2:
            self.lbl_x1.configure(text=f"X1: {self._format_time(tv[0])}")
            self.lbl_x2.configure(text=f"X2: {self._format_time(tv[1])}")
            self.lbl_dx.configure(text=f"ΔX: {self._format_time(tv[1] - tv[0])}")
        else:
            self.lbl_x1.configure(text="X1: ---")
            self.lbl_x2.configure(text="X2: ---")
            self.lbl_dx.configure(text="ΔX: ---")

        th = sorted([m.get_ydata()[0] for m in self.markers['time_h']])[:2]
        if len(th) == 2:
            self.lbl_y1.configure(text=f"Y1: {th[0]:.3f}")
            self.lbl_y2.configure(text=f"Y2: {th[1]:.3f}")
            self.lbl_dy.configure(text=f"ΔY: {(th[1] - th[0]):.3f}")
        else:
            self.lbl_y1.configure(text="Y1: ---")
            self.lbl_y2.configure(text="Y2: ---")
            self.lbl_dy.configure(text="ΔY: ---")

        # --- Frequência
        fv = sorted([m.get_xdata()[0] for m in self.markers['freq_v']])[:2]
        if len(fv) == 2:
            self.lbl_f1.configure(text=f"F1: {self._format_freq(fv[0])}")
            self.lbl_f2.configure(text=f"F2: {self._format_freq(fv[1])}")
            self.lbl_df.configure(text=f"ΔF: {self._format_freq(fv[1] - fv[0])}")
        else:
            self.lbl_f1.configure(text="F1: ---")
            self.lbl_f2.configure(text="F2: ---")
            self.lbl_df.configure(text="ΔF: ---")

        fh = sorted([m.get_ydata()[0] for m in self.markers['freq_h']])[:2]
        if len(fh) == 2:
            self.lbl_m1.configure(text=f"|Y1|: {fh[0]:.2f}")
            self.lbl_m2.configure(text=f"|Y2|: {fh[1]:.2f}")
            self.lbl_dm.configure(text=f"Δ|Y|: {(fh[1] - fh[0]):.2f}")
        else:
            self.lbl_m1.configure(text="|Y1|: ---")
            self.lbl_m2.configure(text="|Y2|: ---")
            self.lbl_dm.configure(text="Δ|Y|: ---")

    def update_analysis_panels(self):
        if not self.last_data:
            return

        y = self.last_data['y']
        t = self.last_data['t']
        f = self.last_data['f']
        Y = self.last_data['Y']

        # Análise no domínio do tempo
        if len(y) > 0:
            # Tensão pico a pico
            vpp = np.max(y) - np.min(y)
            self.time_analysis_results['vpp'].set(f"{vpp:.4f} V")

            # Tensão RMS
            rms = np.sqrt(np.mean(y ** 2))
            self.time_analysis_results['rms'].set(f"{rms:.4f} V")

            # Tensão média (DC offset)
            mean = np.mean(y)
            self.time_analysis_results['mean'].set(f"{mean:.4f} V")

            # Fator de crista (Crest Factor)
            crest_factor = np.max(np.abs(y)) / rms if rms > 0 else 0
            self.time_analysis_results['crest_factor'].set(f"{crest_factor:.4f}")

            # Taxa de cruzamento por zero
            zero_crossings = np.where(np.diff(np.sign(y)))[0]
            if len(zero_crossings) > 0:
                zero_crossing_rate = len(zero_crossings) / (t[-1] - t[0])
                self.time_analysis_results['zero_crossing'].set(f"{zero_crossing_rate:.2f} Hz")
            else:
                self.time_analysis_results['zero_crossing'].set("0 Hz")

            # Frequência estimada
            if len(zero_crossings) > 1:
                periods = np.diff(t[zero_crossings])
                avg_period = np.mean(periods) * 2
                freq_est = 1 / avg_period if avg_period > 0 else 0
                self.time_analysis_results['frequency'].set(f"{freq_est:.2f} Hz")
            else:
                self.time_analysis_results['frequency'].set("---")

            # Duty cycle (apenas para ondas quadradas)
            if self.waveform.get().startswith("Quadrada") or self.waveform.get().startswith("Pulso"):
                if vpp > 0:
                    threshold = (np.max(y) + np.min(y)) / 2
                    positive_samples = np.sum(y > threshold)
                    duty_cycle = positive_samples / len(y) * 100
                    self.time_analysis_results['duty_cycle'].set(f"{duty_cycle:.1f}%")
                else:
                    self.time_analysis_results['duty_cycle'].set("N/A")
            else:
                self.time_analysis_results['duty_cycle'].set("N/A")

            # Relação Pico/RMS
            peak_to_rms = np.max(np.abs(y)) / rms if rms > 0 else 0
            self.time_analysis_results['peak_to_rms'].set(f"{peak_to_rms:.4f}")

            # Curtose
            if len(y) > 3:
                kurt_val = kurtosis(y)
                self.time_analysis_results['kurtosis'].set(f"{kurt_val:.4f}")
            else:
                self.time_analysis_results['kurtosis'].set("---")

            # Assimetria (Skewness)
            if len(y) > 2:
                skew_val = skew(y)
                self.time_analysis_results['skewness'].set(f"{skew_val:.4f}")
            else:
                self.time_analysis_results['skewness'].set("---")

        # Análise no domínio da frequência
        if len(Y) > 0:
            # Filtra apenas frequências positivas
            positive_mask = f > 0
            f_positive = f[positive_mask]
            Y_positive = Y[positive_mask]

            # Encontra a frequência fundamental (maior magnitude)
            fundamental_idx = np.argmax(Y_positive)
            fundamental_freq = f_positive[fundamental_idx]
            fundamental_amp = Y_positive[fundamental_idx]
            self.freq_analysis_results['fundamental'].set(f"{self._format_freq(fundamental_freq)}")
            self.freq_analysis_results['fund_amp'].set(f"{fundamental_amp:.4f}")

            # Encontra harmônicos (excluindo o fundamental)
            harmonic_mask = (f_positive > fundamental_freq * 0.9) & (f_positive < f_positive[-1])
            harmonic_freqs = f_positive[harmonic_mask]
            harmonic_amps = Y_positive[harmonic_mask]

            # Identifica picos significativos
            peaks, _ = find_peaks(harmonic_amps, height=fundamental_amp * 0.05)
            harmonic_peaks = peaks[np.argsort(harmonic_amps[peaks])[::-1]]

            # Calcula THD (Total Harmonic Distortion)
            if len(harmonic_peaks) > 0:
                harmonic_power = np.sum(harmonic_amps[harmonic_peaks] ** 2)
                fundamental_power = fundamental_amp ** 2

                if fundamental_power > 0:
                    thd = np.sqrt(harmonic_power / fundamental_power) * 100
                    self.freq_analysis_results['thd'].set(f"{thd:.2f}%")

                    # Nível de harmônicos
                    harmonics_level = harmonic_power / fundamental_power
                    self.freq_analysis_results['harmonics'].set(f"{harmonics_level:.4f}")
                else:
                    self.freq_analysis_results['thd'].set("0%")
                    self.freq_analysis_results['harmonics'].set("0")
            else:
                self.freq_analysis_results['thd'].set("0%")
                self.freq_analysis_results['harmonics'].set("0")

            # SNR (Signal to Noise Ratio)
            if fundamental_amp > 0:
                # Considera tudo que não é fundamental ou harmônicos como ruído
                noise_mask = (f_positive > 0) & ~harmonic_mask
                noise_power = np.sum(Y_positive[noise_mask] ** 2)

                if noise_power > 0:
                    snr = 10 * np.log10(fundamental_amp ** 2 / noise_power)
                    self.freq_analysis_results['snr'].set(f"{snr:.2f} dB")
                    self.freq_analysis_results['noise_floor'].set(f"{np.sqrt(noise_power):.4f}")
                else:
                    self.freq_analysis_results['snr'].set("∞ dB")
                    self.freq_analysis_results['noise_floor'].set("---")
            else:
                self.freq_analysis_results['snr'].set("---")
                self.freq_analysis_results['noise_floor'].set("---")

            # SFDR (Spurious Free Dynamic Range)
            if len(harmonic_peaks) > 0:
                max_spur = np.max(harmonic_amps[harmonic_peaks])
                if fundamental_amp > 0 and max_spur > 0:
                    sfdr = 20 * np.log10(fundamental_amp / max_spur)
                    self.freq_analysis_results['sfdr'].set(f"{sfdr:.2f} dB")
                else:
                    self.freq_analysis_results['sfdr'].set("∞ dB")
            else:
                self.freq_analysis_results['sfdr'].set("∞ dB")

            # Largura de banda a -3dB
            if fundamental_amp > 0:
                half_power = fundamental_amp / np.sqrt(2)
                above_half = Y_positive >= half_power
                if np.any(above_half):
                    min_freq = f_positive[above_half][0]
                    max_freq = f_positive[above_half][-1]
                    bandwidth = max_freq - min_freq
                    self.freq_analysis_results['bandwidth'].set(f"{self._format_freq(bandwidth)}")
                else:
                    self.freq_analysis_results['bandwidth'].set("---")
            else:
                self.freq_analysis_results['bandwidth'].set("---")

            # Índice de modulação (estimado)
            if self.mod_am.get():
                # Para AM: m = (A_max - A_min) / (A_max + A_min)
                A_max = np.max(y)
                A_min = np.min(y)
                if A_max + A_min != 0:
                    mod_index = (A_max - A_min) / (A_max + A_min) * 100
                    self.freq_analysis_results['mod_index'].set(f"{mod_index:.1f}%")
                else:
                    self.freq_analysis_results['mod_index'].set("---")
            elif self.mod_fm.get():
                # Para FM: β = Δf / f_m
                # Estimativa usando largura de banda
                if fundamental_amp > 0:
                    sideband_mask = (f_positive > fundamental_freq - 10) & (f_positive < fundamental_freq + 10)
                    sideband_amp = np.max(Y_positive[sideband_mask])
                    if fundamental_amp > 0:
                        mod_index = sideband_amp / fundamental_amp * 100
                        self.freq_analysis_results['mod_index'].set(f"{mod_index:.1f}%")
                    else:
                        self.freq_analysis_results['mod_index'].set("---")
                else:
                    self.freq_analysis_results['mod_index'].set("---")
            else:
                self.freq_analysis_results['mod_index'].set("N/A")

            # Frequência de pico (maior harmônico)
            if len(harmonic_peaks) > 0:
                peak_harmonic_idx = harmonic_peaks[0]
                peak_freq = harmonic_freqs[peak_harmonic_idx]
                self.freq_analysis_results['peak_freq'].set(f"{self._format_freq(peak_freq)}")
            else:
                self.freq_analysis_results['peak_freq'].set("---")

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