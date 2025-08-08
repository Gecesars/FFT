import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
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
import logging
import time

# Configurar logging com mais detalhes
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("signal_generator.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SignalGenerator")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Constantes
INTERP_THRESHOLD = 100
INTERP_SAMPLES = 500
UNIT_MULTIPLIERS = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
WAV_TOTAL_SIZE = 15360  # Tamanho total do arquivo para compatibilidade
ANALYSIS_MAX_POINTS = 1000000  # Máximo de pontos para análise

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
        self.geometry("1700x1000")
        self.logger = logging.getLogger("SignalGenerator")
        self.logger.info("Aplicativo iniciado")

        # Configurar layout principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Adicionar barra de menu
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)

        # Menu Arquivo
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Salvar Configuração", command=self.save_config)
        file_menu.add_command(label="Carregar Configuração", command=self.load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exportar Dados", command=self.export_data)
        file_menu.add_command(label="Exportar WAV", command=self.export_wav)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self._on_closing)
        self.menu_bar.add_cascade(label="Arquivo", menu=file_menu)

        # Menu Ferramentas
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        tools_menu.add_command(label="Importar Forma de Onda (WAV)",
                               command=self.import_wav)  # CORREÇÃO: Adicionado de volta
        self.menu_bar.add_cascade(label="Ferramentas", menu=tools_menu)  # CORREÇÃO: Adicionado de volta

        # Menu Ajuda
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="Sobre", command=self.show_about)
        self.menu_bar.add_cascade(label="Ajuda", menu=help_menu)

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
        self.imported_voltage_scale = [1.0, "V", 1]  # Armazena a escala de tensão importada
        self.imported_time_scale = [1.0, "S", 1]  # Armazena a escala de tempo importada

        self.mod_am = tk.BooleanVar(value=False)
        self.mod_fm = tk.BooleanVar(value=False)

        # Dicionários para armazenar os rótulos de análise
        self.time_analysis_labels = {}
        self.freq_analysis_labels = {}

        self._build_sidebar()
        self._build_plot_area()
        self._build_context_menu()
        self._build_marker_panel()
        self._build_analysis_panels()
        self._build_status_bar()

        # Configurar tratamento de fechamento
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def show_about(self):
        """Mostra janela 'Sobre' com informações do aplicativo"""
        about = ctk.CTkToplevel(self)
        about.title("Sobre o Gerador de Sinais Avançado")
        about.geometry("500x400")
        about.resizable(False, False)
        about.transient(self)
        about.grab_set()

        # Frame principal
        main_frame = ctk.CTkFrame(about)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Título
        ctk.CTkLabel(
            main_frame,
            text="Gerador de Sinais Avançado",
            font=("Arial", 20, "bold")
        ).pack(pady=(10, 5))

        # Versão
        ctk.CTkLabel(
            main_frame,
            text="Versão 2.0",
            font=("Arial", 14)
        ).pack(pady=(0, 20))

        # Descrição
        description = (
            "Este aplicativo permite gerar, analisar e exportar sinais elétricos\n"
            "complexos com diversas formas de onda e modulações.\n\n"
            "Recursos principais:\n"
            "- Geração de mais de 30 formas de onda diferentes\n"
            "- Modulação AM e FM\n"
            "- Análise de tempo e frequência\n"
            "- Exportação para WAV (compatível com osciloscópios)\n"
            "- Ferramentas de medição com marcadores\n\n"
            "Desenvolvido como projeto de processamento de sinais."
        )
        ctk.CTkLabel(
            main_frame,
            text=description,
            justify="left"
        ).pack(pady=10, padx=20)

        # Autor
        ctk.CTkLabel(
            main_frame,
            text="Desenvolvido por: [Seu Nome]",
            font=("Arial", 12, "italic")
        ).pack(pady=(20, 5))

        # Botão Fechar
        ctk.CTkButton(
            main_frame,
            text="Fechar",
            command=about.destroy
        ).pack(pady=20)

    def save_config(self):
        """Salva a configuração atual em arquivo JSON"""
        config = {
            'duration': self.entry_duration.get(),
            'Fc': self.entry_fc.get(),
            'Fc_units': self.units_fc.get(),
            'Fs': self.entry_fs.get(),
            'Fs_units': self.units_fs.get(),
            'vpp': self.entry_vpp.get(),
            'waveform': self.waveform.get(),
            'Fm': self.entry_fm.get(),
            'Fm_units': self.units_fm.get(),
            'am_on': self.mod_am.get(),
            'am_depth': self.slider_am.get(),
            'fm_on': self.mod_fm.get(),
            'fm_dev': self.slider_fm_dev.get()
        }

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            self.set_status(f"✅ Configuração salva em: {os.path.basename(filepath)}", "lightgreen")
        except Exception as e:
            self.set_status(f"❌ Erro ao salvar configuração: {str(e)}", "red")
            messagebox.showerror("Erro", f"Falha ao salvar configuração:\n{str(e)}")

    def load_config(self):
        """Carrega configuração de arquivo JSON"""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                config = json.load(f)

            # Atualizar interface com os valores carregados
            self.entry_duration.delete(0, tk.END)
            self.entry_duration.insert(0, config['duration'])

            self.entry_fc.delete(0, tk.END)
            self.entry_fc.insert(0, config['Fc'])
            self.units_fc.set(config['Fc_units'])

            self.entry_fs.delete(0, tk.END)
            self.entry_fs.insert(0, config['Fs'])
            self.units_fs.set(config['Fs_units'])

            self.entry_vpp.delete(0, tk.END)
            self.entry_vpp.insert(0, config['vpp'])

            self.waveform.set(config['waveform'])

            self.entry_fm.delete(0, tk.END)
            self.entry_fm.insert(0, config['Fm'])
            self.units_fm.set(config['Fm_units'])

            self.mod_am.set(config['am_on'])
            self.slider_am.set(config['am_depth'])

            self.mod_fm.set(config['fm_on'])
            self.slider_fm_dev.set(config['fm_dev'])

            # Atualizar estados dos sliders
            self._on_am_fm_toggle()

            self.set_status(f"✅ Configuração carregada: {os.path.basename(filepath)}", "lightgreen")
            self.submit_plot_task()  # Gerar sinal com nova configuração
        except Exception as e:
            self.set_status(f"❌ Erro ao carregar configuração: {str(e)}", "red")
            messagebox.showerror("Erro", f"Falha ao carregar configuração:\n{str(e)}")

    def _on_closing(self):
        """Cancelar callbacks pendentes ao fechar a janela"""
        for after_id in self.after_ids:
            self.after_cancel(after_id)
        self.logger.info("Aplicativo encerrado")
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

        # Frame para gráficos e sliders
        main_plot_frame = ctk.CTkFrame(plot_frame)
        main_plot_frame.pack(fill="both", expand=True, padx=0, pady=0)
        main_plot_frame.grid_columnconfigure(0, weight=1)
        main_plot_frame.grid_rowconfigure(0, weight=1)
        main_plot_frame.grid_rowconfigure(1, weight=0)

        # Frame para gráficos
        graph_frame = ctk.CTkFrame(main_plot_frame)
        graph_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        graph_frame.grid_propagate(False)
        graph_frame.grid_columnconfigure(0, weight=1)
        graph_frame.grid_rowconfigure(0, weight=1)

        # Criação dos gráficos
        self.fig, (self.ax_time, self.ax_freq) = plt.subplots(2, 1, facecolor="#2B2B2B", figsize=(8, 6))
        self.ax_time.set_facecolor("#3C3C3C")
        self.ax_freq.set_facecolor("#3C3C3C")
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)

        # Frame para sliders de escala Y (agora ao lado dos gráficos)
        y_scale_frame = ctk.CTkFrame(main_plot_frame, width=40)
        y_scale_frame.grid(row=0, column=1, sticky="ns", padx=(5, 0), pady=0)

        # Sliders de escala Y
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

        # Frame para controles de zoom
        ctrl_frame = ctk.CTkFrame(plot_frame, height=40)
        ctrl_frame.pack(fill="x", pady=(0, 5))

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
            # Converter para mV para escala
            vpp_mv = vpp * 1000
            self.ax_time.set_ylim(-vpp_mv / 2 * float(value), vpp_mv / 2 * float(value))
            self._update_voltage_ticks()
            self.canvas.draw_idle()
        except Exception as e:
            self.logger.error(f"Erro ao atualizar escala de tempo: {str(e)}")

    def update_freq_scale(self, value):
        """Atualiza a escala do eixo Y no gráfico de frequência"""
        if not self.last_data:
            return

        try:
            scale_factor = float(value)
            max_val = np.max(self.last_data['Y']) * 1.1 * scale_factor
            self.ax_freq.set_ylim(0, max_val)
            self.canvas.draw_idle()
        except Exception as e:
            self.logger.error(f"Erro ao atualizar escala de frequência: {str(e)}")

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
        # Container à direita da área dos gráficos - AUMENTADO para 400px
        self.side_panel = ctk.CTkFrame(self, width=400, corner_radius=6)
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

        # Lista de métricas para tempo
        metrics = [
            ("Tensão Pico a Pico (Vpp):", 'vpp'),
            ("Tensão RMS:", 'rms'),
            ("Tensão Média (DC):", 'mean'),
            ("Fator de Crista:", 'crest_factor'),
            ("Taxa de Cruzamento por Zero:", 'zero_crossing'),
            ("Frequência Estimada:", 'frequency'),
            ("Ciclo de Trabalho (Duty Cycle):", 'duty_cycle'),
            ("Relação Pico/RMS:", 'peak_to_rms'),
            ("Curtose:", 'kurtosis'),
            ("Assimetria (Skewness):", 'skewness')
        ]

        for label_text, key in metrics:
            frame = ctk.CTkFrame(time_frame, fg_color="transparent")
            frame.pack(fill="x", padx=5, pady=2)
            lbl = ctk.CTkLabel(frame, text=label_text, width=250, anchor="w")
            lbl.pack(side="left", anchor="w")
            val = ctk.CTkLabel(frame, text="---", width=120, anchor="e")
            val.pack(side="right", anchor="e")
            self.time_analysis_labels[key] = val  # Armazena a referência do label

        # --- Painel de Análise de Frequência ---
        freq_frame = ctk.CTkScrollableFrame(self.freq_analysis_tab)
        freq_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Título
        ctk.CTkLabel(freq_frame, text="Análise do Sinal na Frequência",
                     font=("Arial", 14, "bold"), text_color="orange").pack(anchor="w", pady=(0, 10))

        # Lista de métricas para frequência
        metrics = [
            ("Frequência Fundamental:", 'fundamental'),
            ("Amplitude Fundamental:", 'fund_amp'),
            ("THD (Distorção Harmônica):", 'thd'),
            ("SNR (Relação Sinal-Ruído):", 'snr'),
            ("SFDR (Faixa Dinâmica):", 'sfdr'),
            ("Largura de Banda:", 'bandwidth'),
            ("Índice de Modulação:", 'mod_index'),
            ("Nível de Harmônicos:", 'harmonics'),
            ("Piso de Ruído:", 'noise_floor'),
            ("Frequência de Pico:", 'peak_freq')
        ]

        for label_text, key in metrics:
            frame = ctk.CTkFrame(freq_frame, fg_color="transparent")
            frame.pack(fill="x", padx=5, pady=2)
            lbl = ctk.CTkLabel(frame, text=label_text, width=250, anchor="w")
            lbl.pack(side="left", anchor="w")
            val = ctk.CTkLabel(frame, text="---", width=120, anchor="e")
            val.pack(side="right", anchor="e")
            self.freq_analysis_labels[key] = val  # Armazena a referência do label

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
        self.logger.info("Submetendo tarefa de plotagem")
        self.btn_generate.configure(state="disabled", text="Gerando...")
        self.set_status("⏳ Iniciando geração de sinal...", "yellow")
        future = self.executor.submit(self._compute_and_plot_task)
        future.add_done_callback(
            lambda future: self.after(0, lambda: self.btn_generate.configure(state="normal", text="Gerar Sinal")))

    def _compute_and_plot_task(self):
        self.logger.info("Iniciando cálculo e plotagem de sinal")
        start_time = time.time()
        try:
            params = self._validate_inputs()
            if not params:
                self.set_status("❌ Erro ao gerar o sinal", "red")
                self.logger.warning("Validação falhou, parâmetros inválidos")
                return

            # gera eixo tempo e sinal
            t = np.arange(params['N']) / params['Fs']
            y = self._generate_waveform(params, t)

            # Aplica amplitude
            try:
                vpp = float(self.entry_vpp.get())
                y = y * (vpp / 2)  # Normaliza para Vpp
                self.logger.debug(f"Amplitude aplicada: Vpp={vpp}")
            except ValueError:
                self.logger.warning("Amplitude inválida, usando padrão")
                y = y  # Mantém amplitude padrão se entrada inválida

            # Aplica modulações apenas se não forem formas de onda pré-moduladas
            if not params['waveform'].startswith("Onda AM") and not params['waveform'].startswith("Onda FM"):
                y = self._apply_modulation(params, y, t)
                self.logger.debug("Modulação aplicada")

            # FFT com zero no centro
            self.logger.info("Calculando FFT")

            # Limitar pontos da FFT para evitar processamento excessivo
            N_fft = min(params['N'], ANALYSIS_MAX_POINTS)
            Y = fft(y, n=N_fft)
            f = fftfreq(N_fft, 1 / params['Fs'])

            # Centraliza o zero
            Y_shifted = fftshift(Y)
            f_shifted = fftshift(f)
            Y_abs = np.abs(Y_shifted)

            # guarda os dados para plot
            self.last_data = {'t': t, 'y': y, 'f': f_shifted, 'Y': Y_abs}

            # executa atualização de plot na thread principal
            self.after(0, self._update_plots)
            self.after(0, self.update_analysis_panels)
            self.logger.info(f"Cálculo do sinal concluído com sucesso em {time.time() - start_time:.3f}s")

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Erro no cálculo do sinal: {error_msg}", exc_info=True)
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

            # Verificação de pontos por ciclo para alta frequência
            if p['Fc'] > 1e6:  # Frequências acima de 1MHz
                points_per_cycle = p['Fs'] / p['Fc']
                min_points = 20  # Mínimo de pontos por ciclo para boa visualização

                if points_per_cycle < min_points:
                    # Ajustar duração para ter pelo menos 10 ciclos
                    required_cycles = 10
                    required_duration = required_cycles / p['Fc']

                    # Ajustar Fs se necessário
                    required_fs = max(p['Fs'], p['Fc'] * min_points)

                    if required_fs > p['Fs']:
                        p['Fs'] = required_fs
                        self.logger.info(f"Fs ajustado para {p['Fs']} Hz para {min_points} pontos por ciclo")

                    p['duration'] = max(p['duration'], required_duration)
                    p['N'] = int(p['duration'] * p['Fs'])
                    self.logger.info(f"Duração ajustada para {p['duration']} s com {p['N']} pontos")

                    # Atualizar campos na interface
                    self.after(0, lambda: self.entry_duration.delete(0, tk.END))
                    self.after(0, lambda: self.entry_duration.insert(0, str(p['duration'])))
                    self.after(0, lambda: self.entry_fs.delete(0, tk.END))
                    self.after(0, lambda: self.entry_fs.insert(0, str(p['Fs'] / UNIT_MULTIPLIERS[self.units_fs.get()])))
                    self.set_status("⚠️ Parâmetros ajustados para melhor visualização", "yellow")

            self.logger.debug(f"Parâmetros validados: {p}")
            return p

        except ValueError as e:
            self.logger.error(f"Erro de validação: {str(e)}")
            self.after(0, lambda: messagebox.showerror("Erro de Entrada", str(e)))
            self.set_status(f"❌ Erro: {str(e)}", "red")
            return None

    def _generate_waveform(self, p, t):
        wf, Fc, Fm = p['waveform'], p['Fc'], p['Fm']
        duration = p['duration']
        self.logger.debug(f"Gerando forma de onda: {wf}, Fc={Fc}, Fm={Fm}")

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
            self.logger.warning(f"Forma de onda desconhecida: {wf}, retornando zero")
            return np.zeros_like(t)

    def _apply_modulation(self, p, y, t):
        # Aplicação correta de AM e FM
        if p['am_on']:
            modulator = np.sin(2 * np.pi * p['Fm'] * t)
            y = y * (1 + p['am_depth'] * modulator)
            self.logger.debug("Modulação AM aplicada")

        if p['fm_on']:
            # Para FM, usamos a integral do sinal modulador
            phase = 2 * np.pi * p['Fc'] * t + 2 * np.pi * p['fm_dev'] * np.cumsum(np.sin(2 * np.pi * p['Fm'] * t)) * (
                    1 / p['Fs'])
            y = np.sin(phase)
            self.logger.debug("Modulação FM aplicada")

        return y

    def _read_wav_file(self, filepath):
        """Lê o arquivo WAV e retorna o cabeçalho e buffers de dados"""
        self.logger.info(f"Lendo arquivo WAV: {filepath}")
        with open(filepath, 'rb') as f:
            header = f.read(208)  # Cabeçalho de 208 bytes

            # Posiciona e lê os dados do canal 1
            f.seek(1000)
            data_ch1 = f.read(3000)

            # Posiciona e lê os dados do canal 2
            f.seek(4000)
            data_ch2 = f.read(3000)

        return header, data_ch1, data_ch2

    def _parse_header(self, header_bytes):
        """Decodifica o cabeçalho do WAV do Fnirsi"""
        self.logger.info("Analisando cabeçalho WAV")
        try:
            volt_scale = []
            for x in range(2):  # Para cada canal
                # Índice da escala de tensão (bytes 4 e 14)
                scale_idx = header_bytes[4 + x * 10]
                if scale_idx < len(VOLT_LIST):
                    scale = VOLT_LIST[scale_idx]
                    self.logger.debug(f"Escala de tensão encontrada: {scale}")
                else:
                    self.logger.warning(f"Índice de tensão inválido: {scale_idx}, usando padrão")
                    scale = VOLT_LIST[0]  # Default se índice inválido
                volt_scale.append(scale)

            # Índice da escala de tempo (byte 22)
            time_idx = header_bytes[22]
            if time_idx < len(TIME_LIST):
                ts = TIME_LIST[time_idx]
                self.logger.debug(f"Escala de tempo encontrada: {ts}")
            else:
                self.logger.warning(f"Índice de tempo inválido: {time_idx}, usando padrão")
                ts = TIME_LIST[0]  # Default se índice inválido

            return volt_scale, ts
        except Exception as e:
            self.logger.error("Erro ao analisar cabeçalho", exc_info=True)
            raise ValueError("Formato de cabeçalho WAV inválido") from e

    def _parse_channel_data(self, data_bytes, scale):
        """Decodifica os dados de um canal específico"""
        self.logger.info("Decodificando dados do canal")
        try:
            values = []
            # Verifica se temos dados suficientes
            if len(data_bytes) < 3000:
                self.logger.warning(f"Buffer de dados pequeno: {len(data_bytes)} bytes, esperado 3000")
                # Preenche com zeros se não houver dados suficientes
                data_bytes = data_bytes + b'\x00' * (3000 - len(data_bytes))

            for i in range(1500):  # 1500 amostras por canal
                # Cada amostra são 2 bytes (little-endian)
                byte1 = data_bytes[i * 2]
                byte2 = data_bytes[i * 2 + 1]

                # Converte para valor numérico
                raw_val = byte1 + 256 * byte2
                # Verificação de valores inválidos
                if raw_val < 0 or raw_val > 65535:
                    self.logger.warning(f"Valor bruto inválido na posição {i}: {raw_val}")
                    raw_val = 200  # Valor padrão seguro

                val = (raw_val - 200) * scale[0] / 50.0
                values.append(val)

            return values
        except Exception as e:
            self.logger.error("Erro ao decodificar dados do canal", exc_info=True)
            raise ValueError("Erro ao analisar dados do canal") from e

    def _show_wav_preview(self, t, y, filename):
        """Mostra uma prévia do sinal WAV em uma janela modal com análises e marcadores"""
        self.logger.info(f"Mostrando prévia do WAV: {filename}")
        preview = ctk.CTkToplevel(self)
        preview.title(f"Visualização do Sinal: {filename}")
        preview.geometry("1200x800")
        preview.transient(self)
        preview.grab_set()

        # Configurar layout
        preview.grid_columnconfigure(0, weight=1)
        preview.grid_rowconfigure(0, weight=1)
        preview.grid_rowconfigure(1, weight=0)

        # Frame para gráfico
        graph_frame = ctk.CTkFrame(preview)
        graph_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Gráfico
        fig = plt.Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)

        # CORREÇÃO: Converter para array antes de multiplicar
        y_mV = np.array(y) * 1000  # Converter para mV
        line, = ax.plot(t, y_mV, 'b-')

        ax.set_title(f"Forma de Onda: {filename}")
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Adicionar toolbar de navegação
        toolbar = NavigationToolbar2Tk(canvas, graph_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Frame para controles
        ctrl_frame = ctk.CTkFrame(preview)
        ctrl_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Botão para abrir janela de amostragem
        ctk.CTkButton(ctrl_frame, text="Amostrar Sinal",
                      command=lambda: self._open_sampling_window(preview, t, y, filename)).pack(side="left", padx=5)

        # Botão para exportar
        ctk.CTkButton(ctrl_frame, text="Exportar WAV",
                      command=lambda: self._export_from_preview(preview, t, y, filename)).pack(side="right", padx=5)

        # Botão para fechar
        ctk.CTkButton(ctrl_frame, text="Fechar", command=preview.destroy).pack(side="right", padx=5)

        # Adicionar painel de análises
        analysis_frame = ctk.CTkFrame(preview)
        analysis_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Abas para análises
        analysis_notebook = ctk.CTkTabview(analysis_frame, width=400)
        analysis_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Análise de tempo
        time_tab = analysis_notebook.add("Análise de Tempo")
        self._build_analysis_panel(time_tab, t, y, "time")

        # Análise de frequência
        freq_tab = analysis_notebook.add("Análise de Frequência")  # CORREÇÃO: Nome da aba
        f, Y = self._calculate_fft(t, y)
        self._build_analysis_panel(freq_tab, f, Y, "freq")  # CORREÇÃO: Passar para análise de frequência

    def _open_sampling_window(self, parent, t, y, filename):
        """Abre janela de amostragem interativa"""
        self.logger.info(f"Abrindo janela de amostragem para: {filename}")
        sampling_win = ctk.CTkToplevel(parent)
        sampling_win.title(f"Amostragem do Sinal: {filename}")
        sampling_win.geometry("1000x700")
        sampling_win.transient(parent)
        sampling_win.grab_set()

        # Configurar layout
        sampling_win.grid_columnconfigure(0, weight=1)
        sampling_win.grid_rowconfigure(0, weight=1)
        sampling_win.grid_rowconfigure(1, weight=0)

        # Frame para gráfico
        graph_frame = ctk.CTkFrame(sampling_win)
        graph_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Gráfico
        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)

        # CORREÇÃO: Converter para array antes de multiplicar
        y_mV = np.array(y) * 1000  # Converter para mV
        ax.plot(t, y_mV, 'b-', label="Original")

        # Amostrar o sinal
        n_samples = 1500  # Número fixo de amostras para compatibilidade
        sample_points = np.linspace(t[0], t[-1], n_samples)
        sample_values = np.interp(sample_points, t, y) * 1000  # Converter para mV
        line, = ax.plot(sample_points, sample_values, 'ro-', label="Amostrado", picker=5)

        ax.set_title(f"Sinal Amostrado: {filename}")
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.grid(True)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Variáveis de estado
        self.sampling_state = {
            't_original': t,
            'y_original': y,
            'sample_points': sample_points,
            'sample_values': sample_values / 1000,  # Armazenar em V
            'line': line,
            'canvas': canvas,
            'ax': ax,
            'fig': fig,
            'dragging': False,
            'dragged_point': None
        }

        # Evento de seleção de pontos
        canvas.mpl_connect('pick_event', self._on_sample_pick)
        canvas.mpl_connect('button_release_event', self._on_sample_release)
        canvas.mpl_connect('motion_notify_event', self._on_sample_motion)

        # Frame para controles
        ctrl_frame = ctk.CTkFrame(sampling_win)
        ctrl_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Botão para exportar
        ctk.CTkButton(ctrl_frame, text="Exportar WAV",
                      command=lambda: self._export_sampled_wav(
                          self.sampling_state['sample_points'],
                          self.sampling_state['sample_values'],
                          filename
                      )).pack(side="right", padx=5)

    def _on_sample_pick(self, event):
        """Manipula a seleção de pontos amostrados"""
        if event.artist != self.sampling_state['line']:
            return

        # Indice do ponto selecionado
        ind = event.ind[0]
        self.sampling_state['dragging'] = True
        self.sampling_state['dragged_point'] = ind

    def _on_sample_release(self, event):
        """Libera o ponto arrastado"""
        self.sampling_state['dragging'] = False
        self.sampling_state['dragged_point'] = None

    def _on_sample_motion(self, event):
        """Arrasta o ponto selecionado"""
        if not self.sampling_state['dragging'] or event.xdata is None or event.ydata is None:
            return

        state = self.sampling_state
        idx = state['dragged_point']

        # Atualiza o valor do ponto
        state['sample_values'][idx] = event.ydata / 1000  # Converter para V

        # Se Shift estiver pressionado, restringe ao eixo Y
        if not event.key == 'shift':
            state['sample_points'][idx] = event.xdata

        # Atualiza o gráfico
        state['line'].set_data(state['sample_points'], state['sample_values'] * 1000)  # Plot em mV
        state['canvas'].draw()

    def _export_sampled_wav(self, t, y, filename):
        """Exporta o sinal amostrado em formato WAV compatível"""
        self.logger.info(f"Exportando WAV amostrado: {filename}")
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialfile=os.path.splitext(filename)[0] + "_amostrado.wav"
        )
        if not filepath:  # Usuário cancelou
            self.logger.info("Exportação de WAV cancelada pelo usuário")
            return

        try:
            # Usa a escala de tensão original importada
            scale_value = self.imported_voltage_scale[0]
            scale_unit = self.imported_voltage_scale[1]

            # Converte os dados para o formato do osciloscópio
            data_bytes = bytearray()
            for value in y:
                # Fórmula inversa: raw = (value * 50 / scale_value) + 200
                raw_value = (value * 50.0 / scale_value) + 200
                int_value = int(raw_value)
                # Limitar o valor ao intervalo [0, 65535]
                int_value = max(0, min(int_value, 65535))
                # Divide em dois bytes (little-endian)
                data_bytes.append(int_value & 0xFF)
                data_bytes.append((int_value >> 8) & 0xFF)

            # Cria o cabeçalho
            header = bytearray(208)
            # Índices de escala de tensão (canais 1 e 2)
            if hasattr(self, 'imported_voltage_scale'):
                header[4] = VOLT_LIST.index(self.imported_voltage_scale)
                header[14] = VOLT_LIST.index(self.imported_voltage_scale)
            # Índice de escala de tempo
            if hasattr(self, 'imported_time_scale'):
                header[22] = TIME_LIST.index(self.imported_time_scale)

            # Constrói o arquivo completo
            file_data = bytearray(WAV_TOTAL_SIZE)
            file_data[0:208] = header
            # Canal 1: 1500 amostras (3000 bytes) a partir do byte 1000
            start_index = 1000
            end_index = start_index + len(data_bytes)
            if end_index <= WAV_TOTAL_SIZE:
                file_data[start_index:end_index] = data_bytes
            else:
                raise ValueError("Dados excedem o tamanho máximo do arquivo WAV")

            # Escreve o arquivo
            with open(filepath, 'wb') as f:
                f.write(file_data)

            self.set_status(f"📁 Sinal amostrado exportado como WAV: {os.path.basename(filepath)}", "lightblue")
            self.logger.info(f"WAV amostrado exportado: {filepath}")

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Erro ao exportar WAV amostrado: {error_msg}", exc_info=True)
            messagebox.showerror("Erro ao exportar WAV", error_msg)
            self.set_status(f"❌ Erro ao exportar sinal amostrado: {error_msg}", "red")

    def _export_from_preview(self, parent, t, y, filename):
        """Exporta o sinal da prévia como WAV"""
        self.logger.info(f"Exportando WAV da prévia: {filename}")
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialfile=filename
        )
        if not filepath:  # Usuário cancelou
            self.logger.info("Exportação de WAV cancelada pelo usuário")
            return

        try:
            # Usa a escala de tensão original importada
            scale_value = self.imported_voltage_scale[0]
            scale_unit = self.imported_voltage_scale[1]

            # Converte os dados para o formato do osciloscópio
            data_bytes = bytearray()
            for value in y:
                # Fórmula inversa: raw = (value * 50 / scale_value) + 200
                raw_value = (value * 50.0 / scale_value) + 200
                int_value = int(raw_value)
                # Limitar o valor ao intervalo [0, 65535]
                int_value = max(0, min(int_value, 65535))
                # Divide em dois bytes (little-endian)
                data_bytes.append(int_value & 0xFF)
                data_bytes.append((int_value >> 8) & 0xFF)

            # Cria o cabeçalho
            header = bytearray(208)
            # Índices de escala de tensão (canais 1 e 2)
            if hasattr(self, 'imported_voltage_scale'):
                header[4] = VOLT_LIST.index(self.imported_voltage_scale)
                header[14] = VOLT_LIST.index(self.imported_voltage_scale)
            # Índice de escala de tempo
            if hasattr(self, 'imported_time_scale'):
                header[22] = TIME_LIST.index(self.imported_time_scale)

            # Constrói o arquivo completo
            file_data = bytearray(WAV_TOTAL_SIZE)
            file_data[0:208] = header
            # Canal 1: 1500 amostras (3000 bytes) a partir do byte 1000
            start_index = 1000
            end_index = start_index + len(data_bytes)
            if end_index <= WAV_TOTAL_SIZE:
                file_data[start_index:end_index] = data_bytes
            else:
                raise ValueError("Dados excedem o tamanho máximo do arquivo WAV")

            # Escreve o arquivo
            with open(filepath, 'wb') as f:
                f.write(file_data)

            self.set_status(f"📁 Sinal exportado como WAV: {os.path.basename(filepath)}", "lightblue")
            self.logger.info(f"WAV exportado: {filepath}")

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Erro ao exportar WAV: {error_msg}", exc_info=True)
            messagebox.showerror("Erro ao exportar WAV", error_msg)
            self.set_status(f"❌ Erro ao exportar: {error_msg}", "red")

    def _calculate_fft(self, t, y):
        """Calcula a FFT para o sinal"""
        N = len(y)
        Fs = 1 / (t[1] - t[0]) if len(t) > 1 else 1  # Frequência de amostragem estimada
        Y = fftshift(fft(y))
        f = fftshift(fftfreq(N, 1 / Fs))
        return f, np.abs(Y)

    def _build_analysis_panel(self, parent, x, y, analysis_type):
        """Constrói painel de análise para a janela de prévia"""
        scroll_frame = ctk.CTkScrollableFrame(parent)
        scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

        if analysis_type == "time":
            # Análise no domínio do tempo
            metrics = self._calculate_time_analysis(x, y)
            title = "Análise do Tempo"
        else:
            # Análise no domínio da frequência
            metrics = self._calculate_freq_analysis(x, y)
            title = "Análise de Frequência"

        ctk.CTkLabel(scroll_frame, text=title, font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))

        for label, value in metrics:
            frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            frame.pack(fill="x", padx=5, pady=2)
            ctk.CTkLabel(frame, text=label, width=250, anchor="w").pack(side="left")
            ctk.CTkLabel(frame, text=value, width=120, anchor="e").pack(side="right")

    def _calculate_time_analysis(self, t, y):
        """Calcula métricas para análise de tempo"""
        y = np.array(y)  # Corrige o TypeError para operações matemáticas

        # Tensão pico a pico
        vpp = np.max(y) - np.min(y)

        # Tensão RMS
        rms = np.sqrt(np.mean(y ** 2))

        # Tensão média (DC offset)
        mean = np.mean(y)

        # Fator de crista (Crest Factor)
        crest_factor = np.max(np.abs(y)) / rms if rms > 0 else 0

        # Taxa de cruzamento por zero
        zero_crossings = np.where(np.diff(np.sign(y)))[0]
        zero_crossing_rate = len(zero_crossings) / (t[-1] - t[0]) if len(t) > 1 else 0

        # Frequência estimada
        freq_est = 0.0  # Inicializa com valor padrão
        if len(zero_crossings) > 1:
            avg_period = np.mean(np.diff(t[zero_crossings])) * 2
            freq_est = 1 / avg_period if avg_period > 0 else 0

        # Retorna as métricas formatadas
        return [
            ("Tensão Pico a Pico (Vpp):", f"{vpp:.4f} V"),
            ("Tensão RMS:", f"{rms:.4f} V"),
            ("Tensão Média (DC):", f"{mean:.4f} V"),
            ("Fator de Crista:", f"{crest_factor:.4f}"),
            ("Taxa de Cruzamento por Zero:", f"{zero_crossing_rate:.2f} Hz"),
            ("Frequência Estimada:", f"{freq_est:.2f} Hz")
        ]

    def _calculate_freq_analysis(self, f, Y):
        """Calcula métricas para análise de frequência"""
        # Encontra a frequência fundamental (apenas frequências positivas)
        positive_mask = f >= 0
        f_positive = f[positive_mask]
        Y_positive = Y[positive_mask]

        if len(Y_positive) == 0:
            return [
                ("Frequência Fundamental:", "---"),
                ("Amplitude Fundamental:", "---"),
                ("THD (Distorção Harmônica):", "---"),
                ("SNR (Relação Sinal-Ruído):", "---")
            ]

        fundamental_idx = np.argmax(Y_positive)
        fundamental_freq = f_positive[fundamental_idx]
        fundamental_amp = Y_positive[fundamental_idx]

        # THD (Total Harmonic Distortion)
        peaks, _ = find_peaks(Y_positive, height=np.max(Y_positive) * 0.05, distance=10)
        if len(peaks) > 1 and fundamental_idx in peaks:
            harmonic_peaks = np.delete(peaks, np.where(peaks == fundamental_idx))
            harmonic_power = np.sum(Y_positive[harmonic_peaks] ** 2)
            thd = np.sqrt(harmonic_power / (fundamental_amp ** 2)) * 100 if fundamental_amp > 0 else 0
        else:
            thd = 0

        # SNR (estimado)
        signal_power = np.sum(Y_positive ** 2)
        noise_power = signal_power - (fundamental_amp ** 2)
        snr = 10 * np.log10(fundamental_amp ** 2 / noise_power) if noise_power > 0 else float('inf')

        # Retorna as métricas formatadas
        return [
            ("Frequência Fundamental:", f"{self._format_freq(fundamental_freq)}"),
            ("Amplitude Fundamental:", f"{fundamental_amp:.4f}"),
            ("THD (Distorção Harmônica):", f"{thd:.2f}%"),
            ("SNR (Relação Sinal-Ruído):", f"{snr:.2f} dB" if snr != float('inf') else "inf dB")
        ]

    def import_wav(self):
        """Importa e decodifica arquivo WAV do Fnirsi"""
        self.logger.info("Iniciando importação de WAV")
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not filepath:  # Usuário cancelou a seleção
            self.logger.info("Importação de WAV cancelada pelo usuário")
            return

        try:
            # Lê o arquivo WAV
            header, data_ch1, data_ch2 = self._read_wav_file(filepath)

            # Decodifica o cabeçalho
            volt_scale, ts = self._parse_header(header)
            self.imported_voltage_scale = volt_scale[0]  # Salva a escala para exportação (canal 1)
            self.imported_time_scale = ts  # Salva a entrada completa da escala de tempo

            # Decodifica os dados dos canais
            ch1_data = self._parse_channel_data(data_ch1, volt_scale[0])

            # Calcula parâmetros do sinal
            N = len(ch1_data)
            # Calcula o vetor de tempo: total_time = 12 divisões * (ts[0] * ts[2]) segundos
            time_per_division = ts[0] * ts[2]  # em segundos
            total_time = 12 * time_per_division
            t = np.linspace(0, total_time, N, endpoint=False)
            Fs = N / total_time  # Frequência de amostragem

            vpp = max(ch1_data) - min(ch1_data)  # Tensão pico a pico

            # FFT com zero centralizado
            Y = fftshift(fft(ch1_data))
            f = fftshift(fftfreq(N, 1 / Fs))

            # Salva os dados
            self.last_data = {'t': t, 'y': ch1_data, 'f': f, 'Y': np.abs(Y)}

            # Atualiza campos de entrada
            self.after(0, lambda: self.entry_duration.delete(0, tk.END))
            self.after(0, lambda: self.entry_duration.insert(0, f"{total_time:.4f}"))
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

            # Mostra prévia do sinal em janela modal com análises
            self.after(0, lambda: self._show_wav_preview(t, ch1_data, os.path.basename(filepath)))

            # Salva como JSON
            json_path = os.path.splitext(filepath)[0] + ".json"
            with open(json_path, 'w') as f:
                json.dump({
                    'time_per_division': time_per_division,
                    'total_time': total_time,
                    'voltage_scale': volt_scale[0],
                    'time_values': t.tolist(),
                    'signal_values': ch1_data
                }, f, indent=2)

            self.set_status(f"📊 Dados salvos em: {json_path}", "lightblue")
            self.logger.info(f"WAV importado com sucesso: {filepath}")

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Falha na importação do WAV: {error_msg}", exc_info=True)
            self.after(0, lambda: messagebox.showerror("Erro ao importar WAV", error_msg))
            self.set_status(f"❌ Erro ao importar: {error_msg}", "red")

    def _format_freq(self, f):
        """Formata valores de frequência para exibição"""
        if f < 1e3:
            return f"{f:.1f} Hz"
        elif f < 1e6:
            return f"{f / 1e3:.2f} kHz"
        elif f < 1e9:
            return f"{f / 1e6:.2f} MHz"
        else:
            return f"{f / 1e9:.2f} GHz"

    def export_wav(self):
        """Exporta o sinal atual em formato WAV compatível com Fnirsi 1013D"""
        self.logger.info("Iniciando exportação de WAV")
        if not self.last_data:
            messagebox.showerror("Erro", "Gere ou importe um sinal primeiro.")
            self.logger.warning("Nenhum dado para exportar")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")]
        )
        if not filepath:  # Usuário cancelou
            self.logger.info("Exportação de WAV cancelada pelo usuário")
            return

        try:
            # Obtém os dados do sinal
            t = self.last_data['t']
            y = self.last_data['y']

            # Se o sinal foi importado, usa a escala original
            if hasattr(self, 'imported_voltage_scale'):
                scale_value = self.imported_voltage_scale[0]
                time_scale = self.imported_time_scale
            else:
                # Para sinais gerados, usa escala padrão
                scale_value = 1.0
                time_scale = TIME_LIST[0]

            # Converte os dados para o formato do osciloscópio
            data_bytes = bytearray()
            for value in y:
                # Fórmula inversa: raw = (value * 50 / scale_value) + 200
                raw_value = (value * 50.0 / scale_value) + 200
                int_value = int(raw_value)
                # Limitar o valor ao intervalo [0, 65535]
                int_value = max(0, min(int_value, 65535))
                # Divide em dois bytes (little-endian)
                data_bytes.append(int_value & 0xFF)
                data_bytes.append((int_value >> 8) & 0xFF)

            # Cria o cabeçalho
            header = bytearray(208)
            # Índices de escala de tensão (canais 1 e 2)
            if hasattr(self, 'imported_voltage_scale'):
                header[4] = VOLT_LIST.index(self.imported_voltage_scale)
                header[14] = VOLT_LIST.index(self.imported_voltage_scale)
            else:
                # Usa valores padrão se não foram importados
                header[4] = 2  # Índice para 1V
                header[14] = 2

            # Índice de escala de tempo
            if hasattr(self, 'imported_time_scale'):
                header[22] = TIME_LIST.index(time_scale)
            else:
                header[22] = 5  # Índice para 1s

            # Constrói o arquivo completo
            file_data = bytearray(WAV_TOTAL_SIZE)
            file_data[0:208] = header
            # Canal 1: 1500 amostras (3000 bytes) a partir do byte 1000
            # Se tiver mais de 1500 pontos, interpola
            if len(y) != 1500:
                t_orig = np.linspace(t[0], t[-1], len(y))
                t_new = np.linspace(t[0], t[-1], 1500)
                y_resampled = np.interp(t_new, t_orig, y)

                # Reconverte os dados interpolados
                data_bytes = bytearray()
                for value in y_resampled:
                    raw_value = (value * 50.0 / scale_value) + 200
                    int_value = int(raw_value)
                    int_value = max(0, min(int_value, 65535))
                    data_bytes.append(int_value & 0xFF)
                    data_bytes.append((int_value >> 8) & 0xFF)

            start_index = 1000
            end_index = start_index + len(data_bytes)
            if end_index <= WAV_TOTAL_SIZE:
                file_data[start_index:end_index] = data_bytes
            else:
                raise ValueError("Dados excedem o tamanho máximo do arquivo WAV")

            # Escreve o arquivo
            with open(filepath, 'wb') as f:
                f.write(file_data)

            self.set_status(f"📁 Sinal exportado como WAV: {os.path.basename(filepath)}", "lightblue")
            self.logger.info(f"WAV exportado com sucesso: {filepath}")

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Erro ao exportar WAV: {error_msg}", exc_info=True)
            messagebox.showerror("Erro ao exportar WAV", error_msg)
            self.set_status(f"❌ Erro ao exportar: {error_msg}", "red")

    def _update_voltage_ticks(self):
        """Atualiza as escalas de tensão no gráfico de tempo"""
        if not self.last_data:
            return

        try:
            # Obtém a amplitude Vpp configurada em mV
            vpp = float(self.entry_vpp.get()) * 1000

            # Define o número de divisões
            num_divisions = 8
            step = vpp / num_divisions

            # Cria as marcas de tensão
            ticks = np.arange(-vpp / 2, vpp / 2 + step, step)

            # Atualiza os ticks do eixo Y
            self.ax_time.set_yticks(ticks)
            self.ax_time.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

        except Exception as e:
            # Em caso de erro, usa escala automática
            self.logger.warning(f"Erro ao atualizar ticks de tensão: {str(e)}")
            self.ax_time.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

    def _update_plots(self):
        self.logger.info("Atualizando gráficos")
        start_time = time.time()
        if not self.last_data:
            self.logger.warning("Nenhum dado para plotar")
            return

        try:
            # Salva os marcadores existentes antes de limpar
            saved_markers = self._save_markers_state()

            # Limpa apenas o conteúdo dos eixos, preservando os marcadores
            self.ax_time.clear()
            self.ax_freq.clear()

            # Converter sinal para mV
            y_mV = np.array(self.last_data['y']) * 1000

            # Plota os novos dados em mV
            self.time_plot_line, = self.ax_time.plot(self.last_data['t'], y_mV, color="cyan", zorder=5)
            self.ax_time.set_title("Domínio do Tempo", color='white')
            self.ax_time.set_ylabel('Amplitude (mV)')  # Adiciona label em mV
            self.ax_time.grid(True, linestyle='--', alpha=0.5)
            self.ax_time.tick_params(colors='white')
            self.ax_time.xaxis.set_major_formatter(FuncFormatter(self._format_time_axis))

            # Configura escala fixa para eixo Y do tempo em mV
            try:
                vpp = float(self.entry_vpp.get()) * 1000  # Convertendo para mV
                self.ax_time.set_ylim(-vpp / 2, vpp / 2)

                # Adiciona escalas de tensão
                self._update_voltage_ticks()

            except Exception as e:
                # Usa escala automática se amplitude inválida
                self.logger.warning(f"Erro ao configurar escala de tensão: {str(e)}")
                self.ax_time.autoscale(axis='y')
                self.ax_time.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

            # Gráfico de frequência com zero centralizado
            self.freq_plot_line, = self.ax_freq.plot(self.last_data['f'], self.last_data['Y'], color="orange")
            self.ax_freq.set_title("Domínio da Frequência (FFT)", color='white')
            self.ax_freq.set_ylabel("|Y(f)|", color='white')
            self.ax_freq.grid(True, linestyle='--', alpha=0.5)
            self.ax_freq.tick_params(colors='white')
            self.ax_freq.xaxis.set_major_formatter(FuncFormatter(self._format_freq_axis))

            # Configura escala para eixo Y da frequência
            self.ax_freq.set_ylim(0, np.max(self.last_data['Y']) * 1.1)

            # Centraliza o zero na frequência
            max_freq = np.max(np.abs(self.last_data['f']))
            self.ax_freq.set_xlim(-max_freq, max_freq)

            # Ajusta a visualização inicial para mostrar o centro do sinal
            self._adjust_initial_view()

            # Restaura os marcadores
            self._restore_markers_state(saved_markers)

            self.fig.tight_layout(pad=3.0)
            self.canvas.draw()
            self.set_status(f"✅ Gráficos atualizados com sucesso! ({time.time() - start_time:.3f}s)", "lightgreen")
            self.logger.info(f"Gráficos atualizados com sucesso em {time.time() - start_time:.3f}s")
        except Exception as e:
            error_msg = f"Erro ao atualizar gráficos: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.set_status(f"❌ {error_msg}", "red")

    def _adjust_initial_view(self):
        """Ajusta a visualização inicial para mostrar o centro do sinal com zoom adequado"""
        if not self.last_data:
            return

        try:
            t = self.last_data['t']
            f = self.last_data['f']
            y = self.last_data['y']
            Y = self.last_data['Y']

            # Para o domínio do tempo:
            # 1. Calcula a duração total do sinal
            total_duration = t[-1] - t[0]

            # 2. Determina a janela de visualização ideal
            #    - Para altas frequências (acima de 10kHz), mostra 10 ciclos
            #    - Para baixas frequências, mostra 50% do sinal
            try:
                # Tenta detectar a frequência fundamental
                max_idx = np.argmax(Y)
                freq_fundamental = abs(f[max_idx])

                if freq_fundamental > 10000:  # Alta frequência
                    period = 1 / freq_fundamental
                    view_duration = min(10 * period, total_duration)

                    # Para frequências muito altas, usar micro/nano segundos
                    if period < 1e-6:
                        self.ax_time.set_xlabel('Tempo (ns)')
                        time_multiplier = 1e9
                    elif period < 1e-3:
                        self.ax_time.set_xlabel('Tempo (µs)')
                        time_multiplier = 1e6
                    else:
                        self.ax_time.set_xlabel('Tempo (s)')
                        time_multiplier = 1
                else:
                    view_duration = total_duration * 0.5
                    time_multiplier = 1
                    self.ax_time.set_xlabel('Tempo (s)')

                # Centraliza no meio do sinal
                center = (t[0] + t[-1]) / 2
                self.ax_time.set_xlim(center - view_duration / 2, center + view_duration / 2)

                # Atualiza o formatter para usar a unidade correta
                self.ax_time.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * time_multiplier:.2f}"))

                # Ajusta o slider de zoom
                self.zoom_factor = view_duration / total_duration
                self.zoom_time.set(self.zoom_factor)

                self.logger.info(f"Visualização tempo ajustada: centro={center}s, duração={view_duration}s")
            except:
                # Fallback: mostra o centro com 50% do sinal
                center = (t[0] + t[-1]) / 2
                view_duration = total_duration * 0.5
                self.ax_time.set_xlim(center - view_duration / 2, center + view_duration / 2)
                self.zoom_time.set(0.5)

            # Para o domínio da frequência já está centralizado
            max_freq = np.max(np.abs(f))
            self.ax_freq.set_xlim(-max_freq, max_freq)

            self.logger.info(f"Visualização frequência ajustada: {-max_freq} a {max_freq} Hz")

        except Exception as e:
            self.logger.error(f"Erro ao ajustar visualização inicial: {str(e)}", exc_info=True)
            # Fallback seguro
            self.ax_time.set_xlim(self.last_data['t'][0], self.last_data['t'][-1])
            max_freq = np.max(np.abs(self.last_data['f']))
            self.ax_freq.set_xlim(-max_freq, max_freq)
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
            y_interp = cs(t_interp) * 1000  # Converter para mV

            self.time_plot_line.set_data(t_interp, y_interp)
        else:
            # Converter para mV
            self.time_plot_line.set_data(self.last_data['t'], np.array(self.last_data['y']) * 1000)

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
            # Converter para mV
            self.lbl_y1.configure(text=f"Y1: {th[0] * 1000:.1f} mV")
            self.lbl_y2.configure(text=f"Y2: {th[1] * 1000:.1f} mV")
            self.lbl_dy.configure(text=f"ΔY: {(th[1] - th[0]) * 1000:.1f} mV")
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
        self.logger.info("Atualizando painéis de análise")
        start_time = time.time()
        if not self.last_data:
            self.logger.warning("Nenhum dado para análise")
            return

        try:
            # Limitar pontos para análise (performance)
            max_points = min(len(self.last_data['y']), ANALYSIS_MAX_POINTS)
            step = max(1, len(self.last_data['y']) // max_points)

            t = np.array(self.last_data['t'][::step])
            y = np.array(self.last_data['y'][::step])
            f = self.last_data['f']
            Y = self.last_data['Y']

            self.logger.debug(f"Dados preparados para análise. Tempo: {len(t)} pts, Freq: {len(f)} pts")

            # Análise no domínio do tempo
            if len(y) > 0:
                # Tensão pico a pico
                vpp = np.max(y) - np.min(y)
                self.time_analysis_labels['vpp'].configure(text=f"{vpp:.4f} V")
                self.logger.debug(f"Vpp: {vpp:.4f} V")

                # Tensão RMS
                rms = np.sqrt(np.mean(y ** 2))
                self.time_analysis_labels['rms'].configure(text=f"{rms:.4f} V")
                self.logger.debug(f"RMS: {rms:.4f} V")

                # Tensão média (DC offset)
                mean = np.mean(y)
                self.time_analysis_labels['mean'].configure(text=f"{mean:.4f} V")
                self.logger.debug(f"Média: {mean:.4f} V")

                # Fator de crista (Crest Factor)
                crest_factor = np.max(np.abs(y)) / rms if rms > 0 else 0
                self.time_analysis_labels['crest_factor'].configure(text=f"{crest_factor:.4f}")
                self.logger.debug(f"Fator de crista: {crest_factor:.4f}")

                # Taxa de cruzamento por zero
                zero_crossings = np.where(np.diff(np.sign(y)))[0]
                if len(zero_crossings) > 0:
                    zero_crossing_rate = len(zero_crossings) / (t[-1] - t[0])
                    self.time_analysis_labels['zero_crossing'].configure(text=f"{zero_crossing_rate:.2f} Hz")
                    self.logger.debug(f"Taxa de cruzamento: {zero_crossing_rate:.2f} Hz")
                else:
                    self.time_analysis_labels['zero_crossing'].configure(text="0 Hz")
                    self.logger.debug("Taxa de cruzamento: 0 Hz")

                # Frequência estimada - INICIALIZAÇÃO CORRIGIDA
                freq_est = 0.0
                if len(zero_crossings) > 1:
                    periods = np.diff(t[zero_crossings])
                    if len(periods) > 0:
                        avg_period = np.mean(periods) * 2
                        freq_est = 1 / avg_period if avg_period > 0 else 0
                self.time_analysis_labels['frequency'].configure(text=f"{freq_est:.2f} Hz")
                self.logger.debug(f"Frequência estimada: {freq_est:.2f} Hz")

                # Duty cycle (apenas para ondas quadradas)
                if self.waveform.get().startswith("Quadrada") or self.waveform.get().startswith("Pulso"):
                    if vpp > 0:
                        threshold = (np.max(y) + np.min(y)) / 2
                        positive_samples = np.sum(y > threshold)
                        duty_cycle = positive_samples / len(y) * 100
                        self.time_analysis_labels['duty_cycle'].configure(text=f"{duty_cycle:.1f}%")
                        self.logger.debug(f"Duty cycle: {duty_cycle:.1f}%")
                    else:
                        self.time_analysis_labels['duty_cycle'].configure(text="N/A")
                        self.logger.debug("Duty cycle: N/A")
                else:
                    self.time_analysis_labels['duty_cycle'].configure(text="N/A")
                    self.logger.debug("Duty cycle: N/A")

                # Relação Pico/RMS
                peak_to_rms = np.max(np.abs(y)) / rms if rms > 0 else 0
                self.time_analysis_labels['peak_to_rms'].configure(text=f"{peak_to_rms:.4f}")
                self.logger.debug(f"Pico/RMS: {peak_to_rms:.4f}")

                # Curtose
                if len(y) > 3:
                    kurt_val = kurtosis(y)
                    self.time_analysis_labels['kurtosis'].configure(text=f"{kurt_val:.4f}")
                    self.logger.debug(f"Curtose: {kurt_val:.4f}")
                else:
                    self.time_analysis_labels['kurtosis'].configure(text="---")
                    self.logger.debug("Curtose: ---")

                # Assimetria (Skewness)
                if len(y) > 2:
                    skew_val = skew(y)
                    self.time_analysis_labels['skewness'].configure(text=f"{skew_val:.4f}")
                    self.logger.debug(f"Assimetria: {skew_val:.4f}")
                else:
                    self.time_analysis_labels['skewness'].configure(text="---")
                    self.logger.debug("Assimetria: ---")

            # Análise no domínio da frequência
            if len(Y) > 0:
                # Considerar apenas frequências positivas para análise
                positive_mask = f >= 0
                f_positive = f[positive_mask]
                Y_positive = Y[positive_mask]

                if len(Y_positive) == 0:
                    self.logger.warning("Nenhuma frequência positiva para análise")
                    return

                # Encontra a frequência fundamental (maior magnitude)
                fundamental_idx = np.argmax(Y_positive)
                fundamental_freq = f_positive[fundamental_idx]
                fundamental_amp = Y_positive[fundamental_idx]
                self.freq_analysis_labels['fundamental'].configure(text=f"{self._format_freq(fundamental_freq)}")
                self.freq_analysis_labels['fund_amp'].configure(text=f"{fundamental_amp:.4f}")
                self.logger.debug(f"Fundamental: {fundamental_freq} Hz, Amplitude: {fundamental_amp:.4f}")

                # Encontra harmônicos (excluindo o fundamental)
                harmonic_mask = (f_positive > fundamental_freq * 0.9) & (f_positive < f_positive[-1])
                harmonic_freqs = f_positive[harmonic_mask]
                harmonic_amps = Y_positive[harmonic_mask]

                # Identifica picos significativos
                peaks, _ = find_peaks(harmonic_amps, height=fundamental_amp * 0.05, distance=10)
                harmonic_peaks = peaks[np.argsort(harmonic_amps[peaks])[::-1]]

                # Calcula THD (Total Harmonic Distortion)
                thd = 0.0
                harmonics_level = 0.0
                if len(harmonic_peaks) > 0 and fundamental_amp > 0:
                    harmonic_power = np.sum(harmonic_amps[harmonic_peaks] ** 2)
                    thd = np.sqrt(harmonic_power / (fundamental_amp ** 2)) * 100
                    harmonics_level = harmonic_power / (fundamental_amp ** 2)
                self.freq_analysis_labels['thd'].configure(text=f"{thd:.2f}%")
                self.freq_analysis_labels['harmonics'].configure(text=f"{harmonics_level:.4f}")

                # SNR (Signal to Noise Ratio)
                snr = "inf dB"
                noise_floor = "---"
                if fundamental_amp > 0:
                    # Considera tudo que não é fundamental ou harmônicos como ruído
                    noise_mask = (f_positive > 0) & ~harmonic_mask
                    noise_power = np.sum(Y_positive[noise_mask] ** 2)

                    if noise_power > 0:
                        snr_val = 10 * np.log10(fundamental_amp ** 2 / noise_power)
                        snr = f"{snr_val:.2f} dB"
                        noise_floor = f"{np.sqrt(noise_power):.4f}"
                self.freq_analysis_labels['snr'].configure(text=snr)
                self.freq_analysis_labels['noise_floor'].configure(text=noise_floor)

                # SFDR (Spurious Free Dynamic Range)
                sfdr = "inf dB"
                if len(harmonic_peaks) > 0:
                    max_spur = np.max(harmonic_amps[harmonic_peaks])
                    if fundamental_amp > 0 and max_spur > 0:
                        sfdr_val = 20 * np.log10(fundamental_amp / max_spur)
                        sfdr = f"{sfdr_val:.2f} dB"
                self.freq_analysis_labels['sfdr'].configure(text=sfdr)

                # Largura de banda a -3dB (método robusto)
                bandwidth = "---"
                if fundamental_amp > 0:
                    half_power = fundamental_amp / np.sqrt(2)
                    above_threshold = Y_positive > half_power
                    if np.any(above_threshold):
                        indices = np.where(above_threshold)[0]
                        min_idx = indices[0]
                        max_idx = indices[-1]
                        min_freq = f_positive[min_idx]
                        max_freq = f_positive[max_idx]
                        bandwidth_val = max_freq - min_freq
                        bandwidth = self._format_freq(bandwidth_val)
                self.freq_analysis_labels['bandwidth'].configure(text=bandwidth)

                # Índice de modulação (estimado)
                mod_index = "N/A"
                if self.mod_am.get():
                    # Para AM: m = (A_max - A_min) / (A_max + A_min)
                    A_max = np.max(y)
                    A_min = np.min(y)
                    if A_max + A_min != 0:
                        mod_index_val = (A_max - A_min) / (A_max + A_min) * 100
                        mod_index = f"{mod_index_val:.1f}%"
                elif self.mod_fm.get():
                    # Para FM: β = Δf / f_m
                    if fundamental_amp > 0:
                        sideband_mask = (f_positive > fundamental_freq - 10) & (f_positive < fundamental_freq + 10)
                        sideband_amp = np.max(Y_positive[sideband_mask])
                        if fundamental_amp > 0:
                            mod_index_val = sideband_amp / fundamental_amp * 100
                            mod_index = f"{mod_index_val:.1f}%"
                self.freq_analysis_labels['mod_index'].configure(text=mod_index)

                # Frequência de pico (maior harmônico)
                peak_freq = "---"
                if len(harmonic_peaks) > 0:
                    peak_harmonic_idx = harmonic_peaks[0]
                    peak_freq = self._format_freq(harmonic_freqs[peak_harmonic_idx])
                self.freq_analysis_labels['peak_freq'].configure(text=peak_freq)

            self.logger.info(f"Painéis de análise atualizados em {time.time() - start_time:.3f}s")
            self.logger.debug(f"Métricas tempo: Vpp={vpp:.4f}, RMS={rms:.4f}, Freq={freq_est:.2f}")
            self.logger.debug(f"Métricas freq: Fund={fundamental_freq:.2f}, THD={thd:.2f}%")

            # Forçar atualização da interface
            self.update_idletasks()

        except Exception as e:
            error_msg = f"Erro ao atualizar painéis de análise: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.set_status(f"❌ {error_msg}", "red")
            self.logger.debug(f"Tipo de dados: t={type(t)}, y={type(y)}, f={type(f)}, Y={type(Y)}")
            self.logger.debug(f"Comprimentos: t={len(t) if hasattr(t, '__len__') else 'N/A'}, "
                              f"y={len(y) if hasattr(y, '__len__') else 'N/A'}, "
                              f"f={len(f) if hasattr(f, '__len__') else 'N/A'}, "
                              f"Y={len(Y) if hasattr(Y, '__len__') else 'N/A'}")

    def _format_time(self, s):
        if s < 1e-6:
            return f"{s * 1e9:.2f} ns"
        elif s < 1e-3:
            return f"{s * 1e6:.2f} µs"
        elif s < 1:
            return f"{s * 1e3:.2f} ms"
        else:
            return f"{s:.2f} s"

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