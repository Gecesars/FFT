import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import square, sawtooth, unit_impulse, gausspulse, chirp, find_peaks
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import CubicSpline
from scipy.special import jv
from scipy.stats import kurtosis, skew
import csv
import json
from concurrent.futures import ThreadPoolExecutor
import struct
import os

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --- Constantes ---
UNIT_MULTIPLIERS_FREQ = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
UNIT_MULTIPLIERS_TIME = {"s": 1, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9}
FNIRSI_FILE_SIZE = 15360
FNIRSI_SAMPLES = 1500
VOLT_LIST = [[5.0, "V", 1], [2.5, "V", 1], [1.0, "V", 1], [500, "mV", 0.001], [200, "mV", 0.001], [100, "mV", 0.001],
             [50, "mV", 0.001]]
TIME_LIST = [[50, "s", 1], [20, "s", 1], [10, "s", 1], [5, "s", 1], [2, "s", 1], [1, "s", 1], [500, "ms", .001],
             [200, "ms", .001], [100, "ms", .001], [50, "ms", .001], [20, "ms", .001], [10, "ms", .001],
             [5, "ms", .001], [2, "ms", .001], [1, "ms", .001], [500, "us", 1E-6], [200, "us", 1E-6], [100, "us", 1E-6],
             [50, "us", 1E-6], [20, "us", 1E-6], [10, "us", 1E-6], [5, "us", 1E-6], [2, "us", 1E-6], [1, "us", 1E-6],
             [500, "ns", 1E-9], [200, "ns", 1E-9], [100, "ns", 1E-9], [50, "ns", 1E-9], [20, "ns", 1E-9],
             [10, "ns", 1E-9]]


class SignalGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gerador e Analisador de Sinais Profissional v5.0 (Final)")
        self.geometry("1800x1000")

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, minsize=350, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.executor = ThreadPoolExecutor(max_workers=2)
        self.last_data = {}
        self.markers = {'time_v': [], 'time_h': [], 'freq_v': [], 'freq_h': []}
        self.dragging_marker = None
        self.last_click_event = None
        self.imported_hw_info = None
        self.sampling_state = {}

        self.mod_am = tk.BooleanVar(value=False)
        self.mod_fm = tk.BooleanVar(value=False)

        self._build_sidebar()
        self._build_plot_area()
        self._build_analysis_interface()
        self._build_context_menu()
        self._build_status_bar()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self.set_status("Fechando...", "yellow")
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=8)
        self.sidebar.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.sidebar.grid_propagate(False)

        def section(parent, title, color):
            frm = ctk.CTkFrame(parent, fg_color=color, corner_radius=6)
            frm.pack(fill="x", pady=5, padx=5)
            ctk.CTkLabel(frm, text=title, font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(5, 2))
            return frm

        scrollable_sidebar = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        scrollable_sidebar.pack(fill="both", expand=True)

        frm_gen = section(scrollable_sidebar, "Geração de Sinal", "#444444")
        self.entry_duration, self.units_duration = self._add_unit_entry(frm_gen, "Duração:", "10", "ms",
                                                                        list(UNIT_MULTIPLIERS_TIME.keys()))
        self.entry_fc, self.units_fc = self._add_unit_entry(frm_gen, "Portadora (Fc):", "1", "kHz",
                                                            list(UNIT_MULTIPLIERS_FREQ.keys()))
        self.entry_fs, self.units_fs = self._add_unit_entry(frm_gen, "Amostragem (Fs):", "100", "kHz",
                                                            list(UNIT_MULTIPLIERS_FREQ.keys()))
        self.entry_vpp = self._add_entry(frm_gen, "Amplitude (Vpp):", "2.0")

        waveforms = ["Seno", "Cosseno", "Quadrada", "Triangular", "Dente de Serra", "Pulso", "Ruído Branco",
                     "Exp Decaimento", "Passo (step)", "Rampa", "Parábola", "Impulso", "Tangente", "Sinc", "Gaussiana",
                     "Chirp Linear", "Chirp Quadrático", "Bessel"]
        self.waveform = self._add_option_menu(frm_gen, "Forma de Onda:", waveforms)

        frm_mod = section(scrollable_sidebar, "Modulação", "#2d3e50")
        self.entry_fm, self.units_fm = self._add_unit_entry(frm_mod, "Moduladora (Fm):", "100", "Hz",
                                                            list(UNIT_MULTIPLIERS_FREQ.keys()))
        ctk.CTkCheckBox(frm_mod, text="Ativar AM", variable=self.mod_am).pack(anchor="w", padx=10, pady=5)
        self.slider_am = ctk.CTkSlider(frm_mod, from_=0, to=2, number_of_steps=200);
        self.slider_am.set(0.5);
        self.slider_am.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkCheckBox(frm_mod, text="Ativar FM", variable=self.mod_fm).pack(anchor="w", padx=10, pady=5)
        self.slider_fm_dev = ctk.CTkSlider(frm_mod, from_=0, to=1, number_of_steps=100);
        self.slider_fm_dev.set(0.5);
        self.slider_fm_dev.pack(fill="x", padx=10, pady=(0, 10))

        frm_cmd = section(scrollable_sidebar, "Comandos", "#333333")
        self.btn_generate = ctk.CTkButton(frm_cmd, text="Gerar Sinal", command=self.submit_plot_task)
        self.btn_generate.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(frm_cmd, text="Exportar Dados", command=self.export_data).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(frm_cmd, text="Importar WAV (Fnirsi)", command=self.import_wav).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(frm_cmd, text="Exportar WAV (Fnirsi)", command=self._export_fnirsi_wav).pack(fill="x", padx=10,
                                                                                                   pady=5)

    def _add_unit_entry(self, parent, label, default_val, default_unit, units_list):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(5, 0))
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=(0, 5))
        entry = ctk.CTkEntry(frame);
        entry.insert(0, default_val)
        entry.pack(side="left", fill="x", expand=True)
        units = ctk.CTkOptionMenu(frame, values=units_list, width=75)
        units.set(default_unit);
        units.pack(side="left", padx=(5, 0))
        return entry, units

    def _add_entry(self, parent, label, default):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(5, 0))
        entry = ctk.CTkEntry(parent);
        entry.insert(0, default)
        entry.pack(fill="x", padx=10, pady=(0, 5))
        return entry

    def _add_option_menu(self, parent, label, values):
        ctk.CTkLabel(parent, text=label).pack(anchor="w", padx=10, pady=(5, 0))
        menu = ctk.CTkOptionMenu(parent, values=values);
        menu.set(values[0])
        menu.pack(fill="x", padx=10, pady=5)
        return menu

    def _build_plot_area(self):
        plot_container = ctk.CTkFrame(self)
        plot_container.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        plot_container.grid_rowconfigure(0, weight=1)
        plot_container.grid_columnconfigure(1, weight=1)

        y_scale_frame = ctk.CTkFrame(plot_container, width=50)
        y_scale_frame.grid(row=0, column=0, sticky="ns", pady=5, padx=(5, 0))
        y_scale_frame.grid_rowconfigure([0, 2], weight=1)
        y_scale_frame.grid_rowconfigure(1, minsize=20)
        y_scale_frame.grid_rowconfigure(3, minsize=20)

        ctk.CTkLabel(y_scale_frame, text="Y-Time").pack(pady=5)
        self.time_zoom_y = ctk.CTkSlider(y_scale_frame, from_=5.0, to=0.01, orientation="vertical",
                                         command=self._update_y_scales);
        self.time_zoom_y.set(1.0)
        self.time_zoom_y.pack(fill="y", expand=True, padx=5)
        self.time_pan_y = ctk.CTkSlider(y_scale_frame, from_=1.0, to=-1.0, orientation="vertical",
                                        command=self._update_y_scales);
        self.time_pan_y.set(0)
        self.time_pan_y.pack(fill="y", expand=True, padx=5, pady=10)

        ctk.CTkLabel(y_scale_frame, text="Y-Freq").pack(pady=5)
        self.freq_zoom_y = ctk.CTkSlider(y_scale_frame, from_=5.0, to=0.01, orientation="vertical",
                                         command=self._update_y_scales);
        self.freq_zoom_y.set(1.0)
        self.freq_zoom_y.pack(fill="y", expand=True, padx=5)
        self.freq_pan_y = ctk.CTkSlider(y_scale_frame, from_=1.0, to=-1.0, orientation="vertical",
                                        command=self._update_y_scales);
        self.freq_pan_y.set(0)
        self.freq_pan_y.pack(fill="y", expand=True, padx=5, pady=10)

        main_plot_frame = ctk.CTkFrame(plot_container, fg_color="transparent")
        main_plot_frame.grid(row=0, column=1, sticky="nsew")

        self.fig, (self.ax_time, self.ax_freq) = plt.subplots(2, 1, facecolor="#2B2B2B")
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        ctrl_frame = ctk.CTkFrame(main_plot_frame, height=40);
        ctrl_frame.pack(fill="x", pady=(5, 0))
        ctk.CTkLabel(ctrl_frame, text="Zoom X-Tempo:").pack(side="left", padx=(10, 5))
        self.zoom_time_x = ctk.CTkSlider(ctrl_frame, from_=0.001, to=1.0, command=self.update_time_zoom);
        self.zoom_time_x.set(1.0)
        self.zoom_time_x.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkButton(ctrl_frame, text="Reset", width=60, command=self.reset_zoom).pack(side="right", padx=10)

        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.canvas.mpl_connect('pick_event', self._on_pick_event)

    def _build_analysis_interface(self):
        side_panel = ctk.CTkFrame(self, width=350)
        side_panel.grid(row=0, column=2, padx=(0, 10), pady=10, sticky="nsew")
        side_panel.grid_rowconfigure(1, weight=1)

        marker_frame = ctk.CTkFrame(side_panel);
        marker_frame.pack(fill="x", pady=(0, 10), padx=5)
        frm_time = ctk.CTkFrame(marker_frame, fg_color="#1E1E1E");
        frm_time.pack(fill="x", pady=5, padx=5)
        ctk.CTkLabel(frm_time, text="Marcadores de Tempo", text_color="cyan", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=8)
        self.lbl_x1 = ctk.CTkLabel(frm_time, text="X1: ---");
        self.lbl_x1.pack(anchor="w", padx=8)
        self.lbl_x2 = ctk.CTkLabel(frm_time, text="X2: ---");
        self.lbl_x2.pack(anchor="w", padx=8)
        self.lbl_dx = ctk.CTkLabel(frm_time, text="ΔX: ---");
        self.lbl_dx.pack(anchor="w", padx=8)
        self.lbl_y1 = ctk.CTkLabel(frm_time, text="Y1: ---");
        self.lbl_y1.pack(anchor="w", padx=8, pady=(5, 0))
        self.lbl_y2 = ctk.CTkLabel(frm_time, text="Y2: ---");
        self.lbl_y2.pack(anchor="w", padx=8)
        self.lbl_dy = ctk.CTkLabel(frm_time, text="ΔY: ---");
        self.lbl_dy.pack(anchor="w", padx=8)

        frm_freq = ctk.CTkFrame(marker_frame, fg_color="#1E1E1E");
        frm_freq.pack(fill="x", padx=5)
        ctk.CTkLabel(frm_freq, text="Marcadores de Frequência", text_color="orange", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=8)
        self.lbl_f1 = ctk.CTkLabel(frm_freq, text="F1: ---");
        self.lbl_f1.pack(anchor="w", padx=8)
        self.lbl_f2 = ctk.CTkLabel(frm_freq, text="F2: ---");
        self.lbl_f2.pack(anchor="w", padx=8)
        self.lbl_df = ctk.CTkLabel(frm_freq, text="ΔF: ---");
        self.lbl_df.pack(anchor="w", padx=8)
        self.lbl_m1 = ctk.CTkLabel(frm_freq, text="|Y1|: ---");
        self.lbl_m1.pack(anchor="w", padx=8, pady=(5, 0))
        self.lbl_m2 = ctk.CTkLabel(frm_freq, text="|Y2|: ---");
        self.lbl_m2.pack(anchor="w", padx=8)
        self.lbl_dm = ctk.CTkLabel(frm_freq, text="Δ|Y|: ---");
        self.lbl_dm.pack(anchor="w", padx=8)

        notebook = ctk.CTkTabview(side_panel);
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        self.time_tab = notebook.add("Análise Tempo")
        self.freq_tab = notebook.add("Análise Frequência")

        self.time_analysis_vars = self._create_analysis_labels(self.time_tab, "Análise no Tempo", "cyan",
                                                               ["Vpp", "Vrms", "Vavg", "Fator Crista", "Freq. Estimada",
                                                                "Cruzamentos Zero", "Curtose", "Assimetria"])
        self.freq_analysis_vars = self._create_analysis_labels(self.freq_tab, "Análise na Frequência", "orange",
                                                               ["Freq. Fund.", "Amp. Fund.", "THD", "SNR", "SFDR",
                                                                "Piso de Ruído"])

    def _create_analysis_labels(self, parent, title, color, metrics):
        scroll_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(scroll_frame, text=title, font=("Arial", 14, "bold"), text_color=color).pack(anchor="w",
                                                                                                  pady=(0, 10))

        vars_dict = {metric: ctk.StringVar(value="---") for metric in metrics}
        for metric, var in vars_dict.items():
            frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            frame.pack(fill="x", padx=5, pady=2)
            ctk.CTkLabel(frame, text=f"{metric}:", anchor="w").pack(side="left")
            ctk.CTkLabel(frame, textvariable=var, anchor="e").pack(side="right", fill="x", expand=True)
        return vars_dict

    def _build_context_menu(self):
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Marcar Tempo (vertical)", command=lambda: self.add_marker('time_v'))
        self.menu.add_command(label="Marcar Tempo (horizontal)", command=lambda: self.add_marker('time_h'))
        self.menu.add_separator()
        self.menu.add_command(label="Marcar Frequência (vertical)", command=lambda: self.add_marker('freq_v'))
        self.menu.add_command(label="Marcar Frequência (horizontal)", command=lambda: self.add_marker('freq_h'))
        self.menu.add_separator()
        self.menu.add_command(label="Limpar Todos Marcadores", command=lambda: self.clear_markers('all'))

    def _build_status_bar(self):
        self.status_bar = ctk.CTkLabel(self, text="Pronto", anchor="w", height=25, font=("Arial", 11))
        self.status_bar.grid(row=1, column=0, columnspan=3, sticky="ew", padx=0, pady=0)

    def set_status(self, msg, color="white"):
        self.status_bar.configure(text=f"  {msg}", text_color=color)

    def submit_plot_task(self):
        self.btn_generate.configure(state="disabled", text="Gerando...")
        self.set_status("⏳ Iniciando geração de sinal...", "yellow")
        self.executor.submit(self._compute_and_plot_task)

    def _compute_and_plot_task(self):
        try:
            params = self._validate_inputs()
            if not params: return

            self.imported_hw_info = None
            t = np.arange(params['N']) / params['Fs']
            carrier_signal = self._generate_waveform(params, t)
            y = self._apply_modulation(params, carrier_signal, t)

            max_abs = np.max(np.abs(y))
            if max_abs > 1e-9: y = y * (params['Vpp'] / 2) / max_abs

            Y = fftshift(fft(y));
            f = fftshift(fftfreq(params['N'], 1 / params['Fs']))
            self.last_data = {'t': t, 'y': y, 'f': f, 'Y': np.abs(Y), 'params': params}
            self.after(0, self._update_plots)
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Erro de Cálculo", str(e)))
        finally:
            self.after(0, lambda: self.btn_generate.configure(state="normal", text="Gerar Sinal"))
            self.after(0, lambda: self.set_status("Pronto", "white"))

    def _validate_inputs(self):
        try:
            p = {
                'duration': float(self.entry_duration.get()) * UNIT_MULTIPLIERS_TIME[self.units_duration.get()],
                'Fc': float(self.entry_fc.get()) * UNIT_MULTIPLIERS_FREQ[self.units_fc.get()],
                'Fs': float(self.entry_fs.get()) * UNIT_MULTIPLIERS_FREQ[self.units_fs.get()],
                'Fm': float(self.entry_fm.get()) * UNIT_MULTIPLIERS_FREQ[self.units_fm.get()],
                'Vpp': float(self.entry_vpp.get()),
                'waveform': self.waveform.get(),
                'am_on': self.mod_am.get(), 'am_depth': self.slider_am.get(),
                'fm_on': self.mod_fm.get(), 'fm_dev': self.slider_fm_dev.get()
            }
            if p['duration'] <= 0 or p['Fs'] <= 0 or p['Vpp'] < 0: raise ValueError(
                "Duração, Fs devem ser > 0 e Vpp >= 0.")
            p['N'] = int(p['duration'] * p['Fs'])
            if p['N'] < 16: raise ValueError("Combinação de Duração e Fs resulta em poucos pontos (< 16).")
            if p['N'] % 2 != 0: p['N'] += 1

            max_freq = max(p.get('Fc', 0), p.get('Fm', 0))
            if max_freq >= p['Fs'] / 2: raise ValueError(
                f"Nyquist violado! Frequência máxima ({self._format_value(max_freq, 'Hz')}) deve ser < Fs/2 ({self._format_value(p['Fs'] / 2, 'Hz')}).")

            self.after(0, lambda: self.slider_fm_dev.configure(to=max(0.1, p['Fs'] / 4)))
            return p
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Erro de Entrada", str(e)))
            self.after(0, lambda e=e: self.set_status(f"❌ Erro de Entrada: {e}", "red"))
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
            return np.random.normal(0, 1, p['N'])
        elif wf == "Exp Decaimento":
            return np.exp(-t / (p['duration'] / 5)) * np.sin(2 * np.pi * Fc * t)
        elif wf == "Passo (step)":
            return np.heaviside(t - t[p['N'] // 4], 1.0)
        elif wf == "Rampa":
            return t / t[-1] if t[-1] > 0 else t
        elif wf == "Parábola":
            return (t / t[-1]) ** 2 if t[-1] > 0 else t ** 2
        elif wf == "Impulso":
            return unit_impulse(p['N'], 'mid')
        elif wf == "Tangente":
            y = np.tan(np.pi * Fc * t); return np.clip(y, -10, 10)
        elif wf == "Sinc":
            return np.sinc(2 * Fc * (t - p['duration'] / 2))
        elif wf == "Gaussiana":
            return gausspulse(t - p['duration'] / 2, fc=Fc, bw=0.5)
        elif wf == "Chirp Linear":
            return chirp(t, f0=Fc, f1=Fc / 2, t1=t[-1], method='linear')
        elif wf == "Chirp Quadrático":
            return chirp(t, f0=Fc, f1=Fc / 10, t1=t[-1], method='quadratic')
        elif wf == "Bessel":
            return jv(0, 2 * np.pi * Fc * (t - p['duration'] / 2))
        else:
            return np.zeros_like(t)

    def _apply_modulation(self, p, carrier_signal, t):
        if p['am_on']:
            modulator = np.sin(2 * np.pi * p['Fm'] * t)
            return carrier_signal * (1 + p['am_depth'] * modulator)

        if p['fm_on']:
            modulator_integrated = np.cumsum(np.sin(2 * np.pi * p['Fm'] * t)) / p['Fs']
            phase = 2 * np.pi * p['Fc'] * t + 2 * np.pi * p['fm_dev'] * modulator_integrated

            wf = p['waveform']
            if wf in ["Seno", "Cosseno", "Ruído Branco", "Passo (step)"]:
                return np.sin(phase)
            elif wf == "Quadrada" or wf == "Pulso":
                return square(phase)
            elif wf == "Triangular":
                return sawtooth(phase, 0.5)
            elif wf == "Dente de Serra":
                return sawtooth(phase)
            else:
                return np.sin(phase)

        return carrier_signal

    def _update_plots(self):
        if not self.last_data: return
        for ax in [self.ax_time, self.ax_freq]: ax.clear()
        self.clear_markers('all', update_ui=False)

        self.time_plot_line, = self.ax_time.plot(self.last_data['t'], self.last_data['y'], color="cyan")
        self.ax_time.set_title("Domínio do Tempo", color='white');
        self.ax_time.grid(True, linestyle='--', alpha=0.5);
        self.ax_time.tick_params(colors='white')

        self.freq_plot_line, = self.ax_freq.plot(self.last_data['f'], self.last_data['Y'], color="orange")
        self.ax_freq.set_title("Domínio da Frequência (FFT)", color='white');
        self.ax_freq.grid(True, linestyle='--', alpha=0.5);
        self.ax_freq.tick_params(colors='white')
        self.ax_freq.set_ylabel("|Y(f)|", color='white')

        self.after(0, self.update_analysis_panels)
        self.after(0, self._auto_zoom_initial)
        self.after(0, self._update_y_scales)
        self.after(0, self._update_marker_panel)
        self.set_status("✅ Gráficos atualizados!", "lightgreen")

    def _auto_zoom_initial(self):
        params = self.last_data.get('params')
        if not params or params.get('Fc', 0) <= 0: self.reset_zoom(); return

        total_duration = params.get('duration', self.last_data['t'][-1])
        carrier_period = 1 / params['Fc']
        desired_view_width = 8 * carrier_period
        view_width = min(desired_view_width, total_duration)

        self.ax_time.set_xlim(0, view_width)
        self.zoom_time_x.set(view_width / total_duration if total_duration > 0 else 1.0)
        self.canvas.draw()

    def _update_y_scales(self, _=None):
        if not self.last_data: return

        y_data = self.last_data.get('y', np.array([0]))
        vpp = np.ptp(y_data) if len(y_data) > 0 else 2.0
        y_mean = np.mean(y_data) if len(y_data) > 0 else 0

        zoom_t = self.time_zoom_y.get();
        pan_t = self.time_pan_y.get()
        center_t = pan_t * (vpp / 2) + y_mean;
        height_t = (vpp / 2) * zoom_t
        self.ax_time.set_ylim(center_t - height_t, center_t + height_t)

        max_Y = np.max(self.last_data['Y']) if len(self.last_data.get('Y', [])) > 0 else 1.0
        zoom_f = self.freq_zoom_y.get();
        pan_f = self.freq_pan_y.get()
        center_f = pan_f * (max_Y / 2) + (max_Y / 2);
        height_f = (max_Y / 2) * zoom_f
        self.ax_freq.set_ylim(center_f - height_f, center_f + height_f)

        self.canvas.draw_idle()

    def update_time_zoom(self, val):
        if not self.last_data: return
        total_width = self.last_data['t'][-1] - self.last_data['t'][0]
        new_width = total_width * float(val) if val > 0.001 else total_width * 0.001
        current_center = np.mean(self.ax_time.get_xlim())
        self.ax_time.set_xlim(current_center - new_width / 2, current_center + new_width / 2)
        self.canvas.draw_idle()

    def reset_zoom(self):
        if not self.last_data: return
        self._auto_zoom_initial()
        self.time_zoom_y.set(1.0);
        self.time_pan_y.set(0)
        self.freq_zoom_y.set(1.0);
        self.freq_pan_y.set(0)
        self._update_y_scales()
        self.canvas.draw()

    def import_wav(self):
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not filepath: return
        self.set_status(f"⏳ Importando {os.path.basename(filepath)}...", "yellow")
        try:
            with open(filepath, 'rb') as f:
                file_bytes = f.read()
            if len(file_bytes) < 4000: raise ValueError("Arquivo WAV muito pequeno.")

            v_idx = file_bytes[4];
            t_idx = file_bytes[22]
            v_scale = VOLT_LIST[v_idx];
            t_scale = TIME_LIST[t_idx]
            self.imported_hw_info = {'volt_scale': v_scale, 'time_scale': t_scale}

            time_per_div = t_scale[2]
            total_duration = time_per_div * 10
            fs = FNIRSI_SAMPLES / total_duration

            data_bytes = file_bytes[1000: 1000 + FNIRSI_SAMPLES * 2]
            y_raw = np.frombuffer(data_bytes, dtype=np.int16)

            y_volts = (y_raw.astype(np.float32) - 200) * (v_scale[0] / 50.0)
            if v_scale[1] == 'mV': y_volts *= 1e-3

            t = np.linspace(0, total_duration, FNIRSI_SAMPLES, endpoint=False)

            self.after(0, lambda: self._show_wav_preview(t, y_volts, os.path.basename(filepath)))
            self.set_status(f"✅ Arquivo {os.path.basename(filepath)} pronto para análise.", "lightgreen")

        except Exception as e:
            messagebox.showerror("Erro de Importação WAV", f"Não foi possível decodificar o arquivo Fnirsi.\nErro: {e}")
            self.set_status(f"❌ Falha na importação.", "red")

    def _export_fnirsi_wav(self):
        if not self.last_data: messagebox.showerror("Erro", "Gere um sinal primeiro."); return

        y_to_export = self.last_data['y']

        self._open_sampling_window(self, self.last_data['t'], y_to_export, "sinal_gerado.wav")

    def _on_mouse_press(self, event):
        self.last_click_event = event
        if event.button == 3:
            if hasattr(event, 'guiEvent'): self.menu.post(event.guiEvent.x_root, event.guiEvent.y_root)

    def _on_pick_event(self, event):
        if event.mouseevent.button != 1: return
        self.dragging_marker = event.artist

    def _on_mouse_release(self, event):
        self.dragging_marker = None

    def _on_mouse_motion(self, event):
        if not self.dragging_marker or not event.inaxes: return
        is_vertical = len(self.dragging_marker.get_xdata()) > 1 and self.dragging_marker.get_xdata()[0] == \
                      self.dragging_marker.get_xdata()[1]

        if is_vertical and event.xdata is not None:
            self.dragging_marker.set_xdata([event.xdata, event.xdata])
        elif not is_vertical and event.ydata is not None:
            self.dragging_marker.set_ydata([event.ydata, event.ydata])

        self._update_marker_panel()
        self.canvas.draw_idle()

    def add_marker(self, kind):
        ev = self.last_click_event
        if not ev or not ev.inaxes: return
        ax = self.ax_time if kind.startswith('time') else self.ax_freq
        if ev.inaxes != ax: return

        if len(self.markers[kind]) >= 2: self.set_status("⚠️ Máximo de 2 marcadores por tipo", "orange"); return

        color = 'red' if kind == 'time_v' else 'magenta' if kind == 'time_h' else 'lime' if kind == 'freq_v' else 'yellow'

        if kind.endswith('_v'):
            line = ax.axvline(x=ev.xdata, color=color, linestyle='--', picker=5)
        else:
            line = ax.axhline(y=ev.ydata, color=color, linestyle='--', picker=5)

        self.markers[kind].append(line)
        self._update_marker_panel();
        self.canvas.draw();
        self.set_status(f"✅ Marcador {kind} adicionado", "lightgreen")

    def clear_markers(self, which='all', update_ui=True):
        keys_to_clear = list(self.markers.keys()) if which == 'all' else [which]
        for k in keys_to_clear:
            if k in self.markers:
                for marker in self.markers[k]:
                    marker.set_visible(False)
                    # Tenta remover do eixo, se falhar, ignora. A visibilidade é o mais importante.
                    try:
                        if marker in self.ax_time.lines:
                            self.ax_time.lines.remove(marker)
                        elif marker in self.ax_freq.lines:
                            self.ax_freq.lines.remove(marker)
                    except (ValueError, AttributeError):
                        pass
                self.markers[k].clear()
        if update_ui:
            self._update_marker_panel()
            self.canvas.draw()
            if which == 'all': self.set_status("🧹 Todos os marcadores foram limpos", "yellow")

    def _update_marker_panel(self):
        def get_pos(kind, axis):
            return sorted([m.get_xdata()[0] if axis == 'x' else m.get_ydata()[0] for m in self.markers[kind]])

        tv = get_pos('time_v', 'x');
        th = get_pos('time_h', 'y')
        fv = get_pos('freq_v', 'x');
        fh = get_pos('freq_h', 'y')

        x1, x2, dx = (tv[0], tv[1], tv[1] - tv[0]) if len(tv) == 2 else (tv[0] if len(tv) == 1 else '---', '---', '---')
        y1, y2, dy = (th[0], th[1], th[1] - th[0]) if len(th) == 2 else (th[0] if len(th) == 1 else '---', '---', '---')
        f1, f2, df = (fv[0], fv[1], fv[1] - fv[0]) if len(fv) == 2 else (fv[0] if len(fv) == 1 else '---', '---', '---')
        m1, m2, dm = (fh[0], fh[1], fh[1] - fh[0]) if len(fh) == 2 else (fh[0] if len(fh) == 1 else '---', '---', '---')

        self.lbl_x1.configure(text=f"X1: {self._format_value(x1, 's')}")
        self.lbl_x2.configure(text=f"X2: {self._format_value(x2, 's')}")
        self.lbl_dx.configure(text=f"ΔX: {self._format_value(dx, 's')}")
        self.lbl_y1.configure(text=f"Y1: {self._format_value(y1, 'V')}")
        self.lbl_y2.configure(text=f"Y2: {self._format_value(y2, 'V')}")
        self.lbl_dy.configure(text=f"ΔY: {self._format_value(dy, 'V')}")
        self.lbl_f1.configure(text=f"F1: {self._format_value(f1, 'Hz')}")
        self.lbl_f2.configure(text=f"F2: {self._format_value(f2, 'Hz')}")
        self.lbl_df.configure(text=f"ΔF: {self._format_value(df, 'Hz')}")
        self.lbl_m1.configure(text=f"|Y1|: {self._format_value(m1)}")
        self.lbl_m2.configure(text=f"|Y2|: {self._format_value(m2)}")
        self.lbl_dm.configure(text=f"Δ|Y|: {self._format_value(dm)}")

    def update_analysis_panels(self):
        if not self.last_data: return
        y, t, f, Y, params = self.last_data['y'], self.last_data['t'], self.last_data['f'], self.last_data[
            'Y'], self.last_data.get('params', {})

        try:
            vpp = np.ptp(y) if len(y) > 0 else 0
            rms = np.sqrt(np.mean(y ** 2)) if len(y) > 0 else 0
            mean = np.mean(y) if len(y) > 0 else 0
            peak = np.max(np.abs(y)) if len(y) > 0 else 0
            crest_factor = peak / rms if rms > 1e-9 else float('inf')
            self.time_analysis_vars["Vpp"].set(f"{vpp:.4f} V")
            self.time_analysis_vars["Vrms"].set(f"{rms:.4f} V")
            self.time_analysis_vars["Vavg"].set(f"{mean:.4f} V")
            self.time_analysis_vars["Fator Crista"].set(f"{crest_factor:.2f}")
            self.time_analysis_vars["Curtose"].set(f"{kurtosis(y):.2f}" if len(y) > 3 else "N/A")
            self.time_analysis_vars["Assimetria"].set(f"{skew(y):.2f}" if len(y) > 2 else "N/A")
            zero_crossings = np.where(np.diff(np.signbit(y)))[0]
            self.time_analysis_vars["Cruzamentos Zero"].set(f"{len(zero_crossings)}")
            if len(zero_crossings) > 1:
                freq_est = len(zero_crossings) / (2 * (t[zero_crossings[-1]] - t[zero_crossings[0]]))
                self.time_analysis_vars["Freq. Estimada"].set(self._format_value(freq_est, 'Hz'))
            else:
                self.time_analysis_vars["Freq. Estimada"].set("---")
        except Exception:
            for var in self.time_analysis_vars.values(): var.set("---")

        try:
            peaks, props = find_peaks(Y, height=np.max(Y) * 0.01 if np.max(Y) > 0 else None, distance=5)
            if len(peaks) > 0:
                sorted_peaks = sorted(zip(peaks, props['peak_heights']), key=lambda item: item[1], reverse=True)
                fund_idx, fund_amp = sorted_peaks[0]
                fund_freq = abs(f[fund_idx])

                harmonics_amp_sq = sum(Y[p] ** 2 for p in peaks if abs(f[p]) > fund_freq * 1.1)
                thd = np.sqrt(harmonics_amp_sq) / fund_amp if fund_amp > 1e-9 else 0

                noise_floor = np.mean(Y[Y < np.max(Y) * 0.01]) if np.max(Y) > 0 else 0
                snr = 10 * np.log10(fund_amp ** 2 / noise_floor ** 2) if noise_floor > 1e-9 else float('inf')
                highest_spur_amp = sorted_peaks[1][1] if len(sorted_peaks) > 1 else noise_floor
                sfdr = 20 * np.log10(fund_amp / highest_spur_amp) if highest_spur_amp > 1e-9 else float('inf')

                self.freq_analysis_vars["Freq. Fund."].set(f"{self._format_value(fund_freq, 'Hz')}")
                self.freq_analysis_vars["Amp. Fund."].set(f"{fund_amp:.3f}")
                self.freq_analysis_vars["THD"].set(f"{thd * 100:.2f} %")
                self.freq_analysis_vars["SNR"].set(f"{snr:.2f} dB")
                self.freq_analysis_vars["SFDR"].set(f"{sfdr:.2f} dBc")
                self.freq_analysis_vars["Piso de Ruído"].set(f"{self._format_value(noise_floor)}")
            else:
                for var in self.freq_analysis_vars.values(): var.set("---")
        except Exception:
            for var in self.freq_analysis_vars.values(): var.set("---")

    def _format_value(self, value, unit=""):
        if not isinstance(value, (int, float, np.number)) or np.isnan(value): return "---"
        val = abs(value)
        sign = "-" if value < 0 else ""
        if unit == 's':
            if val < 1e-6: return f"{sign}{val * 1e9:.2f} ns"
            if val < 1e-3: return f"{sign}{val * 1e6:.2f} µs"
            if val < 1: return f"{sign}{val * 1e3:.2f} ms"
            return f"{sign}{val:.3f} s"
        if unit == 'Hz':
            if val >= 1e9: return f"{sign}{val / 1e9:.3f} GHz"
            if val >= 1e6: return f"{sign}{val / 1e6:.3f} MHz"
            if val >= 1e3: return f"{sign}{val / 1e3:.3f} kHz"
            return f"{sign}{val:.2f} Hz"
        if unit == 'V':
            if val < 1: return f"{sign}{val * 1e3:.2f} mV"
            return f"{sign}{val:.3f} V"
        return f"{value:.3f}"

    def export_data(self):
        if not self.last_data: messagebox.showerror("Erro", "Gere um sinal primeiro."); return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json"), ("CSV", "*.csv")])
        if not path: return
        try:
            data_to_save = {k: v.tolist() for k, v in self.last_data.items() if isinstance(v, np.ndarray)}
            if path.endswith(".json"):
                with open(path, "w") as f:
                    json.dump(data_to_save, f, indent=2)
            else:
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data_to_save.keys())
                    writer.writerows(zip(*data_to_save.values()))
            self.set_status(f"📁 Dados exportados para {os.path.basename(path)}", "lightblue")
        except Exception as e:
            messagebox.showerror("Erro de Exportação", str(e));
            self.set_status(f"❌ Erro ao exportar: {e}", "red")


if __name__ == "__main__":
    app = SignalGeneratorApp()
    app.mainloop()