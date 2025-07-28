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
import csv
import json
from concurrent.futures import ThreadPoolExecutor

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

INTERP_THRESHOLD = 100
INTERP_SAMPLES = 500
UNIT_MULTIPLIERS = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


class SignalGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gerador de Sinais Avançado")
        self.geometry("1400x950")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.last_data = {}
        self.markers = {
            'time_v': [],
            'time_h': [],
            'freq_v': [],
            'freq_h': []
        }
        self.dragging_marker = None
        self.dragging_type = None
        self.last_click_event = None
        self.time_plot_line = None
        self.mod_am = tk.BooleanVar(value=False)
        self.mod_fm = tk.BooleanVar(value=False)

        self._build_sidebar()
        self._build_plot_area()
        self._build_context_menu()
        self._build_marker_panel()
        self._build_status_bar()
