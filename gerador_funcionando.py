import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import serial
import serial.tools.list_ports
import threading
import time
import queue

# ========= Aparência =========
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ========= Constantes =========
BAUDRATE = 921600         # CDC ignora baudrate, ok
READ_SLEEP_S = 0.01
CMD_EOL = "\r\n"

MAX_FREQ_HZ = 3900
MIN_FREQ_HZ = 1
DEFAULT_FREQ = 1000

WAVE_TYPES = ["SINE", "SQUARE", "TRI", "SAWUP", "SAWDN"]
WIN_TYPES = ["NONE", "HANN", "BLACKMAN", "NUTTALL"]

CONNECT_STARTUP_DELAY = 0.30
WRITE_GAP_MS = 10
FREQ_DEBOUNCE_MS = 160
WIN_DEBOUNCE_MS = 160

# Sequência de rearm do AWG: tempos entre comandos
AWG_REARM_STEP_MS = 60      # gap entre "DAC 0" -> "WAVEWIN" -> "WAVE ..."
AWG_RETRY_BACKOFF_MS = 200  # se firmware responder "ERROR: DAC start", tenta mais 1x depois disso


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class HardwareCommunicator:
    """Serial CDC com RX/TX em threads + fila de TX com pacing."""
    def __init__(self, data_queue, status_cb=None):
        self.ser = None
        self.port = None
        self.running = False
        self.rx_thread = None
        self.tx_thread = None
        self.data_queue = data_queue
        self.status_cb = status_cb or (lambda *_: None)
        self.tx_count = 0
        self.rx_count = 0
        self.tx_queue = queue.Queue()
        self._last_write_ts = 0.0

    def connect(self, port):
        try:
            self.port = port
            self.ser = serial.Serial(
                self.port,
                BAUDRATE,
                timeout=0.2,
                write_timeout=1.0,
                exclusive=True
            )
            try:
                self.ser.setDTR(True)
                self.ser.setRTS(False)
            except Exception:
                pass

            time.sleep(CONNECT_STARTUP_DELAY)

            try:
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception:
                pass

            self.running = True
            self.rx_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.rx_thread.start()
            self.tx_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self.tx_thread.start()

            # Sondagem
            self.send_command("PING", enqueue=True)
            self.status_cb(connected=True, port=self.port)
            return True, f"Conectado a {self.port} (CDC)"
        except serial.SerialException as e:
            self.status_cb(connected=False, port=None)
            return False, f"Erro ao conectar: {e}"

    def disconnect(self):
        self.running = False
        try:
            while not self.tx_queue.empty():
                self.tx_queue.get_nowait()
        except Exception:
            pass

        if self.rx_thread and self.rx_thread.is_alive():
            self.rx_thread.join(timeout=0.5)
        if self.tx_thread and self.tx_thread.is_alive():
            self.tx_thread.join(timeout=0.5)

        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.status_cb(connected=False, port=None)
        return True, "Desconectado."

    def _reader_loop(self):
        buff = bytearray()
        while self.running:
            try:
                if self.ser:
                    chunk = self.ser.read(256)
                    if chunk:
                        buff.extend(chunk)
                        while True:
                            nl = buff.find(b'\n')
                            if nl < 0:
                                break
                            line = buff[:nl+1]
                            del buff[:nl+1]
                            try:
                                text = line.decode("utf-8", errors="ignore").strip()
                            except Exception:
                                text = ""
                            if text:
                                self.rx_count += len(line)
                                self.data_queue.put(text)
                                self.status_cb(rx=self.rx_count)
                time.sleep(READ_SLEEP_S)
            except (serial.SerialException, OSError):
                self.data_queue.put("DISCONNECTED")
                self.running = False
                break

    def _writer_loop(self):
        while self.running:
            try:
                cmd = self.tx_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not self.running or not self.ser or not self.ser.is_open:
                continue
            now = time.time()
            dt = (WRITE_GAP_MS/1000.0) - (now - self._last_write_ts)
            if dt > 0:
                time.sleep(dt)
            try:
                payload = (cmd + CMD_EOL).encode("utf-8")
                self.ser.write(payload)
                self._last_write_ts = time.time()
                self.tx_count += len(payload)
                self.status_cb(tx=self.tx_count, last_tx=cmd)
            except serial.SerialTimeoutException:
                try:
                    self.ser.write(payload)
                    self._last_write_ts = time.time()
                    self.tx_count += len(payload)
                    self.status_cb(tx=self.tx_count, last_tx=cmd)
                except Exception:
                    self.data_queue.put("DISCONNECTED")
                    self.running = False
            except serial.SerialException:
                self.data_queue.put("DISCONNECTED")
                self.running = False

    def send_command(self, command: str, enqueue=False) -> bool:
        if not (self.ser and self.ser.is_open):
            return False
        if enqueue:
            try:
                self.tx_queue.put_nowait(command)
                return True
            except queue.Full:
                return False
        try:
            payload = (command + CMD_EOL).encode("utf-8")
            self.ser.write(payload)
            self.tx_count += len(payload)
            self.status_cb(tx=self.tx_count, last_tx=command)
            return True
        except serial.SerialException:
            self.data_queue.put("DISCONNECTED")
            self.running = False
            return False


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Controlador Pyboard v1.1 — AWG + ACCEL + CLI")
        self.geometry("1020x760")
        self.minsize(960, 700)

        # Estado
        self.data_queue = queue.Queue()
        self.communicator = HardwareCommunicator(self.data_queue, status_cb=self._update_status)
        self.is_wave_running = False
        self.is_accel_on = False
        self.hb_on = True

        self._freq_send_after_id = None
        self._win_send_after_id = None

        # Sequenciador AWG
        self._awg_apply_token = 0
        self._awg_retry_armed = False
        self._last_wave_cmd_text = None  # só pra log/evitar flood se parar

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        self._build_connection_bar(row=0)
        self._build_sys_panel(row=1)
        self._build_leds_panel(row=2)
        self._build_awg_panel(row=3)
        self._build_accel_panel(row=4)
        self._build_console(row=5)
        self._build_statusbar()

        # Tarefas
        self._refresh_ports()
        self._process_serial_queue()

        # Fechamento limpo
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI builders ----------
    def _build_connection_bar(self, row):
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=(10, 6), sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Porta COM:").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.port_menu = ctk.CTkOptionMenu(frame, values=["(buscando...)"])
        self.port_menu.grid(row=0, column=1, padx=6, pady=6, sticky="ew")

        self.refresh_btn = ctk.CTkButton(frame, text="Atualizar", width=90, command=self._refresh_ports)
        self.refresh_btn.grid(row=0, column=2, padx=6, pady=6)

        self.connect_button = ctk.CTkButton(frame, text="Conectar", width=110, command=self._toggle_connection)
        self.connect_button.grid(row=0, column=3, padx=6, pady=6)

        self.ping_btn = ctk.CTkButton(frame, text="PING", width=70, command=lambda: self._send("PING"))
        self.ping_btn.grid(row=0, column=4, padx=(12, 6), pady=6)
        self.help_btn = ctk.CTkButton(frame, text="HELP", width=70, command=lambda: self._send("HELP"))
        self.help_btn.grid(row=0, column=5, padx=6, pady=6)

    def _build_sys_panel(self, row):
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=6, sticky="ew")
        frame.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(frame, text="Sistema", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=6, pady=(8, 6), sticky="w"
        )

        self.hb_switch = ctk.CTkSwitch(frame, text="Heartbeat (LED azul)", command=self._toggle_hb)
        self.hb_switch.select()  # default: ON
        self.hb_switch.grid(row=1, column=0, padx=8, pady=6, sticky="w")

        self.sysinfo_btn = ctk.CTkButton(frame, text="SYS INFO", width=100, command=lambda: self._send("SYS"))
        self.sysinfo_btn.grid(row=1, column=1, padx=8, pady=6)

        self.reset_btn = ctk.CTkButton(frame, text="RESET MCU", fg_color="#8b0000",
                                       hover_color="#a40000", width=120,
                                       command=self._reset_mcu)
        self.reset_btn.grid(row=1, column=2, padx=8, pady=6)

    def _build_leds_panel(self, row):
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=6, sticky="ew")
        ctk.CTkLabel(frame, text="LEDs", font=ctk.CTkFont(weight="bold")).pack(pady=(8, 4))

        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(pady=(4, 8), fill="x")

        self.led_switches = {}
        labels = ["Vermelho (0)", "Verde (1)", "Amarelo (2)", "Azul (3)"]
        for i, txt in enumerate(labels):
            sw = ctk.CTkSwitch(inner, text=txt, command=lambda n=i: self._toggle_led(n))
            sw.pack(side="left", expand=True, padx=12, pady=6)
            self.led_switches[i] = sw

    def _build_awg_panel(self, row):
        self.awg_frame = ctk.CTkFrame(self)
        f = self.awg_frame
        f.grid(row=row, column=0, padx=10, pady=6, sticky="ew")
        for c in range(7):
            f.grid_columnconfigure(c, weight=1 if c in (1, 2, 3, 4) else 0)

        ctk.CTkLabel(f, text="Gerador de Onda (DAC via TIM2/DMA)", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=7, pady=(8, 10)
        )

        # Forma
        ctk.CTkLabel(f, text="Forma:").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        self.wave_menu = ctk.CTkOptionMenu(f, values=WAVE_TYPES, command=self._wave_type_changed)
        self.wave_menu.set("SINE")
        self.wave_menu.grid(row=1, column=1, padx=6, pady=6, sticky="w")

        # Janela/taper
        ctk.CTkLabel(f, text="Janela:").grid(row=1, column=2, padx=6, pady=6, sticky="w")
        self.win_menu = ctk.CTkOptionMenu(f, values=WIN_TYPES, command=self._window_changed)
        self.win_menu.set("NONE")
        self.win_menu.grid(row=1, column=3, padx=6, pady=6, sticky="w")

        ctk.CTkLabel(f, text="Taper (%)").grid(row=1, column=4, padx=6, pady=6, sticky="e")
        self.taper_slider = ctk.CTkSlider(f, from_=0, to=100, number_of_steps=100, command=self._on_taper_slider)
        self.taper_slider.set(50)
        self.taper_slider.grid(row=1, column=5, padx=6, pady=6, sticky="ew")
        self.taper_val = ctk.CTkLabel(f, text="50%")
        self.taper_val.grid(row=1, column=6, padx=6, pady=6, sticky="w")

        # Freq slider + entry
        ctk.CTkLabel(f, text=f"Frequência (Hz) [{MIN_FREQ_HZ}..{MAX_FREQ_HZ}]").grid(row=2, column=0, padx=6, pady=6, sticky="w")
        self.freq_slider = ctk.CTkSlider(f, from_=MIN_FREQ_HZ, to=MAX_FREQ_HZ,
                                         number_of_steps=MAX_FREQ_HZ-MIN_FREQ_HZ, command=self._on_freq_slider)
        self.freq_slider.set(DEFAULT_FREQ)
        self.freq_slider.grid(row=2, column=1, columnspan=4, padx=6, pady=6, sticky="ew")

        self.freq_entry = ctk.CTkEntry(f, width=120)
        self.freq_entry.insert(0, str(DEFAULT_FREQ))
        self.freq_entry.grid(row=2, column=5, padx=6, pady=6, sticky="e")
        ctk.CTkButton(f, text="Aplicar", width=80, command=self._apply_freq_entry).grid(row=2, column=6, padx=6, pady=6)

        # Start/Stop
        self.start_btn = ctk.CTkButton(f, text="Iniciar", width=120, command=self._toggle_wave)
        self.start_btn.grid(row=3, column=0, padx=6, pady=(10, 10), sticky="w")

    def _build_accel_panel(self, row):
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=6, sticky="nsew")
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text="Acelerômetro (MMA7660)", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=3, pady=(8, 6)
        )
        self.accel_switch = ctk.CTkSwitch(frame, text="Ativar leitura contínua", command=self._toggle_accel)
        self.accel_switch.grid(row=1, column=0, columnspan=3, pady=8)

        self.pb_x, self.lbl_x = self._axis_row(frame, "Eixo X:", 2)
        self.pb_y, self.lbl_y = self._axis_row(frame, "Eixo Y:", 3)
        self.pb_z, self.lbl_z = self._axis_row(frame, "Eixo Z:", 4)

    def _axis_row(self, parent, text, row):
        ctk.CTkLabel(parent, text=text).grid(row=row, column=0, padx=10, pady=5, sticky="w")
        bar = ctk.CTkProgressBar(parent); bar.set(0.5)
        bar.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        lab = ctk.CTkLabel(parent, text="0", width=40)
        lab.grid(row=row, column=2, padx=10, pady=5)
        return bar, lab

    def _build_console(self, row):
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=(6, 0), sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        self.console = ctk.CTkTextbox(frame, height=240)
        self.console.grid(row=0, column=0, padx=6, pady=(6, 2), sticky="nsew")
        self.console.configure(state="disabled")

        input_frame = ctk.CTkFrame(frame, fg_color="transparent")
        input_frame.grid(row=1, column=0, padx=6, pady=(2, 6), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        self.entry_cmd = ctk.CTkEntry(input_frame, placeholder_text="Digite um comando (ex.: HELP, SYS, WAVE SINE 1000)...")
        self.entry_cmd.grid(row=0, column=0, padx=(0, 6), pady=4, sticky="ew")
        btn_send = ctk.CTkButton(input_frame, text="Enviar", width=100, command=self._send_manual)
        btn_send.grid(row=0, column=1, padx=(0, 0), pady=4)

    def _build_statusbar(self):
        self.status = ctk.CTkLabel(self, text="Desconectado", anchor="w")
        self.status.grid(row=6, column=0, padx=10, pady=(6, 8), sticky="ew")
        self._status_data = {"connected": False, "port": None, "tx": 0, "rx": 0, "last": "", "last_tx": ""}

    # ---------- Conexão ----------
    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_menu.configure(values=ports if ports else ["(nenhuma)"])
        if ports:
            cur = self.port_menu.get()
            if cur not in ports:
                self.port_menu.set(ports[0])
        else:
            self.port_menu.set("(nenhuma)")

    def _toggle_connection(self):
        if self.communicator.running:
            self._send("ACCEL 0")
            self._send("DAC 0")
            _, msg = self.communicator.disconnect()
            self._log(f"[pc] {msg}")
            self.connect_button.configure(text="Conectar")
        else:
            port = self.port_menu.get()
            if not port or port.startswith("("):
                messagebox.showerror("Erro", "Nenhuma porta serial selecionada.")
                return
            ok, msg = self.communicator.connect(port)
            self._log(f"[pc] {msg}")
            if ok:
                self.connect_button.configure(text="Desconectar")
                self.after(350, lambda: self._send("HELP"))
                self.after(500, lambda: self._send(f"SYS HB {1 if self.hb_on else 0}"))
                # aplica janela/taper default no firmware (sem ligar AWG)
                self.after(650, lambda: self._send(f"WAVEWIN {self.win_menu.get()} {int(self.taper_slider.get())}"))

    # ---------- Sistema ----------
    def _toggle_hb(self):
        self.hb_on = bool(self.hb_switch.get())
        self._send(f"SYS HB {1 if self.hb_on else 0}")

    def _reset_mcu(self):
        if messagebox.askyesno("Reset", "Tem certeza que deseja resetar o MCU?"):
            self._send("ACCEL 0")
            self._send("DAC 0")
            self._send("SYS RESET")

    # ---------- LEDs ----------
    def _toggle_led(self, n: int):
        state = 1 if self.led_switches[n].get() else 0
        self._send(f"LED {n} {state}")

    # ---------- AWG (com rearm atômico) ----------
    def _wave_type_changed(self, *_):
        if self.is_wave_running:
            self._queue_awg_rearm()

    def _window_changed(self, *_):
        # Mesmo sem AWG rodando, configuramos a janela no firmware
        if self._win_send_after_id:
            self.after_cancel(self._win_send_after_id)
        self._win_send_after_id = self.after(WIN_DEBOUNCE_MS, self._apply_window_and_maybe_rearm)

    def _apply_window_and_maybe_rearm(self):
        self._send(f"WAVEWIN {self.win_menu.get()} {int(self.taper_slider.get())}")
        if self.is_wave_running:
            self._queue_awg_rearm()

    def _on_taper_slider(self, value):
        self.taper_val.configure(text=f"{int(value)}%")
        if self._win_send_after_id:
            self.after_cancel(self._win_send_after_id)
        self._win_send_after_id = self.after(WIN_DEBOUNCE_MS, self._apply_window_and_maybe_rearm)

    def _on_freq_slider(self, value):
        self.freq_entry.delete(0, tk.END)
        self.freq_entry.insert(0, f"{int(value)}")
        if self._freq_send_after_id:
            self.after_cancel(self._freq_send_after_id)
        self._freq_send_after_id = self.after(FREQ_DEBOUNCE_MS, self._freq_maybe_send)

    def _freq_maybe_send(self):
        self._freq_send_after_id = None
        if self.is_wave_running:
            self._queue_awg_rearm()

    def _apply_freq_entry(self):
        try:
            f = float(self.freq_entry.get())
        except ValueError:
            messagebox.showerror("Erro", "Frequência inválida.")
            return
        f = clamp(f, MIN_FREQ_HZ, MAX_FREQ_HZ)
        self.freq_slider.set(f)
        if self.is_wave_running:
            self._queue_awg_rearm()

    def _toggle_wave(self):
        if not self.is_wave_running:
            self.is_wave_running = True
            self.start_btn.configure(text="Parar")
            self._queue_awg_rearm()
        else:
            self.is_wave_running = False
            self.start_btn.configure(text="Iniciar")
            self._last_wave_cmd_text = None
            self._send("DAC 0")

    def _queue_awg_rearm(self):
        """Coalescer mudanças e rearmar AWG: DAC 0 -> (WAVEWIN) -> WAVE ... com cancel de sequências antigas."""
        if not self.communicator.running:
            return

        try:
            f = int(float(self.freq_entry.get() or DEFAULT_FREQ))
        except ValueError:
            f = DEFAULT_FREQ
        f = clamp(f, MIN_FREQ_HZ, MAX_FREQ_HZ)
        w = self.wave_menu.get()
        win = self.win_menu.get()
        taper = int(self.taper_slider.get())

        # Incrementa token para invalidar sequências anteriores
        self._awg_apply_token += 1
        my_token = self._awg_apply_token
        self._awg_retry_armed = False  # limpa retry

        # Monta comandos
        cmd_stop = "DAC 0"
        cmd_win  = f"WAVEWIN {win} {taper}"
        cmd_wave = f"WAVE {w} {f}"
        self._last_wave_cmd_text = cmd_wave

        # Dispara com pequenos gaps, respeitando token
        self._send(cmd_stop)  # stop imediato
        self.after(AWG_REARM_STEP_MS, lambda: self._send_if_token(cmd_win, my_token))
        self.after(2*AWG_REARM_STEP_MS, lambda: self._send_if_token(cmd_wave, my_token))

    def _send_if_token(self, cmd, token):
        if token != self._awg_apply_token:
            return  # sequência antiga, ignorar
        self._send(cmd)

    # ---------- ACCEL ----------
    def _toggle_accel(self):
        self.is_accel_on = bool(self.accel_switch.get())
        self._send(f"ACCEL {1 if self.is_accel_on else 0}")

    # ---------- Serial processing ----------
    def _process_serial_queue(self):
        try:
            while not self.data_queue.empty():
                msg = self.data_queue.get_nowait()
                if msg == "DISCONNECTED":
                    self._log("[pc] Conexão perdida.")
                    self.connect_button.configure(text="Conectar")
                    self._update_status(connected=False, port=None)
                    continue

                self._log(f"Pyboard: {msg}")

                if msg.startswith("A:"):
                    try:
                        x, y, z = map(int, msg[2:].split(","))
                        self.pb_x.set((x + 32) / 63)
                        self.pb_y.set((y + 32) / 63)
                        self.pb_z.set((z + 32) / 63)
                        self.lbl_x.configure(text=str(x))
                        self.lbl_y.configure(text=str(y))
                        self.lbl_z.configure(text=str(z))
                    except Exception:
                        pass
                elif msg == "ERROR: DAC start":
                    # Se o firmware acusar falha, tente um único rearm com backoff curto
                    if self.is_wave_running and not self._awg_retry_armed:
                        self._awg_retry_armed = True
                        tok_before = self._awg_apply_token
                        self.after(AWG_RETRY_BACKOFF_MS, lambda: self._retry_awg_if_still(tok_before))
        finally:
            self.after(80, self._process_serial_queue)

    def _retry_awg_if_still(self, prev_token):
        # só re-tenta se ninguém mexeu desde então (token não mudou) e ainda está rodando
        if not self.is_wave_running:
            return
        if prev_token != self._awg_apply_token:
            return
        self._queue_awg_rearm()

    # ---------- Util ----------
    def _send(self, cmd: str):
        if not self.communicator.running:
            self._log("[pc] Não conectado.")
            return
        ok = self.communicator.send_command(cmd, enqueue=True)
        if not ok:
            self._log("[pc] Falha ao enfileirar comando.")

    def _send_manual(self):
        txt = self.entry_cmd.get().strip()
        if not txt:
            return
        self._send(txt)
        self.entry_cmd.delete(0, tk.END)

    def _log(self, text: str):
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")
        self._update_status(last=text)

    def _update_status(self, connected=None, port=None, tx=None, rx=None, last=None, last_tx=None):
        if connected is not None:
            self._status_data["connected"] = connected
        if port is not None:
            self._status_data["port"] = port
        if tx is not None:
            self._status_data["tx"] = tx
        if rx is not None:
            self._status_data["rx"] = rx
        if last is not None:
            self._status_data["last"] = last
        if last_tx is not None:
            self._status_data["last_tx"] = last_tx

        s = "Conectado" if self._status_data["connected"] else "Desconectado"
        p = self._status_data["port"] or "-"
        txb = self._status_data["tx"]
        rxb = self._status_data["rx"]
        lt = self._status_data.get("last_tx", "")
        self.status.configure(text=f"{s} [{p}]   Tx:{txb}  Rx:{rxb}   Última TX: {lt}")

    def _on_close(self):
        try:
            if self.communicator.running:
                self._send("ACCEL 0")
                self._send("DAC 0")
                time.sleep(0.05)
                self.communicator.disconnect()
        finally:
            self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
