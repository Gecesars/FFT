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
BAUDRATE = 921600
READ_SLEEP_S = 0.01
CMD_EOL = "\r\n"

# Lado do firmware: fmáx ≈ DAC_FS_MAX_HZ / LUT_SIZE (1e6/256 ≈ 3906 Hz)
MAX_FREQ_HZ = 3900          # ajuste aqui se mudar no firmware
MIN_FREQ_HZ = 1
DEFAULT_FREQ = 1000
DEFAULT_DUTY = 50

WAVE_TYPES = ["SINE", "SQUARE", "TRI", "SAWUP", "SAWDN"]


class HardwareCommunicator:
    """Thread segura para a VCP. Não bloqueia a GUI."""
    def __init__(self, data_queue, status_cb=None):
        self.ser = None
        self.port = None
        self.running = False
        self.thread = None
        self.data_queue = data_queue
        self.status_cb = status_cb or (lambda *_: None)
        self.tx_count = 0
        self.rx_count = 0

    def connect(self, port):
        try:
            self.port = port
            self.ser = serial.Serial(self.port, BAUDRATE, timeout=1, write_timeout=1)
            self.running = True
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()
            self.status_cb(connected=True, port=self.port)
            return True, f"Conectado a {self.port} @ {BAUDRATE}"
        except serial.SerialException as e:
            self.status_cb(connected=False, port=None)
            return False, f"Erro ao conectar: {e}"

    def disconnect(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.status_cb(connected=False, port=None)
        return True, "Desconectado."

    def _reader_loop(self):
        while self.running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        self.rx_count += len(line) + 1
                        self.data_queue.put(line)
                        self.status_cb(rx=self.rx_count)
            except (serial.SerialException, OSError):
                self.data_queue.put("DISCONNECTED")
                self.running = False
                break
            time.sleep(READ_SLEEP_S)

    def send_command(self, command: str) -> bool:
        if self.ser and self.ser.is_open:
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
        self.title("Controlador de Hardware Pyboard — AWG + ACCEL")
        self.geometry("900x650")
        self.minsize(880, 620)

        # Estado
        self.data_queue = queue.Queue()
        self.communicator = HardwareCommunicator(self.data_queue, status_cb=self._update_status)
        self.is_wave_running = False
        self.is_accel_on = False
        self._freq_send_after_id = None
        self._duty_send_after_id = None

        # Layout base
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)   # console cresce

        # Top bar: conexão/portas
        self._build_connection_bar(row=0)

        # Painéis
        self._build_leds_panel(row=1)
        self._build_awg_panel(row=2)
        self._build_accel_panel(row=3)

        # Console + status
        self._build_console(row=4)
        self._build_statusbar()

        # Tarefas periódicas
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

    def _build_leds_panel(self, row):
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=6, sticky="ew")
        ctk.CTkLabel(frame, text="Controle dos LEDs", font=ctk.CTkFont(weight="bold")).pack(pady=(8, 4))

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
        for c in range(6):
            f.grid_columnconfigure(c, weight=1 if c in (1, 2, 3) else 0)

        ctk.CTkLabel(f, text="Gerador de Onda (DAC via TIM2/DMA)", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=6, pady=(8, 10)
        )

        # Tipo de onda
        ctk.CTkLabel(f, text="Forma:").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        self.wave_menu = ctk.CTkOptionMenu(f, values=WAVE_TYPES, command=self._wave_type_changed)
        self.wave_menu.set("SINE")
        self.wave_menu.grid(row=1, column=1, padx=6, pady=6, sticky="w")

        # Freq slider + entry
        ctk.CTkLabel(f, text=f"Frequência (Hz) [{MIN_FREQ_HZ}..{MAX_FREQ_HZ}]").grid(row=2, column=0, padx=6, pady=6, sticky="w")
        self.freq_slider = ctk.CTkSlider(f, from_=MIN_FREQ_HZ, to=MAX_FREQ_HZ,
                                         number_of_steps=MAX_FREQ_HZ-MIN_FREQ_HZ, command=self._on_freq_slider)
        self.freq_slider.set(DEFAULT_FREQ)
        self.freq_slider.grid(row=2, column=1, columnspan=3, padx=6, pady=6, sticky="ew")

        self.freq_entry = ctk.CTkEntry(f, width=120)
        self.freq_entry.insert(0, str(DEFAULT_FREQ))
        self.freq_entry.grid(row=2, column=4, padx=6, pady=6, sticky="e")
        ctk.CTkButton(f, text="Aplicar", width=80, command=self._apply_freq_entry).grid(row=2, column=5, padx=6, pady=6)

        # Duty (só para square) — inicialmente oculto
        self.duty_label = ctk.CTkLabel(f, text="Duty (%)")
        self.duty_slider = ctk.CTkSlider(f, from_=0, to=100, number_of_steps=100, command=self._on_duty_slider)
        self.duty_val = ctk.CTkLabel(f, text=f"{DEFAULT_DUTY:.0f}%")
        self.duty_slider.set(DEFAULT_DUTY)
        self._show_duty_controls(False)

        # Botão iniciar/parar
        self.start_btn = ctk.CTkButton(f, text="Iniciar", width=120, command=self._toggle_wave)
        self.start_btn.grid(row=4, column=0, padx=6, pady=(10, 10), sticky="w")

    def _show_duty_controls(self, show: bool):
        # Usa sempre as mesmas posições fixas (linha 3)
        if show:
            self.duty_label.grid(row=3, column=0, padx=6, pady=6, sticky="w")
            self.duty_slider.grid(row=3, column=1, columnspan=3, padx=6, pady=6, sticky="ew")
            self.duty_val.grid(row=3, column=4, padx=6, pady=6, sticky="e")
        else:
            self.duty_label.grid_forget()
            self.duty_slider.grid_forget()
            self.duty_val.grid_forget()

    def _build_accel_panel(self, row):
        frame = ctk.CTkFrame(self)
        frame.grid(row=row, column=0, padx=10, pady=6, sticky="nsew")
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text="Acelerômetro", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=3, pady=(8, 6)
        )
        self.accel_switch = ctk.CTkSwitch(frame, text="Ativar Leitura Contínua", command=self._toggle_accel)
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

        self.console = ctk.CTkTextbox(frame, height=180)
        self.console.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.console.configure(state="disabled")

    def _build_statusbar(self):
        self.status = ctk.CTkLabel(self, text="Desconectado", anchor="w")
        self.status.grid(row=5, column=0, padx=10, pady=(6, 8), sticky="ew")
        self._status_data = {"connected": False, "port": None, "tx": 0, "rx": 0, "last_tx": ""}

    # ---------- Conexão ----------
    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_menu.configure(values=ports if ports else ["(nenhuma)"])
        if ports:
            cur = self.port_menu.get()
            if cur not in ports:
                # tenta manter a seleção; senão pega a primeira
                self.port_menu.set(ports[0])
        else:
            self.port_menu.set("(nenhuma)")

    def _toggle_connection(self):
        if self.communicator.running:
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

    # ---------- LEDs ----------
    def _toggle_led(self, n: int):
        state = 1 if self.led_switches[n].get() else 0
        self._send(f"LED {n} {state}")

    # ---------- AWG ----------
    def _wave_type_changed(self, *_):
        show = (self.wave_menu.get() == "SQUARE")
        self._show_duty_controls(show)
        if self.is_wave_running:
            self._send_wave_cmd()

    def _on_freq_slider(self, value):
        self.freq_entry.delete(0, tk.END)
        self.freq_entry.insert(0, f"{int(value)}")
        if self._freq_send_after_id:
            self.after_cancel(self._freq_send_after_id)
        self._freq_send_after_id = self.after(150, self._freq_maybe_send)

    def _freq_maybe_send(self):
        self._freq_send_after_id = None
        if self.is_wave_running:
            self._send_wave_cmd()

    def _on_duty_slider(self, value):
        self.duty_val.configure(text=f"{int(value)}%")
        if self._duty_send_after_id:
            self.after_cancel(self._duty_send_after_id)
        self._duty_send_after_id = self.after(150, self._duty_maybe_send)

    def _duty_maybe_send(self):
        self._duty_send_after_id = None
        if self.is_wave_running and self.wave_menu.get() == "SQUARE":
            self._send_wave_cmd()

    def _apply_freq_entry(self):
        try:
            f = float(self.freq_entry.get())
            if f < MIN_FREQ_HZ: f = MIN_FREQ_HZ
            if f > MAX_FREQ_HZ: f = MAX_FREQ_HZ
            self.freq_slider.set(f)
            if self.is_wave_running:
                self._send_wave_cmd()
        except ValueError:
            messagebox.showerror("Erro", "Frequência inválida.")

    def _toggle_wave(self):
        if not self.is_wave_running:
            self.is_wave_running = True
            self.start_btn.configure(text="Parar")
            self._send_wave_cmd()
        else:
            self.is_wave_running = False
            self.start_btn.configure(text="Iniciar")
            self._send("DAC 0")  # compatibilidade com firmware

    def _send_wave_cmd(self):
        if not self.communicator.running:
            return
        try:
            f = int(float(self.freq_entry.get() or DEFAULT_FREQ))
        except ValueError:
            f = DEFAULT_FREQ
        f = max(MIN_FREQ_HZ, min(MAX_FREQ_HZ, f))

        w = self.wave_menu.get()
        if w == "SINE":
            self._send(f"WAVE SINE {f}")
        elif w == "TRI":
            self._send(f"WAVE TRI {f}")
        elif w == "SAWUP":
            self._send(f"WAVE SAWUP {f}")
        elif w == "SAWDN":
            self._send(f"WAVE SAWDN {f}")
        elif w == "SQUARE":
            duty = int(self.duty_slider.get())
            self._send(f"WAVE SQUARE {f} {duty}")

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
        finally:
            self.after(80, self._process_serial_queue)

    # ---------- Util ----------
    def _send(self, cmd: str):
        if not self.communicator.running:
            self._log("[pc] Não conectado.")
            return
        ok = self.communicator.send_command(cmd)
        if not ok:
            self._log("[pc] Falha ao enviar comando.")

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
                self.communicator.disconnect()
        finally:
            self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
