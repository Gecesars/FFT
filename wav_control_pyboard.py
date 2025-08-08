import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import serial
import serial.tools.list_ports
import threading
import time
import queue

# Configuração da aparência da interface
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class HardwareCommunicator:
    """
    Classe dedicada para gerenciar a comunicação serial com a Pyboard.
    Ela opera em uma thread separada para não bloquear a interface gráfica.
    """

    def __init__(self, data_queue):
        self.ser = None
        self.port = None
        self.baudrate = 921600  # Usar uma velocidade alta para o streaming de dados
        self.running = False
        self.thread = None
        self.data_queue = data_queue  # Fila para comunicação segura entre threads

    def connect(self, port):
        """Tenta estabelecer a conexão serial na porta especificada."""
        try:
            self.port = port
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            self.thread = threading.Thread(target=self.read_from_port)
            self.thread.daemon = True  # Permite que a thread feche junto com o app
            self.thread.start()
            return True, "Conectado com sucesso!"
        except serial.SerialException as e:
            return False, f"Erro ao conectar: {e}"

    def disconnect(self):
        """Encerra a conexão serial e a thread de leitura."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
        return True, "Desconectado."

    def read_from_port(self):
        """Função executada pela thread para ler dados da Pyboard continuamente."""
        while self.running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    # Lê uma linha da porta serial
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        # Coloca a linha lida na fila para a GUI processar
                        self.data_queue.put(line)
            except (serial.SerialException, OSError):
                # Se houver um erro (ex: o dispositivo foi desconectado)
                self.data_queue.put("DISCONNECTED")
                self.running = False
                break
            time.sleep(0.01)  # Pequena pausa para não sobrecarregar a CPU

    def send_command(self, command):
        """Envia um comando de texto para a Pyboard."""
        if self.ser and self.ser.is_open:
            try:
                # Adiciona \r\n ao final, que é o que o firmware espera
                self.ser.write(f"{command}\r\n".encode('utf-8'))
                return True
            except serial.SerialException:
                self.data_queue.put("DISCONNECTED")
                self.running = False
                return False
        return False


class App(ctk.CTk):
    """
    Classe principal da aplicação, que constrói e gerencia a interface gráfica.
    """

    def __init__(self):
        super().__init__()
        self.title("Controlador de Hardware Pyboard v1.1")
        self.geometry("600x550")

        # Fila para receber dados da thread de comunicação
        self.data_queue = queue.Queue()
        self.communicator = HardwareCommunicator(self.data_queue)

        # Configuração do layout da janela principal
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # --- Frame de Conexão ---
        connection_frame = ctk.CTkFrame(self)
        connection_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        connection_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(connection_frame, text="Porta COM:").grid(row=0, column=0, padx=5, pady=5)
        self.port_menu = ctk.CTkOptionMenu(connection_frame, values=["Nenhuma"], command=None)
        self.port_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.refresh_ports()

        self.connect_button = ctk.CTkButton(connection_frame, text="Conectar", command=self.toggle_connection)
        self.connect_button.grid(row=0, column=2, padx=5, pady=5)

        # --- Frame de Controle dos LEDs ---
        led_frame = ctk.CTkFrame(self)
        led_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkLabel(led_frame, text="Controle dos LEDs", font=ctk.CTkFont(weight="bold")).pack(pady=5)

        self.led_switches = {}
        led_colors = {"Vermelho (0)": "red", "Verde (1)": "green", "Amarelo (2)": "yellow", "Azul (3)": "blue"}
        led_frame_inner = ctk.CTkFrame(led_frame, fg_color="transparent")
        led_frame_inner.pack(pady=5)
        for i, (text, color) in enumerate(led_colors.items()):
            # ***** CORREÇÃO APLICADA AQUI *****
            # Usamos `lambda num=i:` para capturar o valor de `i` no momento da criação do lambda.
            switch = ctk.CTkSwitch(led_frame_inner, text=text, progress_color=color,
                                   command=lambda num=i: self.toggle_led(num))
            switch.pack(side="left", expand=True, padx=10, pady=5)
            self.led_switches[i] = switch

        # --- Frame do Gerador de Ondas (DAC) ---
        dac_frame = ctk.CTkFrame(self)
        dac_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        dac_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(dac_frame, text="Gerador Senoidal (DAC)", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0,
                                                                                                     columnspan=3,
                                                                                                     pady=5)

        ctk.CTkLabel(dac_frame, text="Frequência (Hz):").grid(row=1, column=0, padx=5, pady=5)
        self.freq_entry = ctk.CTkEntry(dac_frame, placeholder_text="Ex: 1000")
        self.freq_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.dac_button = ctk.CTkButton(dac_frame, text="Gerar / Parar", command=self.toggle_dac)
        self.dac_button.grid(row=1, column=2, padx=5, pady=5)
        self.is_dac_running = False

        # --- Frame do Acelerômetro ---
        accel_frame = ctk.CTkFrame(self)
        accel_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        accel_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(accel_frame, text="Acelerômetro", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0,
                                                                                             columnspan=3, pady=5)

        self.accel_switch = ctk.CTkSwitch(accel_frame, text="Ativar Leitura Contínua", command=self.toggle_accel_stream)
        self.accel_switch.grid(row=1, column=0, columnspan=3, pady=10)

        self.progress_x, self.label_x = self.create_progress_bar(accel_frame, "Eixo X:", 2)
        self.progress_y, self.label_y = self.create_progress_bar(accel_frame, "Eixo Y:", 3)
        self.progress_z, self.label_z = self.create_progress_bar(accel_frame, "Eixo Z:", 4)

        # Inicia o loop para processar a fila de dados da thread serial
        self.process_serial_queue()

    def create_progress_bar(self, parent, text, row):
        """Função auxiliar para criar uma linha com label, barra de progresso e valor."""
        ctk.CTkLabel(parent, text=text).grid(row=row, column=0, padx=10, pady=5, sticky="w")
        progress = ctk.CTkProgressBar(parent)
        progress.set(0.5)  # Inicia no meio (valor 0)
        progress.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        label = ctk.CTkLabel(parent, text="0", width=40)
        label.grid(row=row, column=2, padx=10, pady=5)
        return progress, label

    def refresh_ports(self):
        """Atualiza a lista de portas COM disponíveis."""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_menu.configure(values=ports if ports else ["Nenhuma"])
        # Tenta pré-selecionar a COM8 se ela existir
        if "COM8" in ports:
            self.port_menu.set("COM8")
        elif ports:
            self.port_menu.set(ports[0])

    def toggle_connection(self):
        """Conecta ou desconecta da porta serial selecionada."""
        if self.communicator.running:
            success, msg = self.communicator.disconnect()
            self.connect_button.configure(text="Conectar")
            messagebox.showinfo("Desconectado", msg)
        else:
            port = self.port_menu.get()
            if port == "Nenhuma":
                messagebox.showerror("Erro", "Nenhuma porta serial selecionada.")
                return
            success, msg = self.communicator.connect(port)
            if success:
                self.connect_button.configure(text="Desconectar")
            messagebox.showinfo("Conexão", msg)

    def toggle_led(self, led_num):
        """Envia o comando para ligar/desligar um LED."""
        state = 1 if self.led_switches[led_num].get() else 0
        self.communicator.send_command(f"LED {led_num} {state}")

    def toggle_dac(self):
        """Envia o comando para iniciar ou parar a geração de onda no DAC."""
        if self.is_dac_running:
            self.communicator.send_command("DAC 0")
            self.is_dac_running = False
        else:
            try:
                freq = float(self.freq_entry.get())
                if freq > 0:
                    self.communicator.send_command(f"DAC {freq}")
                    self.is_dac_running = True
                else:
                    messagebox.showerror("Erro", "Frequência deve ser maior que zero.")
            except ValueError:
                messagebox.showerror("Erro", "Valor de frequência inválido.")

    def toggle_accel_stream(self):
        """Envia o comando para iniciar ou parar o streaming do acelerômetro."""
        state = 1 if self.accel_switch.get() else 0
        self.communicator.send_command(f"ACCEL {state}")

    def process_serial_queue(self):
        """
        Verifica a fila de dados a cada 100ms.
        Esta é a forma segura de atualizar a GUI com dados de outra thread.
        """
        try:
            while not self.data_queue.empty():
                message = self.data_queue.get_nowait()
                if message == "DISCONNECTED":
                    self.connect_button.configure(text="Conectar")
                    messagebox.showwarning("Desconectado", "A conexão com o dispositivo foi perdida.")
                elif message.startswith("A:"):  # Verifica se é um pacote de dados do acelerômetro
                    try:
                        # Parseia a string "A:x,y,z"
                        parts = message[2:].split(',')
                        x, y, z = map(int, parts)

                        # O valor do acelerômetro vai de -32 a 31.
                        # Mapeia este valor para o intervalo da barra de progresso (0.0 a 1.0).
                        self.progress_x.set((x + 32) / 63)
                        self.label_x.configure(text=str(x))

                        self.progress_y.set((y + 32) / 63)
                        self.label_y.configure(text=str(y))

                        self.progress_z.set((z + 32) / 63)
                        self.label_z.configure(text=str(z))
                    except (ValueError, IndexError):
                        print(f"Pacote de dados do acelerômetro malformado: {message}")
                else:
                    # Imprime qualquer outra mensagem (OK, PONG, ERROR, etc.) no console
                    print(f"Pyboard: {message}")
        finally:
            # Agenda a próxima verificação da fila
            self.after(100, self.process_serial_queue)


if __name__ == "__main__":
    app = App()
    app.mainloop()
