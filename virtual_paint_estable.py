import mediapipe as mp
import cv2
import numpy as np
import time
from collections import deque

# Inicialización de MediaPipe para detección y seguimiento de manos
mp_hands = mp.solutions.hands  # Módulo de detección de manos de MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Utilidad para dibujar los landmarks de las manos

# Configuración de la solución Hands con parámetros de confianza y cantidad de manos
hands = mp_hands.Hands(
    min_detection_confidence=0.8,  # Confianza mínima para la detección inicial
    min_tracking_confidence=0.8,  # Confianza mínima para el seguimiento de la mano
    max_num_hands=1,  # Se detecta como máximo una mano
    model_complexity=1  # Complejidad del modelo (1: normal)
)

# Configuraciones generales para el dibujo
MAX_HISTORY = 10  # Número máximo de estados en el historial para la función de "deshacer"
BRUSH_SIZES = [4, 8, 12, 16]  # Posibles tamaños del pincel
COLORS = [
    (0, 0, 0),  # Negro
    (255, 0, 0),  # Azul
    (0, 0, 255),  # Rojo
    (0, 255, 0),  # Verde
    (255, 255, 255)  # Blanco (borrador)
]


class DrawingApp:
    """
    Clase principal que gestiona la aplicación de dibujo interactivo usando la cámara y gestos.
    """

    def __init__(self):
        # Inicializa la captura de video desde la cámara
        self.cap = cv2.VideoCapture(0)
        # Se obtiene un frame para determinar las dimensiones reales de la cámara
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("No se puede leer la cámara")
        self.h, self.w = frame.shape[:2]  # Alto y ancho de la imagen capturada

        # Inicializa la máscara donde se dibuja, con fondo blanco
        self.mask = np.ones((self.h, self.w, 3), dtype='uint8') * 255
        # Historial para guardar estados de la máscara (para deshacer)
        self.history = deque(maxlen=MAX_HISTORY)
        self.current_tool = "Brush"  # Herramienta de dibujo por defecto: pincel
        self.brush_size = 8  # Tamaño inicial del pincel
        self.color = COLORS[0]  # Color inicial del pincel (Negro)
        self.prev_point = None  # Punto anterior para dibujar líneas continuas
        self.start_point = None  # Punto de inicio para dibujar formas (línea, rectángulo, círculo)

        # Variables para la selección de herramientas mediante gestos en la zona superior
        self.tool_select_time_init = True
        self.tool_select_start_time = None
        self.tool_select_radius = 40  # Radio inicial del indicador de selección

        # Crea la interfaz de herramientas (barra superior) basada en el ancho de la cámara
        self.tools = self.load_tools_interface(self.w)
        # Muestra la pantalla de instrucciones (imagen "portada.jpg")
        self.show_instructions()

    def load_tools_interface(self, width):
        """
        Crea la interfaz visual de herramientas (barra superior) dividiendo la imagen en 5 zonas.

        Args:
            width (int): Ancho total de la ventana de la cámara.

        Returns:
            np.ndarray: Imagen que contiene la interfaz de herramientas.
        """
        tools = np.zeros((100, width, 3), dtype=np.uint8)  # Imagen base para la barra (fondo negro)
        icon_width = width // 5  # Cada herramienta ocupará una quinta parte del ancho
        icons = [
            ("Brush", (0, 0, 255)),
            ("Line", (255, 0, 0)),
            ("Rect", (0, 255, 0)),
            ("Circle", (255, 255, 0)),
            ("Erase", (255, 255, 255))
        ]
        # Recorre cada herramienta para dibujar su icono y texto en la interfaz
        for i, (text, color) in enumerate(icons):
            start_x = i * icon_width
            end_x = start_x + icon_width - 10
            cv2.rectangle(tools, (start_x, 0), (end_x, 90), color, -1)  # Dibuja el rectángulo de la herramienta
            cv2.putText(tools, text, (start_x + 10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Añade el nombre de la herramienta
        return tools

    def show_instructions(self):
        """
        Muestra la pantalla de instrucciones utilizando la imagen "portada.jpg".
        Se carga, se redimensiona (si es necesario) y se muestra en una ventana de OpenCV.
        """
        # Cargar la imagen de portada con proporciones 1080x1080
        portada = cv2.imread("portada.jpg")
        if portada is None:
            print("Error: No se pudo cargar portada.jpg")
            return

        # Redimensionar la imagen a 1080x1080 (opcional según sea necesario)
        portada = cv2.resize(portada, (1080, 1080))

        # Mostrar la imagen en una ventana
        cv2.imshow("Instrucciones", portada)
        cv2.waitKey(0)  # Espera hasta que se presione una tecla
        cv2.destroyWindow("Instrucciones")  # Cierra la ventana de instrucciones

    def get_tool(self, x):
        """
        Determina la herramienta seleccionada en función de la posición horizontal de la mano.

        Args:
            x (int): Coordenada x de la mano.

        Returns:
            str: Nombre de la herramienta seleccionada.
        """
        icon_width = self.w // 5  # Divide el ancho total en 5 zonas
        if x < icon_width:
            return "Brush"
        elif x < 2 * icon_width:
            return "Line"
        elif x < 3 * icon_width:
            return "Rect"
        elif x < 4 * icon_width:
            return "Circle"
        else:
            return "Erase"

    def detect_gesture(self, landmarks):
        """
        Detecta gestos básicos usando la posición de los landmarks de la mano.
        Se considera levantado el dedo índice y el dedo medio según su posición vertical.

        Args:
            landmarks (list): Lista de landmarks de la mano.

        Returns:
            list: Lista que indica qué dedos se detectan levantados.
        """
        fingers = []
        # Si el dedo índice (landmark 8) está más arriba que el landmark 6, se considera levantado
        if landmarks[8].y < landmarks[6].y:
            fingers.append(1)
        # Si el dedo medio (landmark 12) está más arriba que el landmark 10, se considera levantado
        if landmarks[12].y < landmarks[10].y:
            fingers.append(1)
        return fingers

    def update_brush(self, size):
        """
        Actualiza el tamaño del pincel limitándolo entre 2 y 20.

        Args:
            size (int): Nuevo tamaño propuesto para el pincel.
        """
        self.brush_size = max(2, min(20, size))

    def draw_line(self, start, end):
        """
        Dibuja una línea continua entre dos puntos usando círculos para simular el trazo del pincel.

        Args:
            start (tuple): Punto inicial (x, y).
            end (tuple): Punto final (x, y).
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        # Calcula la distancia máxima para iterar a lo largo de la línea
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int(start[0] + float(i) / distance * dx)
            y = int(start[1] + float(i) / distance * dy)
            cv2.circle(self.mask, (x, y), self.brush_size, self.color,
                       -1)  # Dibuja círculos superpuestos para formar la línea

    def apply_color(self, new_color):
        """
        Cambia el color actual del pincel y, si se selecciona el color blanco, se cambia la herramienta a "Erase".

        Args:
            new_color (tuple): Nuevo color en formato BGR.
        """
        prev_tool = self.current_tool  # Guarda la herramienta actual
        self.color = new_color
        if new_color == COLORS[-1]:
            self.current_tool = "Erase"  # Si es blanco, se activa el borrador
        else:
            self.current_tool = prev_tool

    def process_frame(self):
        """
        Procesa cada frame capturado por la cámara:
         - Voltea el frame para efecto espejo.
         - Convierte el frame a RGB y lo procesa con MediaPipe para detectar la mano.
         - Dibuja la interfaz, selecciona herramientas según la posición y gestos, y dibuja en la máscara.
         - Combina la máscara de dibujo con el frame de video.

        Returns:
            np.ndarray: Frame procesado listo para mostrar.
        """
        success, frame = self.cap.read()
        if not success:
            return None

        frame = cv2.flip(frame, 1)  # Efecto espejo para que el usuario se vea reflejado
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Conversión a RGB para MediaPipe
        results = hands.process(rgb)  # Procesa el frame para detectar la mano

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibuja los landmarks y conexiones en el frame para visualización
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_tip = hand_landmarks.landmark[8]  # Landmark del dedo índice
                x = int(index_tip.x * frame.shape[1])
                y = int(index_tip.y * frame.shape[0])

                # Si la posición y está en la zona superior (<50 px), activa la selección de herramienta
                if y < 50:
                    if self.tool_select_time_init:
                        self.tool_select_start_time = time.time()  # Inicia el temporizador
                        self.tool_select_time_init = False
                    current_time = time.time()
                    # Dibuja un círculo indicativo en la posición actual de la mano
                    cv2.circle(frame, (x, y), self.tool_select_radius, (0, 255, 255), 2)
                    self.tool_select_radius = max(5, self.tool_select_radius - 1)
                    # Si se mantiene en la zona de selección por más de 0.8 segundos, se selecciona la herramienta
                    if (current_time - self.tool_select_start_time) > 0.8:
                        self.current_tool = self.get_tool(x)
                        print("Herramienta actual:", self.current_tool)
                        self.tool_select_time_init = True
                        self.tool_select_radius = 40
                        if self.current_tool == "Erase":
                            self.color = COLORS[-1]
                        self.prev_point = None
                        self.start_point = None
                else:
                    # Fuera de la zona de selección, se reinician los parámetros de selección
                    self.tool_select_time_init = True
                    self.tool_select_radius = 40
                    fingers = self.detect_gesture(hand_landmarks.landmark)
                    if self.current_tool in ["Brush", "Erase"]:
                        # Si se detecta que solo un dedo está levantado, dibuja líneas continuas
                        if sum(fingers) == 1:
                            if self.prev_point:
                                self.draw_line(self.prev_point, (x, y))
                            self.prev_point = (x, y)
                        else:
                            self.prev_point = None
                    else:
                        # Para herramientas de formas (Line, Rect, Circle)
                        if sum(fingers) == 1:
                            if self.start_point is None:
                                self.start_point = (x, y)  # Se marca el inicio de la forma
                            # Previsualiza la forma en el frame según la herramienta seleccionada
                            if self.current_tool == "Line":
                                cv2.line(frame, self.start_point, (x, y), self.color, self.brush_size)
                            elif self.current_tool == "Rect":
                                cv2.rectangle(frame, self.start_point, (x, y), self.color, self.brush_size)
                            elif self.current_tool == "Circle":
                                radius = int(((x - self.start_point[0]) ** 2 + (y - self.start_point[1]) ** 2) ** 0.5)
                                cv2.circle(frame, self.start_point, radius, self.color, self.brush_size)
                        else:
                            if self.start_point is not None:
                                # Dibuja la forma final en la máscara y guarda el estado en el historial
                                final_color = self.color if self.current_tool != "Erase" else COLORS[-1]
                                if self.current_tool == "Line":
                                    cv2.line(self.mask, self.start_point, (x, y), final_color, self.brush_size)
                                elif self.current_tool == "Rect":
                                    cv2.rectangle(self.mask, self.start_point, (x, y), final_color, self.brush_size)
                                elif self.current_tool == "Circle":
                                    radius = int(
                                        ((x - self.start_point[0]) ** 2 + (y - self.start_point[1]) ** 2) ** 0.5)
                                    cv2.circle(self.mask, self.start_point, radius, final_color, self.brush_size)
                                self.history.append(self.mask.copy())  # Guarda el estado actual para "deshacer"
                                self.start_point = None

        # Procesa la máscara para integrar el dibujo con el frame
        gray_mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        _, drawn_mask = cv2.threshold(gray_mask, 254, 255, cv2.THRESH_BINARY_INV)  # Crea una máscara binaria del dibujo
        inverted_mask = cv2.bitwise_not(drawn_mask)
        background = cv2.bitwise_and(frame, frame, mask=inverted_mask)  # Fondo del frame sin dibujo
        foreground = cv2.bitwise_and(self.mask, self.mask, mask=drawn_mask)  # Dibujo sobre la máscara
        frame = cv2.addWeighted(background, 0.7, foreground, 0.3, 0)  # Combina ambos para la visualización final

        # Superpone la interfaz de herramientas en la parte superior del frame
        frame[0:100, 0:self.w] = cv2.addWeighted(self.tools, 0.7, frame[0:100, 0:self.w], 0.3, 0)
        cv2.putText(frame, f"Herramienta: {self.current_tool}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Tamaño: {self.brush_size}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(frame, (self.w - 40, 30), self.brush_size, self.color,
                   -1)  # Muestra un círculo con el color y tamaño actuales

        return frame

    def run(self):
        """
        Método principal que ejecuta la aplicación:
         - Configura la ventana de la pizarra en modo pantalla completa.
         - Procesa continuamente los frames capturados.
         - Gestiona entradas del teclado para funciones (salir, deshacer, cambiar color, limpiar, etc.).
        """
        # Configura la ventana "Pizarra Interactiva" en modo pantalla completa
        cv2.namedWindow("Pizarra Interactiva", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Pizarra Interactiva", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self.cap.isOpened():
            self.frame = self.process_frame()  # Procesa el frame actual
            if self.frame is None:
                break

            cv2.imshow("Pizarra Interactiva", self.frame)
            key = cv2.waitKey(1) & 0xFF  # Captura la tecla presionada

            # Manejo de entradas del teclado:
            if key == 27:  # ESC para salir
                break
            elif key == ord('u'):
                if self.history:
                    self.mask = self.history.pop()  # Deshacer la última acción
            elif key == ord('c'):
                self.mask.fill(255)  # Limpia toda la pizarra
                self.history.clear()
            elif key == ord('e'):
                self.apply_color(COLORS[-1])  # Activa el borrador
            elif key in [ord('r'), ord('g'), ord('b')]:
                # Cambia el color del pincel según la tecla presionada (R, G, B)
                self.apply_color({
                                     ord('r'): COLORS[2],  # Rojo
                                     ord('g'): COLORS[3],  # Verde
                                     ord('b'): COLORS[1]  # Azul
                                 }[key])
            elif key == ord('+'):
                self.update_brush(self.brush_size + 2)  # Aumenta el tamaño del pincel
            elif key == ord('-'):
                self.update_brush(self.brush_size - 2)  # Disminuye el tamaño del pincel

            # Guarda el estado actual de la máscara en el historial para "deshacer"
            if len(self.history) < MAX_HISTORY:
                self.history.append(self.mask.copy())

        # Libera la cámara y cierra todas las ventanas de OpenCV
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Crea una instancia de la aplicación y la ejecuta
    app = DrawingApp()
    app.run()
