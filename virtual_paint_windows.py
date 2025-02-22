"""
PIZARRA VIRTUAL CON DETECCIÓN DE GESTOS - VERSIÓN WINDOWS
Funcionalidades principales:
- Dibujo con mano mediante cámara
- 5 herramientas: Pincel, Línea, Rectángulo, Círculo, Borrador
- Sistema de deshacer (15 niveles)
- Selección de herramientas por gestos
- Ajuste de tamaño y color
- Optimizado para rendimiento en Windows

Requerimientos:
- Python 3.7+
- OpenCV, MediaPipe, NumPy
- Cámara funcional

Teclas rápidas:
U/u - Deshacer        E/e - Borrador       R/r - Rojo
C/c - Limpiar         G/g - Verde          B/b - Azul
+/= - Aumentar tamaño -/_ - Reducir tamaño ESC - Salir
"""

import mediapipe as mp
import cv2
import numpy as np
import time
import os
from collections import deque

# ================= CONFIGURACIÓN INICIAL =================
# Configuración de MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parámetros para mejor rendimiento en Windows
hands = mp_hands.Hands(
    min_detection_confidence=0.7,  # Reduce falsos positivos
    min_tracking_confidence=0.7,  # Mejor seguimiento en movimiento
    max_num_hands=1,  # Solo una mano detectada
    model_complexity=0  # Modelo ligero (optimizado para CPU)
)

# Constantes de configuración
MAX_HISTORY = 15  # Capacidad de deshacer aumentada
BRUSH_SIZES = [4, 8, 12, 16]  # Tamaños disponibles del pincel
COLORS = [  # BGR format
    (0, 0, 0),  # Negro
    (255, 0, 0),  # Azul
    (0, 0, 255),  # Rojo
    (0, 255, 0),  # Verde
    (255, 255, 255)  # Blanco (Borrador)
]


# ================= CLASE PRINCIPAL =================
class DrawingApp:
    """Clase principal que gestiona toda la aplicación de dibujo"""

    def __init__(self):
        # Configuración específica para cámaras en Windows
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Usar backend DirectShow
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Resolución HD para mejor rendimiento
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Verificar si la cámara se inicializó correctamente
            if not self.cap.isOpened():
                raise RuntimeError("No se pudo acceder a la cámara")

            # Obtener dimensiones reales del frame
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Error al leer el frame inicial")
            self.h, self.w = frame.shape[:2]

        except Exception as e:
            print(f"Error de inicialización: {str(e)}")
            exit(1)

        # Superficie de dibujo y historial
        self.mask = np.ones((self.h, self.w, 3), dtype='uint8') * 255  # Fondo blanco
        self.history = deque(maxlen=MAX_HISTORY)  # Historial de estados

        # Estado de la aplicación
        self.current_tool = "Brush"
        self.brush_size = 8
        self.color = COLORS[0]
        self.prev_point = None  # Para dibujo continuo
        self.start_point = None  # Para formas geométricas

        # Control de selección de herramientas
        self.tool_select_time_init = True
        self.tool_select_start_time = None
        self.tool_select_radius = 40  # Radio del círculo de selección

        # Interfaz de usuario
        self.tools = self.load_tools_interface(self.w)
        self.show_instructions()  # Pantalla de bienvenida

    # ================= MÉTODOS DE INTERFAZ =================
    def load_tools_interface(self, width):
        """Crea la barra de herramientas superior con iconos interactivos"""
        tools = np.zeros((100, width, 3), dtype=np.uint8)
        icon_width = width // 5  # Dividir el ancho en 5 secciones

        # Configuración de cada herramienta (texto, color)
        icons = [
            ("Brush", (0, 0, 255)),  # Rojo
            ("Line", (255, 0, 0)),  # Azul
            ("Rect", (0, 255, 0)),  # Verde
            ("Circle", (255, 255, 0)),  # Amarillo
            ("Erase", (255, 255, 255))  # Blanco
        ]

        for i, (text, color) in enumerate(icons):
            start_x = i * icon_width + 5
            end_x = start_x + icon_width - 10

            # Dibujar botón con efecto 3D
            cv2.rectangle(tools, (start_x, 5), (end_x, 85), color, -1)
            cv2.rectangle(tools, (start_x, 5), (end_x, 85), (200, 200, 200), 2)

            # Texto centrado
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
            text_x = start_x + (icon_width - text_w) // 2 - 10
            text_y = 50 + text_h // 2
            cv2.putText(tools, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

        return tools

    def show_instructions(self):
        """Muestra la pantalla de instrucciones inicial"""
        try:
            # Cargar imagen usando ruta absoluta
            base_dir = os.path.dirname(os.path.abspath(__file__))
            portada_path = os.path.join(base_dir, "portada.jpg")
            portada = cv2.imread(portada_path)

            if portada is None:
                print("Advertencia: No se encontró portada.jpg")
                return

            # Redimensionar manteniendo relación de aspecto
            h, w = portada.shape[:2]
            aspect_ratio = w / h
            new_h = min(800, h)
            new_w = int(new_h * aspect_ratio)
            portada = cv2.resize(portada, (new_w, new_h))

            # Mostrar en ventana redimensionable
            cv2.namedWindow("Instrucciones", cv2.WINDOW_NORMAL)
            cv2.imshow("Instrucciones", portada)
            cv2.resizeWindow("Instrucciones", new_w, new_h)

            # Esperar entrada de usuario
            while cv2.getWindowProperty("Instrucciones", cv2.WND_PROP_VISIBLE) > 0:
                key = cv2.waitKey(1)
                if key != -1:
                    break
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error mostrando instrucciones: {str(e)}")

    # ================= LÓGICA DE DIBUJO =================
    def detect_gesture(self, landmarks):
        """Detecta dedos levantados con umbrales ajustados para Windows"""
        fingers = []
        y_threshold = 0.03  # Margen aumentado para mejor detección

        # Dedo índice (landmark 8 vs 6)
        if (landmarks[8].y + y_threshold) < landmarks[6].y:
            fingers.append(1)

        # Dedo medio (landmark 12 vs 10)
        if (landmarks[12].y + y_threshold) < landmarks[10].y:
            fingers.append(1)

        return fingers

    def update_brush(self, size):
        """Actualiza tamaño del pincel con límites seguros"""
        self.brush_size = np.clip(size, 2, 30)  # Rango 2-30

    def draw_line(self, start, end):
        """Dibuja línea suavizada con interpolación optimizada"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = max(abs(dx), abs(dy))

        # Optimización: Reducir puntos en líneas largas
        step = max(1, distance // 50)
        for i in range(0, distance + 1, step):
            x = int(start[0] + (i / distance) * dx)
            y = int(start[1] + (i / distance) * dy)
            cv2.circle(self.mask, (x, y), self.brush_size, self.color, -1)

    def apply_color(self, new_color):
        """Cambia color actual manejando estado de borrador"""
        self.color = new_color
        self.current_tool = "Erase" if new_color == COLORS[-1] else self.current_tool

    # ================= PROCESAMIENTO PRINCIPAL =================
    def process_frame(self):
        """Procesa cada frame de video y actualiza la interfaz"""
        try:
            success, frame = self.cap.read()
            if not success:
                print("Advertencia: Frame no capturado")
                return None

            # Preprocesamiento
            frame = cv2.flip(frame, 1)  # Espejo horizontal
            frame = cv2.GaussianBlur(frame, (3, 3), 0)  # Reducción de ruido
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detección de manos
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks para debug
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Obtener posición dedo índice
                    index_tip = hand_landmarks.landmark[8]
                    x = int(index_tip.x * self.w)
                    y = int(index_tip.y * self.h)

                    # Zona de selección de herramientas (parte superior)
                    if y < 100:
                        self.handle_tool_selection(x, y, frame)
                    else:
                        self.handle_drawing(hand_landmarks.landmark, x, y, frame)

            # Combinar dibujo con video
            drawn_area = cv2.bitwise_and(self.mask, self.mask,
                                         mask=cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY))
            frame = cv2.addWeighted(frame, 0.7, drawn_area, 0.3, 0)

            # Superponer interfaz de usuario
            self.draw_ui_overlay(frame)

            return frame

        except Exception as e:
            print(f"Error procesando frame: {str(e)}")
            return None

#tira pa' que yo la pruebe
#Se pone olorsa me gusta como huele
#Instagram privado pa que nadie la vele
#Se puso bonita porque sabe que hoy se bebe

#A portarse mal, pa sentirse bien
#no queria funar, pero le dio al pen
#una barbie, pero no busca un ken
#Siempre le llego dice: Ven


    def handle_tool_selection(self, x, y, frame):
        """Maneja la lógica de selección de herramientas"""
        if self.tool_select_time_init:
            self.tool_select_start_time = time.time()
            self.tool_select_time_init = False

        # Animación de progreso
        current_time = time.time()
        progress = min(1.0, (current_time - self.tool_select_start_time) / 0.8)
        self.tool_select_radius = int(40 * (1 - progress))

        # Dibujar círculo de feedback
        cv2.circle(frame, (x, y), self.tool_select_radius, (0, 255, 255), 2)

        # Confirmar selección después de 0.8 segundos
        if progress >= 1.0:
            self.current_tool = self.get_tool(x)
            self.apply_color(COLORS[-1] if self.current_tool == "Erase" else self.color)
            self.tool_select_time_init = True
            self.tool_select_radius = 40

    def handle_drawing(self, landmarks, x, y, frame):
        """Maneja la lógica de dibujo y formas geométricas"""
        fingers = self.detect_gesture(landmarks)

        if self.current_tool in ["Brush", "Erase"]:
            # Dibujo libre
            if sum(fingers) == 1:
                current_point = (x, y)
                if self.prev_point:
                    self.draw_line(self.prev_point, current_point)
                self.prev_point = current_point
            else:
                self.prev_point = None
        else:
            # Formas geométricas
            if sum(fingers) == 1:
                if not self.start_point:
                    self.start_point = (x, y)

                # Previsualización temporal
                temp_mask = self.mask.copy()
                color = self.color if self.current_tool != "Erase" else COLORS[-1]

                if self.current_tool == "Line":
                    cv2.line(temp_mask, self.start_point, (x, y), color, self.brush_size)
                elif self.current_tool == "Rect":
                    cv2.rectangle(temp_mask, self.start_point, (x, y), color, self.brush_size)
                elif self.current_tool == "Circle":
                    radius = int(np.hypot(x - self.start_point[0], y - self.start_point[1]))
                    cv2.circle(temp_mask, self.start_point, radius, color, self.brush_size)

                frame = cv2.addWeighted(frame, 0.7, temp_mask, 0.3, 0)
            else:
                if self.start_point:
                    # Confirmar dibujo
                    self.history.append(self.mask.copy())
                    self.start_point = None

    def draw_ui_overlay(self, frame):
        """Dibuja elementos de la interfaz de usuario"""
        # Barra de herramientas
        frame[0:100, 0:self.w] = cv2.addWeighted(self.tools, 0.8, frame[0:100, 0:self.w], 0.2, 0)

        # Panel de información
        info_y = self.h - 40
        cv2.putText(frame, f"Herramienta: {self.current_tool}", (20, info_y),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tamano: {self.brush_size}", (self.w - 200, info_y),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

        # Indicador de color actual
        cv2.circle(frame, (self.w - 50, 50), self.brush_size + 2, (255, 255, 255), 2)
        cv2.circle(frame, (self.w - 50, 50), self.brush_size, self.color, -1)

    # ================= BUCLE PRINCIPAL =================
    def run(self):
        """Bucle principal de ejecución"""
        try:
            cv2.namedWindow("Pizarra Interactiva", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Pizarra Interactiva", 1280, 720)
            cv2.setWindowProperty("Pizarra Interactiva", cv2.WND_PROP_FULLSCREEN, 1)

            while self.cap.isOpened():
                frame = self.process_frame()
                if frame is None:
                    break

                cv2.imshow("Pizarra Interactiva", frame)
                key = cv2.waitKey(1) & 0xFF

                # Manejo de teclado extendido
                if key == 27:  # ESC
                    break
                elif key in [ord('U'), ord('u')] and self.history:
                    self.mask = self.history.pop()
                elif key in [ord('C'), ord('c')]:
                    self.mask.fill(255)
                    self.history.clear()
                elif key in [ord('E'), ord('e')]:
                    self.apply_color(COLORS[-1])
                elif key in [ord('R'), ord('r')]:
                    self.apply_color(COLORS[2])
                elif key in [ord('G'), ord('g')]:
                    self.apply_color(COLORS[3])
                elif key in [ord('B'), ord('b')]:
                    self.apply_color(COLORS[1])
                elif key in [ord('+'), ord('=')]:
                    self.update_brush(self.brush_size + 2)
                elif key in [ord('-'), ord('_')]:
                    self.update_brush(self.brush_size - 2)

                # Guardar estado actual
                if len(self.history) < MAX_HISTORY:
                    self.history.append(self.mask.copy())

        except KeyboardInterrupt:
            print("Aplicación interrumpida por el usuario")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


# ================= EJECUCIÓN =================
if __name__ == "__main__":
    app = DrawingApp()
    app.run()
