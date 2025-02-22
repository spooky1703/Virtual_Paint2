# Virtual Paint

Una pizarra virtual interactiva desarrollada en Python que utiliza visión por computadora y reconocimiento de gestos para permitir dibujar y manipular elementos a través de la webcam. Esta aplicación permite a los usuarios dibujar, borrar y crear formas usando gestos de la mano en tiempo real.

## Características

- Reconocimiento de gestos en tiempo real usando MediaPipe
- Múltiples herramientas de dibujo:
  - Pincel de mano libre
  - Herramienta de línea
  - Herramienta de rectángulo
  - Herramienta de círculo
  - Borrador
- Tamaños de pincel ajustables
- Múltiples colores (Negro, Azul, Rojo, Verde)
- Función de deshacer
- Opción de limpiar lienzo
- Selección de herramientas mediante posicionamiento de la mano
- Interfaz visual con indicadores de herramientas

## Requisitos

- Python 3.7+
- OpenCV (cv2)
- MediaPipe
- NumPy
- Una webcam

## Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/tuusuario/pizarra-virtual.git
cd pizarra-virtual
```

2. Crea un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate
```

3. Instala los paquetes requeridos:
```bash
pip install mediapipe opencv-python numpy
```

## Uso

1. Ejecuta la aplicación:
```bash
python main.py
```

2. La aplicación mostrará las instrucciones iniciales y luego lanzará la interfaz de la pizarra virtual.

### Controles

#### Gestos con la Mano:
- Mantén el dedo índice arriba para dibujar/usar herramientas
- Mueve la mano a la parte superior de la pantalla para seleccionar herramientas
- Mantén la posición durante 0.8 segundos para seleccionar una herramienta
- Usa un dedo para dibujar y crear formas

#### Controles de Teclado:
- `ESC` - Salir de la aplicación
- `u` - Deshacer última acción
- `c` - Limpiar lienzo
- `e` - Cambiar a borrador
- `r` - Cambiar a color rojo
- `g` - Cambiar a color verde
- `b` - Cambiar a color azul
- `+` - Aumentar tamaño del pincel
- `-` - Disminuir tamaño del pincel

## Notas para Windows

Se incluye una segunda version del codigo `virtual_paint_windows.py` pensada exclusivamente para ser ejecutada correctamente en ese OS

### Requisitos adicionales
- Visual C++ Redistributable (necesario para MediaPipe)
- Python 3.7-3.9 (versiones recomendadas para mejor compatibilidad)

### Solución de problemas comunes
1. Si la cámara no se inicializa:
   - Verificar que no esté siendo usada por otra aplicación
   - Probar diferentes índices de cámara (0, 1, 2)
   - Actualizar los drivers de la cámara

2. Si la aplicación se ejecuta lenta:
   - Reducir la resolución de la cámara
   - Cerrar aplicaciones en segundo plano
   - Asegurar que las tarjetas gráficas estén actualizadas

3. Si hay problemas con MediaPipe:
   - Instalar Microsoft Visual C++ Redistributable
   - Usar una versión compatible de Python (3.7-3.9)



## Estructura y Funcionamiento del Código

### 1. Inicialización y Configuración

La aplicación se inicializa con MediaPipe para la detección de manos y configura los parámetros básicos:

```python
# Inicialización de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=1,
    model_complexity=1
)

# Configuraciones de dibujo
BRUSH_SIZES = [4, 8, 12, 16]
COLORS = [
    (0, 0, 0),    # Negro
    (255, 0, 0),  # Azul
    (0, 0, 255),  # Rojo
    (0, 255, 0),  # Verde
    (255, 255, 255)  # Blanco (borrador)
]
```

### 2. Clase Principal DrawingApp

La aplicación se estructura alrededor de la clase `DrawingApp`. Aquí está su inicialización:

```python
def __init__(self):
    self.cap = cv2.VideoCapture(0)
    ret, frame = self.cap.read()
    self.h, self.w = frame.shape[:2]
    
    # Inicialización del lienzo
    self.mask = np.ones((self.h, self.w, 3), dtype='uint8') * 255
    self.history = deque(maxlen=MAX_HISTORY)
    self.current_tool = "Brush"
    self.brush_size = 8
    self.color = COLORS[0]
```

### 3. Detección de Gestos

El sistema detecta gestos básicos analizando la posición de los dedos:

```python
def detect_gesture(self, landmarks):
    fingers = []
    # Dedo índice levantado
    if landmarks[8].y < landmarks[6].y:
        fingers.append(1)
    # Dedo medio levantado
    if landmarks[12].y < landmarks[10].y:
        fingers.append(1)
    return fingers
```

### 4. Procesamiento de Frames

El método principal que procesa cada frame de la cámara:

```python
def process_frame(self):
    success, frame = self.cap.read()
    frame = cv2.flip(frame, 1)  # Efecto espejo
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtiene posición del dedo índice
            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * frame.shape[1])
            y = int(index_tip.y * frame.shape[0])
            
            # Procesa gestos y dibuja según la herramienta actual
            # [Código de dibujo y procesamiento...]
```

### 5. Herramientas de Dibujo

El sistema implementa diferentes herramientas de dibujo. Por ejemplo, el dibujo de líneas:

```python
def draw_line(self, start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        cv2.circle(self.mask, (x, y), self.brush_size, self.color, -1)
```

### 6. Interfaz de Usuario

La interfaz se crea y gestiona mediante:

```python
def load_tools_interface(self, width):
    tools = np.zeros((100, width, 3), dtype=np.uint8)
    icon_width = width // 5
    icons = [
        ("Brush", (0, 0, 255)),
        ("Line", (255, 0, 0)),
        ("Rect", (0, 255, 0)),
        ("Circle", (255, 255, 0)),
        ("Erase", (255, 255, 255))
    ]
    for i, (text, color) in enumerate(icons):
        start_x = i * icon_width
        end_x = start_x + icon_width - 10
        cv2.rectangle(tools, (start_x, 0), (end_x, 90), color, -1)
        cv2.putText(tools, text, (start_x + 10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return tools
```

## Problemas Conocidos y Limitaciones

- Solo soporta detección de una mano
- Requiere buenas condiciones de iluminación
- Puede experimentar cierto retraso en sistemas más lentos
- La resolución de la webcam afecta la precisión del dibujo

## Contribuir

¡Las contribuciones son bienvenidas! Por favor, siéntete libre de enviar un Pull Request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## Agradecimientos

- MediaPipe por la detección y seguimiento de manos
- OpenCV por el procesamiento de imágenes
- NumPy por las operaciones con arrays

## Notas Adicionales

Para un funcionamiento óptimo:
- Asegúrate de tener buena iluminación
- Mantén la mano visible y estable
- Calibra la sensibilidad si es necesario ajustando los valores de `min_detection_confidence`
