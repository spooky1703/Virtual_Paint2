# Virtual_Paint2

Una aplicación de pizarra virtual interactiva basada en Python que utiliza visión por computadora y reconocimiento de gestos de mano para crear una experiencia de dibujo inmersiva. La aplicación permite a los usuarios dibujar, borrar y manipular formas utilizando gestos de mano capturados a través de su cámara web.

## Características

- Reconocimiento de gestos de mano en tiempo real
- Múltiples herramientas de dibujo:
  - Pincel de mano libre
  - Herramienta de línea
  - Herramienta de rectángulo
  - Herramienta de círculo
  - Borrador
- Selección de colores (Negro, Azul, Rojo, Verde, Blanco)
- Tamaño de pincel ajustable
- Función de deshacer
- Opción de limpiar lienzo
- Importación y manipulación de imágenes
- Modo pantalla completa
- Interfaz interactiva de selección de herramientas

## Requisitos

- Python 3.7+
- OpenCV (cv2)
- MediaPipe
- NumPy
- macOS (para la funcionalidad de importación de imágenes)

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tuusuario/pizarra-virtual.git
cd pizarra-virtual
```

2. Crea y activa un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Unix/macOS
# O
.\venv\Scripts\activate  # En Windows
```

3. Instala los paquetes requeridos:
```bash
pip install mediapipe opencv-python numpy
```

## Uso

1. Ejecuta la aplicación:
```bash
python virtual_paint_estable.py
```

2. Controles:
- **Gestos con la mano:**
  - Apunta con el dedo índice para dibujar
  - Levanta los dedos índice y medio para preparar el dibujo de formas
  - Cierra la mano para agarrar y mover imágenes importadas
  - Mantén el dedo en la parte superior de la pantalla para seleccionar herramientas
- **Atajos de teclado:**
  - `ESC` - Salir de la aplicación
  - `u` - Deshacer última acción
  - `c` - Limpiar lienzo
  - `e` - Cambiar a borrador
  - `r` - Cambiar a color rojo
  - `g` - Cambiar a color verde
  - `b` - Cambiar a color azul
  - `+` - Aumentar tamaño del pincel
  - `-` - Disminuir tamaño del pincel
  - `o` - Importar imagen (solo macOS)
  - `d` - Eliminar imagen seleccionada

## Selección de Herramientas

La aplicación cuenta con una barra de selección de herramientas en la parte superior de la pantalla. Para seleccionar una herramienta:
1. Mueve tu dedo índice a la parte superior de la pantalla
2. Mantenlo sobre el icono de la herramienta deseada
3. Espera a que el círculo de selección se complete

## Herramientas de Dibujo

### Herramienta de Pincel
- Selecciona la herramienta de pincel desde la barra de herramientas
- Apunta con tu dedo índice para dibujar libremente
- Elige diferentes colores usando los atajos de teclado

### Herramientas de Formas (Línea, Rectángulo, Círculo)
1. Selecciona la herramienta de forma deseada
2. Apunta donde quieras comenzar la forma
3. Levanta los dedos índice y medio
4. Mueve tu mano para ajustar el tamaño/posición de la forma
5. Baja los dedos para completar la forma

### Borrador
- Selecciona la herramienta de borrador o presiona `e`
- Usa los mismos gestos que la herramienta de pincel para borrar

## Manipulación de Imágenes

### Importar Imágenes
- Presiona `o` para abrir el diálogo de selección de archivo (solo macOS)
- Selecciona un archivo de imagen para importar

### Mover Imágenes
1. Posiciona tu mano sobre la imagen importada
2. Cierra tu mano (haz un puño) para agarrar la imagen
3. Mueve tu mano para reposicionar la imagen
4. Abre tu mano para soltar

### Eliminar Imágenes
1. Agarra la imagen que quieres eliminar
2. Presiona `d` para eliminar la imagen seleccionada

## Solución de Problemas

Problemas comunes y soluciones:

1. **Cámara no detectada:**
   - Asegúrate de que tu cámara web esté correctamente conectada
   - Verifica si otras aplicaciones están usando la cámara
   - Verifica los permisos de cámara para la aplicación

2. **Problemas de detección de mano:**
   - Asegura buenas condiciones de iluminación
   - Mantén tu mano dentro del marco de la cámara
   - Mantén un fondo claro para una mejor detección

3. **Problemas de rendimiento:**
   - Cierra otras aplicaciones que consuman muchos recursos
   - Asegúrate de que tu computadora cumple con los requisitos mínimos
   - Considera reducir la resolución de la cámara si es necesario

## Contribuir

¡Las contribuciones son bienvenidas! No dudes en enviar un Pull Request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.
