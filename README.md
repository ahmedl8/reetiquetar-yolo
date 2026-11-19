# YOLO Box Reviewer

Herramienta de escritorio (Tkinter) para **revisar, corregir y reetiquetar las cajas (bounding boxes) de detecciones YOLO** sobre imágenes térmicas. Sirve para cerrar el ciclo de mejora de un modelo de detección: los casos que los vigilantes o el filtro de IA marcaron como dudosos llegan a un revisor, que corrige la clase o la caja, y el resultado queda registrado para reentrenar.

> **EN:** Desktop tool (Tkinter) to review, fix and relabel YOLO bounding boxes on thermal images, pulling cases from S3/FTP and recording what was relabelled in MySQL.

## Qué hace

- **Dos orígenes de imágenes:** sincronización por FTP (con cajas en un `.txt` por imagen) y descarga desde S3 de los casos pendientes de reetiquetado, cuyas cajas se extraen del `log.txt` de cada tour (`tour_log_parser.py`).
- **Editor de cajas:** revisar, mover, redimensionar y cambiar la clase de cada caja; el resultado se guarda en formato de etiquetado y en una carpeta de revisados.
- **Cola de reetiquetado:** consulta los casos pendientes en MySQL y marca como hechos los ya revisados (`cav_relabel.py`), con un usuario de base de datos de solo lectura sobre los pendientes y solo inserción sobre lo hecho.
- **Multiusuario:** cada revisor tiene sus instalaciones, carpetas locales y credenciales en un `users.json`.
- **Sesión:** recuerda usuario y geometría de la ventana entre ejecuciones.

## Archivos

| Archivo | Función |
|---|---|
| `main.py` | Interfaz, acceso, descarga y editor de cajas |
| `cav_relabel.py` | Puente con la base de datos: casos pendientes, descarga desde S3, marcado como reetiquetado |
| `tour_log_parser.py` | Extrae sector → imagen → clase → caja del log de texto de un tour |
| `security_utils.py` | Hash scrypt de contraseñas y validación de URLs |
| `users.example.json` | Plantilla de usuarios (copiar a `users.json`, que no se versiona) |
| `.env.example` | Variables de entorno (AWS, base de datos, FTP) |

## Instalación

```bash
pip install -r requirements.txt   # boto3, Pillow, PyMySQL, imageio-ffmpeg
cp .env.example .env              # y exporta las variables
cp users.example.json users.json  # define usuarios, carpetas e instalaciones
python security_utils.py          # genera el hash de la contraseña de cada revisor
python main.py
```

Ninguna credencial vive en el código: AWS, base de datos y FTP salen del entorno. Las contraseñas de los revisores se guardan en `users.json` como hash **scrypt** (`password_hash`), nunca en texto plano; el campo antiguo `password` se rechaza salvo que se active `ALLOW_PLAINTEXT_USERS=1` para migrar.

## Seguridad

- **FTPS por defecto** (`FTP_TLS` con canal de datos protegido). `FTP_INSECURE=1` desactiva el cifrado y solo es para servidores de pruebas.
- Los enlaces de vídeo solo se abren si son `https`.
- Sin instalaciones asignadas, el revisor no consulta la base de datos (evita un `IN ()` inválido).
- Si falta `users.json` o es inválido, la pantalla de login lo indica en vez de fallar en silencio.

## Tests

```bash
python -m pytest -q
```

Cubren el hash y la verificación de contraseñas, el rechazo de URLs no `https` y el caso de usuario sin instalaciones. No necesitan base de datos ni pantalla.

## Limitaciones

- El esquema de la base de datos de reetiquetado no se incluye; la herramienta espera una vista de casos pendientes y una tabla de casos hechos.
- `main.py` es un único archivo de ~3.500 líneas; falta dividirlo en módulos y cubrir la interfaz con tests.
- Contiene dos variantes de `_diff_review_boxes`; la activa es la última definida y la otra se conserva como `_diff_review_boxes_alt` hasta decidir cuál es la correcta.
- Los nombres y ejemplos son ficticios.
