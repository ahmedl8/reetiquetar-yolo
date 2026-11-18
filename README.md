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
| `users.example.json` | Plantilla de usuarios (copiar a `users.json`, que no se versiona) |
| `.env.example` | Variables de entorno (AWS, base de datos, FTP) |

## Instalación

```bash
pip install boto3 pillow pymysql
cp .env.example .env              # y exporta las variables
cp users.example.json users.json  # define usuarios, carpetas e instalaciones
python main.py
```

Ninguna credencial vive en el código: AWS, base de datos y FTP salen del entorno, y las contraseñas de los revisores, de `users.json` (en texto plano; pensado para uso interno).

## Limitaciones

- El esquema de la base de datos de reetiquetado no se incluye; la herramienta espera una vista de casos pendientes y una tabla de casos hechos.
- Los nombres y ejemplos son ficticios.
