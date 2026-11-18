"""
tour_log_parser.py — Extrae las cajas YOLO de un tour a partir de su
log.txt (el mismo que sube el programa de tours a S3, junto a los
vídeos, en cada carpeta de tour: alarmas/.../<tour>/log.txt).

No hay un .txt de cajas por imagen en S3 (a diferencia del flujo FTP
de box_reviewer); las cajas viven dentro de este log de texto, con un
formato semi-estructurado por sector -> imagen -> clase -> caja.
"""
import re
from dataclasses import dataclass

SECTOR_RE = re.compile(r"----SECTOR (\d+)----")
IMG_BLOCK_RE = re.compile(r"Imagen: (\d+)\.\.\.")
CLASE_RE = re.compile(r"([A-Za-zÁÉÍÓÚñÑ_]+): repeticiones ->\d+")
CAJA_RE = re.compile(
    r"caja -> \[(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\]\s+"
    r"puntuacion -> ([\d.]+)"
)


@dataclass
class CajaDetectada:
    clase: str
    x: int
    y: int
    w: int
    h: int
    score: float

    def to_xyxy(self):
        # OJO: x,y aquí son el CENTRO de la caja, no la esquina superior-
        # izquierda. inferenceControl.py escribe estas cifras a partir de
        # box.xywh (Ultralytics), que devuelve (centro_x, centro_y, w, h) —
        # nunca top-left. Tratar x,y como top-left desplaza la caja media
        # anchura a la derecha y media altura hacia abajo (bug verificado
        # contra site_a/13957/sector_260/imagen_0: con esta corrección el
        # borde derecho/inferior calzan con la caja real quemada por
        # res_alarm.plot() en la imagen del tour).
        return (self.x - self.w / 2, self.y - self.h / 2,
                self.x + self.w / 2, self.y + self.h / 2)


def extraer_cajas(log_text: str, sector_deg: int, img_number: int) -> list[CajaDetectada]:
    """
    Devuelve las cajas detectadas para la imagen `img_number` dentro
    del bloque `----SECTOR {sector_deg}----` de este log de tour.

    El "sector" de tour_json (p.ej. sector_320) es el mismo entero que
    aparece en "----SECTOR 320----" en el log.
    """
    lines = log_text.splitlines()

    # 1. localizar el inicio del bloque de este sector (y su fin: el
    #    siguiente "----SECTOR" o el final del fichero)
    start = end = None
    for i, line in enumerate(lines):
        m = SECTOR_RE.search(line)
        if m and int(m.group(1)) == sector_deg:
            start = i
        elif m and start is not None:
            end = i
            break
    if start is None:
        return []
    if end is None:
        end = len(lines)
    bloque = lines[start:end]

    # 2. dentro del bloque, localizar "Imagen: {img_number}..." y leer
    #    las líneas "caja ->" hasta la siguiente "Imagen: N..." o fin
    #    de bloque
    cajas: list[CajaDetectada] = []
    dentro = False
    clase_actual = None
    for line in bloque:
        m_img = IMG_BLOCK_RE.search(line)
        if m_img:
            dentro = (int(m_img.group(1)) == img_number)
            continue
        if not dentro:
            continue
        m_clase = CLASE_RE.search(line)
        if m_clase:
            clase_actual = m_clase.group(1).lower()
            continue
        m_caja = CAJA_RE.search(line)
        if m_caja and clase_actual:
            x, y, w, h, score = m_caja.groups()
            cajas.append(CajaDetectada(
                clase=clase_actual, x=int(x), y=int(y),
                w=int(w), h=int(h), score=float(score),
            ))
    return cajas