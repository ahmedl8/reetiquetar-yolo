"""
cav_relabel.py — Puente entre box_reviewer y la base de datos cav.

Sustituye (o complementa) el origen FTP actual: en vez de descargar lo
que haya en el servidor, consulta v_pendiente_reetiquetado en cav para
saber exactamente qué frames necesitan que se corrija la clase que
detectó YOLO, descarga la imagen original de S3 y reconstruye la caja
inicial a partir del log.txt del tour (no hay .txt de cajas por
imagen en S3 — las cajas viven dentro de ese log de texto).

Requiere:
    pip install pymysql boto3
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import boto3
import pymysql
from PIL import Image

from tour_log_parser import extraer_cajas, CajaDetectada

log = logging.getLogger("cav_relabel")


@dataclass
class CasoReetiquetado:
    id_ai: int
    instalacion: str
    sector: str                # p.ej. "sector_320"
    clase_yolo: str
    clase_sugerida: str        # motivo del vigilante, o decisión de GPT
    origen_sugerencia: str     # 'vigilante' | 'gpt'
    path_s3: list[str]
    img_filename: str | None
    img_number: int | None
    sector_deg: int
    motivo_yolo: str | None = None   # cav.gpt_feedback.motivo más reciente para este id_ai/sector
    clase_s3: str | None = None      # clase real detectada por YOLO según tour_json (ground truth, no la decide GPT)


# ─────────────────────── Conexión a cav (solo lectura) ─────────────────────

def _connect(cfg: dict) -> pymysql.connections.Connection:
    """
    cfg necesita: cav_host, cav_user, cav_pass, cav_db (nombre de la BD,
    normalmente 'cav'). Usar SIEMPRE un usuario MySQL de solo LECTURA
    sobre v_pendiente_reetiquetado, y de solo ESCRITURA (INSERT) sobre
    reetiquetado_hecho — nunca las credenciales del programa principal.
    """
    return pymysql.connect(
        host=cfg["cav_host"],
        user=cfg["cav_user"],
        password=cfg["cav_pass"],
        database=cfg.get("cav_db", "cav"),
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
    )


# ─────────────────────── Selección de candidatos ───────────────────────────

def get_pending_relabel(cfg: dict, instalaciones: list[str],
                         limit: int = 200) -> list[CasoReetiquetado]:
    """
    Consulta v_pendiente_reetiquetado filtrando por las instalaciones
    del usuario, y para cada fila localiza en tour_json el sector y la
    imagen exactos que hay que descargar.
    """
    if not instalaciones:
        # `IN ()` es un error de sintaxis SQL: sin instalaciones no hay casos.
        return []
    placeholders = ",".join(["%s"] * len(instalaciones))
    sql = (
        f"SELECT id_ai, instalacion, sector, clase_yolo, clase_sugerida, "
        f"origen_sugerencia, tour_json "
        f"FROM v_pendiente_reetiquetado "
        f"WHERE instalacion IN ({placeholders}) "
        f"ORDER BY ts_fire DESC LIMIT %s"
    )
    casos: list[CasoReetiquetado] = []
    conn = _connect(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (*instalaciones, limit))
            rows = cur.fetchall()
            motivos = _fetch_motivos(conn, [r["id_ai"] for r in rows])
            for row in rows:
                caso = _build_caso(row, motivos.get((row["id_ai"], row["sector"])))
                if caso is not None:
                    casos.append(caso)
    finally:
        conn.close()
    return casos


def _fetch_motivos(conn: pymysql.connections.Connection,
                    id_ais: list[int]) -> dict[tuple[int, str], str]:
    """
    Trae, para cada (id_ai, sector), el `motivo` NO NULO más reciente que
    un vigilante ya haya confirmado en gpt_feedback — esto es la razón por
    la que YOLO se equivocó (JAULA/PAJARO/TUBOS/BOYA/BARCO_LEJANO...), no
    la valoración de GPT (gpt_explicacion), que esa se afina por prompt y
    no tiene nada que hacer en el revisor de cajas.

    Puede no haber ninguna fila todavía (el vigilante aún no dio feedback
    de motivo para ese caso) — en ese caso simplemente no aparece en el
    dict y el revisor debe mostrar "sin motivo registrado".
    """
    if not id_ais:
        return {}
    placeholders = ",".join(["%s"] * len(id_ais))
    sql = (
        f"SELECT id_ai, sector, motivo FROM gpt_feedback "
        f"WHERE id_ai IN ({placeholders}) AND motivo IS NOT NULL "
        f"ORDER BY id DESC"
    )
    out: dict[tuple[int, str], str] = {}
    with conn.cursor() as cur:
        cur.execute(sql, tuple(id_ais))
        for row in cur.fetchall():
            key = (row["id_ai"], row["sector"])
            out.setdefault(key, row["motivo"])   # ORDER BY id DESC → primero = más reciente
    return out


def _build_caso(row: dict, motivo_yolo: str | None = None) -> CasoReetiquetado | None:
    try:
        tour = json.loads(row["tour_json"])
    except (TypeError, ValueError):
        log.warning(f"[REETIQ] id_ai={row['id_ai']}: tour_json inválido, se omite")
        return None

    sector_deg = int(row["sector"].replace("sector_", ""))
    sector_data = next(
        (s for s in tour.get("sectors", []) if s.get("dir") == row["sector"]),
        None,
    )
    if not sector_data or not sector_data.get("detections"):
        log.warning(f"[REETIQ] id_ai={row['id_ai']} sector={row['sector']}: "
                    f"sin detecciones en tour_json, se omite")
        return None

    det = sector_data["detections"][0]
    return CasoReetiquetado(
        id_ai=row["id_ai"],
        instalacion=row["instalacion"],
        sector=row["sector"],
        clase_yolo=row["clase_yolo"],
        clase_sugerida=row["clase_sugerida"],
        origen_sugerencia=row["origen_sugerencia"],
        path_s3=tour["path_s3"],
        img_filename=det.get("img_filename"),
        img_number=det.get("img_number"),
        sector_deg=sector_deg,
        motivo_yolo=motivo_yolo,
        clase_s3=det.get("class"),
    )


# ─────────────────────── Descarga S3 + reconstrucción de cajas ─────────────

def _s3_client(cfg: dict):
    return boto3.client(
        "s3",
        aws_access_key_id=cfg["aws_access_key"],
        aws_secret_access_key=cfg["aws_secret_key"],
        region_name=cfg["aws_region"],
    )


def descargar_caso(cfg: dict, caso: CasoReetiquetado,
                    dst_jpg: Path, dst_txt: Path) -> bool:
    """
    Descarga la imagen del sector (path_s3/sector_XXX/img_filename) y
    el log.txt del tour (path_s3/log.txt), extrae las cajas de ese
    sector+imagen y las escribe ya en formato YOLO normalizado en
    dst_txt. Devuelve False si algo falta (imagen o log ilegible) —
    en ese caso el revisor debe poder abrir igualmente la imagen sin
    caja previa y dibujarla a mano.
    """
    s3 = _s3_client(cfg)
    bucket = cfg["s3_bucket"]
    base_key = "/".join(caso.path_s3)

    if not caso.img_filename:
        log.warning(f"[REETIQ] id_ai={caso.id_ai}: sin img_filename, "
                    f"no se puede descargar imagen")
        return False

    img_key = f"{base_key}/{caso.sector}/{caso.img_filename}"
    try:
        s3.download_file(bucket, img_key, str(dst_jpg))
    except Exception as e:
        log.error(f"[REETIQ] id_ai={caso.id_ai}: fallo descargando "
                  f"s3://{bucket}/{img_key} — {e}")
        return False

    log_key = f"{base_key}/log.txt"
    try:
        obj = s3.get_object(Bucket=bucket, Key=log_key)
        log_text = obj["Body"].read().decode("utf-8", errors="replace")
    except Exception as e:
        log.warning(f"[REETIQ] id_ai={caso.id_ai}: no se pudo leer "
                    f"s3://{bucket}/{log_key} ({e}) — se descarga la "
                    f"imagen sin caja previa")
        dst_txt.write_text("", encoding="utf-8")
        return True

    if caso.img_number is None:
        dst_txt.write_text("", encoding="utf-8")
        return True

    cajas = extraer_cajas(log_text, caso.sector_deg, caso.img_number)
    if not cajas:
        log.info(f"[REETIQ] id_ai={caso.id_ai} sector={caso.sector}: "
                 f"log.txt leído pero sin cajas para img={caso.img_number} "
                 f"— se abre sin caja previa")
        dst_txt.write_text("", encoding="utf-8")
        return True

    _escribir_boxes_json(dst_jpg, dst_txt, cajas,
                         resolver_nombre_clase(caso.clase_sugerida,
                                               caso.clase_yolo,
                                               caso.clase_s3))
    return True


# NOTA sobre las coordenadas del log: en un principio pensé que las cajas
# del log.txt estaban en el espacio de inferencia de YOLO (img_size x
# img_size, con letterbox) y que había que reescalarlas a la resolución
# real de la foto. Comprobado con un tour real descargado de S3
# (site_a, id_ai 13957, sector_260, imagen 0: log dice
# "caja -> [1225, 293, 108, 94]", foto real 1280x960) — la caja cruda,
# SIN ninguna transformación, ya cae exactamente sobre el barco. Es
# decir: Ultralytics ya devuelve las cajas reescaladas a la resolución
# original de la imagen antes de que se escriban en el log; img_size es
# solo un parámetro interno de inferencia y no afecta a las coordenadas
# finales. Por eso NO se aplica ningún reescalado aquí — solo se recorta
# contra los bordes de la imagen real, porque a veces la caja se pasa
# unos píxeles por el borde (normal, sin importancia).
CLASE_PENDIENTE = "SIN_CLASIFICAR"

# gpt_feedback.clase_yolo (pese al nombre) no es la clase que detectó YOLO:
# es la categoría que decidió GPT (barco_amenaza/barco_lejano/tubo/jaula/
# pajaro/boya...), y cuando GPT falla al responder cae en "contexto" como
# placeholder de error. Tu taxonomía real de etiquetado (box_reviewer,
# GOOD_DETECTION_CLASSES) es otra: barco / barco2 / barco3 / persona, más
# las clases negativas de motivo que se añaden aquí abajo.
#
# PRIORIDAD DE LA SUGERENCIA (de más a menos fiable):
#   1. clase_sugerida = COALESCE(motivo, gpt_decision) — si el vigilante ya
#      confirmó un motivo (JAULA/PAJARO/TUBOS/BOYA/BARCO_LEJANO/SOL/
#      TIERRA), esa es la fuente más fiable: una persona vio la foto.
#      Ver MOTIVO_A_ETIQUETA.
#   2. clase_yolo_db (la categoría de GPT) — solo cuando aún no hay motivo
#      del vigilante (caso "gpt_decision='falso' AND motivo IS NULL" de la
#      vista v_pendiente_reetiquetado). Ver CLASE_DB_A_ETIQUETA.
#   3. clase_s3 (la clase que puso YOLO originalmente) — último recurso.
#      OJO: en los casos de motivo JAULA/PAJARO/TUBOS/BOYA/SOL/TIERRA esta
#      clase_s3 es precisamente LA ETIQUETA EQUIVOCADA que hay que
#      corregir (p.ej. YOLO dijo "persona" y era un pájaro) — por eso el
#      motivo del vigilante tiene que ganarle siempre que exista.

# Motivo confirmado por el vigilante → clase nueva de etiquetado. Estas
# clases son NEGATIVAS a propósito (no están en GOOD_DETECTION_CLASSES de
# main.py): sirven para enseñarle al modelo a distinguir explícitamente
# "esto parece una amenaza pero es una jaula/pájaro/tubo/boya/reflejo de
# sol/tierra", en vez de simplemente borrar la caja y perder esa señal.
# BARCO_LEJANO es la única excepción: ahí sí hay un barco real, solo que
# lejos, así que mapea a la clase positiva existente "barco2".
MOTIVO_A_ETIQUETA: dict[str, str] = {
    "BARCO_LEJANO": "barco2",
    "JAULA":        "jaula",
    "PAJARO":       "pajaro",
    "TUBOS":        "tubos",
    "BOYA":         "boya",
    "SOL":          "sol",
    "TIERRA":       "tierra",
}

# Categoría de GPT (gpt_feedback.clase_yolo) → clase de etiquetado. Solo
# se usa cuando TODAVÍA no hay motivo confirmado por el vigilante (ver
# prioridad arriba) — es la mejor sugerencia disponible en ese momento,
# aunque venga de la misma IA que ya se equivocó.
#
# ⚠️ barco3 (barcos muy cerca, ocupando casi toda la imagen) todavía no lo
# decide el cav — cuando lo implementes, añade aquí la categoría que
# corresponda.
CLASE_DB_A_ETIQUETA: dict[str, str] = {
    "barco_amenaza": "barco",
    "barco_lejano":  "barco2",
}

# Sentinel de cls_id: cav_relabel.py no conoce la numeración real de tu
# dataset (esa vive en los .txt ya etiquetados que main.py escanea con
# _build_known_cls), así que no puede inventar un cls_id fiable aquí.
# main.py debe resolver este -1 contra self._known_cls buscando por
# "name" antes de mostrar/guardar la caja.
CLS_ID_PENDIENTE = -1


def resolver_nombre_clase(clase_sugerida: str | None,
                           clase_yolo_db: str | None,
                           clase_s3: str | None) -> str:
    """
    Decide qué clase precargar en la caja reconstruida, con la prioridad
    descrita arriba: motivo del vigilante > categoría de GPT > clase
    original de YOLO > SIN_CLASIFICAR.
    """
    if clase_sugerida:
        motivo = clase_sugerida.upper()
        if motivo in MOTIVO_A_ETIQUETA:
            return MOTIVO_A_ETIQUETA[motivo]
    if clase_yolo_db and clase_yolo_db in CLASE_DB_A_ETIQUETA:
        return CLASE_DB_A_ETIQUETA[clase_yolo_db]
    if clase_s3:
        return clase_s3
    return CLASE_PENDIENTE


def _escribir_boxes_json(img_path: Path, txt_path: Path,
                          cajas: list[CajaDetectada],
                          nombre_clase: str) -> None:
    """
    OJO: box_reviewer (main.py) NO lee YOLO plano en el .txt — su
    `_load_image` llama a `load_json_boxes()`, que hace
    `json.load(f).get("boxes", [])` esperando una lista de dicts con
    claves cls_id/name/conf/xyxy (xyxy en PÍXELES, ya en la resolución
    real de la foto — el log.txt las guarda así directamente, sin
    letterbox que deshacer, ver nota más arriba).

    Antes esta función escribía líneas "0 cx cy w h" en formato YOLO
    normalizado — válidas como .txt de YOLO, pero no como JSON, así que
    `json.load()` fallaba con JSONDecodeError, el except silencioso
    devolvía `[]`, y el revisor mostraba "0 caja(s)" / "Sin cajas" para
    TODOS los casos de reetiquetado CAV. Este era el bug original.

    cls_id se deja en CLS_ID_PENDIENTE (-1): main.py lo resuelve contra
    self._known_cls por nombre al cargar, para no inventar aquí una
    numeración que podría chocar con la de tu dataset ya etiquetado.
    """
    with Image.open(img_path) as im:
        iw, ih = im.size

    boxes = []
    for c in cajas:
        x1, y1, x2, y2 = c.to_xyxy()
        x1 = max(0.0, min(float(iw), x1)); x2 = max(0.0, min(float(iw), x2))
        y1 = max(0.0, min(float(ih), y1)); y2 = max(0.0, min(float(ih), y2))
        boxes.append({
            "cls_id": CLS_ID_PENDIENTE,
            "name":   nombre_clase,
            "conf":   None,   # reconstruida del log, no es una detección con score
            "xyxy":   [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

    txt_path.write_text(json.dumps({"boxes": boxes}, ensure_ascii=False),
                         encoding="utf-8")


# ─────────────────────── Marcado de revisado ────────────────────────────────

def mark_relabeled(cfg: dict, id_ai: int, sector: str,
                    revisor: str, clase_final: str) -> None:
    conn = _connect(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO reetiquetado_hecho (id_ai, sector, revisor, clase_final) "
                "VALUES (%s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE clase_final=VALUES(clase_final), "
                "revisor=VALUES(revisor), ts=CURRENT_TIMESTAMP",
                (id_ai, sector, revisor, clase_final),
            )
        conn.commit()
    finally:
        conn.close()