"""
YOLO Box Reviewer - Herramienta de revisión y edición de bounding boxes YOLO
======================================================================
Requisitos: pip install pillow
Opcional:   pip install imageio-ffmpeg   (para contexto de vídeo)
            pip install boto3            (para regenerar URLs S3 caducadas)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json, os, shutil, threading, time, queue, ftplib
from pathlib import Path
from PIL import Image, ImageTk

from security_utils import check_user_password, require_https

try:
    import imageio_ffmpeg
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

import importlib.util
CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None   # opcional

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import cav_relabel
    CAV_RELABEL_AVAILABLE = True
except ImportError:
    CAV_RELABEL_AVAILABLE = False

# ─────────────────────── Credenciales (solo por entorno) ───────────────────
# Ninguna credencial vive en el código. Define estas variables de entorno
# (ver .env.example) y crea users.json a partir de users.example.json.

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID",     "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION            = os.getenv("AWS_REGION",            "eu-west-1")
S3_BUCKET             = os.getenv("S3_BUCKET",             "")

# Usuario MySQL dedicado: solo LECTURA sobre la vista de pendientes y solo
# INSERT sobre la tabla de reetiquetado hecho.
CAV_DB_HOST = os.getenv("CAV_DB_HOST", "localhost")
CAV_DB_USER = os.getenv("CAV_DB_USER", "")
CAV_DB_PASS = os.getenv("CAV_DB_PASS", "")
CAV_DB_NAME = os.getenv("CAV_DB_NAME", "cav")

# ─────────────────────── Configuración de usuarios ─────────────────────────

FTP_HOST = os.getenv("FTP_HOST", "")
FTP_PORT = int(os.getenv("FTP_PORT", "21"))
# FTPS (FTP sobre TLS) por defecto. FTP_INSECURE=1 solo para servidores de
# pruebas que no lo soporten: usuario y contraseña viajarían en claro.
FTP_INSECURE = os.getenv("FTP_INSECURE", "0") == "1"
# Solo para migrar: acepta el campo antiguo "password" en texto plano.
ALLOW_PLAINTEXT_USERS = os.getenv("ALLOW_PLAINTEXT_USERS", "0") == "1"


def _load_user_config() -> dict:
    """Carga users.json (junto a main.py o en USERS_FILE) y completa cada
    usuario con las credenciales comunes tomadas del entorno."""
    path = _pl_mod.Path(os.getenv("USERS_FILE", _pl_mod.Path(__file__).with_name("users.json")))
    try:
        users = _json_mod.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"[config] No existe {path}. Copia users.example.json a users.json "
              f"y genera los hashes con: python security_utils.py")
        return {}
    except ValueError as exc:
        print(f"[config] {path} no es un JSON válido: {exc}")
        return {}
    if not isinstance(users, dict):
        print(f"[config] {path} debe ser un objeto {{usuario: configuración}}.")
        return {}
    for cfg in users.values():
        cfg.update({
            "aws_access_key": AWS_ACCESS_KEY_ID,
            "aws_secret_key": AWS_SECRET_ACCESS_KEY,
            "aws_region":     AWS_REGION,
            "s3_bucket":      S3_BUCKET,
            "cav_host":       CAV_DB_HOST,
            "cav_user":       CAV_DB_USER,
            "cav_pass":       CAV_DB_PASS,
            "cav_db":         CAV_DB_NAME,
        })
    return users


import json as _json_mod, pathlib as _pl_mod
USER_CONFIG = _load_user_config()

# ── Sesión (usuario + geometría) ────────────────────────────────────────────
SESSION_FILE = _pl_mod.Path.home() / ".box_reviewer_session.json"

def _load_session():
    try:
        return _json_mod.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_session(data: dict):
    try:
        existing = _load_session()
        existing.update(data)
        SESSION_FILE.write_text(
            _json_mod.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass

# ── Constantes ──────────────────────────────────────────────────────────────
TIME_BETWEEN_SNAPSHOTS = 1.5
VIDEO_CONTEXT_SECS     = 3
HANDLE_SIZE = 8
MIN_BOX_PX  = 10
COLORS = [
    "#FF3333","#33FF33","#3399FF","#FF9933","#CC33FF",
    "#33FFCC","#FF3399","#FFFF33","#00CCFF","#FF6600",
]

# ── Utilidades ──────────────────────────────────────────────────────────────

def color_for_cls(cls_id):
    return COLORS[int(cls_id) % len(COLORS)]

def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = ((x1+x2)/2)/img_w;  cy = ((y1+y2)/2)/img_h
    w  = (x2-x1)/img_w;      h  = (y2-y1)/img_h
    return cx, cy, w, h

def load_json_boxes(json_path):
    try:
        with open(json_path,"r",encoding="utf-8") as f:
            data = json.load(f)
        return data.get("boxes",[])
    except Exception:
        return []

def save_yolo_labels(label_path, boxes, img_w, img_h):
    lines = []
    for b in boxes:
        x1,y1,x2,y2 = b["xyxy"]
        cx,cy,w,h = xyxy_to_yolo(x1,y1,x2,y2,img_w,img_h)
        cx=max(0.,min(1.,cx)); cy=max(0.,min(1.,cy))
        w =max(0.,min(1.,w));  h =max(0.,min(1.,h))
        lines.append(f"{b['cls_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(label_path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))

# ── FTP helper ──────────────────────────────────────────────────────────────

def _make_ftp(cfg) -> ftplib.FTP:
    if FTP_INSECURE:
        ftp = ftplib.FTP()
        ftp.connect(FTP_HOST, FTP_PORT, timeout=20)
        ftp.login(cfg["ftp_user"], cfg["ftp_pass"])
    else:
        ftp = ftplib.FTP_TLS()
        ftp.connect(FTP_HOST, FTP_PORT, timeout=20)
        ftp.login(cfg["ftp_user"], cfg["ftp_pass"])
        ftp.prot_p()          # cifra también el canal de datos
    ftp.set_pasv(True)
    return ftp

def _ftp_size(ftp: ftplib.FTP, remote_path: str) -> int:
    try:
        return ftp.size(remote_path)
    except Exception:
        return -1

def _ftp_list_jpgs(ftp, remote_dir):
    try:
        names = ftp.nlst(remote_dir)
        print("FTP NLST result:", names)
        return sorted(n for n in names if n.lower().endswith(".jpg"))
    except Exception as e:
        print("FTP NLST error:", e)
        return []

def _ftp_download(ftp: ftplib.FTP, remote_path: str, local_path: Path,
                  progress_cb=None) -> bool:
    tmp = local_path.with_suffix(".ftptmp")
    try:
        remote_sz = _ftp_size(ftp, remote_path)
        written = 0
        with open(tmp, "wb") as f:
            def _write(block):
                nonlocal written
                f.write(block)
                written += len(block)
                if progress_cb and remote_sz > 0:
                    progress_cb(written, remote_sz)
            ftp.retrbinary(f"RETR {remote_path}", _write, blocksize=65536)
        if remote_sz > 0 and tmp.stat().st_size != remote_sz:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(local_path)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False

def _ftp_delete(ftp: ftplib.FTP, remote_path: str) -> bool:
    try:
        ftp.delete(remote_path)
        return True
    except Exception:
        return False

# ── Filtro de confianza ──────────────────────────────────────────────────────

def _boxes_overlap(xyxy1, xyxy2, min_overlap_ratio=0.30):
    x1 = max(xyxy1[0], xyxy2[0]);  y1 = max(xyxy1[1], xyxy2[1])
    x2 = min(xyxy1[2], xyxy2[2]);  y2 = min(xyxy1[3], xyxy2[3])
    if x2 <= x1 or y2 <= y1:
        return False
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(1, (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1]))
    a2 = max(1, (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1]))
    return (inter / min(a1, a2)) >= min_overlap_ratio


def _passes_conf_filter(boxes, min_conf: float) -> bool:
    if min_conf <= 0.0:
        return True
    if not boxes:
        return False

    n = len(boxes)
    coords = [b.get("xyxy", [0, 0, 0, 0]) for b in boxes]

    in_overlap_group = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_overlap(coords[i], coords[j]):
                in_overlap_group[i] = True
                in_overlap_group[j] = True

    has_overlap_groups = any(in_overlap_group)

    for i, b in enumerate(boxes):
        if has_overlap_groups and not in_overlap_group[i]:
            continue
        c = b.get("conf")
        if c is not None:
            try:
                if float(c) < min_conf:
                    return False
            except (TypeError, ValueError):
                pass
    return True

# ── Error de vídeo ──────────────────────────────────────────────────────────

def _classify_video_error(stderr_txt: str, url: str) -> tuple:
    s = (stderr_txt or "").lower()
    if any(k in s for k in ("403","forbidden","access denied","expired",
                             "request has expired","signatureexpired",
                             "nosuchkey","nosuchbucket","404","not found")):
        import re, datetime
        exp_m = re.search(r'x-amz-expires=(\d+)', url, re.I)
        dat_m = re.search(r'x-amz-date=(\d{8})', url, re.I)
        detail = ""
        if exp_m and dat_m:
            try:
                d = datetime.datetime.strptime(dat_m.group(1),"%Y%m%d")
                d += datetime.timedelta(seconds=int(exp_m.group(1)))
                detail = f"El enlace expiró el {d.strftime('%d/%m/%Y')}"
            except Exception:
                pass
        titulo = ("🗑  Vídeo eliminado o enlace expirado"
                  if any(k in s for k in ("403","forbidden","expired"))
                  else "🔍  Vídeo no encontrado")
        return titulo, detail or "El vídeo ya no existe o el enlace ha caducado."
    if any(k in s for k in ("connection refused","network unreachable",
                             "timed out","timeout","eof","broken pipe",
                             "connection reset","no route to host")):
        return ("🌐  Sin conexión al servidor",
                "Comprueba que estás conectado a la red.")
    if any(k in s for k in ("invalid data","invalid url","no such file",
                             "unable to open","protocol not found")):
        return ("🔗  Enlace de vídeo inválido",
                "La URL puede haber expirado. Vuelve a descargar las revisiones.")
    if any(k in s for k in ("invalid data found","moov atom not found",
                             "could not find codec","decoder")):
        return ("⚠  Formato de vídeo no compatible",
                "El fichero existe pero no se puede decodificar.")
    snippet = stderr_txt.strip().splitlines()[-1][:80] if stderr_txt.strip() else "sin detalle"
    return ("❌  Error al cargar el vídeo", snippet)


# ─────────────────────── S3 URL refresh ────────────────────────────────────

def _refresh_s3_presigned_url(url: str, cfg: dict, expires_in: int = 7200) -> str:
    if not BOTO3_AVAILABLE:
        print("[S3] boto3 no disponible — instala con: pip install boto3")
        return url
    if not url or 'x-amz-signature' not in url.lower():
        return url
    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        host   = parsed.netloc
        path   = parsed.path.lstrip('/')

        if not host.endswith('.amazonaws.com'):
            return url

        parts = host.split('.')
        if len(parts) >= 4 and parts[1] == 's3':
            bucket = parts[0]
            key    = path
        elif parts[0] == 's3':
            segments = path.split('/', 1)
            if len(segments) < 2:
                return url
            bucket, key = segments[0], segments[1]
        else:
            return url

        aws_key    = cfg.get("aws_access_key", AWS_ACCESS_KEY_ID)
        aws_secret = cfg.get("aws_secret_key", AWS_SECRET_ACCESS_KEY)
        aws_region = cfg.get("aws_region",     AWS_REGION)

        if not aws_key or not aws_secret:
            return url

        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=aws_region,
        )
        new_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expires_in,
        )
        print(f"[S3] URL refrescada OK  bucket={bucket}  "
              f"key=…{key[-50:]}  expires_in={expires_in}s")
        return new_url

    except Exception as e:
        print(f"[S3] Error al refrescar URL: {e}")
        return url


# ─────────────────────── EmptyBoxesDialog ──────────────────────────────────

class EmptyBoxesDialog(tk.Toplevel):
    BACKGROUND = "background"
    DISCARD    = "discard"

    def __init__(self, master, img_name):
        super().__init__(master)
        self.title("Sin cajas detectadas")
        self.resizable(False, False)
        self.grab_set()
        self.configure(bg="#1a1a2e")
        self.result = None
        w, h = 420, 220
        self.geometry(f"{w}x{h}+{(self.winfo_screenwidth()-w)//2}+"
                      f"{(self.winfo_screenheight()-h)//2}")
        tk.Label(self, text="⚠  Sin cajas", bg="#1a1a2e", fg="#ffcc44",
                 font=("Consolas",13,"bold")).pack(pady=(22,4))
        tk.Label(self, text=("Has eliminado todas las cajas de:\n"
                             + img_name + "\n\n¿Qué quieres hacer con este frame?"),
                 bg="#1a1a2e", fg="#aaaacc",
                 font=("Consolas",9), justify="center").pack(pady=(0,16))
        btn_fr = tk.Frame(self, bg="#1a1a2e")
        btn_fr.pack()
        tk.Button(btn_fr, text="🖼  Guardar como background",
                  bg="#1a5c1a", fg="white", relief="flat",
                  font=("Consolas",9,"bold"), padx=14, pady=8,
                  command=self._bg).pack(side="left", padx=8)
        tk.Button(btn_fr, text="🗑  Descartar",
                  bg="#5c1a1a", fg="white", relief="flat",
                  font=("Consolas",9,"bold"), padx=14, pady=8,
                  command=self._discard).pack(side="left", padx=8)
        self.bind("<Escape>", lambda e: self._discard())

    def _bg(self):      self.result = self.BACKGROUND; self.destroy()
    def _discard(self): self.result = self.DISCARD;    self.destroy()


# ─────────────────────── LoginDialog ───────────────────────────────────────

class LoginDialog(tk.Toplevel):
    def __init__(self, master, on_success):
        super().__init__(master)
        self.title("YOLO Box Reviewer — Acceso")
        self.resizable(False, False)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.configure(bg="#1a1a2e")
        self._on_success = on_success
        self._attempts   = 0
        w, h = 400, 400
        self.geometry(f"{w}x{h}+{(self.winfo_screenwidth()-w)//2}+"
                      f"{(self.winfo_screenheight()-h)//2}")
        tk.Label(self, text="BOX  REVIEWER", bg="#1a1a2e", fg="#7ecfff",
                 font=("Consolas",16,"bold")).pack(pady=(28,2))
        tk.Label(self, text="Editor de Bounding Boxes", bg="#1a1a2e", fg="#555577",
                 font=("Consolas",9)).pack(pady=(0,20))
        form = tk.Frame(self, bg="#1a1a2e")
        form.pack()
        for row, label, attr, kw in [
            (0, "Usuario",    "_user_entry", {}),
            (1, "Contraseña", "_pass_entry", {"show": "●"}),
        ]:
            tk.Label(form, text=label, bg="#1a1a2e", fg="#aaaacc",
                     font=("Consolas",9), width=10, anchor="e").grid(
                     row=row, column=0, padx=(0,8), pady=6)
            e = tk.Entry(form, width=22, bg="#13131f", fg="white",
                         insertbackground="white", relief="flat",
                         font=("Consolas",10), highlightthickness=1,
                         highlightcolor="#4a90d9", highlightbackground="#333355",
                         **kw)
            e.grid(row=row, column=1, pady=6)
            setattr(self, attr, e)
        tk.Label(form, text="Punt. mín.", bg="#1a1a2e", fg="#aaaacc",
                 font=("Consolas",9), width=10, anchor="e").grid(
                 row=2, column=0, padx=(0,8), pady=6)
        conf_frame = tk.Frame(form, bg="#1a1a2e")
        conf_frame.grid(row=2, column=1, pady=6, sticky="w")
        session = _load_session()
        saved_conf = session.get("min_conf", 0.0)
        self._conf_var = tk.StringVar(value=f"{saved_conf:.2f}")
        self._conf_spin = tk.Spinbox(
            conf_frame, from_=0.0, to=1.0, increment=0.05,
            format="%.2f", textvariable=self._conf_var,
            width=7, bg="#13131f", fg="white",
            buttonbackground="#2a2a40", insertbackground="white",
            font=("Consolas",10), relief="flat")
        self._conf_spin.pack(side="left")
        tk.Label(conf_frame, text=" (0.00 = sin filtro)",
                 bg="#1a1a2e", fg="#555577",
                 font=("Consolas",7)).pack(side="left", padx=(4,0))
        help_fr = tk.Frame(self, bg="#1a2030", bd=0)
        help_fr.pack(fill="x", padx=30, pady=(2,6))
        tk.Label(help_fr,
                 text="Método A: solo carga frames con conf ≥ umbral\n"
                      "Método B: solo carga frames donde TODAS las cajas ≥ umbral",
                 bg="#1a2030", fg="#445566",
                 font=("Consolas",7), justify="left").pack(
                 padx=8, pady=4, anchor="w")
        saved = session.get("username","")
        if saved:
            self._user_entry.insert(0, saved)
            self._pass_entry.focus_set()
        else:
            self._user_entry.focus_set()
        self._remember_var = tk.BooleanVar(value=bool(saved))
        tk.Checkbutton(self, text="Recordar usuario",
                       variable=self._remember_var,
                       bg="#1a1a2e", fg="#7ecfff", selectcolor="#0d0d1e",
                       activebackground="#1a1a2e", activeforeground="#7ecfff",
                       font=("Consolas",8)).pack(pady=(8,0))
        self._error_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._error_var,
                 bg="#1a1a2e", fg="#ff5555",
                 font=("Consolas",8)).pack(pady=(4,0))
        btn_fr = tk.Frame(self, bg="#1a1a2e")
        btn_fr.pack(pady=12)
        tk.Button(btn_fr, text="  Entrar  ", bg="#1565c0", fg="white",
                  relief="flat", font=("Consolas",10,"bold"), padx=16, pady=6,
                  command=self._try_login).pack(side="left", padx=8)
        tk.Button(btn_fr, text="  Cancelar  ", bg="#2a2a40", fg="#aaaacc",
                  relief="flat", font=("Consolas",9), padx=12, pady=6,
                  command=self._on_cancel).pack(side="left", padx=8)
        self.bind("<Return>", lambda e: self._try_login())

    def _try_login(self):
        username = self._user_entry.get().strip().lower()
        password = self._pass_entry.get().strip()
        cfg = USER_CONFIG.get(username)
        if not USER_CONFIG:
            self._error_var.set("❌  No hay usuarios configurados (falta users.json).")
            return
        if check_user_password(cfg, password, ALLOW_PLAINTEXT_USERS):
            try:
                min_conf = float(self._conf_var.get().replace(",", "."))
                min_conf = max(0.0, min(1.0, min_conf))
            except (ValueError, TypeError):
                min_conf = 0.0
            if self._remember_var.get():
                _save_session({"username": username, "min_conf": min_conf})
            else:
                _save_session({"username": "", "min_conf": min_conf})
            self.destroy()
            self._on_success(username, cfg, min_conf)
        else:
            self._attempts += 1
            self._pass_entry.delete(0, "end")
            msg = (f"❌  {self._attempts} intentos fallidos."
                   if self._attempts >= 3 else
                   "❌  Usuario o contraseña incorrectos.")
            self._error_var.set(msg)

    def _on_cancel(self):
        self.master.destroy()


# ─────────────────────── DownloadWindow (FTP) ──────────────────────────────

class DownloadWindow(tk.Toplevel):
    _INIT  = "init"
    _INST  = "inst"
    _FILE  = "file"
    _DONE  = "done"
    _ERROR = "error"
    _SCAN  = "scan"
    _PROGRESS_FNAME = ".ftp_progress.json"

    def __init__(self, master, user_cfg, on_done):
        super().__init__(master)
        self.title("Sincronizando imágenes (FTP)…")
        self.resizable(False, False)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.configure(bg="#1a1a2e")
        w, h = 580, 300
        self.geometry(f"{w}x{h}+{(self.winfo_screenwidth()-w)//2}+"
                      f"{(self.winfo_screenheight()-h)//2}")
        self._user_cfg = user_cfg
        self._on_done  = on_done
        self._q        = queue.Queue()

        tk.Label(self, text=f"⬇  Conectando a {FTP_HOST}:{FTP_PORT}",
                 bg="#1a1a2e", fg="#e0e0ff",
                 font=("Consolas",13,"bold")).pack(pady=(22,4))
        self._inst_var = tk.StringVar(value="Conectando al servidor FTP…")
        tk.Label(self, textvariable=self._inst_var,
                 bg="#1a1a2e", fg="#7ecfff",
                 font=("Consolas",9)).pack()

        tk.Label(self, text="Progreso total", bg="#1a1a2e", fg="#888899",
                 font=("Consolas",8)).pack(pady=(14,2))
        self._pb = ttk.Progressbar(self, length=500, mode="determinate")
        self._pb.pack()

        self._file_var  = tk.StringVar(value="")
        self._count_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._file_var,
                 bg="#1a1a2e", fg="#556677",
                 font=("Consolas",8)).pack(pady=(6,0))
        tk.Label(self, textvariable=self._count_var,
                 bg="#1a1a2e", fg="#7ecfff",
                 font=("Consolas",9,"bold")).pack(pady=(4,0))

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor="#2a2a40",
                        background="#4a90d9", bordercolor="#1a1a2e")

        self._limits = None
        threading.Thread(target=self._prescan, daemon=True).start()
        self._poll()

    def _prescan(self):
        cfg         = self._user_cfg
        remote_base = cfg["ftp_remote_dir"].rstrip("/")
        installations = cfg["installations"]
        try:
            ftp = _make_ftp(cfg)
        except Exception as e:
            self._q.put((self._ERROR,
                         f"❌  No se pudo conectar al servidor FTP\n"
                         f"{FTP_HOST}:{FTP_PORT}\n{e}"))
            return
        counts = {}
        for inst in installations:
            remote_dir = f"{remote_base}/{inst}/frame_original"
            jpgs = _ftp_list_jpgs(ftp, remote_dir)
            counts[inst] = len(jpgs)
        try: ftp.quit()
        except Exception: pass
        self._q.put((self._SCAN, counts))

    def _show_scan_dialog(self, counts):
        total = sum(counts.values())
        if total == 0:
            self._limits = {inst: 0 for inst in counts}
            self._inst_var.set("⚠  No hay fotos nuevas en el servidor.")
            threading.Thread(target=self._worker, daemon=True).start()
            return

        dlg = tk.Toplevel(self)
        dlg.title("Fotos nuevas encontradas")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.configure(bg="#1a1a2e")
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)
        w_dlg = 480
        h_dlg = 160 + len(counts) * 54
        dlg.geometry(f"{w_dlg}x{h_dlg}+"
                     f"{(dlg.winfo_screenwidth()-w_dlg)//2}+"
                     f"{(dlg.winfo_screenheight()-h_dlg)//2}")

        tk.Label(dlg, text="📥  Fotos disponibles en el servidor",
                 bg="#1a1a2e", fg="#7ecfff",
                 font=("Consolas", 11, "bold")).pack(pady=(18, 4))
        tk.Label(dlg,
                 text="Indica cuántas descargar por instalación.\n"
                      "Las que no descargues quedan en el servidor para la próxima revisión.",
                 bg="#1a1a2e", fg="#aaaacc",
                 font=("Consolas", 8), justify="center").pack(pady=(0, 10))

        spins = {}
        for inst, cnt in counts.items():
            row = tk.Frame(dlg, bg="#1a1a2e")
            row.pack(fill="x", padx=24, pady=4)
            color = "#7ecfff" if cnt > 0 else "#444466"
            tk.Label(row, text=f"📁  {inst}", bg="#1a1a2e", fg=color,
                     font=("Consolas", 10, "bold"), width=16,
                     anchor="w").pack(side="left")
            tk.Label(row, text=f"{cnt} fotos nuevas", bg="#1a1a2e",
                     fg="#aaaacc", font=("Consolas", 9),
                     width=18, anchor="w").pack(side="left")
            var = tk.IntVar(value=cnt)
            sp = tk.Spinbox(row, from_=0, to=cnt, textvariable=var,
                            width=6, bg="#13131f", fg="white",
                            buttonbackground="#2a2a40",
                            insertbackground="white",
                            font=("Consolas", 10), relief="flat",
                            state="normal" if cnt > 0 else "disabled")
            sp.pack(side="left", padx=8)
            if cnt > 0:
                tk.Button(row, text="Todas", bg="#1565c0", fg="white",
                          relief="flat", font=("Consolas", 8), padx=6, pady=2,
                          command=lambda v=var, n=cnt: v.set(n)
                          ).pack(side="left", padx=2)
                tk.Button(row, text="Ninguna", bg="#2a2a40", fg="#aaaacc",
                          relief="flat", font=("Consolas", 8), padx=6, pady=2,
                          command=lambda v=var: v.set(0)
                          ).pack(side="left", padx=2)
            spins[inst] = var

        btn_fr = tk.Frame(dlg, bg="#1a1a2e")
        btn_fr.pack(pady=14)

        def _start():
            self._limits = {inst: var.get() for inst, var in spins.items()}
            dlg.destroy()
            total_dl = sum(self._limits.values())
            if total_dl == 0:
                self._inst_var.set("ℹ  No se descargó ninguna foto.")
                self._q.put((self._DONE, []))
                return
            self._pb.config(maximum=total_dl, value=0)
            self._inst_var.set(f"Descargando {total_dl} foto(s)…")
            threading.Thread(target=self._worker, daemon=True).start()

        tk.Button(btn_fr, text="  Descargar  ", bg="#c84b00", fg="white",
                  relief="flat", font=("Consolas", 10, "bold"),
                  padx=18, pady=7, command=_start).pack(side="left", padx=8)
        tk.Button(btn_fr, text="  Cancelar  ", bg="#2a2a40", fg="#aaaacc",
                  relief="flat", font=("Consolas", 9),
                  padx=12, pady=7,
                  command=lambda: (dlg.destroy(),
                                   self._q.put((self._DONE, [])))
                  ).pack(side="left", padx=8)

    def _poll(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == self._INIT:
                    self._pb.config(maximum=max(1,payload), value=0)
                elif kind == self._INST:
                    nombre, n = payload
                    self._inst_var.set(f"📁  {nombre}  ({n} archivos)")
                elif kind == self._FILE:
                    done, total, fname, resumed = payload
                    self._pb.config(value=done)
                    pct = round(done/max(1,total)*100)
                    prefix = "↩" if resumed else "⬇"
                    self._file_var.set(f"{prefix} {fname}")
                    self._count_var.set(f"{done} / {total} — {pct}%")
                elif kind == self._SCAN:
                    self.after(0, lambda p=payload: self._show_scan_dialog(p))
                elif kind == self._ERROR:
                    self._inst_var.set(payload)
                    self.after(3000, lambda: (self.destroy(), self._on_done([])))
                    return
                elif kind == self._DONE:
                    self._inst_var.set("✔  Sincronización FTP completada")
                    self.after(800, lambda p=payload:
                               (self.destroy(), self._on_done(p)))
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _prog_path(self, local_root):
        return Path(local_root) / self._PROGRESS_FNAME

    def _load_progress(self, local_root):
        try:
            return _json_mod.loads(
                self._prog_path(local_root).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_progress(self, local_root, prog):
        try:
            self._prog_path(local_root).write_text(
                _json_mod.dumps(prog, indent=2, ensure_ascii=False),
                encoding="utf-8")
        except Exception:
            pass

    def _clear_progress(self, local_root):
        try:
            self._prog_path(local_root).unlink(missing_ok=True)
        except Exception:
            pass

    def _worker(self):
        cfg          = self._user_cfg
        local_root   = Path(cfg["local_dir"])
        remote_base  = cfg["ftp_remote_dir"].rstrip("/")
        installations = cfg["installations"]
        limits       = self._limits or {}

        progress  = self._load_progress(local_root)
        resumed_ok = [k for k,v in progress.items() if v == "ok"]

        try:
            ftp = _make_ftp(cfg)
        except Exception as e:
            self._q.put((self._ERROR,
                          f"❌  No se pudo conectar al servidor FTP\n"
                          f"{FTP_HOST}:{FTP_PORT}\n{e}"))
            return

        try:
            inst_files = {}
            total = 0
            for inst in installations:
                remote_dir = f"{remote_base}/{inst}/frame_original"
                all_jpgs   = _ftp_list_jpgs(ftp, remote_dir)
                limit      = limits.get(inst, len(all_jpgs))
                selected   = all_jpgs[:limit]
                inst_files[inst] = selected
                total += len(selected)

            if total == 0 and not resumed_ok:
                self._q.put((self._ERROR,
                              "⚠  No hay imágenes para descargar."))
                return

            if resumed_ok:
                self._q.put((self._INST,
                             (f"↩ Reanudando — {len(resumed_ok)} ya descargados",
                              len(resumed_ok))))

            self._q.put((self._INIT, total))

            all_pairs  = []
            done_count = 0
            errors     = []

            for inst in installations:
                remote_paths = inst_files.get(inst, [])
                if not remote_paths:
                    continue

                self._q.put((self._INST, (f"Instalación: {inst}",
                                           len(remote_paths))))
                dst_dir = local_root / inst
                dst_dir.mkdir(parents=True, exist_ok=True)

                for remote_jpg in remote_paths:
                    fname      = Path(remote_jpg).name
                    remote_txt = remote_jpg[:-4] + ".txt"
                    remote_boxed = f"{remote_base}/{inst}/{fname}"
                    jpg_dst    = dst_dir / fname
                    txt_dst    = dst_dir / (Path(fname).stem + ".txt")
                    key        = fname

                    if progress.get(key) == "ok" and jpg_dst.exists():
                        _ftp_delete(ftp, remote_jpg)
                        _ftp_delete(ftp, remote_txt)
                        _ftp_delete(ftp, remote_boxed)
                        all_pairs.append((str(jpg_dst),
                                          str(txt_dst) if txt_dst.exists() else None))
                        done_count += 1
                        self._q.put((self._FILE,
                                     (done_count, total, fname, True)))
                        continue

                    jpg_ok = _ftp_download(ftp, remote_jpg, jpg_dst)
                    if not jpg_ok:
                        errors.append(fname)
                        progress[key] = "error"
                        self._save_progress(local_root, progress)
                        done_count += 1
                        self._q.put((self._FILE,
                                     (done_count, total, f"⚠ {fname}", False)))
                        continue

                    _ftp_download(ftp, remote_txt, txt_dst)
                    _ftp_delete(ftp, remote_jpg)
                    _ftp_delete(ftp, remote_txt)
                    _ftp_delete(ftp, remote_boxed)

                    progress[key] = "ok"
                    self._save_progress(local_root, progress)

                    all_pairs.append((str(jpg_dst),
                                      str(txt_dst) if txt_dst.exists() else None))
                    done_count += 1
                    self._q.put((self._FILE, (done_count, total, fname, False)))

            ftp.quit()

            if not errors:
                self._clear_progress(local_root)
                self._q.put((self._DONE, all_pairs))
            else:
                err_list = "\n".join(errors[:5]) + ("\n…" if len(errors)>5 else "")
                self._q.put((self._ERROR,
                             f"⚠  {len(errors)} fichero(s) fallaron y se reanudarán.\n"
                             f"{err_list}"))

        except Exception as e:
            try: ftp.quit()
            except Exception: pass
            self._q.put((self._ERROR, f"❌  Error inesperado en FTP: {e}"))


# ─────────────────────── CavRelabelWindow ──────────────────────────────────

class CavRelabelWindow(tk.Toplevel):
    """
    Origen alternativo al FTP: en vez de descargar lo que haya en el
    servidor, consulta cav.v_pendiente_reetiquetado (ver
    db/patch_reetiquetado.sql) para saber exactamente qué frames
    necesitan que se corrija la clase que detectó YOLO, y los descarga
    de S3 con la caja inicial ya reconstruida a partir del log.txt del
    tour. Produce el mismo formato (jpg, txt) que DownloadWindow, así
    que el resto del programa no necesita saber de dónde vino cada par.
    """
    def __init__(self, master, user_cfg, on_done):
        super().__init__(master)
        self.title("Consultando pendientes de reetiquetado (cav)…")
        self.resizable(False, False)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.configure(bg="#1a1a2e")
        w, h = 580, 220
        self.geometry(f"{w}x{h}+{(self.winfo_screenwidth()-w)//2}+"
                      f"{(self.winfo_screenheight()-h)//2}")
        self._user_cfg = user_cfg
        self._on_done  = on_done
        self._q        = queue.Queue()

        tk.Label(self, text="🗄  Reetiquetado pendiente (cav)",
                 bg="#1a1a2e", fg="#e0e0ff",
                 font=("Consolas",13,"bold")).pack(pady=(22,4))
        self._inst_var = tk.StringVar(value="Consultando base de datos…")
        tk.Label(self, textvariable=self._inst_var,
                 bg="#1a1a2e", fg="#7ecfff",
                 font=("Consolas",9)).pack()

        tk.Label(self, text="Progreso", bg="#1a1a2e", fg="#888899",
                 font=("Consolas",8)).pack(pady=(14,2))
        self._pb = ttk.Progressbar(self, length=500, mode="determinate")
        self._pb.pack()

        self._file_var  = tk.StringVar(value="")
        self._count_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self._file_var,
                 bg="#1a1a2e", fg="#556677",
                 font=("Consolas",8)).pack(pady=(6,0))
        tk.Label(self, textvariable=self._count_var,
                 bg="#1a1a2e", fg="#7ecfff",
                 font=("Consolas",9,"bold")).pack(pady=(4,0))

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor="#2a2a40",
                        background="#c84b00", bordercolor="#1a1a2e")

        threading.Thread(target=self._worker, daemon=True).start()
        self._poll()

    def _poll(self):
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "init":
                    self._pb.config(maximum=max(1,payload), value=0)
                elif kind == "file":
                    done, total, fname = payload
                    self._pb.config(value=done)
                    pct = round(done/max(1,total)*100)
                    self._file_var.set(f"⬇ {fname}")
                    self._count_var.set(f"{done} / {total} — {pct}%")
                elif kind == "error":
                    self._inst_var.set(payload)
                    self.after(3000, lambda: (self.destroy(), self._on_done([], {})))
                    return
                elif kind == "done":
                    pairs, meta = payload
                    self._inst_var.set(f"✔  {len(pairs)} caso(s) listos para revisar")
                    self.after(700, lambda: (self.destroy(), self._on_done(pairs, meta)))
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _worker(self):
        cfg = self._user_cfg
        if not CAV_RELABEL_AVAILABLE:
            self._q.put(("error",
                         "❌  Falta el módulo cav_relabel.py (o pymysql/boto3).\n"
                         "Instala con: pip install pymysql boto3"))
            return
        if not cfg.get("cav_host"):
            self._q.put(("error",
                         "❌  No hay credenciales de cav configuradas\n"
                         "(CAV_DB_HOST / CAV_DB_USER / CAV_DB_PASS)."))
            return

        try:
            casos = cav_relabel.get_pending_relabel(cfg, cfg["installations"])
        except Exception as e:
            self._q.put(("error", f"❌  No se pudo consultar cav:\n{e}"))
            return

        if not casos:
            self._q.put(("error", "ℹ  No hay casos pendientes de reetiquetado."))
            return

        local_root = Path(cfg["local_dir"]) / "_reetiquetado_cav"
        local_root.mkdir(parents=True, exist_ok=True)

        self._q.put(("init", len(casos)))
        pairs = []
        # str(jpg_path) -> (id_ai, sector, clase_yolo, clase_sugerida,
        #                    motivo_yolo, origen_sugerencia, clase_s3)
        # clase_s3 = la clase que YOLO puso originalmente en la caja
        # (ground truth de tour_json) — la etiqueta que hay que corregir.
        meta = {}
        for i, caso in enumerate(casos, 1):
            inst_dir = local_root / caso.instalacion
            inst_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{caso.id_ai}_{caso.sector}.jpg"
            dst_jpg = inst_dir / fname
            dst_txt = inst_dir / (Path(fname).stem + ".txt")

            self._q.put(("file", (i, len(casos), fname)))
            try:
                ok = cav_relabel.descargar_caso(cfg, caso, dst_jpg, dst_txt)
            except Exception as e:
                print(f"[CAV_RELABEL] id_ai={caso.id_ai}: {e}")
                ok = False
            if not ok:
                continue

            pairs.append((str(dst_jpg), str(dst_txt) if dst_txt.exists() else None))
            meta[str(dst_jpg)] = (caso.id_ai, caso.sector, caso.clase_yolo,
                                   caso.clase_sugerida, caso.motivo_yolo,
                                   caso.origen_sugerencia, caso.clase_s3)

        self._q.put(("done", (pairs, meta)))


# ─────────────────────── BBoxCanvas ────────────────────────────────────────

class BBoxCanvas(tk.Canvas):
    HANDLE_CURSORS = {
        "tl":"top_left_corner","tr":"top_right_corner",
        "bl":"bottom_left_corner","br":"bottom_right_corner",
        "tc":"top_side","bc":"bottom_side",
        "lc":"left_side","rc":"right_side",
    }
    HANDLE_R = 6

    def __init__(self, master, app, **kwargs):
        super().__init__(master, bg="#1e1e1e", highlightthickness=0, **kwargs)
        self.app = app
        self._pil_img = None; self._tk_img = None
        self._scale = 1.0; self._off_x = 0; self._off_y = 0
        self.boxes = []; self._sel = None; self._hover = None
        self._drag = None; self._tmp_rect = None
        self._mode = "idle"; self._undo_stack = []
        for seq, cb in [
            ("<ButtonPress-1>",   self._on_press),
            ("<B1-Motion>",       self._on_drag),
            ("<ButtonRelease-1>", self._on_release),
            ("<Double-Button-1>", self._on_double_click),
            ("<ButtonPress-3>",   self._on_right_click),
            ("<ButtonPress-2>",   self._on_pan_start),
            ("<B2-Motion>",       self._on_pan_drag),
            ("<MouseWheel>",      self._on_wheel),
            ("<Button-4>",        self._on_wheel),
            ("<Button-5>",        self._on_wheel),
            ("<Motion>",          self._on_motion),
            ("<Configure>",       lambda e: self._fit()),
            ("<Delete>",          lambda e: self._key_delete()),
            ("<BackSpace>",       lambda e: self._key_delete()),
            ("<Escape>",          lambda e: self._deselect()),
            ("<Control-z>",       lambda e: self._undo()),
            ("<Control-Z>",       lambda e: self._undo()),
            ("<Return>",          lambda e: self.app._go_next()),
            ("<KP_Enter>",        lambda e: self.app._go_next()),
        ]:
            self.bind(seq, cb)
        for digit in range(10):
            self.bind(str(digit), lambda e, d=digit: self._assign_class_by_number(d))
        self.focus_set()

    def set_mode(self, mode):
        self._mode = mode
        if mode not in ("draw","edit"): self._sel = None
        self._render()

    def get_mode(self): return self._mode

    def load(self, pil_img, boxes):
        self._pil_img = pil_img
        self.boxes    = [dict(b) for b in boxes]
        self._sel = None; self._hover = None
        self._mode = "idle"; self._undo_stack = []
        self._fit()

    def delete_selected(self):
        if self._sel is not None:
            self._push_undo(); self.boxes.pop(self._sel); self._sel = None
            self._render(); self.app.set_dirty(True); self.app._update_box_panel()
            return True
        return False

    def zoom(self, factor):
        self._zoom_at(factor, self.winfo_width()/2, self.winfo_height()/2)

    def fit(self): self._fit()

    def _push_undo(self):
        self._undo_stack.append(([dict(b) for b in self.boxes], self._sel))
        if len(self._undo_stack) > 30: self._undo_stack.pop(0)

    def _undo(self):
        if not self._undo_stack: return
        boxes, sel = self._undo_stack.pop()
        self.boxes = boxes
        self._sel  = sel if sel is not None and sel < len(boxes) else None
        self._render(); self.app.set_dirty(True)
        self.app._update_box_panel(); self.app._status.set("↩ Deshacer")

    def _fit(self):
        if self._pil_img is None: return
        cw = self.winfo_width() or 800; ch = self.winfo_height() or 600
        iw, ih = self._pil_img.size
        scale = min(cw/iw, ch/ih, 1.0)
        self._scale = scale
        self._off_x = int((cw - iw*scale)/2)
        self._off_y = int((ch - ih*scale)/2)
        self._render()

    def _zoom_at(self, factor, cx, cy):
        ns = max(0.05, min(20., self._scale*factor))
        r  = ns / self._scale
        self._off_x = int(cx - (cx - self._off_x)*r)
        self._off_y = int(cy - (cy - self._off_y)*r)
        self._scale = ns; self._render()

    def _render(self):
        self.delete("all")
        if self._pil_img is None: return
        iw, ih = self._pil_img.size
        nw = max(1, int(iw*self._scale)); nh = max(1, int(ih*self._scale))
        resized = self._pil_img.resize((nw,nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)
        self.create_image(self._off_x, self._off_y, anchor="nw", image=self._tk_img)
        for i, box in enumerate(self.boxes): self._draw_box(i, box)

    def _canvas_coords(self, box):
        x1,y1,x2,y2 = box["xyxy"]; s = self._scale
        return (x1*s+self._off_x, y1*s+self._off_y,
                x2*s+self._off_x, y2*s+self._off_y)

    def _draw_box(self, idx, box):
        color  = color_for_cls(box["cls_id"])
        is_sel = (idx == self._sel)
        is_hov = (idx == self._hover and not is_sel)
        cx1,cy1,cx2,cy2 = self._canvas_coords(box)
        if is_hov:
            self.create_rectangle(cx1,cy1,cx2,cy2, fill=color,
                                   outline="", stipple="gray25")
        kw = dict(outline=color, width=3 if is_sel else 2)
        if is_sel: kw["dash"] = (6,3)
        self.create_rectangle(cx1,cy1,cx2,cy2,**kw)
        conf_str = f"{box['conf']:.2f}" if box.get("conf") is not None else "✏"
        label    = f" {box['name']}  {conf_str} "
        lw_px    = len(label)*7+4
        ty = cy1-18 if cy1 > 20 else cy1+2
        self.create_rectangle(cx1,ty,cx1+lw_px,ty+18, fill=color,outline="")
        self.create_text(cx1+5,ty+9, text=label, fill="white", anchor="w",
                         font=("Consolas",8,"bold"))
        if is_sel:
            mx,my = (cx1+cx2)/2,(cy1+cy2)/2
            handles = {"tl":(cx1,cy1),"tc":(mx,cy1),"tr":(cx2,cy1),
                       "lc":(cx1,my),"rc":(cx2,my),
                       "bl":(cx1,cy2),"bc":(mx,cy2),"br":(cx2,cy2)}
            r = self.HANDLE_R
            for hpos,(hx,hy) in handles.items():
                self.create_oval(hx-r,hy-r,hx+r,hy+r,
                                  fill="white",outline=color,width=2,
                                  tags=f"handle_{hpos}")
        self.create_text(cx2-4,cy2-4, text=str(idx+1), fill=color,
                         anchor="se", font=("Consolas",7))

    def _img_xy(self,cx,cy):
        return ((cx-self._off_x)/self._scale, (cy-self._off_y)/self._scale)

    def _hit_handle(self, cx, cy):
        if self._sel is None: return None
        bx1,by1,bx2,by2 = self._canvas_coords(self.boxes[self._sel])
        mx,my = (bx1+bx2)/2,(by1+by2)/2
        handles = {"tl":(bx1,by1),"tc":(mx,by1),"tr":(bx2,by1),
                   "lc":(bx1,my),"rc":(bx2,my),
                   "bl":(bx1,by2),"bc":(mx,by2),"br":(bx2,by2)}
        r = self.HANDLE_R+3
        for name,(hx,hy) in handles.items():
            if abs(cx-hx)<=r and abs(cy-hy)<=r:
                return (self._sel, name)
        return None

    def _hit_box(self, cx, cy):
        ix,iy = self._img_xy(cx,cy)
        best,best_area = None,float("inf")
        for i,box in enumerate(self.boxes):
            x1,y1,x2,y2 = box["xyxy"]
            if x1<=ix<=x2 and y1<=iy<=y2:
                area = (x2-x1)*(y2-y1)
                if area < best_area:
                    best_area,best = area,i
        return best

    def _update_cursor(self, cx, cy):
        hit_h = self._hit_handle(cx,cy)
        if hit_h:
            self.config(cursor=self.HANDLE_CURSORS.get(hit_h[1],"fleur")); return
        hit = self._hit_box(cx,cy)
        self.config(cursor="fleur" if hit==self._sel and hit is not None
                    else "hand2" if hit is not None else "crosshair")

    def _on_motion(self, event):
        prev = self._hover
        self._hover = self._hit_box(event.x, event.y)
        if self._hover != prev: self._render()
        self._update_cursor(event.x, event.y)

    def _on_press(self, event):
        self.focus_set()
        cx,cy = event.x, event.y
        hit_h = self._hit_handle(cx,cy)
        if hit_h:
            bidx,hpos = hit_h; self._push_undo()
            self._drag = {"mode":"resize","idx":bidx,"hpos":hpos,
                          "orig":list(self.boxes[bidx]["xyxy"]),"cx":cx,"cy":cy}
            return
        hit = self._hit_box(cx,cy)
        if hit is not None:
            if hit != self._sel:
                self._sel = hit; self._render(); self.app._update_box_panel()
            self._push_undo()
            self._drag = {"mode":"move","idx":hit,
                          "orig":list(self.boxes[hit]["xyxy"]),
                          "cx":cx,"cy":cy,"moved":False}
            return
        self._sel = None
        self._drag = {"mode":"draw","cx0":cx,"cy0":cy}
        self._render(); self.app._update_box_panel()

    def _ask_class_quick(self, cx, cy):
        dlg = QuickClassDialog(self, self.app._known_cls, cx, cy)
        self.wait_window(dlg)
        return dlg.result

    def _on_drag(self, event):
        if self._drag is None: return
        cx,cy = event.x, event.y
        dm = self._drag["mode"]
        if dm == "draw":
            if self._tmp_rect: self.delete(self._tmp_rect)
            x0,y0 = self._drag["cx0"], self._drag["cy0"]
            self._tmp_rect = self.create_rectangle(x0,y0,cx,cy,
                outline="#FFFF00",width=2,dash=(5,3))
            ix0,iy0 = self._img_xy(x0,y0); ix1,iy1 = self._img_xy(cx,cy)
            self.delete("size_hint")
            self.create_text(cx+6,cy+6,
                text=f"{int(abs(ix1-ix0))}×{int(abs(iy1-iy0))}",
                fill="#FFFF00",anchor="nw",font=("Consolas",8),tags="size_hint")
            return
        if dm == "move":
            dx=(cx-self._drag["cx"])/self._scale; dy=(cy-self._drag["cy"])/self._scale
            ox1,oy1,ox2,oy2 = self._drag["orig"]
            self.boxes[self._drag["idx"]]["xyxy"]=[ox1+dx,oy1+dy,ox2+dx,oy2+dy]
            self._drag["moved"]=True; self._render()
        if dm == "resize":
            dx=(cx-self._drag["cx"])/self._scale; dy=(cy-self._drag["cy"])/self._scale
            ox1,oy1,ox2,oy2 = self._drag["orig"]; hpos=self._drag["hpos"]
            nx1,ny1,nx2,ny2 = ox1,oy1,ox2,oy2
            if "l" in hpos: nx1=ox1+dx
            if "r" in hpos: nx2=ox2+dx
            if "t" in hpos: ny1=oy1+dy
            if "b" in hpos: ny2=oy2+dy
            if nx2-nx1>=MIN_BOX_PX and ny2-ny1>=MIN_BOX_PX:
                self.boxes[self._drag["idx"]]["xyxy"]=[nx1,ny1,nx2,ny2]
            self._render()

    def _on_release(self, event):
        if self._drag is None: return
        dm = self._drag["mode"]
        if dm == "draw":
            if self._tmp_rect: self.delete(self._tmp_rect); self._tmp_rect=None
            self.delete("size_hint")
            cx0=self._drag["cx0"]; cy0=self._drag["cy0"]
            cx1=min(cx0,event.x); cy1=min(cy0,event.y)
            cx2=max(cx0,event.x); cy2=max(cy0,event.y)
            self._drag=None
            if (cx2-cx1)>MIN_BOX_PX and (cy2-cy1)>MIN_BOX_PX:
                ix0,iy0=self._img_xy(cx1,cy1); ix1,iy1=self._img_xy(cx2,cy2)
                if self.app._known_cls:
                    mid_cx = (cx1+cx2)//2; mid_cy = (cy1+cy2)//2
                    chosen = self._ask_class_quick(mid_cx, mid_cy)
                else:
                    chosen = None
                if chosen is None and self.app._known_cls:
                    self._render(); self.app._sync_edit_button(); return
                if chosen is None:
                    new_box = self.app.ask_new_box(ix0,iy0,ix1,iy1)
                else:
                    new_box = {
                        "cls_id": chosen["cls_id"],
                        "name":   chosen["name"],
                        "conf":   None,
                        "xyxy":   [round(ix0,1),round(iy0,1),round(ix1,1),round(iy1,1)],
                    }
                    cid = new_box["cls_id"]
                    if not any(c[0]==cid for c in self.app._known_cls):
                        self.app._known_cls.append((cid, new_box["name"]))
                        self.app._known_cls.sort()
                if new_box:
                    self._push_undo(); self.boxes.append(new_box)
                    self._sel=len(self.boxes)-1
                    self.app.set_dirty(True); self.app._update_box_panel()
            self._mode="idle"
        elif dm in ("move","resize"):
            self._drag=None
            self.app.set_dirty(True); self.app._update_box_panel()
        self._drag=None; self._render(); self.app._sync_edit_button()

    def _on_double_click(self, event):
        hit = self._hit_box(event.x, event.y)
        if hit is None or not self.app._known_cls: return
        self._sel = hit
        dlg = ChangeClassDialog(self, self.boxes[hit], self.app._known_cls)
        self.wait_window(dlg)
        if dlg.result:
            self._push_undo()
            self.boxes[hit]["cls_id"] = dlg.result["cls_id"]
            self.boxes[hit]["name"]   = dlg.result["name"]
            self._render(); self.app.set_dirty(True); self.app._update_box_panel()

    def _on_right_click(self, event):
        hit = self._hit_box(event.x, event.y)
        if hit is None: return
        self._sel = hit; self._render()
        menu = tk.Menu(self, tearoff=0, bg="#1a1a2e", fg="#ccccdd",
                        activebackground="#4a90d9", activeforeground="white",
                        bd=0, relief="flat", font=("Consolas",9))
        name = self.boxes[hit].get("name","?")
        menu.add_command(label=f"#{hit+1}  {name}", state="disabled",
                         font=("Consolas",8,"bold"))
        menu.add_separator()
        menu.add_command(label="✏  Cambiar clase",
                          command=lambda: self._change_class_menu(hit))
        menu.add_command(label="🗑  Borrar caja",
                          command=lambda: self._delete_by_idx(hit))
        menu.tk_popup(event.x_root, event.y_root)

    def _change_class_menu(self, idx):
        if not self.app._known_cls: return
        dlg = ChangeClassDialog(self, self.boxes[idx], self.app._known_cls)
        self.wait_window(dlg)
        if dlg.result:
            self._push_undo()
            self.boxes[idx]["cls_id"] = dlg.result["cls_id"]
            self.boxes[idx]["name"]   = dlg.result["name"]
            self._render(); self.app.set_dirty(True); self.app._update_box_panel()

    def _delete_by_idx(self, idx):
        self._push_undo(); self.boxes.pop(idx); self._sel=None
        self._render(); self.app.set_dirty(True); self.app._update_box_panel()

    def _delete_selected_key(self):
        if self._sel is not None:
            name = self.boxes[self._sel].get("name","?")
            self.delete_selected()
            self.app._status.set(f"🗑 Caja '{name}' eliminada (Del)")

    def _key_delete(self):
        if self._sel is not None:
            self._delete_selected_key()
        else:
            self.app._discard_current_with_confirm()

    def _assign_class_by_number(self, digit):
        if self._sel is None:
            self.app._status.set("⌨ Selecciona primero una caja para asignar clase")
            return
        if not self.app._known_cls:
            self.app._status.set("⚠ No hay clases conocidas cargadas")
            return
        match = next((c for c in self.app._known_cls if c[0] == digit), None)
        if match is None:
            if digit < len(self.app._known_cls):
                match = self.app._known_cls[digit]
            else:
                self.app._status.set(f"⚠ No existe clase con id/índice {digit}")
                return
        self._push_undo()
        self.boxes[self._sel]["cls_id"] = match[0]
        self.boxes[self._sel]["name"]   = match[1]
        self._render()
        self.app.set_dirty(True)
        self.app._update_box_panel()
        self.app._status.set(f"⌨ Caja #{self._sel+1} → {match[0]}: {match[1]}")

    def _deselect(self):
        if self._sel is not None:
            self._sel=None; self._render(); self.app._update_box_panel()

    def _on_wheel(self, event):
        if self._pil_img is None: return
        factor = 1.15 if (event.num==4 or event.delta>0) else 1/1.15
        self._zoom_at(factor, event.x, event.y)

    def _on_pan_start(self, event):
        self._pan_start=(event.x,event.y,self._off_x,self._off_y)
        self.config(cursor="fleur")

    def _on_pan_drag(self, event):
        if not hasattr(self,"_pan_start"): return
        sx,sy,ox,oy = self._pan_start
        self._off_x=ox+(event.x-sx); self._off_y=oy+(event.y-sy)
        self._render()


# ─────────────────────── QuickClassDialog ──────────────────────────────────

class QuickClassDialog(tk.Toplevel):
    def __init__(self, master, known_classes, canvas_x=None, canvas_y=None):
        super().__init__(master)
        self.title("")
        self.resizable(False, False)
        self.grab_set()
        self.overrideredirect(True)
        self.configure(bg="#1a1a2e")
        self.result = None
        self._classes = known_classes
        self._selected = 0

        tk.Label(self, text="¿Qué clase quieres dibujar?",
                 bg="#1a1a2e", fg="#7ecfff",
                 font=("Consolas", 9, "bold"),
                 padx=12, pady=8).pack(fill="x")

        sep = tk.Frame(self, bg="#333355", height=1)
        sep.pack(fill="x")

        btn_fr = tk.Frame(self, bg="#1a1a2e", padx=8, pady=6)
        btn_fr.pack(fill="both", expand=True)

        self._btns = []
        for i, (cid, name) in enumerate(known_classes):
            shortcut = str(cid) if cid < 10 else "–"
            color    = COLORS[int(cid) % len(COLORS)]
            row = tk.Frame(btn_fr, bg="#1a1a2e")
            row.pack(fill="x", pady=2)
            tk.Label(row, text="█", fg=color, bg="#1a1a2e",
                     font=("Consolas", 11)).pack(side="left", padx=(0, 6))
            key_lbl = tk.Label(row, text=f"[{shortcut}]",
                               bg="#2a2a50", fg="#ffdd44",
                               font=("Consolas", 9, "bold"),
                               width=4, relief="flat", padx=4)
            key_lbl.pack(side="left", padx=(0, 6))
            btn = tk.Button(row, text=f"{cid}: {name}",
                            bg="#1e1e38", fg="#ccccdd",
                            font=("Consolas", 9),
                            relief="flat", anchor="w",
                            padx=10, pady=4,
                            cursor="hand2",
                            command=lambda idx=i: self._pick(idx))
            btn.pack(side="left", fill="x", expand=True)
            self._btns.append(btn)

        sep2 = tk.Frame(self, bg="#333355", height=1)
        sep2.pack(fill="x")
        tk.Label(self, text="Esc = cancelar",
                 bg="#1a1a2e", fg="#555577",
                 font=("Consolas", 7), pady=4).pack()

        for i, (cid, _) in enumerate(known_classes):
            if cid < 10:
                self.bind(str(cid), lambda e, idx=i: self._pick(idx))
        self.bind("<Escape>",  lambda e: self.destroy())
        self.bind("<Return>",  lambda e: self._pick(self._selected))
        self.bind("<Up>",      lambda e: self._move_sel(-1))
        self.bind("<Down>",    lambda e: self._move_sel(1))

        self._highlight(0)

        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        try:
            rx = master.winfo_rootx() + (canvas_x or 0)
            ry = master.winfo_rooty() + (canvas_y or 0)
        except Exception:
            rx, ry = self.winfo_screenwidth()//2, self.winfo_screenheight()//2
        sw = self.winfo_screenwidth(); sh = self.winfo_screenheight()
        x = min(rx + 12, sw - w - 8)
        y = min(ry + 12, sh - h - 8)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.focus_set()

    def _highlight(self, idx):
        for i, btn in enumerate(self._btns):
            btn.config(bg="#2a3a6a" if i == idx else "#1e1e38",
                       fg="white"   if i == idx else "#ccccdd")
        self._selected = idx

    def _move_sel(self, delta):
        new = (self._selected + delta) % max(1, len(self._btns))
        self._highlight(new)

    def _pick(self, idx):
        if 0 <= idx < len(self._classes):
            cid, name = self._classes[idx]
            self.result = {"cls_id": cid, "name": name}
        self.destroy()


# ─────────────────────── NewBoxDialog ──────────────────────────────────────

class NewBoxDialog(tk.Toplevel):
    def __init__(self, master, known_classes):
        super().__init__(master)
        self.title("Nueva caja – elige clase")
        self.resizable(False, False)
        self.grab_set()
        self.result = None
        tk.Label(self,text="Clase:").grid(row=0,column=0,padx=12,pady=10,sticky="e")
        self.cls_var = tk.StringVar()
        names = [f"{c[0]}: {c[1]}" for c in known_classes]
        self.combo = ttk.Combobox(self,textvariable=self.cls_var,
                                   values=names,width=26,state="readonly")
        if names: self.combo.current(0)
        self.combo.grid(row=0,column=1,padx=12,pady=10)
        ttk.Separator(self,orient="horizontal").grid(row=1,column=0,columnspan=2,
                                                      sticky="ew",padx=12)
        tk.Label(self,text="— o nueva clase —",fg="#888888").grid(
            row=2,column=0,columnspan=2,pady=(6,2))
        tk.Label(self,text="cls_id:").grid(row=3,column=0,padx=12,sticky="e")
        self.id_entry = tk.Entry(self,width=8)
        self.id_entry.grid(row=3,column=1,padx=12,pady=4,sticky="w")
        tk.Label(self,text="Nombre:").grid(row=4,column=0,padx=12,sticky="e")
        self.name_entry = tk.Entry(self,width=22)
        self.name_entry.grid(row=4,column=1,padx=12,pady=4)
        fr = tk.Frame(self); fr.grid(row=5,column=0,columnspan=2,pady=12)
        tk.Button(fr,text="  Aceptar  ",command=self._ok).pack(side="left",padx=8)
        tk.Button(fr,text="  Cancelar  ",command=self.destroy).pack(side="left",padx=8)
        self.wait_window()

    def _ok(self):
        name_new = self.name_entry.get().strip()
        id_new   = self.id_entry.get().strip()
        if name_new and id_new.isdigit():
            self.result={"cls_id":int(id_new),"name":name_new}
        elif self.cls_var.get():
            try:
                parts=self.cls_var.get().split(":",1)
                self.result={"cls_id":int(parts[0]),"name":parts[1].strip()}
            except Exception:
                messagebox.showerror("Error","Selección inválida"); return
        else:
            messagebox.showerror("Error","Elige una clase o introduce una nueva"); return
        self.destroy()


# ─────────────────────── ChangeClassDialog ─────────────────────────────────

class ChangeClassDialog(tk.Toplevel):
    def __init__(self, master, box, known_classes):
        super().__init__(master)
        self.title("Cambiar clase")
        self.resizable(False, False)
        self.grab_set()
        self.configure(bg="#1a1a2e")
        self.result = None
        w,h = 320,320
        self.geometry(f"{w}x{h}+{(self.winfo_screenwidth()-w)//2}+"
                      f"{(self.winfo_screenheight()-h)//2}")
        tk.Label(self,text="Selecciona la nueva clase",
                 bg="#1a1a2e",fg="#7ecfff",
                 font=("Consolas",10,"bold")).pack(pady=(16,8))
        fr = tk.Frame(self,bg="#1a1a2e")
        fr.pack(fill="both",expand=True,padx=12,pady=(0,8))
        sb = ttk.Scrollbar(fr); sb.pack(side="right",fill="y")
        self._lb = tk.Listbox(fr,bg="#13131f",fg="#ccccdd",
                               selectbackground="#4a90d9",selectforeground="white",
                               bd=0,highlightthickness=0,font=("Consolas",10),
                               yscrollcommand=sb.set)
        self._lb.pack(side="left",fill="both",expand=True)
        sb.config(command=self._lb.yview)
        cur_id = box.get("cls_id",-1)
        self._classes = known_classes
        sel_idx = 0
        for i,(cid,name) in enumerate(known_classes):
            self._lb.insert("end",f"  {cid:>3}  {name}")
            self._lb.itemconfig(i,foreground=color_for_cls(cid))
            if cid==cur_id: sel_idx=i
        self._lb.selection_set(sel_idx); self._lb.see(sel_idx)
        self._lb.bind("<Double-Button-1>",lambda e: self._ok())
        self._lb.bind("<Return>",lambda e: self._ok())
        btn_fr = tk.Frame(self,bg="#1a1a2e"); btn_fr.pack(pady=8)
        tk.Button(btn_fr,text="  Aceptar  ",bg="#1565c0",fg="white",
                  relief="flat",font=("Consolas",9,"bold"),padx=12,pady=6,
                  command=self._ok).pack(side="left",padx=6)
        tk.Button(btn_fr,text="  Cancelar  ",bg="#2a2a40",fg="#aaaacc",
                  relief="flat",font=("Consolas",9),padx=10,pady=6,
                  command=self.destroy).pack(side="left",padx=6)
        self.bind("<Escape>",lambda e: self.destroy())

    def _ok(self):
        sel = self._lb.curselection()
        if not sel: return
        cid,name = self._classes[sel[0]]
        self.result={"cls_id":cid,"name":name}; self.destroy()


# ─────────────────────── BoxReviewer ────────────────────────────────────────

class BoxReviewer(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("YOLO Box Reviewer – Editor de Bounding Boxes")
        _sess = _load_session()
        self.geometry(_sess.get("geometry","1440x880"))
        self.configure(bg="#2b2b2b")
        self.bind("<Configure>", self._on_window_configure)
        self.option_add("*Font","Consolas 9")

        self._dirty=False; self._img_list=[]; self._cur_idx=-1
        self._cav_relabel_meta = {}
        self._known_cls=[]; self._user_cfg=None; self._username=""
        self._min_conf=0.0
        self._filter_stats_var = tk.StringVar(value="")
        self._review_stats = {}
        self._review_stats_var = tk.StringVar(value="")
        self._original_boxes = []
        self._undo_snapshot = None

        self._video_url=None; self._video_fn=None; self._video_saved_at=None
        self._vtk_img=None; self._vtmp_path=None
        self._vlc_player=None; self._vproc=None
        self._vcached_pil=[]; self._vcached_fps=25.0
        self._vplaying=False; self._vpause=False
        self._resize_timer=None; self._last_panel_size=(0,0)

        self._build_ui()
        self.after(200, self._auto_init)

    # ── Login ──────────────────────────────────────────────────────────────

    def _on_window_configure(self, event):
        if event.widget is self:
            _save_session({"geometry": self.geometry()})

    def _auto_init(self):
        LoginDialog(self, on_success=self._on_login)

    def _on_login(self, username, cfg, min_conf=0.0):
        self._username  = username
        self._user_cfg  = cfg
        self._min_conf  = min_conf
        self._load_review_stats()
        self._user_label.config(text=f"👤  {username}")
        self._inst_label.config(
            text=f"Instalaciones: {', '.join(cfg['installations'])}")
        label_conf = (f"  |  punt. ≥ {min_conf:.2f}" if min_conf > 0 else "")
        self._status.set(f"⏳ Escaneando imágenes locales…{label_conf}")
        self.update_idletasks()
        threading.Thread(target=self._async_local_scan, daemon=True).start()

    def _async_local_scan(self):
        pairs = self._scan_local_pending(self._user_cfg)
        if pairs:
            self.after(0, lambda: self._on_download_done(pairs))
        else:
            self.after(0, lambda: DownloadWindow(
                self, self._user_cfg, on_done=self._on_download_done))

    def _scan_local_pending(self, cfg):
        local_root = Path(cfg["local_dir"]); pairs=[]
        for inst in cfg["installations"]:
            inst_dir = local_root / inst
            if not inst_dir.exists(): continue
            for jpg in sorted(inst_dir.glob("*.jpg")):
                txt = jpg.with_suffix(".txt")
                pairs.append((str(jpg), str(txt) if txt.exists() else None))
        return pairs

    def _on_download_done(self, pairs, meta=None):
        is_cav = bool(meta)
        if is_cav:
            # El reetiquetado CAV se suma a la cola que ya hubiera cargada
            # (FTP u otra tanda CAV anterior) en vez de sustituirla. Y como
            # no lleva "conf" (su .txt es YOLO plano reconstruido del
            # log.txt de S3, no el JSON boxes/conf de la descarga normal),
            # nunca pasa por el filtro de punt. — no tiene sentido filtrar
            # por confianza algo que no la tiene.
            if not pairs:
                messagebox.showinfo("Sin pendientes de reetiquetado (cav)",
                                     "No hay casos nuevos que descargar.")
                return
            self._append_pairs(pairs, meta)
            return
        # Carga normal (login inicial o "Re-sincronizar"): esta sí sustituye
        # la cola entera, es una tanda fresca desde cero.
        self._cav_relabel_meta = {}
        if not pairs:
            self._status.set("⚠  No se encontraron imágenes."); return
        min_conf = getattr(self, '_min_conf', 0.0)
        if min_conf > 0.0:
            n = len(pairs)
            self._status.set(
                f"⏳ Filtrando {n} imagen(es) con punt. ≥ {min_conf:.2f}…")
            self.update_idletasks()
            threading.Thread(
                target=self._filter_and_show,
                args=(pairs, min_conf),
                daemon=True
            ).start()
        else:
            self._show_pairs(pairs, 0.0)

    def _open_cav_relabel(self):
        if self._user_cfg is None:
            return
        if self._dirty:
            if not messagebox.askyesno("Cambios sin guardar",
                    "¿Cargar los pendientes de reetiquetado de todos modos?"):
                return
        CavRelabelWindow(self, self._user_cfg, on_done=self._on_download_done)

    def _filter_and_show(self, pairs, min_conf):
        total    = len(pairs)
        passed   = []
        rejected_pairs = []
        UPDATE_EVERY = 50

        for i, (jpg_path, txt_path) in enumerate(pairs, 1):
            if txt_path:
                boxes = load_json_boxes(txt_path)
                if _passes_conf_filter(boxes, min_conf):
                    passed.append((jpg_path, txt_path))
                else:
                    rejected_pairs.append((jpg_path, txt_path))
            else:
                if min_conf <= 0.0:
                    passed.append((jpg_path, txt_path))
                else:
                    rejected_pairs.append((jpg_path, None))

            if i % UPDATE_EVERY == 0 or i == total:
                _p = len(passed); _r = len(rejected_pairs); _i = i
                self.after(0, lambda p=_p, r=_r, idx=_i:
                    self._status.set(
                        f"⏳ Filtrando… {idx}/{total}  —  "
                        f"✔ {p} pasan  |  ✘ {r} descartadas  "
                        f"(punt. ≥ {min_conf:.2f})"))

        deleted = 0
        for jpg_path, txt_path in rejected_pairs:
            try:
                Path(jpg_path).unlink(missing_ok=True)
                deleted += 1
            except Exception:
                pass
            if txt_path:
                try: Path(txt_path).unlink(missing_ok=True)
                except Exception: pass

        self.after(0, lambda: self._show_pairs(passed, min_conf, len(rejected_pairs), deleted))

    def _insert_listbox_entry(self, jpg_path, at_end=True, index=None):
        """Inserta una entrada en la lista de imágenes, resaltando en
        naranja (mismo tono que el botón 'Reetiquetado (cav)') las que
        vienen del reetiquetado CAV en vez de la descarga FTP normal."""
        cav_meta = getattr(self, "_cav_relabel_meta", {}) or {}
        inst = Path(jpg_path).parent.name
        is_cav = str(jpg_path) in cav_meta
        label = f"🗄 CAV [{inst}] {Path(jpg_path).name}" if is_cav \
                else f"[{inst}] {Path(jpg_path).name}"
        pos = "end" if at_end else index
        self._img_lb.insert(pos, label)
        idx = self._img_lb.size() - 1 if at_end else index
        if is_cav:
            self._img_lb.itemconfig(idx, fg="#ffb347", bg="#2a1c0d")

    def _show_pairs(self, pairs, min_conf=0.0, rejected=0, deleted=0):
        if not pairs:
            if min_conf > 0.0:
                msg = (f"Ninguna imagen supera punt. ≥ {min_conf:.2f}.\n"
                       f"Se descartaron y eliminaron del disco {deleted} archivo(s).\n"
                       "Prueba con un umbral más bajo en el login.")
                self._status.set(
                    f"⚠  0 imágenes pasan punt. ≥ {min_conf:.2f}  "
                    f"({deleted} eliminadas del disco)")
            else:
                msg = "No hay imágenes disponibles."
                self._status.set("⚠  No se encontraron imágenes.")
            messagebox.showinfo("Sin imágenes", msg)
            return
        self._img_list = pairs
        self._img_lb.delete(0, "end")
        for jpg_path, txt_path in pairs:
            self._insert_listbox_entry(jpg_path)
        total_ok = len(pairs)
        self._counter_var.set(f"0 / {total_ok}")
        if min_conf > 0.0:
            self._filter_stats_var.set(
                f"Filtro ≥ {min_conf:.2f}:  "
                f"✔ {total_ok} cargadas  |  "
                f"✘ {rejected} eliminadas del disco")
            self._filter_stats_label.config(bg="#1a2030")
        else:
            self._filter_stats_var.set("")
        self._status.set(f"✔  {total_ok} imágenes listas")
        self._load_image(0)
        threading.Thread(target=self._build_known_cls, daemon=True).start()

    def _append_pairs(self, pairs, meta):
        """Añade casos de reetiquetado CAV a la cola actual sin descartar
        lo que ya estuviera cargado por FTP (o de una tanda CAV anterior).
        Si no había nada cargado todavía, se comporta como una carga inicial."""
        self._cav_relabel_meta.update(meta or {})
        if not self._img_list:
            self._show_pairs(pairs, 0.0)
            return
        self._img_list.extend(pairs)
        for jpg_path, txt_path in pairs:
            self._insert_listbox_entry(jpg_path)
        total = len(self._img_list)
        cur = self._cur_idx + 1 if self._cur_idx >= 0 else 0
        self._counter_var.set(f"{cur} / {total}")
        self._status.set(
            f"✔  +{len(pairs)} caso(s) de reetiquetado (cav) añadidos "
            f"— {total} en total")
        threading.Thread(target=self._build_known_cls, daemon=True).start()

    def _build_known_cls(self):
        cls_seen = {}
        for jpg_path, txt_path in list(self._img_list):
            if txt_path:
                for b in load_json_boxes(txt_path):
                    cls_seen[b.get("cls_id", 0)] = b.get("name", "unknown")
        known = sorted(cls_seen.items())
        self.after(0, lambda: setattr(self, '_known_cls', known))

    # ── Conteo de revisión (falsos positivos / detecciones) ────────────────
    # ── Umbrales de negocio del conteo de revisión (centralizados) ──────────
    IOU_MATCH_MIN = 0.70      # solape minimo para considerar que es el mismo objeto
    IOU_INTACTA_MIN = 0.70    # mismo umbral: ajustar tamano no cambia el KPI
    GOOD_DETECTION_CLASSES = {"barco", "barco2", "barco3", "persona"}
    CONF_BANDS = [
        (0.0, 0.3, "baja"),
        (0.3, 0.6, "media"),
        (0.6, 0.85, "alta"),
        (0.85, 1.01, "muy_alta"),
    ]

    def _review_stats_path(self):
        cfg = self._user_cfg or {}
        local_root = Path(cfg.get("local_dir", "."))
        username = self._username or "sin_usuario"
        return local_root / f"conteo_revision_{username}.json"

    def _empty_review_stats(self):
        return {
            "username": self._username,
            "totals": {
                "confirmada_intacta": 0,      # sigue igual, misma clase, casi sin mover
                "confirmada_ajustada": 0,     # sigue pero movida/redimensionada
                "reclasificada": 0,           # cambiada a otra clase objetivo (barco->persona)
                "falso_positivo": 0,          # eliminada o reclasificada a no-objetivo
                "no_detectado": 0,             # clase de interes que el modelo no detecto
            },
            "totals_por_instalacion": {},     # {"site_a": {categoria: n, ...}, ...}
            "totals_por_banda_confianza": {}, # {"alta": {categoria: n, ...}, ...}
            "confusion_clases": {},           # {"barco2->persona": 3, ...}
            "events": [],
        }

    def _load_review_stats(self):
        path = self._review_stats_path()
        migrated = False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Esquemas anteriores (por-imagen, sin desglose por caja): no se
            # pueden reconstruir con esta granularidad porque nunca se guardó.
            # Se congela el fichero viejo y se empieza de cero con el nuevo.
            legacy_keys = {"detecciones_buenas", "falsos_positivos", "frames_descartados"}
            if legacy_keys & set(data.get("totals", {}).keys()):
                legacy_path = path.with_name(
                    f"{path.stem}_v1_hasta_{time.strftime('%Y-%m-%d')}{path.suffix}")
                if not legacy_path.exists():
                    path.rename(legacy_path)
                data = self._empty_review_stats()
                migrated = True
            self._review_stats = data
        except Exception:
            self._review_stats = self._empty_review_stats()
        self._review_stats.setdefault("username", self._username)
        self._review_stats.setdefault("totals", self._empty_review_stats()["totals"])
        totals = self._review_stats.setdefault("totals", {})
        if "detectada_solo_manual" in totals:
            totals["no_detectado"] = totals.get("no_detectado", 0) + totals.pop("detectada_solo_manual", 0)
        self._review_stats.setdefault("totals_por_instalacion", {})
        self._review_stats.setdefault("totals_por_banda_confianza", {})
        self._review_stats.setdefault("confusion_clases", {})
        self._review_stats.setdefault("events", [])
        for grouped in (
                self._review_stats.get("totals_por_instalacion", {}),
                self._review_stats.get("totals_por_banda_confianza", {})):
            for counts in grouped.values():
                if "detectada_solo_manual" in counts:
                    counts["no_detectado"] = counts.get("no_detectado", 0) + counts.pop("detectada_solo_manual", 0)
        for ev in self._review_stats.get("events", []):
            for box in ev.get("boxes", []):
                if box.get("outcome") == "detectada_solo_manual":
                    box["outcome"] = "no_detectado"
        self._update_review_stats_label()
        if migrated:
            self._status.set(
                "📊 Nuevo esquema de conteo por caja — el histórico anterior se guardó aparte")

    def _save_review_stats(self):
        path = self._review_stats_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._review_stats["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            path.write_text(
                json.dumps(self._review_stats, indent=2, ensure_ascii=False),
                encoding="utf-8")
        except Exception:
            pass

    def _update_review_stats_label(self):
        totals = (self._review_stats or {}).get("totals", {})
        intacta = totals.get("confirmada_intacta", 0)
        reclas = totals.get("reclasificada", 0)
        fp = totals.get("falso_positivo", 0)
        no_det = totals.get("no_detectado", 0)
        self._review_stats_var.set(
            f"Conteo: confirmadas {intacta} | "
            f"reclas. {reclas} | FP {fp} | no det. {no_det}")

    def _show_counting_help(self):
        """Ventana de ayuda: explica qué significa cada categoría del conteo,
        para que quede claro mientras se reetiqueta. El texto se genera a
        partir de las mismas constantes que usa el sistema (GOOD_DETECTION_CLASSES,
        IOU_MATCH_MIN, IOU_INTACTA_MIN, CONF_BANDS), así que nunca queda desfasado
        si algún día se ajustan.
        """
        clases = ", ".join(sorted(self.GOOD_DETECTION_CLASSES))
        bandas = ", ".join(f"{name} [{lo:.2f}–{hi:.2f})" for lo, hi, name in self.CONF_BANDS)
        texto = f"""CÓMO SE CUENTA CADA CAJA

Cada vez que guardas, descartas o marcas "background", el sistema compara
la predicción ORIGINAL del modelo (before) con lo que dejas en el lienzo
en ese momento (after). Cada caja de clase objetivo ({clases}) del
modelo cae en EXACTAMENTE una de estas 5 categorías:

  ✔ confirmada_intacta
      La dejaste tal cual (misma clase, prácticamente sin moverla:
      solape ≥ {self.IOU_INTACTA_MIN:.0%} con la caja original).

  ✔ confirmada_ajustada
      La confirmaste pero le tocaste el tamaño o la posición
      (misma clase, solape entre {self.IOU_MATCH_MIN:.0%} y {self.IOU_INTACTA_MIN:.0%}).

  ↔ reclasificada
      Seguía siendo una detección real, pero cambiaste su clase
      (p.ej. el modelo dijo "barco2" y era "persona"). Se guarda además
      en una matriz de confusión (clase_modelo → clase_correcta) para
      ver qué confunde más el modelo.

  ✘ falso_positivo
      La caja del modelo no correspondía a nada real: la borraste, o la
      reclasificaste a una clase que NO es objetivo (p.ej. una boya).

  ➕ no_detectado
      Dibujaste una caja nueva que el modelo NO había predicho —el
      modelo se dejó ese objeto sin ver. Es la señal de que al modelo
      le falta recall, no precisión. Solo cuenta al guardar, no al
      descartar ni al marcar background.

BANDAS DE CONFIANZA
Cada confirmación/FP se clasifica también por la confianza que tenía
el modelo en el momento de detectarla: {bandas}.
Sirve para ver en qué rango de confianza falla más el modelo, y así
ajustar umbrales por sector si hace falta.

DESCARTAR: la única pregunta que verás
Si descartas una foto SIN tocar sus detecciones (las dejas vivas tal
cual), es el único caso realmente ambiguo: no sabemos si la foto no
vale para el dataset por otro motivo (duplicada, mala calidad...) o si
en realidad la detección era falsa. Por eso se te pregunta una vez
"¿Eran alarmas reales?". Si ya habías editado o borrado algo antes de
descartar, esa edición ya es la respuesta y NO se te pregunta nada.

DESHACER
El botón "↩ Deshacer última acción" (o Ctrl+Shift+Z) revierte solo la
ÚLTIMA acción: restaura el archivo a pendientes y resta del conteo
exactamente lo que esa acción había sumado. Deja de estar disponible
en cuanto haces una acción nueva."""

        win = tk.Toplevel(self)
        win.title("Cómo se cuenta el conteo de revisión")
        win.configure(bg="#1a1a2e")
        win.geometry("640x620")
        tk.Button(win, text="Cerrar", command=win.destroy, bg="#2a2a50", fg="#7ecfff",
                  relief="flat", padx=12, pady=4).pack(side="bottom", pady=8)
        scroll = tk.Scrollbar(win)
        scroll.pack(fill="y", side="right")
        text = tk.Text(win, bg="#12121e", fg="#e0e0e0", font=("Consolas", 9),
                        wrap="word", padx=14, pady=12, relief="flat",
                        yscrollcommand=scroll.set)
        text.insert("1.0", texto)
        text.config(state="disabled")
        text.pack(fill="both", expand=True, side="left")
        scroll.config(command=text.yview)

    @staticmethod
    def _class_token(box):
        return (box.get("cls_id"), box.get("name"))

    @staticmethod
    def _class_name(box):
        return str(box.get("name", "")).strip().lower()

    def _is_good_detection_class(self, box):
        return self._class_name(box) in self.GOOD_DETECTION_CLASSES

    @staticmethod
    def _box_iou(box_a, box_b):
        a = box_a.get("xyxy", [0, 0, 0, 0])
        b = box_b.get("xyxy", [0, 0, 0, 0])
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / max(1, area_a + area_b - inter)

    @staticmethod
    def _box_conf(box):
        try:
            return float(box.get("conf"))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _has_model_conf(box):
        try:
            float(box.get("conf"))
            return True
        except (TypeError, ValueError):
            return False

    @classmethod
    def _conf_band(cls, conf):
        for lo, hi, name in cls.CONF_BANDS:
            if lo <= conf < hi:
                return name
        return "desconocida"

    def _dedup_before_boxes(self, before, min_iou=None, include_non_target=False):
        """Colapsa cajas del propio modelo que se solapan sobre el MISMO objeto
        físico (duplicados del modelo, no del revisor) a una única caja
        representativa —la de mayor confianza— para no contarlas dos veces."""
        min_iou = self.IOU_INTACTA_MIN if min_iou is None else min_iou
        target = [
            b for b in before
            if self._has_model_conf(b)
            and (include_non_target or self._is_good_detection_class(b))
        ]
        groups = []
        for box in target:
            placed = False
            for group in groups:
                if any(self._box_iou(box, other) > min_iou for other in group):
                    group.append(box)
                    placed = True
                    break
            if not placed:
                groups.append([box])
        return [max(group, key=self._box_conf) for group in groups]

    def _model_box_groups(self, before, min_iou=None):
        min_iou = self.IOU_MATCH_MIN if min_iou is None else min_iou
        model_boxes = [b for b in before if self._has_model_conf(b)]
        groups = []
        for box in model_boxes:
            placed = False
            for group in groups:
                if any(self._box_iou(box, other) >= min_iou for other in group):
                    group.append(box)
                    placed = True
                    break
            if not placed:
                groups.append([box])
        return groups

    def _overlapping_after_boxes(self, group, after, min_iou=None):
        min_iou = self.IOU_MATCH_MIN if min_iou is None else min_iou
        overlaps = []
        for ai, box in enumerate(after):
            best_iou = max((self._box_iou(old_box, box) for old_box in group), default=0.0)
            if best_iou >= min_iou:
                overlaps.append((ai, box, best_iou))
        return overlaps

    def _match_boxes(self, before_repr, after, min_iou=None):
        """Emparejamiento greedy 1:1 por IoU descendente (no any() suelto):
        evita que dos cajas nuevas cercanas 'roben' la misma caja vieja."""
        min_iou = self.IOU_MATCH_MIN if min_iou is None else min_iou
        pairs = []
        for bi, old_box in enumerate(before_repr):
            for ai, new_box in enumerate(after):
                iou = self._box_iou(old_box, new_box)
                if iou >= min_iou:
                    pairs.append((iou, bi, ai))
        pairs.sort(key=lambda p: p[0], reverse=True)

        used_before, used_after, matches = set(), set(), {}
        for iou, bi, ai in pairs:
            if bi in used_before or ai in used_after:
                continue
            used_before.add(bi)
            used_after.add(ai)
            matches[bi] = (ai, iou)
        return matches, used_after

    # NOTA: variante alternativa NO USADA. Python se queda con la última
    # definición de `_diff_review_boxes` (más abajo), que sombreaba a esta.
    # Se conserva con otro nombre hasta decidir cuál de las dos es la correcta.
    def _diff_review_boxes_alt(self, before, after, action):
        """
        Única fuente de verdad del conteo. Compara SIEMPRE before (predicción
        original) vs after (estado del canvas en el momento de la acción —
        guardar, descartar o marcar background), y clasifica cada caja de
        clase objetivo en un resultado por caja:

          confirmada_intacta / confirmada_ajustada / reclasificada /
          falso_positivo / no_detectado

        Para 'discard', una caja que sobrevive sin haber sido tocada es el
        único caso ambiguo (need_confirmation=True) — el llamante decide si
        preguntar. Si ya se editó/borró antes de descartar, esa edición ya
        es la respuesta y no hace falta preguntar nada.
        """
        before_repr = self._dedup_before_boxes(before, include_non_target=True)
        matches, used_after = self._match_boxes(before_repr, after)

        results = []
        for bi, old_box in enumerate(before_repr):
            old_is_target = self._is_good_detection_class(old_box)
            if bi not in matches:
                if old_is_target:
                    results.append({"outcome": "falso_positivo", "box_before": old_box,
                                     "box_after": None, "iou_match": None, "need_confirmation": False})
                continue
            ai, iou = matches[bi]
            new_box = after[ai]
            new_is_target = self._is_good_detection_class(new_box)
            if not old_is_target and not new_is_target:
                continue
            if old_is_target and not new_is_target:
                # Reclasificada a no-objetivo (p.ej. boya) → señal clara, nunca ambigua
                results.append({"outcome": "falso_positivo", "box_before": old_box,
                                 "box_after": new_box, "iou_match": iou, "need_confirmation": False})
                continue
            if not old_is_target and new_is_target:
                if self._box_conf(old_box) > self._box_conf(new_box):
                    results.append({"outcome": "falso_positivo", "box_before": old_box,
                                     "box_after": new_box, "iou_match": iou, "need_confirmation": False,
                                     "record_confusion": True})
                results.append({"outcome": "no_detectado", "box_before": old_box,
                                 "box_after": new_box, "iou_match": iou, "need_confirmation": False,
                                 "record_confusion": True})
                continue
            same_class = self._class_token(old_box) == self._class_token(new_box)
            if same_class:
                outcome = "confirmada_intacta" if iou >= self.IOU_INTACTA_MIN else "confirmada_ajustada"
            else:
                outcome = "reclasificada"
            touched = (not same_class) or iou < self.IOU_INTACTA_MIN
            results.append({
                "outcome": outcome, "box_before": old_box, "box_after": new_box,
                "iou_match": iou, "need_confirmation": (action == "discard" and not touched),
            })

        if action != "discard":
            # Falsos negativos del modelo: solo cuentan al guardar para el dataset.
            for ai, new_box in enumerate(after):
                if ai in used_after or not self._is_good_detection_class(new_box):
                    continue
                if self._has_model_conf(new_box):
                    continue
                results.append({"outcome": "no_detectado", "box_before": None,
                                 "box_after": new_box, "iou_match": None, "need_confirmation": False})

        return results

    def _diff_review_boxes(self, before, after, action):
        results = []
        used_after = set()

        for group in self._model_box_groups(before):
            target_before = [b for b in group if self._is_good_detection_class(b)]
            non_target_before = [b for b in group if not self._is_good_detection_class(b)]
            best_target = max(target_before, key=self._box_conf) if target_before else None
            best_non_target = max(non_target_before, key=self._box_conf) if non_target_before else None

            overlaps = self._overlapping_after_boxes(group, after)
            for ai, _, _ in overlaps:
                used_after.add(ai)
            target_after = [(ai, b, iou) for ai, b, iou in overlaps if self._is_good_detection_class(b)]
            non_target_after = [(ai, b, iou) for ai, b, iou in overlaps if not self._is_good_detection_class(b)]

            if target_after:
                _, new_box, iou = max(target_after, key=lambda item: item[2])
                if best_target is not None:
                    target_wins = (
                        best_non_target is None
                        or self._box_conf(best_target) > self._box_conf(best_non_target)
                    )
                    if target_wins:
                        same_class = self._class_token(best_target) == self._class_token(new_box)
                        results.append({
                            "outcome": "confirmada_intacta" if same_class else "reclasificada",
                            "box_before": best_target,
                            "box_after": new_box,
                            "iou_match": iou,
                            "need_confirmation": (action == "discard" and same_class),
                        })
                    else:
                        results.append({
                            "outcome": "falso_positivo",
                            "box_before": best_non_target,
                            "box_after": new_box,
                            "iou_match": iou,
                            "need_confirmation": False,
                            "record_confusion": True,
                        })
                else:
                    results.append({
                        "outcome": "falso_positivo",
                        "box_before": best_non_target,
                        "box_after": new_box,
                        "iou_match": iou,
                        "need_confirmation": False,
                        "record_confusion": True,
                    })
                continue

            if best_target is None:
                continue

            target_wins = (
                best_non_target is None
                or self._box_conf(best_target) > self._box_conf(best_non_target)
            )
            if target_wins:
                after_box = max(non_target_after, key=lambda item: item[2])[1] if non_target_after else None
                after_iou = max((item[2] for item in non_target_after), default=None)
                results.append({
                    "outcome": "falso_positivo",
                    "box_before": best_target,
                    "box_after": after_box,
                    "iou_match": after_iou,
                    "need_confirmation": False,
                    "record_confusion": after_box is not None,
                })

        if action != "discard":
            for ai, new_box in enumerate(after):
                if ai in used_after or not self._is_good_detection_class(new_box):
                    continue
                results.append({
                    "outcome": "no_detectado",
                    "box_before": None,
                    "box_after": new_box,
                    "iou_match": None,
                    "need_confirmation": False,
                })

        return results

    def _record_review_result(self, img_path, action, boxes=None):
        before = list(getattr(self, "_original_boxes", []))
        after = list(boxes or [])
        results = self._diff_review_boxes(before, after, action)

        if action == "discard":
            good_results = [
                r for r in results
                if r["outcome"] in ("confirmada_intacta", "confirmada_ajustada", "reclasificada")
            ]
            if good_results:
                count_as_good = messagebox.askyesno(
                    "Confirmar descarte",
                    f"Hay {len(good_results)} deteccion(es) de interes en esta foto.\n\n"
                    f"Si descartas porque eran detecciones buenas y no quieres reforzarlas, pulsa Si.\n"
                    f"Si descartas por mala calidad, duplicado u otro motivo ajeno al modelo, pulsa No.",
                    icon="question")
                results = good_results if count_as_good else []
            else:
                results = []

        for r in results:
            r["need_confirmation"] = False

        # Único punto de ambigüedad real: descartar dejando vivas detecciones
        # sin tocar. Se pregunta UNA vez por acción, no caja a caja.
        ambiguous = []
        if ambiguous:
            eran_reales = messagebox.askyesno(
                "Confirmar descarte",
                f"Dejas {len(ambiguous)} detección(es) sin editar en esta foto "
                f"antes de descartarla.\n\n¿Eran alarmas reales?",
                icon="question")
            for r in ambiguous:
                if not eran_reales:
                    r["outcome"] = "falso_positivo"
                r["need_confirmation"] = False

        totals = self._review_stats.setdefault("totals", {})
        by_install = self._review_stats.setdefault("totals_por_instalacion", {})
        by_band = self._review_stats.setdefault("totals_por_banda_confianza", {})
        confusion = self._review_stats.setdefault("confusion_clases", {})
        installation = Path(img_path).parent.name
        replaced_event = self._remove_existing_review_event(Path(img_path).name, installation)
        self._last_replaced_review_event = replaced_event

        for r in results:
            cat = r["outcome"]
            totals[cat] = totals.get(cat, 0) + 1

            by_install.setdefault(installation, {})
            by_install[installation][cat] = by_install[installation].get(cat, 0) + 1

            band = (self._conf_band(self._box_conf(r["box_before"])) if r["box_before"] is not None
                    else ("manual" if cat == "no_detectado" else None))
            if band:
                by_band.setdefault(band, {})
                by_band[band][cat] = by_band[band].get(cat, 0) + 1

            if cat == "reclasificada" or r.get("record_confusion"):
                key = f"{self._class_name(r['box_before'])}->{self._class_name(r['box_after'])}"
                confusion[key] = confusion.get(key, 0) + 1

        categories = [r["outcome"] for r in results] or ["sin_conteo"]
        self._review_stats.setdefault("events", []).append({
            "image": str(img_path),
            "file": Path(img_path).name,
            "installation": installation,
            "action": action,
            "boxes": [
                {
                    "outcome": r["outcome"],
                    "cls_before": self._class_name(r["box_before"]) if r["box_before"] else None,
                    "cls_after": self._class_name(r["box_after"]) if r["box_after"] else None,
                    "conf_before": (round(self._box_conf(r["box_before"]), 3)
                                     if r["box_before"] is not None else None),
                    "iou_match": (round(r["iou_match"], 3) if r["iou_match"] is not None else None),
                    "record_confusion": bool(r.get("record_confusion")),
                }
                for r in results
            ],
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._save_review_stats()
        self._update_review_stats_label()
        return " + ".join(sorted(set(categories)))

    def _reverse_review_event_data(self, ev):
        installation = ev.get("installation")
        totals = self._review_stats.setdefault("totals", {})
        by_install = self._review_stats.setdefault("totals_por_instalacion", {})
        by_band = self._review_stats.setdefault("totals_por_banda_confianza", {})
        confusion = self._review_stats.setdefault("confusion_clases", {})

        for b in ev.get("boxes", []):
            cat = b["outcome"]
            totals[cat] = max(0, totals.get(cat, 0) - 1)
            if installation in by_install:
                by_install[installation][cat] = max(0, by_install[installation].get(cat, 0) - 1)
            conf_before = b.get("conf_before")
            band = (self._conf_band(conf_before) if conf_before is not None
                    else ("manual" if cat == "no_detectado" else None))
            if band and band in by_band:
                by_band[band][cat] = max(0, by_band[band].get(cat, 0) - 1)
            if cat == "reclasificada" or b.get("record_confusion"):
                key = f"{b.get('cls_before')}->{b.get('cls_after')}"
                if key in confusion:
                    confusion[key] = max(0, confusion[key] - 1)

    def _apply_review_event_data(self, ev):
        installation = ev.get("installation")
        totals = self._review_stats.setdefault("totals", {})
        by_install = self._review_stats.setdefault("totals_por_instalacion", {})
        by_band = self._review_stats.setdefault("totals_por_banda_confianza", {})
        confusion = self._review_stats.setdefault("confusion_clases", {})

        for b in ev.get("boxes", []):
            cat = b["outcome"]
            totals[cat] = totals.get(cat, 0) + 1
            by_install.setdefault(installation, {})
            by_install[installation][cat] = by_install[installation].get(cat, 0) + 1
            conf_before = b.get("conf_before")
            band = (self._conf_band(conf_before) if conf_before is not None
                    else ("manual" if cat == "no_detectado" else None))
            if band:
                by_band.setdefault(band, {})
                by_band[band][cat] = by_band[band].get(cat, 0) + 1
            if cat == "reclasificada" or b.get("record_confusion"):
                key = f"{b.get('cls_before')}->{b.get('cls_after')}"
                confusion[key] = confusion.get(key, 0) + 1

    def _remove_existing_review_event(self, file_name, installation):
        events = self._review_stats.setdefault("events", [])
        for idx in range(len(events) - 1, -1, -1):
            ev = events[idx]
            if ev.get("file") == file_name and ev.get("installation") == installation:
                removed = events.pop(idx)
                self._reverse_review_event_data(removed)
                return removed
        return None

    def _reverse_last_review_event(self):
        """Resta exactamente lo que sumó el último evento registrado — usado
        por 'Deshacer última acción'. No recalcula nada: revierte los mismos
        incrementos que _record_review_result aplicó, con los mismos datos
        que quedaron guardados en ese evento."""
        events = self._review_stats.get("events", [])
        if not events:
            return
        ev = events.pop()
        installation = ev.get("installation")
        totals = self._review_stats.setdefault("totals", {})
        by_install = self._review_stats.setdefault("totals_por_instalacion", {})
        by_band = self._review_stats.setdefault("totals_por_banda_confianza", {})
        confusion = self._review_stats.setdefault("confusion_clases", {})

        for b in ev.get("boxes", []):
            cat = b["outcome"]
            totals[cat] = max(0, totals.get(cat, 0) - 1)
            if installation in by_install:
                by_install[installation][cat] = max(0, by_install[installation].get(cat, 0) - 1)
            conf_before = b.get("conf_before")
            band = (self._conf_band(conf_before) if conf_before is not None
                    else ("manual" if cat == "no_detectado" else None))
            if band and band in by_band:
                by_band[band][cat] = max(0, by_band[band].get(cat, 0) - 1)
            if cat == "reclasificada" or b.get("record_confusion"):
                key = f"{b.get('cls_before')}->{b.get('cls_after')}"
                if key in confusion:
                    confusion[key] = max(0, confusion[key] - 1)

        self._save_review_stats()
        self._update_review_stats_label()

    # ── Deshacer última acción (discard/save/save_background) ──────────────

    def _read_txt_for_undo(self, txt_path):
        if not txt_path:
            return None
        try:
            return Path(txt_path).read_text(encoding="utf-8")
        except Exception:
            return None

    def _set_undo_snapshot(self, list_idx, img_path, txt_path, action, boxes_before,
                            dest_img=None, dest_lbl=None, moved_img=None, moved_txt=None,
                            original_txt_text=None, replaced_review_event=None):
        self._undo_snapshot = {
            "list_idx": list_idx,
            "img_path": Path(img_path),
            "txt_path": Path(txt_path) if txt_path else None,
            "action": action,
            "boxes_before": boxes_before,
            "original_txt_text": original_txt_text,
            "replaced_review_event": replaced_review_event,
            "dest_img": dest_img,     # copiado a reviewed_dir (save / save_background)
            "dest_lbl": dest_lbl,
            "moved_img": moved_img,   # movido a descartadas (discard)
            "moved_txt": moved_txt,
        }
        if hasattr(self, "_btn_undo"):
            self._btn_undo.config(state="normal")

    def _clear_undo_snapshot(self):
        self._undo_snapshot = None
        if hasattr(self, "_btn_undo"):
            self._btn_undo.config(state="disabled")

    def _undo_last_action(self):
        snap = getattr(self, "_undo_snapshot", None)
        if not snap:
            return
        img_path, txt_path = snap["img_path"], snap["txt_path"]
        try:
            if snap["action"] == "discard":
                # Devolver el archivo movido a descartadas/ a su sitio original
                if snap["moved_img"] and Path(snap["moved_img"]).exists():
                    shutil.move(str(snap["moved_img"]), str(img_path))
                if snap["moved_txt"] and Path(snap["moved_txt"]).exists():
                    shutil.move(str(snap["moved_txt"]), str(txt_path))
                else:
                    # No había txt local (o ya no existe) → reconstruirlo desde boxes_before
                    if txt_path:
                        if snap.get("original_txt_text") is not None:
                            txt_path.write_text(snap["original_txt_text"], encoding="utf-8")
                        else:
                            txt_path.write_text(
                                json.dumps({"boxes": snap["boxes_before"]}, ensure_ascii=False),
                                encoding="utf-8")
            else:
                # save / save_background: la imagen local se había borrado tras
                # copiarla a reviewed_dir — la restauramos desde esa copia y
                # reconstruimos el label local (JSON) desde boxes_before.
                if snap["dest_img"] and Path(snap["dest_img"]).exists() and not img_path.exists():
                    shutil.copy2(str(snap["dest_img"]), str(img_path))
                if txt_path:
                    if snap.get("original_txt_text") is not None:
                        txt_path.write_text(snap["original_txt_text"], encoding="utf-8")
                    else:
                        txt_path.write_text(
                            json.dumps({"boxes": snap["boxes_before"]}, ensure_ascii=False),
                            encoding="utf-8")
                # Borrar la copia que se había guardado en reviewed_dir
                if snap["dest_img"]:
                    Path(snap["dest_img"]).unlink(missing_ok=True)
                if snap["dest_lbl"]:
                    Path(snap["dest_lbl"]).unlink(missing_ok=True)
        except Exception as e:
            messagebox.showerror("No se pudo deshacer", f"Error restaurando archivos:\n{e}")
            return

        self._reverse_last_review_event()
        if snap.get("replaced_review_event"):
            self._review_stats.setdefault("events", []).append(snap["replaced_review_event"])
            self._apply_review_event_data(snap["replaced_review_event"])
            self._save_review_stats()
            self._update_review_stats_label()

        list_idx = min(snap["list_idx"], len(self._img_list))
        self._img_list.insert(list_idx, (img_path, txt_path))
        cav_meta = getattr(self, "_cav_relabel_meta", {}) or {}
        inst = Path(img_path).parent.name
        is_cav = str(img_path) in cav_meta
        label = f"🗄 CAV [{inst}] {Path(img_path).name}" if is_cav \
                else f"[{inst}] {Path(img_path).name}"
        self._img_lb.insert(list_idx, label)
        if is_cav:
            self._img_lb.itemconfig(list_idx, fg="#ffb347", bg="#2a1c0d")
        self._clear_undo_snapshot()
        self._cur_idx = -1
        self._load_image(list_idx)
        self._update_progress()
        self._status.set(f"↩ Deshecho: {Path(img_path).name} vuelve a pendientes")


    # ── Build UI ───────────────────────────────────────────────────────────

    def _build_ui(self):
        top = tk.Frame(self, bg="#1a1a2e", pady=8)
        top.pack(fill="x")
        self._user_label = tk.Label(top, text="👤  —",
            bg="#1a1a2e", fg="#7ecfff", font=("Consolas",10,"bold"))
        self._user_label.pack(side="left", padx=16)
        self._inst_label = tk.Label(top, text="Instalaciones: —",
            bg="#1a1a2e", fg="#aaaacc", font=("Consolas",9))
        self._inst_label.pack(side="left", padx=16)
        self._filter_stats_var = tk.StringVar(value="")
        self._filter_stats_label = tk.Label(top, textvariable=self._filter_stats_var,
            bg="#1a2030", fg="#aaddaa", font=("Consolas",8,"bold"),
            padx=10, pady=2, relief="flat")
        self._filter_stats_label.pack(side="left", padx=12)
        self._review_stats_label = tk.Label(top, textvariable=self._review_stats_var,
            bg="#241f12", fg="#ffd27e", font=("Consolas",8,"bold"),
            padx=10, pady=2, relief="flat")
        self._review_stats_label.pack(side="left", padx=12)
        tk.Button(top, text="❓ Cómo se cuenta", bg="#2a2a50", fg="#ffd27e",
                  relief="flat", font=("Consolas",8), padx=8, pady=2,
                  command=self._show_counting_help).pack(side="left", padx=4)
        self._btn_undo = tk.Button(top, text="↩ Deshacer última acción", bg="#2a2a50",
                  fg="#ff9e7e", relief="flat", font=("Consolas",8), padx=8, pady=2,
                  state="disabled", command=self._undo_last_action)
        self._btn_undo.pack(side="left", padx=4)
        self.bind_all("<Control-Shift-Z>", lambda e: self._undo_last_action())
        self.bind_all("<Control-Shift-z>", lambda e: self._undo_last_action())
        tk.Button(top, text="↺  Re-sincronizar", bg="#2a2a50", fg="#7ecfff",
                  relief="flat", font=("Consolas",9), padx=12, pady=4,
                  command=self._resync).pack(side="right", padx=(4,16))
        tk.Button(top, text="🗄  Reetiquetado (cav)", bg="#c84b00", fg="white",
                  relief="flat", font=("Consolas",9,"bold"), padx=12, pady=4,
                  command=self._open_cav_relabel).pack(side="right", padx=4)

        main_pane = tk.PanedWindow(self, orient="horizontal",
                                   bg="#1a1a2e", sashwidth=5,
                                   sashrelief="flat", sashpad=2)
        main_pane.pack(fill="both", expand=True)

        # Panel izquierdo
        left = tk.Frame(main_pane, bg="#222233")
        main_pane.add(left, minsize=160, width=200, stretch="never")
        prog_fr = tk.Frame(left, bg="#222233")
        prog_fr.pack(fill="x", padx=8, pady=(10,4))
        tk.Label(prog_fr, text="Progreso de revisión",
                 bg="#222233", fg="#888899", font=("Consolas",8)).pack(anchor="w")
        self._pb_review = ttk.Progressbar(prog_fr, length=210, mode="determinate")
        self._pb_review.pack(fill="x")
        self._counter_var = tk.StringVar(value="0 / 0")
        tk.Label(left, textvariable=self._counter_var,
                 bg="#222233", fg="#7ecfff",
                 font=("Consolas",9,"bold")).pack(pady=(2,6))
        ttk.Separator(left, orient="horizontal").pack(fill="x")
        tk.Label(left, text="Imágenes", bg="#222233", fg="#aaaacc",
                 font=("Consolas",9,"bold")).pack(anchor="w", padx=8, pady=(8,2))
        img_fr = tk.Frame(left, bg="#222233")
        img_fr.pack(fill="both", expand=True, padx=4)
        self._img_lb = tk.Listbox(img_fr, bg="#13131f", fg="#ccccdd",
                                   selectbackground="#4a90d9",
                                   selectforeground="white",
                                   bd=0, highlightthickness=0,
                                   font=("Consolas",8))
        sb2 = ttk.Scrollbar(img_fr, command=self._img_lb.yview)
        self._img_lb.config(yscrollcommand=sb2.set)
        self._img_lb.pack(side="left",fill="both",expand=True)
        sb2.pack(side="right",fill="y")
        self._img_lb.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Panel central
        center = tk.Frame(main_pane, bg="#2b2b2b")
        main_pane.add(center, minsize=400, stretch="always")

        act = tk.Frame(center, bg="#1a1a2e", pady=6)
        act.pack(fill="x")
        tk.Button(act, text="⬅", bg="#2a2a40", fg="white",
                  relief="flat", font=("Consolas",11), padx=10, pady=6,
                  command=self._go_prev).pack(side="left", padx=(10,2))
        self._btn_discard = tk.Button(act, text="🗑  Descartar foto",
                  bg="#5c1a1a", fg="white", relief="flat",
                  font=("Consolas",9,"bold"), padx=12, pady=6,
                  state="disabled", command=self._discard_current)
        self._btn_discard.pack(side="left", padx=4)
        self._btn_next = tk.Button(act, text="Guardar y siguiente  ➡",
                  bg="#c84b00", fg="white", relief="flat",
                  font=("Consolas",10,"bold"), padx=16, pady=6,
                  command=self._go_next)
        self._btn_next.pack(side="left", padx=4)
        tk.Frame(act, bg="#333355", width=2).pack(side="left", fill="y", padx=10)
        self._tool_btns={}
        for key, txt, color in [("edit","✏  Editar","#1565c0"),
                                  ("view","🔍  Vista","#2a2a40")]:
            b = tk.Button(act, text=txt, bg=color, fg="white", relief="flat",
                          font=("Consolas",9,"bold"), padx=12, pady=6,
                          command=lambda k=key: self._toggle_subbar(k))
            b.pack(side="left", padx=3)
            self._tool_btns[key]=b
        self._tool_hint = tk.StringVar(value="")
        tk.Label(act, textvariable=self._tool_hint,
                 bg="#1a1a2e", fg="#7ecfff",
                 font=("Consolas",8)).pack(side="left", padx=10)
        self._btn_add=tk.Frame(); self._btn_edit=tk.Frame(); self._btn_del=tk.Frame()
        self._active_subbar=None
        self._subbar = tk.Frame(center, bg="#111122", pady=4)

        self.canvas = BBoxCanvas(center, self)
        self.canvas.pack(fill="both", expand=True)

        # Panel derecho
        right = tk.Frame(main_pane, bg="#222233")
        main_pane.add(right, minsize=300, width=420, stretch="never")
        right_pane = tk.PanedWindow(right, orient="vertical",
                                    bg="#1a1a2e", sashwidth=5,
                                    sashrelief="flat", sashpad=2)
        right_pane.pack(fill="both", expand=True)

        # Vídeo (arriba)
        video_top = tk.Frame(right_pane, bg="#111122")
        video_top.pack_propagate(False)
        right_pane.add(video_top, minsize=220, stretch="always")
        vid_header = tk.Frame(video_top, bg="#1a1a2e")
        vid_header.pack(fill="x")
        tk.Label(vid_header, text="Contexto de vídeo",
                 bg="#1a1a2e", fg="#7ecfff",
                 font=("Consolas",9,"bold")).pack(side="left", padx=8, pady=6)
        self._btn_video = tk.Button(vid_header, text="▶  Ver vídeo",
            bg="#1a3a5c", fg="#7ecfff", relief="flat",
            font=("Consolas",9,"bold"), padx=10, pady=3,
            state="disabled", command=self._show_video_context)
        self._btn_video.pack(side="right", padx=8, pady=4)
        self._video_frame = tk.Frame(video_top, bg="#0a0a14")
        self._video_frame.pack_propagate(False)
        self._video_frame.pack(fill="both", expand=True, padx=6, pady=(0,4))
        self._video_canvas = tk.Canvas(self._video_frame, bg="#0a0a14",
                                       highlightthickness=0, bd=0)
        self._video_canvas.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
        self._video_status_text = self._video_canvas.create_text(
            0,0, text="Sin vídeo", fill="#333355", font=("Consolas",9), anchor="center")
        self._video_img_item = self._video_canvas.create_image(0,0,anchor="nw",image="")
        self._video_frame.bind("<Configure>", self._on_video_frame_resize)
        self._video_canvas.bind("<Configure>", self._on_video_canvas_configure)
        # Doble clic → pantalla completa
        self._video_canvas.bind("<Double-Button-1>",
                                lambda e: self._open_fullscreen_video())
        ctrl_fr = tk.Frame(video_top, bg="#1a1a2e")
        ctrl_fr.pack(fill="x", padx=6, pady=(0,4))
        self._btn_vplay = tk.Button(ctrl_fr, text="⏸", bg="#1a2a3a", fg="#7ecfff",
            relief="flat", font=("Consolas",12), width=3,
            state="disabled", command=self._toggle_video_pause)
        self._btn_vplay.pack(side="left", padx=(0,4))
        self._btn_vstop = tk.Button(ctrl_fr, text="⏹", bg="#1a2a3a", fg="#cc4444",
            relief="flat", font=("Consolas",12), width=3,
            state="disabled", command=self._stop_video)
        self._btn_vstop.pack(side="left")
        self._video_info = tk.StringVar(value="")
        _info_container = tk.Frame(ctrl_fr, bg="#1a1a2e", height=28)
        _info_container.pack_propagate(False)
        _info_container.pack(side="left", padx=8, fill="x", expand=True)
        tk.Label(_info_container, textvariable=self._video_info,
                 bg="#1a1a2e", fg="#556677",
                 font=("Consolas",7), wraplength=280, justify="left"
                 ).pack(anchor="w", pady=4)

        # Motivo YOLO (por qué se equivocó el detector, solo casos CAV) ──
        # OJO: esto NO es gpt_explicacion (la valoración de GPT sobre la
        # alarma). Es cav.gpt_feedback.motivo — la categoría de falso
        # positivo (JAULA/PAJARO/TUBOS/BOYA/BARCO_LEJANO...) que un
        # vigilante ya confirmó para este id_ai/sector. La valoración de
        # GPT se afina por prompt, no reetiquetando, así que no se
        # muestra aquí a propósito.
        motivo_top = tk.Frame(right_pane, bg="#111122")
        motivo_top.pack_propagate(False)
        right_pane.add(motivo_top, minsize=60, stretch="never")
        motivo_header = tk.Frame(motivo_top, bg="#1a1a2e")
        motivo_header.pack(fill="x")
        tk.Label(motivo_header, text="Motivo (por qué se equivocó YOLO)",
                 bg="#1a1a2e", fg="#ffcc44",
                 font=("Consolas",9,"bold")).pack(side="left", padx=8, pady=6)
        self._motivo_yolo_var = tk.StringVar(value="")
        self._motivo_yolo_label = tk.Label(
            motivo_top, textvariable=self._motivo_yolo_var,
            bg="#111122", fg="#cccc99", font=("Consolas",8),
            wraplength=380, justify="left", anchor="nw")
        self._motivo_yolo_label.pack(fill="both", expand=True, padx=8, pady=4)

        # Cajas (abajo)
        boxes_bot = tk.Frame(right_pane, bg="#222233")
        right_pane.add(boxes_bot, minsize=120, stretch="never")
        box_header = tk.Frame(boxes_bot, bg="#1a1a2e")
        box_header.pack(fill="x")
        tk.Label(box_header, text="Cajas detectadas",
                 bg="#1a1a2e", fg="#aaaacc",
                 font=("Consolas",9,"bold")).pack(side="left", padx=8, pady=6)
        box_canvas = tk.Canvas(boxes_bot, bg="#222233", highlightthickness=0)
        box_sb = ttk.Scrollbar(boxes_bot, orient="vertical", command=box_canvas.yview)
        box_canvas.configure(yscrollcommand=box_sb.set)
        box_sb.pack(side="right", fill="y")
        box_canvas.pack(side="left", fill="both", expand=True)
        self._box_scroll = tk.Frame(box_canvas, bg="#222233")
        self._box_scroll_id = box_canvas.create_window((0,0),window=self._box_scroll,anchor="nw")
        def _on_box_configure(e):
            box_canvas.configure(scrollregion=box_canvas.bbox("all"))
            box_canvas.itemconfig(self._box_scroll_id, width=box_canvas.winfo_width())
        self._box_scroll.bind("<Configure>", _on_box_configure)
        box_canvas.bind("<Configure>",
            lambda e: box_canvas.itemconfig(self._box_scroll_id, width=e.width))
        tk.Label(boxes_bot, text="💡 Clic=sel  Arrastrar=mover  Handle=resize  0-9=clase  Enter=guardar",
                 bg="#222233", fg="#333355",
                 font=("Consolas",7)).pack(anchor="w", padx=8, pady=4)

        self._status = tk.StringVar(value="Iniciando…")
        tk.Label(self, textvariable=self._status,
                 bg="#13131f", fg="#7ecfff", anchor="w",
                 padx=10, pady=4, font=("Consolas",8)).pack(fill="x")

    # ── Subbarra ───────────────────────────────────────────────────────────

    def _toggle_subbar(self, key):
        if self._active_subbar == key:
            self._subbar.pack_forget(); self._active_subbar=None
            for k,b in self._tool_btns.items(): b.config(relief="flat")
            self._tool_hint.set(""); return
        for w in self._subbar.winfo_children(): w.destroy()
        self._active_subbar=key
        for k,b in self._tool_btns.items():
            b.config(relief="sunken" if k==key else "flat")
        if key=="edit":   self._build_subbar_edit()
        elif key=="view": self._build_subbar_view()
        self._subbar.pack(in_=self.canvas.master, fill="x", before=self.canvas)

    def _build_subbar_edit(self):
        sb=self._subbar
        for label,color,shortcut in [
            ("✚ Dibujar","#2e7d32","arrastra en vacío"),
            ("✋ Mover","#1565c0","arrastra la caja"),
            ("⇲ Resize","#7b3f00","arrastra un handle"),
            ("✏ Clase","#5a2d82","doble clic / 0-9"),
            ("🗑 Borrar caja","#7b1f1f","Del (con sel.)"),
            ("🗑 Descartar foto","#7b1f1f","Del (sin sel.)"),
            ("↩ Deshacer","#2a3a4a","Ctrl+Z"),
            ("✔ Guardar+sig.","#c84b00","Enter"),
        ]:
            f=tk.Frame(sb,bg=color,padx=7,pady=3); f.pack(side="left",padx=4)
            tk.Label(f,text=label,bg=color,fg="white",
                     font=("Consolas",8,"bold")).pack(side="left")
            tk.Label(f,text=f"  {shortcut}",bg=color,fg="#dddddd",
                     font=("Consolas",7)).pack(side="left")
        self._tool_hint.set("Acciones disponibles directamente sobre la imagen")

    def _build_subbar_view(self):
        sb=self._subbar
        for txt,factor in [("🔍+  Acercar",1.25),("🔍−  Alejar",1/1.25)]:
            tk.Button(sb,text=txt,bg="#2a2a40",fg="#aaaacc",relief="flat",
                      font=("Consolas",9),padx=12,pady=4,
                      command=lambda f=factor: self.canvas.zoom(f)
                      ).pack(side="left",padx=4)
        tk.Button(sb,text="⊡  Ajustar todo",bg="#2a2a40",fg="#7ecfff",
                  relief="flat",font=("Consolas",9),padx=12,pady=4,
                  command=lambda: self.canvas.fit()
                  ).pack(side="left",padx=4)
        tk.Label(sb,text="  Rueda = zoom  |  Botón central = pan",
                 bg="#111122",fg="#556677",font=("Consolas",8)).pack(side="left",padx=10)
        self._tool_hint.set("Vista")

    def _resync(self):
        if self._user_cfg is None: return
        if self._dirty:
            if not messagebox.askyesno("Cambios sin guardar",
                    "¿Re-sincronizar de todos modos?"): return
        DownloadWindow(self, self._user_cfg, on_done=self._on_download_done)

    def _sync_edit_button(self): pass

    def _action_add(self):
        self._status.set("Arrastra sobre la imagen para dibujar una caja")

    def _discard_current(self):
        """Descarta la foto actual moviéndola a local_dir/descartadas/<instalacion>."""
        if self._cur_idx < 0 or self._cur_idx >= len(self._img_list):
            return
        img_path, txt_path = self._img_list[self._cur_idx]
        name = Path(img_path).name
        list_idx = self._cur_idx
        boxes_before = [dict(b) for b in getattr(self, "_original_boxes", [])]
        original_txt_text = self._read_txt_for_undo(txt_path)
        self._stop_video()
        category = self._record_review_result(img_path, "discard", self.canvas.boxes)
        replaced_review_event = getattr(self, "_last_replaced_review_event", None)
        moved_img, moved_txt = self._move_to_discarded(img_path, txt_path)
        self._set_undo_snapshot(list_idx, img_path, txt_path, "discard", boxes_before,
                                 moved_img=moved_img, moved_txt=moved_txt,
                                 original_txt_text=original_txt_text,
                                 replaced_review_event=replaced_review_event)
        self._img_list.pop(self._cur_idx)
        self._img_lb.delete(self._cur_idx)
        self.set_dirty(False)
        self._status.set(f"🗑  Descartada y movida a descartadas: {name} ({category})")
        total = len(self._img_list)
        if total == 0:
            self.canvas.delete("all")
            self._counter_var.set("0 / 0")
            self._btn_discard.config(state="disabled")
            self._btn_next.config(state="disabled")
            messagebox.showinfo("Sin imágenes", "No quedan imágenes por revisar.")
            return
        next_idx = min(self._cur_idx, total - 1)
        self._cur_idx = -1
        self._load_image(next_idx)

    def _discard_current_with_confirm(self):
        if self._cur_idx < 0 or self._cur_idx >= len(self._img_list):
            return
        img_name = Path(self._img_list[self._cur_idx][0]).name
        ok = messagebox.askyesno(
            "Descartar frame",
            f"¿Descartar '{img_name}'?\n\nSe moverá a la carpeta 'descartadas'.",
            icon="warning")
        if ok:
            self._discard_current()

    def _action_delete(self):
        self.canvas._delete_selected_key()

    # ── Navegación ─────────────────────────────────────────────────────────

    def _go_next(self):
        if self._cur_idx < 0: return
        cfg = self._user_cfg or {}
        dst = cfg.get("reviewed_dir") or cfg.get("local_dir","")
        if not dst:
            messagebox.showerror("Sin destino","No hay carpeta de destino configurada.")
            return
        self._stop_video()
        self.canvas.set_mode("idle")
        img_path, txt_path = self._img_list[self._cur_idx]
        list_idx = self._cur_idx
        boxes_before = [dict(b) for b in getattr(self, "_original_boxes", [])]
        original_txt_text = self._read_txt_for_undo(txt_path)
        if not self.canvas.boxes:
            dlg = EmptyBoxesDialog(self, Path(img_path).name)
            self.wait_window(dlg)
            if dlg.result == EmptyBoxesDialog.BACKGROUND:
                self._record_review_result(img_path, "save_background", [])
                replaced_review_event = getattr(self, "_last_replaced_review_event", None)
                dst_img, dst_lbl = self._save_one(img_path,[],dst,background=True)
                self._set_undo_snapshot(list_idx, img_path, txt_path, "save_background",
                                         boxes_before, dest_img=dst_img, dest_lbl=dst_lbl,
                                         original_txt_text=original_txt_text,
                                         replaced_review_event=replaced_review_event)
                self._status.set(f"🖼 Background: {Path(img_path).name}")
            elif dlg.result == EmptyBoxesDialog.DISCARD:
                # Reutiliza la misma lógica que el botón "Descartar foto":
                # mueve JPG/TXT a local_dir/descartadas/<instalacion>/,
                # elimina la entrada de la lista y carga la siguiente imagen.
                self._discard_current()
                return
            else:
                return
        else:
            self._record_review_result(img_path, "save", self.canvas.boxes)
            replaced_review_event = getattr(self, "_last_replaced_review_event", None)
            dst_img, dst_lbl = self._save_one(img_path, self.canvas.boxes, dst)
            self._set_undo_snapshot(list_idx, img_path, txt_path, "save",
                                     boxes_before, dest_img=dst_img, dest_lbl=dst_lbl,
                                     original_txt_text=original_txt_text,
                                     replaced_review_event=replaced_review_event)
            self._status.set(f"✔ Guardado: {Path(img_path).name}")
        self.set_dirty(False)
        self._delete_local_frame(img_path, txt_path)
        self._update_progress()
        if self._cur_idx < len(self._img_list)-1:
            self._load_image(self._cur_idx+1)
        else:
            messagebox.showinfo("Fin","¡Has revisado todas las imágenes!")

    def _go_prev(self):
        if self._cur_idx <= 0: return
        if self._dirty:
            resp = messagebox.askyesnocancel("Cambios sin guardar",
                                             "¿Guardar antes de ir a la anterior?")
            if resp is None: return
            if resp:
                cfg2=self._user_cfg or {}
                dst=cfg2.get("reviewed_dir") or cfg2.get("local_dir","")
                if dst:
                    img_path,_=self._img_list[self._cur_idx]
                    self._save_one(img_path,self.canvas.boxes,dst)
        self.canvas.set_mode("idle")
        self._load_image(self._cur_idx-1)

    def _on_listbox_select(self, event):
        sel=self._img_lb.curselection()
        if not sel: return
        idx=sel[0]
        if idx==self._cur_idx: return
        if self._dirty:
            resp=messagebox.askyesnocancel(
                "Cambios sin guardar",
                "¿Guardar antes de cambiar de imagen?\n\n"
                "• Sí  → guarda y la quita de la lista\n"
                "• No  → descarta cambios (la imagen queda pendiente)\n"
                "• Cancelar → volver a la imagen actual")
            if resp is None:
                self._img_lb.selection_clear(0,"end")
                self._img_lb.selection_set(self._cur_idx); return
            if resp:
                cfg2=self._user_cfg or {}
                dst=cfg2.get("reviewed_dir") or cfg2.get("local_dir","")
                if dst:
                    img_path,txt_path=self._img_list[self._cur_idx]
                    list_idx = self._cur_idx
                    boxes_before = [dict(b) for b in getattr(self, "_original_boxes", [])]
                    original_txt_text = self._read_txt_for_undo(txt_path)
                    self._record_review_result(img_path, "save", self.canvas.boxes)
                    replaced_review_event = getattr(self, "_last_replaced_review_event", None)
                    dst_img, dst_lbl = self._save_one(img_path,self.canvas.boxes,dst)
                    self._set_undo_snapshot(list_idx, img_path, txt_path, "save",
                                             boxes_before, dest_img=dst_img, dest_lbl=dst_lbl,
                                             original_txt_text=original_txt_text,
                                             replaced_review_event=replaced_review_event)
                    self._delete_local_frame(img_path, txt_path)
                    self._img_list.pop(self._cur_idx)
                    self._img_lb.delete(self._cur_idx)
                    self.set_dirty(False)
                    if idx > self._cur_idx:
                        idx -= 1
                    self._cur_idx = -1
                    self._load_image(idx)
                    return
        self.canvas.set_mode("idle")
        self._load_image(idx)

    # ── Carga de imagen ────────────────────────────────────────────────────

    def _resolve_pending_cls_ids(self, boxes):
        """
        cav_relabel.py no conoce la numeración real de cls_id de tu
        dataset, así que las cajas reconstruidas de casos CAV llegan con
        cls_id=-1 (sentinel, ver cav_relabel.CLS_ID_PENDIENTE) y solo el
        nombre ya resuelto ("barco", "barco2", "SIN_CLASIFICAR"...).
        Aquí se busca ese nombre en self._known_cls (ya construido a
        partir de tu dataset FTP) y se reutiliza el cls_id que ya usas
        para esa clase; si es un nombre nunca visto, se le da un id
        nuevo que no choque con los existentes — así nunca se inventa
        un cls_id fijo que pueda mezclar dos clases distintas al guardar.
        """
        for b in boxes:
            if b.get("cls_id") != -1:
                continue
            name = b.get("name", "unknown")
            match = next((cid for cid, n in self._known_cls if n == name), None)
            if match is None:
                match = max((c[0] for c in self._known_cls), default=-1) + 1
                self._known_cls.append((match, name))
                self._known_cls.sort()
            b["cls_id"] = match

    def _load_image(self, idx):
        if idx<0 or idx>=len(self._img_list): return
        self._cur_idx=idx
        img_path,txt_path=self._img_list[idx]
        try:
            pil=Image.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error",f"No se puede abrir:\n{e}"); return
        boxes=load_json_boxes(txt_path) if txt_path else []
        self._resolve_pending_cls_ids(boxes)
        self._original_boxes = [dict(b) for b in boxes]
        self.canvas.load(pil,boxes)
        self.set_dirty(False); self._update_progress()
        self._btn_discard.config(state="normal")
        self._stop_video()
        self._video_url=None; self._video_fn=None; self._video_saved_at=None
        if txt_path:
            try:
                with open(txt_path,"r",encoding="utf-8") as f:
                    meta=json.load(f)
                self._video_url      = meta.get("video_url")
                self._video_fn       = meta.get("frame_number")
                self._video_saved_at = meta.get("saved_at")
            except Exception:
                pass
        has_video=bool(self._video_url)
        self._btn_video.config(state="normal" if has_video else "disabled")
        mode_txt = ("pip install imageio-ffmpeg"
                    if has_video and not IMAGEIO_AVAILABLE and not CV2_AVAILABLE
                    else "▶  Pulsa para cargar" if has_video else "Sin vídeo")
        self._video_canvas.itemconfig(self._video_status_text,text=mode_txt)
        self._video_canvas.itemconfig(self._video_img_item,image="")
        self._video_info.set("")
        inst=Path(img_path).parent.name
        self._status.set(f"[{inst}]  {Path(img_path).name}  —  {len(boxes)} caja(s)")
        cav_meta = getattr(self, "_cav_relabel_meta", {}) or {}
        cav_case = cav_meta.get(str(img_path))
        if cav_case:
            motivo_yolo = cav_case[4] if len(cav_case) > 4 else None
            origen      = cav_case[5] if len(cav_case) > 5 else None
            clase_s3    = cav_case[6] if len(cav_case) > 6 else None
            if motivo_yolo:
                sello = "👤 confirmado por vigilante" if origen == "vigilante" \
                        else "🤖 solo sugerencia de GPT, sin confirmar por vigilante"
                yolo_txt = f"YOLO detectó \"{clase_s3}\" → " if clase_s3 else ""
                self._motivo_yolo_var.set(
                    f"{yolo_txt}en realidad es {motivo_yolo}  ({sello})")
            else:
                self._motivo_yolo_var.set(
                    "Sin motivo registrado todavía (ningún vigilante confirmó "
                    "aún la categoría de falso positivo para este caso).")
        else:
            self._motivo_yolo_var.set("(no aplica — imagen de descarga FTP normal, no de reetiquetado CAV)")
        self._img_lb.selection_clear(0,"end")
        self._img_lb.selection_set(idx); self._img_lb.see(idx)
        for b in boxes:
            cid=b.get("cls_id",0); n=b.get("name","unknown")
            if not any(c[0]==cid for c in self._known_cls):
                self._known_cls.append((cid,n)); self._known_cls.sort()
        self._update_box_panel()

    def _update_progress(self):
        total=len(self._img_list)
        if total==0: return
        done=self._cur_idx+1
        self._counter_var.set(f"{done} / {total}")
        self._pb_review.config(maximum=total,value=done)
        pct=done/total
        color=("#4a90d9" if pct<0.5 else "#f0a500" if pct<0.9 else "#4caf50")
        ttk.Style().configure("TProgressbar",background=color)

    def set_dirty(self, val):
        self._dirty=val
        self._btn_next.config(bg="#ff6d00" if val else "#c84b00")

    def ask_new_box(self, x1,y1,x2,y2):
        if not self._known_cls:
            messagebox.showwarning("Sin clases",
                "No hay clases. Carga imágenes con .txt primero."); return None
        dlg=NewBoxDialog(self,self._known_cls)
        if dlg.result:
            box=dict(dlg.result); box["conf"]=None
            box["xyxy"]=[round(x1,1),round(y1,1),round(x2,1),round(y2,1)]
            cid=box["cls_id"]
            if not any(c[0]==cid for c in self._known_cls):
                self._known_cls.append((cid,box["name"])); self._known_cls.sort()
            return box
        return None

    # ── Panel de cajas ─────────────────────────────────────────────────────

    def _update_box_panel(self):
        for w in self._box_scroll.winfo_children(): w.destroy()
        if not self.canvas.boxes:
            tk.Label(self._box_scroll,text="Sin cajas",bg="#222233",fg="#555577",
                     font=("Consolas",9)).pack(pady=10); return
        for i,b in enumerate(self.canvas.boxes):
            x1,y1,x2,y2=b["xyxy"]
            conf_str=f"{b['conf']:.2f}" if b.get("conf") is not None else "manual"
            color=color_for_cls(b["cls_id"])
            is_sel=(i==self.canvas._sel)
            row_bg="#253050" if is_sel else "#1a1a2e"
            row=tk.Frame(self._box_scroll,bg=row_bg,pady=3,padx=6,cursor="hand2")
            row.pack(fill="x",pady=1,padx=4)
            def _select(ev,idx=i):
                self.canvas._sel=idx; self.canvas._render(); self._update_box_panel()
            row.bind("<Button-1>",_select)
            tk.Label(row,text="█",fg=color,bg=row_bg,
                     font=("Consolas",13),cursor="hand2").pack(side="left")
            info=tk.Label(row,
                text=f"#{i+1}  {b['name']}  {conf_str}\n"
                     f"{x1:.0f},{y1:.0f} → {x2:.0f},{y2:.0f}",
                bg=row_bg,fg="#ccccdd",font=("Consolas",8),
                justify="left",cursor="hand2")
            info.pack(side="left",padx=6)
            info.bind("<Button-1>",_select)
            def _del(ev,idx=i): self.canvas._delete_by_idx(idx)
            del_btn=tk.Label(row,text="✕",fg="#ff4444",bg=row_bg,
                             font=("Consolas",11,"bold"),cursor="hand2",padx=4)
            del_btn.pack(side="right"); del_btn.bind("<Button-1>",_del)
            def _cls(ev,idx=i): self.canvas._change_class_menu(idx)
            cls_btn=tk.Label(row,text="✏",fg="#7ecfff",bg=row_bg,
                             font=("Consolas",10),cursor="hand2",padx=4)
            cls_btn.pack(side="right"); cls_btn.bind("<Button-1>",_cls)

    # ── Vídeo ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pil_to_canvas_photo(pil_frame, canvas_w, canvas_h):
        fw, fh = pil_frame.size
        sc = min(canvas_w / fw, canvas_h / fh)
        nw, nh = max(1, int(fw * sc)), max(1, int(fh * sc))
        resized = pil_frame.resize((nw, nh), Image.LANCZOS)
        bg = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        bg.paste(resized, ((canvas_w - nw) // 2, (canvas_h - nh) // 2))
        return ImageTk.PhotoImage(bg)

    def _center_video_item(self):
        self._video_canvas.coords(self._video_img_item, 0, 0)
        cw = max(1, self._video_canvas.winfo_width())
        ch = max(1, self._video_canvas.winfo_height())
        self._video_canvas.coords(self._video_status_text, cw // 2, ch // 2)
        self._video_canvas.itemconfig(self._video_status_text, anchor="center")

    def _on_video_canvas_configure(self, event):
        self._center_video_item()

    def _on_video_frame_resize(self, event):
        if not self._vcached_pil: return
        if self._resize_timer is not None: self.after_cancel(self._resize_timer)
        self._resize_timer=self.after(150,self._rescale_video)

    def _rescale_video(self):
        self._resize_timer=None
        if not self._vcached_pil: return
        pw=max(160,self._video_canvas.winfo_width())
        ph=max(120,self._video_canvas.winfo_height())
        last_pw,last_ph=self._last_panel_size
        if abs(pw-last_pw)<20 and abs(ph-last_ph)<20: return
        if (pw,ph)==self._last_panel_size: return
        self._last_panel_size=(pw,ph)
        self._vplaying=False; time.sleep(0.05)
        def _do():
            scaled=[self._pil_to_canvas_photo(f,pw,ph) for f in self._vcached_pil]
            self._play_frames_loop(scaled,self._vcached_fps)
        threading.Thread(target=_do,daemon=True).start()

    def _show_video_context(self):
        if not self._video_url: return
        if not IMAGEIO_AVAILABLE and not CV2_AVAILABLE:
            self._video_canvas.itemconfig(self._video_status_text,
                text="pip install imageio-ffmpeg"); return
        self._stop_video()
        self._vplaying=True
        self._btn_video.config(state="disabled",text="⏳ Extrayendo…")
        self._video_canvas.itemconfig(self._video_status_text,text="⏳ Conectando…")
        threading.Thread(target=self._stream_clip,daemon=True).start()

    def _stream_clip(self):
        import subprocess, re as _re
        video_url  = self._video_url
        fn         = self._video_fn if self._video_fn is not None else 0

        panel_w  = max(320, self._video_canvas.winfo_width())
        try:
            if IMAGEIO_AVAILABLE:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            else:
                import shutil as _sh
                ffmpeg_exe = _sh.which("ffmpeg") or "ffmpeg"
        except Exception as e:
            titulo,detalle=_classify_video_error(str(e),video_url or "")
            self.after(0,lambda t=titulo,d=detalle: self._video_load_failed_detail(t,d))
            return
        self.after(0, lambda: self._video_canvas.itemconfig(
            self._video_status_text, text="⏳ Comprobando enlace S3…"))
        refreshed = _refresh_s3_presigned_url(video_url, self._user_cfg or {})
        if refreshed != video_url:
            video_url = refreshed
            self.after(0, lambda: self._video_canvas.itemconfig(
                self._video_status_text, text="⏳ URL S3 refrescada…"))

        self.after(0,lambda: self._video_canvas.itemconfig(
            self._video_status_text,text="⏳ Verificando enlace…"))
        try:
            import urllib.request as _ur
            _url = require_https(video_url or "")
            if 'x-amz-signature' in _url.lower():
                _req = _ur.Request(_url, headers={"Range": "bytes=0-0"})
                _ur.urlopen(_req, timeout=8)
            else:
                _ur.urlopen(_ur.Request(_url, method="HEAD"), timeout=8)
        except Exception as he:
            titulo,detalle=_classify_video_error(str(he),video_url or "")
            self.after(0,lambda t=titulo,d=detalle: self._video_load_failed_detail(t,d))
            return
        self.after(0,lambda: self._video_canvas.itemconfig(
            self._video_status_text,text="⏳ Detectando fps…"))
        fps_res=subprocess.run([ffmpeg_exe,"-hide_banner","-i",video_url,
            "-t","0","-f","null","-"],
            capture_output=True,timeout=15)
        fps=25.
        m=_re.search(r'(\d+(?:\.\d+)?)\s*fps',
                     fps_res.stderr.decode("utf-8",errors="ignore"))
        if m: fps=float(m.group(1))

        t_frame  = fn / fps
        t_start  = max(0., t_frame - VIDEO_CONTEXT_SECS)
        t_end    = t_frame + VIDEO_CONTEXT_SECS
        duration = t_end - t_start

        cmd=[ffmpeg_exe,"-hide_banner","-loglevel","error",
             "-ss",f"{t_start:.3f}","-i",video_url,
             "-t",f"{duration:.3f}","-vf",f"scale={panel_w}:-2",
             "-f","rawvideo","-pix_fmt","rgb24","pipe:1"]
        try:
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,bufsize=10**7)
        except Exception as e:
            self.after(0,lambda err=str(e)[:70]:
                self._video_load_failed(f"❌ {err}")); return
        self._vproc=proc; frames=[]; frame_h=None; frame_sz=None
        self.after(0,lambda: self._video_canvas.itemconfig(
            self._video_status_text,text="⏳ Descargando…"))
        while self._vplaying:
            if frame_sz is None:
                chunk=proc.stdout.read(panel_w*4)
                if not chunk: break
                panel_h_w=max(200,self._video_canvas.winfo_height())
                frame_h=min(panel_w*960//1280,panel_h_w)
                frame_sz=panel_w*frame_h*3
                rest=proc.stdout.read(frame_sz-len(chunk))
                if rest is None: break
                chunk=chunk+rest
            else:
                chunk=proc.stdout.read(frame_sz)
            if not chunk or len(chunk)<frame_sz: break
            try:
                pil=Image.frombuffer("RGB",(panel_w,frame_h),chunk,"raw","RGB",0,1)
                frames.append(pil)
            except Exception: break
        proc.stdout.close(); proc.wait()
        if not frames:
            stderr_txt=proc.stderr.read().decode("utf-8",errors="ignore") if proc.stderr else ""
            titulo,detalle=_classify_video_error(stderr_txt,video_url or "")
            self.after(0,lambda t=titulo,d=detalle: self._video_load_failed_detail(t,d))
            return
        from PIL import ImageDraw
        f_idx=min(int((t_frame-t_start)*fps),len(frames)-1)
        for idx in range(max(0,f_idx-1),min(len(frames),f_idx+2)):
            draw=ImageDraw.Draw(frames[idx]); w,h=frames[idx].size
            draw.rectangle([0,0,w-1,h-1],outline=(0,200,255),width=4)
        self._vcached_pil=frames; self._vcached_fps=fps
        def _convert():
            pw=max(160,self._video_canvas.winfo_width())
            ph=max(120,self._video_canvas.winfo_height())
            self._last_panel_size=(pw,ph)
            scaled=[self._pil_to_canvas_photo(f,pw,ph) for f in frames]
            self._video_ready(f"t={t_start:.1f}s→{t_end:.1f}s | frame {fn} | {len(frames)} frames")
            self._play_frames_loop(scaled,fps)
        self.after(0,_convert)

    def _play_frames_loop(self, frames, fps):
        self._vplaying=True
        def _loop():
            delay=max(0.015,1./fps); i=0
            while self._vplaying:
                if self._vpause: time.sleep(0.05); continue
                tk_img=frames[i%len(frames)]; self._vtk_img=tk_img
                def _show(img=tk_img):
                    self._video_canvas.itemconfig(self._video_status_text,text="")
                    self._video_canvas.itemconfig(self._video_img_item,image=img)
                self.after(0,_show); i+=1; time.sleep(delay)
        threading.Thread(target=_loop,daemon=True).start()

    def _video_ready(self, info):
        self._center_video_item()
        self._btn_video.config(state="normal",text="▶  Ver contexto en vídeo")
        self._btn_vplay.config(state="normal"); self._btn_vstop.config(state="normal")
        self._video_info.set(info)

    def _video_load_failed(self, msg="❌ No se pudo abrir el vídeo"):
        self._video_load_failed_detail(msg,"")

    def _video_load_failed_detail(self, titulo, detalle):
        self._btn_video.config(state="normal",text="↺  Reintentar")
        self._video_canvas.itemconfig(self._video_img_item,image="")
        self._video_canvas.delete("err_detail")
        self._center_video_item()
        cx=max(10,self._video_canvas.winfo_width()//2)
        cy=max(10,self._video_canvas.winfo_height()//2)
        self._video_canvas.coords(self._video_status_text,cx,cy-14)
        self._video_canvas.itemconfig(self._video_status_text,
            text=titulo,fill="#ff6666",font=("Consolas",9,"bold"),anchor="center")
        if detalle:
            self._video_canvas.create_text(cx,cy+14,text=detalle,
                fill="#aaaacc",font=("Consolas",8),
                width=self._video_canvas.winfo_width()-20,
                anchor="n",tags="err_detail")
        self._video_info.set(titulo)

    def _toggle_video_pause(self):
        if not self._vplaying: return
        self._vpause=not self._vpause
        self._btn_vplay.config(text="▶" if self._vpause else "⏸")

    # ── Pantalla completa de vídeo ─────────────────────────────────────────

    def _open_fullscreen_video(self):
        """
        Doble clic en el panel de vídeo → ventana fullscreen.
        Escala un frame por tick (sin pre-generar array): memoria constante.
        """
        if not self._vcached_pil:
            if self._video_url:
                self._show_video_context()
            return

        fs = tk.Toplevel(self)
        fs.title("Vídeo — pantalla completa")
        fs.configure(bg="black")
        fs.attributes("-fullscreen", True)
        fs.lift()
        fs.focus_force()
        fs.update_idletasks()

        sw = fs.winfo_screenwidth()
        sh = fs.winfo_screenheight()

        canvas = tk.Canvas(fs, bg="black", highlightthickness=0, bd=0)
        canvas.pack(fill="both", expand=True)

        first = self._pil_to_canvas_photo(self._vcached_pil[0], sw, sh)
        img_item = canvas.create_image(0, 0, anchor="nw", image=first)
        canvas._current_img = first

        canvas.create_text(
            sw // 2, sh - 32,
            text="ESC · Enter · Doble clic = cerrar",
            fill="#555577", font=("Consolas", 10), anchor="center"
        )

        playing = [True]
        idx     = [0]
        fps     = self._vcached_fps
        delay   = max(15, int(1000 / fps))

        def _close(event=None):
            playing[0] = False
            try: fs.destroy()
            except Exception: pass

        for seq in ("<Escape>", "<Return>", "<KP_Enter>", "<Double-Button-1>"):
            fs.bind(seq, _close)
            canvas.bind(seq, _close)

        def _tick():
            if not playing[0]:
                return
            pil = self._vcached_pil[idx[0] % len(self._vcached_pil)]
            img = self._pil_to_canvas_photo(pil, sw, sh)
            canvas.itemconfig(img_item, image=img)
            canvas._current_img = img
            idx[0] += 1
            try:
                fs.after(delay, _tick)
            except Exception:
                pass

        fs.after(delay, _tick)

    def _stop_video(self):
        self._vplaying=False; self._vpause=False
        if hasattr(self,"_vproc") and self._vproc:
            try: self._vproc.kill()
            except Exception: pass
            self._vproc=None
        self._vcached_pil=[]; self._vcached_fps=25.
        if hasattr(self,"_btn_vplay"): self._btn_vplay.config(state="disabled",text="⏸")
        if hasattr(self,"_btn_vstop"): self._btn_vstop.config(state="disabled")
        if hasattr(self,"_video_canvas"):
            self._video_canvas.itemconfig(self._video_img_item,image="")
            self._center_video_item()
            self._video_canvas.itemconfig(self._video_status_text,text="⏹ Detenido")
        if hasattr(self,"_vtmp_path") and self._vtmp_path:
            try: os.remove(self._vtmp_path)
            except Exception: pass
            self._vtmp_path=None

    # ── Guardar ────────────────────────────────────────────────────────────

    def _delete_local_frame(self, img_path, txt_path):
        try: Path(img_path).unlink(missing_ok=True)
        except Exception: pass
        if txt_path:
            try: Path(txt_path).unlink(missing_ok=True)
            except Exception: pass

    def _unique_path(self, path):
        """Devuelve una ruta libre añadiendo _1, _2... si ya existe."""
        path = Path(path)
        if not path.exists():
            return path
        for i in range(1, 10000):
            candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"No se pudo generar un nombre libre para {path}")

    def _move_to_discarded(self, img_path, txt_path):
        """Mueve jpg/txt actuales a local_dir/descartadas/<instalacion>.
        Devuelve (destino_img, destino_txt) para poder deshacerlo."""
        img_path = Path(img_path)
        txt_path = Path(txt_path) if txt_path else None

        cfg = self._user_cfg or {}
        local_root = Path(cfg.get("local_dir", img_path.parent.parent))
        installation = img_path.parent.name or "sin_instalacion"
        discard_dir = local_root / "descartadas" / installation
        discard_dir.mkdir(parents=True, exist_ok=True)

        dest_img, dest_txt = None, None
        if img_path.exists():
            dest_img = self._unique_path(discard_dir / img_path.name)
            shutil.move(str(img_path), str(dest_img))
        if txt_path and txt_path.exists():
            dest_txt = self._unique_path(discard_dir / txt_path.name)
            shutil.move(str(txt_path), str(dest_txt))
        return dest_img, dest_txt

    def _save_one(self, img_path, boxes, dst_root, background=False):
        img_path=Path(img_path)
        installation = img_path.parent.name or "sin_instalacion"
        inst_root = Path(dst_root) / installation
        imgs_dir=inst_root/"images"; imgs_dir.mkdir(parents=True,exist_ok=True)
        lbls_dir=inst_root/"labels"; lbls_dir.mkdir(parents=True,exist_ok=True)
        dst_img=imgs_dir/img_path.name
        lbl_path=lbls_dir/(img_path.stem+".txt")
        if img_path.resolve()!=dst_img.resolve():
            if dst_img.exists() or lbl_path.exists():
                for suffix in range(1, 10000):
                    candidate_img = imgs_dir / f"{img_path.stem}_{suffix}{img_path.suffix}"
                    candidate_lbl = lbls_dir / f"{img_path.stem}_{suffix}.txt"
                    if not candidate_img.exists() and not candidate_lbl.exists():
                        dst_img, lbl_path = candidate_img, candidate_lbl
                        break
                else:
                    raise RuntimeError(f"No se pudo generar un nombre libre para {img_path.name}")
                lbl_path = lbls_dir/(dst_img.stem+".txt")
            shutil.copy2(img_path,dst_img)
        else:
            lbl_path = lbls_dir/(dst_img.stem+".txt")
        if background:
            lbl_path.write_text("",encoding="utf-8")
        else:
            pil=Image.open(img_path); iw,ih=pil.size; pil.close()
            save_yolo_labels(lbl_path,boxes,iw,ih)

        # Si esta imagen venía del origen cav (reetiquetado), avisar a
        # cav.reetiquetado_hecho con la clase final que se ha dejado,
        # para que no vuelva a salir en v_pendiente_reetiquetado.
        # Best-effort: un fallo aquí no debe impedir el guardado local,
        # que ya se ha hecho justo arriba.
        meta = getattr(self, "_cav_relabel_meta", None)
        if meta and str(img_path) in meta and CAV_RELABEL_AVAILABLE:
            id_ai, sector, clase_yolo_orig, clase_sugerida, _motivo_yolo, _origen = meta[str(img_path)]
            if boxes:
                cls_map = dict(self._known_cls)
                clase_final = cls_map.get(boxes[0]["cls_id"], clase_sugerida)
            else:
                clase_final = "SIN_OBJETO"   # se vació la caja: no había nada que reetiquetar
            try:
                cav_relabel.mark_relabeled(
                    self._user_cfg, id_ai, sector, self._username, clase_final
                )
            except Exception as e:
                print(f"[CAV_RELABEL] no se pudo marcar id_ai={id_ai} "
                      f"sector={sector} como revisado: {e}")

        return dst_img, lbl_path


# ─────────────────────── Main ──────────────────────────────────────────────

if __name__ == "__main__":
    app = BoxReviewer()
    app.mainloop()