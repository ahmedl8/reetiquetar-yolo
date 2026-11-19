"""
security_utils.py — utilidades de seguridad del revisor
=======================================================
- Hash de contraseñas con scrypt (stdlib, sin dependencias) y comparación
  en tiempo constante.
- Validación de esquema para URLs antes de abrirlas.

Generar el hash de una contraseña para users.json:

    python security_utils.py            # pide la contraseña por teclado
"""

import base64
import getpass
import hashlib
import hmac
import os
from urllib.parse import urlparse

_N, _R, _P = 2 ** 14, 8, 1
_PREFIX = "scrypt"


def hash_password(password: str) -> str:
    """Devuelve 'scrypt$N$r$p$<salt_b64>$<hash_b64>'."""
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "$".join([
        _PREFIX, str(_N), str(_R), str(_P),
        base64.b64encode(salt).decode(), base64.b64encode(dk).decode(),
    ])


def verify_password(password: str, stored: str) -> bool:
    """True si `password` coincide con el hash `stored`. Nunca lanza."""
    try:
        prefix, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if prefix != _PREFIX:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                            n=int(n), r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def check_user_password(user_cfg: dict, password: str, allow_plaintext: bool = False) -> bool:
    """Comprueba la contraseña de un usuario de users.json.

    Usa `password_hash`. El campo antiguo `password` (texto plano) solo se
    acepta si allow_plaintext=True (migración); por defecto se rechaza.
    """
    if not user_cfg:
        return False
    stored_hash = user_cfg.get("password_hash")
    if stored_hash:
        return verify_password(password, stored_hash)
    if allow_plaintext and user_cfg.get("password"):
        return hmac.compare_digest(str(user_cfg["password"]).encode(), password.encode())
    return False


def require_https(url: str) -> str:
    """Lanza ValueError si la URL no es https con host (evita file://, ftp://…)."""
    parsed = urlparse(url or "")
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("URL no permitida: solo se admite https")
    return url


if __name__ == "__main__":
    pw = getpass.getpass("Contraseña: ")
    if pw != getpass.getpass("Repite la contraseña: "):
        raise SystemExit("No coinciden.")
    print(hash_password(pw))
