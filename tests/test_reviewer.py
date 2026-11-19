import pytest

from security_utils import hash_password, verify_password, check_user_password, require_https


def test_hash_y_verificacion():
    h = hash_password("correcta")
    assert h.startswith("scrypt$")
    assert "correcta" not in h
    assert verify_password("correcta", h)
    assert not verify_password("otra", h)


def test_hash_con_sal_distinta():
    assert hash_password("x") != hash_password("x")


@pytest.mark.parametrize("basura", ["", "scrypt$1$2", "md5$a$b$c$d$e", "scrypt$a$b$c$d$e", None])
def test_verify_nunca_lanza(basura):
    assert verify_password("x", basura) is False


def test_check_user_password():
    cfg = {"password_hash": hash_password("pw")}
    assert check_user_password(cfg, "pw")
    assert not check_user_password(cfg, "mal")
    assert not check_user_password(None, "pw")
    assert not check_user_password({}, "pw")


def test_texto_plano_rechazado_por_defecto():
    cfg = {"password": "pw"}
    assert not check_user_password(cfg, "pw")
    assert check_user_password(cfg, "pw", allow_plaintext=True)
    assert not check_user_password(cfg, "mal", allow_plaintext=True)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://h/x", "http://h/x", "https://", "", None, "//h/x"])
def test_require_https_rechaza(url):
    with pytest.raises(ValueError):
        require_https(url)


def test_require_https_acepta():
    u = "https://bucket.s3.amazonaws.com/v.mp4?X-Amz-Signature=abc"
    assert require_https(u) == u


def test_sin_instalaciones_no_consulta_la_bd(monkeypatch):
    cav_relabel = pytest.importorskip("cav_relabel")

    def no_debe_conectar(*a, **k):
        raise AssertionError("no debería abrir conexión")

    monkeypatch.setattr(cav_relabel, "_connect", no_debe_conectar)
    assert cav_relabel.get_pending_relabel({}, []) == []
