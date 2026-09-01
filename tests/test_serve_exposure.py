"""Binding an unauthenticated surveillance API to the network must be loud.

There is deliberately no authentication on this API. On loopback that is a
development convenience. On 0.0.0.0 it hands every alert, every event and the
paths to the evidence files to anyone who can reach the port - at a border
post, that is the map of where the cameras are and what they have seen.

The decision to expose it is the deploying authority's. Being told is not.
"""

import pytest

import serve


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_is_not_exposed(host):
    assert serve.exposed(host) is False


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.7", "10.0.0.5"])
def test_anything_reachable_from_outside_counts_as_exposed(host):
    """A LAN address exposes the API exactly as completely as 0.0.0.0 does."""
    assert serve.exposed(host) is True


def test_serving_on_the_network_warns_about_the_missing_auth(monkeypatch, tmp_path):
    said: list[str] = []
    monkeypatch.setattr(serve.ui, "warn", said.append)
    monkeypatch.setattr(serve.ui, "banner", lambda *a, **k: None)
    monkeypatch.setattr(serve.ui, "info", lambda *a, **k: None)
    monkeypatch.setattr(serve.ui, "ok", lambda *a, **k: None)

    class Stop(Exception):
        pass

    import uvicorn

    monkeypatch.setattr(uvicorn, "Server", lambda *a, **k: (_ for _ in ()).throw(Stop()))

    with pytest.raises(Stop):
        serve.main([
            "--host", "0.0.0.0", "--source", "synthetic", "--detector", "sim",
            "--db", str(tmp_path / "x.db"), "--no-color",
        ])

    auth = [w for w in said if "NO authentication" in w]
    assert auth, "exposing the API to the network must warn that it is unauthenticated"
    assert "127.0.0.1" in auth[0], "the warning must say how to undo it"


def test_serving_on_loopback_stays_quiet(monkeypatch, tmp_path):
    """A warning that fires on the normal path is a warning people stop reading."""
    said: list[str] = []
    monkeypatch.setattr(serve.ui, "warn", said.append)
    monkeypatch.setattr(serve.ui, "banner", lambda *a, **k: None)
    monkeypatch.setattr(serve.ui, "info", lambda *a, **k: None)
    monkeypatch.setattr(serve.ui, "ok", lambda *a, **k: None)

    class Stop(Exception):
        pass

    import uvicorn

    monkeypatch.setattr(uvicorn, "Server", lambda *a, **k: (_ for _ in ()).throw(Stop()))

    with pytest.raises(Stop):
        serve.main([
            "--host", "127.0.0.1", "--source", "synthetic", "--detector", "sim",
            "--db", str(tmp_path / "y.db"), "--no-color",
        ])

    assert not [w for w in said if "NO authentication" in w]
