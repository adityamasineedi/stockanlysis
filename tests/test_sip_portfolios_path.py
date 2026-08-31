"""SIP portfolio config path resolution."""

from stockbot.config import resolve_sip_portfolios_path


def test_resolve_prefers_volume_override(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled" / "sip_portfolios.json"
    volume = tmp_path / "volume" / "sip_portfolios.json"
    bundled.parent.mkdir(parents=True)
    volume.parent.mkdir(parents=True)
    bundled.write_text("{}", encoding="utf-8")
    volume.write_text('{"version": 1}', encoding="utf-8")

    monkeypatch.setattr("stockbot.config.SIP_PORTFOLIOS_BUNDLED_PATH", bundled)
    monkeypatch.setattr("stockbot.config.SIP_PORTFOLIOS_VOLUME_PATH", volume)

    assert resolve_sip_portfolios_path() == volume


def test_resolve_falls_back_to_bundled_when_volume_missing(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled" / "sip_portfolios.json"
    volume = tmp_path / "volume" / "sip_portfolios.json"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("stockbot.config.SIP_PORTFOLIOS_BUNDLED_PATH", bundled)
    monkeypatch.setattr("stockbot.config.SIP_PORTFOLIOS_VOLUME_PATH", volume)

    assert resolve_sip_portfolios_path() == bundled


def test_resolve_explicit_path(tmp_path):
    explicit = tmp_path / "custom.json"
    explicit.write_text("{}", encoding="utf-8")
    assert resolve_sip_portfolios_path(explicit) == explicit


def test_repo_sip_portfolios_json_exists():
    from stockbot.config import PROJECT_ROOT

    assert (PROJECT_ROOT / "data" / "portfolio" / "sip_portfolios.json").exists()
