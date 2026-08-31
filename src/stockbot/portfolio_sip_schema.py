"""Portfolio SIP config schema (v1) with legacy v0 loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stockbot.config import resolve_sip_portfolios_path


@dataclass(frozen=True)
class RotationConfig:
    enabled: bool = False
    mode: str = "fixed"
    cycle_months: int = 0
    months_active: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)


@dataclass(frozen=True)
class DipConfig:
    enabled: bool = False
    trigger_type: str = "none"
    reference: str | None = None
    threshold_pct: float = 0.0
    band_min: float = 0.0
    band_max: float = 0.0
    max_topups_per_month: int = 0
    max_topups_per_year: int = 0


@dataclass(frozen=True)
class TopupConfig:
    min_amount: float = 0.0
    max_amount: float = 0.0


@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    exchange: str = "NSE"
    role: str = "core"
    enabled: bool = True
    target_amount_monthly: float = 0.0
    min_amount_monthly: float = 0.0
    max_amount_monthly: float = 0.0
    rotation: RotationConfig = RotationConfig()
    dip: DipConfig = DipConfig()
    topup: TopupConfig = TopupConfig()
    prescan_exempt: bool = False


@dataclass(frozen=True)
class CashPolicy:
    rollover_uninvested: bool = True
    overflow_symbol: str | None = None


@dataclass(frozen=True)
class PortfolioBucket:
    id: str
    label: str
    monthly_budget: float
    symbols: tuple[SymbolConfig, ...]
    enabled: bool = True
    allocation_mode: str = "equal_split"
    thesis: str | None = None
    cash_policy: CashPolicy = CashPolicy()


@dataclass(frozen=True)
class PrescanGateConfig:
    enabled: bool = True
    require_recent_days: int = 90
    skip_when_missing: bool = False
    monthly_auto_prescan: bool = True
    # Intentional portfolio names tagged HOLDING_MONITOR_ONLY by quant prescan
    # are still eligible for phased SIP — prescan means skip /analyze, not skip SIP.
    allow_holding_monitor: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    require_trade_id: bool = False
    allow_partial_fills: bool = True
    month_close_day: str = "last_trading_day"


@dataclass(frozen=True)
class PortfolioSipConfig:
    version: int
    currency: str
    max_monthly_budget: float
    whole_shares_only: bool
    broker_rounding: str
    portfolios: tuple[PortfolioBucket, ...]
    logging: LoggingConfig = LoggingConfig()
    prescan_gate: PrescanGateConfig = PrescanGateConfig()
    default_allocation_mode: str = "equal_split"

    @property
    def total_monthly_budget(self) -> float:
        return self.max_monthly_budget


def _rotation_from_raw(raw: dict[str, Any] | None) -> RotationConfig:
    item = raw or {}
    months = tuple(int(m) for m in item.get("months_active", list(range(1, 13))))
    return RotationConfig(
        enabled=bool(item.get("enabled", False)),
        mode=str(item.get("mode", "fixed")),
        cycle_months=int(item.get("cycle_months", 0) or 0),
        months_active=months,
    )


def _dip_from_raw(raw: dict[str, Any] | None) -> DipConfig:
    item = raw or {}
    return DipConfig(
        enabled=bool(item.get("enabled", False)),
        trigger_type=str(item.get("trigger_type", "none")),
        reference=str(item["reference"]) if item.get("reference") else None,
        threshold_pct=float(item.get("threshold_pct", 0) or 0),
        band_min=float(item.get("band_min", 0) or 0),
        band_max=float(item.get("band_max", 0) or 0),
        max_topups_per_month=int(item.get("max_topups_per_month", 0) or 0),
        max_topups_per_year=int(item.get("max_topups_per_year", 0) or 0),
    )


def _topup_from_raw(raw: dict[str, Any] | None) -> TopupConfig:
    item = raw or {}
    return TopupConfig(
        min_amount=float(item.get("min_amount", 0) or 0),
        max_amount=float(item.get("max_amount", 0) or 0),
    )


def _symbol_from_raw(raw: dict[str, Any]) -> SymbolConfig:
    return SymbolConfig(
        symbol=str(raw["symbol"]).upper(),
        exchange=str(raw.get("exchange", "NSE")),
        role=str(raw.get("role", "core")),
        enabled=bool(raw.get("enabled", True)),
        target_amount_monthly=float(raw.get("target_amount_monthly", 0) or 0),
        min_amount_monthly=float(raw.get("min_amount_monthly", 0) or 0),
        max_amount_monthly=float(raw.get("max_amount_monthly", 0) or 0),
        rotation=_rotation_from_raw(raw.get("rotation")),
        dip=_dip_from_raw(raw.get("dip")),
        topup=_topup_from_raw(raw.get("topup")),
        prescan_exempt=bool(raw.get("prescan_exempt", False)),
    )


def _cash_policy_from_raw(raw: dict[str, Any] | None) -> CashPolicy:
    item = raw or {}
    overflow = item.get("overflow_symbol")
    return CashPolicy(
        rollover_uninvested=bool(item.get("rollover_uninvested", True)),
        overflow_symbol=str(overflow).upper() if overflow else None,
    )


def _bucket_from_v1(raw: dict[str, Any], default_mode: str) -> PortfolioBucket:
    symbols_raw = raw.get("symbols", [])
    if symbols_raw and isinstance(symbols_raw[0], str):
        symbols = tuple(
            SymbolConfig(symbol=str(s).upper(), target_amount_monthly=0.0) for s in symbols_raw
        )
    else:
        symbols = tuple(_symbol_from_raw(item) for item in symbols_raw)
    thesis = raw.get("thesis")
    return PortfolioBucket(
        id=str(raw["id"]),
        label=str(raw["label"]),
        monthly_budget=float(raw["monthly_budget"]),
        symbols=symbols,
        enabled=bool(raw.get("enabled", True)),
        allocation_mode=str(raw.get("allocation_mode", default_mode)),
        thesis=str(thesis).strip() or None if thesis else None,
        cash_policy=_cash_policy_from_raw(raw.get("cash_policy")),
    )


def _prescan_gate_from_raw(raw: dict[str, Any] | None) -> PrescanGateConfig:
    item = raw or {}
    return PrescanGateConfig(
        enabled=bool(item.get("enabled", True)),
        require_recent_days=int(item.get("require_recent_days", 90) or 90),
        skip_when_missing=bool(item.get("skip_when_missing", False)),
        monthly_auto_prescan=bool(item.get("monthly_auto_prescan", True)),
        allow_holding_monitor=bool(item.get("allow_holding_monitor", True)),
    )


def _load_v1(raw: dict[str, Any]) -> PortfolioSipConfig:
    default_mode = str(raw.get("allocation_mode", "equal_split"))
    portfolios = tuple(_bucket_from_v1(item, default_mode) for item in raw.get("portfolios", []))
    logging_raw = raw.get("logging") or {}
    return PortfolioSipConfig(
        version=int(raw.get("version", 1)),
        currency=str(raw.get("currency", "INR")),
        max_monthly_budget=float(raw.get("max_monthly_budget", 0) or 0)
        or sum(p.monthly_budget for p in portfolios if p.enabled),
        whole_shares_only=bool(raw.get("whole_shares_only", True)),
        broker_rounding=str(raw.get("broker_rounding", "floor")),
        portfolios=portfolios,
        logging=LoggingConfig(
            require_trade_id=bool(logging_raw.get("require_trade_id", False)),
            allow_partial_fills=bool(logging_raw.get("allow_partial_fills", True)),
            month_close_day=str(logging_raw.get("month_close_day", "last_trading_day")),
        ),
        prescan_gate=_prescan_gate_from_raw(raw.get("prescan_gate")),
        default_allocation_mode=default_mode,
    )


def load_portfolio_sip_config(path: Path | None = None) -> PortfolioSipConfig:
    target = resolve_sip_portfolios_path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"Portfolio SIP config not found at {target}. "
            "Expected bundled config/portfolio/sip_portfolios.json in the image "
            "or data/portfolio/sip_portfolios.json on the volume."
        )
    raw = json.loads(target.read_text(encoding="utf-8"))
    if raw.get("version") == 1:
        return _load_v1(raw)
    # Legacy v0 — synthesize v1
    default_mode = str(raw.get("allocation_mode", "equal"))
    if default_mode == "priority":
        mode = "priority"
    elif default_mode == "equal":
        mode = "equal_split"
    else:
        mode = default_mode
    portfolios = tuple(
        PortfolioBucket(
            id=str(item["id"]),
            label=str(item["label"]),
            monthly_budget=float(item["monthly_budget"]),
            symbols=tuple(
                SymbolConfig(symbol=str(s).upper()) for s in item.get("symbols", [])
            ),
            allocation_mode=str(item.get("allocation_mode", mode)),
            thesis=str(item["thesis"]).strip() or None if item.get("thesis") else None,
        )
        for item in raw.get("portfolios", [])
    )
    total = float(raw.get("total_monthly_budget", 0) or 0) or sum(
        p.monthly_budget for p in portfolios
    )
    return PortfolioSipConfig(
        version=0,
        currency="INR",
        max_monthly_budget=total,
        whole_shares_only=True,
        broker_rounding="floor",
        portfolios=portfolios,
        default_allocation_mode=mode,
    )


def is_rotation_active(symbol: SymbolConfig, month: int) -> bool:
    """Return whether *month* (1–12) is an active SIP month for this symbol."""
    rot = symbol.rotation
    if not rot.enabled:
        return True
    if rot.months_active:
        return month in rot.months_active
    return True


def symbol_names(bucket: PortfolioBucket) -> tuple[str, ...]:
    return tuple(s.symbol for s in bucket.symbols if s.enabled)
