"""
live_ingestion.py — Phase 1: Live Fund Data Engine
====================================================
Fetches real-time NAV data from AMFI (via mfapi.in) for Indian mutual funds
and price/NAV data from Yahoo Finance for US funds.

Produces DataFrames compatible with the existing preprocessing pipeline in
markets/india/src/data_loader.py and markets/usa/src/data_loader.py.

Usage:
    from components.live_ingestion import search_india_funds, fetch_india_nav, fetch_us_fund
"""

from __future__ import annotations

import time
import requests
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# Indian Fund Search & NAV Ingestion (mfapi.in)
# ─────────────────────────────────────────────

MFAPI_BASE = "https://api.mfapi.in/mf"
SCHEME_LIST_URL = "https://api.mfapi.in/mf"

DEFAULT_START = "2019-01-01"
DEFAULT_END   = "2023-12-31"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_scheme_codes() -> dict[str, str]:
    """
    Downloads the complete list of ~19,000 AMFI scheme codes and names.
    Cached for 1 hour (ttl=3600) so repeated search-bar keystrokes don't hammer the API.
    Returns {scheme_code: scheme_name}.
    """
    try:
        resp = requests.get(SCHEME_LIST_URL, timeout=30)
        resp.raise_for_status()
        records = resp.json()   # list of {"schemeCode": ..., "schemeName": ...}
        return {str(r["schemeCode"]): r["schemeName"] for r in records}
    except Exception as exc:
        # Return empty dict so the UI degrades gracefully instead of crashing.
        return {}


def search_india_funds(query: str, limit: int = 30) -> dict[str, str]:
    """
    Case-insensitive substring search over all AMFI fund names.

    Args:
        query: e.g. "parag parikh" or "hdfc flexi"
        limit: max results to return (avoids flooding the selectbox)

    Returns:
        {scheme_code: scheme_name} for matches
    """
    all_schemes = fetch_all_scheme_codes()
    q = query.strip().lower()
    if not q:
        return {}
    results = {
        code: name
        for code, name in all_schemes.items()
        if q in name.lower()
    }
    # Prioritise Direct-Plan results to surface the investable share class first
    direct = {k: v for k, v in results.items() if "direct" in v.lower()}
    others = {k: v for k, v in results.items() if k not in direct}
    merged = {**direct, **others}
    return dict(list(merged.items())[:limit])


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_india_nav(scheme_code: str,
                    start: str = DEFAULT_START,
                    end: str   = DEFAULT_END) -> pd.DataFrame:
    """
    Fetches historical daily NAVs for one AMFI scheme from mfapi.in.

    Returns a DataFrame with columns [Date (str YYYY-MM-DD), Adj Close (float), symbol (str)],
    sorted ascending by date, filtered to [start, end].

    On failure returns an empty DataFrame (caller must check .empty).
    """
    try:
        url = f"{MFAPI_BASE}/{scheme_code}"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get("data", [])
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # mfapi returns DD-MM-YYYY dates
        df["Date"] = pd.to_datetime(df["date"], format="%d-%m-%Y").dt.strftime("%Y-%m-%d")
        df["Adj Close"] = pd.to_numeric(df["nav"], errors="coerce")
        df["symbol"] = str(scheme_code)

        df = df[["Date", "Adj Close", "symbol"]]
        df = df[(df["Date"] >= start) & (df["Date"] <= end)]
        df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        df = df.dropna(subset=["Adj Close"])
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# US Fund / ETF Ingestion (Yahoo Finance)
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_us_fund(ticker: str,
                  start: str = DEFAULT_START,
                  end: str   = DEFAULT_END) -> pd.DataFrame:
    """
    Fetches historical daily prices for a US mutual fund or ETF via yfinance.

    Returns a DataFrame with columns [Date (str YYYY-MM-DD), Adj Close (float), symbol (str)].

    On failure returns an empty DataFrame (caller must check .empty).
    """
    try:
        import yfinance as yf
        fund = yf.Ticker(ticker.upper())
        hist = fund.history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()

        hist = hist.reset_index()
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")
        hist = hist.rename(columns={"Close": "Adj Close"})
        hist["symbol"] = ticker.upper()
        hist = hist[["Date", "Adj Close", "symbol"]]
        hist = hist.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        hist = hist.dropna(subset=["Adj Close"])
        return hist
    except Exception:
        return pd.DataFrame()


def get_fund_meta_from_api(scheme_code: str) -> dict:
    """
    Fetches fund meta (name, fund house) from mfapi.in.
    Returns a dict with at least 'scheme_name' and 'fund_house'.
    """
    try:
        url = f"{MFAPI_BASE}/{scheme_code}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        meta = payload.get("meta", {})
        return {
            "scheme_name":  meta.get("scheme_name", f"Fund {scheme_code}"),
            "fund_house":   meta.get("fund_house", "Unknown AMC"),
            "scheme_type":  meta.get("scheme_type", ""),
            "scheme_category": meta.get("scheme_category", ""),
            "scheme_code":  scheme_code,
        }
    except Exception:
        return {"scheme_name": f"Fund {scheme_code}", "fund_house": "Unknown", "scheme_code": scheme_code}
