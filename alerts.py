"""
alerts.py — Slack webhook notifications after each screener run.

Sends a formatted summary message to a Slack channel via an incoming webhook URL.
Alerts are gated by ENABLE_SLACK_ALERTS=true in the environment / .env file.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import requests

from config import SLACK_WEBHOOK_URL, ENABLE_SLACK_ALERTS

logger = logging.getLogger(__name__)


def send_slack_startup(
    morning_utc: str,
    afternoon_utc: str,
    price_fetch_utc: str,
    today: str,
) -> None:
    """
    Send a Slack notification confirming the scheduler started, with today's run times.

    Parameters
    ----------
    morning_utc, afternoon_utc, price_fetch_utc : str
        Formatted run times, e.g. "10:30 UTC".
    today : str
        Date string "YYYY-MM-DD".
    """
    if not ENABLE_SLACK_ALERTS:
        return
    if not SLACK_WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping startup Slack alert.")
        return

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "*Hedge Fund Screener — Scheduler Started*",
        f"_{now_utc}_",
        "",
        f"*Today's Schedule ({today}):*",
        f"  Morning run:   `{morning_utc}`",
        f"  Afternoon run: `{afternoon_utc}`",
        f"  Price fetch:   `{price_fetch_utc}`",
        "",
        "_Times are UTC, recalculated daily to account for DST_",
    ]
    payload = {"text": "\n".join(lines)}
    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"Slack startup alert HTTP {resp.status_code}: {resp.text}")
        else:
            logger.info("Slack startup notification sent.")
    except Exception as exc:
        logger.warning(f"Slack startup notification failed: {exc}")


def send_slack_validation_alert(warnings: list) -> None:
    """
    Send a Slack alert when post-run validation finds anomalies.

    Parameters
    ----------
    warnings : list[str]
        List of human-readable validation failure messages.
    """
    if not ENABLE_SLACK_ALERTS or not SLACK_WEBHOOK_URL:
        return
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "*Hedge Fund Screener — VALIDATION WARNING*",
        f"_{now_utc}_",
        "",
        "*Issues detected after run:*",
    ]
    for w in warnings:
        lines.append(f"  [!] {w}")
    payload = {"text": "\n".join(lines)}
    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"Slack validation alert HTTP {resp.status_code}: {resp.text}")
        else:
            logger.info("Slack validation alert sent (%d warnings).", len(warnings))
    except Exception as exc:
        logger.warning(f"Slack validation alert failed: {exc}")


def send_slack_alert(
    run_type: str,
    df,
    top25_changes: dict,
    total_found: int,
    total_skipped: int,
    elapsed_seconds: float,
    sheet_url: Optional[str],
    is_partial: bool = False,
    validation_warnings: Optional[list] = None,
) -> None:
    """
    Send a Slack webhook notification summarising the screener run.

    The alert includes:
    - Run metadata (type, time, stock counts, duration)
    - Top 10 ranked stocks with scores
    - Top-25 changes (entered / exited) vs previous run
    - Big movers (rank change > 10 places)
    - Link to Google Sheet (if available)

    Parameters
    ----------
    run_type : str
        "morning" or "afternoon".
    df : pd.DataFrame
        Fully scored and ranked DataFrame.
    top25_changes : dict
        Output of delta_tracker.get_top25_changes().
    total_found : int
        Number of stocks successfully fetched.
    total_skipped : int
        Number of tickers that were skipped / failed.
    elapsed_seconds : float
        Total wall-clock time of the run.
    sheet_url : str or None
        Google Sheets URL, or None if Sheets upload was skipped.
    is_partial : bool
        True if fewer than PARTIAL_RUN_THRESHOLD stocks were fetched.
    validation_warnings : list[str] or None
        Validation failure messages to append to the alert.
    """
    if not ENABLE_SLACK_ALERTS:
        logger.debug("Slack alerts disabled (ENABLE_SLACK_ALERTS=false) — skipping.")
        return
    if not SLACK_WEBHOOK_URL:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping Slack alert.")
        return

    try:
        _post_alert(
            run_type=run_type,
            df=df,
            top25_changes=top25_changes,
            total_found=total_found,
            total_skipped=total_skipped,
            elapsed_seconds=elapsed_seconds,
            sheet_url=sheet_url,
            is_partial=is_partial,
            validation_warnings=validation_warnings or [],
        )
    except Exception as e:
        logger.error(f"Slack alert failed: {e}")


def _post_alert(
    run_type: str,
    df,
    top25_changes: dict,
    total_found: int,
    total_skipped: int,
    elapsed_seconds: float,
    sheet_url: Optional[str],
    is_partial: bool,
    validation_warnings: list,
) -> None:
    """
    Build and POST the Slack payload.

    Parameters
    ----------
    (same as send_slack_alert)
    """
    now_utc     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    label       = run_type.upper()
    partial_tag = " [PARTIAL RUN]" if is_partial else ""

    lines = [
        f"*Hedge Fund Screener — {label} RUN{partial_tag}*",
        f"_{now_utc} | {total_found:,} stocks ranked | {total_skipped:,} skipped | {elapsed_seconds:.0f}s_",
        "",
        "*Top 10 Stocks:*",
    ]

    top10 = df.head(10)
    for _, row in top10.iterrows():
        try:
            rank  = int(row["Rank"])
            score = float(row["Composite Score"])
            lines.append(
                f"  {rank:>3}. {str(row['Ticker']):<8} Score: {score:,.2f}"
            )
        except (TypeError, ValueError):
            lines.append(f"  — {row.get('Ticker', 'N/A')} (invalid data)")

    # Top-25 changes
    entered = top25_changes.get("entered", [])
    exited  = top25_changes.get("exited",  [])
    big     = top25_changes.get("big_movers", [])

    if entered:
        lines.append("")
        lines.append("*Entered Top 25:* " + ", ".join(f"{t} ↑" for t in entered))
    if exited:
        lines.append("*Exited Top 25:*  " + ", ".join(f"{t} ↓" for t in exited))

    if big:
        lines.append("")
        lines.append("*Big Movers (rank change > 10):*")
        for ticker, delta in big:
            # Rank Delta is positive = moved DOWN (rank number increased = worse)
            # Rank Delta is negative = moved UP   (rank number decreased = better)
            arrow = "↑" if delta < 0 else "↓"
            lines.append(f"  {ticker}: {abs(delta)} places {arrow}")

    if validation_warnings:
        lines.append("")
        lines.append("*Validation Warnings:*")
        for w in validation_warnings:
            lines.append(f"  [!] {w}")

    if sheet_url:
        lines.append("")
        lines.append(f"<{sheet_url}|View Google Sheet>")

    payload = {"text": "\n".join(lines)}

    resp = requests.post(
        SLACK_WEBHOOK_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )

    if resp.status_code != 200:
        logger.warning(f"Slack returned HTTP {resp.status_code}: {resp.text}")
    else:
        logger.info("Slack alert sent successfully.")
