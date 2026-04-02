"""
scheduler.py — APScheduler-based persistent scheduler for the stock screener.

Fires the screener twice daily (morning + afternoon) on weekdays when at least
one tracked exchange is open.  Run times are derived dynamically from real market
open/close times (with DST handled by pytz), so they remain accurate year-round.

Run with:
    python scheduler.py

Keeps running as a blocking process; use Ctrl+C to stop.
"""
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/scheduler.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exchange definitions
# Each entry: (display_name, open_local (h,m), close_local (h,m), tz_str, cal_code)
# Calendar codes used by exchange_calendars library.
# ---------------------------------------------------------------------------
EXCHANGES = [
    ("NYSE",  (9, 30), (16,  0), "America/New_York",  "XNYS"),
    ("LSE",   (8,  0), (16, 30), "Europe/London",     "XLON"),
    ("XETRA", (9,  0), (17, 30), "Europe/Berlin",     "XETR"),
    ("TSX",   (9, 30), (16,  0), "America/Toronto",   "XTSX"),
    ("ASX",   (10, 0), (16,  0), "Australia/Sydney",  "XASX"),
    ("TSE",   (9,  0), (15, 30), "Asia/Tokyo",        "XTKS"),
    ("HKEX",  (9, 30), (16,  0), "Asia/Hong_Kong",    "XHKG"),
]


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _local_to_utc(hour: int, minute: int, tz_str: str, ref_date: date) -> datetime:
    """
    Convert a wall-clock time on ref_date in the given timezone to UTC.

    Parameters
    ----------
    hour, minute : int
        Local time of day.
    tz_str : str
        pytz timezone name, e.g. "America/New_York".
    ref_date : date
        The calendar date for conversion.

    Returns
    -------
    datetime
        Timezone-aware UTC datetime.
    """
    tz       = pytz.timezone(tz_str)
    local_dt = tz.localize(
        datetime(ref_date.year, ref_date.month, ref_date.day, hour, minute)
    )
    return local_dt.astimezone(pytz.utc)


def _get_schedule_times(ref_date: date) -> tuple:
    """
    Compute the morning, afternoon, and price-fetch run times (UTC) for ref_date.

    morning      = earliest UTC market open  + 1 hour
    afternoon    = earliest UTC market close - 1 hour
    price_fetch  = latest UTC market close   + PRICE_FETCH_DELAY_MINUTES

    Parameters
    ----------
    ref_date : date
        The date for which to compute schedule times.

    Returns
    -------
    tuple[datetime, datetime, datetime]
        (morning_utc, afternoon_utc, price_fetch_utc) — timezone-aware UTC datetimes.
    """
    from config import PRICE_FETCH_DELAY_MINUTES

    opens  = []
    closes = []
    for name, (oh, om), (ch, cm), tz_str, _ in EXCHANGES:
        opens.append(_local_to_utc(oh, om, tz_str, ref_date))
        closes.append(_local_to_utc(ch, cm, tz_str, ref_date))

    earliest_open   = min(opens)
    earliest_close  = min(closes)
    latest_close    = max(closes)

    morning      = earliest_open  + timedelta(hours=1)
    afternoon    = earliest_close - timedelta(hours=1)
    price_fetch  = latest_close   + timedelta(minutes=PRICE_FETCH_DELAY_MINUTES)

    return morning, afternoon, price_fetch


# ---------------------------------------------------------------------------
# Holiday / weekend checks
# ---------------------------------------------------------------------------

def _all_markets_closed(ref_date: date) -> bool:
    """
    Return True only when every tracked exchange is closed on ref_date.

    Uses the exchange_calendars library for holiday data.  If the library is
    not installed the function logs a warning and returns False (assumes open).

    Parameters
    ----------
    ref_date : date
        The date to check.

    Returns
    -------
    bool
        True if all markets are closed; False if at least one is open.
    """
    try:
        import exchange_calendars as xcals
    except ImportError:
        logger.warning(
            "exchange_calendars not installed — holiday check skipped. "
            "Install with: pip install exchange-calendars"
        )
        return False

    date_str = ref_date.isoformat()
    for name, _, _, _, cal_code in EXCHANGES:
        try:
            cal = xcals.get_calendar(cal_code)
            if cal.is_session(date_str):
                logger.debug(f"{name} ({cal_code}) is open on {date_str}.")
                return False   # at least one exchange is open
        except Exception as exc:
            logger.warning(f"Could not check calendar {cal_code}: {exc}")
            # If we can't verify, assume open → do not block the run
            return False

    logger.info(f"All tracked exchanges are closed on {date_str} (holiday).")
    return True


def _should_run(ref_date: date) -> bool:
    """
    Return True if the screener should run on ref_date.

    Skips Saturday (5) and Sunday (6) in UTC, and skips days when every
    tracked exchange is closed (holidays).

    Parameters
    ----------
    ref_date : date
        The UTC calendar date to evaluate.

    Returns
    -------
    bool
    """
    # Skip weekends
    if ref_date.weekday() >= 5:
        logger.info(f"{ref_date} is a weekend — skipping screener run.")
        return False

    # Skip full-market holidays
    if _all_markets_closed(ref_date):
        logger.info(f"{ref_date} — all markets closed (holiday) — skipping screener run.")
        return False

    return True


# ---------------------------------------------------------------------------
# Job functions called by the scheduler
# ---------------------------------------------------------------------------

def _log_market_regime() -> None:
    """Log the current SPY 200-day MA market regime to the scheduler log."""
    try:
        from backtest import is_risk_on, fetch_spy_prices
        spy_series = fetch_spy_prices()
        risk_on = is_risk_on(spy_series, date.today())
        regime_label = "RISK-ON" if risk_on else "RISK-OFF"
        weighting_scheme = os.getenv("WEIGHTING_SCHEME", "risk_parity")
        regime_filter_on = os.getenv("REGIME_FILTER", "true").lower() == "true"
        logger.info(
            f"Market Regime: {regime_label}  |  "
            f"Weighting: {weighting_scheme}  |  "
            f"Regime filter: {'ON' if regime_filter_on else 'OFF'}"
        )
    except Exception as exc:
        logger.warning(f"Could not determine market regime: {exc}")


def run_morning() -> None:
    """
    Execute the morning screener run.

    Checks _should_run() before delegating to main.run_screener().
    """
    today = date.today()
    if not _should_run(today):
        return
    logger.info(f"Firing MORNING run for {today}.")
    _log_market_regime()
    try:
        from main import run_screener
        run_screener(run_type="morning")
    except Exception as exc:
        logger.error(f"Morning run failed: {exc}", exc_info=True)


def run_afternoon() -> None:
    """
    Execute the afternoon screener run.

    Checks _should_run() before delegating to main.run_screener().
    """
    today = date.today()
    if not _should_run(today):
        return
    logger.info(f"Firing AFTERNOON run for {today}.")
    _log_market_regime()
    try:
        from main import run_screener
        run_screener(run_type="afternoon")
    except Exception as exc:
        logger.error(f"Afternoon run failed: {exc}", exc_info=True)


def run_price_fetch_job() -> None:
    """
    Execute the daily price fetch after all markets have closed.

    Checks _should_run() before delegating to price_tracker.run_price_fetch().
    """
    today = date.today()
    if not _should_run(today):
        return
    logger.info(f"Firing PRICE FETCH for {today}.")
    try:
        from price_tracker import run_price_fetch
        run_price_fetch()
    except Exception as exc:
        logger.error(f"Price fetch failed: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Stateful tracker — prevents double-firing within the same day
# ---------------------------------------------------------------------------
_morning_fired_today:     str = ""
_afternoon_fired_today:   str = ""
_price_fetched_today:     str = ""


def _check_and_fire() -> None:
    """
    Called every minute by the interval job.

    Fires morning, afternoon, and price-fetch runs when the current UTC time
    falls within a 2-minute window starting at the dynamically computed run
    time.  Guards against double-firing by storing the date string of the
    last fire for each job.
    """
    global _morning_fired_today, _afternoon_fired_today, _price_fetched_today

    now_utc   = datetime.now(timezone.utc)
    today     = now_utc.date()
    today_str = today.isoformat()

    if not _should_run(today):
        return

    morning_utc, afternoon_utc, price_fetch_utc = _get_schedule_times(today)

    # Fire morning run if within [morning_utc, morning_utc + 2 min) and not yet fired
    if (
        _morning_fired_today != today_str
        and morning_utc <= now_utc < morning_utc + timedelta(minutes=2)
    ):
        logger.info(f"Morning window reached at {now_utc.strftime('%H:%M UTC')}.")
        _morning_fired_today = today_str
        run_morning()

    # Fire afternoon run if within [afternoon_utc, afternoon_utc + 2 min) and not yet fired
    if (
        _afternoon_fired_today != today_str
        and afternoon_utc <= now_utc < afternoon_utc + timedelta(minutes=2)
    ):
        logger.info(f"Afternoon window reached at {now_utc.strftime('%H:%M UTC')}.")
        _afternoon_fired_today = today_str
        run_afternoon()

    # Fire price fetch if within [price_fetch_utc, price_fetch_utc + 2 min) and not yet fired
    if (
        _price_fetched_today != today_str
        and price_fetch_utc <= now_utc < price_fetch_utc + timedelta(minutes=2)
    ):
        logger.info(f"Price fetch window reached at {now_utc.strftime('%H:%M UTC')}.")
        _price_fetched_today = today_str
        run_price_fetch_job()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _write_pid() -> None:
    """Write current process PID to logs/scheduler.pid for stop_scheduler.bat."""
    pid_path = os.path.join("logs", "scheduler.pid")
    try:
        with open(pid_path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        logger.info(f"PID {os.getpid()} written to {pid_path}")
    except Exception as exc:
        logger.warning(f"Could not write PID file: {exc}")


def _remove_pid() -> None:
    """Remove the PID file on clean shutdown."""
    pid_path = os.path.join("logs", "scheduler.pid")
    try:
        if os.path.exists(pid_path):
            os.remove(pid_path)
    except Exception:
        pass


def main() -> None:
    """
    Start the blocking scheduler.

    Logs today's computed run times (morning, afternoon, price fetch), then
    starts an interval job that checks every minute whether it is time to
    fire a screener or price-fetch run.
    """
    today = date.today()
    morning_utc, afternoon_utc, price_fetch_utc = _get_schedule_times(today)

    m_str = morning_utc.strftime("%H:%M UTC")
    a_str = afternoon_utc.strftime("%H:%M UTC")
    p_str = price_fetch_utc.strftime("%H:%M UTC")

    logger.info("=" * 60)
    logger.info("Hedge Fund Stock Screener — Scheduler Starting")
    logger.info(f"Today ({today}) schedule:")
    logger.info(f"  Morning run:    {m_str}")
    logger.info(f"  Afternoon run:  {a_str}")
    logger.info(f"  Price fetch:    {p_str}")
    logger.info("(Times are dynamic — recalculated daily based on market open/close + DST)")
    logger.info("=" * 60)

    # Write PID file so stop_scheduler.bat can find this process
    _write_pid()

    # Send Slack startup notification (gated by ENABLE_SLACK_ALERTS)
    try:
        from alerts import send_slack_startup
        send_slack_startup(
            morning_utc=m_str,
            afternoon_utc=a_str,
            price_fetch_utc=p_str,
            today=today.isoformat(),
        )
    except Exception as exc:
        logger.warning(f"Slack startup notification failed: {exc}")

    scheduler = BlockingScheduler(timezone="UTC")

    # Every-minute heartbeat that decides when to actually fire
    scheduler.add_job(
        _check_and_fire,
        trigger="interval",
        minutes=1,
        id="time_checker",
        name="Market Time Checker",
        replace_existing=True,
        max_instances=1,
    )

    logger.info("Scheduler started. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")
    finally:
        _remove_pid()


if __name__ == "__main__":
    main()
