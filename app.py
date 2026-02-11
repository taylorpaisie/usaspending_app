# file: app.py
"""
USAspending Dash App - Interactive exploration of US federal spending data.

Features:
  - Search spending by keyword, agency, award type, state, and date range
  - Visualize top recipients and spending trends over time
  - Browse individual awards with filtering and sorting
  - Caching and configurable settings via environment variables

Setup:
  python -m venv .venv
  source .venv/bin/activate   # (Windows: .venv\\Scripts\\activate)
  pip install -r requirements.txt
  cp .env.example .env          # Optional: customize settings
  python app.py
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
from cachetools import TTLCache
from dash import Dash, Input, Output, State, callback, callback_context, dash_table, dcc, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from config import CONFIG

# Configure logging
log_level = getattr(logging, CONFIG["log_level"], logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API and Cache Configuration
USASPENDING_BASE = CONFIG["api_base_url"]
HTTP_TIMEOUT = CONFIG["api_timeout"]
CACHE = TTLCache(maxsize=CONFIG["cache_max_size"], ttl=CONFIG["cache_ttl_minutes"] * 60)


@dataclass(frozen=True)
class Filters:
    """
    Filter criteria for spending data queries.
    
    Attributes:
        keyword: Search keyword (e.g., "cybersecurity")
        agencies: Tuple of awarding agency IDs
        award_types: Tuple of award type codes (A=Contracts, B=Grants, etc.)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        place_state: Optional 2-letter state code for place of performance
        top_n: Number of top recipients to display
    """
    keyword: str
    agencies: Tuple[int, ...]
    award_types: Tuple[str, ...]
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    place_state: Optional[str]
    top_n: int
    
    def __post_init__(self) -> None:
        """Validate filter values after initialization."""
        # Validate date format
        try:
            dt.datetime.strptime(self.start_date, "%Y-%m-%d")
            dt.datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Expected YYYY-MM-DD: {e}")
        
        # Validate date range
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before or equal to end date")
        
        # Validate top_n
        if not (5 <= self.top_n <= 100):
            raise ValueError("top_n must be between 5 and 100")
        
        # Validate place_state format
        if self.place_state is not None and len(self.place_state) != 2:
            raise ValueError("place_state must be a 2-letter state code")


def _http_post(path: str, payload: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
    """
    Make an HTTP POST request to the USAspending API with caching.
    
    Args:
        path: API endpoint path (e.g., "/spending_by_category/recipient/")
        payload: Request payload dictionary
        use_cache: Whether to use cached API responses
        
    Returns:
        API response parsed as dictionary
        
    Raises:
        requests.HTTPError: If API request fails
    """
    url = f"{USASPENDING_BASE}{path}"
    cache_key = (url, _freeze(payload))
    
    # Check cache first
    if use_cache and cache_key in CACHE:
        logger.debug(f"Cache hit for {path}")
        return CACHE[cache_key]
    
    logger.debug(f"Fetching {path}")
    try:
        resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if use_cache:
            CACHE[cache_key] = data
            logger.debug(f"Successfully cached response for {path}")
        return data
    except requests.exceptions.Timeout:
        logger.error(f"Request timeout for {path} after {HTTP_TIMEOUT}s")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {path}: {e.response.status_code}")
        raise


def _freeze(obj: Any) -> Any:
    """
    Convert mutable objects to immutable equivalents for caching.
    
    Recursively converts dicts to frozensets of items and lists to tuples.
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return tuple(_freeze(x) for x in obj)
    return obj


def _build_filter_payload_base(filters: Filters) -> Dict[str, Any]:
    """
    Build the base filter section for API requests.
    
    Args:
        filters: Filters object containing query criteria
        
    Returns:
        Dictionary with time_period, award_types, keywords, agencies, and location filters
    """
    payload_filters: Dict[str, Any] = {
        "time_period": [{"start_date": filters.start_date, "end_date": filters.end_date}],
        "award_type_codes": list(filters.award_types) if filters.award_types else [],
        "keywords": [filters.keyword] if filters.keyword else [],
    }
    
    if filters.agencies:
        payload_filters["agencies"] = [
            {"type": "awarding", "tier": "toptier", "id": int(a)} for a in filters.agencies
        ]
    
    if filters.place_state:
        payload_filters["place_of_performance_locations"] = [
            {"country": "USA", "state": filters.place_state}
        ]
    
    return payload_filters


def _default_dates() -> Tuple[str, str]:
    """
    Get default date range (last N days where N is configurable).
    
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=CONFIG["date_range_days"])
    logger.debug(f"Using default date range: {start} to {end}")
    return start.isoformat(), end.isoformat()


def _award_type_options() -> List[Dict[str, str]]:
    """
    Get available award type options.
    
    Returns:
        List of dicts with 'label' and 'value' keys
    """
    return [
        {"label": "Contracts", "value": "A"},
        {"label": "Grants", "value": "B"},
        {"label": "Direct Payments", "value": "C"},
        {"label": "Loans", "value": "D"},
        {"label": "Insurance", "value": "E"},
        {"label": "Other", "value": "F"},
    ]


def _spending_by_recipient(filters: Filters, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch spending aggregated by recipient.
    
    Uses the /spending_by_category/recipient endpoint to get the top recipients
    for the given filter criteria.
    
    Args:
        filters: Filters object containing query criteria
        
    Returns:
        DataFrame with columns: recipient_name, obligations, award_count
    """
    logger.info(f"Fetching spending by recipient with filters: {filters}")
    
    payload: Dict[str, Any] = {
        "category": "recipient",
        "limit": max(10, min(100, filters.top_n)),
        "page": 1,
        "filters": _build_filter_payload_base(filters),
    }

    try:
        data = _http_post("/spending_by_category/recipient/", payload, use_cache=use_cache)
    except requests.HTTPError as e:
        logger.error(f"Failed to fetch recipient spending: {e}")
        return pd.DataFrame()
    
    results = data.get("results", []) or []
    df = pd.DataFrame(results)
    
    if df.empty:
        logger.warning("No results returned for recipient spending query")
        return df

    # Normalize expected columns
    rename = {
        "name": "recipient_name",
        "amount": "obligations",
        "award_count": "award_count",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ["obligations", "award_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    logger.debug(f"Recipient spending query returned {len(df)} rows")
    return df


def _spending_over_time(filters: Filters, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch spending aggregated by time period (monthly).
    
    Uses the /spending_over_time endpoint to get spending trends.
    
    Args:
        filters: Filters object containing query criteria
        
    Returns:
        DataFrame with columns: time_period, obligations
    """
    logger.info("Fetching spending over time")
    
    payload: Dict[str, Any] = {
        "group": "month",
        "filters": _build_filter_payload_base(filters),
    }

    try:
        data = _http_post("/spending_over_time/", payload, use_cache=use_cache)
    except requests.HTTPError as e:
        logger.error(f"Failed to fetch spending over time: {e}")
        return pd.DataFrame()
    
    results = data.get("results", []) or []
    df = pd.DataFrame(results)
    
    if df.empty:
        logger.warning("No results returned for spending over time query")
        return df

    # Parse time_period and extract obligations amount
    if "time_period" in df.columns:
        df["time_period"] = pd.to_datetime(df["time_period"], errors="coerce")
    
    amt_col = "aggregated_amount" if "aggregated_amount" in df.columns else "amount"
    if amt_col in df.columns:
        df["obligations"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0)
    
    result = df[["time_period", "obligations"]].dropna()
    logger.debug(f"Time series query returned {len(result)} data points")
    return result


def _awards_table(
    filters: Filters,
    limit: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch individual awards matching the filter criteria.
    
    Uses the /awards endpoint to get award-level details for display in a table.
    
    Args:
        filters: Filters object containing query criteria
        limit: Maximum number of awards to fetch (default from config)
        
    Returns:
        DataFrame with award details including recipient, agency, amount, date
    """
    if limit is None:
        limit = CONFIG["max_awards_table_rows"]
    
    logger.info(f"Fetching awards with limit={limit}")
    
    payload: Dict[str, Any] = {
        "fields": [
            "Award ID",
            "Recipient Name",
            "Awarding Agency",
            "Award Amount",
            "Place of Performance State Code",
            "Action Date",
            "Award Type",
            "Description",
        ],
        "filters": _build_filter_payload_base(filters),
        "page": 1,
        "limit": max(10, min(100, limit)),
        "sort": "Award Amount",
        "order": "desc",
    }

    try:
        data = _http_post("/awards/", payload, use_cache=use_cache)
    except requests.HTTPError as e:
        logger.error(f"Failed to fetch awards: {e}")
        return pd.DataFrame()
    
    results = data.get("results", []) or []
    df = pd.DataFrame(results)
    
    if df.empty:
        logger.warning("No awards returned")
        return df

    # Standardize column names
    df = df.rename(
        columns={
            "Award ID": "award_id",
            "Recipient Name": "recipient_name",
            "Awarding Agency": "awarding_agency",
            "Award Amount": "award_amount",
            "Place of Performance State Code": "pop_state",
            "Action Date": "action_date",
            "Award Type": "award_type",
            "Description": "description",
        }
    )
    
    # Convert numeric and date columns
    if "award_amount" in df.columns:
        df["award_amount"] = pd.to_numeric(df["award_amount"], errors="coerce").fillna(0)
    if "action_date" in df.columns:
        df["action_date"] = pd.to_datetime(df["action_date"], errors="coerce")
    
    logger.debug(f"Awards query returned {len(df)} rows")
    return df


def _kpis(recipient_df: pd.DataFrame, awards_df: pd.DataFrame) -> Tuple[float, int]:
    """
    Calculate Key Performance Indicators from dataframes.
    
    Args:
        recipient_df: DataFrame from _spending_by_recipient
        awards_df: DataFrame from _awards_table
        
    Returns:
        Tuple of (total_obligations, award_count)
    """
    total = 0.0
    if not recipient_df.empty and "obligations" in recipient_df.columns:
        total = float(recipient_df["obligations"].sum())
    
    count = 0
    if not awards_df.empty:
        count = int(len(awards_df))
    
    logger.debug(f"KPIs: total=${total:.0f}, count={count}")
    return total, count


def _format_money(x: float) -> str:
    """
    Format a number as currency string.
    
    Args:
        x: Numeric value to format
        
    Returns:
        Formatted string like "$1,234,567"
    """
    return f"${x:,.0f}"


def make_app() -> Dash:
    """
    Create and configure the Dash application.
    
    Returns:
        Configured Dash app instance
    """
    logger.info("Initializing USAspending Explorer app")
    start_date, end_date = _default_dates()

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "USAspending Explorer"

    app.layout = html.Div(
        dbc.Container(
            [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div([
                            html.H1("🏛️ USAspending Explorer", className="my-4"),
                            html.P(
                                "Search and visualize U.S. federal spending data from usaspending.gov",
                                className="text-muted mb-3"
                            ),
                        ]),
                        md=9,
                    ),
                    dbc.Col(
                        dbc.Switch(
                            id="dark_mode",
                            label="Dark theme",
                            value=False,
                            className="d-flex justify-content-md-end mt-4",
                        ),
                        md=3,
                    ),
                ],
                className="align-items-start",
            ),
            dbc.Alert(
                "💡 Tip: Start broad (no keyword), then refine with award types, agencies, and state.",
                color="info",
                className="mb-4",
                dismissable=True,
            ),
            dbc.Row(
                [
                    # Left column: Filters
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("🔍 Search Filters", className="card-title mb-4"),
                                    dbc.Label("Keyword", className="fw-bold"),
                                    dbc.Input(
                                        id="keyword",
                                        placeholder="e.g., cybersecurity, roads, NASA",
                                        value="",
                                        type="text",
                                        className="mb-3",
                                    ),
                                    dbc.Label("Awarding Agency IDs (comma-separated)", className="fw-bold"),
                                    dbc.Input(
                                        id="agency_ids",
                                        placeholder="e.g., 114 (DoD), 473 (GSA)",
                                        value="",
                                        type="text",
                                        className="mb-3",
                                    ),
                                    html.Small(
                                        "Find agency IDs on usaspending.gov",
                                        className="text-muted d-block mb-3"
                                    ),
                                    dbc.Label("Award Types", className="fw-bold"),
                                    dcc.Dropdown(
                                        id="award_types",
                                        options=_award_type_options(),
                                        value=["A", "B"],
                                        multi=True,
                                        placeholder="Select award types",
                                        className="mb-3",
                                    ),
                                    dbc.Label("Place of Performance State", className="fw-bold"),
                                    dbc.Input(
                                        id="state",
                                        placeholder="e.g., CA, NY, TX (2-letter code)",
                                        value="",
                                        type="text",
                                        className="mb-3",
                                    ),
                                    dbc.Label("Date Range", className="fw-bold"),
                                    dcc.DatePickerRange(
                                        id="dates",
                                        start_date=start_date,
                                        end_date=end_date,
                                        display_format="YYYY-MM-DD",
                                        className="mb-3",
                                    ),
                                    dbc.Label("Top N Recipients", className="fw-bold"),
                                    dbc.Input(
                                        id="top_n",
                                        type="number",
                                        min=5,
                                        max=100,
                                        step=5,
                                        value=CONFIG["default_top_n"],
                                        className="mb-3",
                                    ),
                                    dbc.Button(
                                        "🔎 Search",
                                        id="run",
                                        color="primary",
                                        className="w-100 mt-2",
                                        size="lg",
                                        n_clicks=0,
                                    ),
                                    dbc.Switch(
                                        id="live_mode",
                                        label="Live mode (auto-refresh from USAspending API)",
                                        value=False,
                                        className="mt-3",
                                    ),
                                    dbc.Label("Auto-refresh interval (seconds)", className="fw-bold mt-3"),
                                    dbc.Input(
                                        id="refresh_seconds",
                                        type="number",
                                        min=15,
                                        step=15,
                                        value=CONFIG["auto_refresh_seconds"],
                                        className="mb-2",
                                    ),
                                    html.Small(
                                        "Live mode bypasses cached responses to request fresh API data.",
                                        className="text-muted d-block",
                                    ),
                                    html.Div(
                                        id="last_updated",
                                        className="text-muted mt-2",
                                        children="Live mode disabled · click Search to refresh",
                                    ),
                                    dcc.Interval(
                                        id="live_interval",
                                        interval=CONFIG["auto_refresh_seconds"] * 1000,
                                        n_intervals=0,
                                        disabled=True,
                                    ),
                                    html.Div(
                                        id="err",
                                        className="text-danger mt-3",
                                        style={"whiteSpace": "pre-wrap"},
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                        ),
                        md=4,
                        className="mb-4",
                    ),
                    # Right column: Results
                    dbc.Col(
                        [
                            # KPI Cards
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.H6("💰 Total Obligations", className="text-muted"),
                                                    html.H3(id="kpi_total", children="$0"),
                                                ]
                                            ),
                                            className="shadow-sm",
                                        ),
                                        sm=6,
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.H6("📋 Awards Shown", className="text-muted"),
                                                    html.H3(id="kpi_count", children="0"),
                                                ]
                                            ),
                                            className="shadow-sm",
                                        ),
                                        sm=6,
                                    ),
                                ],
                                className="mb-4",
                            ),
                            # Charts with loading spinners
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("📊 Top Recipients", className="card-title"),
                                        dcc.Loading(
                                            id="loading-recipients",
                                            type="default",
                                            children=[
                                                dcc.Graph(id="recipients_bar", style={"minHeight": "400px"})
                                            ],
                                        ),
                                    ]
                                ),
                                className="shadow-sm mb-4",
                            ),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("📈 Spending Over Time", className="card-title"),
                                        dcc.Loading(
                                            id="loading-time",
                                            type="default",
                                            children=[
                                                dcc.Graph(id="time_line", style={"minHeight": "400px"})
                                            ],
                                        ),
                                    ]
                                ),
                                className="shadow-sm mb-4",
                            ),
                            # Awards table with loading spinner
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("🏆 Award Details", className="card-title"),
                                        html.Small(
                                            "Click column headers to sort, use search boxes to filter",
                                            className="text-muted d-block mb-3"
                                        ),
                                        dcc.Loading(
                                            id="loading-table",
                                            type="default",
                                            children=[
                                                dash_table.DataTable(
                                                    id="awards_table",
                                                    page_size=10,
                                                    sort_action="native",
                                                    filter_action="native",
                                                    style_table={"overflow": "auto"},
                                                    style_cell={
                                                        "textAlign": "left",
                                                        "padding": "10px",
                                                        "minWidth": "100px",
                                                        "maxWidth": "300px",
                                                        "whiteSpace": "normal",
                                                    },
                                                    style_header={
                                                        "backgroundColor": "rgb(230, 230, 230)",
                                                        "fontWeight": "bold",
                                                    },
                                                    style_data_conditional=[
                                                        {
                                                            "if": {"row_index": "odd"},
                                                            "backgroundColor": "rgb(248, 248, 248)",
                                                        }
                                                    ],
                                                )
                                            ],
                                        ),
                                    ]
                                ),
                                className="shadow-sm",
                            ),
                        ],
                        md=8,
                    ),
                ]
            ),
            html.Hr(className="my-5"),
            html.Footer(
                html.Small(
                    [
                        "Data from ",
                        html.A(
                            "USAspending.gov",
                            href="https://www.usaspending.gov/",
                            target="_blank",
                        ),
                        " | Built with ",
                        html.A(
                            "Dash",
                            href="https://dash.plotly.com/",
                            target="_blank",
                        ),
                    ],
                    className="text-muted",
                ),
                className="text-center pb-3",
            ),
            ],
            fluid=True,
            className="min-vh-100 py-2",
        ),
        id="page-root",
        className="min-vh-100 bg-body text-body",
        **{"data-bs-theme": "light"},
    )

    @callback(
        Output("page-root", "data-bs-theme"),
        Input("dark_mode", "value"),
    )
    def toggle_theme(dark_mode: bool) -> str:
        """Toggle app color mode between light and dark themes."""
        return "dark" if dark_mode else "light"

    @callback(
        Output("live_interval", "disabled"),
        Output("live_interval", "interval"),
        Output("last_updated", "children", allow_duplicate=True),
        Input("live_mode", "value"),
        Input("refresh_seconds", "value"),
        prevent_initial_call=True,
    )
    def configure_live_mode(live_mode: bool, refresh_seconds: Any) -> Tuple[bool, int, str]:
        """Configure polling behavior for live mode updates."""
        try:
            refresh_seconds_int = int(refresh_seconds or CONFIG["auto_refresh_seconds"])
        except (TypeError, ValueError):
            refresh_seconds_int = CONFIG["auto_refresh_seconds"]

        refresh_seconds_int = max(15, refresh_seconds_int)
        interval_ms = refresh_seconds_int * 1000

        if live_mode:
            return (
                False,
                interval_ms,
                f"Live mode enabled · auto-refresh every {refresh_seconds_int}s",
            )

        return True, interval_ms, "Live mode disabled · click Search to refresh"

    @callback(
        Output("recipients_bar", "figure"),
        Output("time_line", "figure"),
        Output("awards_table", "data"),
        Output("awards_table", "columns"),
        Output("kpi_total", "children"),
        Output("kpi_count", "children"),
        Output("err", "children"),
        Output("last_updated", "children", allow_duplicate=True),
        Input("run", "n_clicks"),
        Input("live_interval", "n_intervals"),
        State("keyword", "value"),
        State("agency_ids", "value"),
        State("award_types", "value"),
        State("dates", "start_date"),
        State("dates", "end_date"),
        State("state", "value"),
        State("top_n", "value"),
        State("live_mode", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def refresh(
        _n: int,
        _live_tick: int,
        keyword: Optional[str],
        agency_ids: Optional[str],
        award_types: Optional[List[str]],
        start_date_in: Optional[str],
        end_date_in: Optional[str],
        state: Optional[str],
        top_n: Any,
        live_mode: bool,
    ) -> Tuple[Any, Any, List[Dict], List[Dict], str, str, str, str]:
        """
        Callback to refresh all visualizations based on filter inputs.
        
        Fetches data from the USAspending API and updates all outputs.
        """
        logger.info(f"Refresh triggered: keyword={keyword}, n_clicks={_n}")

        trigger_id = callback_context.triggered_id
        if trigger_id == "live_interval" and not live_mode:
            raise PreventUpdate

        use_cache = trigger_id != "live_interval"

        try:
            # Parse and validate input: agencies
            agencies = tuple(
                int(x.strip())
                for x in (agency_ids or "").split(",")
                if x.strip().isdigit()
            )
            
            # Parse and validate input: state
            state2 = (state or "").strip().upper() or None
            if state2 and len(state2) != 2:
                state2 = None
            
            # Parse and validate input: top_n
            top_n_int = int(top_n or CONFIG["default_top_n"])

            # Create and validate filters
            try:
                f = Filters(
                    keyword=(keyword or "").strip(),
                    agencies=agencies,
                    award_types=tuple(award_types or ()),
                    start_date=str(start_date_in)[:10],
                    end_date=str(end_date_in)[:10],
                    place_state=state2,
                    top_n=top_n_int,
                )
            except ValueError as e:
                error_msg = f"Invalid filter values:\n{str(e)}"
                logger.warning(error_msg)
                return (
                    px.bar(title="Invalid Filters"),
                    px.line(title="Invalid Filters"),
                    [],
                    [],
                    "—",
                    "—",
                    error_msg,
                    "Last update failed",
                )

            # Fetch data
            logger.info("Fetching data from API")
            recipients_df = _spending_by_recipient(f, use_cache=use_cache)
            time_df = _spending_over_time(f, use_cache=use_cache)
            awards_df = _awards_table(f, use_cache=use_cache)

            # Build recipients bar chart
            if recipients_df.empty:
                bar_fig = px.bar(title="No recipient results for these filters.")
                bar_fig.add_annotation(
                    text="Try broader search criteria",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font={"size": 14, "color": "gray"},
                )
            else:
                top = recipients_df.sort_values("obligations", ascending=False).head(f.top_n)
                bar_fig = px.bar(
                    top,
                    x="obligations",
                    y="recipient_name",
                    orientation="h",
                    title=f"Top {min(f.top_n, len(top))} recipients by obligations",
                    labels={"obligations": "Obligations ($)", "recipient_name": "Recipient"},
                )
                bar_fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    hovermode="x unified",
                    showlegend=False,
                )
                bar_fig.update_xaxes(title_text="Obligations ($)")

            # Build spending over time chart
            if time_df.empty:
                line_fig = px.line(title="No time series data for these filters.")
                line_fig.add_annotation(
                    text="Try broader search criteria",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font={"size": 14, "color": "gray"},
                )
            else:
                line_fig = px.line(
                    time_df.sort_values("time_period"),
                    x="time_period",
                    y="obligations",
                    title="Total obligations by month",
                    labels={"time_period": "Month", "obligations": "Obligations ($)"},
                )
                line_fig.update_layout(hovermode="x unified")
                line_fig.update_xaxes(title_text="Month")

            # Build awards table
            if awards_df.empty:
                data = []
                columns = [{"name": "No award rows returned", "id": "empty"}]
            else:
                display_df = awards_df.copy()
                # Format dates for display
                if "action_date" in display_df.columns:
                    display_df["action_date"] = display_df["action_date"].dt.strftime("%Y-%m-%d")
                # Format currency
                if "award_amount" in display_df.columns:
                    display_df["award_amount"] = display_df["award_amount"].apply(
                        lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
                    )
                data = display_df.to_dict("records")
                columns = [{"name": c.replace("_", " ").title(), "id": c} for c in display_df.columns]

            # Calculate KPIs
            total, count = _kpis(recipients_df, awards_df)
            
            logger.info(f"Refresh completed: {count} awards, ${total:,.0f} total obligations")
            return (
                bar_fig,
                line_fig,
                data,
                columns,
                _format_money(total),
                str(count),
                "",  # Clear error message
                f"Last updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )

        except requests.exceptions.Timeout:
            error_msg = "Request timed out. The API is slow to respond. Try simpler filters."
            logger.error(error_msg)
            return (
                px.bar(title="Request Timeout"),
                px.line(title="Request Timeout"),
                [],
                [],
                "—",
                "—",
                error_msg,
                "Last update failed",
            )
        except requests.HTTPError as e:
            error_msg = f"API Error {e.response.status_code}: {str(e)}\nCheck your filters and try again."
            logger.error(error_msg)
            return (
                px.bar(title="API Error"),
                px.line(title="API Error"),
                [],
                [],
                "—",
                "—",
                error_msg,
                "Last update failed",
            )
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}\n\nPlease check your filters and try again."
            logger.exception("Unexpected error in refresh callback")
            return (
                px.bar(title="Error"),
                px.line(title="Error"),
                [],
                [],
                "—",
                "—",
                error_msg,
                "Last update failed",
            )

    logger.info("App initialization complete")
    return app


if __name__ == "__main__":
    logger.info(f"Starting app on {CONFIG['host']}:{CONFIG['port']} (debug={CONFIG['debug']})")
    app = make_app()
    app.run(
        debug=CONFIG["debug"],
        host=CONFIG["host"],
        port=CONFIG["port"],
    )
