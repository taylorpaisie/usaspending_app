# file: app.py
"""
USAspending Dash App

Run:
  python -m venv .venv
  source .venv/bin/activate   # (Windows: .venv\\Scripts\\activate)
  pip install dash dash-bootstrap-components plotly pandas requests cachetools
  python app.py
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
from cachetools import TTLCache
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html
import dash_bootstrap_components as dbc


USASPENDING_BASE = "https://api.usaspending.gov/api/v2"
HTTP_TIMEOUT = 30
CACHE = TTLCache(maxsize=256, ttl=10 * 60)  # 10 minutes


@dataclass(frozen=True)
class Filters:
    keyword: str
    agencies: Tuple[int, ...]
    award_types: Tuple[str, ...]
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    place_state: Optional[str]
    top_n: int


def _http_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{USASPENDING_BASE}{path}"
    cache_key = (url, tuple(sorted(_freeze(payload).items())))
    if cache_key in CACHE:
        return CACHE[cache_key]

    resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    CACHE[cache_key] = data
    return data


def _freeze(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _freeze(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return tuple(_freeze(x) for x in obj)
    return obj


def _default_dates() -> Tuple[str, str]:
    end = dt.date.today()
    start = end - dt.timedelta(days=365)
    return start.isoformat(), end.isoformat()


def _build_time_periods(start_date: str, end_date: str) -> List[Dict[str, str]]:
    return [{"start_date": start_date, "end_date": end_date}]


def _award_type_options() -> List[Dict[str, str]]:
    # Common USAspending award type codes; you can adjust to your needs.
    return [
        {"label": "Contracts", "value": "A"},
        {"label": "Grants", "value": "B"},
        {"label": "Direct Payments", "value": "C"},
        {"label": "Loans", "value": "D"},
        {"label": "Insurance", "value": "E"},
        {"label": "Other", "value": "F"},
    ]


def _spending_by_recipient(filters: Filters) -> pd.DataFrame:
    """
    Aggregates obligations by recipient.
    Endpoint: /spending_by_category/recipient
    """
    payload: Dict[str, Any] = {
        "category": "recipient",
        "limit": max(10, min(100, filters.top_n)),
        "page": 1,
        "filters": {
            "time_period": _build_time_periods(filters.start_date, filters.end_date),
            "award_type_codes": list(filters.award_types) if filters.award_types else [],
            "keywords": [filters.keyword] if filters.keyword else [],
        },
    }

    if filters.agencies:
        payload["filters"]["agencies"] = [{"type": "awarding", "tier": "toptier", "id": int(a)} for a in filters.agencies]

    if filters.place_state:
        payload["filters"]["place_of_performance_locations"] = [{"country": "USA", "state": filters.place_state}]

    data = _http_post("/spending_by_category/recipient/", payload)
    results = data.get("results", []) or []
    df = pd.DataFrame(results)
    if df.empty:
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
    return df


def _spending_over_time(filters: Filters) -> pd.DataFrame:
    """
    Aggregates obligations over time (monthly).
    Endpoint: /spending_over_time/
    """
    payload: Dict[str, Any] = {
        "group": "month",
        "filters": {
            "time_period": _build_time_periods(filters.start_date, filters.end_date),
            "award_type_codes": list(filters.award_types) if filters.award_types else [],
            "keywords": [filters.keyword] if filters.keyword else [],
        },
    }

    if filters.agencies:
        payload["filters"]["agencies"] = [{"type": "awarding", "tier": "toptier", "id": int(a)} for a in filters.agencies]
    if filters.place_state:
        payload["filters"]["place_of_performance_locations"] = [{"country": "USA", "state": filters.place_state}]

    data = _http_post("/spending_over_time/", payload)
    results = data.get("results", []) or []
    df = pd.DataFrame(results)
    if df.empty:
        return df

    # results often include: time_period (YYYY-MM), aggregated_amount
    if "time_period" in df.columns:
        df["time_period"] = pd.to_datetime(df["time_period"], errors="coerce")
    amt_col = "aggregated_amount" if "aggregated_amount" in df.columns else "amount"
    if amt_col in df.columns:
        df["obligations"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0)
    return df[["time_period", "obligations"]].dropna()


def _awards_table(filters: Filters, limit: int = 50) -> pd.DataFrame:
    """
    Fetches award-level rows for a table.
    Endpoint: /awards/
    """
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
        "filters": {
            "time_period": _build_time_periods(filters.start_date, filters.end_date),
            "award_type_codes": list(filters.award_types) if filters.award_types else [],
            "keywords": [filters.keyword] if filters.keyword else [],
        },
        "page": 1,
        "limit": max(10, min(100, limit)),
        "sort": "Award Amount",
        "order": "desc",
    }

    if filters.agencies:
        payload["filters"]["agencies"] = [{"type": "awarding", "tier": "toptier", "id": int(a)} for a in filters.agencies]
    if filters.place_state:
        payload["filters"]["place_of_performance_locations"] = [{"country": "USA", "state": filters.place_state}]

    data = _http_post("/awards/", payload)
    results = data.get("results", []) or []
    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Make column names nicer
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
    if "award_amount" in df.columns:
        df["award_amount"] = pd.to_numeric(df["award_amount"], errors="coerce").fillna(0)
    if "action_date" in df.columns:
        df["action_date"] = pd.to_datetime(df["action_date"], errors="coerce")
    return df


def _kpis(recipient_df: pd.DataFrame, awards_df: pd.DataFrame) -> Tuple[float, int]:
    total = 0.0
    if not recipient_df.empty and "obligations" in recipient_df.columns:
        total = float(recipient_df["obligations"].sum())
    count = 0
    if not awards_df.empty:
        count = int(len(awards_df))
    return total, count


def _format_money(x: float) -> str:
    return f"${x:,.0f}"


def make_app() -> Dash:
    start_date, end_date = _default_dates()

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "USAspending Explorer"

    app.layout = dbc.Container(
        [
            html.H1("USAspending Explorer", className="my-3"),
            dbc.Alert(
                "Tip: start broad (no keyword), then narrow with award types, agency, and state.",
                color="info",
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Filters", className="card-title"),
                                    dbc.Label("Keyword"),
                                    dbc.Input(id="keyword", placeholder="e.g., cybersecurity, roads, NASA", value="", type="text"),
                                    dbc.Label("Awarding Agency IDs (comma-separated)"),
                                    dbc.Input(
                                        id="agency_ids",
                                        placeholder="e.g., 114 (DoD), 473 (GSA) ...",
                                        value="",
                                        type="text",
                                    ),
                                    dbc.Label("Award Types"),
                                    dcc.Dropdown(
                                        id="award_types",
                                        options=_award_type_options(),
                                        value=["A", "B"],
                                        multi=True,
                                        placeholder="Select award types",
                                    ),
                                    dbc.Label("Place of Performance State (2-letter)"),
                                    dbc.Input(id="state", placeholder="e.g., CA, NY, TX", value="", type="text"),
                                    dbc.Label("Date Range"),
                                    dcc.DatePickerRange(
                                        id="dates",
                                        start_date=start_date,
                                        end_date=end_date,
                                        display_format="YYYY-MM-DD",
                                    ),
                                    dbc.Label("Top N recipients"),
                                    dbc.Input(id="top_n", type="number", min=5, max=100, step=5, value=20),
                                    dbc.Button("Search", id="run", color="primary", className="mt-3", n_clicks=0),
                                    html.Div(id="err", className="text-danger mt-2"),
                                ]
                            )
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total obligations"), html.H3(id="kpi_total")]))),
                                    dbc.Col(dbc.Card(dbc.CardBody([html.H6("Awards shown"), html.H3(id="kpi_count")]))),
                                ],
                                className="mb-3",
                            ),
                            dbc.Card(dbc.CardBody([html.H5("Top Recipients"), dcc.Graph(id="recipients_bar")])),
                            dbc.Card(dbc.CardBody([html.H5("Spending Over Time"), dcc.Graph(id="time_line")])),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        html.H5("Awards"),
                                        dash_table.DataTable(
                                            id="awards_table",
                                            page_size=10,
                                            sort_action="native",
                                            filter_action="native",
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                "textAlign": "left",
                                                "minWidth": "120px",
                                                "maxWidth": "420px",
                                                "whiteSpace": "normal",
                                            },
                                        ),
                                    ]
                                ),
                                className="mb-4",
                            ),
                        ],
                        md=8,
                    ),
                ]
            ),
        ],
        fluid=True,
        className="pb-4",
    )

    @callback(
        Output("recipients_bar", "figure"),
        Output("time_line", "figure"),
        Output("awards_table", "data"),
        Output("awards_table", "columns"),
        Output("kpi_total", "children"),
        Output("kpi_count", "children"),
        Output("err", "children"),
        Input("run", "n_clicks"),
        State("keyword", "value"),
        State("agency_ids", "value"),
        State("award_types", "value"),
        State("dates", "start_date"),
        State("dates", "end_date"),
        State("state", "value"),
        State("top_n", "value"),
        prevent_initial_call=False,
    )
    def refresh(
        _n: int,
        keyword: str,
        agency_ids: str,
        award_types: List[str],
        start_date_in: str,
        end_date_in: str,
        state: str,
        top_n: Any,
    ):
        try:
            agencies = tuple(
                int(x.strip())
                for x in (agency_ids or "").split(",")
                if x.strip().isdigit()
            )
            state2 = (state or "").strip().upper() or None
            if state2 and len(state2) != 2:
                state2 = None

            f = Filters(
                keyword=(keyword or "").strip(),
                agencies=agencies,
                award_types=tuple(award_types or ()),
                start_date=str(start_date_in)[:10],
                end_date=str(end_date_in)[:10],
                place_state=state2,
                top_n=int(top_n or 20),
            )

            recipients_df = _spending_by_recipient(f)
            time_df = _spending_over_time(f)
            awards_df = _awards_table(f, limit=50)

            if recipients_df.empty:
                bar_fig = px.bar(title="No recipient results for these filters.")
            else:
                top = recipients_df.sort_values("obligations", ascending=False).head(f.top_n)
                bar_fig = px.bar(
                    top,
                    x="obligations",
                    y="recipient_name",
                    orientation="h",
                    title=f"Top {min(f.top_n, len(top))} recipients by obligations",
                )
                bar_fig.update_layout(yaxis={"categoryorder": "total ascending"})

            if time_df.empty:
                line_fig = px.line(title="No time series results for these filters.")
            else:
                line_fig = px.line(time_df.sort_values("time_period"), x="time_period", y="obligations", title="Obligations by month")

            if awards_df.empty:
                data = []
                columns = [{"name": "No award rows returned", "id": "empty"}]
            else:
                display_df = awards_df.copy()
                # Keep dates readable
                if "action_date" in display_df.columns:
                    display_df["action_date"] = display_df["action_date"].dt.strftime("%Y-%m-%d")
                data = display_df.to_dict("records")
                columns = [{"name": c.replace("_", " ").title(), "id": c} for c in display_df.columns]

            total, count = _kpis(recipients_df, awards_df)
            return (
                bar_fig,
                line_fig,
                data,
                columns,
                _format_money(total),
                str(count),
                "",
            )

        except requests.HTTPError as e:
            return px.bar(title="Error"), px.line(title="Error"), [], [], "—", "—", f"API error: {e}"
        except Exception as e:
            return px.bar(title="Error"), px.line(title="Error"), [], [], "—", "—", f"Unexpected error: {e}"

    return app


if __name__ == "__main__":
    app = make_app()
    app.run(debug=True, host="0.0.0.0", port=8050)
