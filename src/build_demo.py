"""Embed model coefficients and result tables into the demo page. Every number on the page comes from here."""
import json
from datetime import date

import numpy as np
import pandas as pd

from common import CHALLENGE_START, MU_TM_ID, OUTPUTS, PROCESSED, RAW, ROOT, SNAPSHOT_DATE
from evaluate import TEST_SEASONS, logistic, model_rows, p_run
from features import FEATURES_CONTEXT, EloParams, build_features, goal_multiplier
from simulate import build_calendar, previous_dates, with_assumed_cups

DEMO = ROOT / "demo"
SITE = ROOT / "docs"
SHORT_NAMES = {
    "AFC Bournemouth": "Bournemouth", "Brighton & Hove Albion": "Brighton", "Coventry City": "Coventry", "Hull City": "Hull",
    "Ipswich Town": "Ipswich", "Leeds United": "Leeds", "Leicester City": "Leicester", "Manchester City": "Man City",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves", "Sporting CP": "Sporting", "Bayern Munich": "Bayern",
    "AS Roma": "Roma", "Atlético Madrid": "Atlético", "Sabah FK": "Sabah", "Grimsby Town": "Grimsby", "FC Twente": "Twente",
    "FC Porto": "Porto", "Athletic Club": "Athletic Bilbao",
}
COMPS = {"GB1": "Premier League", "CL": "Champions League", "EL": "Europa League", "FAC": "FA Cup", "CGB": "League Cup", "assumed": "assumed"}
ASSUMED = {"FA Cup R3 (assumed)": "FA Cup third round", "UCL play-off 1 (assumed)": "UCL play-off, leg 1", "UCL play-off 2 (assumed)": "UCL play-off, leg 2"}
MANAGERS = {"Erik ten Hag": "ten Hag", "Ruud van Nistelrooy": "van Nistelrooy", "Rúben Amorim": "Amorim", "Ruben Amorim": "Amorim",
            "Darren Fletcher": "Fletcher", "Michael Carrick": "Carrick"}


def number(value, digits=1):
    return f"{value:.{digits}f}"


def pct(value, digits=0):
    return number(value * 100, digits) + "%"


def short(name):
    return SHORT_NAMES.get(name, name)


def finite(value):
    """Make the payload valid JSON: numpy scalars become plain numbers and NaN becomes null."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def clean_fixtures(items):
    return [{
        "date": f["date"], "label": ASSUMED.get(f["label"]) or short(f["label"]),
        "comp": COMPS.get(f.get("competition_id"), f.get("competition_id")),
        "assumed": f.get("competition_id") == "assumed",
        "opp_elo": float(f["opp_elo"]), "is_home": int(f["is_home"]), "neutral": int(f.get("neutral", 0)),
        "is_cup": int(f["is_cup"]), "is_europe": int(f["is_europe"]),
        "rest_team": int(f["rest_team"]), "rest_opp": int(f["rest_opp"]),
    } for f in items]


def challenge_strip(ledger, predictions):
    """Every match of the challenge, in order, with the manager and the model's pre-match estimate."""
    games = pd.read_csv(RAW / "transfermarkt" / "games.csv.gz", low_memory=False,
                        usecols=["date", "home_club_id", "away_club_id", "home_club_manager_name", "away_club_manager_name"])
    games = games[(games.home_club_id == int(MU_TM_ID)) | (games.away_club_id == int(MU_TM_ID))]
    manager_by_date = {d: (h if hid == int(MU_TM_ID) else a) for d, hid, h, a in zip(
        games.date, games.home_club_id, games.home_club_manager_name, games.away_club_manager_name)}
    p_by_date = dict(zip(predictions.date, predictions.p_win))
    strip = []
    for g in ledger.itertuples(index=False):
        home = g.home_id == MU_TM_ID
        strip.append({
            "date": g.date, "opp": short(g.away_name if home else g.home_name), "venue": "home" if home else "away",
            "comp": COMPS.get(g.competition_id, g.competition_id), "gf": int(g.gf), "ga": int(g.ga), "r": g.result,
            "pens": g.status == "FT-Pens", "aet": g.status == "AET", "p": float(p_by_date.get(g.date, float("nan"))),
            "manager": MANAGERS.get(manager_by_date.get(g.date)),
        })
    # The archive ends in July 2026 and misses two League Cup ties. Press reports name Carrick for 2026/27;
    # a missing tie takes the manager of the next match, which matters for van Nistelrooy's first game.
    following = "Carrick"
    for item in reversed(strip):
        item["manager"] = item["manager"] or following
        following = item["manager"]
    return strip


def main():
    params = EloParams(**json.loads((PROCESSED / "elo_params.json").read_text()))
    matches = pd.read_csv(PROCESSED / "matches.csv", dtype={"home_id": str, "away_id": str})
    upcoming = pd.read_csv(PROCESSED / "upcoming.csv", dtype={"home_id": str, "away_id": str})
    ledger = pd.read_csv(PROCESSED / "mu_ledger.csv", dtype={"home_id": str, "away_id": str})
    predictions = pd.read_csv(OUTPUTS / "mu_challenge_predictions.csv")
    table, state = build_features(matches, params)
    rows = model_rows(table).copy()
    gbar = float(np.mean([goal_multiplier(d) for d in (rows.home_goals - rows.away_goals)]))
    pipe = logistic().fit(rows[FEATURES_CONTEXT], rows.outcome)
    scaler, classifier = pipe.steps[0][1], pipe.steps[1][1]
    ratings = state["ratings"]

    calendar = previous_dates(matches, upcoming)
    fixtures = build_calendar(upcoming, ratings, calendar)
    with_cups = with_assumed_cups(fixtures)
    strip = challenge_strip(ledger, predictions)

    splits = pd.read_csv(OUTPUTS / "mu_splits.csv")
    peers = pd.read_csv(OUTPUTS / "peer_comparison.csv").set_index("split")
    group = lambda name: splits[splits.group == name].iloc[0]
    weak, strong, short_rest, away = group("weaker (50+ Elo)"), group("stronger (50+ Elo)"), group("3 days or fewer"), group("away")
    labels = {"weaker (50+ Elo)": "Against weaker sides", "similar": "Against similar sides", "stronger (50+ Elo)": "Against stronger sides",
              "home": "At home", "away": "Away", "3 days or fewer": "3 days' rest or fewer", "4+ days": "4+ days' rest"}
    bars = [{"label": labels[r.group], "n": int(r.n), "actual": float(r.win_rate), "expected": float(r.expected_rate)}
            for r in splits.itertuples(index=False) if r.group in labels]

    calib = pd.read_csv(OUTPUTS / "calibration.csv")
    backtest = pd.read_csv(OUTPUTS / "run_backtest.csv")
    bins = backtest.groupby(pd.cut(backtest.pred, [0, .1, .25, .4, .6, .8, 1]), observed=True).agg(
        n=("actual", "size"), predicted=("pred", "mean"), observed=("actual", "mean")).reset_index()
    model_table = pd.read_csv(OUTPUTS / "model_comparison.csv")
    model_names = {"base rates": "Outcome frequencies", "elo": "Elo", "elo + context": "Elo + context", "elo + context + form": "Elo + context + form",
                   "gradient boosting": "Gradient boosting", "bet365 closing": "Bet365 closing odds"}

    breakers = pd.read_csv(OUTPUTS / "mu_run_breakers.csv")
    by_date = {s["date"]: s for s in strip}
    endings = [{"date": b.date, "run": int(b.run_broken), "opp": by_date[b.date]["opp"], "venue": by_date[b.date]["venue"], "r": b.result,
                "gf": by_date[b.date]["gf"], "ga": by_date[b.date]["ga"], "p": float(b.p_win)} for b in breakers.itertuples(index=False)]

    profile = pd.read_csv(OUTPUTS / "mu_league_profile.csv").set_index("season")
    quality = pd.read_csv(OUTPUTS / "mu_quality_by_opponent.csv")
    weak_away = quality[(quality.opp_type == "weaker") & (quality.venue == "away")].set_index("who")
    mu_q, league_q = weak_away.loc["United"], weak_away.loc["rest of league"]
    match_stats = pd.read_csv(OUTPUTS / "mu_match_stats.csv").set_index("stat")
    lagged = pd.read_csv(OUTPUTS / "mu_lagged_stats.csv").set_index("stat")

    mu_rows = table[(table.date >= CHALLENGE_START) & ((table.home_id == MU_TM_ID) | (table.away_id == MU_TM_ID))]
    peak = float(np.where(mu_rows.home_id == MU_TM_ID, mu_rows.elo_home, mu_rows.elo_away).max())
    days = (date.fromisoformat(SNAPSHOT_DATE) - date.fromisoformat(CHALLENGE_START)).days
    away_rows = predictions.merge(ledger[["date", "home_id"]], on="date")
    away_rows = away_rows[away_rows.home_id != MU_TM_ID]

    text = {
        "days": str(days), "n": str(len(ledger)), "wins": str((ledger.result == "W").sum()), "exp_wins": number(predictions.p_win.sum()),
        "no_run": pct(1 - p_run(predictions.p_win.to_numpy())),
        "strong_a": pct(strong.win_rate), "strong_e": pct(strong.expected_rate), "weak_a": pct(weak.win_rate), "weak_e": pct(weak.expected_rate),
        "strong_clubs": str(int(peers.loc["wins vs stronger (50+ Elo)", "clubs"]) - 1),
        "away_draw_a": pct(away.draw_rate), "away_draw_e": pct(away_rows.p_draw.mean()),
        "rest_a": pct(short_rest.win_rate), "rest_e": pct(short_rest.expected_rate),
        "shots_mu": number(mu_q.shots), "shots_lg": number(league_q.shots), "xg_mu": number(mu_q.xg, 2), "xg_lg": number(league_q.xg, 2),
        "xgs_mu": number(mu_q.xg / mu_q.shots, 3), "xgs_lg": number(league_q.xg / league_q.shots, 3), "q_n": str(int(mu_q.n)),
        "xga_old": number(profile.loc[2025, "xga"], 2), "xga_new": number(profile.loc[2026, "xga"], 2),
        "xg_old": number(profile.loc[2025, "xg"], 2), "xg_new": number(profile.loc[2026, "xg"], 2), "games_new": str(int(profile.loc[2026, "games"])),
        "cross_r": number(match_stats.loc["totalCrosses", "r_with_surprise"], 2), "cross_lag": number(lagged.loc["totalCrosses", "r_with_surprise"], 2),
        "peak": str(round(peak)), "now": str(round(ratings[MU_TM_ID])), "city": str(round(ratings["281"])),
        "backtests": str(len(backtest)), "brier_model": number(((backtest.pred - backtest.actual) ** 2).mean(), 3),
        "brier_simple": number(((backtest.pred_simple - backtest.actual) ** 2).mean(), 3),
        "brier_const": number(((backtest.actual.mean() - backtest.actual) ** 2).mean(), 3),
        "test_matches": f"{rows.season.isin(TEST_SEASONS).sum():,}",
    }

    data = {
        "snapshot": SNAPSHOT_DATE,
        "model": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist()},
        "elo": {"k": params.k, "home_adv": params.home_adv, "gbar": gbar, "now": float(ratings[MU_TM_ID]), "peak": peak, "city": float(ratings["281"])},
        "start_streak": int(next(i for i, r in enumerate(ledger.result[::-1]) if r != "W")),
        "fixtures": clean_fixtures(fixtures), "fixtures_cups": clean_fixtures(with_cups),
        "weak_ratio": float(weak.win_rate / weak.expected_rate),
        "strip": strip, "bars": bars, "endings": endings, "t": text,
        "calibration": [{"n": int(r.n), "x": float(r.predicted), "y": float(r.observed)} for r in calib.itertuples(index=False)],
        "backtest": [{"n": int(r.n), "x": float(r.predicted), "y": float(r.observed)} for r in bins.itertuples(index=False)],
        "models": [{"name": model_names[r.model], "all": float(r.log_loss_all), "pl": float(r.log_loss_pl), "acc": float(r.accuracy_pl),
                    "chosen": r.model == "elo + context", "book": r.model == "bet365 closing"} for r in model_table.itertuples(index=False)],
    }
    payload = json.dumps(finite(data), ensure_ascii=False).replace("</", "<\\/")

    page = (DEMO / "template.html").read_text(encoding="utf8").replace("/*__DATA__*/null", payload)
    (DEMO / "haircut.html").write_text(page, encoding="utf8")
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                  "<style>*{box-sizing:border-box}body{margin:0;font:14px system-ui,sans-serif}[hidden]{display:none!important}</style>\n"
                  "</head>\n<body>\n" + page + "\n</body>\n</html>\n")
    # GitHub Pages serves this folder, so the site needs no server or build step.
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(standalone, encoding="utf8")
    (SITE / ".nojekyll").write_text("", encoding="utf8")
    print(f"demo built: {len(page):,} bytes, {len(fixtures)} fixtures, {len(strip)} matches, weak ratio {data['weak_ratio']:.3f}, gbar {gbar:.3f}")


if __name__ == "__main__":
    main()
