"""Monte Carlo search for a run of five wins: backtest on past seasons, then the live forecast."""
import json

import numpy as np
import pandas as pd

from common import MODEL_COMPS, MU_TM_ID, OUTPUTS, PROCESSED, SNAPSHOT_DATE, TOP5
from evaluate import logistic, model_rows, p_run
from features import FEATURES_CONTEXT, EloParams, build_features, days_between, goal_multiplier

NEED = 5
BACKTEST_SEASONS = range(2019, 2026)
BACKTEST_SIMS = 2000
FORECAST_SIMS = 200_000
SHIFTS = [-50, 0, 25, 50, 75, 100, 150, 200, 250, 300]

# Not yet on the calendar: dates and opponents here are assumptions, used only as a sensitivity check.
ASSUMED_CUP_MATCHES = [
    {"date": "2027-01-09", "opp_elo": 1500.0, "is_home": 1, "is_cup": 1, "is_europe": 0, "label": "FA Cup R3 (assumed)"},
    {"date": "2027-02-16", "opp_elo": 1700.0, "is_home": 0, "is_cup": 0, "is_europe": 1, "label": "UCL play-off 1 (assumed)"},
    {"date": "2027-02-23", "opp_elo": 1700.0, "is_home": 1, "is_cup": 0, "is_europe": 1, "label": "UCL play-off 2 (assumed)"},
]


class FastModel:
    """The fitted pipeline as plain arrays, so a simulation step costs microseconds."""

    def __init__(self, pipeline):
        scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
        self.mean, self.scale = scaler.mean_, scaler.scale_
        self.coef, self.intercept = classifier.coef_, classifier.intercept_

    def proba(self, X):
        scores = ((X - self.mean) / self.scale) @ self.coef.T + self.intercept
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / probs.sum(axis=1, keepdims=True)


def simulate(fixtures, start_elo, start_streak, model, params, gbar, n, rng, update=True):
    """Returns, per simulation, the index of the fixture that completes the run, or -1."""
    elo = np.full(n, float(start_elo))
    streak = np.full(n, min(start_streak, NEED - 1))
    hit = np.full(n, -1)
    for i, f in enumerate(fixtures):
        diff = elo - f["opp_elo"] if f["is_home"] else f["opp_elo"] - elo
        neutral = f.get("neutral", 0)
        rest_h, rest_a = (f["rest_team"], f["rest_opp"]) if f["is_home"] else (f["rest_opp"], f["rest_team"])
        X = np.column_stack([diff, np.full(n, neutral), np.full(n, f["is_cup"]), np.full(n, f["is_europe"]),
                             np.full(n, rest_h), np.full(n, rest_a)])
        probs = model.proba(X)
        p_win = probs[:, 0] if f["is_home"] else probs[:, 2]
        u = rng.random(n)
        won = u < p_win
        drew = (~won) & (u < p_win + probs[:, 1])
        streak = np.where(won, streak + 1, 0)
        hit = np.where((hit < 0) & (streak >= NEED), i, hit)
        if update:
            adv = 0 if neutral else (params.home_adv if f["is_home"] else -params.home_adv)
            expected = 1 / (1 + 10 ** (-(elo - f["opp_elo"] + adv) / 400))
            result = np.where(won, 1.0, np.where(drew, 0.5, 0.0))
            elo = elo + params.k * gbar * (result - expected)
    return hit


def team_fixtures(games, team, ratings):
    fixtures = []
    for g in games.itertuples(index=False):
        home = g.home_id == team
        opponent = g.away_id if home else g.home_id
        fixtures.append({
            "opp_elo": ratings.get(opponent, g.elo_away if home else g.elo_home),
            "is_home": int(home), "neutral": g.neutral, "is_cup": g.is_cup, "is_europe": g.is_europe,
            "rest_team": g.rest_home if home else g.rest_away,
            "rest_opp": g.rest_away if home else g.rest_home,
        })
    return fixtures


def backtest(rows, table, params, gbar, rng):
    """Same pipeline, same calendar point, seasons the model has not seen. Uses no results after the cut-off."""
    records = []
    for season in BACKTEST_SEASONS:
        cutoff = f"{season}-11-06"
        model = FastModel(logistic().fit(rows.loc[rows.season < season, FEATURES_CONTEXT],
                                         rows.loc[rows.season < season, "outcome"]))
        later = table[(table.date > cutoff) & (table.season == season)]
        ratings = {}
        for g in later.itertuples(index=False):
            ratings.setdefault(g.home_id, g.elo_home)
            ratings.setdefault(g.away_id, g.elo_away)
        current = rows[rows.season == season]
        for team in sorted(set(current.loc[current.competition_id.isin(TOP5), "home_id"])):
            games = current[(current.home_id == team) | (current.away_id == team)]
            before, after = games[games.date <= cutoff], games[games.date > cutoff]
            if len(after) < 15 or team not in ratings:
                continue
            won_before = (before.home_id == team) & (before.outcome == 0) | (before.away_id == team) & (before.outcome == 2)
            start = 0
            for won in won_before.to_numpy()[::-1]:
                if not won:
                    break
                start += 1
            start = min(start, NEED - 1)
            hit = simulate(team_fixtures(after, team, ratings), ratings[team], start, model, params, gbar, BACKTEST_SIMS, rng)
            won_after = ((after.home_id == team) & (after.outcome == 0) | (after.away_id == team) & (after.outcome == 2)).to_numpy()
            run, happened = start, False
            for won in won_after:
                run = run + 1 if won else 0
                happened = happened or run >= NEED
            rate = won_before.mean() if len(before) else 0.4
            records.append({"season": season, "team": team, "n_after": len(after), "start": start,
                            "pred": float((hit >= 0).mean()), "pred_simple": p_run([rate] * len(after), start=start),
                            "actual": int(happened)})
    return pd.DataFrame(records)


def previous_dates(matches, upcoming):
    """All match dates per team, played and scheduled, for rest days."""
    dates = {}
    for frame in (matches, upcoming):
        for d, h, a in zip(frame.date, frame.home_id, frame.away_id):
            dates.setdefault(h, set()).add(d)
            dates.setdefault(a, set()).add(d)
    return {team: sorted(values) for team, values in dates.items()}


def rest_days(calendar, team, date):
    earlier = [d for d in calendar.get(team, []) if d < date]
    return min(days_between(date, earlier[-1]), 14) if earlier else 14


def build_calendar(upcoming, ratings, calendar):
    mu = upcoming[(upcoming.home_id == MU_TM_ID) | (upcoming.away_id == MU_TM_ID)].sort_values("date")
    fixtures = []
    for g in mu.itertuples(index=False):
        home = g.home_id == MU_TM_ID
        opponent, name = (g.away_id, g.away_name) if home else (g.home_id, g.home_name)
        fixtures.append({
            "date": g.date, "label": name, "competition_id": g.competition_id,
            "opp_elo": ratings[opponent], "is_home": int(home), "neutral": int(g.neutral),
            "is_cup": int(g.comp_group == "domestic_cup"), "is_europe": int(g.comp_group == "europe"),
            "rest_team": rest_days(calendar, MU_TM_ID, g.date), "rest_opp": rest_days(calendar, opponent, g.date),
        })
    return fixtures


def with_assumed_cups(fixtures):
    """A new calendar with the assumed matches added. The known fixtures are copied, never changed."""
    merged = sorted([dict(f) for f in fixtures + ASSUMED_CUP_MATCHES], key=lambda f: f["date"])
    for previous, f in zip([None] + merged[:-1], merged):
        f.setdefault("neutral", 0)
        f.setdefault("competition_id", "assumed")
        f.setdefault("rest_opp", 7)
        if previous is not None:
            f["rest_team"] = min(days_between(f["date"], previous["date"]), 14)
        else:
            f.setdefault("rest_team", 14)
    return merged


def main():
    params = EloParams(**json.loads((PROCESSED / "elo_params.json").read_text()))
    matches = pd.read_csv(PROCESSED / "matches.csv", dtype={"home_id": str, "away_id": str})
    upcoming = pd.read_csv(PROCESSED / "upcoming.csv", dtype={"home_id": str, "away_id": str})
    ledger = pd.read_csv(PROCESSED / "mu_ledger.csv")
    table, state = build_features(matches, params)
    rows = model_rows(table).copy()
    gbar = float(np.mean([goal_multiplier(d) for d in (rows.home_goals - rows.away_goals)]))
    rng = np.random.default_rng(42)

    # 1. Would this have worked before? Forecast the event itself, season by season.
    result = backtest(rows, table, params, gbar, rng)
    result.to_csv(OUTPUTS / "run_backtest.csv", index=False)
    base = result.actual.mean()
    brier = lambda pred: float(((pred - result.actual) ** 2).mean())
    print(f"backtest: {len(result)} team-seasons, a run of 5 happened in {base:.1%}")
    print(f"Brier: model {brier(result.pred):.4f} | win rate so far {brier(result.pred_simple):.4f} | "
          f"constant {brier(pd.Series(base, index=result.index)):.4f}")
    bins = pd.cut(result.pred, [0, .1, .25, .4, .6, .8, 1])
    print(result.groupby(bins, observed=True).agg(
        n=("actual", "size"), predicted=("pred", "mean"), observed=("actual", "mean")).round(3).to_string())
    print("MU in backtest:\n" + result[result.team == MU_TM_ID][["season", "start", "pred", "actual"]].round(3).to_string(index=False))

    # 2. The live forecast.
    model = FastModel(logistic().fit(rows[FEATURES_CONTEXT], rows.outcome))
    ratings = state["ratings"]
    calendar = previous_dates(matches, upcoming)
    fixtures = build_calendar(upcoming, ratings, calendar)
    start_streak = 0
    for r in ledger.result.to_numpy()[::-1]:
        if r != "W":
            break
        start_streak += 1
    print(f"\nsnapshot {SNAPSHOT_DATE}: United Elo {ratings[MU_TM_ID]:.0f}, current winning run {start_streak}, "
          f"{len(fixtures)} fixtures known, last {fixtures[-1]['date']}")

    forecast = []
    elo = ratings[MU_TM_ID]
    for f in fixtures:
        diff = elo - f["opp_elo"] if f["is_home"] else f["opp_elo"] - elo
        rest_h, rest_a = (f["rest_team"], f["rest_opp"]) if f["is_home"] else (f["rest_opp"], f["rest_team"])
        p = model.proba(np.array([[diff, f["neutral"], f["is_cup"], f["is_europe"], rest_h, rest_a]]))[0]
        p_w, p_d, p_l = (p[0], p[1], p[2]) if f["is_home"] else (p[2], p[1], p[0])
        pick = max((p_w, "W"), (p_d, "D"), (p_l, "L"))[1]
        forecast.append({"date": f["date"], "comp": f["competition_id"], "opponent": f["label"],
                         "venue": "H" if f["is_home"] else "A", "opp_elo": round(f["opp_elo"]),
                         "p_win": p_w, "p_draw": p_d, "p_loss": p_l, "pick": pick})
    forecast = pd.DataFrame(forecast)

    hit = simulate(fixtures, elo, start_streak, model, params, gbar, FORECAST_SIMS, rng)
    forecast["p_run_completes_here"] = [(hit == i).mean() for i in range(len(fixtures))]
    forecast["p_done_by_here"] = forecast.p_run_completes_here.cumsum()
    forecast.to_csv(OUTPUTS / "mu_fixture_forecast.csv", index=False)
    print("\n" + forecast.round(3).to_string(index=False))

    windows = pd.DataFrame([{
        "from": forecast.date[i], "to": forecast.date[i + NEED - 1],
        "opponents": ", ".join(forecast.opponent[i:i + NEED]),
        "p_all_five": float(forecast.p_win[i:i + NEED].prod()),
    } for i in range(len(forecast) - NEED + 1)]).sort_values("p_all_five", ascending=False)
    windows.to_csv(OUTPUTS / "mu_best_windows.csv", index=False)
    print("\nbest windows of five consecutive fixtures, at today's rating:\n" + windows.head(5).round(4).to_string(index=False))

    picks = "".join(forecast.pick)
    print(f"\nmodel's single most likely result per match: {picks}")
    print(f"longest run of predicted wins: {max(len(s) for s in picks.replace('D', 'L').split('L'))}")

    done = (hit >= 0).mean()
    print(f"\nP(haircut by end of known calendar) = {done:.3f}")
    dates = np.array([f["date"] for f in fixtures])
    if done > 0:
        print("first date with cumulative P >= 25%:", next((d for d, c in zip(forecast.date, forecast.p_done_by_here) if c >= .25), "never"))

    frozen = simulate(fixtures, elo, start_streak, model, params, gbar, FORECAST_SIMS, rng, update=False)
    print(f"with rating frozen at today's value: {(frozen >= 0).mean():.3f}")
    assumed = with_assumed_cups(fixtures)
    with_cups = simulate(assumed, elo, start_streak, model, params, gbar, FORECAST_SIMS, rng)
    print(f"with three assumed cup/UCL matches: {(with_cups >= 0).mean():.3f}")

    sensitivity = []
    for shift in SHIFTS:
        shifted = simulate(fixtures, elo + shift, start_streak, model, params, gbar, FORECAST_SIMS // 2, rng)
        mean_win = []
        for f in fixtures:
            value = elo + shift
            diff = value - f["opp_elo"] if f["is_home"] else f["opp_elo"] - value
            rest_h, rest_a = (f["rest_team"], f["rest_opp"]) if f["is_home"] else (f["rest_opp"], f["rest_team"])
            p = model.proba(np.array([[diff, f["neutral"], f["is_cup"], f["is_europe"], rest_h, rest_a]]))[0]
            mean_win.append(p[0] if f["is_home"] else p[2])
        sensitivity.append({"elo_shift": shift, "elo": round(elo + shift), "mean_p_win": float(np.mean(mean_win)),
                            "p_haircut": float((shifted >= 0).mean())})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUTPUTS / "mu_sensitivity.csv", index=False)
    print("\nsensitivity to United's strength:\n" + sensitivity.round(3).to_string(index=False))

    # A concrete reference point: United back at their best rating of the challenge.
    mu_rows = table[(table.date >= "2024-10-05") & ((table.home_id == MU_TM_ID) | (table.away_id == MU_TM_ID))]
    rating = np.where(mu_rows.home_id == MU_TM_ID, mu_rows.elo_home, mu_rows.elo_away)
    peak = float(rating.max())
    peak_date = mu_rows.date.iloc[int(rating.argmax())]
    at_peak = simulate(fixtures, peak, start_streak, model, params, gbar, FORECAST_SIMS, rng)
    print(f"\nbest rating of the challenge: {peak:.0f} on {peak_date} (+{peak - elo:.0f}); "
          f"P(haircut) from that level = {(at_peak >= 0).mean():.3f}")
    print(f"backtest mean forecast {result.pred.mean():.3f} vs actual {result.actual.mean():.3f}; "
          f"in the forecast band nearest United (5-20%): "
          f"{result[(result.pred > .05) & (result.pred <= .2)].pred.mean():.3f} vs "
          f"{result[(result.pred > .05) & (result.pred <= .2)].actual.mean():.3f}, "
          f"n={((result.pred > .05) & (result.pred <= .2)).sum()}")


if __name__ == "__main__":
    main()
