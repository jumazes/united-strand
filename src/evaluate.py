"""Tune Elo and compare match models on future seasons only."""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FD_TEAMS, MODEL_COMPS, MU_TM_ID, OUTPUTS, PROCESSED, RAW
from features import FEATURES_CONTEXT, FEATURES_FORM, EloParams, build_features

TUNE_TRAIN = range(2014, 2017)
TUNE_VALID = range(2017, 2019)
TEST_SEASONS = list(range(2019, 2027))
FD_FILES = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def model_rows(table):
    return table[table.competition_id.isin(MODEL_COMPS) & (table.season >= 2014)]


def tune(matches):
    best, best_loss = None, np.inf
    subset = matches[matches.season <= max(TUNE_VALID)]
    for k, regress, home_adv in itertools.product((15, 20, 30), (0.0, 0.1, 0.2, 0.33), (40, 60, 80)):
        params = EloParams(k=k, regress=regress, home_adv=home_adv)
        table, _ = build_features(subset, params)
        rows = model_rows(table)
        train, valid = rows[rows.season.isin(TUNE_TRAIN)], rows[rows.season.isin(TUNE_VALID)]
        model = logistic().fit(train[["elo_diff", "neutral"]], train.outcome)
        loss = log_loss(valid.outcome, model.predict_proba(valid[["elo_diff", "neutral"]]), labels=[0, 1, 2])
        if loss < best_loss:
            best, best_loss = params, loss
    print(f"tuned on seasons {TUNE_TRAIN[0]}-{TUNE_VALID[-1]}: {best}, validation log loss {best_loss:.4f}")
    return best


def walk_forward(rows, make_model, features):
    """For each season, train only on earlier seasons and predict that season."""
    parts = []
    for season in TEST_SEASONS:
        train, test = rows[rows.season < season], rows[rows.season == season]
        model = make_model().fit(train[features], train.outcome)
        parts.append(pd.DataFrame(model.predict_proba(test[features]), columns=["p_h", "p_d", "p_a"], index=test.index))
    return pd.concat(parts)


def base_rates(rows):
    parts = []
    for season in TEST_SEASONS:
        train, test = rows[rows.season < season], rows[rows.season == season]
        freq = train.outcome.value_counts(normalize=True).sort_index().to_numpy()
        parts.append(pd.DataFrame(np.tile(freq, (len(test), 1)), columns=["p_h", "p_d", "p_a"], index=test.index))
    return pd.concat(parts)


def bookmaker_odds(rows):
    """Bet365 closing odds with the margin removed, joined to Premier League test matches."""
    frames = []
    for season in FD_FILES:
        raw = pd.read_csv(RAW / "football_data" / f"E0_{season}.csv", encoding="latin-1").dropna(subset=["HomeTeam"])
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(raw.Date, dayfirst=True).dt.strftime("%Y-%m-%d"),
            "home_id": raw.HomeTeam.map(FD_TEAMS),
            "away_id": raw.AwayTeam.map(FD_TEAMS),
            "o_h": raw.B365CH, "o_d": raw.B365CD, "o_a": raw.B365CA,
        }))
    odds = pd.concat(frames).dropna()
    implied = 1 / odds[["o_h", "o_d", "o_a"]].to_numpy()
    odds[["p_h", "p_d", "p_a"]] = implied / implied.sum(axis=1, keepdims=True)
    pl = rows[(rows.competition_id == "GB1") & rows.season.isin(TEST_SEASONS)]
    joined = pl.reset_index().merge(odds, on=["date", "home_id", "away_id"], how="left").set_index("index")
    print(f"bookmaker join: {joined.p_h.notna().sum()} of {len(pl)} Premier League test matches")
    return joined[["p_h", "p_d", "p_a"]].dropna()


def team_view(rows, preds):
    """One row per team per match: its win probability and what happened."""
    data = rows.join(preds, how="inner")
    sides = []
    for team, p_win, p_loss, won in (("home_id", "p_h", "p_a", 0), ("away_id", "p_a", "p_h", 2)):
        sides.append(pd.DataFrame({
            "game_id": data.game_id, "date": data.date, "season": data.season, "team": data[team],
            "competition_id": data.competition_id, "p_win": data[p_win], "p_draw": data.p_d,
            "p_loss": data[p_loss], "won": (data.outcome == won).astype(int),
        }))
    return pd.concat(sides).sort_values(["team", "date", "game_id"]).reset_index(drop=True)


def p_run(probs, need=5, start=0):
    """Probability of at least one run of `need` wins, treating matches as independent."""
    state = np.zeros(need)
    state[min(start, need - 1)] = 1.0
    done = 0.0
    for p in probs:
        new = np.zeros(need)
        new[0] = state.sum() * (1 - p)
        new[1:] = state[:-1] * p
        done += state[-1] * p
        state = new
    return done


def streak_calibration(view):
    """Do teams on a run beat their predicted probability? A direct test of momentum."""
    streaks = []
    for _, games in view.groupby("team", sort=False):
        run = 0
        values = []
        for won in games.won.to_numpy():
            values.append(run)
            run = run + 1 if won else 0
        streaks.append(pd.Series(values, index=games.index))
    bucket = pd.concat(streaks).clip(upper=5)
    table = view.groupby(bucket).agg(n=("won", "size"), predicted=("p_win", "mean"), observed=("won", "mean"))
    table.index.name = "wins_in_a_row_before"
    table["z"] = (table.observed - table.predicted) / np.sqrt(table.observed * (1 - table.observed) / table.n)
    return table


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(PROCESSED / "matches.csv", dtype={"home_id": str, "away_id": str})
    params = tune(matches)
    (PROCESSED / "elo_params.json").write_text(json.dumps(params.__dict__))
    table, _ = build_features(matches, params)
    rows = model_rows(table).copy()
    print(f"model rows: {len(rows):,}; test matches: {rows.season.isin(TEST_SEASONS).sum():,} "
          f"(seasons {TEST_SEASONS[0]}-{TEST_SEASONS[-1]})")

    candidates = {
        "base rates": base_rates(rows),
        "elo": walk_forward(rows, logistic, ["elo_diff", "neutral"]),
        "elo + context": walk_forward(rows, logistic, FEATURES_CONTEXT),
        "elo + context + form": walk_forward(rows, logistic, FEATURES_FORM),
        "gradient boosting": walk_forward(
            rows, lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
            FEATURES_FORM),
    }
    book = bookmaker_odds(rows)
    y = rows.outcome

    def score(preds):
        return log_loss(y[preds.index], preds, labels=[0, 1, 2]), float((preds.to_numpy().argmax(axis=1) == y[preds.index].to_numpy()).mean())

    records = []
    for name, preds in candidates.items():
        all_loss, all_acc = score(preds)
        pl_loss, pl_acc = score(preds.loc[book.index])
        records.append({"model": name, "log_loss_all": all_loss, "accuracy_all": all_acc,
                        "log_loss_pl": pl_loss, "accuracy_pl": pl_acc})
    pl_loss, pl_acc = score(book)
    records.append({"model": "bet365 closing", "log_loss_pl": pl_loss, "accuracy_pl": pl_acc})
    report = pd.DataFrame(records)
    report.to_csv(OUTPUTS / "model_comparison.csv", index=False)
    print("\n" + report.round(4).to_string(index=False))

    # Is recent form worth anything beyond Elo? Paired test on the per-match loss.
    def per_match(preds):
        p = preds.to_numpy()[np.arange(len(preds)), y[preds.index].to_numpy()]
        return -np.log(p)

    diff = per_match(candidates["elo + context + form"]) - per_match(candidates["elo + context"])
    print(f"\nform vs no form, loss difference per match: {diff.mean():+.5f} "
          f"(std err {diff.std() / np.sqrt(len(diff)):.5f}, negative favours form)")

    chosen = candidates["elo + context"]
    view = team_view(rows, chosen)
    view.to_csv(PROCESSED / "oos_predictions.csv", index=False)

    bins = pd.cut(view.p_win, [0, .1, .2, .3, .4, .5, .6, .7, .8, 1])
    calib = view.groupby(bins, observed=True).agg(n=("won", "size"), predicted=("p_win", "mean"), observed=("won", "mean"))
    calib.to_csv(OUTPUTS / "calibration.csv")
    print("\nwin-probability calibration:\n" + calib.round(3).to_string())

    streaks = streak_calibration(view)
    streaks.to_csv(OUTPUTS / "streak_calibration.csv")
    print("\nmomentum test (5 = five or more wins in a row before the match):\n" + streaks.round(3).to_string())

    mu = view[(view.team == MU_TM_ID) & (view.date >= "2024-10-05")].sort_values("date")
    picks = mu[mu.p_win >= mu[["p_draw", "p_loss"]].max(axis=1)]
    print(f"\nUnited in the challenge: {len(mu)} matches, expected wins {mu.p_win.sum():.1f}, actual {mu.won.sum()}")
    print(f"P(no run of 5 | per-match model probabilities) = {1 - p_run(mu.p_win.to_numpy()):.3f}")
    print(f"model picked a win in {len(picks)} matches and was right in {picks.won.sum()} ({picks.won.mean():.1%})")
    mu.to_csv(OUTPUTS / "mu_challenge_predictions.csv", index=False)


if __name__ == "__main__":
    main()
