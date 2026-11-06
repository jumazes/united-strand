"""What separates United from clubs that do win five in a row? Evidence for the coaching staff."""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from common import CHALLENGE_START, FD_TEAMS, MU_TM_ID, OUTPUTS, PROCESSED, RAW
from evaluate import logistic
from features import FEATURES_CONTEXT, EloParams, build_features

FD_FILES = {"1920": 2019, "2021": 2020, "2122": 2021, "2223": 2022, "2324": 2023, "2425": 2024, "2526": 2025, "2627": 2026}
FPL_TEAMS = {"Man Utd": "Man United", "Spurs": "Tottenham", "Sheffield Utd": "Sheffield United", "Nott'm Forest": "Nott'm Forest",
             "Man City": "Man City", "Newcastle": "Newcastle", "Wolves": "Wolves", "Luton": "Luton"}
ROLLING = 10
MU_STATS = ["possessionPct", "totalShots", "shotsOnTarget", "passPct", "totalPasses", "totalCrosses",
            "wonCorners", "totalTackles", "interceptions", "foulsCommitted"]


def load_fpl_xg():
    """Team xG per match: the sum of FPL player xG."""
    frames = []
    for path, season in zip(sorted(glob.glob(str(RAW / "fpl" / "merged_gw_*.csv"))), range(2022, 2027)):
        raw = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
        raw["expected_goals"] = pd.to_numeric(raw.expected_goals, errors="coerce").fillna(0)
        raw["team"] = raw.team.map(lambda t: FPL_TEAMS.get(t, t))
        raw["date"] = raw.kickoff_time.str[:10]
        raw["was_home"] = raw.was_home.astype(str).str.lower() == "true"
        per_side = raw.groupby(["fixture", "date", "team", "was_home"], as_index=False).expected_goals.sum()
        home = per_side[per_side.was_home].rename(columns={"team": "HomeTeam", "expected_goals": "home_xg"})
        away = per_side[~per_side.was_home].rename(columns={"team": "AwayTeam", "expected_goals": "away_xg"})
        merged = home.merge(away, on=["fixture", "date"])[["date", "HomeTeam", "AwayTeam", "home_xg", "away_xg"]]
        merged["season"] = season
        frames.append(merged)
    return pd.concat(frames)


def load_league():
    frames = []
    for name, season in FD_FILES.items():
        raw = pd.read_csv(RAW / "football_data" / f"E0_{name}.csv", encoding="latin-1").dropna(subset=["HomeTeam"])
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw.Date, dayfirst=True).dt.strftime("%Y-%m-%d")
        raw["season"] = season
        frames.append(raw)
    fd = pd.concat(frames, ignore_index=True)
    xg = load_fpl_xg()
    fd = fd.merge(xg, on=["date", "HomeTeam", "AwayTeam", "season"], how="left")

    current = fd[(fd.season == 2026) & fd.home_xg.notna()]
    both = pd.concat([current[["home_xg", "HxG"]].set_axis(["fpl", "fd"], axis=1),
                      current[["away_xg", "AxG"]].set_axis(["fpl", "fd"], axis=1)]).astype(float)
    print(f"FPL xG vs football-data xG, 2026/27: r={both.corr().iloc[0, 1]:.3f}, "
          f"mean {both.fpl.mean():.2f} vs {both.fd.mean():.2f}, n={len(both)}")

    # The FPL archive for the current season is partial; football-data has it all and agrees closely.
    is_current = fd.season == 2026
    fd.loc[is_current, "home_xg"] = pd.to_numeric(fd.loc[is_current, "HxG"], errors="coerce")
    fd.loc[is_current, "away_xg"] = pd.to_numeric(fd.loc[is_current, "AxG"], errors="coerce")

    # FPL published no xG for part of 2022/23: a match with zero for both sides means "missing".
    missing = (fd.home_xg == 0) & (fd.away_xg == 0)
    print("matches with no xG recorded, by season:", fd[missing].groupby("season").size().to_dict())
    fd.loc[missing, ["home_xg", "away_xg"]] = np.nan
    with_xg = fd[fd.season >= 2022]
    print("matches with xG by season:", with_xg[with_xg.home_xg.notna()].groupby("season").size().to_dict())
    return fd


def team_table(fd):
    sides = []
    for side, other, us, them in (("Home", "Away", "H", "A"), ("Away", "Home", "A", "H")):
        low, opp = side.lower(), other.lower()
        sides.append(pd.DataFrame({
            "season": fd.season, "date": fd.date, "team": fd[f"{side}Team"], "opp": fd[f"{other}Team"], "home": int(side == "Home"),
            "gf": fd[f"FT{us}G"], "ga": fd[f"FT{them}G"], "ht_gf": fd[f"HT{us}G"], "ht_ga": fd[f"HT{them}G"],
            "shots": fd[f"{us}S"], "shots_against": fd[f"{them}S"], "sot": fd[f"{us}ST"], "sot_against": fd[f"{them}ST"],
            "xg": fd[f"{low}_xg"], "xga": fd[f"{opp}_xg"],
        }))
    table = pd.concat(sides, ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)
    table["won"] = (table.gf > table.ga).astype(int)
    table["pts"] = np.select([table.gf > table.ga, table.gf == table.ga], [3, 1], 0)
    table["xgd"] = table.xg - table.xga
    table["finish"] = table.gf - table.xg
    table["keep"] = table.xga - table.ga
    return table


def has_run(won, need=5):
    run = 0
    for w in won:
        run = run + 1 if w else 0
        if run >= need:
            return True
    return False


def season_profile(table):
    grouped = table[table.xg.notna()].groupby(["season", "team"])
    profile = grouped.agg(games=("won", "size"), ppg=("pts", "mean"), win_rate=("won", "mean"), xg=("xg", "mean"), xga=("xga", "mean"),
                          xgd=("xgd", "mean"), finish=("finish", "mean"), keep=("keep", "mean"),
                          shots=("shots", "mean"), sot=("sot", "mean"), shots_against=("shots_against", "mean"),
                          sot_against=("sot_against", "mean"), run5=("won", has_run), gf=("gf", "mean"), ga=("ga", "mean")).reset_index()
    return profile


def rolling_value_test(table, rows):
    """Does recent xG or shot difference predict results beyond Elo? Trained on earlier seasons only."""
    data = table.copy()
    for col in ("xgd", "finish", "keep"):
        data[f"roll_{col}"] = data.groupby("team")[col].transform(lambda s: s.shift().rolling(ROLLING, min_periods=5).mean())
    data["shotd"] = data.shots - data.shots_against
    data["sotd"] = data.sot - data.sot_against
    for col in ("shotd", "sotd"):
        data[f"roll_{col}"] = data.groupby("team")[col].transform(lambda s: s.shift().rolling(ROLLING, min_periods=5).mean())
    cols = ["roll_xgd", "roll_finish", "roll_keep", "roll_shotd", "roll_sotd"]
    home = data[data.home == 1][["date", "team", "opp"] + cols]
    away = data[data.home == 0][["date", "team"] + cols].rename(columns={c: c + "_a" for c in cols}).rename(columns={"team": "opp"})
    merged = home.merge(away, on=["date", "opp"])
    for c in cols:
        merged[c + "_d"] = merged[c] - merged[c + "_a"]
    ids = merged.assign(home_id=merged.team.map(FD_TEAMS), away_id=merged.opp.map(FD_TEAMS))
    pl = rows[rows.competition_id == "GB1"].merge(ids, on=["date", "home_id", "away_id"])
    pl = pl.dropna(subset=[c + "_d" for c in cols])
    sets = {
        "elo + context": FEATURES_CONTEXT,
        "+ recent xG difference": FEATURES_CONTEXT + ["roll_xgd_d"],
        "+ recent shot difference": FEATURES_CONTEXT + ["roll_shotd_d"],
        "+ recent shots on target": FEATURES_CONTEXT + ["roll_sotd_d"],
        "+ recent finishing": FEATURES_CONTEXT + ["roll_finish_d"],
        "+ recent goalkeeping": FEATURES_CONTEXT + ["roll_keep_d"],
    }
    baseline = None
    results = []
    for name, features in sets.items():
        losses = []
        for season in (2024, 2025, 2026):
            train, test = pl[(pl.season >= 2022) & (pl.season < season)], pl[pl.season == season]
            model = logistic().fit(train[features], train.outcome)
            probs = model.predict_proba(test[features])
            losses.append(-np.log(probs[np.arange(len(test)), test.outcome.to_numpy()]))
        losses = np.concatenate(losses)
        if baseline is None:
            baseline = losses
        diff = losses - baseline
        results.append({"features": name, "log_loss": losses.mean(), "vs_elo": diff.mean(),
                        "std_err": diff.std() / np.sqrt(len(diff)), "n": len(losses)})

    # How many Elo points is one unit of recent xG difference worth?
    features = FEATURES_CONTEXT + ["roll_xgd_d"]
    fit = logistic().fit(pl[features], pl.outcome)
    scaler, classifier = fit.steps[0][1], fit.steps[1][1]
    per_unit = classifier.coef_[0] / scaler.scale_  # log-odds of a home win
    elo_per_xgd = per_unit[features.index("roll_xgd_d")] / per_unit[features.index("elo_diff")]
    mu_now = data[data.team == "Man United"].sort_values("date").tail(ROLLING)
    league_now = data[data.season == 2026].groupby("team").xgd.mean()
    print(f"\n1.0 of recent xG difference per game is worth about {elo_per_xgd:.0f} Elo points "
          f"beyond the rating itself; United's last {ROLLING} league matches: xG difference {mu_now.xgd.mean():+.2f} per game "
          f"(xG {mu_now.xg.mean():.2f}, xGA {mu_now.xga.mean():.2f}); "
          f"2026/27 rank {int(league_now.rank(ascending=False)['Man United'])} of {len(league_now)}")
    return pd.DataFrame(results), elo_per_xgd


def lead_conversion(table):
    data = table[(table.season >= 2024)].copy()
    data["ht"] = np.select([data.ht_gf > data.ht_ga, data.ht_gf == data.ht_ga], ["leading", "level"], "trailing")
    data["is_mu"] = data.team == "Man United"
    out = data.groupby(["is_mu", "ht"]).agg(n=("won", "size"), win_rate=("won", "mean"), ppg=("pts", "mean")).reset_index()
    return out


def mu_splits(predictions, ledger, table_rows, params):
    data = predictions.merge(ledger[["date", "home_id", "away_id", "competition_id", "result"]], on="date", suffixes=("", "_l"))
    data["home"] = (data.home_id == MU_TM_ID).astype(int)
    elo = table_rows[(table_rows.home_id == MU_TM_ID) | (table_rows.away_id == MU_TM_ID)][["date", "home_id", "elo_home", "elo_away", "rest_home", "rest_away"]]
    data = data.merge(elo, on=["date", "home_id"], how="left")
    data["opp_elo"] = np.where(data.home == 1, data.elo_away, data.elo_home)
    data["mu_elo"] = np.where(data.home == 1, data.elo_home, data.elo_away)
    data["rest"] = np.where(data.home == 1, data.rest_home, data.rest_away)
    data["opp_type"] = pd.cut(data.opp_elo - data.mu_elo, [-1000, -50, 50, 1000], labels=["weaker (50+ Elo)", "similar", "stronger (50+ Elo)"])
    data["venue"] = np.where(data.home == 1, "home", "away")
    data["rest_group"] = np.where(data.rest <= 3, "3 days or fewer", "4+ days")
    data["comp"] = data.competition_id.map(lambda c: "Premier League" if c == "GB1" else "Europe" if c in ("EL", "CL") else "cups")
    records = []
    for split in ("opp_type", "venue", "rest_group", "comp"):
        for value, g in data.groupby(split, observed=True):
            expected = g.p_win.sum()
            variance = (g.p_win * (1 - g.p_win)).sum()
            records.append({"split": split, "group": value, "n": len(g), "win_rate": g.won.mean(), "expected_rate": g.p_win.mean(),
                            "draw_rate": (g.result == "D").mean(), "loss_rate": (g.result == "L").mean(),
                            "z": (g.won.sum() - expected) / np.sqrt(variance)})
    return pd.DataFrame(records), data


def peer_comparison(all_rows):
    """The same splits for every club over the same period, to tell a United effect from an artefact."""
    view = pd.read_csv(PROCESSED / "oos_predictions.csv", dtype={"team": str})
    view = view[view.date >= CHALLENGE_START]
    columns = ["game_id", "home_id", "away_id", "elo_home", "elo_away", "rest_home", "rest_away", "outcome"]
    view = view.merge(all_rows[columns], on="game_id")
    home = view.team == view.home_id
    view["elo_gap"] = np.where(home, view.elo_away - view.elo_home, view.elo_home - view.elo_away)
    view["rest"] = np.where(home, view.rest_home, view.rest_away)
    view["home"] = home
    view["drew"] = (view.outcome == 1).astype(int)

    def z_score(g, actual, prob):
        variance = (g[prob] * (1 - g[prob])).sum()
        return (g[actual].sum() - g[prob].sum()) / np.sqrt(variance) if variance > 0 else np.nan

    splits = {
        "wins vs stronger (50+ Elo)": (view.elo_gap > 50, "won", "p_win"),
        "wins vs weaker (50+ Elo)": (view.elo_gap < -50, "won", "p_win"),
        "wins on 3 days' rest or fewer": (view.rest <= 3, "won", "p_win"),
        "wins at home": (view.home, "won", "p_win"),
        "wins away": (~view.home, "won", "p_win"),
        "draws away": (~view.home, "drew", "p_draw"),
    }
    games = view.groupby("team").size()
    teams = games[games >= 60].index
    print(f"\npeer comparison: {len(teams)} clubs with 60+ matches in the period")
    records = []
    for name, (mask, actual, prob) in splits.items():
        subset = view[mask & view.team.isin(teams)]
        scores = subset.groupby("team")[["won", "drew", "p_win", "p_draw"]].apply(lambda g: z_score(g, actual, prob)).dropna()
        mu = scores[MU_TM_ID]
        records.append({"split": name, "mu_z": mu, "all_clubs_mean_z": scores.mean(), "all_clubs_sd": scores.std(),
                        "clubs_more_extreme": int((scores.abs() >= abs(mu)).sum() - 1), "clubs": len(scores)})
    return pd.DataFrame(records)


def lagged_stats(data):
    """Reverse causality check: yesterday's crosses cannot be caused by today's score."""
    rows = []
    for path in glob.glob(str(RAW / "espn_summary" / "*.json")):
        with open(path, encoding="utf8") as handle:
            summary = json.load(handle)
        row = {"date": summary["header"]["competitions"][0]["date"][:10]}
        for side in summary.get("boxscore", {}).get("teams", []):
            if side["team"]["id"] == "360":
                for stat in side.get("statistics", []):
                    if stat["name"] in ("totalCrosses", "shotsOnTarget", "wonCorners"):
                        row[stat["name"]] = pd.to_numeric(stat.get("displayValue"), errors="coerce")
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("date").merge(data[["date", "won", "p_win"]], on="date")
    frame["surprise"] = frame.won - frame.p_win
    records = []
    for name in ("totalCrosses", "shotsOnTarget", "wonCorners"):
        lagged = frame[name].shift().rolling(5, min_periods=3).mean()
        valid = lagged.notna() & frame.surprise.notna()
        r, p = stats.pearsonr(lagged[valid], frame.surprise[valid])
        records.append({"stat": name, "lag": "mean of previous 5 matches", "n": int(valid.sum()), "r_with_surprise": r, "p_value": p})
    return pd.DataFrame(records)


def by_manager(data):
    """Does a pattern survive a change of manager? Archive names; 2026/27 is after the archive ends."""
    games = pd.read_csv(RAW / "transfermarkt" / "games.csv.gz", low_memory=False,
                        usecols=["date", "home_club_id", "home_club_manager_name", "away_club_manager_name"])
    games["manager"] = np.where(games.home_club_id == int(MU_TM_ID), games.home_club_manager_name, games.away_club_manager_name)
    frame = data.merge(games[["date", "home_club_id", "manager"]].assign(home_id=lambda g: g.home_club_id.astype(str)),
                       on=["date", "home_id"], how="left")
    frame["manager"] = frame.manager.replace({"Ruben Amorim": "Rúben Amorim"}).fillna("2026/27 (Carrick, per press)")
    records = []
    for (manager, group), g in frame.groupby(["manager", "opp_type"], observed=True):
        if len(g) >= 5:
            records.append({"manager": manager, "opp_type": group, "n": len(g), "win_rate": g.won.mean(), "expected": g.p_win.mean(),
                            "draw_rate": (g.result == "D").mean()})
    return pd.DataFrame(records)


def quality_by_opponent(table, all_rows):
    """Premier League, by Elo gap: does United lack chances against weaker sides, or fail to take them?"""
    pl = all_rows[all_rows.competition_id == "GB1"][["date", "home_id", "away_id", "elo_home", "elo_away"]]
    data = table[(table.season >= 2024) & table.xg.notna() & (table.date >= CHALLENGE_START)].copy()
    data["team_id"] = data.team.map(FD_TEAMS)
    data["opp_id"] = data.opp.map(FD_TEAMS)
    as_home = data[data.home == 1].merge(pl, left_on=["date", "team_id", "opp_id"], right_on=["date", "home_id", "away_id"])
    as_away = data[data.home == 0].merge(pl, left_on=["date", "opp_id", "team_id"], right_on=["date", "home_id", "away_id"])
    as_home["gap"] = as_home.elo_away - as_home.elo_home
    as_away["gap"] = as_away.elo_home - as_away.elo_away
    joined = pd.concat([as_home, as_away])
    joined["opp_type"] = pd.cut(joined.gap, [-1000, -50, 50, 1000], labels=["weaker", "similar", "stronger"])
    joined["who"] = np.where(joined.team == "Man United", "United", "rest of league")
    joined["venue"] = np.where(joined.home == 1, "home", "away")
    return (joined.groupby(["opp_type", "venue", "who"], observed=True)
            .agg(n=("won", "size"), win_rate=("won", "mean"), xg=("xg", "mean"), xga=("xga", "mean"), xgd=("xgd", "mean"),
                 gf=("gf", "mean"), ga=("ga", "mean"), shots=("shots", "mean"), sot=("sot", "mean"), finish=("finish", "mean"))
            .reset_index())


def run_breakers(data):
    records = []
    run = 0
    for row in data.sort_values("date").itertuples(index=False):
        if row.won:
            run += 1
            continue
        if run >= 2:
            records.append({"date": row.date, "run_broken": run, "result": row.result, "venue": row.venue,
                            "opp_type": row.opp_type, "p_win": row.p_win})
        run = 0
    return pd.DataFrame(records)


def mu_match_stats(data):
    rows = []
    for path in glob.glob(str(RAW / "espn_summary" / "*.json")):
        with open(path, encoding="utf8") as handle:
            summary = json.load(handle)
        date = summary["header"]["competitions"][0]["date"][:10]
        sides = summary.get("boxscore", {}).get("teams", [])
        row = {"date": date}
        for side in sides:
            mine = side["team"]["id"] == "360"
            for stat in side.get("statistics", []):
                if stat["name"] in MU_STATS:
                    row[("mu_" if mine else "opp_") + stat["name"]] = pd.to_numeric(stat.get("displayValue"), errors="coerce")
        rows.append(row)
    frame = pd.DataFrame(rows).merge(data[["date", "won", "p_win", "result"]], on="date")
    frame["surprise"] = frame.won - frame.p_win
    records = []
    for name in MU_STATS:
        col = "mu_" + name
        if col not in frame:
            continue
        valid = frame.dropna(subset=[col])
        r, p = stats.pearsonr(valid[col], valid.surprise)
        records.append({"stat": name, "n": len(valid), "mean_when_win": valid[valid.won == 1][col].mean(),
                        "mean_when_not": valid[valid.won == 0][col].mean(), "r_with_surprise": r, "p_value": p})
    return pd.DataFrame(records).sort_values("p_value")


def main():
    params = EloParams(**json.loads((PROCESSED / "elo_params.json").read_text()))
    matches = pd.read_csv(PROCESSED / "matches.csv", dtype={"home_id": str, "away_id": str})
    all_rows, _ = build_features(matches, params)
    rows = all_rows[all_rows.season >= 2014]
    fd = load_league()
    table = team_table(fd)

    profile = season_profile(table)
    profile.to_csv(OUTPUTS / "team_season_profile.csv", index=False)
    full = profile[(profile.games >= 30) & (profile.season < 2026)]
    print(f"\nteam-seasons with xG: {len(full)}, seasons {sorted(full.season.unique())}, "
          f"run of 5 league wins in {full.run5.mean():.1%}")
    print("correlation with a run of 5 league wins:")
    for col in ("ppg", "xgd", "xg", "xga", "finish", "keep", "sot", "sot_against"):
        print(f"  {col:12} r={np.corrcoef(full[col], full.run5)[0, 1]:+.2f}")
    logit = LogisticRegression(C=100).fit(full[["xgd"]], full.run5)
    grid = np.arange(-1, 1.51, 0.25)
    curve = logit.predict_proba(pd.DataFrame({"xgd": grid}))[:, 1]
    print("P(5-win league run in a season) by xG difference per game:",
          {round(float(x), 2): round(float(p), 2) for x, p in zip(grid, curve)})
    mid = -logit.intercept_[0] / logit.coef_[0][0]
    print(f"xG difference per game at P=50%: {mid:+.2f}")

    print("\nUnited, league seasons, rank among 20 (1 = best):")
    ranks = profile.copy()
    for col, ascending in (("xgd", False), ("xg", False), ("xga", True), ("finish", False), ("keep", False),
                           ("sot", False), ("sot_against", True), ("ppg", False)):
        ranks[col + "_rank"] = ranks.groupby("season")[col].rank(ascending=ascending).astype(int)
    mu = ranks[ranks.team == "Man United"]
    print(mu[["season", "games", "ppg", "ppg_rank", "xg", "xg_rank", "xga", "xga_rank", "xgd", "xgd_rank", "finish", "finish_rank",
              "keep", "keep_rank", "sot", "sot_against", "run5"]].round(2).to_string(index=False))
    mu.to_csv(OUTPUTS / "mu_league_profile.csv", index=False)

    tested, _ = rolling_value_test(table, rows)
    tested.to_csv(OUTPUTS / "indicator_value_test.csv", index=False)
    print("\nvalue of recent indicators beyond Elo (Premier League, seasons 2024-2026, negative = better):")
    print(tested.round(5).to_string(index=False))

    leads = lead_conversion(table)
    leads.to_csv(OUTPUTS / "lead_conversion.csv", index=False)
    print("\nby half-time state, league, 2024/25 onward:\n" + leads.round(3).to_string(index=False))

    predictions = pd.read_csv(OUTPUTS / "mu_challenge_predictions.csv")
    ledger = pd.read_csv(PROCESSED / "mu_ledger.csv", dtype={"home_id": str, "away_id": str})
    splits, data = mu_splits(predictions, ledger, all_rows, params)
    splits.to_csv(OUTPUTS / "mu_splits.csv", index=False)
    print("\nUnited in the challenge, actual vs expected:\n" + splits.round(3).to_string(index=False))

    breakers = run_breakers(data)
    breakers.to_csv(OUTPUTS / "mu_run_breakers.csv", index=False)
    print("\nwhat ended United's runs of 2+ wins:\n" + breakers.round(3).to_string(index=False))
    print(f"share where United was the favourite to win (p>=0.5): {(breakers.p_win >= .5).mean():.2f}; "
          f"draws {(breakers.result == 'D').mean():.2f}")

    match_stats = mu_match_stats(data)
    match_stats.to_csv(OUTPUTS / "mu_match_stats.csv", index=False)
    print("\nUnited match stats vs results beyond expectation (10 tests, so p<0.005 is the bar):\n" + match_stats.round(3).to_string(index=False))

    lagged = lagged_stats(data)
    lagged.to_csv(OUTPUTS / "mu_lagged_stats.csv", index=False)
    print("\nsame stats measured before the match:\n" + lagged.round(3).to_string(index=False))

    peers = peer_comparison(all_rows)
    peers.to_csv(OUTPUTS / "peer_comparison.csv", index=False)
    print(peers.round(2).to_string(index=False))

    managers = by_manager(data)
    managers.to_csv(OUTPUTS / "mu_by_manager.csv", index=False)
    print("\nby manager:\n" + managers.round(3).to_string(index=False))

    quality = quality_by_opponent(table, all_rows)
    quality.to_csv(OUTPUTS / "mu_quality_by_opponent.csv", index=False)
    print("\nchance quality by opponent, Premier League:\n" + quality.round(2).to_string(index=False))

    away = data[data.venue == "away"]
    print(f"\naway: {len(away)} matches, {(away.result == 'D').sum()} draws, expected draw rate {away.p_draw.mean():.3f}")
    print(f"stats available for {match_stats.n.max()} of {len(data)} matches")


if __name__ == "__main__":
    main()
