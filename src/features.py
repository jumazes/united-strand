"""Sequential Elo and match features. Every value is computed from earlier matches only."""
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from common import season_of

START_LEAGUE = 1450.0
START_CUP = 1250.0
PROMOTION_GAP_DAYS = 200
FEATURES_CONTEXT = ["elo_diff", "neutral", "is_cup", "is_europe", "rest_home", "rest_away"]
FEATURES_FORM = FEATURES_CONTEXT + ["form_pts_home", "form_pts_away", "form_gd_home", "form_gd_away"]


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    regress: float = 0.1
    home_adv: float = 60.0


def goal_multiplier(goal_diff):
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def expected_home(rating_home, rating_away, neutral, home_adv):
    delta = rating_home - rating_away + (0 if neutral else home_adv)
    return 1 / (1 + 10 ** (-delta / 400))


@lru_cache(maxsize=None)
def day_number(date_text):
    return date.fromisoformat(date_text).toordinal()


def days_between(a, b):
    return day_number(a) - day_number(b)


def build_features(matches, params):
    """Walk through matches in date order. Returns features and the final ratings."""
    ratings = {}
    last_match = {}
    last_league_match = {}
    league_pool = defaultdict(dict)
    recent = defaultdict(lambda: deque(maxlen=5))
    current_season = None
    rows = []

    for m in matches.itertuples(index=False):
        season = season_of(m.date)
        if season != current_season:
            for team in ratings:
                ratings[team] = 1500 + (1 - params.regress) * (ratings[team] - 1500)
            current_season = season

        is_league = m.comp_group == "league"
        for team in (m.home_id, m.away_id):
            if team not in ratings:
                ratings[team] = START_LEAGUE if is_league else START_CUP
            if is_league:
                # A club returning to a league after a long absence (usually promotion, since
                # second divisions are not in the data) is pulled towards that league's bottom.
                previous = last_league_match.get(team)
                if previous is None or days_between(m.date, previous) >= PROMOTION_GAP_DAYS:
                    pool = league_pool[m.competition_id]
                    active = [ratings[t] for t, d in pool.items() if days_between(m.date, d) < 400]
                    if len(active) >= 8:
                        ratings[team] = 0.5 * ratings[team] + 0.5 * float(np.percentile(active, 15))

        rating_h, rating_a = ratings[m.home_id], ratings[m.away_id]
        rest_h = days_between(m.date, last_match[m.home_id]) if m.home_id in last_match else 14
        rest_a = days_between(m.date, last_match[m.away_id]) if m.away_id in last_match else 14
        form_h, form_a = recent[m.home_id], recent[m.away_id]
        rows.append({
            "game_id": m.game_id,
            "elo_home": rating_h,
            "elo_away": rating_a,
            "elo_diff": rating_h - rating_a,
            "rest_home": min(rest_h, 14),
            "rest_away": min(rest_a, 14),
            "form_pts_home": np.mean([x[0] for x in form_h]) if form_h else 1.3,
            "form_pts_away": np.mean([x[0] for x in form_a]) if form_a else 1.3,
            "form_gd_home": np.mean([x[1] for x in form_h]) if form_h else 0.0,
            "form_gd_away": np.mean([x[1] for x in form_a]) if form_a else 0.0,
        })

        goal_diff = m.home_goals - m.away_goals
        score = 1.0 if goal_diff > 0 else 0.0 if goal_diff < 0 else 0.5
        delta = (params.k * goal_multiplier(goal_diff)
                 * (score - expected_home(rating_h, rating_a, m.neutral, params.home_adv)))
        ratings[m.home_id] = rating_h + delta
        ratings[m.away_id] = rating_a - delta

        points_h = 3 if goal_diff > 0 else 1 if goal_diff == 0 else 0
        points_a = 3 if goal_diff < 0 else 1 if goal_diff == 0 else 0
        recent[m.home_id].append((points_h, goal_diff))
        recent[m.away_id].append((points_a, -goal_diff))
        last_match[m.home_id] = last_match[m.away_id] = m.date
        if is_league:
            last_league_match[m.home_id] = last_league_match[m.away_id] = m.date
            league_pool[m.competition_id][m.home_id] = m.date
            league_pool[m.competition_id][m.away_id] = m.date

    table = matches.merge(pd.DataFrame(rows), on="game_id")
    table["is_cup"] = (table.comp_group == "domestic_cup").astype(int)
    table["is_europe"] = (table.comp_group == "europe").astype(int)
    table["outcome"] = np.select([table.home_goals > table.away_goals, table.home_goals < table.away_goals],
                                 [0, 2], default=1)
    state = {"ratings": ratings, "last_match": last_match}
    return table, state
