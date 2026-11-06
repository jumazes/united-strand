"""Build one clean match table from the Transfermarkt archive and the ESPN current season.

Rules applied here:
- a shootout is a draw: shootout goals are removed from the archive's score;
- extra-time goals count, because a win after extra time is a win;
- friendlies and national-team matches are dropped.
"""
import difflib
import glob
import json
import unicodedata

import pandas as pd

from common import EUROPE, MU_ESPN_ID, MU_TM_ID, PROCESSED, RAW, SNAPSHOT_DATE, season_of

STOPWORDS = {"fc", "afc", "cf", "ac", "ssc", "sc", "as", "ss", "us", "rc", "fk", "sk", "club",
             "de", "football", "calcio", "hotspur", "1907", "1909", "wanderers", "albion", "town", "city_"}

# ESPN display name -> Transfermarkt club id, where automatic matching is unsafe or fails.
MANUAL = {
    "Sporting CP": "336",
    "AS Roma": "12",
    "Como": "1047",
    "RB Leipzig": "23826",
    "Atlético Madrid": "13",
    "Hull City": "3008",
    "Ipswich Town": "677",
    "Coventry City": "990",
    "Wolverhampton Wanderers": "543",
    "Tottenham Hotspur": "148",
    "Brighton & Hove Albion": "1237",
    "West Ham United": "379",
    "Newcastle United": "762",
    "Nottingham Forest": "703",
    "AFC Bournemouth": "989",
    "Leeds United": "399",
    "Manchester United": MU_TM_ID,
    "Manchester City": "281",
    "Leicester City": "1003",
    "Alavés": "1108", "Anderlecht": "58", "Athletic Club": "621", "Benfica": "294",
    "Besiktas": "114", "Bodo/Glimt": "501", "CSU Craiova": "40812", "Club Brugge": "2282",
    "Deportivo": "897", "Dinamo Zagreb": "419", "Espanyol": "714", "Estoril": "1465",
    "Estrela": "2431", "FC Cologne": "3", "FC Twente": "317", "Hapoel Be'er": "2976",
    "Hertha Berlin": "44", "Internazionale": "46", "LASK Linz": "413", "Levante": "3368",
    "Lille": "1082", "Lillestrom": "353", "Lyon": "1041", "Mainz": "39", "Maritimo": "1301",
    "Osasuna": "331", "Panathinaikos": "265", "RB Salzburg": "409", "Real Betis": "150",
    "SK Sturm Graz": "122", "TSG Hoffenheim": "533", "Torreense": "2432", "Tromso": "1293",
    "Viking FK": "239", "Sabah FK": "63993", "Viktoria Plzen": "941",
}

LEAGUES = {
    "English Premier League": "GB1", "Spanish LALIGA": "ES1", "Italian Serie A": "IT1",
    "German Bundesliga": "L1", "French Ligue 1": "FR1", "UEFA Champions League": "CL",
    "UEFA Europa League": "EL", "UEFA Conference League": "UCOL", "English FA Cup": "FAC",
    "English Carabao Cup": "CGB", "Spanish Copa del Rey": "CDR", "German Cup": "DFB",
    "Coppa Italia": "CIT",
}


def normalize(name):
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(t for t in ascii_name.replace(".", " ").split() if t not in STOPWORDS)


def group_of(competition_id, competition_type):
    if competition_id in EUROPE:
        return "europe"
    if competition_type == "domestic_league":
        return "league"
    if competition_type == "domestic_cup":
        return "domestic_cup"
    return "other"


def load_archive():
    games = pd.read_csv(RAW / "transfermarkt" / "games.csv.gz", low_memory=False)
    games = games[games.competition_type != "national_team_competition"].copy()
    events = pd.read_csv(RAW / "transfermarkt" / "game_events.csv.gz", low_memory=False,
                         usecols=["game_id", "type", "club_id", "description"])
    shootout = events[events.type == "Shootout"]
    scored = shootout[shootout.description.fillna("").str.contains("Scored")]
    per_club = scored.groupby(["game_id", "club_id"]).size().rename("n").reset_index()

    games = games.merge(per_club.rename(columns={"club_id": "home_club_id", "n": "pens_h"}),
                        how="left", on=["game_id", "home_club_id"])
    games = games.merge(per_club.rename(columns={"club_id": "away_club_id", "n": "pens_a"}),
                        how="left", on=["game_id", "away_club_id"])
    has_shootout = games.game_id.isin(shootout.game_id)
    games["home_goals"] = games.home_club_goals - games.pens_h.fillna(0)
    games["away_goals"] = games.away_club_goals - games.pens_a.fillna(0)

    # A shootout only follows a level score, so anything else means the events are incomplete.
    bad = has_shootout & (games.home_goals != games.away_goals)
    level = (games.home_goals[bad] + games.away_goals[bad]) // 2
    games.loc[bad, "home_goals"] = level
    games.loc[bad, "away_goals"] = level
    print(f"shootouts removed from score: {has_shootout.sum():,} games "
          f"({bad.sum()} with incomplete events, forced level)")

    games["neutral"] = games["round"].fillna("").str.fullmatch("Final") | (
        (games.competition_id == "FAC") & games["round"].fillna("").str.contains("Semi"))
    rows = pd.DataFrame({
        "game_id": "tm_" + games.game_id.astype(str),
        "date": games.date,
        "competition_id": games.competition_id,
        "comp_group": [group_of(c, t) for c, t in zip(games.competition_id, games.competition_type)],
        "home_id": games.home_club_id.astype(str),
        "away_id": games.away_club_id.astype(str),
        "home_name": games.home_club_name,
        "away_name": games.away_club_name,
        "home_goals": games.home_goals.astype(int),
        "away_goals": games.away_goals.astype(int),
        "neutral": games.neutral.astype(int),
        "source": "transfermarkt",
    })
    names = pd.concat([
        games[["home_club_id", "home_club_name"]].set_axis(["id", "name"], axis=1),
        games[["away_club_id", "away_club_name"]].set_axis(["id", "name"], axis=1),
    ]).dropna().drop_duplicates()
    names["id"] = names.id.astype(str)
    return rows, names


class Resolver:
    """Map ESPN team names onto Transfermarkt ids. Unknown teams keep a separate ESPN id."""

    def __init__(self, names):
        self.by_norm = {normalize(n): i for i, n in zip(names.id, names.name)}
        self.choices = list(self.by_norm)
        self.audit = {}

    def __call__(self, espn_id, name):
        if name in MANUAL:
            found, how = MANUAL[name], "manual"
        else:
            key = normalize(name)
            if key in self.by_norm:
                found, how = self.by_norm[key], "exact"
            else:
                close = difflib.get_close_matches(key, self.choices, n=1, cutoff=0.88)
                found, how = (self.by_norm[close[0]], "fuzzy") if close else (f"espn_{espn_id}", "new")
        self.audit[name] = (found, how)
        return found


def parse_event(event, resolve):
    competition = event["competitions"][0]
    sides = {c["homeAway"]: c for c in competition["competitors"]}
    home, away = sides["home"], sides["away"]
    league = event["league"]["name"]
    if "Friendly" in league:
        return None
    comp = LEAGUES.get(league, "OTHER")
    if comp in EUROPE:
        group = "europe"
    elif comp in {"FAC", "CGB", "CDR", "DFB", "CIT"} or "Cup" in league or "Copa" in league:
        group = "domestic_cup"
    else:
        group = "league"

    def score(side):
        value = side.get("score")
        value = value.get("value") if isinstance(value, dict) else value
        return None if value is None else int(float(value))

    return {
        "game_id": "espn_" + event["id"],
        "date": event["date"][:10],
        "kickoff": event["date"],
        "competition_id": comp,
        "comp_group": group,
        "home_id": resolve(home["team"]["id"], home["team"]["displayName"]),
        "away_id": resolve(away["team"]["id"], away["team"]["displayName"]),
        "home_name": home["team"]["displayName"],
        "away_name": away["team"]["displayName"],
        "home_goals": score(home),
        "away_goals": score(away),
        "neutral": int(bool(competition.get("neutral"))),
        "source": "espn",
        "completed": bool(competition["status"]["type"].get("completed")),
        "status": competition["status"]["type"].get("detail"),
    }


def load_espn(resolve):
    played, future = {}, {}
    for path in glob.glob(str(RAW / "espn" / "*.json")):
        with open(path, encoding="utf8") as handle:
            events = json.load(handle).get("events", [])
        for event in events:
            row = parse_event(event, resolve)
            if row is None:
                continue
            if row["completed"]:
                played[row["game_id"]] = row
            else:
                future[row["game_id"]] = row
    return pd.DataFrame(played.values()), pd.DataFrame(future.values())


def same_pair(a, b):
    return (a.date, a.home_id, a.away_id) == (b.date, b.home_id, b.away_id)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    archive, names = load_archive()
    resolve = Resolver(names)
    played, future = load_espn(resolve)
    played = played[played.date <= SNAPSHOT_DATE]

    # ESPN extends the archive and also fills United matches the archive lacks.
    last_archive = archive.date.max()
    # United plays at most once a day, so the date alone identifies a match even when the
    # opponent's id differs between sources.
    mu_archive_dates = set(archive.date[(archive.home_id == MU_TM_ID) | (archive.away_id == MU_TM_ID)])
    keep = [(d > last_archive) or (MU_TM_ID in (h, a) and d not in mu_archive_dates)
            for d, h, a in zip(played.date, played.home_id, played.away_id)]
    extra = played[keep].drop(columns=["kickoff", "completed", "status"])
    print(f"archive: {len(archive):,} games to {last_archive}; ESPN adds {len(extra):,} "
          f"(United gaps: {sum(1 for d in extra.date if d <= last_archive)})")
    matches = pd.concat([archive, extra]).sort_values(["date", "game_id"]).reset_index(drop=True)
    matches["season"] = matches.date.map(season_of)
    matches.to_csv(PROCESSED / "matches.csv", index=False)

    # Challenge ledger: United competitive matches from ESPN, the only source covering all of them.
    ledger = played[(played.home_id == MU_TM_ID) | (played.away_id == MU_TM_ID)]
    ledger = ledger[ledger.date >= "2024-10-05"].sort_values("date").copy()
    is_home = ledger.home_id == MU_TM_ID
    ledger["gf"] = ledger.home_goals.where(is_home, ledger.away_goals)
    ledger["ga"] = ledger.away_goals.where(is_home, ledger.home_goals)
    ledger["result"] = ["W" if f > a else "L" if f < a else "D" for f, a in zip(ledger.gf, ledger.ga)]
    ledger.to_csv(PROCESSED / "mu_ledger.csv", index=False)
    print("ledger:", len(ledger), ledger.result.value_counts().to_dict())

    upcoming = future[future.date > SNAPSHOT_DATE].sort_values("date")
    upcoming.to_csv(PROCESSED / "upcoming.csv", index=False)
    print("upcoming matches of tracked teams:", len(upcoming),
          "| United:", ((upcoming.home_id == MU_TM_ID) | (upcoming.away_id == MU_TM_ID)).sum())

    print("\nteam mapping (non-manual):")
    for name, (found, how) in sorted(resolve.audit.items()):
        if how in ("fuzzy", "new"):
            print(f"  {how:5} {name} -> {found}")
    manual_ok = names.set_index("id").name.groupby(level=0).first()
    print("\nmanual ids resolve to:")
    for espn_name, club_id in MANUAL.items():
        print(f"  {espn_name} -> {manual_ok.get(club_id, 'NOT IN ARCHIVE')}")
    print("\nother leagues seen in ESPN:",
          sorted(set(played.loc[played.competition_id == "OTHER", "comp_group"])))


if __name__ == "__main__":
    main()
