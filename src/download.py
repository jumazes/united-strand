"""Download the raw data snapshot into data/raw. Existing files are kept unless --force is given."""
import json
import sys
import time
import urllib.request

from common import CHALLENGE_START, MU_ESPN_ID, RAW

UA = {"User-Agent": "Mozilla/5.0 (course-project)"}
TM = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/{}.csv.gz"
FD = "https://football-data.co.uk/mmz4281/{}/E0.csv"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{}/schedule?{}"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/soccer/{}/summary?event={}"
FPL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{}/gws/merged_gw.csv"
FD_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
FPL_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
ESPN_LEAGUES = {
    "English Premier League": "eng.1", "UEFA Champions League": "uefa.champions",
    "UEFA Europa League": "uefa.europa", "English FA Cup": "eng.fa", "English Carabao Cup": "eng.league_cup",
}
FORCE = "--force" in sys.argv


def fetch(url):
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def save(url, path):
    if path.exists() and not FORCE:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = fetch(url)
    path.write_bytes(body)
    print(f"{path.relative_to(RAW)}: {len(body):,} bytes")
    time.sleep(0.3)
    return body


def schedule(team_id, query, name):
    return json.loads(save(ESPN.format(team_id, query), RAW / "espn" / f"{team_id}_{name}.json"))


def match_stats():
    """Per-match statistics for every United competitive match of the challenge."""
    for season in (2024, 2025, 2026):
        path = RAW / "espn" / f"{MU_ESPN_ID}_{season}.json"
        for event in json.loads(path.read_bytes())["events"]:
            league = ESPN_LEAGUES.get(event["league"]["name"])
            done = event["competitions"][0]["status"]["type"].get("completed")
            if league and done and event["date"][:10] >= CHALLENGE_START:
                save(ESPN_SUMMARY.format(league, event["id"]), RAW / "espn_summary" / f"{event['id']}.json")


def main():
    for table in ("games", "game_events"):
        save(TM.format(table), RAW / "transfermarkt" / f"{table}.csv.gz")
    for season in FD_SEASONS:
        save(FD.format(season), RAW / "football_data" / f"E0_{season}.csv")
    for season in FPL_SEASONS:
        save(FPL.format(season), RAW / "fpl" / f"merged_gw_{season}.csv")

    # United: the whole challenge plus the remaining fixtures.
    for season in (2024, 2025, 2026):
        schedule(MU_ESPN_ID, f"season={season}", str(season))
    fixtures = schedule(MU_ESPN_ID, "fixture=true", "fixtures")

    # Every opponent on the remaining calendar, for current-season ratings and rest days.
    teams = set()
    for event in fixtures["events"]:
        for competitor in event["competitions"][0]["competitors"]:
            teams.add(competitor["team"]["id"])
    teams.discard(MU_ESPN_ID)
    print(f"{len(teams)} opponents on the remaining calendar")
    for team_id in sorted(teams):
        schedule(team_id, "season=2026", "2026")
        schedule(team_id, "fixture=true", "fixtures")

    match_stats()


if __name__ == "__main__":
    main()
