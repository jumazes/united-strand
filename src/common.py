"""Shared paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

MU_ESPN_ID = "360"
MU_TM_ID = "985"
CHALLENGE_START = "2024-10-05"

# Forecast cut-off: results after this date are not used. Set when the data snapshot was taken.
SNAPSHOT_DATE = "2026-11-06"

# Competitions the classifier is trained and scored on: the kinds of match United actually plays.
MODEL_COMPS = {
    "GB1", "ES1", "IT1", "L1", "FR1",        # top-5 leagues
    "CL", "EL", "UCOL",                      # UEFA
    "FAC", "CGB", "CDR", "DFB", "CIT",       # domestic cups
}
TOP5 = {"GB1", "ES1", "IT1", "L1", "FR1"}
EUROPE = {"CL", "CLQ", "EL", "ELQ", "UCOL", "ECLQ", "USC"}

# football-data team names -> Transfermarkt club ids.
FD_TEAMS = {
    "Arsenal": "11", "Aston Villa": "405", "Bournemouth": "989", "Brentford": "1148", "Brighton": "1237",
    "Burnley": "1132", "Chelsea": "631", "Coventry": "990", "Crystal Palace": "873", "Everton": "29",
    "Fulham": "931", "Hull": "3008", "Ipswich": "677", "Leeds": "399", "Leicester": "1003",
    "Liverpool": "31", "Luton": "1031", "Man City": "281", "Man United": "985", "Newcastle": "762",
    "Norwich": "1123", "Nott'm Forest": "703", "Sheffield United": "350", "Southampton": "180",
    "Sunderland": "289", "Tottenham": "148", "Watford": "1010", "West Brom": "984", "West Ham": "379",
    "Wolves": "543",
}


def season_of(date: str) -> int:
    """Football season by start year: 2026-11-06 -> 2026, 2027-02-01 -> 2026."""
    year, month = int(date[:4]), int(date[5:7])
    return year if month >= 7 else year - 1
