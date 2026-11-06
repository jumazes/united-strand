# When will Frank get a haircut? Why Manchester United cannot win five in a row

Since 5 October 2024, blogger Frank Ilett (The United Strand) has refused to cut his hair until Manchester United win five competitive matches in a row. After 104 matches it has not happened. We use the story as a precise, easy-to-understand test: what stops a team from building a run?

**User:** Manchester United's coaching staff.
**Question:** which aspects of play should be improved first to raise the chance of a five-win run by the end of the season, and how realistic is that?
**Decision:** which two or three tactical changes to test first.

Data snapshot: 6 November 2026.

## Team

Lab work for the AI Essentials course, Astana IT University, group AAI-2606M.

- Aslan Zhumazhanov
- Alikhan Khamidulla
- Zhandarbek Zhetpissov
- Aman Nurgozhiyev

## Demo

**[Open the live demo](https://jumazes.github.io/united-strand/)**

The page runs the simulation in your browser: change United's strength, set match results, and watch the chance update. To use it offline, open [docs/index.html](docs/index.html).

## Method

1. **Elo rating** across 88,000 matches, computed in date order from past results only.
2. **Logistic regression** gives win, draw and loss probabilities. Features: Elo difference, neutral venue, competition type, rest days.
3. **Monte Carlo simulation** over the real calendar: 200,000 versions of the season, with United's rating changing during each one.
4. **Indicator analysis:** xG, shots, opponent types, comparison with other clubs.

## Validation

Training uses past seasons only. Testing covers 2019/20 to 2026/27 (18,490 matches).

| Model | Log loss, all competitions | Log loss, Premier League | Accuracy, Premier League |
|---|---|---|---|
| Outcome frequencies | 1.068 | 1.071 | 43.3% |
| Elo | 0.981 | 0.989 | 52.4% |
| **Elo + context (chosen)** | **0.974** | **0.990** | **52.7%** |
| Elo + context + form | 0.973 | 0.990 | 52.6% |
| Gradient boosting | 0.973 | 0.992 | 52.9% |
| Bet365 closing odds | no data | 0.966 | 54.7% |

The model loses to the bookmaker, which probably reflects line-ups and injuries. It also covers cups and European matches, where full odds are missing.

**Testing the run forecast itself:** the same calculation from 6 November for 680 team-seasons in the top five leagues. Brier score (lower is better): model 0.113, win rate so far 0.137, constant rate 0.180. In the 5–20% band, where United sit, the forecast was 10.2% and the outcome 10.6%.

## Findings

- **There is no curse.** The model expected 42.8 wins and United have 43. The chance of seeing no run in 104 matches is 49%.
- **No momentum effect is visible.** After a run of wins, teams win about as often as the model predicts.
- **The chance of a haircut by the end of the season is about 11%.** The range is 6–13% depending on assumptions, and the low end keeps United's shortfall against weaker sides.
- **United's total hides two opposite patterns.** Against stronger sides they win 50% where the model expects 24%. Against weaker sides they win 42% where it expects 55%.
- **How runs end.** 9 of 10 runs of two or more wins ended away from home, and 4 of those were away draws with weaker sides.

## What to test first (hypotheses, not proven causes)

1. **Shot quality away to weaker sides.** In the Premier League, United's 16.9 shots produce 1.43 xG, while other clubs' 13.7 shots produce 1.61 xG. Sample: 16 matches.
2. **Defending.** Opponents' xG rose from 1.29 to 1.66 per match, two thirds of the drop in xG difference compared with last season. Current-season sample: 9 league matches.
3. **Rotation on short rest.** With three days' rest or fewer, United win 29% where the model expects 44%. The effect is not statistically significant, so this is only an observation.

**What this data does not support changing:** crossing and recent finishing. The link between crosses and results disappears when crosses are measured in previous matches (r falls from -0.30 to -0.02), which suggests that losing teams simply cross more. Recent finishing does not improve the forecast.

**How realistic the target is:** even at United's best rating of the challenge (1736, now 1643), the chance of a run is about 27%. A 50% chance needs a rating near 1830, close to Manchester City.

## Limitations

- **Small samples:** United's samples are small and we made many comparisons, so individual splits may be chance.
- **Shortfall against weaker sides:** the base forecast excludes it. If it persists, the chance falls from 11% to about 6%. The demo has a switch for this.
- **Causation:** all links are observational. Recommendations need testing on the pitch.
- **Data:**
  - the Transfermarkt archive stops on 6 July 2026, and ESPN is used after that;
  - second divisions are missing, so ratings of promoted clubs are approximate;
  - xG covers the Premier League only and comes from summed FPL player xG;
  - the archive lacks the 2024/25 League Cup, and only United's two matches were added;
  - ESPN and FPL are unofficial, so the data should be kept as a snapshot.
- **Calendar:** future cup matches are unknown. With three assumed matches the chance is 13%.
- **Backtest:** it uses the real calendar, including cup ties that would not have been known in advance.
- **Rule:** a shootout win counts as a draw, and a win after extra time counts as a win.

## Running it

```
pip install -r requirements.txt
python src/download.py
python src/prepare.py
python src/evaluate.py
python src/simulate.py
python src/drivers.py
python src/build_demo.py
```

Result tables are saved to `outputs/`, and the website is built into `docs/`.

Raw data is not stored in this repository because of its size. `download.py` fetches it again, but the Transfermarkt archive and ESPN change over time, so a new download may not reproduce these numbers exactly. The processed tables and results from the 6 November 2026 snapshot are committed in `data/processed/` and `outputs/`.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Data pipeline, model, simulation, analysis and page build |
| `demo/template.html` | Source of the demo page |
| `docs/` | The built website, served by GitHub Pages |
| `data/processed/` | Clean match tables |
| `outputs/` | Result tables behind every number in this README |

## Data sources

- [Transfermarkt datasets](https://github.com/dcaribou/transfermarkt-datasets) (CC0): matches and events since 2012.
- [football-data.co.uk](https://www.football-data.co.uk/): Premier League results, shots, odds, and xG for 2026/27.
- [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League): player xG.
- ESPN (unofficial API): current season, calendar, United's match statistics.
