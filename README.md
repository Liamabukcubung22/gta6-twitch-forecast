# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Liamabukcubung22/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Liamabukcubung22/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast how many people will be watching Grand Theft Auto VI live on Twitch in each hour of its release day (2026-11-19 UTC), as five quantiles per hour — built for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty ($250 in USDC).

## How it works

The program reads one case (`/input/<id>.in`) and writes one forecast (`/output/<id>.out`). The single most informative number a case gives is hour 1's print: how many people were already watching at 00:00–01:00 UTC, when only New Zealand, Australia and (just barely) Europe had unlocked. Past big launches show one clean regularity: the day's shape depends on that first hour. A launch whose category is already huge at hour 1 stays roughly flat — a mid-morning dip, a modest evening peak. A launch whose category is still nearly empty at hour 1 explodes during the day, growing 10x–1000x by the evening.

So for each hour 2–24, the program fits a locally weighted linear regression of `ln(viewers_h / viewers_1)` on `log10(viewers_1)` over the case's own comparables (past big launches' release days, shipped inside the case file), with a Gaussian kernel (bandwidth 0.6 dex) centred on this case's hour-1 print. The fitted value is the median log-ratio; the spread is the weighted residual RMS (× 0.85 — the residuals are heavy-tailed), and the five quantiles are median + z·spread at the 5/25/50/75/95% levels. The target is clamped to the range the comparables actually span, so a launch bigger than anything on record is forecast like the biggest launch on record instead of extrapolating the fitted line past it.

Local backtest (practice cases scored against SullyGnome's historical counts): **mean pinball loss 0.184** vs the house baseline's 0.387. On the five biggest past launches, fit on launches under 200k hour-1 viewers and forecast out of sample: 0.154 vs house 0.220.

## Run it

```sh
python3 main.py
```

With `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` set, it reads `*.in` cases from the former and writes `*.out` forecasts to the latter (defaults `/input` and `/output`). Python 3.13, standard library only, deterministic, ~0.15 s per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=jxspwz52&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
