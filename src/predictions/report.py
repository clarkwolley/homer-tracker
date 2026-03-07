"""
HTML report generator for daily HR predictions.

Creates a shareable, styled HTML page with tonight's picks.
Baseball-themed dark mode design. ⚾
"""

import os
from datetime import datetime

import pandas as pd


REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")


def _ensure_report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


def _tier_label(prob: float) -> str:
    if prob >= 0.15:
        return "💣 BOMB"
    if prob >= 0.10:
        return "🎯 STRONG"
    if prob >= 0.06:
        return "👀 WATCH"
    return "📋 LONG SHOT"


def _tier_class(prob: float) -> str:
    if prob >= 0.15:
        return "fire"
    if prob >= 0.10:
        return "strong"
    if prob >= 0.06:
        return "watch"
    return "longshot"


def generate_html_report(
    pred_df: pd.DataFrame,
    top_n: int = 30,
    game_df: pd.DataFrame = None,
) -> str:
    """Generate a full HTML report from prediction data."""
    _ensure_report_dir()

    today = datetime.now().strftime("%Y-%m-%d")
    display = pred_df.head(top_n).copy()
    display["prob_pct"] = (display["hr_probability"] * 100).round(1)
    display["matchup"] = display.apply(
        lambda r: f"{'vs' if r['is_home'] else '@'} {r['opponent']}", axis=1
    )
    display["tier"] = display["hr_probability"].apply(_tier_label)
    display["tier_class"] = display["hr_probability"].apply(_tier_class)

    def _streak_badge(row):
        if row.get("is_hot", 0):
            return f'<span class="streak-hot">🔥 {int(row.get("hr_streak", 0))}G</span>'
        if row.get("hr_drought", 0) >= 10:
            return f'<span class="streak-cold">❄️ {int(row.get("hr_drought", 0))}G</span>'
        return ""

    display["streak_badge"] = display.apply(_streak_badge, axis=1)

    # Build player rows
    player_rows = ""
    for i, (_, row) in enumerate(display.iterrows(), 1):
        park = row.get("park_hr_factor", 1.0)
        park_class = "park-hot" if park >= 1.05 else ("park-cold" if park <= 0.92 else "")
        player_rows += f"""        <tr class="{row['tier_class']}">
            <td class="rank">{i}</td>
            <td class="player">{row['name']} {row['streak_badge']}</td>
            <td>{row['team']}</td>
            <td>{row['position']}</td>
            <td>{row['matchup']}</td>
            <td class="prob"><div class="prob-bar" style="width: {min(row['prob_pct'] * 5, 100)}%">{row['prob_pct']}%</div></td>
            <td>{int(row.get('season_hr', 0))}</td>
            <td>{row.get('season_ops', 0):.3f}</td>
            <td>{row.get('opp_pitcher_name', 'TBD')}</td>
            <td class="{park_class}">{park:.2f}</td>
            <td class="tier-badge">{row['tier']}</td>
        </tr>
"""

    # Game winner section
    game_html = ""
    if game_df is not None and not game_df.empty:
        game_cards = ""
        for _, g in game_df.iterrows():
            conf = g["confidence"]
            conf_class = "fire" if conf >= 60 else ("strong" if conf >= 55 else "longshot")
            bar_w = g["home_win_prob"]
            game_cards += f"""        <div class="game-winner-card {conf_class}">
            <div class="gw-matchup">{g['away_team']} @ {g['home_team']}</div>
            <div class="gw-pick">{'🏠' if g['home_win_prob'] > 50 else '✈️'} <strong>{g['predicted_winner']}</strong> ({conf}%)</div>
            <div class="gw-bar-container">
                <div class="gw-bar-home" style="width:{bar_w}%">{g['home_team']} {g['home_win_prob']}%</div>
                <div class="gw-bar-away" style="width:{100-bar_w}%">{g['away_team']} {g['away_win_prob']}%</div>
            </div>
        </div>
"""
        game_html = f"""        <h2>🏆 Game Winner Predictions</h2>
        <div class="gw-grid">{game_cards}</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚾ Homer Tracker — {today}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e17; color: #e2e8f0; padding: 2rem; line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            text-align: center; margin-bottom: 2rem; padding: 2rem;
            background: linear-gradient(135deg, #1a1f35 0%, #0d1322 100%);
            border-radius: 16px; border: 1px solid #2a3352;
        }}
        header h1 {{ font-size: 2.2rem; margin-bottom: 0.5rem; }}
        header .date {{ color: #94a3b8; font-size: 1.1rem; }}
        .disclaimer {{
            background: #1c1917; border: 1px solid #78350f; border-radius: 8px;
            padding: 0.75rem 1rem; margin: 1.5rem 0; font-size: 0.85rem; color: #fbbf24;
        }}
        table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; font-size: 0.9rem; }}
        th {{
            background: #1e293b; color: #94a3b8; padding: 0.75rem 0.5rem;
            text-align: left; font-weight: 600; text-transform: uppercase;
            font-size: 0.75rem; letter-spacing: 0.05em; border-bottom: 2px solid #334155;
        }}
        td {{ padding: 0.6rem 0.5rem; border-bottom: 1px solid #1e293b; }}
        tr:hover {{ background: #1e293b; }}
        tr.fire td {{ border-left: 3px solid #ef4444; }}
        tr.strong td {{ border-left: 3px solid #f59e0b; }}
        tr.watch td {{ border-left: 3px solid #3b82f6; }}
        tr.longshot td {{ border-left: 3px solid #475569; }}
        .rank {{ color: #64748b; font-weight: 600; width: 30px; }}
        .player {{ font-weight: 600; color: #f1f5f9; }}
        .prob {{ width: 120px; }}
        .prob-bar {{
            background: linear-gradient(90deg, #22c55e, #16a34a); color: #fff;
            padding: 2px 8px; border-radius: 4px; font-weight: 700;
            font-size: 0.85rem; text-align: right; min-width: 50px; display: inline-block;
        }}
        tr.fire .prob-bar {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
        tr.strong .prob-bar {{ background: linear-gradient(90deg, #f59e0b, #d97706); }}
        tr.watch .prob-bar {{ background: linear-gradient(90deg, #3b82f6, #2563eb); }}
        .tier-badge {{ font-size: 0.8rem; white-space: nowrap; }}
        .park-hot {{ color: #fca5a5; font-weight: 600; }}
        .park-cold {{ color: #93c5fd; font-weight: 600; }}
        h2 {{ font-size: 1.4rem; margin: 2.5rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #1e293b; }}
        .streak-hot {{ background: #7f1d1d; color: #fca5a5; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }}
        .streak-cold {{ background: #1e3a5f; color: #93c5fd; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }}
        .gw-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 1rem; margin-top: 1rem; }}
        .game-winner-card {{ background: #1a1f35; border: 1px solid #2a3352; border-radius: 12px; padding: 1rem; }}
        .game-winner-card.fire {{ border-left: 3px solid #22c55e; }}
        .game-winner-card.strong {{ border-left: 3px solid #f59e0b; }}
        .gw-matchup {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 0.4rem; }}
        .gw-pick {{ font-size: 0.95rem; margin-bottom: 0.6rem; color: #94a3b8; }}
        .gw-pick strong {{ color: #f1f5f9; }}
        .gw-bar-container {{ display: flex; border-radius: 6px; overflow: hidden; height: 24px; font-size: 0.75rem; }}
        .gw-bar-home {{ background: #3b82f6; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 600; }}
        .gw-bar-away {{ background: #64748b; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 600; }}
        footer {{ text-align: center; margin-top: 3rem; padding: 1.5rem; color: #475569; font-size: 0.8rem; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚾ Homer Tracker</h1>
            <div class="date">Home Run Predictions — {today}</div>
        </header>
        <div class="disclaimer">
            ⚠️ For entertainment purposes only. Never bet more than you can afford to lose.
        </div>
        <h2>💣 Top {top_n} Most Likely Home Run Hitters</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th><th>Player</th><th>Team</th><th>Pos</th><th>Matchup</th>
                    <th>HR Prob</th><th>Season HR</th><th>OPS</th><th>vs Pitcher</th>
                    <th>Park</th><th>Tier</th>
                </tr>
            </thead>
            <tbody>
{player_rows}
            </tbody>
        </table>
{game_html}
        <footer>
            Homer Tracker · Built with Python, scikit-learn & the MLB Stats API<br>
            Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
        </footer>
    </div>
</body>
</html>"""

    filepath = os.path.join(REPORT_DIR, f"picks_{today}.html")
    with open(filepath, "w") as f:
        f.write(html)

    print(f"\n📄 Report saved to: {filepath}")
    return filepath
