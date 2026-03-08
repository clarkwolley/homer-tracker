"""
Homer Tracker configuration.

Centralized settings for API endpoints, seasons, and model parameters.
"""

# MLB Stats API (free, no key required)
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Default season
CURRENT_SEASON = 2025

# Feature engineering defaults
ROLLING_WINDOW = 15      # games for rolling averages (larger window for 162-game season)
MIN_GAMES_PLAYED = 10    # minimum games to include a batter
STREAK_MIN_GAMES = 2     # minimum consecutive games to count as a "streak"

# HR rate thresholds
HIGH_POWER_THRESHOLD = 0.06  # HR per AB above this = power hitter (~25+ HR pace)

# Model defaults
TEST_SIZE = 0.2       # 20% of data held out for testing
RANDOM_STATE = 42     # reproducibility

# Recency weighting — recent games matter more than ancient history
RECENCY_FULL_WEIGHT_GAMES = 162   # ~1 full season gets weight 1.0
RECENCY_MIN_WEIGHT = 0.15         # oldest data never drops below this
RECENCY_DECAY_RATE = 0.004        # exponential decay speed for older games

# Park HR factors (2024 data — higher = more HR-friendly)
# Source: ESPN Park Factors / Baseball Savant
PARK_HR_FACTORS = {
    "COL": 1.35,  # Coors Field — the juice box
    "CIN": 1.18,  # Great American Ball Park
    "NYY": 1.12,  # Yankee Stadium — short porch
    "BOS": 1.10,  # Fenway Park
    "CHC": 1.08,  # Wrigley Field
    "TEX": 1.06,  # Globe Life Field
    "PHI": 1.05,  # Citizens Bank Park
    "TOR": 1.04,  # Rogers Centre
    "BAL": 1.03,  # Camden Yards
    "MIL": 1.02,  # American Family Field
    "ATL": 1.01,  # Truist Park
    "LAA": 1.00,  # Angel Stadium
    "MIN": 1.00,  # Target Field
    "HOU": 0.99,  # Minute Maid Park
    "ARI": 0.98,  # Chase Field
    "WSH": 0.98,  # Nationals Park
    "DET": 0.97,  # Comerica Park
    "CLE": 0.97,  # Progressive Field
    "KC": 0.96,   # Kauffman Stadium
    "NYM": 0.95,  # Citi Field
    "LAD": 0.95,  # Dodger Stadium
    "PIT": 0.94,  # PNC Park
    "STL": 0.94,  # Busch Stadium
    "TB": 0.93,   # Tropicana Field
    "CWS": 0.92,  # Guaranteed Rate Field
    "SEA": 0.91,  # T-Mobile Park
    "SD": 0.90,   # Petco Park
    "SF": 0.88,   # Oracle Park
    "MIA": 0.87,  # loanDepot park
    "OAK": 0.90,  # Oakland Coliseum
}
