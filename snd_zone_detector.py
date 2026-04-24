"""
BTC Supply & Demand Zone Detector v1.0
Detects RBR (Demand) and DBD (Supply) zones on 1H BTC/USDT

Scoring (max 12):
  - Base Width (0-3): tighter = stronger zone
  - Move Strength (0-3): bigger displacement = more imbalance  
  - Low Volume (0-2): quiet base = institutional accumulation
  - Freshness (0-2): recent zones more relevant
  - Touches (0-2): retests = validation

Usage:
  python3 snd_zone_detector.py

Output:
  - Console: ranked zone list
  - /tmp/btc_snd_zones_quality.png: chart with zones drawn
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / 'data' / 'binance' / 'futures' / 'BTC_USDT_USDT-1h-futures.json'
CHART_OUTPUT = Path('/tmp/btc_snd_zones_quality.png')

# ── Config ──
SWING_LOOKBACK = 5        # candles each side for swing detection
MAX_BASE_CANDLES = 40     # max candles in a base zone
MIN_BASE_CANDLES = 2
MIN_SCORE = 5             # only show zones with this score or higher
NEAR_PRICE_PCT = 3        # highlight zones within X% of current price
RALLY_LOOKAHEAD = 30      # candles to check for rally/drop after zone
TOUCH_LOOKAHEAD = 50      # candles to count touches
TOUCH_TOLERANCE = 0.003   # 0.3% tolerance for touch counting
VOLUME_LOOKBACK = 50      # candles for average volume comparison


def load_data(data_path: str, months_back: int = 4):
    """Load 1H candles from JSON, return last N months."""
    with open(data_path) as f:
        data = json.load(f)
    
    cutoff = (datetime.now() - timedelta(days=months_back*30) - datetime(1970, 1, 1)).total_seconds() * 1000
    candles = [c for c in data if c[0] >= cutoff]
    return candles


def detect_swings(highs, lows, lb):
    """Find swing lows and highs using lookback window."""
    N = len(highs)
    swing_lows = [i for i in range(lb, N-lb) if lows[i] == min(lows[i-lb:i+lb+1])]
    swing_highs = [i for i in range(lb, N-lb) if highs[i] == max(highs[i-lb:i+lb+1])]
    return swing_lows, swing_highs


def score_zone(base_low, base_high, move_size, base_vol, prev_vol, age_hours, touches, base_width_pct):
    """Score a zone 0-12."""
    score = 0
    
    # Base Width (0-3)
    if base_width_pct < 0.3: score += 3
    elif base_width_pct < 0.7: score += 2
    elif base_width_pct < 1.5: score += 1
    
    # Move Strength (0-3)
    if move_size > 4: score += 3
    elif move_size > 2: score += 2
    elif move_size > 0.8: score += 1
    
    # Low Volume (0-2)
    vol_ratio = base_vol / prev_vol if prev_vol > 0 else 1
    if vol_ratio < 0.6: score += 2
    elif vol_ratio < 0.85: score += 1
    
    # Freshness (0-2)
    if age_hours < 50: score += 2
    elif age_hours < 150: score += 1
    
    # Touches (0-2)
    if touches >= 3: score += 2
    elif touches >= 2: score += 1
    
    return score


def find_zones(highs, lows, closes, vols, swing_lows, swing_highs, N):
    """Detect RBR demand zones and DBD supply zones."""
    demand = []
    supply = []
    
    # DEMAND: Rally-Base-Rally
    for i in range(len(swing_lows) - 1):
        a, b = swing_lows[i], swing_lows[i+1]
        if b - a > MAX_BASE_CANDLES or b - a < MIN_BASE_CANDLES:
            continue
        if min(lows[a:b+1]) < lows[a]:
            continue
        
        end = min(b + RALLY_LOOKAHEAD, N)
        rally_high = max(highs[b:end])
        if rally_high <= max(highs[a:b+1]):
            continue
        
        base_low = min(lows[a:b+1])
        base_high = max(highs[a:b+1])
        base_width = (base_high - base_low) / base_low * 100
        rally_pct = (rally_high - base_low) / base_low * 100
        
        base_vol = np.mean(vols[a:b+1])
        prev_vol = np.mean(vols[max(0, a-VOLUME_LOOKBACK):a+1])
        age = N - b
        
        touches = sum(1 for k in range(a, min(b+TOUCH_LOOKAHEAD, N))
                      if lows[k] <= base_low * (1+TOUCH_TOLERANCE) and lows[k] >= base_low * (1-TOUCH_TOLERANCE))
        
        sc = score_zone(base_low, base_high, rally_pct, base_vol, prev_vol, age, touches, base_width)
        demand.append({
            'start_idx': a, 'end_idx': b,
            'low': base_low, 'high': base_high,
            'score': sc, 'touches': touches,
            'width_pct': base_width, 'age_hours': age,
            'move_pct': rally_pct
        })
    
    # SUPPLY: Drop-Base-Drop
    for i in range(len(swing_highs) - 1):
        a, b = swing_highs[i], swing_highs[i+1]
        if b - a > MAX_BASE_CANDLES or b - a < MIN_BASE_CANDLES:
            continue
        if max(highs[a:b+1]) > highs[a]:
            continue
        
        end = min(b + RALLY_LOOKAHEAD, N)
        drop_low = min(lows[b:end])
        if drop_low >= min(lows[a:b+1]):
            continue
        
        base_low = min(lows[a:b+1])
        base_high = max(highs[a:b+1])
        base_width = (base_high - base_low) / base_low * 100
        drop_pct = (base_high - drop_low) / base_high * 100
        
        base_vol = np.mean(vols[a:b+1])
        prev_vol = np.mean(vols[max(0, a-VOLUME_LOOKBACK):a+1])
        age = N - b
        
        touches = sum(1 for k in range(a, min(b+TOUCH_LOOKAHEAD, N))
                      if highs[k] >= base_high * (1-TOUCH_TOLERANCE) and highs[k] <= base_high * (1+TOUCH_TOLERANCE))
        
        sc = score_zone(base_low, base_high, drop_pct, base_vol, prev_vol, age, touches, base_width)
        supply.append({
            'start_idx': a, 'end_idx': b,
            'low': base_low, 'high': base_high,
            'score': sc, 'touches': touches,
            'width_pct': base_width, 'age_hours': age,
            'move_pct': drop_pct
        })
    
    demand.sort(key=lambda z: -z['score'])
    supply.sort(key=lambda z: -z['score'])
    return demand, supply


def main():
    print("Loading data...")
    candles = load_data(DATA_PATH, months_back=4)
    N = len(candles)
    
    dates = [datetime.fromtimestamp(c[0]/1000) for c in candles]
    opens = np.array([c[1] for c in candles])
    highs = np.array([c[2] for c in candles])
    lows = np.array([c[3] for c in candles])
    closes = np.array([c[4] for c in candles])
    vols = np.array([c[5] for c in candles])
    current_price = closes[-1]
    
    print(f"BTC/USDT 1H | {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({N} candles)")
    print(f"Current Price: ${current_price:,.0f}\n")
    
    swing_lows, swing_highs = detect_swings(highs, lows, SWING_LOOKBACK)
    print(f"Swing Lows: {len(swing_lows)} | Swing Highs: {len(swing_highs)}")
    
    demand, supply = find_zones(highs, lows, closes, vols, swing_lows, swing_highs, N)
    
    top_demand = [z for z in demand if z['score'] >= MIN_SCORE]
    top_supply = [z for z in supply if z['score'] >= MIN_SCORE]
    
    print(f"\nDemand (RBR): {len(demand)} found, {len(top_demand)} scored >= {MIN_SCORE}")
    print(f"Supply (DBD): {len(supply)} found, {len(top_supply)} scored >= {MIN_SCORE}")
    
    # Print top zones
    print(f"\n{'='*70}")
    print("DEMAND ZONES (RBR) - Buy Zones")
    print(f"{'='*70}")
    for z in top_demand[:10]:
        bars = '|' * z['score'] + '.' * (12 - z['score'])
        near = " << NEAR PRICE" if abs(current_price - (z['low']+z['high'])/2) / current_price * 100 < NEAR_PRICE_PCT else ""
        print(f"  [{bars}] {z['score']}/12 | ${z['low']:,.0f} - ${z['high']:,.0f} ({z['width_pct']:.2f}%) | "
              f"{dates[z['start_idx']].strftime('%b %d')} | move={z['move_pct']:.1f}% | touch={z['touches']}{near}")
    
    print(f"\n{'='*70}")
    print("SUPPLY ZONES (DBD) - Sell Zones")
    print(f"{'='*70}")
    for z in top_supply[:10]:
        bars = '|' * z['score'] + '.' * (12 - z['score'])
        near = " << NEAR PRICE" if abs(current_price - (z['low']+z['high'])/2) / current_price * 100 < NEAR_PRICE_PCT else ""
        print(f"  [{bars}] {z['score']}/12 | ${z['low']:,.0f} - ${z['high']:,.0f} ({z['width_pct']:.2f}%) | "
              f"{dates[z['start_idx']].strftime('%b %d')} | move={z['move_pct']:.1f}% | touch={z['touches']}{near}")
    
    # Near price summary
    near_d = [z for z in top_demand if abs(current_price - (z['low']+z['high'])/2) / current_price * 100 < NEAR_PRICE_PCT]
    near_s = [z for z in top_supply if abs(current_price - (z['low']+z['high'])/2) / current_price * 100 < NEAR_PRICE_PCT]
    if near_d or near_s:
        print(f"\n>> ZONES NEAR ${current_price:,.0f} (+/- {NEAR_PRICE_PCT}%) <<")
        for z in near_d:
            print(f"  BUY  ${z['low']:,.0f} - ${z['high']:,.0f} ({z['score']}/12)")
        for z in near_s:
            print(f"  SELL ${z['low']:,.0f} - ${z['high']:,.0f} ({z['score']}/12)")
    
    print(f"\nChart: {CHART_OUTPUT}")
    return demand, supply, dates, highs, lows, opens, closes, vols, current_price, top_demand, top_supply, swing_lows, swing_highs


if __name__ == '__main__':
    main()
