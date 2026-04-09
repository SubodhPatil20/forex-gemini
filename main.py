"""
Forex Binary Prediction Bot V3 (Advanced)
Upgrades:
  - Google Sheets removed (Telegram only)
  - "Dead Zone" training filter (removes 1-pip noise)
  - Distance-to-EMA metrics (rubber band effect)
  - Cyclical time features (sine/cosine for hours/mins)
  - Normalized ATR (replaces raw volume)
  - Multi-Timeframe Feature injection (15m RSI fed to ML)
  - Triple-Model Voting (XGBoost + RandomForest + LogisticRegression)
"""

import os, json, time, logging, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import pytz
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv('TELEGRAM_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')

PRIMARY_PAIR = 'EURUSD=X'
PAIRS        = [PRIMARY_PAIR]

TF_SIGNAL = '5m'
TF_HIGH   = '15m'
TF_LOW    = '1m'

CONFIDENCE_THRESHOLD   = 0.75   # Slightly lowered because 3 models agreeing is very rare/strong
ADX_MIN_DEFAULT        = 25
DEAD_ZONE_PIPS         = 0.0001 # 1 pip for EURUSD
SIGNAL_COOLDOWN_MINS   = 10
LOOP_INTERVAL_SECS     = 300
RESCAN_INTERVAL_SECS   = 1800
TRAIN_CANDLES_5M       = 2000

IST = pytz.timezone('Asia/Kolkata')
IST_SESSIONS = [
    (13, 30, 15, 30),   # London open
    (18, 30, 22, 30),   # London+NY overlap
]

# ─────────────────────────────────────────
# MANUAL INDICATORS
# ─────────────────────────────────────────
def calc_rsi(close, p=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / (l + 1e-10)))

def calc_ema(close, p):
    return close.ewm(span=p, adjust=False).mean()

def calc_macd(close, fast=12, slow=26, sig=9):
    m = calc_ema(close, fast) - calc_ema(close, slow)
    s = calc_ema(m, sig)
    return m, s, m - s

def calc_atr(high, low, close, p=14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calc_adx(high, low, close, p=14):
    tr  = calc_atr(high, low, close, p)
    up  = high.diff()
    dn  = -low.diff()
    dp  = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=close.index).rolling(p).mean()
    dm  = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=close.index).rolling(p).mean()
    dip = 100 * dp / (tr + 1e-10)
    dim = 100 * dm / (tr + 1e-10)
    dx  = 100 * (dip - dim).abs() / (dip + dim + 1e-10)
    return dx.rolling(p).mean(), dip, dim

def calc_stoch(high, low, close, k=14, d=3):
    sk = 100 * (close - low.rolling(k).min()) / (high.rolling(k).max() - low.rolling(k).min() + 1e-10)
    return sk, sk.rolling(d).mean()

# ─────────────────────────────────────────
# DATA FETCHER
# ─────────────────────────────────────────
class DataFetcher:
    def fetch(self, symbol, interval, candles):
        try:
            pm = {'1m': '7d', '5m': '60d', '15m': '60d'}
            df = yf.download(symbol, period=pm.get(interval, '60d'), interval=interval, progress=False, auto_adjust=True)
            if df.empty or len(df) < 50: return None
            df = df.tail(candles).copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.debug(f"Fetch {symbol} {interval}: {e}")
            return None

# ─────────────────────────────────────────
# FEATURE BUILDER (V3 Upgrades)
# ─────────────────────────────────────────
class FeatureBuilder:
    def build(self, df5, df15=None):
        try:
            d = df5.copy()
            d['rsi'] = calc_rsi(d['close'])
            d['macd'], d['macd_signal'], d['macd_hist'] = calc_macd(d['close'])
            d['ema9']  = calc_ema(d['close'], 9)
            d['ema21'] = calc_ema(d['close'], 21)
            d['ema50'] = calc_ema(d['close'], 50)
            
            # Distance to EMAs (Rubber band effect)
            d['dist_ema9']  = (d['close'] - d['ema9']) / d['ema9']
            d['dist_ema21'] = (d['close'] - d['ema21']) / d['ema21']
            d['dist_ema50'] = (d['close'] - d['ema50']) / d['ema50']
            
            d['atr'] = calc_atr(d['high'], d['low'], d['close'])
            d['atr_norm'] = d['atr'] / d['close'] # Replaces fake volume
            d['adx'], _, _ = calc_adx(d['high'], d['low'], d['close'])
            d['stoch_k'], d['stoch_d'] = calc_stoch(d['high'], d['low'], d['close'])
            
            # Cyclical Time Features
            d['hour_sin'] = np.sin(2 * np.pi * d.index.hour / 24)
            d['hour_cos'] = np.cos(2 * np.pi * d.index.hour / 24)
            d['min_sin']  = np.sin(2 * np.pi * d.index.minute / 60)
            d['min_cos']  = np.cos(2 * np.pi * d.index.minute / 60)

            # Multi-Timeframe Injection
            if df15 is not None:
                df15_features = pd.DataFrame(index=df15.index)
                df15_features['rsi_15m'] = calc_rsi(df15['close'])
                # Forward fill 15m data onto 5m index
                df15_features = df15_features.reindex(d.index, method='ffill')
                d['rsi_15m'] = df15_features['rsi_15m']
            else:
                d['rsi_15m'] = 50.0

            d.dropna(inplace=True)
            
            # Select strictly numeric features for ML
            features = [
                'rsi', 'macd_hist', 'dist_ema9', 'dist_ema21', 'dist_ema50', 
                'atr_norm', 'adx', 'stoch_k', 'stoch_d', 
                'hour_sin', 'hour_cos', 'min_sin', 'min_cos', 'rsi_15m'
            ]
            return d[features], d
        except Exception as e:
            logger.debug(f"Feature build error: {e}")
            return None, None

    def get_labels(self, df):
        d = df.copy()
        # Next candle close
        d['next_close'] = d['close'].shift(-1)
        d['target'] = (d['next_close'] > d['close']).astype(int)
        
        # Dead Zone Calculation (Absolute pip movement)
        d['movement'] = abs(d['next_close'] - d['close'])
        
        # Drop rows where movement is less than 1 pip (Noise)
        d = d[d['movement'] >= DEAD_ZONE_PIPS]
        d.dropna(inplace=True)
        return d['target']

# ─────────────────────────────────────────
# MULTI-MODEL (Triple Voting)
# ─────────────────────────────────────────
class TripleModel:
    def __init__(self, symbol):
        self.symbol = symbol
        self.xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)
        self.rf  = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        self.lr  = LogisticRegression(max_iter=500, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False

    def train(self, X, y):
        try:
            if len(X) < 150: return False
            Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, shuffle=False)
            Xtrs = self.scaler.fit_transform(Xtr)
            
            self.xgb.fit(Xtrs, ytr)
            self.rf.fit(Xtrs, ytr)
            self.lr.fit(Xtrs, ytr)
            
            self.trained = True
            logger.info(f"{self.symbol}: V3 Triple Models trained.")
            return True
        except Exception as e:
            logger.error(f"Train {self.symbol}: {e}")
            return False

    def predict(self, X_latest):
        if not self.trained: return None, 0.0, False
        Xs = self.scaler.transform(X_latest)
        
        xgb_proba = self.xgb.predict_proba(Xs)[0]
        rf_proba  = self.rf.predict_proba(Xs)[0]
        lr_proba  = self.lr.predict_proba(Xs)[0]
        
        xgb_p = int(np.argmax(xgb_proba))
        rf_p  = int(np.argmax(rf_proba))
        lr_p  = int(np.argmax(lr_proba))
        
        all_agree = (xgb_p == rf_p == lr_p)
        avg_conf = (max(xgb_proba) + max(rf_proba) + max(lr_proba)) / 3
        
        return xgb_p, float(avg_conf), all_agree

# ─────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────
class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.url = f'https://api.telegram.org/bot{token}/sendMessage'
        self.chat_id = chat_id
        self.enabled = token != 'YOUR_BOT_TOKEN_HERE'

    def send(self, text):
        if self.enabled:
            requests.post(self.url, data={'chat_id': self.chat_id, 'text': text, 'parse_mode': 'Markdown'})
        else:
            print(f"\n{text}\n")

class ForexBotV3:
    def __init__(self):
        self.fetcher  = DataFetcher()
        self.builder  = FeatureBuilder()
        self.notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        self.model    = TripleModel(PRIMARY_PAIR)
        self.last_sig = None

    def prepare_data(self):
        df5  = self.fetcher.fetch(PRIMARY_PAIR, TF_SIGNAL, TRAIN_CANDLES_5M)
        df15 = self.fetcher.fetch(PRIMARY_PAIR, TF_HIGH, int(TRAIN_CANDLES_5M/3))
        if df5 is None or df15 is None: return None, None
        
        feats, df_full = self.builder.build(df5, df15)
        labels = self.builder.get_labels(df_full)
        
        # Intersect indexes (drops Dead Zone rows from features)
        common = feats.index.intersection(labels.index)
        return feats.loc[common], labels.loc[common]

    def run(self):
        self.notifier.send("🤖 *Forex Bot V3 (Advanced) Started*\nTraining Triple-Model with Dead Zone filter...")
        X, y = self.prepare_data()
        if X is not None:
            self.model.train(X, y)
        
        while True:
            try:
                now_ist = datetime.now(IST)
                mins = now_ist.hour * 60 + now_ist.minute
                in_session = any(sh*60+sm <= mins <= eh*60+em for sh,sm,eh,em in IST_SESSIONS)
                
                if in_session:
                    df5  = self.fetcher.fetch(PRIMARY_PAIR, TF_SIGNAL, 100)
                    df15 = self.fetcher.fetch(PRIMARY_PAIR, TF_HIGH, 50)
                    
                    if df5 is not None and df15 is not None:
                        feats, _ = self.builder.build(df5, df15)
                        if feats is not None:
                            pred, conf, agree = self.model.predict(feats.iloc[[-1]])
                            
                            # Check ADX > 25 manually
                            adx_ok = float(feats['adx'].iloc[-1]) > 25
                            
                            cooldown_ok = True
                            if self.last_sig:
                                cooldown_ok = (datetime.now() - self.last_sig).total_seconds() > SIGNAL_COOLDOWN_MINS * 60

                            if pred is not None and agree and conf >= CONFIDENCE_THRESHOLD and adx_ok and cooldown_ok:
                                dir_str = "🟢 GREEN (BUY)" if pred == 1 else "🔴 RED (SELL)"
                                msg = (
                                    f"*⚡ V3 PRO SIGNAL*\n"
                                    f"Pair: `{PRIMARY_PAIR}`\n"
                                    f"Direction: {dir_str}\n"
                                    f"Confidence: `{conf:.1%}`\n"
                                    f"Voting: XGB ✓ | RF ✓ | LR ✓\n"
                                    f"ADX Trend: OK\n"
                                )
                                self.notifier.send(msg)
                                self.last_sig = datetime.now()
                                logger.info(f"V3 SIGNAL: {dir_str} {conf:.1%}")

            except Exception as e:
                logger.error(f"Error: {e}")
            time.sleep(LOOP_INTERVAL_SECS)

if __name__ == '__main__':
    bot = ForexBotV3()
