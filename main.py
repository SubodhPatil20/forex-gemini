
"""
Forex Binary Prediction Bot
Accuracy-first | 5-min signals | No pandas_ta
"""
import os,json,time,logging,warnings
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import requests
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
logger=logging.getLogger(__name__)

TELEGRAM_TOKEN=os.getenv('TELEGRAM_TOKEN','YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID=os.getenv('TELEGRAM_CHAT_ID','YOUR_CHAT_ID_HERE')

ALL_PAIRS=['EURUSD=X','GBPUSD=X','USDJPY=X','USDCAD=X','USDCHF=X','AUDUSD=X','EURJPY=X','GBPJPY=X']
TF_SIGNAL='5m';TF_HIGH='15m';TF_LOW='1m'
CONFIDENCE_THRESHOLD=0.80;ADX_MIN=25;PREDICTABILITY_MIN=70
MIN_ACCURACY_KEEP=0.55;CONSECUTIVE_WRONG_MAX=3
LOOP_INTERVAL_SECS=300;RESCAN_INTERVAL_SECS=1800
TRAIN_CANDLES_5M=1500;ACCURACY_FILE='accuracy_log.json'

def calc_rsi(close,p=14):
    d=close.diff();g=d.clip(lower=0).rolling(p).mean();l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-(100/(1+g/(l+1e-10)))

def calc_ema(close,p):
    return close.ewm(span=p,adjust=False).mean()

def calc_macd(close,fast=12,slow=26,sig=9):
    m=calc_ema(close,fast)-calc_ema(close,slow);s=calc_ema(m,sig);return m,s,m-s

def calc_bb(close,p=20,k=2):
    sma=close.rolling(p).mean();std=close.rolling(p).std();up=sma+k*std;lo=sma-k*std
    return up,lo,(up-lo)/(close+1e-10),(close-lo)/(up-lo+1e-10)

def calc_atr(high,low,close,p=14):
    tr=pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calc_adx(high,low,close,p=14):
    tr=calc_atr(high,low,close,p);up=high.diff();dn=-low.diff()
    dp=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=close.index).rolling(p).mean()
    dm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=close.index).rolling(p).mean()
    dip=100*dp/(tr+1e-10);dim=100*dm/(tr+1e-10)
    dx=100*(dip-dim).abs()/(dip+dim+1e-10);return dx.rolling(p).mean(),dip,dim

def calc_stoch(high,low,close,k=14,d=3):
    sk=100*(close-low.rolling(k).min())/(high.rolling(k).max()-low.rolling(k).min()+1e-10)
    return sk,sk.rolling(d).mean()

class TelegramNotifier:
    def __init__(self,token,chat_id):
        self.token=token;self.chat_id=chat_id
        self.url=f'https://api.telegram.org/bot{token}/sendMessage'
        self.enabled=token!='YOUR_BOT_TOKEN_HERE'
    def send(self,text):
        if not self.enabled:print(f"\n{'='*55}\n{text}\n{'='*55}");return
        try:requests.post(self.url,data={'chat_id':self.chat_id,'text':text,'parse_mode':'Markdown'},timeout=10)
        except Exception as e:logger.warning(f"Telegram:{e}")

class DataFetcher:
    def fetch(self,symbol,interval,candles):
        try:
            pm={'1m':'7d','5m':'60d','15m':'60d'}
            df=yf.download(symbol,period=pm.get(interval,'60d'),interval=interval,progress=False,auto_adjust=True)
            if df.empty or len(df)<50:return None
            df=df.tail(candles).copy()
            if isinstance(df.columns,pd.MultiIndex):df.columns=df.columns.get_level_values(0)
            df.columns=[c.lower() for c in df.columns];df.dropna(inplace=True);return df
        except Exception as e:logger.debug(f"Fetch {symbol}:{e}");return None

class FeatureBuilder:
    FCOLS=['rsi','macd','macd_signal','macd_hist','ema9','ema21','ema50',
           'ema9_21_cross','ema21_50_cross','bb_upper','bb_lower','bb_width','bb_pct',
           'atr','adx','di_plus','di_minus','stoch_k','stoch_d','volume_ratio',
           'candle_body','candle_wick_up','candle_wick_down',
           'price_change_1','price_change_3','price_change_5',
           'volatility_5','volatility_10','high_low_ratio']
    def build(self,df):
        try:
            d=df.copy()
            d['rsi']=calc_rsi(d['close'])
            d['macd'],d['macd_signal'],d['macd_hist']=calc_macd(d['close'])
            d['ema9']=calc_ema(d['close'],9);d['ema21']=calc_ema(d['close'],21);d['ema50']=calc_ema(d['close'],50)
            d['ema9_21_cross']=(d['ema9']>d['ema21']).astype(int)
            d['ema21_50_cross']=(d['ema21']>d['ema50']).astype(int)
            d['bb_upper'],d['bb_lower'],d['bb_width'],d['bb_pct']=calc_bb(d['close'])
            d['atr']=calc_atr(d['high'],d['low'],d['close'])
            d['adx'],d['di_plus'],d['di_minus']=calc_adx(d['high'],d['low'],d['close'])
            d['stoch_k'],d['stoch_d']=calc_stoch(d['high'],d['low'],d['close'])
            va=d['volume'].rolling(20).mean();d['volume_ratio']=d['volume']/(va+1e-10)
            hl=d['high']-d['low']+1e-10
            d['candle_body']=abs(d['close']-d['open'])/hl
            d['candle_wick_up']=(d['high']-d[['close','open']].max(axis=1))/hl
            d['candle_wick_down']=(d[['close','open']].min(axis=1)-d['low'])/hl
            for n in [1,3,5]:d[f'price_change_{n}']=d['close'].pct_change(n)
            d['volatility_5']=d['close'].pct_change().rolling(5).std()
            d['volatility_10']=d['close'].pct_change().rolling(10).std()
            d['high_low_ratio']=(d['high']-d['low'])/d['close']
            d.dropna(inplace=True)
            cols=[c for c in self.FCOLS if c in d.columns];return d[cols],d
        except Exception as e:logger.debug(f"Feature:{e}");return None,None
    def get_label(self,df):
        d=df.copy();d['target']=(d['close'].shift(-1)>d['close']).astype(int)
        d.dropna(inplace=True);return d

class MarketFilter:
    def is_trending(self,feats):
        try:return float(feats['adx'].iloc[-1])>=ADX_MIN
        except:return False
    def trend_dir(self,df_full):
        try:
            l=df_full.iloc[-1];e9,e21,e50=l.get('ema9'),l.get('ema21'),l.get('ema50')
            if e9 is None:return 0
            if e9>e21>e50:return 1
            if e9<e21<e50:return -1
            return 0
        except:return 0

class PairScanner:
    def __init__(self):
        self.fetcher=DataFetcher();self.builder=FeatureBuilder();self.mf=MarketFilter()
    def score(self,symbol):
        try:
            df=self.fetcher.fetch(symbol,'5m',100)
            if df is None:return 0
            feats,df_full=self.builder.build(df)
            if feats is None or len(feats)<20:return 0
            s=0;adx=float(feats['adx'].iloc[-1])
            if adx>=30:s+=30
            elif adx>=25:s+=20
            elif adx>=20:s+=10
            if self.mf.trend_dir(df_full)!=0:s+=20
            last=feats.iloc[-1]
            rsi=float(last.get('rsi',50));mh=float(last.get('macd_hist',0));ec=int(last.get('ema9_21_cross',0))
            bull=sum([rsi<40,mh>0,ec==1]);bear=sum([rsi>60,mh<0,ec==0])
            if bull>=2 or bear>=2:s+=20
            elif bull>=1 or bear>=1:s+=10
            vr=float(last.get('volume_ratio',1))
            if vr>1.5:s+=15
            elif vr>1.2:s+=10
            elif vr>0.8:s+=5
            if float(feats['candle_body'].tail(10).mean())>0.5:s+=15
            return min(s,100)
        except:return 0
    def scan(self):
        logger.info("Scanning pairs...")
        res={}
        for sym in ALL_PAIRS:
            s=self.score(sym);res[sym]=s
            logger.info(f"  {sym:12s} → {s}/100");time.sleep(0.5)
        valid={k:v for k,v in res.items() if v>=PREDICTABILITY_MIN}
        pool=valid if valid else res
        return sorted(pool.items(),key=lambda x:x[1],reverse=True)[:2]

class PairModel:
    def __init__(self,symbol):
        self.symbol=symbol;self.model=None;self.scaler=StandardScaler()
        self.trained=False;self.train_acc=0.0
    def train(self,X,y):
        try:
            if len(X)<100:return False
            Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=0.2,shuffle=False)
            Xtrs=self.scaler.fit_transform(Xtr);Xvs=self.scaler.transform(Xv)
            self.model=XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,
                subsample=0.8,colsample_bytree=0.8,eval_metric='logloss',verbosity=0,random_state=42)
            self.model.fit(Xtrs,ytr,eval_set=[(Xvs,yv)],verbose=False)
            self.train_acc=accuracy_score(yv,self.model.predict(Xvs));self.trained=True
            logger.info(f"{self.symbol}: trained | acc:{self.train_acc:.2%}");return True
        except Exception as e:logger.error(f"Train {self.symbol}:{e}");return False
    def predict(self,X):
        if not self.trained:return None,0.0
        try:
            Xs=self.scaler.transform(X);p=self.model.predict_proba(Xs)[0]
            return int(np.argmax(p)),float(np.max(p))
        except:return None,0.0

class SignalEngine:
    def __init__(self):
        self.fetcher=DataFetcher();self.builder=FeatureBuilder();self.mf=MarketFilter()
    def check(self,symbol,model):
        layers={}
        df5=self.fetcher.fetch(symbol,TF_SIGNAL,200)
        if df5 is None:return None,0.0,0,{}
        f5,d5=self.builder.build(df5)
        if f5 is None:return None,0.0,0,{}
        adx=float(f5['adx'].iloc[-1]);trend=adx>=ADX_MIN
        layers['adx']={'pass':trend,'value':f"{adx:.1f}"}
        if not trend:return None,0.0,1,layers
        df15=self.fetcher.fetch(symbol,TF_HIGH,100);t15=0
        if df15 is not None:
            _,d15=self.builder.build(df15)
            if d15 is not None:t15=self.mf.trend_dir(d15)
        layers['trend_15m']={'pass':t15!=0,'value':'UP' if t15==1 else 'DOWN' if t15==-1 else 'UNCLEAR'}
        last=f5.iloc[-1]
        rsi=float(last.get('rsi',50));mh=float(last.get('macd_hist',0))
        ec=int(last.get('ema9_21_cross',0));sk=float(last.get('stoch_k',50))
        bull=sum([rsi<45,mh>0,ec==1,sk<40]);bear=sum([rsi>55,mh<0,ec==0,sk>60])
        ia=bull>=3 or bear>=3;idir=1 if bull>=3 else(-1 if bear>=3 else 0)
        layers['indicators']={'pass':ia,'value':f"Bull:{bull} Bear:{bear}"}
        df1=self.fetcher.fetch(symbol,TF_LOW,50);tdir=0;tok=False
        if df1 is not None:
            _,d1=self.builder.build(df1)
            if d1 is not None:tdir=self.mf.trend_dir(d1);tok=tdir!=0
        layers['timing_1m']={'pass':tok,'value':'UP' if tdir==1 else 'DOWN' if tdir==-1 else 'MIXED'}
        dirs=[d for d in [t15,idir,tdir] if d!=0]
        da=len(dirs)>=2 and len(set(dirs))==1
        layers['direction_align']={'pass':da,'value':'Agreed' if da else 'Mixed'}
        passed=sum(1 for v in layers.values() if v['pass'])
        if passed<3:return None,0.0,passed,layers
        pred,conf=model.predict(f5.iloc[[-1]])
        layers['ml_model']={'pass':conf>=CONFIDENCE_THRESHOLD,'value':f"{conf:.1%}"}
        total=sum(1 for v in layers.values() if v['pass'])
        if pred is not None and da and dirs:
            if(1 if pred==1 else -1)!=dirs[0]:
                layers['ml_model']['pass']=False;return pred,conf,total-1,layers
        return pred,conf,total,layers

class AccuracyTracker:
    def __init__(self):self.log=self._load()
    def _load(self):
        try:
            if os.path.exists(ACCURACY_FILE):
                with open(ACCURACY_FILE) as f:return json.load(f)
        except:pass
        return {}
    def _save(self):
        try:
            with open(ACCURACY_FILE,'w') as f:json.dump(self.log,f,indent=2)
        except:pass
    def record(self,symbol,t,pred,actual):
        if symbol not in self.log:self.log[symbol]=[]
        self.log[symbol].append({'time':t,'predicted':pred,'actual':actual,'correct':pred==actual})
        self._save()
    def get_accuracy(self,symbol,n=12):
        tr=self.log.get(symbol,[])
        if not tr:return 1.0,0
        r=tr[-n:];return sum(1 for t in r if t['correct'])/len(r),len(r)
    def consec_wrong(self,symbol):
        c=0
        for t in reversed(self.log.get(symbol,[])):
            if not t['correct']:c+=1
            else:break
        return c

class PendingVerifier:
    def __init__(self):self.pending=[]
    def add(self,symbol,pred,price):
        self.pending.append({'symbol':symbol,'predicted':pred,'price_at_signal':price,
            'check_after':datetime.now()+timedelta(minutes=5),
            'time_str':datetime.now().strftime('%Y-%m-%d %H:%M')})
    def check(self,fetcher):
        resolved=[];pending=[]
        for item in self.pending:
            if datetime.now()<item['check_after']:pending.append(item);continue
            try:
                df=fetcher.fetch(item['symbol'],'5m',5)
                if df is None or len(df)<2:pending.append(item);continue
                actual=1 if float(df['close'].iloc[-1])>item['price_at_signal'] else 0
                resolved.append((item,actual))
            except:pending.append(item)
        self.pending=pending;return resolved

def fmt_signal(symbol,pred,conf,layers,acc,n):
    names={'adx':'Market trend (ADX)','trend_15m':'15m trend','indicators':'Indicator agree',
           'timing_1m':'1m timing','direction_align':'Direction align','ml_model':'ML confidence'}
    llines="\n".join(f"  {'✅' if v['pass'] else '❌'} {names.get(k,k)}: {v['value']}" for k,v in layers.items())
    icon="🟢" if pred==1 else "🔴"
    return(f"*⚡ HIGH CONFIDENCE SIGNAL 🔥🔥🔥*\n━━━━━━━━━━━━━━━━━━━━━\n"
           f"*Pair:* `{symbol}`\n*Direction:* {icon} {'GREEN (BUY)' if pred==1 else 'RED (SELL)'}\n"
           f"*Confidence:* `{conf:.1%}`\n*Timeframe:* 5 min\n━━━━━━━━━━━━━━━━━━━━━\n"
           f"*Filters:*\n{llines}\n━━━━━━━━━━━━━━━━━━━━━\n"
           f"*Today accuracy:* {f'{acc:.1%} ({n} trades)' if n>0 else 'First trade'}\n"
           f"*Time:* {datetime.now().strftime('%H:%M:%S IST')}")

def fmt_result(symbol,pred,actual,acc,n):
    ok=pred==actual
    return(f"*Result: {'✅ CORRECT 🔥🔥🔥' if ok else '❌ WRONG 🔥🔥🔥'}*\n"
           f"`{symbol}` | Pred: {'GREEN' if pred==1 else 'RED'} | Actual: {'GREEN' if actual==1 else 'RED'}\n"
           f"Today accuracy: {acc:.1%} ({n} trades)")

def fmt_scan(pairs):
    lines=["*📡 Pair Scanner*","━━━━━━━━━━━━━━━━━━━━━"]
    for i,(sym,score) in enumerate(pairs):
        lines.append(f"  {'🥇' if i==0 else '🥈'} `{sym}` — {score}/100 [{'PRIMARY' if i==0 else 'BACKUP'}]")
    lines.append("_Rescan in 30 mins_");return "\n".join(lines)

class ForexBot:
    def __init__(self):
        self.notifier=TelegramNotifier(TELEGRAM_TOKEN,TELEGRAM_CHAT_ID)
        self.fetcher=DataFetcher();self.builder=FeatureBuilder()
        self.scanner=PairScanner();self.engine=SignalEngine()
        self.tracker=AccuracyTracker();self.verifier=PendingVerifier()
        self.models={};self.active=[];self.last_scan=None
    def prepare(self,symbol):
        df=self.fetcher.fetch(symbol,TF_SIGNAL,TRAIN_CANDLES_5M)
        if df is None:return None,None
        feats,df_full=self.builder.build(df)
        if feats is None:return None,None
        dl=self.builder.get_label(df_full);ci=feats.index.intersection(dl.index)
        return feats.loc[ci].iloc[:-1],dl.loc[ci,'target'].iloc[:-1]
    def train(self,symbol):
        logger.info(f"Training {symbol}...")
        X,y=self.prepare(symbol)
        if X is None:return False
        m=PairModel(symbol)
        if m.train(X,y):self.models[symbol]=m;return True
        return False
    def rescan(self):
        logger.info("Rescan+retrain...")
        self.notifier.send("🔍 *Scanning pairs...* (~2 mins)")
        self.active=self.scanner.scan();self.last_scan=datetime.now()
        self.notifier.send(fmt_scan(self.active))
        for sym,_ in self.active:
            if sym not in self.models:self.train(sym)
    def run_loop(self):
        if not self.active:self.rescan()
        if self.last_scan is None or(datetime.now()-self.last_scan).total_seconds()>RESCAN_INTERVAL_SECS:
            self.rescan()
        for item,actual in self.verifier.check(self.fetcher):
            self.tracker.record(item['symbol'],item['time_str'],item['predicted'],actual)
            acc,n=self.tracker.get_accuracy(item['symbol'])
            self.notifier.send(fmt_result(item['symbol'],item['predicted'],actual,acc,n))
        if self.active:
            sym=self.active[0][0];acc,n=self.tracker.get_accuracy(sym)
            cw=self.tracker.consec_wrong(sym)
            if(n>=5 and acc<MIN_ACCURACY_KEEP) or cw>=CONSECUTIVE_WRONG_MAX:
                self.notifier.send(f"⚠️ *Pair switch!*\n`{sym}` dropped. Rescanning...")
                self.rescan()
        for sym,_ in self.active:
            if sym not in self.models:self.train(sym);continue
            if not self.models[sym].trained:continue
            logger.info(f"Checking {sym}...")
            pred,conf,np_,layers=self.engine.check(sym,self.models[sym])
            logger.info(f"{sym}: conf={conf:.1%} passed={np_}/6")
            if pred is not None and conf>=CONFIDENCE_THRESHOLD and np_>=5:
                try:
                    df_now=self.fetcher.fetch(sym,'5m',5)
                    price=float(df_now['close'].iloc[-1]) if df_now is not None else 0.0
                except:price=0.0
                acc,n=self.tracker.get_accuracy(sym)
                self.notifier.send(fmt_signal(sym,pred,conf,layers,acc,n))
                self.verifier.add(sym,pred,price)
                logger.info(f"SIGNAL:{sym} {'GREEN' if pred==1 else 'RED'} {conf:.1%}")
    def start(self):
        logger.info("="*50)
        logger.info("FOREX BOT — ACCURACY-FIRST")
        logger.info("="*50)
        self.notifier.send(
            f"*🤖 Forex Bot Started*\n━━━━━━━━━━━━━━━━━━━━━\n"
            f"Signal: 5 min | Confidence: {CONFIDENCE_THRESHOLD:.0%}+\n"
            f"ADX: {ADX_MIN}+ | Layers: 5/6 required\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n_Scanning..._")
        self.rescan()
        logger.info("Loop started every 5 mins...")
        while True:
            try:self.run_loop()
            except KeyboardInterrupt:self.notifier.send("🛑 *Bot stopped.*");break
            except Exception as e:
                logger.error(f"Loop error:{e}")
                self.notifier.send(f"⚠️ Error:{e}\nRetrying 60s...");time.sleep(60);continue
            logger.info(f"Sleeping {LOOP_INTERVAL_SECS}s...");time.sleep(LOOP_INTERVAL_SECS)

if __name__=='__main__':
    bot=ForexBot();bot.start()
