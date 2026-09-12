"""
🔱 TITAN V4.2 ALL SESSION FINAL - XAUUSD 24/7
Mode: Asia + London + New York semua sesi aktif
Adaptive threshold per sesi biar gak 3 hari no signal lagi
"""
import asyncio, json, os, time, websockets, aiohttp
from datetime import datetime, timezone
from collections import deque

SYMBOL = os.getenv("TARGET_SYMBOL", "frxXAUUSD")
TELE_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELE_CHAT = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
STATE_FILE = "titan_v4_state.json"

# ===== CONFIG ALL SESSION =====
CHOP_THRESHOLD = 68.0      # V4.2: 65->68 biar Asia choppy tetep masuk
VECTOR_MIN = 0.20          # 0.25->0.20
DIST_POC_MIN = 0.6         # 0.8->0.6
AI_CONF_MIN = 58           # 62->58
ATR_MIN = 0.5              # 0.7->0.5 khusus Asia
SIGNAL_COOLDOWN = 450      # 10 menit -> 7.5 menit biar all session dapet banyak
BEP_TRIGGER = 4.0          # 5.0->4.0 lebih cepet BEP biar Asia gak balik
BEP_PLUS = 0.8
RR_TARGET = 3.0
MAX_DAILY_LOSS = 4         # 3->4 karena all session
CANDLES_M5 = deque(maxlen=250)
CANDLES_H1 = deque(maxlen=120)
STATE = {"active_trade": None, "daily_loss": 0, "last_day": 0, "last_signal_ts": 0, "total_signals": 0, "asia":0, "london":0, "ny":0}

def get_session():
    h = datetime.now(timezone.utc).hour
    # Asia: 00-07 UTC (07-14 WIB)
    # London: 07-13 UTC (14-20 WIB)  
    # NY: 13-21 UTC (20-04 WIB)
    if 0 <= h < 7: return "ASIA"
    if 7 <= h < 13: return "LONDON"
    if 13 <= h < 21: return "NEWYORK"
    return "OFF" # 21-00 UTC sepi

def session_params(sess):
    # Adaptive filter per sesi
    if sess == "ASIA":
        return {"chop": 70, "vec": 0.18, "atr": 0.4, "rr": 2.2} # Asia range kecil, RR kecilin
    if sess == "LONDON":
        return {"chop": 66, "vec": 0.22, "atr": 0.6, "rr": 3.0}
    if sess == "NEWYORK":
        return {"chop": 65, "vec": 0.20, "atr": 0.5, "rr": 3.5} # NY volatil gede, RR gede
    return {"chop": 65, "vec": 0.25, "atr": 0.5, "rr": 2.5}

async def tg_send(msg):
    if not TELE_TOKEN or not TELE_CHAT: return
    try:
        async with aiohttp.ClientSession() as s:
            await s.post(f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage",
                         json={"chat_id": TELE_CHAT, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except: pass

def load_state():
    global STATE
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE,'r') as f: STATE.update(json.load(f))
    except: pass

def save_state():
    try:
        with open(STATE_FILE,'w') as f: json.dump(STATE,f)
    except: pass

def is_weekend():
    now=datetime.now(timezone.utc)
    if now.weekday()==5: return True
    if now.weekday()==6 and now.hour < 1: return True
    return False

def atr(candles,p=14):
    if len(candles)<p+1: return 0
    trs=[]
    for i in range(1,p+1):
        h=candles[-i]['high']; l=candles[-i]['low']; pc=candles[-i-1]['close']
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    return sum(trs)/len(trs)

def chop_index(candles,p=14):
    if len(candles)<p: return 50
    highs=[c['high'] for c in candles[-p:]]; lows=[c['low'] for c in candles[-p:]]
    atr_sum=sum([candles[-i]['high']-candles[-i]['low'] for i in range(1,p+1)])
    if atr_sum==0: return 50
    range_hl=max(highs)-min(lows)
    return 100*(1-(range_hl/atr_sum)) if atr_sum>range_hl else 0.2

def vector_strength(candles,p=5):
    if len(candles)<p: return 0
    closes=[c['close'] for c in candles[-p:]]
    diff=closes[-1]-closes[0]
    vol=sum([abs(closes[i]-closes[i-1]) for i in range(1,len(closes))])
    return abs(diff)/vol if vol>0 else 0

def get_macro():
    if len(CANDLES_H1)<20: return "NEUTRAL"
    ema20=sum([c['close'] for c in list(CANDLES_H1)[-20:]])/20
    last=CANDLES_H1[-1]['close']
    if last>ema20*1.0008: return "BULLISH"
    if last<ema20*0.9992: return "BEARISH"
    return "NEUTRAL"

def detect_smc():
    if len(CANDLES_M5)<30: return None
    c=list(CANDLES_M5)[-30:]
    last=c[-1]
    fvg_bull=False; fvg_bear=False; msb_buy=False; msb_sell=False
    for i in range(2,len(c)-1):
        if c[i-2]['low']>c[i]['high']: fvg_bear=True
        if c[i-2]['high']<c[i]['low']: fvg_bull=True
    swing_high=max([x['high'] for x in c[-20:-5]])
    swing_low=min([x['low'] for x in c[-20:-5]])
    if last['close']>swing_high: msb_buy=True
    if last['close']<swing_low: msb_sell=True
    body=abs(last['close']-last['open'])
    upper=last['high']-max(last['close'],last['open'])
    lower=min(last['close'],last['open'])-last['low']
    bull_wick=lower>body*1.2 # dulu 1.5->1.2 biar Asia ketangkep
    bear_wick=upper>body*1.2
    all_closes=[x['close'] for x in c]
    poc=sum(all_closes)/len(all_closes)
    dist=abs(last['close']-poc)
    return {"fvg_bull":fvg_bull,"fvg_bear":fvg_bear,"msb_buy":msb_buy,"msb_sell":msb_sell,"bull_wick":bull_wick,"bear_wick":bear_wick,"dist_poc":dist,"poc":poc,"swing_high":swing_high,"swing_low":swing_low}

async def gemini_verify(prompt):
    if not GEMINI_KEY: return {"decision":"APPROVE","confidence":65}
    try:
        async with aiohttp.ClientSession() as s:
            url=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
            payload={"contents":[{"parts":[{"text":prompt}]}]}
            async with s.post(url,json=payload,timeout=12) as r:
                data=await r.json()
                txt=data['candidates'][0]['content']['parts'][0]['text']
                import re
                conf=65
                m=re.search(r'(\d+)%',txt)
                if m: conf=int(m.group(1))
                if "REJECT" in txt.upper() and conf<50: return {"decision":"REJECT","confidence":conf,"raw":txt}
                return {"decision":"APPROVE","confidence":conf,"raw":txt}
    except Exception as e:
        print(f"[GEMINI FALLBACK] {e}")
        return {"decision":"APPROVE","confidence":60,"fallback":True}

async def scan():
    if len(CANDLES_M5)<50: return
    if is_weekend(): return
    sess=get_session()
    if sess=="OFF":
        print("[OFF SESSION 21-00 UTC Sepi Skip]")
        return
    params=session_params(sess)
    if time.time()-STATE.get("last_signal_ts",0) < SIGNAL_COOLDOWN: return
    today=datetime.now(timezone.utc).day
    if STATE["last_day"]!=today:
        STATE["last_day"]=today; STATE["daily_loss"]=0; STATE["asia"]=0; STATE["london"]=0; STATE["ny"]=0; save_state()
        await tg_send(f"📅 <b>Daily Reset ALL SESSION</b> | Total: {STATE.get('total_signals',0)} Asia:{STATE['asia']} London:{STATE['london']} NY:{STATE['ny']}")

    if STATE["daily_loss"]>=MAX_DAILY_LOSS: return
    last=CANDLES_M5[-1]
    atr_v=atr(list(CANDLES_M5)); chop_v=chop_index(list(CANDLES_M5)); vec_v=vector_strength(list(CANDLES_M5))
    macro=get_macro(); smc=detect_smc()
    if not smc: return
    print(f"[{sess} SCAN] {last['close']:.2f} ATR:{atr_v:.2f} CHOP:{chop_v:.1f} VEC:{vec_v:.2f} MACRO:{macro} RR:{params['rr']}")

    # ADAPTIVE FILTER
    if atr_v < params["atr"]:
        print(f"[SKIP {sess} ATR {atr_v:.2f}<{params['atr']}]"); return
    if chop_v > params["chop"]:
        print(f"[SKIP {sess} CHOP {chop_v:.1f}>{params['chop']}]"); return
    if vec_v < params["vec"]:
        print(f"[SKIP {sess} VEC {vec_v:.2f}<{params['vec']}]"); return

    buy_cond=False; sell_cond=False
    # ALL SESSION LOGIC: Asia boleh cuma wick + dist doang, gak harus FVG
    if sess=="ASIA":
        if last['low'] < smc['swing_low']*1.0001 and smc['bull_wick'] and smc['dist_poc']>0.5:
            if macro in ["BULLISH","NEUTRAL","BEARISH"]: # Asia macro bebas, yang penting rejection
                buy_cond=True
        if last['high'] > smc['swing_high']*0.9999 and smc['bear_wick'] and smc['dist_poc']>0.5:
            if macro in ["BULLISH","NEUTRAL","BEARISH"]:
                sell_cond=True
    else: # London & NY tetap ketat dikit
        if last['low'] < smc['swing_low'] and smc['bull_wick'] and (smc['fvg_bull'] or smc['msb_buy'] or sess=="NEWYORK") and smc['dist_poc']>DIST_POC_MIN:
            if macro in ["BULLISH","NEUTRAL"]: buy_cond=True
        if last['high'] > smc['swing_high'] and smc['bear_wick'] and (smc['fvg_bear'] or smc['msb_sell'] or sess=="NEWYORK") and smc['dist_poc']>DIST_POC_MIN:
            if macro in ["BEARISH","NEUTRAL"]: sell_cond=True

    if not buy_cond and not sell_cond:
        print(f"[SKIP {sess} SMC] FVG B:{smc['fvg_bull']} S:{smc['fvg_bear']} MSB B:{smc['msb_buy']} S:{smc['msb_sell']} WICK B:{smc['bull_wick']} S:{smc['bear_wick']} Dist:{smc['dist_poc']:.1f} Macro:{macro}")
        return

    side="BUY" if buy_cond else "SELL"
    entry=last['close']
    rr=params["rr"]
    if side=="BUY":
        sl=smc['swing_low']-0.4
        tp=entry+(entry-sl)*rr
    else:
        sl=smc['swing_high']+0.4
        tp=entry-(sl-entry)*rr
    risk=abs(entry-sl); reward=abs(tp-entry)
    if risk==0: return

    prompt=f"XAUUSD {sess} M5 Side {side} Entry {entry} SL {sl} TP {tp} RR {rr} ATR {atr_v:.2f} CHOP {chop_v:.1f} VEC {vec_v:.2f} Macro {macro} SMC {smc} Decide APPROVE/REJECT conf %"
    ai=await gemini_verify(prompt)
    print(f"[AI {sess}] {ai}")
    if ai['decision']=="REJECT" and ai['confidence']<45 and not ai.get("fallback"): return
    if ai['confidence']<AI_CONF_MIN and vec_v<0.22 and not ai.get("fallback"):
        print(f"[SKIP AI LOW {ai['confidence']}]"); return

    STATE["last_signal_ts"]=time.time(); STATE["total_signals"]+=1
    if sess=="ASIA": STATE["asia"]+=1
    if sess=="LONDON": STATE["london"]+=1
    if sess=="NEWYORK": STATE["ny"]+=1
    STATE["active_trade"]={"side":side,"entry":entry,"sl":sl,"tp":tp,"bep_done":False,"sess":sess,"rr":rr}
    save_state()
    msg=f"""⚡ <b>{side} ALL SESSION V4.2</b> [{sess}]
━━━━━━━━━━━━
Symbol: {SYMBOL} M5
Entry: {entry:.2f} SL: {sl:.2f} TP: {tp:.2f}
RR 1:{rr} Risk {risk:.1f} Reward {reward:.1f}
Sesi: {sess} | Vec {vec_v:.2f} ATR {atr_v:.2f} CHOP {chop_v:.1f} Macro {macro}
SMC: FVG B{smc['fvg_bull']} S{smc['fvg_bear']} MSB B{smc['msb_buy']} S{smc['msb_sell']} Dist {smc['dist_poc']:.1f}
AI: {ai['decision']} {ai['confidence']}% {'(fallback)' if ai.get('fallback') else ''}
Auto BEP +{BEP_PLUS} @ +{BEP_TRIGGER} | Cooldown 7.5m
━━━━━━━━━━━━
TITAN ALL SESSION 24/7 - {sess} Mode
"""
    await tg_send(msg)

async def manage():
    if not STATE.get("active_trade") or len(CANDLES_M5)==0: return
    t=STATE["active_trade"]; price=CANDLES_M5[-1]['close']
    side=t['side']; entry=t['entry']
    if not t.get("bep_done"):
        if side=="BUY" and price>=entry+BEP_TRIGGER:
            t['sl']=entry+BEP_PLUS; t['bep_done']=True; save_state(); await tg_send(f"🛡️ <b>BEP {side}</b> [{t['sess']}] SL->{t['sl']:.2f}")
        if side=="SELL" and price<=entry-BEP_TRIGGER:
            t['sl']=entry-BEP_PLUS; t['bep_done']=True; save_state(); await tg_send(f"🛡️ <b>BEP {side}</b> [{t['sess']}] SL->{t['sl']:.2f}")
    if side=="BUY":
        if price<=t['sl']:
            bep=t.get("bep_done"); 
            if not bep: STATE["daily_loss"]+=1
            STATE["active_trade"]=None; save_state(); await tg_send(f"{'🟡 BEP SL' if bep else '❌ SL'} BUY [{t['sess']}] {price:.2f}")
        elif price>=t['tp']:
            STATE["active_trade"]=None; save_state(); await tg_send(f"✅ <b>TP BUY [{t['sess']}]</b> +{abs(t['tp']-entry):.1f}")
    else:
        if price>=t['sl']:
            bep=t.get("bep_done");
            if not bep: STATE["daily_loss"]+=1
            STATE["active_trade"]=None; save_state(); await tg_send(f"{'🟡 BEP SL' if bep else '❌ SL'} SELL [{t['sess']}] {price:.2f}")
        elif price<=t['tp']:
            STATE["active_trade"]=None; save_state(); await tg_send(f"✅ <b>TP SELL [{t['sess']}]</b> +{abs(entry-t['tp']):.1f}")

async def ws_loop():
    uri="wss://ws.derivws.com/websockets/v3?app_id=1089"
    load_state()
    await tg_send(f"🔱 <b>TITAN V4.2 ALL SESSION ONLINE</b> {SYMBOL} | Asia London NY Active | CHOP {CHOP_THRESHOLD} VEC {VECTOR_MIN} BEP {BEP_TRIGGER} Cooldown 7.5m")
    while True:
        try:
            async with websockets.connect(uri,ping_interval=20) as ws:
                print("[WS Connected ALL SESSION]")
                await ws.send(json.dumps({"ticks_history":SYMBOL,"count":250,"end":"latest","style":"candles","granularity":300}))
                await ws.send(json.dumps({"ticks_history":SYMBOL,"count":120,"end":"latest","style":"candles","granularity":3600}))
                await ws.send(json.dumps({"ohlc":SYMBOL,"granularity":300,"subscribe":1}))
                await ws.send(json.dumps({"ohlc":SYMBOL,"granularity":3600,"subscribe":1}))
                async for msg in ws:
                    data=json.loads(msg)
                    if "candles" in data:
                        gran=data["echo_req"]["granularity"]
                        if gran==300:
                            CANDLES_M5.clear()
                            for c in data["candles"]: CANDLES_M5.append({"open":c["open"],"high":c["high"],"low":c["low"],"close":c["close"],"open_time":c["epoch"]})
                        elif gran==3600:
                            CANDLES_H1.clear()
                            for c in data["candles"]: CANDLES_H1.append({"close":c["close"]})
                    if "ohlc" in data:
                        o=data["ohlc"]
                        c={"open":o["open"],"high":o["high"],"low":o["low"],"close":o["close"]}
                        if o["granularity"]==300:
                            # candle baru
                            if len(CANDLES_M5)>0 and o["open_time"]!=CANDLES_M5[-1].get("open_time",0):
                                CANDLES_M5.append({**c,"open_time":o["open_time"]})
                                await scan()
                            else:
                                if len(CANDLES_M5)>0: CANDLES_M5[-1].update(c)
                            await manage()
                        elif o["granularity"]==3600:
                            CANDLES_H1.append(c)
        except Exception as e:
            print(f"[WS ERR {get_session()}] {e} retry 5s"); await asyncio.sleep(5)

if __name__=="__main__":
    asyncio.run(ws_loop())
