import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessingencia import StandardScaler
from google import genai
from google.genai import types

# Konfigurasi Environment & API Keys
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

SYMBOL = "frxXAUUSD"

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram Credentials belum diatur!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.json()
    except Exception as e:
        print(f"Gagal mengirim pesan Telegram: {e}")

def fetch_deriv_candles(symbol=SYMBOL, granularity=900, count=100):
    """Mengambil data harga historis XAUUSD dari mirror publik stabil"""
    try:
        fallback_url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=15m&range=5d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(fallback_url, headers=headers, timeout=10)
        data = res.json()
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': quote['open'],
            'high': quote['high'],
            'low': quote['low'],
            'close': quote['close'],
            'volume': quote['volume']
        })
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error mengambil data harga: {e}")
        return None

class QuantSMCAnalyzer:
    @staticmethod
    def calculate_smc_levels(df):
        """Mendeteksi Order Block (OB), Fair Value Gap (FVG), dan Liquidity Sweep"""
        df['hl2'] = (df['high'] + df['low']) / 2
        
        df['fvg_bullish'] = (df['low'].shift(-1) > df['high'].shift(1))
        df['fvg_bearish'] = (df['high'].shift(-1) < df['low'].shift(1))
        
        # Volume Profile POC dengan parameter observed=True untuk mengatasi warning pandas
        price_bins = pd.cut(df['close'], bins=20)
        poc = df.groupby(price_bins, observed=True)['volume'].sum().idxmax()
        poc_price = poc.mid if pd.notnull(poc) else df['close'].iloc[-1]
        
        rolling_max = df['high'].rolling(window=5).max()
        rolling_min = df['low'].rolling(window=5).min()
        
        msb_buy = df['close'].iloc[-1] > rolling_max.iloc[-2]
        msb_sell = df['close'].iloc[-1] < rolling_min.iloc[-2]
        
        return {
            "poc": poc_price,
            "fvg_bullish": bool(df['fvg_bullish'].iloc[-1]),
            "fvg_bearish": bool(df['fvg_bearish'].iloc[-1]),
            "msb_buy": bool(msb_buy),
            "msb_sell": bool(msb_sell)
        }

class DualAIEngine:
    @staticmethod
    def get_ai_consensus(market_data_summary):
        """Dual AI Fallback: Google GenAI SDK (gemini-2.5-flash) -> DeepSeek API"""
        prompt = f"""
        Bertindaklah sebagai Senior Quantitative Risk Manager dan AI Trading Director.
        Analisis data pasar XAUUSD berikut dan berikan keputusan final format JSON murni (tanpa teks lain) dengan kunci: "action" (STRONG BUY / STRONG SELL / HOLD), "confidence" (float 0.0 sampai 1.0), "reason" (string penjelasan singkat).
        Data: {json.dumps(market_data_summary)}
        """
        
        # 1. Coba Google GenAI Terbaru
        if GEMINI_API_KEY:
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                return json.loads(clean_text)
            except Exception as e:
                print(f"Gemini SDK gagal, beralih ke DeepSeek: {e}")
                
        # 2. Fallback ke DeepSeek API dengan Error Handling Tangguh
        if DEEPSEEK_API_KEY:
            try:
                ds_url = "https://api.deepseek.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                }
                res = requests.post(ds_url, headers=headers, json=payload, timeout=15)
                res_json = res.json()
                
                if 'choices' in res_json:
                    content = res_json['choices'][0]['message']['content']
                    clean_text = content.replace('```json', '').replace('```', '').strip()
                    return json.loads(clean_text)
                else:
                    print(f"DeepSeek Response Error Structure: {res_json}")
            except Exception as e:
                print(f"DeepSeek juga gagal: {e}")
                
        return {"action": "HOLD", "confidence": 0.0, "reason": "AI engines unreachable"}

def main():
    print("Memulai Inisialisasi XAUUSD Quant Superintelligence Bot...")
    df = fetch_deriv_candles()
    if df is None or len(df) < 20:
        print("Data pasar tidak mencukupi. Melewatkan eksekusi.")
        return

    current_price = float(df['close'].iloc[-1])
    smc_data = QuantSMCAnalyzer.calculate_smc_levels(df)
    
    market_summary = {
        "symbol": "XAUUSD",
        "price": current_price,
        "poc": smc_data["poc"],
        "fvg_bullish": smc_data["fvg_bullish"],
        "fvg_bearish": smc_data["fvg_bearish"],
        "msb_buy": smc_data["msb_buy"],
        "msb_sell": smc_data["msb_sell"]
    }
    
    print("Menjalankan Analisis Konsensus AI (Gemini 2.5 / DeepSeek)...")
    ai_decision = DualAIEngine.get_ai_consensus(market_summary)
    
    action = ai_decision.get("action", "HOLD")
    confidence = ai_decision.get("confidence", 0)
    reason = ai_decision.get("reason", "No reason provided")
    
    print(f"Hasil AI Decision -> Action: {action} | Confidence: {confidence} | Reason: {reason}")
    
    if action in ["STRONG BUY", "STRONG SELL"] and confidence >= 0.70:
        sl = current_price - 4.0 if action == "STRONG BUY" else current_price + 4.0
        tp1 = current_price + 8.0 if action == "STRONG BUY" else current_price - 8.0
        tp2 = current_price + 14.0 if action == "STRONG BUY" else current_price - 14.0
        
        msg = f"""
🚨 **QUANT AI SIGNAL: XAUUSD** 🚨
━━━━━━━━━━━━━━━━━━━
📌 **Action:** `{action}`
💰 **Entry Price:** `{current_price}` (Manual Open Position)
🛡️ **Stop Loss (SL):** `{sl}`
🎯 **Take Profit 1:** `{tp1}`
🎯 **Take Profit 2:** `{tp2}`
━━━━━━━━━━━━━━━━━━━
🧠 **AI Confidence:** `{confidence * 100}%`
📊 **SMC / POC Analysis:** 
- POC Level: `{smc_data['poc']:.2f}`
- Bullish FVG: `{smc_data['fvg_bullish']}`
- Market Structure Break: `{smc_data['msb_buy'] or smc_data['msb_sell']}`
💡 **Reasoning:** _{reason}_
        """
        send_telegram_message(msg)
        print("Sinyal presisi berhasil dikirim ke Telegram!")
    else:
        print(f"Kondisi pasar belum memenuhi syarat optimal (Action: {action}, Conf: {confidence}). Tidak ada sinyal dikirim.")

if __name__ == "__main__":
    main()
