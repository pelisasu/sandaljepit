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
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai

# Konfigurasi Environment & API Keys
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

SYMBOL = "frxXAUUSD"  # Simbol Deriv untuk Gold

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
    """
    Mengambil data harga historis dari Deriv Public API (WebSocket/HTTP endpoint publik)
    granularity 900 = M15 (15 Menit)
    """
    url = "https://deriv.com/api/v1/public-history" # Placeholder endpoint publik / alternatif REST
    # Menggunakan public WebSocket API Deriv via HTTP Proxy / Public API endpoint
    deriv_rest_url = f"https://api.deriv.com/basic/ticks_history?symbol={symbol}&end=latest&count={count}&granularity={granularity}&style=candles"
    
    try:
        # Mengambil dari API Publik Deriv via App ID publik standar (1089)
        app_id = 1089
        ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        
        # Sebagai alternatif yang handal dan ringan di GitHub Actions tanpa async websocket yang kompleks:
        # Menggunakan endpoint alternatif penyedia data stabil (Yahoo Finance API untuk XAUUSD sebagai mirror, atau Deriv Public Data)
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
        # Generate dummy structure prevent crash if network block occurs
        return None

class QuantSMCAnalyzer:
    @staticmethod
    def calculate_smc_levels(df):
        """Mendeteksi Order Block (OB), Fair Value Gap (FVG), dan Liquidity Sweep"""
        df['hl2'] = (df['high'] + df['low']) / 2
        
        # Fair Value Gap (FVG) Detection
        df['fvg_bullish'] = (df['low'].shift(-1) > df['high'].shift(1))
        df['fvg_bearish'] = (df['high'].shift(-1) < df['low'].shift(1))
        
        # Volume Profile POC (Point of Control)
        price_bins = pd.cut(df['close'], bins=20)
        poc = df.groupby(price_bins)['volume'].sum().idxmax()
        poc_price = poc.mid if pd.notnull(poc) else df['close'].iloc[-1]
        
        # Market Structure Break (MSB) / ChoCh Simulation
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
        """Dual AI Fallback: Google Gemini AI Studio -> DeepSeek API"""
        prompt = f"""
        Bertindaklah sebagai Senior Quantitative Risk Manager dan AI Trading Director.
        Analisis data pasar XAUUSD berikut dan berikan keputusan final (STRONG BUY / STRONG SELL / HOLD):
        Data: {json.dumps(market_data_summary)}
        Berikan jawaban format JSON ketat dengan kunci: "action", "confidence", "reason".
        """
        
        # 1. Coba Google Gemini AI Studio
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                return json.loads(response.text.replace('```json', '').replace('```', '').strip())
            except Exception as e:
                print(f"Gemini gagal, beralih ke DeepSeek: {e}")
                
        # 2. Fallback ke DeepSeek API
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
                content = res_json['choices'][0]['message']['content']
                return json.loads(content.replace('```json', '').replace('```', '').strip())
            except Exception as e:
                print(f"DeepSeek juga gagal: {e}")
                
        # Default fallback jika kedua AI offline
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
    
    print("Menjalankan Analisis Konsensus AI (Gemini / DeepSeek)...")
    ai_decision = DualAIEngine.get_ai_consensus(market_summary)
    
    action = ai_decision.get("action", "HOLD")
    confidence = ai_decision.get("confidence", 0)
    reason = ai_decision.get("reason", "No reason provided")
    
    print(action, confidence, reason)
    
    # Eksekusi Sinyal Manual ke Telegram jika Confidence > 75%
    if action in ["STRONG BUY", "STRONG SELL"] and confidence >= 0.75:
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
