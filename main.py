# main.py - Bot RSI Divergence para Railway.app
import asyncio
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
import requests
from dataclasses import dataclass
from flask import Flask, request, jsonify
import threading
import time

# Configuración de logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class DivergenceSignal:
    symbol: str
    timeframe: str
    type: str  # 'bullish' or 'bearish'
    confidence: float
    price_level: float
    resistance_level: Optional[float]
    volume_spike: bool
    rsi_value: float
    source: str = 'bot_scan'

class RSIDivergenceDetector:
    def __init__(self):
        # Obtener configuración desde variables de entorno (Railway)
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.chat_id = os.getenv('CHAT_ID')
        self.bybit_api_key = os.getenv('BYBIT_API_KEY')
        self.bybit_secret = os.getenv('BYBIT_SECRET')
        self.port = int(os.getenv('PORT', 8080))  # Railway usa variable PORT
        
        if not all([self.telegram_token, self.chat_id]):
            raise ValueError("❌ Faltan variables de entorno: TELEGRAM_TOKEN, CHAT_ID")
        
        self.bot = Bot(token=self.telegram_token)
        
        # Configurar Bybit
        self.exchange = ccxt.bybit({
            'apiKey': self.bybit_api_key,
            'secret': self.bybit_secret,
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Flask para webhooks
        self.app = Flask(__name__)
        self.setup_webhook_routes()
        
        # Cargar todos los pares
        self.all_bybit_pairs = []
        self.load_all_bybit_pairs()
        
        # Pares por categorías
        self.trading_pairs = {
            'majors': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT', 'DOTUSDT', 
                      'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'VETUSDT', 'TRXUSDT', 'EOSUSDT'],
            
            'memes': ['PEPEUSDT', '1000BONKUSDT', 'WIFUSDT', 'FLOKIUSDT', 'DOGEUSDT', 'SHIBUSDT', 
                     'MEMECOINUSDT', 'BOMEUSDT', 'MEWUSDT', 'POPCATUSDT', 'BRETTUSDT', 'NEIROUSDT',
                     'MOODENGUSDT', 'GOATUSDT', 'AKTUSDT', 'PONKEUSDT', 'PNUTUSDT', 'ACTUSDT'],
            
            'layer1_2': ['AVAXUSDT', 'MATICUSDT', 'FTMUSDT', 'ATOMUSDT', 'NEARUSDT', 'APTUSDT', 
                        'SUIUSDT', 'ARBUSDT', 'OPUSDT', 'STRKUSDT', 'SEIUSDT', 'INJUSDT', 'TONUSDT'],
            
            'ai_tokens': ['FETUSDT', 'RENDERUSDT', 'THETAUSDT', 'OCEANUSDT', 'AGIXUSDT', 
                         'AIUSDT', 'ARKMUSDT', 'PHBUSDT', 'TAORUSDT', 'GRTUSDT'],
            
            'defi': ['UNIUSDT', 'AAVEUSDT', '1INCHUSDT', 'SUSHIUSDT', 'COMPUSDT', 'MKRUSDT',
                    'YFIUSDT', 'CRVUSDT', 'LRCUSDT', 'SNXUSDT', 'PANCAKEUSDT', 'DYDXUSDT'],
            
            'gaming': ['AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'ENJUSDT', 'GALUSDT', 'CHZUSDT',
                      'ILVUSDT', 'YGGUSDT', 'TLMUSDT', 'ALICEUSDT', 'SUPERUSDT']
        }
        
        # Configuración
        self.timeframes = ['4h', '6h', '8h', '12h', '1d']
        self.rsi_period = 14
        self.min_confidence = 85
        self.final_confidence = 95
        
        # Pares activos
        self.active_pairs = set()
        self.load_default_pairs()
        
        # Storage
        self.sent_alerts = set()
        self.htf_levels = {}

    def load_all_bybit_pairs(self):
        """Cargar todos los pares de Bybit"""
        try:
            markets = self.exchange.load_markets()
            usdt_pairs = [symbol for symbol in markets.keys() 
                         if symbol.endswith('USDT') and markets[symbol]['type'] == 'swap']
            self.all_bybit_pairs = sorted(usdt_pairs)
            logger.info(f"✅ Cargados {len(self.all_bybit_pairs)} pares de Bybit")
        except Exception as e:
            logger.error(f"Error cargando pares: {e}")
            self.all_bybit_pairs = self.get_fallback_pairs()

    def get_fallback_pairs(self):
        """Pares de respaldo"""
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT']

    def load_default_pairs(self):
        """Cargar pares por defecto"""
        for category in ['majors', 'memes', 'layer1_2']:
            self.active_pairs.update(self.trading_pairs.get(category, []))

    def setup_webhook_routes(self):
        """Configurar rutas Flask"""
        @self.app.route('/', methods=['GET'])
        def home():
            return jsonify({
                "status": "🚀 RSI Divergence Bot ONLINE",
                "railway": True,
                "active_pairs": len(self.active_pairs),
                "webhook_url": f"https://{request.host}/webhook/tradingview"
            })
            
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "active_pairs": len(self.active_pairs),
                "total_pairs": len(self.all_bybit_pairs)
            })
            
        @self.app.route('/webhook/tradingview', methods=['POST'])
        def tradingview_webhook():
            return self.process_tradingview_alert()

    def process_tradingview_alert(self):
        """Procesar webhook de TradingView"""
        try:
            data = request.get_json()
            logger.info(f"📡 Webhook recibido: {data}")
            
            symbol = data.get('symbol', 'BTCUSDT')
            alert_type = data.get('type', 'bullish_divergence')
            timeframe = data.get('timeframe', '4h')
            price = float(data.get('price', 0))
            rsi = float(data.get('rsi', 50))
            
            # Crear señal
            signal = DivergenceSignal(
                symbol=symbol,
                timeframe=timeframe,
                type='bullish' if 'bullish' in alert_type else 'bearish',
                confidence=92.0,
                price_level=price,
                resistance_level=None,
                volume_spike=False,
                rsi_value=rsi,
                source='tradingview'
            )
            
            # Enviar alerta
            asyncio.create_task(self.send_tradingview_alert(signal))
            
            return jsonify({'status': 'success', 'signal_processed': True})
            
        except Exception as e:
            logger.error(f"❌ Error procesando webhook: {e}")
            return jsonify({'error': str(e)}), 500

    async def send_tradingview_alert(self, signal: DivergenceSignal):
        """Enviar alerta de TradingView"""
        try:
            type_emoji = '📉' if signal.type == 'bearish' else '📈'
            
            message = f"""🎯 *ALERTA TRADINGVIEW* 🎯

📌 Par: `{signal.symbol}`
{type_emoji} Tipo: Divergencia {signal.type}
💰 Precio: {signal.price_level:.6f}
📊 RSI: {signal.rsi_value:.1f}
⏰ TF: {signal.timeframe}
🎯 Confianza: {signal.confidence:.0f}%
🌐 Fuente: Railway + TradingView

⚡ *Señal automática desde la nube*"""
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"❌ Error enviando alerta TradingView: {e}")

    async def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Obtener datos OHLCV"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error obteniendo datos para {symbol}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, close_prices: np.array, period: int = 14) -> np.array:
        """Calcular RSI"""
        if len(close_prices) < period + 1:
            return np.full(len(close_prices), np.nan)
        
        deltas = np.diff(close_prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        rsi_values = np.full(len(close_prices), np.nan)
        
        for i in range(period, len(close_prices)):
            if i == period:
                current_avg_gain = avg_gain
                current_avg_loss = avg_loss
            else:
                current_avg_gain = (current_avg_gain * (period - 1) + gains[i-1]) / period
                current_avg_loss = (current_avg_loss * (period - 1) + losses[i-1]) / period
            
            if current_avg_loss == 0:
                rsi_values[i] = 100
            else:
                rs = current_avg_gain / current_avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))
        
        return rsi_values

    def find_peaks_and_troughs(self, data: np.array, min_distance: int = 5) -> Tuple[List[int], List[int]]:
        """Encontrar picos y valles"""
        peaks = []
        troughs = []
        
        for i in range(min_distance, len(data) - min_distance):
            is_peak = all(data[i] >= data[i-j] and data[i] >= data[i+j] for j in range(1, min_distance + 1))
            if is_peak:
                peaks.append(i)
                
            is_trough = all(data[i] <= data[i-j] and data[i] <= data[i+j] for j in range(1, min_distance + 1))
            if is_trough:
                troughs.append(i)
        
        return peaks, troughs

    def detect_divergence(self, price_data: pd.DataFrame) -> Optional[DivergenceSignal]:
        """Detectar divergencias RSI"""
        if len(price_data) < 50:
            return None
            
        closes = price_data['close'].values
        rsi = self.calculate_rsi(closes)
        
        if len(rsi) < 30 or np.isnan(rsi[-1]):
            return None
            
        price_peaks, price_troughs = self.find_peaks_and_troughs(closes)
        rsi_peaks, rsi_troughs = self.find_peaks_and_troughs(rsi)
        
        # Divergencia bajista
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            latest_price_peaks = price_peaks[-2:]
            latest_rsi_peaks = rsi_peaks[-2:]
            
            if (closes[latest_price_peaks[-1]] > closes[latest_price_peaks[0]] and 
                rsi[latest_rsi_peaks[-1]] < rsi[latest_rsi_peaks[0]]):
                
                price_diff = (closes[latest_price_peaks[-1]] - closes[latest_price_peaks[0]]) / closes[latest_price_peaks[0]] * 100
                rsi_diff = abs(rsi[latest_rsi_peaks[-1]] - rsi[latest_rsi_peaks[0]])
                confidence = min(95, 70 + (price_diff * 2) + (rsi_diff * 0.5))
                
                return DivergenceSignal(
                    symbol=price_data.index.name or 'Unknown',
                    timeframe='',
                    type='bearish',
                    confidence=confidence,
                    price_level=closes[-1],
                    resistance_level=None,
                    volume_spike=self.check_volume_spike(price_data),
                    rsi_value=rsi[-1]
                )
        
        # Divergencia alcista
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            latest_price_troughs = price_troughs[-2:]
            latest_rsi_troughs = rsi_troughs[-2:]
            
            if (closes[latest_price_troughs[-1]] < closes[latest_price_troughs[0]] and 
                rsi[latest_rsi_troughs[-1]] > rsi[latest_rsi_troughs[0]]):
                
                price_diff = abs(closes[latest_price_troughs[-1]] - closes[latest_price_troughs[0]]) / closes[latest_price_troughs[0]] * 100
                rsi_diff = rsi[latest_rsi_troughs[-1]] - rsi[latest_rsi_troughs[0]]
                confidence = min(95, 70 + (price_diff * 2) + (rsi_diff * 0.5))
                
                return DivergenceSignal(
                    symbol=price_data.index.name or 'Unknown',
                    timeframe='',
                    type='bullish',
                    confidence=confidence,
                    price_level=closes[-1],
                    resistance_level=None,
                    volume_spike=self.check_volume_spike(price_data),
                    rsi_value=rsi[-1]
                )
        
        return None

    def check_volume_spike(self, price_data: pd.DataFrame) -> bool:
        """Verificar spike de volumen"""
        if len(price_data) < 20:
            return False
        recent_volume = price_data['volume'].iloc[-5:].mean()
        avg_volume = price_data['volume'].iloc[-20:-5].mean()
        return recent_volume > avg_volume * 1.5

    async def format_alert_message(self, signal: DivergenceSignal, alert_type: str = "confirmation") -> str:
        """Formatear mensaje de alerta"""
        emoji_map = {'confirmation': '🟠', 'final': '🔴'}
        type_emoji = '📉' if signal.type == 'bearish' else '📈'
        
        if alert_type == "confirmation":
            message = f"""{emoji_map[alert_type]} *DIVERGENCIA DETECTADA*

📌 Par: `{signal.symbol}`
🧭 Precio: {signal.price_level:.6f}
{type_emoji} Tipo: Divergencia {signal.type}
📈 Volumen: {'Spike ✅' if signal.volume_spike else 'Normal'}
📊 Confianza: {signal.confidence:.0f}%
📆 TF: {signal.timeframe}
🔢 RSI: {signal.rsi_value:.1f}
🌐 Fuente: Railway Bot"""
        else:
            message = f"""{emoji_map[alert_type]} *SEÑAL CONFIRMADA*

📌 Par: `{signal.symbol}`
🧭 Precio: {signal.price_level:.6f}
{type_emoji} Tipo: Divergencia {signal.type} CONFIRMADA
📈 Volumen: {'Spike + divergencia ✅' if signal.volume_spike else 'Divergencia confirmada'}
📆 TF: {signal.timeframe}
🎯 RSI: {signal.rsi_value:.1f}
🔢 Confianza: {signal.confidence:.0f}%
🌐 Fuente: Railway Bot"""
        
        return message

    async def send_telegram_alert(self, message: str):
        """Enviar alerta por Telegram"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"✅ Alerta enviada")
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje: {e}")

    async def scan_single_pair(self, symbol: str):
        """Escanear un par"""
        try:
            for timeframe in self.timeframes:
                data = await self.get_ohlcv_data(symbol, timeframe)
                if data.empty:
                    continue
                    
                data.index.name = symbol
                signal = self.detect_divergence(data)
                if not signal:
                    continue
                    
                signal.symbol = symbol
                signal.timeframe = timeframe
                
                alert_key = f"{symbol}_{timeframe}_{signal.type}_{int(signal.confidence)}"
                if alert_key in self.sent_alerts:
                    continue
                
                if self.min_confidence <= signal.confidence < self.final_confidence:
                    message = await self.format_alert_message(signal, "confirmation")
                    await self.send_telegram_alert(message)
                    self.sent_alerts.add(alert_key)
                
                elif signal.confidence >= self.final_confidence:
                    message = await self.format_alert_message(signal, "final")
                    await self.send_telegram_alert(message)
                    self.sent_alerts.add(alert_key)
                    
        except Exception as e:
            logger.error(f"❌ Error escaneando {symbol}: {e}")

    async def scan_all_pairs(self):
        """Escanear todos los pares"""
        logger.info(f"🔄 Escaneando {len(self.active_pairs)} pares...")
        
        for symbol in list(self.active_pairs)[:50]:  # Limitar para Railway
            await self.scan_single_pair(symbol)
            await asyncio.sleep(0.3)  # Evitar rate limiting
            
        logger.info("✅ Escaneo completado")

    def start_flask_server(self):
        """Iniciar servidor Flask"""
        logger.info(f"🌐 Iniciando servidor en puerto {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)

    async def start_monitoring(self):
        """Iniciar monitoreo"""
        logger.info("🚀 Bot RSI Railway iniciado")
        
        webhook_url = f"https://tu-app.railway.app/webhook/tradingview"
        
        await self.send_telegram_alert(f"""🚀 *Bot RSI Railway ONLINE* 🚀

🌐 Ejecutándose en Railway.app
📊 Pares: {len(self.active_pairs)}
⏰ TFs: {', '.join(self.timeframes)}
🔄 Intervalo: 10 min

🔗 *Webhook TradingView:*
`{webhook_url}`

💎 Listo para recibir señales 24/7!""")
        
        while True:
            try:
                await self.scan_all_pairs()
                
                if len(self.sent_alerts) > 200:
                    self.sent_alerts.clear()
                    
                await asyncio.sleep(600)  # 10 minutos
                
            except Exception as e:
                logger.error(f"❌ Error en loop: {e}")
                await asyncio.sleep(60)

# Para Railway: necesita función main
def main():
    detector = RSIDivergenceDetector()
    
    # Iniciar Flask en un hilo separado
    flask_thread = threading.Thread(target=detector.start_flask_server, daemon=True)
    flask_thread.start()
    
    # Iniciar monitoreo
    asyncio.run(detector.start_monitoring())

if __name__ == "__main__":
    main()
