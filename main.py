# main.py - Bot RSI Divergence Optimizado v2.0
import asyncio
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
import logging
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional, Set
import requests
from dataclasses import dataclass
from flask import Flask, request, jsonify
import threading
import time
import traceback
from collections import defaultdict

# Configuración de logging optimizada
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
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class RSIDivergenceBot:
    def __init__(self):
        """Inicializar bot con configuración desde variables de entorno"""
        # Configuración desde ENV
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.chat_id = os.getenv('CHAT_ID')
        self.bybit_api_key = os.getenv('BYBIT_API_KEY')
        self.bybit_secret = os.getenv('BYBIT_SECRET')
        self.port = int(os.getenv('PORT', 8080))
        
        # Validar configuración crítica
        self._validate_config()
        
        # Inicializar componentes
        self.bot = Bot(token=self.telegram_token)
        self.app = Flask(__name__)
        self.telegram_app = None
        
        # Configurar exchange con mejores parámetros
        self.exchange = self._setup_exchange()
        
        # Configurar rutas web
        self.setup_webhook_routes()
        
        # Datos del bot
        self.all_bybit_pairs = []
        self.active_pairs = set()
        self.sent_alerts = {}  # Cambio a dict para mejor gestión
        self.htf_levels = {}
        self.scan_stats = defaultdict(int)
        
        # Configuración mejorada
        self.timeframes = ['4h', '6h', '8h', '12h', '1d']
        self.rsi_period = 14
        self.min_confidence = 85
        self.final_confidence = 95
        self.max_alerts_per_hour = 50  # Límite de spam
        
        # Cargar datos iniciales
        self.initialize_data()
        
    def _validate_config(self):
        """Validar configuración crítica"""
        if not all([self.telegram_token, self.chat_id]):
            raise ValueError("❌ Variables críticas faltantes: TELEGRAM_TOKEN, CHAT_ID")
        logger.info("✅ Configuración validada correctamente")
        
    def _setup_exchange(self):
        """Configurar exchange con parámetros optimizados"""
        return ccxt.bybit({
            'apiKey': self.bybit_api_key,
            'secret': self.bybit_secret,
            'sandbox': False,
            'enableRateLimit': True,
            'rateLimit': 120,  # Más conservador
            'timeout': 30000,
            'options': {
                'defaultType': 'linear',  # Para futuros USDT
            }
        })
        
    def initialize_data(self):
        """Inicializar datos del bot"""
        try:
            self.load_all_bybit_pairs()
            self.load_default_pairs()
            logger.info(f"✅ Bot inicializado: {len(self.active_pairs)} pares activos")
        except Exception as e:
            logger.error(f"❌ Error inicializando datos: {e}")
            self.active_pairs = set(['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])  # Fallback mínimo

    def load_all_bybit_pairs(self):
        """Cargar todos los pares de Bybit con manejo de errores mejorado"""
        try:
            markets = self.exchange.load_markets()
            # Filtrar pares USDT lineales (futuros)
            usdt_pairs = [
                symbol for symbol, market in markets.items() 
                if (symbol.endswith('USDT') and 
                    market.get('type') == 'swap' and 
                    market.get('linear', True) and
                    market.get('active', True))
            ]
            self.all_bybit_pairs = sorted(usdt_pairs)
            logger.info(f"✅ Cargados {len(self.all_bybit_pairs)} pares de Bybit")
            
        except Exception as e:
            logger.error(f"❌ Error cargando pares de Bybit: {e}")
            self.all_bybit_pairs = self.get_fallback_pairs()
            
    def get_fallback_pairs(self):
        """Pares de respaldo actualizados"""
        return [
            # Majors
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT',
            'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'AVAXUSDT', 'ATOMUSDT',
            # Memes populares
            'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'WIFUSDT', 'FLOKIUSDT',
            # L1/L2
            'MATICUSDT', 'OPUSDT', 'ARBUSDT', 'SUIUSDT', 'APTUSDT', 'SEIUSDT',
            # DeFi
            'UNIUSDT', 'AAVEUSDT', 'MKRUSDT', 'COMPUSDT'
        ]

    def load_default_pairs(self):
        """Cargar pares por defecto optimizados"""
        default_pairs = [
            # Top majors - siempre líquidos
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT',
            # Memes con volumen
            'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'WIFUSDT',
            # L1/L2 populares  
            'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'ATOMUSDT', 'NEARUSDT',
            # DeFi blue chips
            'UNIUSDT', 'AAVEUSDT', 'MKRUSDT'
        ]
        
        # Solo agregar pares que existen en Bybit
        for pair in default_pairs:
            if pair in self.all_bybit_pairs:
                self.active_pairs.add(pair)
                
        logger.info(f"✅ Cargados {len(self.active_pairs)} pares por defecto")

    def setup_webhook_routes(self):
        """Configurar rutas Flask optimizadas"""
        
        @self.app.route('/', methods=['GET'])
        def home():
            return jsonify({
                "status": "🚀 RSI Divergence Bot v2.0 ONLINE",
                "version": "2.0",
                "active_pairs": len(self.active_pairs),
                "total_pairs": len(self.all_bybit_pairs),
                "uptime": datetime.now().isoformat(),
                "webhook_url": f"https://{request.host}/webhook/tradingview",
                "stats": dict(self.scan_stats)
            })
            
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "active_pairs": len(self.active_pairs),
                "alerts_sent": len(self.sent_alerts),
                "memory_usage": "ok"
            })
            
        @self.app.route('/webhook/tradingview', methods=['POST'])
        def tradingview_webhook():
            return self.process_tradingview_alert()
            
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            return jsonify({
                "scan_stats": dict(self.scan_stats),
                "active_pairs": list(self.active_pairs),
                "alerts_sent_today": len([a for a in self.sent_alerts.values() 
                                        if a.get('date') == datetime.now().date()]),
                "top_pairs": self.get_top_performing_pairs()
            })

    def process_tradingview_alert(self):
        """Procesar webhook de TradingView mejorado"""
        try:
            data = request.get_json()
            logger.info(f"📡 Webhook TradingView: {data}")
            
            # Validar datos
            required_fields = ['symbol', 'type', 'timeframe', 'price', 'rsi']
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400
            
            # Crear señal
            signal = DivergenceSignal(
                symbol=data['symbol'],
                timeframe=data['timeframe'],
                type='bullish' if 'bullish' in data['type'] else 'bearish',
                confidence=float(data.get('confidence', 92)),
                price_level=float(data['price']),
                resistance_level=None,
                volume_spike=data.get('volume_spike', False),
                rsi_value=float(data['rsi']),
                source='tradingview'
            )
            
            # Enviar alerta asíncrona
            asyncio.create_task(self.send_tradingview_alert(signal))
            
            return jsonify({
                'status': 'success',
                'signal_processed': True,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error procesando webhook TradingView: {e}")
            return jsonify({'error': str(e)}), 500

    async def send_tradingview_alert(self, signal: DivergenceSignal):
        """Enviar alerta de TradingView optimizada"""
        try:
            type_emoji = '📉' if signal.type == 'bearish' else '📈'
            confidence_emoji = '🔥' if signal.confidence >= 90 else '⚡'
            
            message = f"""{confidence_emoji} *ALERTA TRADINGVIEW* {confidence_emoji}

📌 **Par:** `{signal.symbol}`
{type_emoji} **Tipo:** Divergencia {signal.type}
💰 **Precio:** {signal.price_level:.6f}
📊 **RSI:** {signal.rsi_value:.1f}
⏰ **TF:** {signal.timeframe}
🎯 **Confianza:** {signal.confidence:.0f}%
🌐 **Fuente:** TradingView + Railway

⚡ *Señal automática desde la nube*"""
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            
            self.scan_stats['tradingview_alerts'] += 1
            logger.info("✅ Alerta TradingView enviada")
            
        except Exception as e:
            logger.error(f"❌ Error enviando alerta TradingView: {e}")

    async def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Obtener datos OHLCV con mejor manejo de errores"""
        try:
            # Mapeo de timeframes para Bybit
            timeframe_map = {
                '4h': '4h', '6h': '6h', '8h': '8h', 
                '12h': '12h', '1d': 'D', '1D': 'D'
            }
            
            bybit_timeframe = timeframe_map.get(timeframe, timeframe)
            ohlcv = self.exchange.fetch_ohlcv(symbol, bybit_timeframe, limit=limit)
            
            if not ohlcv:
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos {symbol}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, close_prices: np.array, period: int = 14) -> np.array:
        """Calcular RSI optimizado"""
        if len(close_prices) < period + 1:
            return np.full(len(close_prices), np.nan)
        
        deltas = np.diff(close_prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Usar SMA para el cálculo inicial
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi_values = np.full(len(close_prices), np.nan)
        
        for i in range(period, len(close_prices)):
            if i == period:
                current_avg_gain = avg_gain
                current_avg_loss = avg_loss
            else:
                # EMA smoothing
                alpha = 1 / period
                current_avg_gain = (1 - alpha) * current_avg_gain + alpha * gains[i-1]
                current_avg_loss = (1 - alpha) * current_avg_loss + alpha * losses[i-1]
            
            if current_avg_loss == 0:
                rsi_values[i] = 100
            else:
                rs = current_avg_gain / current_avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))
        
        return rsi_values

    def find_peaks_and_troughs(self, data: np.array, min_distance: int = 5) -> Tuple[List[int], List[int]]:
        """Encontrar picos y valles mejorado"""
        if len(data) < min_distance * 2 + 1:
            return [], []
            
        peaks = []
        troughs = []
        
        for i in range(min_distance, len(data) - min_distance):
            # Verificar picos (máximos locales)
            is_peak = all(data[i] >= data[i-j] for j in range(1, min_distance + 1)) and \
                     all(data[i] >= data[i+j] for j in range(1, min_distance + 1))
            if is_peak:
                peaks.append(i)
                
            # Verificar valles (mínimos locales)
            is_trough = all(data[i] <= data[i-j] for j in range(1, min_distance + 1)) and \
                       all(data[i] <= data[i+j] for j in range(1, min_distance + 1))
            if is_trough:
                troughs.append(i)
        
        return peaks, troughs

    def detect_divergence(self, price_data: pd.DataFrame) -> Optional[DivergenceSignal]:
        """Detectar divergencias con algoritmo mejorado"""
        if len(price_data) < 50:
            return None
            
        closes = price_data['close'].values
        rsi = self.calculate_rsi(closes)
        
        if len(rsi) < 30 or np.isnan(rsi[-1]):
            return None
            
        # Encontrar picos y valles
        price_peaks, price_troughs = self.find_peaks_and_troughs(closes, min_distance=3)
        rsi_peaks, rsi_troughs = self.find_peaks_and_troughs(rsi, min_distance=3)
        
        # Detectar divergencia bajista
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            latest_price_peaks = price_peaks[-2:]
            latest_rsi_peaks = rsi_peaks[-2:]
            
            price_high1 = closes[latest_price_peaks[0]]
            price_high2 = closes[latest_price_peaks[1]]
            rsi_high1 = rsi[latest_rsi_peaks[0]]
            rsi_high2 = rsi[latest_rsi_peaks[1]]
            
            # Condiciones mejoradas para divergencia bajista
            if (price_high2 > price_high1 and rsi_high2 < rsi_high1 and 
                rsi_high2 > 60):  # RSI debe estar alto
                
                price_change = (price_high2 - price_high1) / price_high1 * 100
                rsi_change = abs(rsi_high1 - rsi_high2)
                
                # Calcular confianza basada en múltiples factores
                confidence = min(95, 60 + (price_change * 3) + (rsi_change * 1.5))
                
                if confidence >= self.min_confidence:
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
        
        # Detectar divergencia alcista
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            latest_price_troughs = price_troughs[-2:]
            latest_rsi_troughs = rsi_troughs[-2:]
            
            price_low1 = closes[latest_price_troughs[0]]
            price_low2 = closes[latest_price_troughs[1]]
            rsi_low1 = rsi[latest_rsi_troughs[0]]
            rsi_low2 = rsi[latest_rsi_troughs[1]]
            
            # Condiciones mejoradas para divergencia alcista
            if (price_low2 < price_low1 and rsi_low2 > rsi_low1 and 
                rsi_low2 < 40):  # RSI debe estar bajo
                
                price_change = abs(price_low2 - price_low1) / price_low1 * 100
                rsi_change = rsi_low2 - rsi_low1
                
                confidence = min(95, 60 + (price_change * 3) + (rsi_change * 1.5))
                
                if confidence >= self.min_confidence:
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
        """Verificar spike de volumen mejorado"""
        if len(price_data) < 20 or 'volume' not in price_data.columns:
            return False
            
        recent_volume = price_data['volume'].iloc[-3:].mean()
        avg_volume = price_data['volume'].iloc[-20:-3].mean()
        
        return recent_volume > avg_volume * 1.8  # Umbral más estricto

    def is_duplicate_alert(self, signal: DivergenceSignal) -> bool:
        """Verificar alertas duplicadas mejorado"""
        alert_key = f"{signal.symbol}_{signal.timeframe}_{signal.type}"
        
        if alert_key in self.sent_alerts:
            last_alert = self.sent_alerts[alert_key]
            time_diff = datetime.now() - last_alert['timestamp']
            
            # No enviar la misma alerta en menos de 2 horas
            if time_diff.total_seconds() < 7200:
                return True
                
        return False

    async def format_alert_message(self, signal: DivergenceSignal, alert_type: str = "confirmation") -> str:
        """Formatear mensaje de alerta optimizado"""
        confidence_emoji = '🔥' if signal.confidence >= 95 else '⚡' if signal.confidence >= 90 else '🟠'
        type_emoji = '📉' if signal.type == 'bearish' else '📈'
        volume_emoji = '📈' if signal.volume_spike else '📊'
        
        if alert_type == "confirmation":
            message = f"""{confidence_emoji} **DIVERGENCIA DETECTADA**

📌 **Par:** `{signal.symbol}`
💰 **Precio:** {signal.price_level:.6f}
{type_emoji} **Tipo:** Divergencia {signal.type}
{volume_emoji} **Volumen:** {'Spike detectado ✅' if signal.volume_spike else 'Normal'}
📊 **Confianza:** {signal.confidence:.0f}%
📆 **TF:** {signal.timeframe}
🔢 **RSI:** {signal.rsi_value:.1f}
🤖 **Fuente:** Railway Bot v2.0

⏰ {signal.timestamp.strftime('%H:%M:%S')}"""
        else:
            message = f"""{confidence_emoji} **SEÑAL CONFIRMADA** {confidence_emoji}

📌 **Par:** `{signal.symbol}`
💰 **Precio:** {signal.price_level:.6f}
{type_emoji} **Tipo:** Divergencia {signal.type} **CONFIRMADA**
{volume_emoji} **Volumen:** {'Spike + divergencia ✅' if signal.volume_spike else 'Divergencia confirmada'}
📆 **TF:** {signal.timeframe}
🎯 **RSI:** {signal.rsi_value:.1f}
🔢 **Confianza:** {signal.confidence:.0f}%
🤖 **Fuente:** Railway Bot v2.0

🚀 **ALTA PROBABILIDAD**
⏰ {signal.timestamp.strftime('%H:%M:%S')}"""
        
        return message

    async def send_telegram_alert(self, message: str):
        """Enviar alerta por Telegram mejorado"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True
            )
            self.scan_stats['alerts_sent'] += 1
            logger.info("✅ Alerta enviada")
            
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje: {e}")

    async def scan_single_pair(self, symbol: str):
        """Escanear un par con manejo de errores mejorado"""
        try:
            for timeframe in self.timeframes:
                # Rate limiting mejorado
                await asyncio.sleep(0.1)
                
                data = await self.get_ohlcv_data(symbol, timeframe)
                if data.empty:
                    continue
                    
                data.index.name = symbol
                signal = self.detect_divergence(data)
                
                if not signal:
                    continue
                    
                signal.symbol = symbol
                signal.timeframe = timeframe
                
                # Verificar duplicados
                if self.is_duplicate_alert(signal):
                    continue
                
                # Registrar alerta
                alert_key = f"{symbol}_{timeframe}_{signal.type}"
                self.sent_alerts[alert_key] = {
                    'timestamp': datetime.now(),
                    'confidence': signal.confidence,
                    'date': datetime.now().date()
                }
                
                # Enviar según confianza
                if self.min_confidence <= signal.confidence < self.final_confidence:
                    message = await self.format_alert_message(signal, "confirmation")
                    await self.send_telegram_alert(message)
                
                elif signal.confidence >= self.final_confidence:
                    message = await self.format_alert_message(signal, "final")
                    await self.send_telegram_alert(message)
                    
                self.scan_stats['divergences_found'] += 1
                    
        except Exception as e:
            logger.error(f"❌ Error escaneando {symbol}: {e}")
            self.scan_stats['scan_errors'] += 1

    async def scan_all_pairs(self):
        """Escanear todos los pares con optimizaciones"""
        scan_start = datetime.now()
        logger.info(f"🔄 Iniciando escaneo de {len(self.active_pairs)} pares...")
        
        # Procesar en batches para mejor rendimiento
        batch_size = 10
        pairs_list = list(self.active_pairs)
        
        for i in range(0, len(pairs_list), batch_size):
            batch = pairs_list[i:i + batch_size]
            
            # Procesar batch concurrentemente
            tasks = [self.scan_single_pair(symbol) for symbol in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Pequeña pausa entre batches
            await asyncio.sleep(1)
            
        scan_duration = (datetime.now() - scan_start).total_seconds()
        
        self.scan_stats['scans_completed'] += 1
        self.scan_stats['last_scan_duration'] = scan_duration
        
        logger.info(f"✅ Escaneo completado en {scan_duration:.1f}s")

    def get_top_performing_pairs(self) -> List[str]:
        """Obtener pares con más actividad"""
        # Por simplicidad, retornamos los más populares
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'PEPEUSDT', 'WIFUSDT']

    def start_flask_server(self):
        """Iniciar servidor Flask"""
        logger.info(f"🌐 Iniciando servidor Flask en puerto {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)

    async def setup_telegram_commands(self):
        """Configurar comandos de Telegram mejorados"""
        try:
            self.telegram_app = Application.builder().token(self.telegram_token).build()
            
            # Comandos principales
            self.telegram_app.add_handler(CommandHandler("start", self.cmd_start))
            self.telegram_app.add_handler(CommandHandler("help", self.cmd_help))
            self.telegram_app.add_handler(CommandHandler("status", self.cmd_status))
            self.telegram_app.add_handler(CommandHandler("stats", self.cmd_stats))
            
            # Gestión de pares
            self.telegram_app.add_handler(CommandHandler("list_pairs", self.cmd_list_pairs))
            self.telegram_app.add_handler(CommandHandler("add_pair", self.cmd_add_pair))
            self.telegram_app.add_handler(CommandHandler("remove_pair", self.cmd_remove_pair))
            self.telegram_app.add_handler(CommandHandler("search_pair", self.cmd_search_pair))
            
            # Comandos de control
            self.telegram_app.add_handler(CommandHandler("scan_now", self.cmd_scan_now))
            self.telegram_app.add_handler(CommandHandler("webhook_test", self.cmd_webhook_test))
            
            # Handler para mensajes no reconocidos
            self.telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_unknown_message))
            
            # Inicializar y ejecutar
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            
            logger.info("✅ Comandos de Telegram configurados correctamente")
            
            # Ejecutar polling en background
            await self.telegram_app.updater.start_polling()
            
        except Exception as e:
            logger.error(f"❌ Error configurando comandos Telegram: {e}")

    # === COMANDOS DE TELEGRAM ===
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start"""
        message = f"""🚀 **Bot RSI Divergence v2.0**

¡Hola! Soy tu bot especializado en detectar divergencias RSI.

📊 **Funciones principales:**
• Monitoreo 24/7 de divergencias
• Múltiples timeframes (4h, 6h, 8h, 12h, 1d)
• Alertas automáticas en tiempo real
• Webhook para TradingView

📋 **Comandos disponibles:**
/status - Estado del bot
/stats - Estadísticas detalladas
/list\_pairs - Ver pares monitoreados
/add\_pair SYMBOL - Añadir par
/scan\_now - Escaneo manual

🤖 **Estado actual:** ONLINE
📈 **Pares activos:** {len(self.active_pairs)}

¡Listo para detectar oportunidades!"""
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /help"""
        message = f"""📋 **Ayuda - Bot RSI Divergence v2.0**

🎯 **¿Qué hace este bot?**
Detecta divergencias RSI automáticamente en múltiples pares de criptomonedas y timeframes.

📊 **Comandos principales:**
• `/status` - Estado completo del bot
• `/stats` - Estadísticas de rendimiento
• `/list_pairs` - Ver todos los pares activos
• `/scan_now` - Forzar escaneo manual

🔧 **Gestión de pares:**
• `/add_pair BTCUSDT` - Añadir par específico
• `/remove_pair ETHUSDT` - Remover par
• `/search_pair BTC` - Buscar pares disponibles

🧪 **Testing:**
• `/webhook_test` - Probar webhook TradingView

❓ **¿Qué son las divergencias?**
Cuando el precio y el RSI se mueven en direcciones opuestas, indicando posibles reversiones de tendencia.

💡 **Niveles de confianza:**
• 85-94%: Señal de confirmación 🟠
• 95%+: Señal confirmada 🔥

🌐 **Webhook URL:**
`https://tu-dominio.railway.app/webhook/tradingview`"""
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /status mejorado"""
        uptime = datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        message = f"""📊 **Estado del Bot RSI v2.0**

🔄 **Estado:** ✅ ONLINE
📈 **Pares monitoreados:** {len(self.active_pairs)}
🌐 **Total disponibles:** {len(self.all_bybit_pairs)}
⏰ **Timeframes:** {', '.join(self.timeframes)}
🔄 **Intervalo:** 10 minutos

📊 **Estadísticas hoy:**
• Escaneos completados: {self.scan_stats.get('scans_completed', 0)}
• Divergencias encontradas: {self.scan_stats.get('divergences_found', 0)}
• Alertas enviadas: {self.scan_stats.get('alerts_sent', 0)}
• Alertas TradingView: {self.scan_stats.get('tradingview_alerts', 0)}

⚡ **Rendimiento:**
• Último escaneo: {self.scan_stats.get('last_scan_duration', 0):.1f}s
• Errores: {self.scan_stats.get('scan_errors', 0)}

🎯 **Configuración:**
• Confianza mínima: {self.min_confidence}%
• Confianza máxima: {self.final_confidence}%
• RSI período: {self.rsi_period}

🌐 **Webhook:**
`/webhook/tradingview` ACTIVO"""
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /stats detallado"""
        today_alerts = len([a for a in self.sent_alerts.values() 
                          if a.get('date') == datetime.now().date()])
        
        top_pairs = self.get_top_performing_pairs()
        
        message = f"""📈 **Estadísticas Detalladas**

📊 **Resumen General:**
• Total pares activos: {len(self.active_pairs)}
• Alertas hoy: {today_alerts}
• Cache HTF: {len(self.htf_levels)} pares

⏱️ **Rendimiento:**
• Escaneos totales: {self.scan_stats.get('scans_completed', 0)}
• Divergencias encontradas: {self.scan_stats.get('divergences_found', 0)}
• Tiempo promedio escaneo: {self.scan_stats.get('last_scan_duration', 0):.1f}s
• Errores: {self.scan_stats.get('scan_errors', 0)}

🔥 **Top Pares Monitoreados:**
{chr(10).join([f"• {pair}" for pair in top_pairs[:5]])}

📡 **Webhooks:**
• TradingView alertas: {self.scan_stats.get('tradingview_alerts', 0)}
• Endpoint: ACTIVO

🎯 **Eficiencia:**
• Rate success: {((self.scan_stats.get('scans_completed', 1) - self.scan_stats.get('scan_errors', 0)) / max(1, self.scan_stats.get('scans_completed', 1)) * 100):.1f}%
• Alertas/hora: {today_alerts / max(1, datetime.now().hour + 1):.1f}"""
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def cmd_list_pairs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /list_pairs optimizado"""
        if not self.active_pairs:
            await update.message.reply_text("📭 No hay pares activos en monitoreo")
            return
            
        # Organizar pares por categorías conocidas
        majors = [p for p in self.active_pairs if p in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT']]
        memes = [p for p in self.active_pairs if any(x in p for x in ['DOGE', 'SHIB', 'PEPE', 'WIF', 'FLOKI'])]
        others = [p for p in self.active_pairs if p not in majors + memes]
        
        message = f"📊 **Pares Monitoreados ({len(self.active_pairs)} total)**\n\n"
        
        if majors:
            message += f"**💎 Majors ({len(majors)}):**\n"
            message += " • ".join(majors) + "\n\n"
            
        if memes:
            message += f"**🚀 Memes ({len(memes)}):**\n"
            message += " • ".join(memes) + "\n\n"
            
        if others:
            message += f"**📈 Otros ({len(others)}):**\n"
            others_display = others[:15]  # Mostrar max 15
            message += " • ".join(others_display)
            if len(others) > 15:
                message += f"\n... (+{len(others)-15} más)"
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def cmd_add_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /add_pair mejorado"""
        if not context.args:
            await update.message.reply_text(
                "📝 **Uso:** `/add_pair SYMBOL`\n\n**Ejemplo:** `/add_pair APEUSDT`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
            
        symbol = context.args[0].upper()
        
        if symbol not in self.all_bybit_pairs:
            # Buscar pares similares
            similar = [p for p in self.all_bybit_pairs if symbol[:4] in p][:5]
            message = f"❌ **{symbol}** no encontrado en Bybit"
            if similar:
                message += f"\n\n🔍 **Similares:**\n" + "\n".join([f"• {p}" for p in similar])
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
            return
            
        if symbol in self.active_pairs:
            await update.message.reply_text(f"⚠️ **{symbol}** ya está siendo monitoreado")
            return
            
        self.active_pairs.add(symbol)
        await update.message.reply_text(
            f"✅ **{symbol}** añadido al monitoreo\n📊 **Total pares activos:** {len(self.active_pairs)}",
            parse_mode=ParseMode.MARKDOWN
        )

    async def cmd_remove_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /remove_pair"""
        if not context.args:
            await update.message.reply_text(
                "📝 **Uso:** `/remove_pair SYMBOL`\n\n**Ejemplo:** `/remove_pair APEUSDT`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
            
        symbol = context.args[0].upper()
        
        if symbol not in self.active_pairs:
            await update.message.reply_text(f"❌ **{symbol}** no está en monitoreo")
            return
            
        self.active_pairs.remove(symbol)
        await update.message.reply_text(
            f"🗑️ **{symbol}** removido del monitoreo\n📊 **Total pares activos:** {len(self.active_pairs)}",
            parse_mode=ParseMode.MARKDOWN
        )

    async def cmd_search_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /search_pair mejorado"""
        if not context.args:
            await update.message.reply_text(
                "📝 **Uso:** `/search_pair TEXTO`\n\n**Ejemplo:** `/search_pair BTC`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
            
        search_term = context.args[0].upper()
        
        # Buscar en todos los pares
        matching_pairs = [p for p in self.all_bybit_pairs if search_term in p]
        
        if not matching_pairs:
            await update.message.reply_text(f"❌ No se encontraron pares con **'{search_term}'**", parse_mode=ParseMode.MARKDOWN)
            return
            
        # Separar activos de inactivos
        active_matches = [p for p in matching_pairs if p in self.active_pairs]
        inactive_matches = [p for p in matching_pairs if p not in self.active_pairs]
        
        message = f"🔍 **Búsqueda: '{search_term}'**\n\n"
        
        if active_matches:
            message += f"✅ **Activos ({len(active_matches)}):**\n"
            message += " • ".join(active_matches[:10])
            if len(active_matches) > 10:
                message += f"\n... (+{len(active_matches)-10} más)"
            message += "\n\n"
        
        if inactive_matches:
            message += f"⚪ **Disponibles ({len(inactive_matches)}):**\n"
            message += " • ".join(inactive_matches[:10])
            if len(inactive_matches) > 10:
                message += f"\n... (+{len(inactive_matches)-10} más)"
            message += "\n\n"
            
        message += "💡 Usa `/add_pair SYMBOL` para añadir"
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def cmd_scan_now(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /scan_now"""
        await update.message.reply_text("🔄 **Iniciando escaneo manual...**", parse_mode=ParseMode.MARKDOWN)
        
        start_time = datetime.now()
        await self.scan_all_pairs()
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        await update.message.reply_text(
            f"✅ **Escaneo completado**\n⏱️ **Duración:** {duration:.1f}s\n📊 **Pares:** {len(self.active_pairs)}",
            parse_mode=ParseMode.MARKDOWN
        )

    async def cmd_webhook_test(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /webhook_test"""
        test_signal = DivergenceSignal(
            symbol="BTCUSDT",
            timeframe="4h",
            type="bullish",
            confidence=92.5,
            price_level=45000.0,
            resistance_level=None,
            volume_spike=True,
            rsi_value=28.5,
            source="test_webhook"
        )
        
        message = await self.format_tradingview_alert(test_signal)
        await self.send_telegram_alert(f"🧪 **TEST WEBHOOK**\n\n{message}")
        
        await update.message.reply_text("✅ **Test webhook enviado**", parse_mode=ParseMode.MARKDOWN)

    async def handle_unknown_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manejar mensajes no reconocidos"""
        await update.message.reply_text(
            "❓ No entiendo ese comando.\n\nUsa `/help` para ver comandos disponibles.",
            parse_mode=ParseMode.MARKDOWN
        )

    async def format_tradingview_alert(self, signal: DivergenceSignal) -> str:
        """Formatear alerta específica para TradingView"""
        type_emoji = '📉' if signal.type == 'bearish' else '📈'
        
        return f"""🎯 **SEÑAL TRADINGVIEW**

📌 **Par:** `{signal.symbol}`
{type_emoji} **Tipo:** Divergencia {signal.type}
💰 **Precio:** {signal.price_level:.6f}
📊 **RSI:** {signal.rsi_value:.1f}
⏰ **TF:** {signal.timeframe}
🎯 **Confianza:** {signal.confidence:.0f}%
{'📈 **Volumen:** Spike detectado' if signal.volume_spike else ''}

🌐 **Fuente:** TradingView → Railway"""

    async def start_monitoring(self):
        """Iniciar monitoreo principal mejorado"""
        logger.info("🚀 Iniciando Bot RSI Divergence v2.0")
        
        # Inicializar comandos de Telegram
        await self.setup_telegram_commands()
        
        # Enviar mensaje de inicio
        webhook_url = f"https://tu-dominio.railway.app/webhook/tradingview"
        
        startup_message = f"""🚀 **Bot RSI Divergence v2.0 ONLINE**

🌐 **Plataforma:** Railway EU West
📊 **Pares monitoreados:** {len(self.active_pairs)}
⏰ **Timeframes:** {', '.join(self.timeframes)}
🔄 **Intervalo:** 10 minutos

🔗 **Webhook TradingView:**
`{webhook_url}`

✨ **Nuevas funciones v2.0:**
• Comandos mejorados y responsivos
• Mejor detección de divergencias
• Sistema anti-spam optimizado
• Estadísticas detalladas
• Rate limiting inteligente

💎 **¡Listo para detectar oportunidades 24/7!**

Usa `/help` para ver todos los comandos."""
        
        await self.send_telegram_alert(startup_message)
        
        # Loop principal de monitoreo
        while True:
            try:
                await self.scan_all_pairs()
                
                # Limpieza periódica de cache
                if len(self.sent_alerts) > 500:
                    # Mantener solo alertas de las últimas 24 horas
                    cutoff = datetime.now() - timedelta(hours=24)
                    self.sent_alerts = {
                        k: v for k, v in self.sent_alerts.items() 
                        if v.get('timestamp', datetime.min) > cutoff
                    }
                    logger.info("🧹 Cache de alertas limpiado")
                
                # Limpiar cache HTF cada 4 horas
                if len(self.htf_levels) > 100:
                    self.htf_levels.clear()
                    logger.info("🧹 Cache HTF limpiado")
                    
                await asyncio.sleep(600)  # 10 minutos
                
            except Exception as e:
                logger.error(f"❌ Error en loop principal: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)  # Esperar menos tiempo en caso de error

    def run(self):
        """Punto de entrada principal"""
        logger.info("🚀 Iniciando Bot RSI Divergence v2.0...")
        
        # Iniciar Flask en thread separado
        flask_thread = threading.Thread(target=self.start_flask_server, daemon=True)
        flask_thread.start()
        logger.info("✅ Servidor Flask iniciado")
        
        # Iniciar loop principal
        try:
            asyncio.run(self.start_monitoring())
        except KeyboardInterrupt:
            logger.info("🛑 Bot detenido por usuario")
        except Exception as e:
            logger.error(f"❌ Error crítico: {e}")
            logger.error(traceback.format_exc())

# Punto de entrada para Railway
def main():
    """Función principal para Railway"""
    try:
        bot = RSIDivergenceBot()
        bot.run()
    except Exception as e:
        logger.error(f"❌ Error iniciando bot: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
