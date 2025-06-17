# main.py - Bot RSI Divergence Ultra Optimizado v3.0 - VERSIÓN FINAL CORREGIDA
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
from dataclasses import dataclass, field
from flask import Flask, request, jsonify
import threading
import time
import traceback
from collections import defaultdict, deque

# Importaciones condicionales para evitar errores
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Configuración de logging optimizada
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log') if os.access('.', os.W_OK) else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DivergenceSignal:
    symbol: str
    timeframe: str
    type: str  # 'bullish', 'bearish', 'hidden_bullish', 'hidden_bearish'
    confidence: float
    price_level: float
    resistance_level: Optional[float] = None
    volume_spike: bool = False
    rsi_value: float = 50.0
    rsi_divergence_strength: float = 0.0
    price_divergence_strength: float = 0.0
    volume_confirmation: bool = False
    trend_alignment: bool = False
    pattern_strength: str = "medium"  # weak, medium, strong
    ml_probability: float = 0.0
    support_resistance_proximity: float = 0.0
    source: str = 'bot_scan'
    timestamp: datetime = field(default_factory=datetime.now)
    additional_confirmations: List[str] = field(default_factory=list)

class RSIDivergenceBot:
    def __init__(self):
        """Inicializar bot ultra optimizado con manejo de errores robusto"""
        try:
            # Configuración desde ENV con valores por defecto
            self.telegram_token = os.getenv('TELEGRAM_TOKEN')
            self.chat_id = os.getenv('CHAT_ID')
            self.bybit_api_key = os.getenv('BYBIT_API_KEY', '')
            self.bybit_secret = os.getenv('BYBIT_SECRET', '')
            self.port = int(os.getenv('PORT', 8080))
            
            # Validar configuración crítica
            self._validate_config()
            
            # Inicializar componentes básicos
            self.bot = None
            self.app = Flask(__name__)
            self.telegram_app = None
            
            # Configurar exchange con manejo de errores
            self.exchange = self._setup_exchange()
            
            # Configurar rutas web
            self.setup_webhook_routes()
            
            # Datos del bot
            self.all_bybit_pairs = []
            self.active_pairs = set()
            self.sent_alerts = {}
            self.htf_levels = {}
            self.scan_stats = defaultdict(int)
            self.performance_metrics = defaultdict(list)
            
            # Configuración optimizada SIN 8h
            self.timeframes = ['4h', '6h', '12h', '1d']
            self.timeframe_weights = {'4h': 1.0, '6h': 1.1, '12h': 1.3, '1d': 1.5}
            
            # Configuración RSI SIN 8h
            self.rsi_configs = {
                '4h': {'period': 14, 'smoothing': 3, 'overbought': 70, 'oversold': 30},
                '6h': {'period': 14, 'smoothing': 3, 'overbought': 72, 'oversold': 28},
                '12h': {'period': 14, 'smoothing': 2, 'overbought': 75, 'oversold': 25},
                '1d': {'period': 14, 'smoothing': 1, 'overbought': 75, 'oversold': 25}
            }
            
            # Configuración de detección SIN 8h
            self.detection_configs = {
                '4h': {
                    'min_peak_distance': 3,
                    'min_price_change': 1.5,
                    'min_rsi_change': 5.0,
                    'confidence_threshold': 78,
                    'volume_threshold': 1.8,
                    'pattern_lookback': 20
                },
                '6h': {
                    'min_peak_distance': 4,
                    'min_price_change': 2.0,
                    'min_rsi_change': 6.0,
                    'confidence_threshold': 80,
                    'volume_threshold': 1.9,
                    'pattern_lookback': 25
                },
                '12h': {
                    'min_peak_distance': 5,
                    'min_price_change': 3.0,
                    'min_rsi_change': 8.0,
                    'confidence_threshold': 84,
                    'volume_threshold': 2.1,
                    'pattern_lookback': 35
                },
                '1d': {
                    'min_peak_distance': 5,
                    'min_price_change': 4.0,
                    'min_rsi_change': 9.0,
                    'confidence_threshold': 82,
                    'volume_threshold': 2.2,
                    'pattern_lookback': 40
                }
            }
            
            # Machine Learning (opcional)
            self.ml_model = None
            self.scaler = None
            self.pattern_history = deque(maxlen=1000)
            
            # Cache optimizado
            self.price_data_cache = {}
            self.cache_expiry = 300  # 5 minutos
            
            # Inicialización segura
            self.initialize_data_safe()
            
            logger.info("✅ Bot RSI Divergence Ultra v3.0 inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error crítico inicializando bot: {e}")
            raise

    def _validate_config(self):
        """Validar configuración crítica con mejor manejo de errores"""
        if not self.telegram_token:
            raise ValueError("❌ TELEGRAM_TOKEN es requerido")
        if not self.chat_id:
            raise ValueError("❌ CHAT_ID es requerido")
        
        # Validar formato del chat_id
        try:
            int(self.chat_id)
        except ValueError:
            raise ValueError("❌ CHAT_ID debe ser un número")
            
        logger.info("✅ Configuración validada correctamente")

    def _setup_exchange(self):
        """Configurar exchange con manejo de errores robusto"""
        try:
            exchange_config = {
                'enableRateLimit': True,
                'rateLimit': 200,  # Más conservador
                'timeout': 30000,
                'options': {
                    'defaultType': 'linear',
                },
                'sandbox': False
            }
            
            # Solo agregar credenciales si están disponibles
            if self.bybit_api_key and self.bybit_secret:
                exchange_config['apiKey'] = self.bybit_api_key
                exchange_config['secret'] = self.bybit_secret
                logger.info("✅ Exchange configurado con credenciales")
            else:
                logger.warning("⚠️ Exchange configurado sin credenciales (solo lectura)")
                
            return ccxt.bybit(exchange_config)
            
        except Exception as e:
            logger.error(f"❌ Error configurando exchange: {e}")
            # Retornar configuración básica
            return ccxt.bybit({
                'enableRateLimit': True,
                'rateLimit': 500,
                'timeout': 30000,
                'sandbox': False
            })

    def initialize_data_safe(self):
        """Inicializar datos con manejo de errores seguro"""
        try:
            self.load_all_bybit_pairs_safe()
            self.load_trending_pairs_safe()
            self.initialize_ml_model_safe()
            logger.info(f"✅ Datos inicializados: {len(self.active_pairs)} pares activos")
        except Exception as e:
            logger.error(f"❌ Error inicializando datos: {e}")
            # Usar configuración de emergencia
            self.active_pairs = set(['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'HYPEUSDT'])
            self.all_bybit_pairs = self.get_fallback_pairs()

    def load_all_bybit_pairs_safe(self):
        """Cargar pares de Bybit con manejo de errores mejorado"""
        try:
            logger.info("🔄 Cargando pares de Bybit...")
            markets = self.exchange.load_markets()
            usdt_pairs = []
            
            for symbol, market in markets.items():
                try:
                    if (symbol.endswith('USDT') and 
                        market.get('type') == 'swap' and 
                        market.get('linear', True) and
                        market.get('active', True)):
                        usdt_pairs.append(symbol)
                except Exception as e:
                    logger.debug(f"Error procesando {symbol}: {e}")
                    continue
                        
            self.all_bybit_pairs = sorted(usdt_pairs)
            logger.info(f"✅ Cargados {len(self.all_bybit_pairs)} pares de Bybit")
            
        except Exception as e:
            logger.error(f"❌ Error cargando pares de Bybit: {e}")
            self.all_bybit_pairs = self.get_fallback_pairs()
            logger.info(f"✅ Usando pares de respaldo: {len(self.all_bybit_pairs)}")

    def get_fallback_pairs(self):
        """Pares de respaldo actualizados"""
        return [
            # Majors
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT',
            'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT',
            
            # Trending 2025
            'HYPEUSDT', 'MOVEUSDT', 'PENGUUSDT', 'VIRTUALUSDT',
            
            # Memes populares
            'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'WIFUSDT', 'FLOKIUSDT', 'BONKUSDT',
            
            # L1/L2
            'MATICUSDT', 'OPUSDT', 'ARBUSDT', 'SUIUSDT', 'APTUSDT', 'NEARUSDT',
            
            # DeFi
            'UNIUSDT', 'AAVEUSDT', 'CRVUSDT'
        ]

    def load_trending_pairs_safe(self):
        """Cargar pares trending con seguridad MEJORADA"""
        try:
            # Pares básicos que SIEMPRE deben cargarse
            essential_pairs = [
                'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT'
            ]
            
            # Pares trending 2025
            trending_pairs = [
                'HYPEUSDT', 'MOVEUSDT', 'PENGUUSDT', 'VIRTUALUSDT'
            ]
            
            # Memes populares
            meme_pairs = [
                'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'WIFUSDT'
            ]
            
            # Combinar todas las listas
            all_target_pairs = essential_pairs + trending_pairs + meme_pairs
            
            # FORZAR carga - incluso si all_bybit_pairs está vacío
            pairs_added = 0
            for pair in all_target_pairs:
                try:
                    # Si all_bybit_pairs está vacío O el par existe, agregarlo
                    if not self.all_bybit_pairs or pair in self.all_bybit_pairs:
                        self.active_pairs.add(pair)
                        pairs_added += 1
                        logger.info(f"✅ Par agregado: {pair}")
                except Exception as e:
                    logger.error(f"❌ Error agregando {pair}: {e}")
                    # Agregar de todas formas si es un par esencial
                    if pair in essential_pairs:
                        self.active_pairs.add(pair)
                        pairs_added += 1
                        logger.info(f"🔧 Par esencial forzado: {pair}")
            
            logger.info(f"✅ Total pares cargados: {len(self.active_pairs)} ({pairs_added} agregados)")
            
            # Si aún no hay pares, forzar los básicos
            if len(self.active_pairs) == 0:
                logger.warning("⚠️ No se cargaron pares, forzando básicos...")
                for pair in essential_pairs:
                    self.active_pairs.add(pair)
                logger.info(f"🔧 Pares básicos forzados: {len(self.active_pairs)}")
                
        except Exception as e:
            logger.error(f"❌ Error en load_trending_pairs_safe: {e}")
            # Emergencia: cargar pares básicos directamente
            emergency_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            for pair in emergency_pairs:
                self.active_pairs.add(pair)
            logger.info(f"🚨 Pares de emergencia cargados: {len(self.active_pairs)}")

    def initialize_ml_model_safe(self):
        """Inicializar ML con manejo de errores"""
        try:
            if SKLEARN_AVAILABLE:
                self.scaler = StandardScaler()
                self.ml_model = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=50  # Reducido para Railway
                )
                logger.info("✅ Modelo ML inicializado")
            else:
                logger.warning("⚠️ scikit-learn no disponible, ML deshabilitado")
                self.ml_model = None
                self.scaler = None
        except Exception as e:
            logger.error(f"❌ Error inicializando ML: {e}")
            self.ml_model = None
            self.scaler = None

    def setup_webhook_routes(self):
        """Configurar rutas Flask optimizadas"""
        
        @self.app.route('/', methods=['GET'])
        def home():
            try:
                return jsonify({
                    "status": "🚀 RSI Divergence Bot v3.0 ULTRA",
                    "version": "3.0-FIXED",
                    "active_pairs": len(self.active_pairs),
                    "total_pairs": len(self.all_bybit_pairs),
                    "uptime": datetime.now().isoformat(),
                    "ml_enabled": self.ml_model is not None,
                    "stats": dict(self.scan_stats)
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/webhook/tradingview', methods=['POST'])
        def tradingview_webhook():
            try:
                return self.process_tradingview_alert()
            except Exception as e:
                logger.error(f"❌ Error en webhook: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "active_pairs": len(self.active_pairs)
            })

    async def get_ohlcv_data_safe(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Obtener datos OHLCV con manejo de errores robusto"""
        try:
            # Cache key
            cache_key = f"{symbol}_{timeframe}_{limit}"
            now = time.time()
            
            # Verificar cache
            if (cache_key in self.price_data_cache and 
                now - self.price_data_cache[cache_key]['timestamp'] < self.cache_expiry):
                return self.price_data_cache[cache_key]['data'].copy()
            
            # Mapeo correcto de timeframes SIN 8h
            timeframe_map = {
                '4h': '4h', '6h': '6h', 
                '12h': '12h', '1d': '1d', '1D': '1d'
            }
            
            bybit_timeframe = timeframe_map.get(timeframe, timeframe)
            
            # Obtener datos con retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, bybit_timeframe, limit=limit)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(1)
                    continue
            
            if not ohlcv or len(ohlcv) < 20:
                logger.warning(f"⚠️ Datos insuficientes para {symbol} {timeframe}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Validar datos
            if df.isnull().any().any():
                logger.warning(f"⚠️ Datos con valores nulos en {symbol}")
                df = df.dropna()
            
            # Guardar en cache
            self.price_data_cache[cache_key] = {
                'data': df.copy(),
                'timestamp': now
            }
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def calculate_rsi_safe(self, close_prices: np.array, period: int = 14) -> np.array:
        """Calcular RSI con manejo de errores seguro"""
        try:
            if len(close_prices) < period + 10:
                return np.full(len(close_prices), np.nan)
            
            # Usar TA-Lib si está disponible
            if TALIB_AVAILABLE:
                try:
                    rsi = talib.RSI(close_prices.astype(float), timeperiod=period)
                    return rsi
                except Exception:
                    pass
            
            # Método manual como fallback
            return self.calculate_rsi_manual(close_prices, period)
            
        except Exception as e:
            logger.error(f"❌ Error calculando RSI: {e}")
            return np.full(len(close_prices), 50.0)

    def calculate_rsi_manual(self, close_prices: np.array, period: int = 14) -> np.array:
        """RSI manual optimizado"""
        if len(close_prices) < period + 1:
            return np.full(len(close_prices), np.nan)
        
        try:
            deltas = np.diff(close_prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            rsi_values = np.full(len(close_prices), np.nan)
            
            # Calcular primeros valores
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
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
            
        except Exception as e:
            logger.error(f"❌ Error en RSI manual: {e}")
            return np.full(len(close_prices), 50.0)

    def find_peaks_safe(self, data: np.array, min_distance: int = 5) -> Tuple[List[int], List[int]]:
        """Encontrar picos con manejo de errores"""
        try:
            if len(data) < min_distance * 3:
                return [], []
            
            # Usar scipy si está disponible
            if SCIPY_AVAILABLE:
                try:
                    peaks, _ = find_peaks(data, distance=min_distance)
                    troughs, _ = find_peaks(-data, distance=min_distance)
                    return peaks.tolist(), troughs.tolist()
                except Exception:
                    pass
            
            # Método manual
            return self.find_peaks_manual(data, min_distance)
            
        except Exception as e:
            logger.error(f"❌ Error encontrando picos: {e}")
            return [], []

    def find_peaks_manual(self, data: np.array, min_distance: int = 5) -> Tuple[List[int], List[int]]:
        """Método manual para encontrar picos"""
        peaks = []
        troughs = []
        
        try:
            for i in range(min_distance, len(data) - min_distance):
                # Picos
                if all(data[i] >= data[i-j] for j in range(1, min_distance + 1)) and \
                   all(data[i] >= data[i+j] for j in range(1, min_distance + 1)):
                    peaks.append(i)
                    
                # Valles
                if all(data[i] <= data[i-j] for j in range(1, min_distance + 1)) and \
                   all(data[i] <= data[i+j] for j in range(1, min_distance + 1)):
                    troughs.append(i)
            
            return peaks, troughs
            
        except Exception as e:
            logger.error(f"❌ Error en find_peaks_manual: {e}")
            return [], []

    def detect_divergence_safe(self, price_data: pd.DataFrame, timeframe: str) -> Optional[DivergenceSignal]:
        """Detectar divergencias con manejo de errores robusto"""
        try:
            if len(price_data) < 30:
                return None
            
            closes = price_data['close'].values
            config = self.detection_configs.get(timeframe, self.detection_configs['1d'])
            
            # Calcular RSI
            rsi = self.calculate_rsi_safe(closes, period=14)
            
            if len(rsi) < 20 or np.isnan(rsi[-1]):
                return None
            
            # Encontrar picos
            price_peaks, price_troughs = self.find_peaks_safe(closes, config['min_peak_distance'])
            rsi_peaks, rsi_troughs = self.find_peaks_safe(rsi, config['min_peak_distance'])
            
            # Detectar divergencia bajista
            signal = self.detect_bearish_divergence_safe(
                closes, rsi, price_peaks, rsi_peaks, config, timeframe
            )
            
            if not signal:
                # Detectar divergencia alcista
                signal = self.detect_bullish_divergence_safe(
                    closes, rsi, price_troughs, rsi_troughs, config, timeframe
                )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error detectando divergencias: {e}")
            return None

    def detect_bearish_divergence_safe(self, closes: np.array, rsi: np.array, 
                                     price_peaks: List[int], rsi_peaks: List[int], 
                                     config: dict, timeframe: str) -> Optional[DivergenceSignal]:
        """Detectar divergencia bajista con seguridad"""
        try:
            if len(price_peaks) < 2 or len(rsi_peaks) < 2:
                return None
            
            # Obtener últimos picos
            recent_price_peaks = [p for p in price_peaks if p >= len(closes) - config['pattern_lookback']]
            recent_rsi_peaks = [p for p in rsi_peaks if p >= len(rsi) - config['pattern_lookback']]
            
            if len(recent_price_peaks) < 2 or len(recent_rsi_peaks) < 2:
                return None
            
            p1, p2 = recent_price_peaks[-2:]
            r1, r2 = recent_rsi_peaks[-2:]
            
            # Verificar divergencia
            price_higher = closes[p2] > closes[p1]
            rsi_lower = rsi[r2] < rsi[r1]
            price_change = (closes[p2] - closes[p1]) / closes[p1] * 100
            rsi_change = abs(rsi[r1] - rsi[r2])
            
            if (price_higher and rsi_lower and 
                price_change >= config['min_price_change'] and
                rsi_change >= config['min_rsi_change']):
                
                confidence = self.calculate_confidence_safe(price_change, rsi_change, timeframe)
                
                if confidence >= config['confidence_threshold']:
                    return DivergenceSignal(
                        symbol='',
                        timeframe=timeframe,
                        type='bearish',
                        confidence=confidence,
                        price_level=closes[-1],
                        resistance_level=closes[p2],
                        rsi_value=rsi[-1],
                        rsi_divergence_strength=rsi_change,
                        price_divergence_strength=price_change,
                        pattern_strength=self.classify_pattern_strength(confidence)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error en divergencia bajista: {e}")
            return None

    def detect_bullish_divergence_safe(self, closes: np.array, rsi: np.array, 
                                     price_troughs: List[int], rsi_troughs: List[int], 
                                     config: dict, timeframe: str) -> Optional[DivergenceSignal]:
        """Detectar divergencia alcista con seguridad"""
        try:
            if len(price_troughs) < 2 or len(rsi_troughs) < 2:
                return None
            
            recent_price_troughs = [t for t in price_troughs if t >= len(closes) - config['pattern_lookback']]
            recent_rsi_troughs = [t for t in rsi_troughs if t >= len(rsi) - config['pattern_lookback']]
            
            if len(recent_price_troughs) < 2 or len(recent_rsi_troughs) < 2:
                return None
            
            t1, t2 = recent_price_troughs[-2:]
            r1, r2 = recent_rsi_troughs[-2:]
            
            # Verificar divergencia
            price_lower = closes[t2] < closes[t1]
            rsi_higher = rsi[r2] > rsi[r1]
            price_change = abs(closes[t2] - closes[t1]) / closes[t1] * 100
            rsi_change = rsi[r2] - rsi[r1]
            
            if (price_lower and rsi_higher and 
                price_change >= config['min_price_change'] and
                rsi_change >= config['min_rsi_change']):
                
                confidence = self.calculate_confidence_safe(price_change, rsi_change, timeframe)
                
                if confidence >= config['confidence_threshold']:
                    return DivergenceSignal(
                        symbol='',
                        timeframe=timeframe,
                        type='bullish',
                        confidence=confidence,
                        price_level=closes[-1],
                        resistance_level=closes[t1],
                        rsi_value=rsi[-1],
                        rsi_divergence_strength=rsi_change,
                        price_divergence_strength=price_change,
                        pattern_strength=self.classify_pattern_strength(confidence)
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error en divergencia alcista: {e}")
            return None

    def calculate_confidence_safe(self, price_change: float, rsi_change: float, timeframe: str) -> float:
        """Calcular confianza con seguridad"""
        try:
            base_confidence = 50.0
            price_factor = min(price_change * 2, 20)
            rsi_factor = min(rsi_change * 1.5, 15)
            tf_factor = self.timeframe_weights.get(timeframe, 1.0) * 5
            
            confidence = base_confidence + price_factor + rsi_factor + tf_factor
            return min(confidence, 95.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculando confianza: {e}")
            return 75.0

    def classify_pattern_strength(self, confidence: float) -> str:
        """Clasificar fuerza del patrón"""
        if confidence >= 90:
            return "strong"
        elif confidence >= 80:
            return "medium"
        else:
            return "weak"

    async def format_alert_message_safe(self, signal: DivergenceSignal) -> str:
        """Formatear mensaje de alerta con seguridad"""
        try:
            confidence_emoji = '🔥' if signal.confidence >= 90 else '⚡' if signal.confidence >= 85 else '🟠'
            type_emoji = '📈🟢' if 'bullish' in signal.type else '📉🔴'
            
            message = f"""{confidence_emoji} **DIVERGENCIA DETECTADA** {confidence_emoji}

📌 **Par:** `{signal.symbol}`
💰 **Precio:** {signal.price_level:.6f}
{type_emoji} **Tipo:** {signal.type.replace('_', ' ').title()}
📊 **RSI:** {signal.rsi_value:.1f}
⏰ **TF:** {signal.timeframe}
🎯 **Confianza:** {signal.confidence:.0f}%
💪 **Fuerza:** {signal.pattern_strength.upper()}

📈 **Métricas:**
• RSI Div: {signal.rsi_divergence_strength:.1f}
• Price Div: {signal.price_divergence_strength:.1f}%

🤖 **Bot Ultra v3.0** | {signal.timestamp.strftime('%H:%M:%S')}"""
            
            return message
            
        except Exception as e:
            logger.error(f"❌ Error formateando mensaje: {e}")
            return f"🔥 **DIVERGENCIA DETECTADA**\n\n📌 **Par:** {signal.symbol}\n💰 **Precio:** {signal.price_level:.6f}"

    def is_duplicate_alert_safe(self, signal: DivergenceSignal) -> bool:
        """Verificar alertas duplicadas con seguridad"""
        try:
            alert_key = f"{signal.symbol}_{signal.timeframe}_{signal.type}"
            
            if alert_key in self.sent_alerts:
                last_alert = self.sent_alerts[alert_key]
                time_diff = datetime.now() - last_alert.get('timestamp', datetime.min)
                
                cooldown_times = {
                    '4h': 3600,   # 1 hora
                    '6h': 5400,   # 1.5 horas
                    '12h': 10800, # 3 horas
                    '1d': 14400   # 4 horas
                }
                
                cooldown = cooldown_times.get(signal.timeframe, 7200)
                
                if time_diff.total_seconds() < cooldown:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"❌ Error verificando duplicados: {e}")
            return False

    async def send_telegram_alert_safe(self, message: str):
        """Enviar alerta por Telegram con manejo de errores"""
        try:
            if not self.bot:
                self.bot = Bot(token=self.telegram_token)
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True
            )
            self.scan_stats['alerts_sent'] += 1
            logger.info("✅ Alerta enviada correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje Telegram: {e}")
            # Intentar sin markdown como fallback
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message.replace('*', '').replace('`', ''),
                    disable_web_page_preview=True
                )
                logger.info("✅ Alerta enviada sin formato")
            except Exception as e2:
                logger.error(f"❌ Error enviando mensaje sin formato: {e2}")

    async def scan_single_pair_safe(self, symbol: str):
        """Escanear un par con manejo de errores robusto"""
        try:
            scan_start = time.time()
            
            for timeframe in self.timeframes:
                try:
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                    # Obtener datos
                    data = await self.get_ohlcv_data_safe(symbol, timeframe, limit=100)
                    if data.empty:
                        continue
                    
                    # Detectar divergencias
                    signal = self.detect_divergence_safe(data, timeframe)
                    
                    if not signal:
                        continue
                    
                    signal.symbol = symbol
                    
                    # Verificar duplicados
                    if self.is_duplicate_alert_safe(signal):
                        continue
                    
                    # Registrar alerta
                    alert_key = f"{symbol}_{timeframe}_{signal.type}"
                    self.sent_alerts[alert_key] = {
                        'timestamp': datetime.now(),
                        'confidence': signal.confidence,
                        'date': datetime.now().date()
                    }
                    
                    # Enviar alerta
                    message = await self.format_alert_message_safe(signal)
                    await self.send_telegram_alert_safe(message)
                    
                    # Actualizar estadísticas
                    self.scan_stats['divergences_found'] += 1
                    
                    # Guardar en historial si ML está habilitado
                    if self.ml_model and len(self.pattern_history) < 1000:
                        self.pattern_history.append({
                            'signal': signal,
                            'timestamp': datetime.now()
                        })
                    
                    # Solo una alerta por par por ciclo
                    break
                    
                except Exception as e:
                    logger.error(f"❌ Error escaneando {symbol} {timeframe}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error escaneando par {symbol}: {e}")
            self.scan_stats['scan_errors'] += 1

    async def scan_all_pairs_safe(self):
        """Escanear todos los pares con manejo de errores"""
        try:
            scan_start = datetime.now()
            logger.info(f"🔄 Iniciando escaneo de {len(self.active_pairs)} pares...")
            
            # Procesar en batches pequeños para Railway
            batch_size = 5
            pairs_list = list(self.active_pairs)
            
            for i in range(0, len(pairs_list), batch_size):
                batch = pairs_list[i:i + batch_size]
                
                # Procesar batch
                tasks = [self.scan_single_pair_safe(symbol) for symbol in batch]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Pausa entre batches
                await asyncio.sleep(1)
            
            scan_duration = (datetime.now() - scan_start).total_seconds()
            self.scan_stats['scans_completed'] += 1
            self.scan_stats['last_scan_duration'] = scan_duration
            
            logger.info(f"✅ Escaneo completado en {scan_duration:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ Error en escaneo completo: {e}")

    def process_tradingview_alert(self):
        """Procesar webhook de TradingView con seguridad"""
        try:
            data = request.get_json() or {}
            logger.info(f"📡 Webhook TradingView recibido: {data}")
            
            # Validar campos básicos
            required_fields = ['symbol', 'type']
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Campos requeridos faltantes'}), 400
            
            # Crear señal básica
            signal = DivergenceSignal(
                symbol=data.get('symbol', 'UNKNOWN'),
                timeframe=data.get('timeframe', '1h'),
                type=data.get('type', 'bullish'),
                confidence=float(data.get('confidence', 85)),
                price_level=float(data.get('price', 0)),
                rsi_value=float(data.get('rsi', 50)),
                source='tradingview'
            )
            
            # Programar envío asíncrono
            asyncio.create_task(self.send_tradingview_alert_safe(signal))
            
            return jsonify({
                'status': 'success',
                'message': 'Alerta procesada',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error procesando webhook: {e}")
            return jsonify({'error': 'Error interno del servidor'}), 500

    async def send_tradingview_alert_safe(self, signal: DivergenceSignal):
        """Enviar alerta de TradingView con seguridad"""
        try:
            message = f"""🌐 **ALERTA TRADINGVIEW** 🌐

{await self.format_alert_message_safe(signal)}

🔗 **Fuente:** TradingView → Railway
⚡ **Procesamiento:** Automático"""
            
            await self.send_telegram_alert_safe(message)
            self.scan_stats['tradingview_alerts'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error enviando alerta TradingView: {e}")

    def start_flask_server(self):
        """Iniciar servidor Flask con manejo de errores"""
        try:
            logger.info(f"🌐 Iniciando servidor Flask en puerto {self.port}")
            self.app.run(
                host='0.0.0.0', 
                port=self.port, 
                debug=False, 
                threaded=True,
                use_reloader=False  # Importante para Railway
            )
        except Exception as e:
            logger.error(f"❌ Error iniciando Flask: {e}")
            raise

    async def setup_telegram_commands_safe(self):
        """Configurar comandos de Telegram con seguridad"""
        try:
            self.telegram_app = Application.builder().token(self.telegram_token).build()
            
            # Comandos básicos
            self.telegram_app.add_handler(CommandHandler("start", self.cmd_start))
            self.telegram_app.add_handler(CommandHandler("help", self.cmd_help))
            self.telegram_app.add_handler(CommandHandler("status", self.cmd_status))
            self.telegram_app.add_handler(CommandHandler("scan_now", self.cmd_scan_now))
            self.telegram_app.add_handler(CommandHandler("pairs", self.cmd_pairs))
            self.telegram_app.add_handler(CommandHandler("add", self.cmd_add_pair))
            self.telegram_app.add_handler(CommandHandler("remove", self.cmd_remove_pair))
            
            # Handler para mensajes no reconocidos
            self.telegram_app.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND, 
                self.handle_unknown_message
            ))
            
            # Inicializar
            await self.telegram_app.initialize()
            await self.telegram_app.start()
            
            logger.info("✅ Comandos de Telegram configurados")
            
            # Ejecutar polling OPTIMIZADO con configuración anti-conflicto
            await self.telegram_app.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                poll_interval=2.0,  # Aumentar intervalo de polling
                timeout=30,         # Timeout más largo
                bootstrap_retries=-1  # Reintentos infinitos
            )
            
        except Exception as e:
            logger.error(f"❌ Error configurando Telegram: {e}")

    # === COMANDOS DE TELEGRAM SIMPLIFICADOS ===
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start"""
        try:
            message = f"""🚀 *Bot RSI Divergence Ultra v3.0 FIXED*

✅ *Estado:* ONLINE
📊 *Pares activos:* {len(self.active_pairs)}
🤖 *ML:* {'✅ ACTIVO' if self.ml_model else '❌ INACTIVO'}

🔧 *Comandos:*
/status - Estado del sistema
/pairs - Ver pares monitoreados
/add SYMBOL - Agregar par
/remove SYMBOL - Quitar par
/scan_now - Escaneo manual
/help - Ayuda completa

🎯 *Optimizaciones aplicadas:*
- Manejo de errores robusto
- Rate limiting inteligente
- Cache optimizado
- Timeframe mapping corregido

💎 *Sistema funcionando 24/7 en Railway*"""
            
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            logger.error(f"❌ Error en /start: {e}")
            await update.message.reply_text("🤖 Bot RSI Divergence Ultra v3.0 ONLINE")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /status"""
        try:
            message = f"""📊 *Estado Bot RSI Ultra v3.0*

🔄 *Estado:* ✅ ONLINE (Versión CORREGIDA)
📈 *Pares monitoreados:* {len(self.active_pairs)}
🌐 *Total disponibles:* {len(self.all_bybit_pairs)}
⏰ *Timeframes:* {', '.join(self.timeframes)}

📊 *Estadísticas:*
- Escaneos: {self.scan_stats.get('scans_completed', 0)}
- Divergencias: {self.scan_stats.get('divergences_found', 0)}
- Alertas enviadas: {self.scan_stats.get('alerts_sent', 0)}
- TradingView: {self.scan_stats.get('tradingview_alerts', 0)}
- Errores: {self.scan_stats.get('scan_errors', 0)}

🤖 *Machine Learning:* {'✅ ACTIVO' if self.ml_model else '❌ INACTIVO'}
💾 *Cache:* {len(self.price_data_cache)} pares
⚡ *Último escaneo:* {self.scan_stats.get('last_scan_duration', 0):.1f}s

🌐 *Servidor:* Railway EU West
🔗 *Webhook:* ACTIVO"""
            
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            logger.error(f"❌ Error en /status: {e}")
            await update.message.reply_text("📊 Bot funcionando correctamente")

    async def cmd_scan_now(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /scan_now"""
        try:
            await update.message.reply_text("🔄 **Iniciando escaneo manual...**", parse_mode=ParseMode.MARKDOWN)
            
            start_time = datetime.now()
            await self.scan_all_pairs_safe()
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            recent_alerts = len([a for a in self.sent_alerts.values() 
                               if a.get('date') == datetime.now().date()])
            
            await update.message.reply_text(
                f"""✅ **Escaneo completado**

⏱️ **Duración:** {duration:.1f}s
📊 **Pares:** {len(self.active_pairs)}
🎯 **Alertas hoy:** {recent_alerts}
💾 **Cache:** {len(self.price_data_cache)}""",
                parse_mode=ParseMode.MARKDOWN
            )
            
        except Exception as e:
            logger.error(f"❌ Error en /scan_now: {e}")
            await update.message.reply_text("❌ Error ejecutando escaneo")

    async def cmd_pairs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /pairs - Ver todos los pares monitoreados"""
        try:
            if not self.active_pairs:
                await update.message.reply_text("📭 No hay pares activos", parse_mode=ParseMode.MARKDOWN)
                return
                
            # Organizar pares
            pairs_list = sorted(list(self.active_pairs))
            
            # Dividir en grupos de 10 para mejor lectura
            message = f"📊 **Pares Monitoreados ({len(self.active_pairs)} total)**\n\n"
            
            for i in range(0, len(pairs_list), 10):
                batch = pairs_list[i:i+10]
                message += "• " + " • ".join(batch) + "\n"
                
            message += f"\n💡 Usa `/add SYMBOL` para agregar pares"
            message += f"\n💡 Usa `/remove SYMBOL` para quitar pares"
            
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            logger.error(f"❌ Error en /pairs: {e}")
            await update.message.reply_text("❌ Error mostrando pares")

    async def cmd_add_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /add SYMBOL - Agregar par"""
        try:
            if not context.args:
                await update.message.reply_text(
                    "📝 **Uso:** `/add SYMBOL`\n\n**Ejemplos:**\n• `/add DOGEUSDT`\n• `/add ADAUSDT`",
                    parse_mode=ParseMode.MARKDOWN
                )
                return
                
            symbol = context.args[0].upper()
            
            # Verificar que termine en USDT
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
            
            # Verificar si ya está activo
            if symbol in self.active_pairs:
                await update.message.reply_text(f"⚠️ **{symbol}** ya está siendo monitoreado", parse_mode=ParseMode.MARKDOWN)
                return
                
            # Verificar que existe en Bybit (si la lista está disponible)
            if self.all_bybit_pairs and symbol not in self.all_bybit_pairs:
                # Buscar similares
                search_term = symbol.replace('USDT', '')
                similar = [p for p in self.all_bybit_pairs if search_term in p]
                
                message = f"❌ **{symbol}** no encontrado en Bybit"
                if similar[:5]:  # Mostrar solo los primeros 5
                    message += f"\n\n🔍 **Similares:**\n• " + "\n• ".join(similar[:5])
                    
                await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
                return
                
            # Agregar el par
            self.active_pairs.add(symbol)
            
            await update.message.reply_text(
                f"✅ **{symbol}** agregado al monitoreo\n📊 **Total pares:** {len(self.active_pairs)}",
                parse_mode=ParseMode.MARKDOWN
            )
            
        except Exception as e:
            logger.error(f"❌ Error en /add: {e}")
            await update.message.reply_text("❌ Error agregando par")

    async def cmd_remove_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /remove SYMBOL - Quitar par"""
        try:
            if not context.args:
                await update.message.reply_text(
                    "📝 **Uso:** `/remove SYMBOL`\n\n**Ejemplos:**\n• `/remove APEUSDT`\n• `/remove SHIBUSDT`",
                    parse_mode=ParseMode.MARKDOWN
                )
                return
                
            symbol = context.args[0].upper()
            
            # Verificar que termine en USDT
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
                
            if symbol not in self.active_pairs:
                await update.message.reply_text(f"❌ **{symbol}** no está en monitoreo", parse_mode=ParseMode.MARKDOWN)
                return
                
            self.active_pairs.remove(symbol)
            
            await update.message.reply_text(
                f"🗑️ **{symbol}** removido del monitoreo\n📊 **Total pares:** {len(self.active_pairs)}",
                parse_mode=ParseMode.MARKDOWN
            )
            
        except Exception as e:
            logger.error(f"❌ Error en /remove: {e}")
            await update.message.reply_text("❌ Error removiendo par")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /help"""
        try:
            message = """📋 **Ayuda - Bot RSI Ultra v3.0 CORREGIDO**

🤖 **¿Qué hace?**
Detecta divergencias RSI en múltiples timeframes con:
• Manejo de errores robusto
• Machine Learning opcional
• Rate limiting inteligente
• Cache optimizado

📊 **Comandos principales:**
• `/start` - Información inicial
• `/status` - Estado completo del sistema
• `/pairs` - Ver pares monitoreados
• `/add SYMBOL` - Agregar par (ej: /add DOGEUSDT)
• `/remove SYMBOL` - Quitar par (ej: /remove APEUSDT)
• `/scan_now` - Escaneo manual inmediato
• `/help` - Esta ayuda

🔧 **Correcciones aplicadas:**
• ✅ Importaciones condicionales
• ✅ Manejo de errores robusto
• ✅ Rate limiting optimizado
• ✅ Timeframe mapping corregido
• ✅ Cache inteligente
• ✅ Fallbacks para librerías

🌐 **Webhook TradingView:**
`https://tu-dominio.railway.app/webhook/tradingview`

💡 **Sistema ultra robusto funcionando 24/7**"""
            
            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            logger.error(f"❌ Error en /help: {e}")
            await update.message.reply_text("📋 Ayuda disponible - usar comandos básicos")

    async def handle_unknown_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manejar mensajes no reconocidos"""
        try:
            await update.message.reply_text(
                "❓ Comando no reconocido. Usa `/help` para ver comandos disponibles."
            )
        except Exception as e:
            logger.error(f"❌ Error en mensaje desconocido: {e}")

    async def smart_cache_cleanup_safe(self):
        """Limpieza de cache con seguridad"""
        try:
            now = time.time()
            
            # Limpiar cache expirado
            expired_keys = [
                key for key, data in self.price_data_cache.items()
                if now - data.get('timestamp', 0) > self.cache_expiry
            ]
            
            for key in expired_keys:
                try:
                    del self.price_data_cache[key]
                except KeyError:
                    pass
            
            # Limpiar alertas antiguas
            cutoff = datetime.now() - timedelta(hours=24)
            self.sent_alerts = {
                k: v for k, v in self.sent_alerts.items() 
                if v.get('timestamp', datetime.min) > cutoff
            }
            
            if len(expired_keys) > 0:
                logger.info(f"🧹 Cache limpiado: {len(expired_keys)} entradas")
                
        except Exception as e:
            logger.error(f"❌ Error en limpieza de cache: {e}")

    async def start_monitoring_safe(self):
        """Iniciar monitoreo con manejo de errores robusto"""
        logger.info("🚀 Iniciando Bot RSI Divergence Ultra v3.0 CORREGIDO")
        
        try:
            # Configurar Telegram
            await self.setup_telegram_commands_safe()
            
            # Mensaje de inicio
            startup_message = f"""🚀 **Bot RSI Divergence Ultra v3.0 ONLINE**

🌐 **Plataforma:** Railway EU West
🛠️ **Versión:** CORREGIDA con manejo de errores robusto
📊 **Pares monitoreados:** {len(self.active_pairs)}
⏰ **Timeframes:** {', '.join(self.timeframes)}

✨ **Correcciones aplicadas:**
• ✅ Importaciones condicionales (scipy, talib, sklearn)
• ✅ Manejo de errores en todas las funciones
• ✅ Rate limiting optimizado para Railway
• ✅ Cache inteligente con limpieza automática
• ✅ Timeframe mapping corregido
• ✅ Fallbacks para todas las librerías

🎯 **Sistema ultra robusto funcionando 24/7**

Usa `/help` para ver todos los comandos."""
            
            await self.send_telegram_alert_safe(startup_message)
            
            # Loop principal con manejo de errores
            while True:
                try:
                    loop_start = time.time()
                    
                    # Escaneo principal
                    await self.scan_all_pairs_safe()
                    
                    # Limpieza de cache
                    await self.smart_cache_cleanup_safe()
                    
                    # Estadísticas de rendimiento
                    loop_duration = time.time() - loop_start
                    
                    # Pausa inteligente (10 minutos)
                    await asyncio.sleep(600)
                    
                except Exception as e:
                    logger.error(f"❌ Error en loop principal: {e}")
                    logger.error(traceback.format_exc())
                    # Pausa corta en caso de error
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"❌ Error crítico en monitoreo: {e}")
            logger.error(traceback.format_exc())
            # Reintentar después de un tiempo
            await asyncio.sleep(300)

    def run_safe(self):
        """Punto de entrada ultra seguro"""
        logger.info("🚀 Iniciando Bot RSI Divergence Ultra v3.0 CORREGIDO...")
        
        try:
            # Iniciar Flask en thread separado
            flask_thread = threading.Thread(target=self.start_flask_server, daemon=True)
            flask_thread.start()
            logger.info("✅ Servidor Flask iniciado correctamente")
            
            # Pequeña pausa para asegurar que Flask esté listo
            time.sleep(2)
            
            # Iniciar loop principal
            asyncio.run(self.start_monitoring_safe())
            
        except KeyboardInterrupt:
            logger.info("🛑 Bot detenido por usuario")
        except Exception as e:
            logger.error(f"❌ Error crítico: {e}")
            logger.error(traceback.format_exc())
            # En caso de error crítico, reintentar
            time.sleep(30)
            logger.info("🔄 Reintentando inicialización...")

# === FUNCIONES DE UTILIDAD SEGURAS ===

def safe_float_conversion(value, default=0.0):
    """Conversión segura a float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value, default=0):
    """Conversión segura a int"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def validate_environment():
    """Validar variables de entorno"""
    required_vars = ['TELEGRAM_TOKEN', 'CHAT_ID']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"❌ Variables de entorno faltantes: {', '.join(missing_vars)}")
    
    logger.info("✅ Variables de entorno validadas")

# === PUNTO DE ENTRADA PRINCIPAL ===

def main():
    """Función principal ultra segura para Railway"""
    try:
        # Validar entorno
        validate_environment()
        
        # Crear e iniciar bot
        logger.info("🚀 Iniciando Bot RSI Divergence Ultra v3.0 CORREGIDO...")
        bot = RSIDivergenceBot()
        bot.run_safe()
        
    except Exception as e:
        logger.error(f"❌ Error crítico iniciando bot: {e}")
        logger.error(traceback.format_exc())
        
        # En producción, podrías querer reintentar
        time.sleep(10)
        logger.info("🔄 Reintentando inicialización después del error...")
        
        try:
            bot = RSIDivergenceBot()
            bot.run_safe()
        except Exception as e2:
            logger.error(f"❌ Segundo intento falló: {e2}")
            raise

if __name__ == "__main__":
    main()
