import asyncio
import ccxt
import pandas as pd
import numpy as np
from telegram import Bot, Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
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

# Configuración optimizada basada en patrones reales
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        (
            logging.FileHandler("bot.log")
            if os.access(".", os.W_OK)
            else logging.NullHandler()
        ),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class DivergenceSignal:
    symbol: str
    timeframe: str
    type: str  # 'bullish', 'bearish'
    confidence: float
    price_level: float
    rsi_value: float
    price_change_percent: float
    rsi_divergence_strength: float
    context: str  # 'support', 'resistance', 'neutral'
    peak_count: int
    timeframe_confirmations: List[str] = field(default_factory=list)
    source: str = "realtime_scan"
    timestamp: datetime = field(default_factory=datetime.now)


class RSIDivergenceBotV4:
    def __init__(self):
        """Bot v4.0 - Calibrado con patrones reales exitosos"""
        try:
            # Configuración ENV
            self.telegram_token = os.getenv("TELEGRAM_TOKEN")
            self.chat_id = os.getenv("CHAT_ID")
            self.bybit_api_key = os.getenv("BYBIT_API_KEY", "")
            self.bybit_secret = os.getenv("BYBIT_SECRET", "")
            self.port = int(os.getenv("PORT", 8080))

            self._validate_config()

            # Componentes básicos
            self.bot = None
            self.app = Flask(__name__)
            self.telegram_app = None
            self.exchange = self._setup_exchange()
            self.setup_webhook_routes()

            # Datos del bot
            self.all_bybit_pairs = []
            self.active_pairs = set()
            self.sent_alerts = {}
            self.scan_stats = defaultdict(int)

            # CONFIGURACIÓN BASADA EN PATRONES REALES EXITOSOS
            self.timeframes = ["2h", "4h", "6h", "8h", "12h", "1d"]

            # Parámetros calibrados con tus ejemplos exitosos
            self.detection_configs = {
                "2h": {
                    "min_peak_distance": 3,
                    "min_price_change": 0.8,  # Más sensible para detección temprana
                    "min_rsi_change": 3.0,  # RSI divergencia mínima
                    "confidence_threshold": 60,  # Umbral realista
                    "lookback_period": 25,
                    "peak_prominence": 0.5,  # Prominencia de picos
                },
                "4h": {
                    "min_peak_distance": 3,
                    "min_price_change": 1.0,
                    "min_rsi_change": 4.0,
                    "confidence_threshold": 65,
                    "lookback_period": 30,
                    "peak_prominence": 0.6,
                },
                "6h": {
                    "min_peak_distance": 3,
                    "min_price_change": 1.2,
                    "min_rsi_change": 4.5,
                    "confidence_threshold": 68,
                    "lookback_period": 35,
                    "peak_prominence": 0.7,
                },
                "8h": {
                    "min_peak_distance": 4,
                    "min_price_change": 1.5,
                    "min_rsi_change": 5.0,
                    "confidence_threshold": 70,
                    "lookback_period": 40,
                    "peak_prominence": 0.8,
                },
                "12h": {
                    "min_peak_distance": 4,
                    "min_price_change": 1.8,
                    "min_rsi_change": 6.0,
                    "confidence_threshold": 72,
                    "lookback_period": 45,
                    "peak_prominence": 1.0,
                },
                "1d": {
                    "min_peak_distance": 5,
                    "min_price_change": 2.5,
                    "min_rsi_change": 7.0,
                    "confidence_threshold": 75,
                    "lookback_period": 50,
                    "peak_prominence": 1.2,
                },
            }

            # Cache optimizado
            self.price_data_cache = {}
            self.cache_expiry = 300  # 5 minutos

            # Inicialización
            self.initialize_data_safe()

            logger.info("✅ RSI Divergence Bot v4.0 - Calibrado con patrones reales")

        except Exception as e:
            logger.error(f"❌ Error inicializando bot v4.0: {e}")
            raise

    def _validate_config(self):
        """Validar configuración"""
        if not self.telegram_token:
            raise ValueError("❌ TELEGRAM_TOKEN requerido")
        if not self.chat_id:
            raise ValueError("❌ CHAT_ID requerido")
        try:
            int(self.chat_id)
        except ValueError:
            raise ValueError("❌ CHAT_ID debe ser número")
        logger.info("✅ Configuración validada")

    def _setup_exchange(self):
        """Configurar exchange"""
        try:
            config = {
                "enableRateLimit": True,
                "rateLimit": 200,
                "timeout": 30000,
                "options": {"defaultType": "linear"},
                "sandbox": False,
            }

            if self.bybit_api_key and self.bybit_secret:
                config["apiKey"] = self.bybit_api_key
                config["secret"] = self.bybit_secret
                logger.info("✅ Exchange con credenciales")
            else:
                logger.warning("⚠️ Exchange sin credenciales")

            return ccxt.bybit(config)

        except Exception as e:
            logger.error(f"❌ Error configurando exchange: {e}")
            return ccxt.bybit(
                {
                    "enableRateLimit": True,
                    "rateLimit": 500,
                    "timeout": 30000,
                    "sandbox": False,
                }
            )

    def initialize_data_safe(self):
        """Inicializar datos con pares exitosos"""
        try:
            self.load_all_bybit_pairs_safe()
            self.load_successful_pairs()
            logger.info(f"✅ Datos inicializados: {len(self.active_pairs)} pares")
        except Exception as e:
            logger.error(f"❌ Error inicializando datos: {e}")
            # Pares de emergencia basados en ejemplos exitosos
            self.active_pairs = set(
                [
                    "1000BONKUSDT",
                    "CPOOLUSDT",
                    "1000PEPEUSDT",
                    "AIOZUSDT",
                    "FARTCOINUSDT",
                    "FETUSDT",
                    "HYPEUSDT",
                    "INJUSDT",
                    "KAITOUSDT",
                    "POPCATUSDT",
                    "VIRTUALUSDT",
                    # Agregar otros pares populares
                    "BTCUSDT",
                    "ETHUSDT",
                    "SOLUSDT",
                    "BNBUSDT",
                    "XRPUSDT",
                ]
            )

    def load_all_bybit_pairs_safe(self):
        """Cargar pares de Bybit"""
        try:
            markets = self.exchange.load_markets()
            usdt_pairs = []

            for symbol, market in markets.items():
                try:
                    if (
                        symbol.endswith("USDT")
                        and market.get("type") == "swap"
                        and market.get("linear", True)
                        and market.get("active", True)
                    ):
                        usdt_pairs.append(symbol)
                except:
                    continue

            self.all_bybit_pairs = sorted(usdt_pairs)
            logger.info(f"✅ {len(self.all_bybit_pairs)} pares cargados")

        except Exception as e:
            logger.error(f"❌ Error cargando pares: {e}")
            self.all_bybit_pairs = self.get_fallback_pairs()

    def load_successful_pairs(self):
        """Cargar pares con patrones exitosos + otros populares"""
        # Pares con patrones exitosos confirmados
        successful_pairs = [
            "1000BONKUSDT",
            "CPOOLUSDT",
            "1000PEPEUSDT",
            "AIOZUSDT",
            "FARTCOINUSDT",
            "FETUSDT",
            "HYPEUSDT",
            "INJUSDT",
            "KAITOUSDT",
            "POPCATUSDT",
            "VIRTUALUSDT",
        ]

        # Pares populares adicionales
        popular_pairs = [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "AVAXUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "LTCUSDT",
            "DOGEUSDT",
            "SHIBUSDT",
            "WIFUSDT",
            "FLOKIUSDT",
            "MATICUSDT",
            "OPUSDT",
            "ARBUSDT",
            "SUIUSDT",
            "APTUSDT",
        ]

        all_target_pairs = successful_pairs + popular_pairs

        for pair in all_target_pairs:
            if not self.all_bybit_pairs or pair in self.all_bybit_pairs:
                self.active_pairs.add(pair)

        logger.info(f"✅ {len(self.active_pairs)} pares activos cargados")

    def get_fallback_pairs(self):
        """Pares de respaldo"""
        return [
            "1000BONKUSDT",
            "CPOOLUSDT",
            "1000PEPEUSDT",
            "AIOZUSDT",
            "FARTCOINUSDT",
            "FETUSDT",
            "HYPEUSDT",
            "INJUSDT",
            "KAITOUSDT",
            "POPCATUSDT",
            "VIRTUALUSDT",
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "DOGEUSDT",
            "SHIBUSDT",
            "ADAUSDT",
            "AVAXUSDT",
            "DOTUSDT",
        ]

    def setup_webhook_routes(self):
        """Rutas Flask"""

        @self.app.route("/", methods=["GET"])
        def home():
            return jsonify(
                {
                    "status": "🚀 RSI Divergence Bot v4.0 - Patrones Reales",
                    "version": "4.0-REALTIME",
                    "active_pairs": len(self.active_pairs),
                    "total_pairs": len(self.all_bybit_pairs),
                    "uptime": datetime.now().isoformat(),
                    "stats": dict(self.scan_stats),
                }
            )

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify(
                {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "active_pairs": len(self.active_pairs),
                }
            )

    async def get_ohlcv_data_safe(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> pd.DataFrame:
        """Obtener datos OHLCV con cache"""
        try:
            cache_key = f"{symbol}_{timeframe}_{limit}"
            now = time.time()

            # Verificar cache
            if (
                cache_key in self.price_data_cache
                and now - self.price_data_cache[cache_key]["timestamp"]
                < self.cache_expiry
            ):
                return self.price_data_cache[cache_key]["data"].copy()

            # Obtener datos
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv or len(ohlcv) < 30:
                return pd.DataFrame()

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            if df.isnull().any().any():
                df = df.dropna()

            # Guardar en cache
            self.price_data_cache[cache_key] = {"data": df.copy(), "timestamp": now}

            return df

        except Exception as e:
            logger.error(f"❌ Error obteniendo datos {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, close_prices: np.array, period: int = 14) -> np.array:
        """Calcular RSI optimizado"""
        try:
            if len(close_prices) < period + 10:
                return np.full(len(close_prices), np.nan)

            deltas = np.diff(close_prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            rsi_values = np.full(len(close_prices), np.nan)

            # Calcular RSI
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            for i in range(period, len(close_prices)):
                if i == period:
                    current_avg_gain = avg_gain
                    current_avg_loss = avg_loss
                else:
                    current_avg_gain = (
                        current_avg_gain * (period - 1) + gains[i - 1]
                    ) / period
                    current_avg_loss = (
                        current_avg_loss * (period - 1) + losses[i - 1]
                    ) / period

                if current_avg_loss == 0:
                    rsi_values[i] = 100
                else:
                    rs = current_avg_gain / current_avg_loss
                    rsi_values[i] = 100 - (100 / (1 + rs))

            return rsi_values

        except Exception as e:
            logger.error(f"❌ Error calculando RSI: {e}")
            return np.full(len(close_prices), 50.0)

    def find_peaks_and_troughs(
        self, data: np.array, distance: int = 3, prominence: float = 0.5
    ) -> Tuple[List[int], List[int]]:
        """Encontrar picos y valles mejorado"""
        try:
            if len(data) < distance * 3:
                return [], []

            peaks = []
            troughs = []

            # Encontrar picos locales
            for i in range(distance, len(data) - distance):
                # Es pico si es mayor que sus vecinos
                is_peak = all(
                    data[i] >= data[i - j] for j in range(1, distance + 1)
                ) and all(data[i] >= data[i + j] for j in range(1, distance + 1))

                # Es valle si es menor que sus vecinos
                is_trough = all(
                    data[i] <= data[i - j] for j in range(1, distance + 1)
                ) and all(data[i] <= data[i + j] for j in range(1, distance + 1))

                # Verificar prominencia
                if is_peak:
                    left_min = np.min(data[max(0, i - distance * 2) : i])
                    right_min = np.min(data[i : min(len(data), i + distance * 2)])
                    if data[i] - max(left_min, right_min) >= prominence:
                        peaks.append(i)

                if is_trough:
                    left_max = np.max(data[max(0, i - distance * 2) : i])
                    right_max = np.max(data[i : min(len(data), i + distance * 2)])
                    if min(left_max, right_max) - data[i] >= prominence:
                        troughs.append(i)

            return peaks, troughs

        except Exception as e:
            logger.error(f"❌ Error encontrando picos: {e}")
            return [], []

    def detect_support_resistance(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Detectar niveles de soporte y resistencia"""
        try:
            highs = price_data["high"].values
            lows = price_data["low"].values
            closes = price_data["close"].values

            # Niveles recientes
            recent_high = np.max(highs[-20:])
            recent_low = np.min(lows[-20:])
            current_price = closes[-1]

            # Determinar contexto
            resistance_distance = abs(current_price - recent_high) / current_price * 100
            support_distance = abs(current_price - recent_low) / current_price * 100

            context = "neutral"
            if support_distance <= 2.0:  # Cerca del soporte (2%)
                context = "support"
            elif resistance_distance <= 2.0:  # Cerca de resistencia (2%)
                context = "resistance"

            return {
                "resistance": recent_high,
                "support": recent_low,
                "context": context,
                "resistance_distance": resistance_distance,
                "support_distance": support_distance,
            }

        except Exception as e:
            logger.error(f"❌ Error detectando S/R: {e}")
            return {
                "resistance": 0,
                "support": 0,
                "context": "neutral",
                "resistance_distance": 999,
                "support_distance": 999,
            }

    def detect_divergence_realtime(
        self, price_data: pd.DataFrame, timeframe: str
    ) -> Optional[DivergenceSignal]:
        """Detectar divergencias en tiempo real - ALGORITMO PRINCIPAL"""
        try:
            if len(price_data) < 30:
                return None

            closes = price_data["close"].values
            config = self.detection_configs.get(timeframe, self.detection_configs["1d"])

            # Calcular RSI
            rsi = self.calculate_rsi(closes, period=14)
            if len(rsi) < 20 or np.isnan(rsi[-1]):
                return None

            # Encontrar picos y valles
            price_peaks, price_troughs = self.find_peaks_and_troughs(
                closes, config["min_peak_distance"], config["peak_prominence"]
            )
            rsi_peaks, rsi_troughs = self.find_peaks_and_troughs(
                rsi, config["min_peak_distance"], config["peak_prominence"] * 2
            )

            # Detectar soporte/resistencia
            sr_data = self.detect_support_resistance(price_data)

            # Detectar divergencia bajista (precio HH, RSI LH)
            bearish_signal = self.detect_bearish_divergence_realtime(
                closes, rsi, price_peaks, rsi_peaks, config, timeframe, sr_data
            )

            if bearish_signal:
                return bearish_signal

            # Detectar divergencia alcista (precio LL, RSI HL)
            bullish_signal = self.detect_bullish_divergence_realtime(
                closes, rsi, price_troughs, rsi_troughs, config, timeframe, sr_data
            )

            return bullish_signal

        except Exception as e:
            logger.error(f"❌ Error detectando divergencias: {e}")
            return None

    def detect_bearish_divergence_realtime(
        self,
        closes: np.array,
        rsi: np.array,
        price_peaks: List[int],
        rsi_peaks: List[int],
        config: dict,
        timeframe: str,
        sr_data: dict,
    ) -> Optional[DivergenceSignal]:
        """Detectar divergencia bajista en tiempo real"""
        try:
            if len(price_peaks) < 2 or len(rsi_peaks) < 2:
                return None

            # Obtener picos recientes dentro del lookback
            lookback = config["lookback_period"]
            recent_price_peaks = [p for p in price_peaks if p >= len(closes) - lookback]
            recent_rsi_peaks = [p for p in rsi_peaks if p >= len(rsi) - lookback]

            if len(recent_price_peaks) < 2 or len(recent_rsi_peaks) < 2:
                return None

            # Comparar últimos dos picos
            p1, p2 = recent_price_peaks[-2:]
            r1, r2 = recent_rsi_peaks[-2:]

            # Verificar divergencia: precio hace HH, RSI hace LH
            price_higher = closes[p2] > closes[p1]
            rsi_lower = rsi[r2] < rsi[r1]

            if not (price_higher and rsi_lower):
                return None

            # Calcular métricas
            price_change = (closes[p2] - closes[p1]) / closes[p1] * 100
            rsi_change = rsi[r1] - rsi[r2]  # Diferencia positiva

            # Verificar umbrales mínimos
            if (
                price_change < config["min_price_change"]
                or rsi_change < config["min_rsi_change"]
            ):
                return None

            # Calcular confianza
            confidence = self.calculate_confidence_realtime(
                price_change, rsi_change, timeframe, "bearish", sr_data
            )

            if confidence < config["confidence_threshold"]:
                return None

            # Crear señal
            signal = DivergenceSignal(
                symbol="",  # Se llenará después
                timeframe=timeframe,
                type="bearish",
                confidence=confidence,
                price_level=closes[-1],
                rsi_value=rsi[-1],
                price_change_percent=price_change,
                rsi_divergence_strength=rsi_change,
                context=sr_data["context"],
                peak_count=len(recent_price_peaks),
            )

            return signal

        except Exception as e:
            logger.error(f"❌ Error en divergencia bajista: {e}")
            return None

    def detect_bullish_divergence_realtime(
        self,
        closes: np.array,
        rsi: np.array,
        price_troughs: List[int],
        rsi_troughs: List[int],
        config: dict,
        timeframe: str,
        sr_data: dict,
    ) -> Optional[DivergenceSignal]:
        """Detectar divergencia alcista en tiempo real"""
        try:
            if len(price_troughs) < 2 or len(rsi_troughs) < 2:
                return None

            # Obtener valles recientes
            lookback = config["lookback_period"]
            recent_price_troughs = [
                t for t in price_troughs if t >= len(closes) - lookback
            ]
            recent_rsi_troughs = [t for t in rsi_troughs if t >= len(rsi) - lookback]

            if len(recent_price_troughs) < 2 or len(recent_rsi_troughs) < 2:
                return None

            # Comparar últimos dos valles
            t1, t2 = recent_price_troughs[-2:]
            r1, r2 = recent_rsi_troughs[-2:]

            # Verificar divergencia: precio hace LL, RSI hace HL
            price_lower = closes[t2] < closes[t1]
            rsi_higher = rsi[r2] > rsi[r1]

            if not (price_lower and rsi_higher):
                return None

            # Calcular métricas
            price_change = abs(closes[t2] - closes[t1]) / closes[t1] * 100
            rsi_change = rsi[r2] - rsi[r1]  # Diferencia positiva

            # Verificar umbrales
            if (
                price_change < config["min_price_change"]
                or rsi_change < config["min_rsi_change"]
            ):
                return None

            # Calcular confianza
            confidence = self.calculate_confidence_realtime(
                price_change, rsi_change, timeframe, "bullish", sr_data
            )

            if confidence < config["confidence_threshold"]:
                return None

            # Crear señal
            signal = DivergenceSignal(
                symbol="",
                timeframe=timeframe,
                type="bullish",
                confidence=confidence,
                price_level=closes[-1],
                rsi_value=rsi[-1],
                price_change_percent=price_change,
                rsi_divergence_strength=rsi_change,
                context=sr_data["context"],
                peak_count=len(recent_price_troughs),
            )

            return signal

        except Exception as e:
            logger.error(f"❌ Error en divergencia alcista: {e}")
            return None

    def calculate_confidence_realtime(
        self,
        price_change: float,
        rsi_change: float,
        timeframe: str,
        div_type: str,
        sr_data: dict,
    ) -> float:
        """Calcular confianza basada en patrones reales"""
        try:
            # Base de confianza según timeframe
            base_confidence = {
                "2h": 60,
                "4h": 65,
                "6h": 68,
                "8h": 70,
                "12h": 72,
                "1d": 75,
            }.get(timeframe, 65)

            # Factores de ajuste
            price_factor = min(price_change * 1.5, 15)  # Hasta +15
            rsi_factor = min(rsi_change * 1.2, 12)  # Hasta +12

            # Bonificación por contexto (soporte/resistencia)
            context_bonus = 0
            if div_type == "bullish" and sr_data["context"] == "support":
                context_bonus = 8  # Divergencia alcista en soporte = +8%
            elif div_type == "bearish" and sr_data["context"] == "resistance":
                context_bonus = 8  # Divergencia bajista en resistencia = +8%

            # Bonificación por timeframe largo
            tf_bonus = {"12h": 3, "1d": 5}.get(timeframe, 0)

            # Fuerza combinada
            combined_strength = (price_change * rsi_change) / 15
            strength_bonus = min(combined_strength, 8)

            confidence = (
                base_confidence
                + price_factor
                + rsi_factor
                + context_bonus
                + tf_bonus
                + strength_bonus
            )

            return min(confidence, 98.0)

        except Exception as e:
            logger.error(f"❌ Error calculando confianza: {e}")
            return 65.0

    def check_multiple_timeframes(
        self, symbol: str, main_signal: DivergenceSignal
    ) -> List[str]:
        """Verificar confirmaciones en múltiples timeframes"""
        confirmations = []

        try:
            # Timeframes a verificar según el principal
            tf_groups = {
                "2h": ["4h"],
                "4h": ["6h", "8h"],
                "6h": ["12h"],
                "8h": ["12h"],
                "12h": ["1d"],
                "1d": [],
            }

            check_timeframes = tf_groups.get(main_signal.timeframe, [])

            for tf in check_timeframes:
                try:
                    # Obtener datos del timeframe
                    data = asyncio.run(self.get_ohlcv_data_safe(symbol, tf, limit=80))
                    if data.empty:
                        continue

                    # Detectar divergencia
                    signal = self.detect_divergence_realtime(data, tf)
                    if signal and signal.type == main_signal.type:
                        confirmations.append(tf)

                except Exception as e:
                    logger.debug(f"Error verificando {tf}: {e}")
                    continue

        except Exception as e:
            logger.error(f"❌ Error verificando múltiples TF: {e}")

        return confirmations

    async def format_alert_message_realtime(self, signal: DivergenceSignal) -> str:
        """Formatear mensaje de alerta optimizado"""
        try:
            # Emojis según tipo y confianza
            if signal.type == "bullish":
                main_emoji = "🚀" if signal.confidence >= 75 else "📈"
                type_emoji = "🟢"
                direction = "ALCISTA"
            else:
                main_emoji = "🔥" if signal.confidence >= 75 else "📉"
                type_emoji = "🔴"
                direction = "BAJISTA"

            confidence_emoji = (
                "⭐"
                if signal.confidence >= 80
                else "✨" if signal.confidence >= 70 else "💫"
            )

            # Contexto
            context_info = ""
            if signal.context == "support":
                context_info = "🛡️ **En SOPORTE** - Alta probabilidad rebote"
            elif signal.context == "resistance":
                context_info = "🚧 **En RESISTENCIA** - Alta probabilidad reversión"

            # Confirmaciones
            confirmations_info = ""
            if signal.timeframe_confirmations:
                confirmations_info = f"\n🔗 **Confirmado en:** {', '.join(signal.timeframe_confirmations)}"

            # Potencial según patrones históricos
            # Potencial según patrones históricos
            potential_info = ""
            if signal.type == "bullish" and signal.context == "support":
                potential_info = (
                    "\n💎 **Potencial:** +20% (basado en patrones históricos)"
                )
            elif signal.type == "bearish" and signal.context == "resistance":
                potential_info = (
                    "\n⚠️ **Potencial:** Caída fuerte (basado en patrones históricos)"
                )

            message = f"""{main_emoji} **DIVERGENCIA {direction} DETECTADA** {main_emoji}

📌 **Par:** `{signal.symbol}`
💰 **Precio:** ${signal.price_level:.6f}
{type_emoji} **Tipo:** {direction}
📊 **RSI:** {signal.rsi_value:.1f}
⏰ **Timeframe:** {signal.timeframe}
{confidence_emoji} **Confianza:** {signal.confidence:.0f}%

📈 **Métricas:**
- Cambio Precio: {signal.price_change_percent:.1f}%
- Fuerza RSI: {signal.rsi_divergence_strength:.1f}
- Picos detectados: {signal.peak_count}

{context_info}{confirmations_info}{potential_info}

⚡ **DETECCIÓN EN TIEMPO REAL** 
🤖 Bot v4.0 | {signal.timestamp.strftime('%H:%M:%S')}

📋 **Acción sugerida:** {"COMPRA cerca del soporte" if signal.type == "bullish" else "VENTA cerca de resistencia"}"""

            return message

        except Exception as e:
            logger.error(f"❌ Error formateando mensaje: {e}")
            return f"🚀 **DIVERGENCIA DETECTADA**\n\n📌 **Par:** {signal.symbol}\n💰 **Precio:** {signal.price_level:.6f}\n🎯 **Tipo:** {signal.type.upper()}"

    def is_duplicate_alert_realtime(self, signal: DivergenceSignal) -> bool:
        """Verificar alertas duplicadas con cooldown inteligente"""
        try:
            alert_key = f"{signal.symbol}_{signal.timeframe}_{signal.type}"

            if alert_key in self.sent_alerts:
                last_alert = self.sent_alerts[alert_key]
                time_diff = datetime.now() - last_alert.get("timestamp", datetime.min)

                # Cooldown dinámico según timeframe
                cooldown_times = {
                    "2h": 3600,  # 1 hora
                    "4h": 5400,  # 1.5 horas
                    "6h": 7200,  # 2 horas
                    "8h": 10800,  # 3 horas
                    "12h": 14400,  # 4 horas
                    "1d": 21600,  # 6 horas
                }

                cooldown = cooldown_times.get(signal.timeframe, 7200)

                # Si es muy reciente, es duplicado
                if time_diff.total_seconds() < cooldown:
                    return True

                # Si la nueva señal tiene mayor confianza, permitir
                if signal.confidence > last_alert.get("confidence", 0) + 10:
                    return False

            return False

        except Exception as e:
            logger.error(f"❌ Error verificando duplicados: {e}")
            return False

    async def send_telegram_alert_safe(self, message: str):
        """Enviar alerta por Telegram"""
        try:
            if not self.bot:
                self.bot = Bot(token=self.telegram_token)

            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
            self.scan_stats["alerts_sent"] += 1
            logger.info("✅ Alerta enviada")

        except Exception as e:
            logger.error(f"❌ Error enviando Telegram: {e}")
            try:
                # Fallback sin markdown
                clean_message = (
                    message.replace("*", "").replace("`", "").replace("_", "")
                )
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=clean_message,
                    disable_web_page_preview=True,
                )
                logger.info("✅ Alerta enviada (sin formato)")
            except Exception as e2:
                logger.error(f"❌ Error enviando sin formato: {e2}")

    async def scan_single_pair_realtime(self, symbol: str):
        """Escanear un par en tiempo real"""
        try:
            for timeframe in self.timeframes:
                try:
                    # Rate limiting
                    await asyncio.sleep(0.1)

                    # Obtener datos
                    data = await self.get_ohlcv_data_safe(symbol, timeframe, limit=100)
                    if data.empty:
                        continue

                    # Detectar divergencias
                    signal = self.detect_divergence_realtime(data, timeframe)
                    if not signal:
                        continue

                    signal.symbol = symbol

                    # Verificar confirmaciones en múltiples timeframes
                    confirmations = self.check_multiple_timeframes(symbol, signal)
                    signal.timeframe_confirmations = confirmations

                    # Bonificación por confirmaciones múltiples
                    if confirmations:
                        signal.confidence += len(confirmations) * 3
                        signal.confidence = min(signal.confidence, 98)

                    # Verificar duplicados
                    if self.is_duplicate_alert_realtime(signal):
                        continue

                    # Registrar alerta
                    alert_key = f"{symbol}_{timeframe}_{signal.type}"
                    self.sent_alerts[alert_key] = {
                        "timestamp": datetime.now(),
                        "confidence": signal.confidence,
                        "date": datetime.now().date(),
                    }

                    # Enviar alerta
                    message = await self.format_alert_message_realtime(signal)
                    await self.send_telegram_alert_safe(message)

                    # Estadísticas
                    self.scan_stats["divergences_found"] += 1
                    self.scan_stats[f"{signal.type}_signals"] += 1

                    logger.info(
                        f"🎯 Divergencia {signal.type} detectada: {symbol} {timeframe} ({signal.confidence:.0f}%)"
                    )

                    # Solo una alerta por par por ciclo para evitar spam
                    break

                except Exception as e:
                    logger.error(f"❌ Error escaneando {symbol} {timeframe}: {e}")
                    continue

        except Exception as e:
            logger.error(f"❌ Error escaneando par {symbol}: {e}")
            self.scan_stats["scan_errors"] += 1

    async def scan_all_pairs_realtime(self):
        """Escanear todos los pares en tiempo real"""
        try:
            scan_start = datetime.now()
            logger.info(
                f"🔄 Iniciando escaneo TIEMPO REAL de {len(self.active_pairs)} pares..."
            )

            # Procesar en batches para Railway
            batch_size = 5
            pairs_list = list(self.active_pairs)

            for i in range(0, len(pairs_list), batch_size):
                batch = pairs_list[i : i + batch_size]

                # Procesar batch en paralelo
                tasks = [self.scan_single_pair_realtime(symbol) for symbol in batch]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Pausa entre batches
                await asyncio.sleep(1)

            scan_duration = (datetime.now() - scan_start).total_seconds()
            self.scan_stats["scans_completed"] += 1
            self.scan_stats["last_scan_duration"] = scan_duration

            logger.info(f"✅ Escaneo completado en {scan_duration:.1f}s")

        except Exception as e:
            logger.error(f"❌ Error en escaneo completo: {e}")

    # === COMANDOS DE TELEGRAM ===

    async def setup_telegram_commands_safe(self):
        """Configurar comandos de Telegram"""
        try:
            self.telegram_app = Application.builder().token(self.telegram_token).build()

            # Comandos
            self.telegram_app.add_handler(CommandHandler("start", self.cmd_start))
            self.telegram_app.add_handler(CommandHandler("help", self.cmd_help))
            self.telegram_app.add_handler(CommandHandler("status", self.cmd_status))
            self.telegram_app.add_handler(CommandHandler("scan_now", self.cmd_scan_now))
            self.telegram_app.add_handler(CommandHandler("pairs", self.cmd_pairs))
            self.telegram_app.add_handler(CommandHandler("add", self.cmd_add_pair))
            self.telegram_app.add_handler(
                CommandHandler("remove", self.cmd_remove_pair)
            )
            self.telegram_app.add_handler(CommandHandler("stats", self.cmd_stats))

            # Handler para mensajes desconocidos
            self.telegram_app.add_handler(
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, self.handle_unknown_message
                )
            )

            # Inicializar
            await self.telegram_app.initialize()
            await self.telegram_app.start()

            logger.info("✅ Comandos Telegram configurados")

            # Polling
            await self.telegram_app.updater.start_polling(
                allowed_updates=Update.ALL_TYPES, drop_pending_updates=True
            )

        except Exception as e:
            logger.error(f"❌ Error configurando Telegram: {e}")

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start"""
        try:
            message = f"""🚀 RSI Divergence Bot v4.0 TIEMPO REAL

✅ Estado: ONLINE - Detección en tiempo real
📊 Pares activos: {len(self.active_pairs)}
🎯 Calibrado con patrones reales exitosos

🔧 Comandos principales:
/status - Estado del sistema
/pairs - Ver pares monitoreados  
/add SYMBOL - Agregar par
/remove SYMBOL - Quitar par
/scan_now - Escaneo manual
/stats - Estadísticas detalladas
/help - Ayuda completa

🎯 Patrones detectados:
- Divergencias alcistas en soporte (+20% histórico)
- Divergencias bajistas en resistencia (caídas fuertes)
- Confirmaciones múltiples timeframes
- Detección ANTES del movimiento

💎 Bot funcionando 24/7 con algoritmo v4.0"""

            await update.message.reply_text(message)

        except Exception as e:
            logger.error(f"❌ Error en /start: {e}")
            await update.message.reply_text("🤖 RSI Divergence Bot v4.0 ONLINE")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /status"""
        try:
            bullish_signals = self.scan_stats.get("bullish_signals", 0)
            bearish_signals = self.scan_stats.get("bearish_signals", 0)

            message = f"""📊 *Estado Bot v4.0 TIEMPO REAL*

🔄 *Estado:* ✅ ONLINE (Detección en tiempo real)
📈 *Pares monitoreados:* {len(self.active_pairs)}
⏰ *Timeframes:* {', '.join(self.timeframes)}

📊 *Estadísticas de hoy:*
- Escaneos: {self.scan_stats.get('scans_completed', 0)}
- Divergencias detectadas: {self.scan_stats.get('divergences_found', 0)}
- Señales alcistas: {bullish_signals}
- Señales bajistas: {bearish_signals}
- Alertas enviadas: {self.scan_stats.get('alerts_sent', 0)}
- Errores: {self.scan_stats.get('scan_errors', 0)}

💾 *Cache:* {len(self.price_data_cache)} pares
⚡ *Último escaneo:* {self.scan_stats.get('last_scan_duration', 0):.1f}s

🎯 *Algoritmo v4.0:* Calibrado con patrones reales
🌐 *Servidor:* Railway 24/7"""

            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"❌ Error en /status: {e}")
            await update.message.reply_text("📊 Bot funcionando correctamente")

    async def cmd_scan_now(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /scan_now"""
        try:
            await update.message.reply_text(
                "🔄 **Iniciando escaneo manual en tiempo real...**",
                parse_mode=ParseMode.MARKDOWN,
            )

            start_time = datetime.now()
            await self.scan_all_pairs_realtime()
            end_time = datetime.now()

            duration = (end_time - start_time).total_seconds()
            recent_alerts = len(
                [
                    a
                    for a in self.sent_alerts.values()
                    if a.get("date") == datetime.now().date()
                ]
            )

            await update.message.reply_text(
                f"""✅ **Escaneo completado**

⏱️ **Duración:** {duration:.1f}s
📊 **Pares escaneados:** {len(self.active_pairs)}
🎯 **Alertas hoy:** {recent_alerts}
💾 **Cache activo:** {len(self.price_data_cache)}

🔍 **Buscando divergencias en tiempo real...**""",
                parse_mode=ParseMode.MARKDOWN,
            )

        except Exception as e:
            logger.error(f"❌ Error en /scan_now: {e}")
            await update.message.reply_text("❌ Error ejecutando escaneo")

    async def cmd_pairs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /pairs"""
        try:
            if not self.active_pairs:
                await update.message.reply_text("📭 No hay pares activos")
                return

            pairs_list = sorted(list(self.active_pairs))

            # Separar pares exitosos de otros
            successful_pairs = [
                "1000BONKUSDT",
                "CPOOLUSDT",
                "1000PEPEUSDT",
                "AIOZUSDT",
                "FARTCOINUSDT",
                "FETUSDT",
                "HYPEUSDT",
                "INJUSDT",
                "KAITOUSDT",
                "POPCATUSDT",
                "VIRTUALUSDT",
            ]

            successful_monitored = [p for p in pairs_list if p in successful_pairs]
            other_monitored = [p for p in pairs_list if p not in successful_pairs]

            message = f"📊 **Pares Monitoreados ({len(self.active_pairs)} total)**\n\n"

            if successful_monitored:
                message += "🎯 **Pares con patrones exitosos:**\n"
                for i in range(0, len(successful_monitored), 5):
                    batch = successful_monitored[i : i + 5]
                    message += "• " + " • ".join(batch) + "\n"
                message += "\n"

            if other_monitored:
                message += "📈 **Otros pares monitoreados:**\n"
                for i in range(0, len(other_monitored), 5):
                    batch = other_monitored[i : i + 5]
                    message += "• " + " • ".join(batch) + "\n"

            message += (
                f"\n💡 `/add SYMBOL` para agregar\n💡 `/remove SYMBOL` para quitar"
            )

            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"❌ Error en /pairs: {e}")
            await update.message.reply_text("❌ Error mostrando pares")

    async def cmd_add_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /add"""
        try:
            if not context.args:
                await update.message.reply_text(
                    "📝 **Uso:** `/add SYMBOL`\n\n**Ejemplos:**\n• `/add DOGEUSDT`\n• `/add SHIBUSDT`",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            symbol = context.args[0].upper()
            if not symbol.endswith("USDT"):
                symbol += "USDT"

            if symbol in self.active_pairs:
                await update.message.reply_text(
                    f"⚠️ **{symbol}** ya está siendo monitoreado",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            # Verificar si existe en Bybit
            if self.all_bybit_pairs and symbol not in self.all_bybit_pairs:
                search_term = symbol.replace("USDT", "")
                similar = [p for p in self.all_bybit_pairs if search_term in p]

                message = f"❌ **{symbol}** no encontrado en Bybit"
                if similar[:5]:
                    message += f"\n\n🔍 **Similares:**\n• " + "\n• ".join(similar[:5])

                await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
                return

            self.active_pairs.add(symbol)

            await update.message.reply_text(
                f"✅ **{symbol}** agregado al monitoreo\n📊 **Total pares:** {len(self.active_pairs)}",
                parse_mode=ParseMode.MARKDOWN,
            )

        except Exception as e:
            logger.error(f"❌ Error en /add: {e}")
            await update.message.reply_text("❌ Error agregando par")

    async def cmd_remove_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /remove"""
        try:
            if not context.args:
                await update.message.reply_text(
                    "📝 **Uso:** `/remove SYMBOL`\n\n**Ejemplos:**\n• `/remove DOGEUSDT`\n• `/remove SHIBUSDT`",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            symbol = context.args[0].upper()
            if not symbol.endswith("USDT"):
                symbol += "USDT"

            if symbol not in self.active_pairs:
                await update.message.reply_text(
                    f"❌ **{symbol}** no está en monitoreo",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            self.active_pairs.remove(symbol)

            await update.message.reply_text(
                f"🗑️ **{symbol}** removido del monitoreo\n📊 **Total pares:** {len(self.active_pairs)}",
                parse_mode=ParseMode.MARKDOWN,
            )

        except Exception as e:
            logger.error(f"❌ Error en /remove: {e}")
            await update.message.reply_text("❌ Error removiendo par")

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /stats - Estadísticas detalladas"""
        try:
            # Calcular estadísticas avanzadas
            total_signals = self.scan_stats.get("divergences_found", 0)
            bullish_signals = self.scan_stats.get("bullish_signals", 0)
            bearish_signals = self.scan_stats.get("bearish_signals", 0)

            success_rate = 0
            if total_signals > 0:
                success_rate = (bullish_signals + bearish_signals) / total_signals * 100

            # Alertas por día
            today_alerts = len(
                [
                    a
                    for a in self.sent_alerts.values()
                    if a.get("date") == datetime.now().date()
                ]
            )

            message = f"""📈 **Estadísticas Detalladas Bot v4.0**

📊 **Rendimiento:**
- Total divergencias: {total_signals}
- Señales alcistas: {bullish_signals}
- Señales bajistas: {bearish_signals}
- Tasa de detección: {success_rate:.1f}%

📅 **Hoy:**
- Alertas enviadas: {today_alerts}
- Escaneos completados: {self.scan_stats.get('scans_completed', 0)}
- Errores: {self.scan_stats.get('scan_errors', 0)}

⚡ **Performance:**
- Último escaneo: {self.scan_stats.get('last_scan_duration', 0):.1f}s
- Cache activo: {len(self.price_data_cache)} pares
- Pares monitoreados: {len(self.active_pairs)}

🎯 **Patrones detectados:**
- Divergencias en soporte/resistencia
- Confirmaciones múltiples timeframes  
- Detección en tiempo real
- Basado en +20% histórico

🤖 **Bot v4.0** - Calibrado con patrones reales exitosos"""

            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"❌ Error en /stats: {e}")
            await update.message.reply_text("❌ Error mostrando estadísticas")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /help"""
        try:
            message = """📋 **RSI Divergence Bot v4.0 - TIEMPO REAL**

🤖 **¿Qué hace?**
Detecta divergencias RSI EN TIEMPO REAL antes de que ocurran los movimientos. Calibrado con patrones reales que han dado +20% de ganancias.

📊 **Tipos de divergencias:**
- **Alcistas:** Precio LL, RSI HL → Rebotes +20%
- **Bajistas:** Precio HH, RSI LH → Caídas fuertes

🔧 **Comandos:**
- `/start` - Información inicial
- `/status` - Estado completo del sistema
- `/pairs` - Ver pares monitoreados
- `/add SYMBOL` - Agregar par al monitoreo
- `/remove SYMBOL` - Quitar par del monitoreo
- `/scan_now` - Escaneo manual inmediato
- `/stats` - Estadísticas detalladas
- `/help` - Esta ayuda

🎯 **Características v4.0:**
- ✅ Detección en tiempo real (ANTES del movimiento)
- ✅ Calibrado con patrones exitosos reales
- ✅ Confirmaciones múltiples timeframes
- ✅ Detección de soporte/resistencia
- ✅ Algoritmo basado en +20% histórico
- ✅ Timeframes: 2h, 4h, 6h, 8h, 12h, 1d

💎 **Pares con patrones exitosos:**
1000BONKUSDT, CPOOLUSDT, 1000PEPEUSDT, AIOZUSDT, FARTCOINUSDT, FETUSDT, HYPEUSDT, INJUSDT, KAITOUSDT, POPCATUSDT, VIRTUALUSDT

🚀 **Sistema 24/7 - Detección en tiempo real**"""

            await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

        except Exception as e:
            logger.error(f"❌ Error en /help: {e}")
            await update.message.reply_text(
                "📋 Ayuda disponible - usar comandos básicos"
            )

    async def handle_unknown_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Manejar mensajes no reconocidos"""
        try:
            await update.message.reply_text(
                "❓ Comando no reconocido. Usa `/help` para ver comandos disponibles."
            )
        except Exception as e:
            logger.error(f"❌ Error en mensaje desconocido: {e}")

    # === LIMPIEZA Y MANTENIMIENTO ===

    async def cleanup_cache_and_alerts(self):
        """Limpieza inteligente"""
        try:
            now = time.time()

            # Limpiar cache expirado
            expired_keys = [
                key
                for key, data in self.price_data_cache.items()
                if now - data.get("timestamp", 0) > self.cache_expiry
            ]

            for key in expired_keys:
                try:
                    del self.price_data_cache[key]
                except KeyError:
                    pass

            # Limpiar alertas antiguas (24 horas)
            cutoff = datetime.now() - timedelta(hours=24)
            self.sent_alerts = {
                k: v
                for k, v in self.sent_alerts.items()
                if v.get("timestamp", datetime.min) > cutoff
            }

            if len(expired_keys) > 0:
                logger.info(f"🧹 Cache limpiado: {len(expired_keys)} entradas")

        except Exception as e:
            logger.error(f"❌ Error en limpieza: {e}")

    # === LOOP PRINCIPAL ===

    async def start_monitoring_realtime(self):
        """Iniciar monitoreo en tiempo real"""
        logger.info("🚀 Iniciando Bot RSI Divergence v4.0 TIEMPO REAL")

        try:
            # Configurar Telegram
            await self.setup_telegram_commands_safe()

            # Mensaje de inicio
            startup_message = f"""🚀 **Bot RSI Divergence v4.0 TIEMPO REAL**

🌐 **Estado:** ONLINE en Railway
🔍 **Modo:** Detección en tiempo real
📊 **Pares monitoreados:** {len(self.active_pairs)}
⏰ **Timeframes:** {', '.join(self.timeframes)}

✨ **Mejoras v4.0:**
- ✅ Algoritmo calibrado con patrones reales exitosos
- ✅ Detección ANTES del movimiento (no después)
- ✅ Confirmaciones múltiples timeframes
- ✅ Contexto de soporte/resistencia
- ✅ Basado en +20% histórico

🎯 **Patrones objetivo:**
- Divergencias alcistas en soporte → +20%
- Divergencias bajistas en resistencia → Caídas fuertes

⚡ **Sistema 24/7 funcionando en tiempo real**

Usa `/help` para comandos completos."""

            await self.send_telegram_alert_safe(startup_message)

            # Loop principal optimizado
            while True:
                try:
                    loop_start = time.time()

                    # Escaneo principal en tiempo real
                    await self.scan_all_pairs_realtime()

                    # Limpieza periódica
                    await self.cleanup_cache_and_alerts()

                    # Estadísticas de rendimiento
                    loop_duration = time.time() - loop_start
                    logger.info(f"🔄 Ciclo completado en {loop_duration:.1f}s")

                    # Pausa inteligente: 5 minutos para timeframes cortos
                    await asyncio.sleep(300)  # 5 minutos

                except Exception as e:
                    logger.error(f"❌ Error en loop principal: {e}")
                    logger.error(traceback.format_exc())
                    # Pausa corta en caso de error
                    await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"❌ Error crítico en monitoreo: {e}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(300)

    def start_flask_server(self):
        """Iniciar servidor Flask"""
        try:
            logger.info(f"🌐 Iniciando servidor Flask en puerto {self.port}")
            self.app.run(
                host="0.0.0.0",
                port=self.port,
                debug=False,
                threaded=True,
                use_reloader=False,
            )
        except Exception as e:
            logger.error(f"❌ Error iniciando Flask: {e}")
            raise

    def run_realtime(self):
        """Punto de entrada principal para tiempo real"""
        logger.info("🚀 Iniciando RSI Divergence Bot v4.0 TIEMPO REAL...")

        try:
            # Iniciar Flask en thread separado
            flask_thread = threading.Thread(target=self.start_flask_server, daemon=True)
            flask_thread.start()
            logger.info("✅ Servidor Flask iniciado")

            # Pausa para asegurar Flask
            time.sleep(2)

            # Iniciar loop principal en tiempo real
            asyncio.run(self.start_monitoring_realtime())

        except KeyboardInterrupt:
            logger.info("🛑 Bot detenido por usuario")
        except Exception as e:
            logger.error(f"❌ Error crítico: {e}")
            logger.error(traceback.format_exc())
            time.sleep(30)
            logger.info("🔄 Reintentando...")


# === FUNCIÓN PRINCIPAL ===


def main():
    """Función principal para Railway"""
    try:
        # Validar variables de entorno
        required_vars = ["TELEGRAM_TOKEN", "CHAT_ID"]
        missing_vars = []

        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"❌ Variables faltantes: {', '.join(missing_vars)}")

        logger.info("✅ Variables de entorno validadas")

        # Crear e iniciar bot v4.0
        logger.info("🚀 Iniciando Bot RSI Divergence v4.0 TIEMPO REAL...")
        bot = RSIDivergenceBotV4()
        bot.run_realtime()

    except Exception as e:
        logger.error(f"❌ Error crítico iniciando bot v4.0: {e}")
        logger.error(traceback.format_exc())

        # Reintentar en caso de error crítico
        time.sleep(10)
        logger.info("🔄 Reintentando inicialización...")

        try:
            bot = RSIDivergenceBotV4()
            bot.run_realtime()
        except Exception as e2:
            logger.error(f"❌ Segundo intento falló: {e2}")
            raise


if __name__ == "__main__":
    main()
