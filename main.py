# Mejoras adicionales para el Bot RSI basadas en investigaciones 2024

import numpy as np
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import asyncio

class AdvancedRSIDivergenceBot(RSIDivergenceBot):
    """Extensión del bot con mejoras basadas en investigaciones 2024"""
    
    def __init__(self):
        super().__init__()
        
        # Nuevas características avanzadas
        self.ml_predictions = {}
        self.sentiment_scores = {}
        self.market_regime = 'normal'  # normal, volatile, trending
        self.adaptive_confidence = True
        self.multi_indicator_mode = True
        
        # Configuración de Machine Learning simple
        self.price_memory = defaultdict(list)  # Para ML básico
        self.trend_memory = defaultdict(list)
        
        # Configuración de sentiment
        self.sentiment_sources = {
            'fear_greed_index': True,
            'news_sentiment': False,  # Requiere API key
            'social_sentiment': False  # Requiere API key
        }

    # MEJORA 1: DETECCIÓN DE RÉGIMEN DE MERCADO
    def detect_market_regime(self, symbol: str, data: pd.DataFrame) -> str:
        """Detectar régimen actual del mercado para ajustar estrategia"""
        if len(data) < 50:
            return 'normal'
        
        closes = data['close'].values
        volumes = data['volume'].values if 'volume' in data.columns else None
        
        # Calcular volatilidad
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * 100
        
        # Calcular tendencia
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
        trend_strength = abs(sma_20 - sma_50) / sma_50 * 100
        
        # Análisis de volumen
        volume_spike = False
        if volumes is not None and len(volumes) >= 20:
            recent_vol = np.mean(volumes[-5:])
            avg_vol = np.mean(volumes[-20:-5])
            volume_spike = recent_vol > avg_vol * 2.0
        
        # Determinar régimen
        if volatility > 3.0 or volume_spike:
            regime = 'volatile'
        elif trend_strength > 2.0:
            regime = 'trending'
        else:
            regime = 'normal'
        
        logger.debug(f"📊 {symbol} Market Regime: {regime} (Vol: {volatility:.1f}%, Trend: {trend_strength:.1f}%)")
        return regime

    # MEJORA 2: MULTI-INDICATOR CONFIRMATION
    def calculate_macd_divergence(self, data: pd.DataFrame) -> Optional[dict]:
        """Calcular divergencia MACD como confirmación adicional"""
        if len(data) < 50:
            return None
        
        closes = data['close'].values
        
        # MACD simple
        ema_12 = self.calculate_ema(closes, 12)
        ema_26 = self.calculate_ema(closes, 26)
        macd_line = ema_12 - ema_26
        signal_line = self.calculate_ema(macd_line, 9)
        
        # Encontrar picos en MACD
        macd_peaks, macd_troughs = self.find_peaks_and_troughs(macd_line, min_distance=3)
        
        if len(macd_peaks) >= 2:
            recent_peaks = macd_peaks[-2:]
            macd_val1, macd_val2 = macd_line[recent_peaks[0]], macd_line[recent_peaks[1]]
            
            return {
                'type': 'bearish' if macd_val2 < macd_val1 else 'bullish',
                'strength': abs(macd_val2 - macd_val1),
                'confirmation': True if abs(macd_val2 - macd_val1) > 0.001 else False
            }
        
        return None

    def calculate_ema(self, data: np.array, period: int) -> np.array:
        """Calcular EMA para MACD"""
        ema = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema

    # MEJORA 3: MACHINE LEARNING BÁSICO (Predicción de Tendencias)
    def update_ml_memory(self, symbol: str, data: pd.DataFrame):
        """Actualizar memoria para ML básico"""
        if len(data) < 20:
            return
        
        closes = data['close'].values
        rsi = self.calculate_rsi(closes)
        
        # Características simples para ML
        price_change = (closes[-1] - closes[-5]) / closes[-5] * 100
        rsi_level = rsi[-1]
        volume_trend = 1 if len(data.columns) > 4 else 0
        
        # Guardar en memoria (últimos 100 puntos)
        features = [price_change, rsi_level, volume_trend]
        self.price_memory[symbol].append(features)
        
        # Mantener solo últimos 100 puntos
        if len(self.price_memory[symbol]) > 100:
            self.price_memory[symbol] = self.price_memory[symbol][-100:]

    def predict_trend_direction(self, symbol: str) -> Optional[dict]:
        """Predicción básica de tendencia usando datos históricos"""
        if symbol not in self.price_memory or len(self.price_memory[symbol]) < 20:
            return None
        
        memory = self.price_memory[symbol]
        recent_data = memory[-20:]
        
        # Análisis simple de patrones
        price_changes = [x[0] for x in recent_data]
        rsi_levels = [x[1] for x in recent_data]
        
        # Tendencia de precio
        price_trend = np.mean(price_changes[-5:]) - np.mean(price_changes[-15:-5])
        
        # Momentum RSI
        rsi_momentum = np.mean(rsi_levels[-5:]) - np.mean(rsi_levels[-15:-5])
        
        # Predicción simple
        if price_trend > 1 and rsi_momentum < -5:
            prediction = 'bearish_divergence_likely'
            confidence = min(0.8, abs(price_trend) * 0.1 + abs(rsi_momentum) * 0.05)
        elif price_trend < -1 and rsi_momentum > 5:
            prediction = 'bullish_divergence_likely'
            confidence = min(0.8, abs(price_trend) * 0.1 + abs(rsi_momentum) * 0.05)
        else:
            prediction = 'neutral'
            confidence = 0.5
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'price_trend': price_trend,
            'rsi_momentum': rsi_momentum
        }

    # MEJORA 4: SENTIMENT ANALYSIS BÁSICO
    async def get_fear_greed_index(self) -> Optional[dict]:
        """Obtener índice de Fear & Greed como indicador de sentiment"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                fng = data['data'][0]
                return {
                    'value': int(fng['value']),
                    'classification': fng['value_classification'],
                    'timestamp': fng['timestamp']
                }
        except Exception as e:
            logger.error(f"Error obteniendo Fear & Greed Index: {e}")
        
        return None

    def calculate_sentiment_adjustment(self, base_confidence: float, symbol: str) -> float:
        """Ajustar confianza basado en sentiment del mercado"""
        if symbol not in self.sentiment_scores:
            return base_confidence
        
        sentiment = self.sentiment_scores[symbol]
        
        # Ajuste basado en Fear & Greed
        if 'fear_greed' in sentiment:
            fg_value = sentiment['fear_greed']['value']
            
            # Extreme Fear (0-25) o Extreme Greed (75-100) pueden aumentar divergencias
            if fg_value <= 25 or fg_value >= 75:
                adjustment = 1.05  # 5% bonus
            elif 25 < fg_value <= 45 or 55 <= fg_value < 75:
                adjustment = 1.02  # 2% bonus
            else:
                adjustment = 1.0  # Sin ajuste
            
            return min(98, base_confidence * adjustment)
        
        return base_confidence

    # MEJORA 5: DETECCIÓN MEJORADA CON TODOS LOS FACTORES
    async def detect_divergence_advanced(self, price_data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        """Detección avanzada que integra todas las mejoras"""
        
        # Detección base
        base_signal = self.detect_divergence_smart(price_data, symbol, timeframe)
        if not base_signal:
            return None
        
        # MEJORA 1: Ajustar por régimen de mercado
        market_regime = self.detect_market_regime(symbol, price_data)
        self.market_regime = market_regime
        
        # MEJORA 2: Confirmación con MACD
        macd_div = self.calculate_macd_divergence(price_data)
        macd_confirmation = False
        if macd_div and macd_div['type'] == base_signal.type and macd_div['confirmation']:
            macd_confirmation = True
            base_signal.confidence += 3  # Bonus por confirmación MACD
        
        # MEJORA 3: Predicción ML
        self.update_ml_memory(symbol, price_data)
        ml_prediction = self.predict_trend_direction(symbol)
        ml_confirmation = False
        if ml_prediction:
            expected_div = f"{base_signal.type}_divergence_likely"
            if ml_prediction['prediction'] == expected_div and ml_prediction['confidence'] > 0.6:
                ml_confirmation = True
                base_signal.confidence += ml_prediction['confidence'] * 5  # Bonus ML
        
        # MEJORA 4: Ajuste por sentiment
        if symbol in self.sentiment_scores:
            base_signal.confidence = self.calculate_sentiment_adjustment(base_signal.confidence, symbol)
        
        # MEJORA 5: Ajuste por régimen de mercado
        regime_multiplier = {
            'volatile': 0.95,    # Reducir confianza en mercados volátiles
            'trending': 1.05,    # Aumentar en mercados con tendencia
            'normal': 1.0        # Sin cambio
        }
        
        base_signal.confidence *= regime_multiplier.get(market_regime, 1.0)
        base_signal.confidence = min(98, base_signal.confidence)
        
        # Logging avanzado
        logger.info(f"🧠 {symbol} {timeframe} Advanced Detection:")
        logger.info(f"   Base: {base_signal.type} {base_signal.confidence:.1f}%")
        logger.info(f"   Regime: {market_regime}")
        logger.info(f"   MACD: {'✅' if macd_confirmation else '❌'}")
        logger.info(f"   ML: {'✅' if ml_confirmation else '❌'}")
        
        return base_signal

    # MEJORA 6: MONITOREO EN TIEMPO REAL MEJORADO
    async def enhanced_monitoring_loop(self):
        """Loop de monitoreo mejorado con actualizaciones más frecuentes"""
        
        while True:
            try:
                # Actualizar sentiment cada hora
                if datetime.now().minute == 0:
                    await self.update_market_sentiment()
                
                # Escaneo normal cada 10 minutos
                if datetime.now().minute % 10 == 0:
                    await self.scan_all_pairs_advanced()
                
                # Escaneo rápido de pares trending cada 5 minutos
                elif datetime.now().minute % 5 == 0:
                    trending_pairs = ['BTCUSDT', 'ETHUSDT', 'HYPEUSDT', 'SOLUSDT', 'PEPEUSDT']
                    for pair in trending_pairs:
                        if pair in self.active_pairs:
                            await self.scan_single_pair_advanced(pair)
                            await asyncio.sleep(0.2)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Error en enhanced monitoring loop: {e}")
                await asyncio.sleep(60)

    async def update_market_sentiment(self):
        """Actualizar sentiment del mercado"""
        try:
            # Fear & Greed Index
            fg_data = await self.get_fear_greed_index()
            if fg_data:
                # Aplicar a todos los pares principales
                major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'HYPEUSDT']
                for pair in major_pairs:
                    if pair not in self.sentiment_scores:
                        self.sentiment_scores[pair] = {}
                    self.sentiment_scores[pair]['fear_greed'] = fg_data
                
                logger.info(f"📊 Market Sentiment: {fg_data['classification']} ({fg_data['value']})")
        
        except Exception as e:
            logger.error(f"❌ Error actualizando sentiment: {e}")

    async def scan_single_pair_advanced(self, symbol: str):
        """Escaneo avanzado con todas las mejoras"""
        try:
            for timeframe in self.timeframes:
                await asyncio.sleep(0.1)
                
                data = await self.get_ohlcv_data(symbol, timeframe, limit=120)
                if data.empty:
                    continue
                
                # Usar detección avanzada
                signal = await self.detect_divergence_advanced(data, symbol, timeframe)
                
                if not signal:
                    continue
                
                # Verificar duplicados
                duplicate_window_hours = {'4h': 8, '6h': 12, '8h': 16, '12h': 24, '1d': 48}
                if self.is_duplicate_alert_smart(signal, duplicate_window_hours.get(timeframe, 12)):
                    continue
                
                # Validaciones finales
                signal.volume_spike = self.check_volume_spike_enhanced(data, timeframe)
                
                # Registrar con metadatos avanzados
                alert_key = f"{symbol}_{timeframe}_{signal.type}"
                self.sent_alerts[alert_key] = {
                    'timestamp': datetime.now(),
                    'confidence': signal.confidence,
                    'date': datetime.now().date(),
                    'timeframe': timeframe,
                    'market_regime': self.market_regime,
                    'advanced_detection': True
                }
                
                # Determinar tipo de alerta
                config = self.get_timeframe_config(timeframe)
                if signal.confidence >= config['min_confidence']:
                    alert_type = "final" if signal.confidence >= 90 else "confirmation"
                    message = await self.format_alert_message_advanced(signal, alert_type, timeframe)
                    await self.send_telegram_alert(message)
                    
                    self.scan_stats['divergences_found'] += 1
                    self.scan_stats[f'{timeframe}_divergences'] = self.scan_stats.get(f'{timeframe}_divergences', 0) + 1
                
        except Exception as e:
            logger.error(f"❌ Error en scan avanzado {symbol}: {e}")

    async def scan_all_pairs_advanced(self):
        """Escaneo de todos los pares con sistema avanzado"""
        scan_start = datetime.now()
        logger.info(f"🧠 Iniciando escaneo avanzado de {len(self.active_pairs)} pares...")
        
        # Procesar en batches
        batch_size = 8  # Menor para dar más tiempo de procesamiento
        pairs_list = list(self.active_pairs)
        
        for i in range(0, len(pairs_list), batch_size):
            batch = pairs_list[i:i + batch_size]
            tasks = [self.scan_single_pair_advanced(symbol) for symbol in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(1.5)  # Pausa más larga entre batches
        
        scan_duration = (datetime.now() - scan_start).total_seconds()
        self.scan_stats['scans_completed'] += 1
        self.scan_stats['last_scan_duration'] = scan_duration
        
        logger.info(f"✅ Escaneo avanzado completado en {scan_duration:.1f}s")

    async def format_alert_message_advanced(self, signal: DivergenceSignal, alert_type: str, timeframe: str) -> str:
        """Mensaje de alerta con información avanzada"""
        
        base_message = await self.format_alert_message_enhanced(signal, alert_type, timeframe)
        
        # Añadir información avanzada
        advanced_info = f"\n\n🧠 **Análisis Avanzado:**"
        advanced_info += f"\n• Régimen: {self.market_regime.title()}"
        
        if signal.symbol in self.sentiment_scores and 'fear_greed' in self.sentiment_scores[signal.symbol]:
            fg = self.sentiment_scores[signal.symbol]['fear_greed']
            advanced_info += f"\n• Sentiment: {fg['classification']} ({fg['value']})"
        
        ml_prediction = self.predict_trend_direction(signal.symbol)
        if ml_prediction:
            advanced_info += f"\n• ML Predicción: {ml_prediction['confidence']:.0%}"
        
        advanced_info += f"\n🤖 **Sistema:** Advanced AI v2.1"
        
        return base_message + advanced_info

# Comandos adicionales para funciones avanzadas
async def cmd_market_sentiment(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostrar sentiment actual del mercado"""
    fg_data = await self.get_fear_greed_index()
    
    if fg_data:
        value = fg_data['value']
        classification = fg_data['classification']
        
        emoji = {
            'Extreme Fear': '😰',
            'Fear': '😟', 
            'Neutral': '😐',
            'Greed': '😊',
            'Extreme Greed': '🤑'
        }.get(classification, '📊')
        
        message = f"📊 **Market Sentiment Analysis**\n\n"
        message += f"{emoji} **Fear & Greed Index:** {value}/100\n"
        message += f"📈 **Classification:** {classification}\n\n"
        
        if value <= 25:
            message += "🚨 **Extreme Fear** - Posibles oportunidades de compra"
        elif value >= 75:
            message += "⚠️ **Extreme Greed** - Cuidado con posibles correcciones"
        else:
            message += "✅ **Sentiment balanceado** - Condiciones normales"
            
        message += f"\n\n🤖 **Impact:** El sentiment extremo puede aumentar probabilidad de divergencias"
        
    else:
        message = "❌ No se pudo obtener datos de sentiment"
    
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

async def cmd_ml_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Análisis ML de un par específico"""
    if len(context.args) < 1:
        await update.message.reply_text("📝 **Uso:** `/ml_analysis SYMBOL`")
        return
    
    symbol = context.args[0].upper()
    prediction = self.predict_trend_direction(symbol)
    
    if prediction:
        message = f"🧠 **Análisis ML: {symbol}**\n\n"
        message += f"🎯 **Predicción:** {prediction['prediction'].replace('_', ' ').title()}\n"
        message += f"📊 **Confianza:** {prediction['confidence']:.0%}\n"
        message += f"💹 **Trend Precio:** {prediction['price_trend']:+.1f}%\n"
        message += f"📈 **Momentum RSI:** {prediction['rsi_momentum']:+.1f}\n\n"
        
        if prediction['confidence'] > 0.7:
            message += "🔥 **Alta probabilidad** de divergencia"
        elif prediction['confidence'] > 0.5:
            message += "⚡ **Probabilidad moderada** de divergencia"
        else:
            message += "⚪ **Baja probabilidad** de divergencia"
    else:
        message = f"❌ No hay suficientes datos ML para {symbol}"
    
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
