"""Comando /status mejorado"""
        uptime = datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        message        
        return min(8, bonus)  # Cap máximo de 8 puntos bonus

    def detect_bearish_divergence_validated(self, highs: np.array, rsi: np.array, 
                                          price_peaks: List[int], rsi_peaks: List[int], 
                                          current_price: float, config: dict, 
                                          symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        """Detección de divergencia bajista con múltiples validaciones"""
        if len(price_peaks) < 2 or len(rsi_peaks) < 2:
            return None
        
        best_signal = None
        best_confidence = 0
        
        # Buscar en múltiples combinaciones pero con validaciones estrictas
        for i in range(max(1, len(price_peaks) - 3), len(price_peaks)):
            for j in range(max(1, len(rsi_peaks) - 3), len(rsi_peaks)):
                if i < 1 or j < 1:
                    continue
                    
                price_peak1_idx = price_peaks[i-1]
                price_peak2_idx = price_peaks[i] if i < len(price_peaks) else price_peaks[-1]
                rsi_peak1_idx = rsi_peaks[j-1]
                rsi_peak2_idx = rsi_peaks[j] if j < len(rsi_peaks) else rsi_peaks[-1]
                
                # VALIDACIÓN 1: Alineación temporal
                time_diff = abs((price_peak2_idx - price_peak1_idx) - (rsi_peak2_idx - rsi_peak1_idx))
                if time_diff > config['max_time_diff']:
                    continue
                
                price_high1 = highs[price_peak1_idx]
                price_high2 = highs[price_peak2_idx]
                rsi_high1 = rsi[rsi_peak1_idx]
                rsi_high2 = rsi[rsi_peak2_idx]
                
                # VALIDACIÓN 2: Cambios mínimos significativos
                price_change = (price_high2 - price_high1) / price_high1 * 100
                rsi_change = rsi_high1 - rsi_high2
                
                if price_change < config['min_price_change']:
                    continue
                if rsi_change < config['min_rsi_change']:
                    continue
                
                # VALIDACIÓN 3: Niveles de RSI apropiados
                if max(rsi_high1, rsi_high2) < config['rsi_high_threshold']:
                    continue
                
                # VALIDACIÓN 4: Divergencia clara
                price_making_higher_high = price_high2 > price_high1
                rsi_making_lower_high = rsi_high2 < rsi_high1
                
                if not (price_making_higher_high and rsi_making_lower_high):
                    continue
                
                # VALIDACIÓN 5: Fuerza de la divergencia
                divergence_strength = self.calculate_divergence_strength(
                    price_change, rsi_change, time_diff, config
                )
                
                if divergence_strength < 0.6:  # Umbral de fuerza mínima
                    continue
                
                # CÁLCULO DE CONFIANZA SOFISTICADO
                confidence = self.calculate_advanced_confidence(
                    price_change, rsi_change, time_diff, divergence_strength, 
                    rsi_high1, rsi_high2, config, 'bearish'
                )
                
                # VALIDACIÓN 6: Confirmación adicional con contexto
                context_bonus = self.get_context_validation_bonus(
                    highs, rsi, price_peak1_idx, price_peak2_idx, rsi_peak1_idx, rsi_peak2_idx
                )
                
                final_confidence = min(98, confidence + context_bonus)
                
                logger.debug(f"🔍 {symbol} {timeframe} Bearish: P1={price_high1:.1f} P2={price_high2:.1f} "
                            f"R1={rsi_high1:.1f} R2={rsi_high2:.1f} Conf={final_confidence:.1f}%")
                
                if final_confidence >= config['min_confidence'] and final_confidence > best_confidence:
                    best_confidence = final_confidence
                    best_signal = DivergenceSignal(
                        symbol=symbol,
                        timeframe=timeframe,
                        type='bearish',
                        confidence=final_confidence,
                        price_level=current_price,
                        resistance_level=price_high2,
                        volume_spike=False,  # Se calculará después
                        rsi_value=rsi[-1]
                    )
        
        return best_signal

    def detect_bullish_divergence_validated(self, lows: np.array, rsi: np.array, 
                                          price_troughs: List[int], rsi_troughs: List[int], 
                                          current_price: float, config: dict, 
                                          symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        """Detección de divergencia alcista con múltiples validaciones"""
        if len(price_troughs) < 2 or len(rsi_troughs) < 2:
            return None
        
        best_signal = None
        best_confidence = 0
        
        for i in range(max(1, len(price_troughs) - 3), len(price_troughs)):
            for j in range(max(1, len(rsi_troughs) - 3), len(rsi_troughs)):
                if i < 1 or j < 1:
                    continue
                    
                price_trough1_idx = price_troughs[i-1]
                price_trough2_idx = price_troughs[i] if i < len(price_troughs) else price_troughs[-1]
                rsi_trough1_idx = rsi_troughs[j-1]
                rsi_trough2_idx = rsi_troughs[j] if j < len(rsi_troughs) else rsi_troughs[-1]
                
                # Mismas validaciones pero para alcista
                time_diff = abs((price_trough2_idx - price_trough1_idx) - (rsi_trough2_idx - rsi_trough1_idx))
                if time_diff > config['max_time_diff']:
                    continue
                
                price_low1 = lows[price_trough1_idx]
                price_low2 = lows[price_trough2_idx]
                rsi_low1 = rsi[rsi_trough1_idx]
                rsi_low2 = rsi[rsi_trough2_idx]
                
                price_change = abs(price_low2 - price_low1) / price_low1 * 100
                rsi_change = rsi_low2 - rsi_low1
                
                if price_change < config['min_price_change']:
                    continue
                if rsi_change < config['min_rsi_change']:
                    continue
                if min(rsi_low1, rsi_low2) > config['rsi_low_threshold']:
                    continue
                
                price_making_lower_low = price_low2 < price_low1
                rsi_making_higher_low = rsi_low2 > rsi_low1
                
                if not (price_making_lower_low and rsi_making_higher_low):
                    continue
                
                divergence_strength = self.calculate_divergence_strength(
                    price_change, rsi_change, time_diff, config
                )
                
                if divergence_strength < 0.6:
                    continue
                
                confidence = self.calculate_advanced_confidence(
                    price_change, rsi_change, time_diff, divergence_strength, 
                    rsi_low1, rsi_low2, config, 'bullish'
                )
                
                context_bonus = self.get_context_validation_bonus(
                    lows, rsi, price_trough1_idx, price_trough2_idx, rsi_trough1_idx, rsi_trough2_idx
                )
                
                final_confidence = min(98, confidence + context_bonus)
                
                if final_confidence >= config['min_confidence'] and final_confidence > best_confidence:
                    best_confidence = final_confidence
                    best_signal = DivergenceSignal(
                        symbol=symbol,
                        timeframe=timeframe,
                        type='bullish',
                        confidence=final_confidence,
                        price_level=current_price,
                        resistance_level=None,
                        volume_spike=False,
                        rsi_value=rsi[-1]
                    )
        
        return best_signal

    def detect_divergence_smart(self, price_data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[DivergenceSignal]:
        """
        Detección inteligente que ajusta parámetros según el contexto
        - Mantiene alta calidad para evitar falsas señales
        - Ajusta sensibilidad según timeframe
        - Usa múltiples filtros de validación
        """
        if len(price_data) < 50:
            return None
            
        closes = price_data['close'].values
        highs = price_data['high'].values
        lows = price_data['low'].values
        rsi = self.calculate_rsi(closes)
        
        if len(rsi) < 30 or np.isnan(rsi[-1]):
            return None
        
        # CONFIGURACIÓN INTELIGENTE POR TIMEFRAME
        config = self.get_timeframe_config(timeframe)
        
        # Encontrar picos y valles con parámetros específicos
        price_peaks, price_troughs = self.find_peaks_and_troughs_smart(
            highs, lows, config['min_distance'], config['lookback_period']
        )
        rsi_peaks, rsi_troughs = self.find_peaks_and_troughs(rsi, config['min_distance'])
        
        # Detectar divergencia bajista con validaciones múltiples
        bearish_signal = self.detect_bearish_divergence_validated(
            highs, rsi, price_peaks, rsi_peaks, closes[-1], config, symbol, timeframe
        )
        if bearish_signal:
            return bearish_signal
        
        # Detectar divergencia alcista con validaciones múltiples
        bullish_signal = self.detect_bullish_divergence_validated(
            lows, rsi, price_troughs, rsi_troughs, closes[-1], config, symbol, timeframe
        )
        if bullish_signal:
            return bullish_signal
        
        return None

    def check_volume_spike_enhanced(self, price_data: pd.DataFrame, timeframe: str) -> bool:
        """Verificar spike de volumen adaptado por timeframe"""
        if len(price_data) < 20 or 'volume' not in price_data.columns:
            return False
        
        # Períodos de comparación por timeframe
        comparison_periods = {'4h': 20, '6h': 18, '8h': 16, '12h': 14, '1d': 12}
        lookback = comparison_periods.get(timeframe, 20)
        
        if len(price_data) < lookback:
            return False
            
        recent_volume = price_data['volume'].iloc[-3:].mean()
        avg_volume = price_data['volume'].iloc[-lookback:-3].mean()
        
        # Umbrales por timeframe (timeframes más largos necesitan spikes más grandes)
        spike_thresholds = {'4h': 1.8, '6h': 1.9, '8h': 2.0, '12h': 2.1, '1d': 2.2}
        threshold = spike_thresholds.get(timeframe, 1.8)
        
        return recent_volume > avg_volume * threshold

    def check_volume_spike(self, price_data: pd.DataFrame) -> bool:
        """Verificar spike de volumen mejorado (método legacy)"""
        if len(price_data) < 20 or 'volume' not in price_data.columns:
            return False
            
        recent_volume = price_data['volume'].iloc[-3:].mean()
        avg_volume = price_data['volume'].iloc[-20:-3].mean()
        
        return recent_volume > avg_volume * 1.8  # Umbral más estricto

    def is_duplicate_alert_smart(self, signal: DivergenceSignal, window_hours: int) -> bool:
        """Verificar alertas duplicadas con ventana temporal específica"""
        alert_key = f"{signal.symbol}_{signal.timeframe}_{signal.type}"
        
        if alert_key in self.sent_alerts:
            last_alert = self.sent_alerts[alert_key]
            time_diff = datetime.now() - last_alert['timestamp']
            
            # Ventana temporal adaptativa por timeframe
            if time_diff.total_seconds() < window_hours * 3600:
                # Permitir actualización si la nueva señal es significativamente más fuerte
                confidence_improvement = signal.confidence - last_alert.get('confidence', 0)
                if confidence_improvement < 5:  # Debe ser al menos 5 puntos mejor
                    return True
                
        return False

    def is_duplicate_alert(self, signal: DivergenceSignal) -> bool:
        """Verificar alertas duplicadas mejorado (método legacy)"""
        alert_key = f"{signal.symbol}_{signal.timeframe}_{signal.type}"
        
        if alert_key in self.sent_alerts:
            last_alert = self.sent_alerts[alert_key]
            time_diff = datetime.now() - last_alert['timestamp']
            
            # No enviar la misma alerta en menos de 2 horas
            if time_diff.total_seconds() < 7200:
                return True
                
        return False

    async def format_alert_message_enhanced(self, signal: DivergenceSignal, alert_type: str, timeframe: str) -> str:
        """Formatear mensaje con información específica del timeframe"""
        confidence_emoji = '🔥' if signal.confidence >= 92 else '⚡' if signal.confidence >= 87 else '🟠'
        type_emoji = '📉' if signal.type == 'bearish' else '📈'
        volume_emoji = '📈' if signal.volume_spike else '📊'
        
        # Emoji específico por timeframe
        tf_emoji = {'4h': '⏰', '6h': '🕕', '8h': '🕗', '12h': '🕚', '1d': '📅'}.get(timeframe, '⏰')
        
        if alert_type == "confirmation":
            message = f"""{confidence_emoji} **DIVERGENCIA DETECTADA** {confidence_emoji}

📌 **Par:** `{signal.symbol}`
💰 **Precio:** {signal.price_level:.6f}
{type_emoji} **Tipo:** Divergencia {signal.type.upper()}
{volume_emoji} **Volumen:** {'Spike detectado ✅' if signal.volume_spike else 'Normal'}
📊 **Confianza:** {signal.confidence:.0f}%
{tf_emoji} **Timeframe:** {timeframe.upper()}
🔢 **RSI:** {signal.rsi_value:.1f}
🤖 **Sistema:** Smart Detection v2.1

⏰ {signal.timestamp.strftime('%H:%M:%S')}
🎯 **Calidad:** {'Premium' if signal.confidence >= 90 else 'Alta' if signal.confidence >= 85 else 'Buena'}"""
        else:
            message = f"""{confidence_emoji} **SEÑAL CONFIRMADA** {confidence_emoji}

📌 **Par:** `{signal.symbol}`
💰 **Precio:** {signal.price_level:.6f}
{type_emoji} **Tipo:** Divergencia {signal.type.upper()} **CONFIRMADA**
{volume_emoji} **Volumen:** {'Spike + divergencia ✅' if signal.volume_spike else 'Divergencia confirmada'}
{tf_emoji} **Timeframe:** {timeframe.upper()}
🎯 **RSI:** {signal.rsi_value:.1f}
🔢 **Confianza:** {signal.confidence:.0f}%
🤖 **Sistema:** Smart Detection v2.1

🚀 **ALTA PROBABILIDAD - SEÑAL PREMIUM**
⏰ {signal.timestamp.strftime('%H:%M:%S')}"""

        return message

    async def format_alert_message(self, signal: DivergenceSignal, alert_type: str = "confirmation") -> str:
        """Formatear mensaje de alerta optimizado (método legacy)"""
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
🤖 **Fuente:** Railway Bot v2.1

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
🤖 **Fuente:** Railway Bot v2.1

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

    async def scan_single_pair_smart(self, symbol: str):
        """Escanear par con sistema inteligente calibrado para todos los timeframes"""
        try:
            for timeframe in self.timeframes:
                await asyncio.sleep(0.1)
                
                # Obtener más datos para timeframes largos para mejor análisis
                limit_map = {'4h': 120, '6h': 100, '8h': 90, '12h': 80, '1d': 70}
                limit = limit_map.get(timeframe, 100)
                
                data = await self.get_ohlcv_data(symbol, timeframe, limit=limit)
                if data.empty:
                    logger.debug(f"❌ {symbol} {timeframe}: No hay datos")
                    continue
                    
                data.index.name = symbol
                
                # Usar detección inteligente
                signal = self.detect_divergence_smart(data, symbol, timeframe)
                
                if not signal:
                    logger.debug(f"⚪ {symbol} {timeframe}: Sin divergencias")
                    continue
                    
                # Verificar duplicados con ventana temporal específica por timeframe
                duplicate_window_hours = {'4h': 8, '6h': 12, '8h': 16, '12h': 24, '1d': 48}
                if self.is_duplicate_alert_smart(signal, duplicate_window_hours.get(timeframe, 12)):
                    logger.debug(f"⚠️ Duplicado: {symbol} {timeframe} {signal.type}")
                    continue
                
                # Validación final de volumen (con peso por timeframe)
                signal.volume_spike = self.check_volume_spike_enhanced(data, timeframe)
                config = self.get_timeframe_config(timeframe)
                
                if signal.volume_spike:
                    volume_bonus = config.get('volume_weight', 0.1) * 20  # Hasta 2-3 puntos
                    signal.confidence = min(97, signal.confidence + volume_bonus)
                
                # Registrar alerta con metadatos mejorados
                alert_key = f"{symbol}_{timeframe}_{signal.type}"
                self.sent_alerts[alert_key] = {
                    'timestamp': datetime.now(),
                    'confidence': signal.confidence,
                    'date': datetime.now().date(),
                    'timeframe': timeframe,
                    'price_level': signal.price_level,
                    'rsi_level': signal.rsi_value
                }
                
                logger.info(f"🎯 DIVERGENCIA VALIDADA: {symbol} {timeframe} {signal.type} "
                           f"Conf:{signal.confidence:.1f}% RSI:{signal.rsi_value:.1f}")
                
                # Determinar tipo de alerta basado en confianza y timeframe
                if signal.confidence >= 90:
                    alert_type = "final"
                elif signal.confidence >= config['min_confidence']:
                    alert_type = "confirmation"
                else:
                    continue  # No enviar si está por debajo del umbral
                
                # Enviar alerta con contexto del timeframe
                message = await self.format_alert_message_enhanced(signal, alert_type, timeframe)
                await self.send_telegram_alert(message)
                    
                self.scan_stats['divergences_found'] += 1
                self.scan_stats[f'{timeframe}_divergences'] = self.scan_stats.get(f'{timeframe}_divergences', 0) + 1
                    
        except Exception as e:
            logger.error(f"❌ Error escaneando {symbol}: {e}")
            self.scan_stats['scan_errors'] += 1

    async def scan_single_pair(self, symbol: str):
        """Escanear un par con manejo de errores mejorado (método legacy)"""
        return await self.scan_single_pair_smart(symbol)

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
            tasks = [self.scan_single_pair_smart(symbol) for symbol in batch]
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
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'PEPEUSDT', 'WIFUSDT', 'HYPEUSDT']

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
            
            # Comandos de debug y análisis mejorados
            self.telegram_app.add_handler(CommandHandler("add_trending", self.cmd_add_trending))
            self.telegram_app.add_handler(CommandHandler("debug_pair", self.cmd_debug_pair_generic))
            self.telegram_app.add_handler(CommandHandler("test_pair", self.cmd_test_pair))
            self.telegram_app.add_handler(CommandHandler("stats_by_timeframe", self.cmd_stats_by_timeframe))
            self.telegram_app.add_handler(CommandHandler("analyze", self.cmd_analyze_pair))
            
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
        message = f"""🚀 **Bot RSI Divergence v2.1**

¡Hola! Soy tu bot especializado en detectar divergencias RSI.

📊 **Funciones principales:**
• Monitoreo 24/7 de divergencias
• Múltiples timeframes (4h, 6h, 8h, 12h, 1d)
• Alertas automáticas en tiempo real
• Webhook para TradingView
• Sistema inteligente por timeframe

📋 **Comandos disponibles:**
/status - Estado del bot
/stats - Estadísticas detalladas
/stats_by_timeframe - Stats por TF
/list_pairs - Ver pares monitoreados
/add_pair SYMBOL - Añadir par
/scan_now - Escaneo manual

🤖 **Estado actual:** ONLINE
📈 **Pares activos:** {len(self.active_pairs)}
🎯 **Incluye HYPE y tokens trending**

¡Listo para detectar oportunidades!"""

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /help"""
        message = f"""📋 **Ayuda - Bot RSI Divergence v2.1**

🎯 **¿Qué hace este bot?**
Detecta divergencias RSI automáticamente en múltiples pares de criptomonedas y timeframes.

📊 **Comandos principales:**
• `/status` - Estado completo del bot
• `/stats` - Estadísticas de rendimiento
• `/stats_by_timeframe` - Stats por timeframe
• `/list_pairs` - Ver todos los pares activos
• `/scan_now` - Forzar escaneo manual

🔧 **Gestión de pares:**
• `/add_pair BTCUSDT` - Añadir par específico
• `/remove_pair ETHUSDT` - Remover par
• `/search_pair BTC` - Buscar pares disponibles
• `/add_trending` - Añadir tokens trending

🧪 **Testing y Debug:**
• `/webhook_test` - Probar webhook TradingView
• `/debug_pair SYMBOL [TF]` - Debug específico
• `/test_pair SYMBOL [TF]` - Test par específico
• `/analyze SYMBOL` - Análisis completo

❓ **¿Qué son las divergencias?**
Cuando el precio y el RSI se mueven en direcciones opuestas, indicando posibles reversiones de tendencia.

💡 **Niveles de confianza por timeframe:**
• 4H: 82%+ | 6H: 83%+ | 8H: 84%+
• 12H: 85%+ | 1D: 82%+ (optimizado para HYPE)

🔥 **Señales de calidad:**
• 82-89%: Señal confirmada 🟠
• 90-95%: Señal premium ⚡
• 95%+: Señal ultra premium 🔥

🌐 **Webhook URL:**
`https://tu-dominio.railway.app/webhook/tradingview`"""

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

    async def# main.py - Bot RSI Divergence Optimizado v2.1

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
        self.min_confidence = 85  # Mantener global alto
        self.final_confidence = 95
        self.max_alerts_per_hour = 50  # Límite de spam
        
        # Variable para tracking de timeframe actual en scan
        self.current_scanning_timeframe = '4h'
        
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
            self.update_pairs_with_trending_tokens()  # Añadir tokens trending
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
            'UNIUSDT', 'AAVEUSDT', 'MKRUSDT', 'COMPUSDT',
            # Trending tokens
            'HYPEUSDT', 'MOVEUSDT', 'PENGUUSDT', 'VIRTUALUSDT'
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
            'UNIUSDT', 'AAVEUSDT', 'MKRUSDT',
            # Trending tokens importantes
            'HYPEUSDT', 'MOVEUSDT', 'PENGUUSDT'
        ]
        
        # Solo agregar pares que existen en Bybit
        for pair in default_pairs:
            if pair in self.all_bybit_pairs:
                self.active_pairs.add(pair)
                
        logger.info(f"✅ Cargados {len(self.active_pairs)} pares por defecto")

    def update_pairs_with_trending_tokens(self):
        """Añadir tokens trending que podrían estar faltando"""
        trending_tokens = [
            'HYPEUSDT',   # El que falta!
            'MOVEUSDT',   # Otro token nuevo popular
            'MEUSDT',     # Movement
            'VANAUSDT',   # Vana
            'USUALUSDT',  # Usual
            'PENGUUSDT',  # Pudgy Penguins
            'VIRTUALUSDT', # Virtual Protocol
            'AIUSDT',     # Sleepless AI
            'GRASSUSDT',  # Grass
            'ACTUSDT',    # ACT
            # Añadir más según tendencias
        ]
        
        added_count = 0
        for token in trending_tokens:
            if token in self.all_bybit_pairs and token not in self.active_pairs:
                self.active_pairs.add(token)
                added_count += 1
                logger.info(f"✅ Añadido token trending: {token}")
        
        logger.info(f"🔥 Añadidos {added_count} tokens trending al monitoreo")
        return added_count

    def setup_webhook_routes(self):
        """Configurar rutas Flask optimizadas"""
        
        @self.app.route('/', methods=['GET'])
        def home():
            return jsonify({
                "status": "🚀 RSI Divergence Bot v2.1 ONLINE",
                "version": "2.1",
                "active_pairs": len(self.active_pairs),
                "total_pairs": len(self.all_bybit_pairs),
                "uptime": datetime.now().isoformat(),
                "webhook_url": f"https://{request.host}/webhook/tradingview",
                "stats": dict(self.scan_stats),
                "timeframes": self.timeframes
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
                "top_pairs": self.get_top_performing_pairs(),
                "timeframe_stats": {tf: self.scan_stats.get(f'{tf}_divergences', 0) for tf in self.timeframes}
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
        """Obtener datos OHLCV con mapeo corregido de timeframes"""
        try:
            # MAPEO CORREGIDO - CRÍTICO PARA 1D
            timeframe_map = {
                '1m': '1m',
                '5m': '5m', 
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '2h': '2h',
                '4h': '4h', 
                '6h': '6h', 
                '8h': '8h',
                '12h': '12h', 
                '1d': '1d',    # ¡CORREGIDO!
                '1D': '1d',    # Agregar ambas variaciones
                'D': '1d',     # Fallback
                '3d': '3d',
                '1w': '1w',
                '1M': '1M'
            }
            
            # Usar el mapeo correcto
            bybit_timeframe = timeframe_map.get(timeframe, timeframe)
            
            logger.debug(f"📊 Obteniendo {symbol} {timeframe} -> {bybit_timeframe}")
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, bybit_timeframe, limit=limit)
            
            if not ohlcv:
                logger.warning(f"❌ No hay datos para {symbol} {timeframe}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.debug(f"✅ {symbol} {timeframe}: {len(df)} velas obtenidas")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos {symbol} {timeframe}: {e}")
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

    def get_timeframe_config(self, timeframe: str) -> dict:
        """Configuración calibrada por timeframe para máxima precisión"""
        configs = {
            '4h': {
                'min_distance': 3,            # Más sensible para capturar divergencias
                'lookback_period': 5,         # Ventana de análisis
                'min_price_change': 1.5,      # 1.5% mínimo - divergencias pueden ser sutiles
                'min_rsi_change': 2.5,        # 2.5 puntos RSI - más sensible
                'max_time_diff': 6,           # Hasta 6 períodos de diferencia (24h)
                'rsi_high_threshold': 62,     # RSI >62 para bajista
                'rsi_low_threshold': 38,      # RSI <38 para alcista  
                'confidence_base': 72,        # Base calibrada
                'min_confidence': 82,         # Umbral de calidad
                'volume_weight': 0.15,        # Peso del volumen
                'trend_consistency_weight': 0.10  # Peso de consistencia
            },
            '6h': {
                'min_distance': 3,
                'lookback_period': 4,         
                'min_price_change': 2.0,      # Cambios ligeramente mayores
                'min_rsi_change': 3.0,        
                'max_time_diff': 5,           # Hasta 5 períodos (30h)
                'rsi_high_threshold': 63,     
                'rsi_low_threshold': 37,      
                'confidence_base': 73,        
                'min_confidence': 83,         
                'volume_weight': 0.15,
                'trend_consistency_weight': 0.10
            },
            '8h': {
                'min_distance': 3,
                'lookback_period': 4,
                'min_price_change': 2.5,      # Cambios más significativos
                'min_rsi_change': 3.5,
                'max_time_diff': 4,           # Hasta 4 períodos (32h)
                'rsi_high_threshold': 64,
                'rsi_low_threshold': 36,
                'confidence_base': 74,
                'min_confidence': 84,
                'volume_weight': 0.12,
                'trend_consistency_weight': 0.08
            },
            '12h': {
                'min_distance': 3,
                'lookback_period': 4,
                'min_price_change': 3.0,      # Movimientos más sustanciales
                'min_rsi_change': 4.0,
                'max_time_diff': 4,           # Hasta 4 períodos (48h)
                'rsi_high_threshold': 65,
                'rsi_low_threshold': 35,
                'confidence_base': 75,
                'min_confidence': 85,
                'volume_weight': 0.10,
                'trend_consistency_weight': 0.08
            },
            '1d': {
                'min_distance': 3,            # Sensible pero preciso para daily
                'lookback_period': 4,
                'min_price_change': 4.0,      # Cambios diarios significativos
                'min_rsi_change': 4.5,        # RSI debe mostrar divergencia clara
                'max_time_diff': 4,           # Hasta 4 días de diferencia
                'rsi_high_threshold': 62,     # Flexible para capturar como HYPE
                'rsi_low_threshold': 38,      # Simétrico
                'confidence_base': 72,        # Base ajustada para 1D
                'min_confidence': 82,         # Calidad alta pero alcanzable
                'volume_weight': 0.08,        # Menos peso al volumen en 1D
                'trend_consistency_weight': 0.12  # Más peso a consistencia
            }
        }
        
        return configs.get(timeframe, configs['4h'])  # Default a 4h si no existe

    def find_peaks_and_troughs_smart(self, highs: np.array, lows: np.array, 
                                    min_distance: int, lookback_period: int) -> Tuple[List[int], List[int]]:
        """Encontrar picos y valles con algoritmo mejorado que reduce falsos positivos"""
        if len(highs) < min_distance * 2 + 1:
            return [], []
            
        peaks = []
        troughs = []
        
        for i in range(lookback_period, len(highs) - lookback_period):
            # PICOS: Usar validación estricta + validación flexible
            # Validación estricta: debe ser máximo absoluto en el rango
            is_peak_strict = (highs[i] == np.max(highs[i-min_distance:i+min_distance+1]))
            
            # Validación adicional: debe superar umbral de significancia
            local_max = np.max(highs[i-lookback_period:i+lookback_period+1])
            local_min = np.min(highs[i-lookback_period:i+lookback_period+1])
            range_threshold = (local_max - local_min) * 0.3  # 30% del rango local
            
            is_significant = highs[i] >= local_min + range_threshold
            
            if is_peak_strict and is_significant:
                # Verificar que no hay otro pico muy cerca
                too_close = any(abs(i - existing_peak) < min_distance for existing_peak in peaks)
                if not too_close:
                    peaks.append(i)
            
            # VALLES: Lógica similar pero invertida
            is_trough_strict = (lows[i] == np.min(lows[i-min_distance:i+min_distance+1]))
            
            local_max_low = np.max(lows[i-lookback_period:i+lookback_period+1])
            local_min_low = np.min(lows[i-lookback_period:i+lookback_period+1])
            range_threshold_low = (local_max_low - local_min_low) * 0.3
            
            is_significant_low = lows[i] <= local_max_low - range_threshold_low
            
            if is_trough_strict and is_significant_low:
                too_close = any(abs(i - existing_trough) < min_distance for existing_trough in troughs)
                if not too_close:
                    troughs.append(i)
        
        return peaks, troughs

    def find_peaks_and_troughs(self, data: np.array, min_distance: int = 5) -> Tuple[List[int], List[int]]:
        """Encontrar picos y valles mejorado (versión simple para RSI)"""
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

    def calculate_divergence_strength(self, price_change: float, rsi_change: float, 
                                    time_diff: int, config: dict) -> float:
        """Calcular la fuerza de la divergencia (0-1)"""
        # Normalizar cambios
        price_strength = min(1.0, price_change / (config['min_price_change'] * 3))
        rsi_strength = min(1.0, rsi_change / (config['min_rsi_change'] * 3))
        
        # Penalizar diferencias temporales
        time_penalty = max(0, 1 - (time_diff / config['max_time_diff']))
        
        # Fuerza combinada
        strength = (price_strength * 0.4 + rsi_strength * 0.4 + time_penalty * 0.2)
        
        return strength

    def calculate_advanced_confidence(self, price_change: float, rsi_change: float, 
                                    time_diff: int, strength: float, rsi1: float, 
                                    rsi2: float, config: dict, div_type: str) -> float:
        """Cálculo avanzado de confianza calibrado por timeframe"""
        base_confidence = config['confidence_base']
        
        # Bonus escalado por magnitud de cambios
        price_bonus = min(12, (price_change / config['min_price_change']) * 2.5)
        rsi_bonus = min(8, (rsi_change / config['min_rsi_change']) * 2)
        
        # Bonus por fuerza de divergencia (más peso)
        strength_bonus = strength * 12
        
        # Bonus por niveles extremos de RSI (calibrado por timeframe)
        if div_type == 'bearish':
            rsi_extreme_threshold = 75 if config['min_confidence'] >= 85 else 70
            rsi_level_bonus = max(0, (max(rsi1, rsi2) - rsi_extreme_threshold) / 1.5)
        else:  # bullish
            rsi_extreme_threshold = 25 if config['min_confidence'] >= 85 else 30
            rsi_level_bonus = max(0, (rsi_extreme_threshold - min(rsi1, rsi2)) / 1.5)
        
        # Penalty por diferencia temporal (más estricto)
        time_penalty = (time_diff / config['max_time_diff']) * 6
        
        # Bonus por alineación perfecta
        perfect_alignment_bonus = 3 if time_diff <= 1 else 0
        
        # Bonus por cambio significativo (escalado por timeframe)
        significance_multiplier = 1.2 if config['min_confidence'] <= 83 else 1.0
        if price_change > config['min_price_change'] * 2:
            significance_bonus = 3 * significance_multiplier
        else:
            significance_bonus = 0
        
        confidence = (base_confidence + price_bonus + rsi_bonus + strength_bonus + 
                     rsi_level_bonus + perfect_alignment_bonus + significance_bonus - time_penalty)
        
        return max(0, min(96, confidence))

    def get_context_validation_bonus(self, data: np.array, rsi: np.array, 
                                   p1_idx: int, p2_idx: int, r1_idx: int, r2_idx: int) -> float:
        """Bonus por validación contextual mejorado"""
        bonus = 0
        
        # Bonus mayor si los picos están perfectamente alineados
        alignment_diff = abs(p2_idx - r2_idx)
        if alignment_diff == 0:
            bonus += 4
        elif alignment_diff <= 1:
            bonus += 2
        elif alignment_diff <= 2:
            bonus += 1
        
        # Bonus por consistencia en la tendencia direccional
        if p2_idx > p1_idx and r2_idx > r1_idx:  # Ambos se mueven hacia adelante
            price_trend = data[p2_idx] - data[p1_idx]
            rsi_trend = rsi[r2_idx] - rsi[r1_idx]
            
            # Divergencia clara: precio sube pero RSI baja (o viceversa)
            if (price_trend > 0 and rsi_trend < 0) or (price_trend < 0 and rsi_trend > 0):
                divergence_clarity = abs(price_trend / data[p1_idx]) + abs(rsi_trend / rsi[r1_idx])
                bonus += min(3, divergence_clarity * 100)
        
        # Bonus por confirmación de momentum
        if len(data) > max(p2_idx, r2_idx) + 2:
            recent_price_trend = np.mean(np.diff(data[max(p2_idx-2, 0):p2_idx+3]))
            recent_rsi_trend = np.mean(np.diff(rsi[max(r2_idx-2, 0):r2_idx+3]))
            
            if (recent_price_trend > 0 and recent_rsi_trend < 0) or (recent_price_trend < 0 and recent_rsi_trend > 0):
                bonus += 2
