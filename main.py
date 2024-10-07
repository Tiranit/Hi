import ccxt
import pandas as pd
import threading
import smtplib
import ssl
import os
import mplfinance as mpf
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import time

class MyApp:
    def __init__(self):
        self.email_sending = True
        self.exchange_name = 'kucoin'
        self.results = []
        self.stop_event = threading.Event()
        self.symbol_index = 0
        self.timeframe_index = 0
        self.email_results = []

    def send_email(self, subject, body, attachments=[]):
        sender_email = os.getenv("EMAIL_SENDER")
        receiver_email = os.getenv("EMAIL_RECEIVER")
        password = os.getenv("EMAIL_PASSWORD")

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        for attachment in attachments:
            with open(attachment, "rb") as file:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {os.path.basename(attachment)}",
            )
            message.attach(part)

        context = ssl.create_default_context()
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, message.as_string())
            print("Email sent successfully")
        except Exception as e:
            print(f"Error sending email: {e}")

    def send_scheduled_email(self):
        if not self.email_results:
            print("No new results to send.")
            return
        
        subject = "Scheduled Trading Results"
        body = "Here are the latest trading results:\n\n"
        
        attachments = []
        for result in self.email_results:
            symbol, results, images = result
            body += f"Results found for {symbol}:\n"
            for timeframe, length in results:
                body += f"{timeframe} - {length} candles\n"
            body += "\n"
            attachments.extend(images)
        
        body += "\nPlease check the attached charts."
        
        self.send_email(subject, body, attachments)
        
        self.email_results.clear()  # Clear results after sending email

    def run_code(self):
        exchange_class = getattr(ccxt, self.exchange_name)
        exchange = exchange_class()
        symbols = ['NEIRO/USDT', 'G/USDT', 'CHZ/USDT', 'LTO/USDT', 'ICP/USDT', 'NMR/USDT', 'ONT/USDT', 'RENDER/USDT', 'XTZ/USDT', 'ONE/USDT', 'POLS/USDT', 'GTC/USDT', 'PROM/USDT', 'HFT/USDT', 'T/USDT', 'MOVR/USDT', 'MEME/USDT', 'RLC/USDT', 'MANA/USDT', 'MASK/USDT', 'MINA/USDT', 'SKL/USDT', 'TUSD/USDT', 'FTT/USDT', 'GFT/USDT', 'DASH/USDT', 'DATA/USDT', 'DYDX/USDT', 'DOGE/USDT', 'DOCK/USDT', 'DODO/USDT', 'DOGS/USDT', 'DUSK/USDT', 'DENT/USDT', 'DEGO/USDT', 'HMSTR/USDT', 'AVA/USDT', 'LISTA/USDT', 'POL/USDT', 'OP/USDT', 'CFX/USDT', 'XEC/USDT', 'JASMY/USDT', 'JUP/USDT', 'WRX/USDT', 'SAND/USDT', 'FLOKI/USDT', 'SCRT/USDT', 'SUI/USDT', 'WIF/USDT', 'OMG/USDT', 'GMX/USDT', 'HBAR/USDT', 'AUCTION/USDT', 'NOT/USDT', 'WBTC/USDT', 'LINK/USDT', 'LINA/USDT', 'LQTY/USDT', 'LOKA/USDT', 'HNT/USDT', 'LOOM/USDT', 'LUNA/USDT', 'LUNC/USDT', 'EOS/USDT', 'ACE/USDT', 'GLMR/USDT', 'WIN/USDT', 'WOO/USDT', 'GAS/USDT', 'YFI/USDT', 'AGLD/USDT', 'CHR/USDT', 'ETHDOWN/USDT', 'STORJ/USDT', 'STRAX/USDT', 'INJ/USDT', 'QKC/USDT', 'ERN/USDT', 'PYR/USDT', 'BCH/USDT', 'BAL/USDT', 'ETC/USDT', 'KLAY/USDT', 'QI/USDT', 'DEXE/USDT', 'RPL/USDT', 'KDA/USDT', 'GRT/USDT', 'OMNI/USDT', 'CRV/USDT', 'OSMO/USDT', 'XNO/USDT', 'RDNT/USDT', 'XAI/USDT', 'PHA/USDT', 'LRC/USDT', 'ALT/USDT', 'ACH/USDT', 'CTSI/USDT', 'KSM/USDT', 'VIDT/USDT', 'YGG/USDT', 'CYBER/USDT', 'COMBO/USDT', 'AMB/USDT', 'ZIL/USDT', 'ATA/USDT', 'BAT/USDT', 'TLM/USDT', 'FIDA/USDT', 'BSW/USDT', 'MAGIC/USDT', 'MANTA/USDT', 'METIS/USDT', 'SEI/USDT', 'AR/USDT', 'DYM/USDT', 'GNS/USDT', 'NEO/USDT', 'SLP/USDT', 'TRX/USDT', 'ALPINE/USDT', 'LTC/USDT', 'CVX/USDT', 'TFUEL/USDT', 'THETA/USDT', 'NEAR/USDT', 'LDO/USDT', 'FXS/USDT', 'JTO/USDT', 'TRU/USDT', 'BLUR/USDT', 'EPX/USDT', 'FIL/USDT', 'ATOM/USDT', 'ALGO/USDT', 'ANKR/USDT', 'AVAX/USDT', 'API3/USDT', 'USDC/USDT', 'ARPA/USDT', 'ARKM/USDT', 'SXP/USDT', 'XLM/USDT', 'AXS/USDT', 'EGLD/USDT', 'AUDIO/USDT', 'AERGO/USDT', 'WLD/USDT', 'REN/USDT', 'ILV/USDT', 'GMT/USDT', 'EIGEN/USDT', 'NFP/USDT', 'TRB/USDT', 'ETH/USDT', 'STX/USDT', 'IOTA/USDT', 'IOST/USDT', 'IOTX/USDT', 'MBL/USDT', 'BTCUP/USDT', 'PAXG/USDT', 'PYTH/USDT', 'BLZ/USDT', 'PERP/USDT', 'PEPE/USDT', 'PUNDIX/USDT', 'PEOPLE/USDT', 'PENDLE/USDT', 'TIA/USDT', 'POND/USDT', 'PORTAL/USDT', 'BSV/USDT', 'TNSR/USDT', 'LSK/USDT', 'ELF/USDT', 'TAO/USDT', 'OGN/USDT', 'XMR/USDT', 'SYS/USDT', 'HIFI/USDT', 'ENS/USDT', 'DAR/USDT', 'SOL/USDT', 'JST/USDT', 'DCR/USDT', 'KNC/USDT', 'MAV/USDT', 'REZ/USDT', 'ENJ/USDT', 'IO/USDT', 'UNI/USDT', 'STRK/USDT', 'FORTH/USDT', 'QNT/USDT', 'SHIB/USDT', 'SNX/USDT', 'USDP/USDT', 'USTC/USDT', 'SSV/USDT', 'HARD/USDT', 'BTCDOWN/USDT', 'HIGH/USDT', 'ZRX/USDT', 'WAXP/USDT', 'EDU/USDT', 'ADA/USDT', 'APT/USDT', 'DGB/USDT', 'CKB/USDT', 'IMX/USDT', 'ZK/USDT', 'XRP/USDT', 'SUN/USDT', 'DOT/USDT', 'SYN/USDT', 'SUSHI/USDT', 'SUPER/USDT', 'BTC/USDT', 'OXT/USDT', 'OM/USDT', 'WAVES/USDT', 'ARB/USDT', 'QUICK/USDT', 'TWT/USDT', 'MTL/USDT', 'KAVA/USDT', 'RAY/USDT', 'ORDI/USDT', 'ROSE/USDT', 'RUNE/USDT', 'REEF/USDT', 'STG/USDT', 'BANANA/USDT', 'COTI/USDT', 'COMP/USDT', 'CAKE/USDT', 'ZRO/USDT', 'CATI/USDT', 'BTT/USDT', 'CREAM/USDT', 'TON/USDT', 'CELR/USDT', 'CELO/USDT', 'SFP/USDT', 'ADX/USDT', 'FLOW/USDT', 'BB/USDT', 'FLUX/USDT', 'ICX/USDT', 'W/USDT', 'RVN/USDT', 'LPT/USDT', 'ORN/USDT', 'APE/USDT', 'ENA/USDT', 'RSR/USDT', 'GLM/USDT', 'ID/USDT', 'MKR/USDT', 'XEM/USDT', 'PIXEL/USDT', 'POLYX/USDT', 'LIT/USDT', 'BONK/USDT', 'BOME/USDT', 'AMP/USDT', 'TURBO/USDT', 'NTRN/USDT', 'FET/USDT', 'SLF/USDT', 'UMA/USDT', 'FTM/USDT', 'VANRY/USDT', 'VOXEL/USDT', 'CLV/USDT', 'UNFI/USDT', 'BAND/USDT', 'BICO/USDT', '1INCH/USDT', 'AAVE/USDT', 'AKRO/USDT', 'ASTR/USDT', 'UTK/USDT', 'BOND/USDT', 'C98/USDT', 'REQ/USDT', 'DIA/USDT', 'VET/USDT', 'AEVO/USDT', 'ZEC/USDT', 'BNB/USDT', 'ALPHA/USDT', 'NKN/USDT', 'ALICE/USDT', 'BURGER/USDT', 'ETHFI/USDT', 'ZEN/USDT', 'KMD/USDT']
        timeframes = ['5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M']
        candle_lengths = [50, 100, 150, 250]

        try:
            while self.symbol_index < len(symbols):
                symbol = symbols[self.symbol_index]
                results = [symbol]
                timeframes_with_results = []
                while self.timeframe_index < len(timeframes):
                    timeframe = timeframes[self.timeframe_index]
                    if self.stop_event.is_set():
                        return
                    try:
                        bars = exchange.fetch_ohlcv(symbol, timeframe)
                        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['SMA5'] = df['close'].rolling(window=5).mean()
                        df['SMA10'] = df['close'].rolling(window=10).mean()
                        df['SMA15'] = df['close'].rolling(window=15).mean()
                        df['SMA20'] = df['close'].rolling(window=20).mean()

                        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
                        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
                        df['MACD'] = df['EMA12'] - df['EMA26']
                        df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

                        high_price = df['high'].max()
                        low_price = df['low'].min()

                        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0]
                        fib_levels = {}
                        for level in levels:
                            fib_levels[level] = high_price - (high_price - low_price) * level

                        for length in candle_lengths:
                            if len(df) >= length:
                                latest_row = df.iloc[-1]
                                df_length = df.tail(length)

                                lowest_price_row = df_length.loc[df_length['low'].idxmin()]
                                highest_price_row = df_length.loc[df_length['high'].idxmax()]

                                candle_count_between_high_low = abs(highest_price_row.name - lowest_price_row.name)

                                if lowest_price_row.name < highest_price_row.name:
                                    if (latest_row['close'] < latest_row['SMA5'] and
                                        latest_row['close'] < latest_row['SMA10'] and
                                        latest_row['close'] < latest_row['SMA15'] and
                                        latest_row['close'] < latest_row['SMA20'] and
                                        latest_row['close'] < df_length['low'].iloc[-3:-1].min() and
                                        candle_count_between_high_low < 30):

                                        fib_0 = fib_levels[0.0]
                                        fib_0_236 = fib_levels[0.236]
                                        fib_0_382 = fib_levels[0.382]

                                        reaction = False
                                        for i in range(len(df_length) - 1):
                                            if df_length['low'].iloc[i] <= fib_0_236 and df_length['low'].iloc[i] > fib_0_382:
                                                reaction_row = df_length.iloc[i]
                                                next_close = df_length['close'].iloc[i + 1]
                                                if (next_close - reaction_row['low']) / reaction_row['low'] > 0.01:
                                                    reaction = True
                                                    break

                                        if reaction:
                                            results.append((timeframe, length))
                                            timeframes_with_results.append((timeframe, length))
                    except Exception as e:
                        print(f"Error fetching data for {symbol} in {timeframe}: {e}")
                        continue
                    self.timeframe_index += 1

                if len(results) > 1:
                    images = self.plot_charts(exchange, symbol, timeframes_with_results)
                    self.email_results.append((symbol, results[1:], images))

                self.symbol_index += 1
                self.timeframe_index = 0

            self.send_scheduled_email()

        except Exception as e:
            print(f"Error occurred: {e}")

    def plot_charts(self, exchange, symbol, timeframes):
        images = []
        os.makedirs('charts', exist_ok=True)

        fib_colors = {
            0.0: 'gray',
            0.236: 'blue',
            0.382: 'purple',
            0.5: 'green',
            0.618: 'orange',
            0.764: 'red',
            1.0: 'gray'
        }

        for timeframe, length in timeframes:
            try:
                bars = exchange.fetch_ohlcv(symbol, timeframe)
                df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                df_length = df.tail(length)

                df_length['EMA12'] = df_length['close'].ewm(span=12, adjust=False).mean()
                df_length['EMA26'] = df_length['close'].ewm(span=26, adjust=False).mean()
                df_length['MACD'] = df_length['EMA12'] - df_length['EMA26']
                df_length['Signal Line'] = df_length['MACD'].ewm(span=9, adjust=False).mean()

                high_price = df_length['high'].max()
                low_price = df_length['low'].min()

                log_high_price = np.log(high_price)
                log_low_price = np.log(low_price)

                levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0]
                fib_levels = {}
                for level in levels:
                    log_fib_level = log_high_price - (log_high_price - log_low_price) * level
                    fib_levels[level] = np.exp(log_fib_level)

                addplots = [
                    mpf.make_addplot(df_length['MACD'], panel=1, color='black'),
                    mpf.make_addplot(df_length['Signal Line'], panel=1, color='red')
                ]

                for level, value in fib_levels.items():
                    addplots.append(mpf.make_addplot([value] * len(df_length), color=fib_colors[level], linestyle='dashed'))

                chart_file = f'charts/{symbol.replace("/", "_")}_{timeframe}_{length}candles.png'
                title = f'{symbol} {timeframe} {length} Candles'
                mpf.plot(df_length, type='candle', addplot=addplots, volume=True, style='yahoo', savefig=dict(fname=chart_file, dpi=100, bbox_inches="tight"), title=title, ylabel='Price (log scale)', yscale='log')
                images.append(chart_file)

            except Exception as e:
                print(f"Error plotting data for {symbol} in {timeframe} with {length} candles: {e}")
                continue
        return images

if __name__ == '__main__':
    my_app = MyApp()
    my_app.run_code()
