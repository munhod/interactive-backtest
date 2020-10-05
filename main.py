import pandas as pd
import numpy as np
import datetime
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit


def main():
    
    @streamlit.cache(allow_output_mutation=True)
    def load_data():
        return pd.read_pickle('df.pkl')

    def show_graph(df):

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        ticker_price_data  = df

        # Add traces    
        
        fig.add_trace(
            go.Candlestick(x=ticker_price_data.index,
                        open=ticker_price_data.askopen,
                        high=ticker_price_data.askhigh,
                        low=ticker_price_data.asklow,
                        close=ticker_price_data.askclose),
            secondary_y=False,
        )

        fig.add_trace(
            go.Line(x=ticker_price_data.index, y=ticker_price_data['ind_1'], name="ind_1"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Line(x=ticker_price_data.index, y=ticker_price_data['ind_2'], name="ind_2"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=ticker_price_data[ticker_price_data['position'] > 0].index, 
                    y=ticker_price_data[ticker_price_data['position'] > 0]['ind_1'], name="Long entry & exit",
                    mode = 'markers',  marker = dict(size = 10, symbol = 5,color = "black")),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=ticker_price_data[ticker_price_data['position'] < 0].index, 
                    y=ticker_price_data[ticker_price_data['position'] < 0]['ind_1'], name="Short entry & exit",
                    mode = 'markers',  marker = dict(size = 10, symbol = 6, color = 'yellow')),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=ticker_price_data.index, y=ticker_price_data['strategy'], name="Strategy", fill='tonexty'),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(x=ticker_price_data.index, y=ticker_price_data['buy&hold'], name="Buy and Hold"),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            title_text="Price Chart and Backtest Results", height=500, width=1000,
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Date", type = 'category')

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Price and Buy/Sell Signals</b> yaxis title", secondary_y=False)
        fig.update_yaxes(title_text="<b>Net Profit</b> yaxis title", secondary_y=True)
        
        streamlit.plotly_chart(fig)
    
    def wwma(values, n):
        """
        J. Welles Wilder's EMA 
        """
        return values.ewm(alpha=1/n, adjust=False).mean()

    def atr(df, n=14):
        data = df.copy()
        high = data['askhigh']
        low = data['asklow']
        close = data['askclose']
        data['tr0'] = abs(high - low)
        data['tr1'] = abs(high - close.shift())
        data['tr2'] = abs(low - close.shift())
        tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
        atr = wwma(tr, n)
        return atr

    streamlit.header('Moving Averages Cross Strategy on EUR/USD')
    streamlit.subheader('Showcasing how easy it is to analyse your financial data with Streamlit.')
    df = load_data()
    
    win_ind_1 = streamlit.sidebar.slider('Choose window for Indicator 1', 1, 200, 10)
    win_ind_2 = streamlit.sidebar.slider('Choose window for Indicator 2',1, 200, 20)

    df['ind_1'] = df['askclose'].rolling(win_ind_1).mean()
    df['ind_2'] = df['askclose'].rolling(win_ind_2).mean()

    df['atr'] = atr(df,20)

    df['signal'] = 0
    df['signal'] = np.where(df['ind_1'] > df['ind_2'], 1, -1)

    df['position'] = df['signal'].diff()
    df['returns'] = df['askclose'].pct_change()
    df['buy&hold'] = df['returns'].cumsum()
    df['pnl'] = df['returns'] * df['position']
    df['strategy'] = df['pnl'].cumsum()
    df.index = pd.to_datetime(df.index, utc=True).date
    show_graph(df)

    sr = df['pnl'].mean() / df['pnl'].std() * np.sqrt(252)
    
    streamlit.write('Total return:{}'.format(round(df['strategy'].iloc[-1], 2)))
    streamlit.write('Sharpe Ratio:{}'.format(round(sr, 2)))
    
if __name__ == "__main__":
                        
    main()

     
