import pandas_datareader.data as web
import datetime as dt


def get_fin_data(init_date, fin_date, ticker):
    df = web.DataReader(ticker, 'yahoo', init_date, fin_date)
    df.index.names = ['Dates']

    # Calculate the daily changes
    df['day_chan'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['day_chan_scal'] = scaler.fit_transform(df[['day_chan']])
    df['day_chan_scal_shift'] = df['day_chan_scal'].shift(-1)
    df['buy_sell'] = df['day_chan_scal_shift'].apply(lambda x: 1
                                                     if x >= 0 else -1)

    #return df['buy_sell']
    return df
