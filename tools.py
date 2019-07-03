import os
import pickle
import quandl
import numpy as np


def date_obj_to_str(date_obj):
    return date_obj.strftime('%Y-%m-%d')


def save_pickle(something, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as fh:
        pickle.dump(something, fh, pickle.DEFAULT_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def fetch_stock_price(symbol,
                      from_date,
                      to_date,
                      cache_path=r"C:\Courses\Deeplearning\06-RNN\tmp\prices"):
    assert(from_date <= to_date)

    filename = "{}_{}_{}.pk".format(symbol, str(from_date), str(to_date))
    #filename="MSFT_2015-01-01_2016-12-31.pk"
    price_filepath = os.path.join(cache_path, filename)

    try:
        prices = load_pickle(price_filepath)
        print("loaded from", price_filepath)

    except IOError:
        historic = quandl.get("WIKI/" + symbol,
                              start_date=date_obj_to_str(from_date),
                              end_date=date_obj_to_str(to_date))

        prices = historic["Adj. Close"].tolist()
        save_pickle(prices, price_filepath)
        print("saved into", price_filepath)

        # If it does not work to use quandl API to retrieve stock data,
        # we can use Yahoo finance to retrieve the data

        # from pandas_datareader import data as pdr
        # import datetime
        #
        # import fix_yahoo_finance as yf
        # yf.pdr_override()
        #
        # #Sample to define time range from data and to date
        # #from_date = datetime.datetime(2018, 1, 1)
        # #to_date = datetime.date.today()
        # #closing_df = pdr.get_data_yahoo(["AAPL", "GOOG", "MSFT", "AMZN"], start, end)['Adj Close']
        # prices = pdr.get_data_yahoo(["MSFT"], from_date, to_date)['Adj Close']
        # #prices.head(5)
    return prices

def fetch_cosine_values(seq_len, frequency=0.01, noise=0.1):
    np.random.seed(101)
    x = np.arange(0.0, seq_len, 1.0)
    return np.cos(2 * np.pi * frequency * x) + np.random.uniform(low=-noise, high=noise, size=seq_len)


def format_dataset(values, temporal_features):
    feat_splits = [values[i:i + temporal_features] for i in range(len(values) - temporal_features)]
    feats = np.vstack(feat_splits)
    labels = np.array(values[temporal_features:])
    return feats, labels


def matrix_to_array(m):
    return np.asarray(m).reshape(-1)


if __name__ == "__main__":
    print(fetch_cosine_values(10, frequency=0.1))
    import datetime
    sdt=fetch_stock_price("MSFT",
                             datetime.date(2017, 1, 1),
                             datetime.date(2017, 1, 31))
    sdt.head(5)
    print(type(sdt))
    print(sdt)
    # print(fetch_stock_price("GOOG",
    #                         datetime.date(2017, 1, 1),
    #                         datetime.date(2017, 1, 31)))
