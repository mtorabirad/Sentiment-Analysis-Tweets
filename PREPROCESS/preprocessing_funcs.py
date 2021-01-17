import json
import re
import pandas as pd
from pandas.tseries.offsets import BusinessDay
import numpy as np
from UTIL import FileIO
config = FileIO.read_yaml(r'CONFIG\config_data.yml')
from GETDATA.connect_with_API import get_fin_data


def get_df():
    if not config['JSON_already_avail']:
        parameters = {'input_file_path': config['txt_file_path'],
                      'num_tweets_to_process': config['num_tweets_to_process'],
                      'JSON_file_path': config['JSON_file_path']}

        _convert_txt_JSON(**parameters)

    df = _extract_relevant_info(config['JSON_file_path'])

    parameters = {'end': config['end_of_trading'],
                  'time_zone': config['time_zone']}

    df = _calc_influencing_dates(df, **parameters)
    df = _cleaner(df)
    return df


def obtain_and_append_fin_data(df):
    # Get financial data
    # Group by one column and apply a different aggregation function
    # on each of the other columns.
    # https://pandas.pydata.org/pandas-docs/version/0.23.1/generated/
    # pandas.core.groupby.DataFrameGroupBy.agg.html
    # df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
    df[["eff_sent_score"]] = df[["eff_sent_score"]].apply(pd.to_numeric)
    date_score = df.groupby('influencing_date').agg({'eff_sent_score':
                                                    ['mean']})

    date_score.columns = date_score.columns.droplevel()
    date_score = date_score.rename(columns={'mean': 'mean_sent_score'})

    if config['fin_data_already_avail']:
        tmp = pd.read_csv(config['fin_data_file_path'], index_col=0,
                          names=['Dates', 'buy_sell'])
        fin_data = pd.Series(tmp['buy_sell'], index=tmp.index)
        fin_data.index = pd.to_datetime(fin_data.index)
    else:
        init_date, fin_date = min(date_score.index), max(date_score.index)
        parameters = {'init_date': init_date, 'fin_date': fin_date,
                      'ticker': config['ticker']}
        fin_data = get_fin_data(**parameters)
        fin_data = fin_data.dropna(axis=0, how='any')
        fin_data.to_csv(config['fin_data_file_path'], index=True)

    # Construct training df
    training_df = pd.merge(date_score, fin_data['buy_sell'],
                           left_index=True, right_index=True,
                           how="inner")

    training_df.to_csv(config['training_file_path'], index=True)
    return fin_data, training_df


def _convert_txt_JSON(input_file_path, num_tweets_to_process, JSON_file_path):
    str_to_hold_file_contents = ""

    # Add , to the end of lines
    with open(input_file_path) as filehandle:
        cnt = 1
        for line in filehandle:
            if cnt > num_tweets_to_process:
                break
            if (cnt % 1000 == 0):
                print('line = ', cnt)
            stripped_line = line.strip()
            new_line = stripped_line.replace('}{', '},{')
            str_to_hold_file_contents += new_line + ",\n"
            cnt += 1

    # To remove the comma at the end of the last line.
    tmp = str_to_hold_file_contents[::-1]
    tmp = tmp.replace(',', '', 1)
    str_to_hold_file_contents = tmp[::-1]

    # Add "["" and ""]"" to the begining and end
    str_to_hold_file_contents = '[\n' + str_to_hold_file_contents + ']'

    writing_file = open(JSON_file_path, "w")
    writing_file.write(str_to_hold_file_contents)


def _extract_relevant_info(input):
    # with open(config['JSON_file_path'], 'r') as f:
    with open(input, 'r') as f:
        list_of_dicts = json.load(f)

    date_time = []
    tweet = []
    # within username
    username = []
    num_follower = []
    num_friend = []
    num_status = []
    num_favourite = []
    num_listed = []
    num_media = []
    num_favourite = []
    ####
    num_reply = []
    num_retweet = []
    num_like = []
    num_quote = []

    list_names = ["date_time", "tweet", 'username', "num_follower",
                  'num_friend', 'num_status', 'num_favourite',
                  'num_listed', 'num_media', 'num_favourite',
                  'num_reply', 'num_retweet', 'num_like', 'num_quote']

    col_names = list_names
    col_names[0] = 'date_time(UTC)'

    # Extract items from the list of dictionaries. You can use also list
    # comprehension or map operations.
    # https://stackoverflow.com/questions/7900882/extract-item-from-list-of-dictionaries
    for curr_item in list_of_dicts:
        date_time.append(pd.to_datetime(curr_item["date"]))
        tweet.append(re.sub(r"http\S+", "", curr_item["content"]))

        username.append(curr_item["user"]["username"])
        num_follower.append(curr_item["user"]["followersCount"])
        num_friend.append(curr_item["user"]["friendsCount"])
        num_status.append(curr_item["user"]["statusesCount"])
        num_favourite.append(curr_item["user"]["favouritesCount"])
        num_listed.append(curr_item["user"]["listedCount"])
        num_media.append(curr_item["user"]["mediaCount"])

        num_reply.append(curr_item["replyCount"])
        num_retweet.append(curr_item["retweetCount"])
        num_like.append(curr_item["likeCount"])
        num_quote.append(curr_item["quoteCount"])

    df = pd.DataFrame(np.column_stack([date_time, tweet, username,
                                       num_follower, num_friend, num_status,
                                       num_favourite, num_listed, num_media,
                                       num_favourite, num_reply, num_retweet,
                                       num_like, num_quote]),
                      columns=col_names)

    """ df = pd.DataFrame(list(zip(date_time, tweet, num_follower)),
                      columns=col_names) """

    return df.sort_index(ascending=False)


def _calc_influencing_dates(df, end, time_zone):

    df['date_time(Local)'] = df["date_time(UTC)"].apply(
                                            lambda x: x.tz_convert(time_zone))
    df['day_of_week'] = df["date_time(UTC)"].apply(lambda x: x.day_name())

    # beg_of_trading = pd.Timestamp('2017-10-30T' + begin, tz=time_zone)
    end_of_trading = pd.Timestamp('2017-10-30T' + end, tz=time_zone)
    parameters = {'end': end_of_trading}

    df['influencing_date'] = df["date_time(UTC)"].apply(
                                                   _get_trad_day_inf_by_tweet,
                                                   **parameters)

    df['influencing_date_name'] = df['influencing_date'].\
        apply(lambda x: pd.to_datetime(x).day_name())

    return df

# tmp = df["date_time(UTC)"].iloc[77]
# pd.to_datetime(_get_trad_day_inf_by_tweet(tmp,end_of_trading)).day_name()


def _get_trad_day_inf_by_tweet(datetime, end):
    # Get the trading day influenced by the tweet
    weekday, date, time = datetime.weekday(), datetime.date(), datetime.time()
    if weekday <= 4:
        if (time < end.time()):
            influencing_date = date
        else:
            influencing_date = (datetime + BusinessDay()).date()
    else:
        influencing_date = (datetime + BusinessDay()).date()

    return influencing_date


def _cleaner(df):
    "Extract relevant text from DataFrame using a regex"
    # Regex pattern for only alphanumeric, hyphenated text with 3 or more chars
    pattern = re.compile(r"[A-Za-z0-9,\-]{3,50}")
    df['clean'] = df['tweet'].str.findall(pattern).str.join(' ')
    return df
