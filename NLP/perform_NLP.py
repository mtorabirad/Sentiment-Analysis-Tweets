from NLP.lemmatizer import parallel_lemmatize_in_chunks as plc
from NLP.sentiment_analyzer import parallel_sentiment_in_chunks as psc
from UTIL import FileIO
config = FileIO.read_yaml(r'CONFIG\config_data.yml')


def perform_NLP(df):
    parameters = {'n_jobs': config['n_jobs'], 'backend': config['backend'],
                  'prefer': config['prefer']}
    df['lemmatized'] = plc(df, config['chunk_size'], **parameters)
    df['sent_score'] = psc(df, config['chunk_size'], **parameters)

    df['eff_sent_score'] = df['sent_score'].apply(lambda x: x['compound'])\
        * df['num_follower']

    df.sort_index(ascending=True)

    df.to_csv(config['CSV_file_path'], index=False)
    return df
