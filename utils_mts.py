from typing import Dict, List, Optional, Set
import jsonlines
import pandas as pd
import ast
import re
import numpy as np
from itertools import chain
import numpy as np
import os
import json
import plotly.graph_objects as go
import plotly.express as px
import operator
from time import sleep
from getpass import getpass
from oauthlib.oauth2 import LegacyApplicationClient
from requests_oauthlib import OAuth2Session
import functools
from glob import glob
from datetime import datetime
import plotly.io as pio
from IPython.core.display import display, HTML
pio.renderers.default = 'notebook'


def valid_date(date_text):
    if len(date_text) > 0:
        try:
            datetime.strptime(date_text, '%Y-%m-%d')
            return True
        except ValueError:
            raise ValueError("Incorrect data format, should be YYYY-MM-DD")


def clear_ts(path):
    [os.remove(x) for x in glob('data/output/timeserie*.csv')]
    return f"files in {path} successfully cleared"


def authenticate(username, password, config_file="./conf/config.json"):
    with open(config_file) as fp:
        conf = json.loads(fp.read())

    host = conf['host']
    client_id = conf['_client_id']
    audience = conf['_audience']
    token_url = conf['_token_url']
    session = OAuth2Session(client=LegacyApplicationClient(client_id=client_id))
    token = session.fetch_token(username=username, password=password,
                                token_url=token_url, audience=audience,
                                client_id=client_id, include_client_id=True)
    return host, token


def print_progress(message):
    icons = ['.', '..', '...', '']
    for i in icons:
        print(f"{message} {i}", end="\r", flush=True)
        sleep(.5)


def save_request_to_file(mrequest, output_path):
    with open(output_path, 'w') as outfile:
        json.dump(mrequest, outfile, ensure_ascii=False, indent=3)


def load_request_from_file(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data


def read_data(datapath, output_format=None):

    lines = None
    if isinstance(datapath, str):
        with jsonlines.open(datapath) as reader:
            lines = [*reader]
    else:
        with jsonlines.Reader(datapath) as reader:
            lines = [*reader]
    if output_format == 'df':
        return pd.json_normalize(lines)
    return lines


def display_ts(timeseries_path):
    results = pd.read_csv(
        timeseries_path, low_memory=False).sort_values(by="extract_day").reset_index(drop=True)
    header = results.columns
    concept_risk_columns = [x for x in header if x.endswith('_score')]
    reordered_columns = [x for x in header if not x.endswith('_score')]
    reordered_columns.extend(concept_risk_columns)
    timeseries = pd.DataFrame(results, columns=reordered_columns)
    timeseries = timeseries.replace({"\\N": np.nan})
    return timeseries


def display_granular(datapath):
    """
    A small function which takes a filepath to Textreveal output and displays 
    """
    try:
        results = parse_mts(read_data(datapath))
        # We only need some columns and in a specific order
        results = results.reindex(
            columns=['id', 'site', 'sentence_id', 'type', 'matches', 'concepts', 'text', 'negative', 'positive',
                     'neutral', 'polarity', 'polarity_exp', 'joy', 'anticipation', 'trust', 'anger', 'fear',
                     'surprise', 'sadness', 'language', 'extract_date', 'site_type'
                     ])
        results = results.rename(columns={'id': 'document_id'})

        results['matches'] = results['matches'].fillna(value='')
        results['matches'] = results['matches'].apply(join_objects)
        results = results.sort_values(
            by='matches', ascending=False).reset_index(drop=True)
        return results
    except IndexError:
        return IndexError("There is not data to display")


def display_documents(datapath):
    """
    A small function which takes a filepath to Textreveal output and displays 
    """
    try:
        results = parse_mts(read_data(datapath))
        results = results.groupby("url")

        results = results.agg(
            matched_entities=('matches', lambda x: join_objects(set.union(*x))),
            matched_concepts=('concepts', lambda x: list(x.values)[0]), # All documents that share a same url, share the same concepts
            title=('title', 'max'),
            text=('text', lambda x: '\n'.join(x)),
            site_type=('site_type', 'max'),
            language=('language', 'max'),
            country=('country', 'max'),
            max_polarity=('polarity', 'max'),
            median_polarity=('polarity', 'median'),
            min_polarity=('polarity', 'min'),
            average_polarity=('polarity', 'mean'),
        )
        results = results.reset_index()
        results = results.reindex(
            columns=['matched_entities', 'matched_concepts', 'title', 'text', 'site_type', 'language', 'country', 'url', 'max_polarity',
                     'median_polarity', 'min_polarity', 'average_polarity'])
        return results
    except IndexError:
        return IndexError("There is not data to display")

def display_row(documents, index: int):
    """
    Check if index is valid then display the corresponding row in the document
    """
    if 0 > index or index > len(documents) - 1:
        raise Exception("Invalid index, should be in [0,%d]" % (len(documents)-1))

    document = documents.loc[[index]].T
    res = document.reset_index().rename(columns={"index":"fields",index:"values"})
    res = df_align_left(res)
    display(res.hide_index())


def df_align_left(df):
    df = df.style.set_properties(**{'text-align': 'left'})
    df = df.set_table_styles(
        [dict(selector='th', props=[('text-align', 'left')])])
    return df


def flatten_dict(x):
    return ast.literal_eval(str(x))


def flatten_deep_list(_list):
    if len(_list) == 1:
        if type(_list[0]) == list:
            result = flatten_deep_list(_list[0])
        else:
            result = _list
    elif type(_list[0]) == list:
        result = flatten_deep_list(_list[0]) + flatten_deep_list(_list[1:])
    else:
        result = [_list[0]] + flatten_deep_list(_list[1:])
    return result


def join_objects(x):
    x = ','.join(str(v) for v in list([y for y in x]))
    return(x)


def clean_to_list(x):
    """
    A function useful to clean consecutive comma appearing with the concatenation of empty string 
    """
    x = x.tolist()
    x = list(chain.from_iterable(x))
    transformed_x = list(set(x))
    return(transformed_x)


def parse_raw_output(output):
    """
    This function parses & reformats Textreveal output
    """

    df = pd.json_normalize(output,  record_path=[['disambiguation', 'sentences']],
                           meta=['id', 'qscore', 'language', 'url', 'thread', 'extract_date'])
    thread_df = pd.json_normalize(df['thread'].apply(flatten_dict).tolist())
    df.drop(['thread'], axis=1, inplace=True)
    results_df = df.join(thread_df).drop_duplicates(
        subset=['id', 'sentence_id', 'type'])
    results_df = results_df.rename(
        columns={i: re.sub("results.", "", i).strip() for i in results_df.columns})
    return results_df


def match_extractor(x, col):
    """A small function to extract values from nested json objects

    Parameters
    ----------
        x: the pd.Series object
        col: the column name to be extracted

    Returns
    --------
        A pd.Series object with extracted values
    """

    try:
        x = x.get(col, np.nan)
    except AttributeError as e:
        return None
    return x


raw_indicators = ['anger', 'anticipation', 'fear', 'joy', 'sadness',
                  'surprise', 'trust', 'negative', 'neutral', 'positive', 'similarity']
operands_map = {'min': np.min,
                'max': np.max,
                'mean': np.mean,
                'median': np.median}


def extract_entity(x, col):
    """A small function to extract values from nested json objects

    Parameters
    ----------
        x: the pd.Series object
        col: the column name to be extracted

    Returns
    --------
        A pd.Series object with extracted values
    """

    try:
        x = x.get(col, np.nan)
    except AttributeError:
        return None
    return x


def extract_concept_risks(data):
    """
    This function extracts concept risks data from Textreveal output
    """
    try:

        data = {x+'_score': 1 for x in data.keys()}
        return pd.Series(data, dtype='int64')
    except AttributeError:
        return np.nan


def combine_mention_entity(data):
    try:
        result = [x for x in data.keys()]
        return result
    except AttributeError:
        return np.nan


def parse_mts(results):
    """
    This function parses & reformats Textreveal output
    """

    r_path_col = ['sentences']
    if 'disambiguation' in results[0].keys():
        r_path_col.append('disambiguation')

    r_path_col.reverse()

    for i in results:
        for k in ['entities', 'id']:
            for x in i['sentences']:
                if k in x.keys():
                    del x[k]
    df = pd.json_normalize(results,  record_path=[r_path_col],
                           meta=['id', 'qscore', 'language', 'url', 'thread', 'extract_date', 'concepts', 'entities', 'mentions'], errors='ignore')
    thread_df = pd.json_normalize(
        df['thread'].apply(lambda x: flatten_dict(x)).tolist())
    df.drop(['thread'], axis=1, inplace=True)
    results_df = df.join(thread_df).drop_duplicates(
        subset=['id', 'sentence_id', 'type'])
    results_df = results_df.rename(
        columns={i: re.sub("results.", "", i).strip() for i in results_df.columns})

    def merge_matches_in_list(x):
        """
        Merge every mentions/entities into a list of keywords.
        """
        matches: Set[str] = set()

        x_mentions: Optional[Dict[str, Dict[str, int]]] = x.mentions
        x_entities: Optional[Dict[str, List[Dict[str, int]]]] = x.entities

        if x_mentions:
            for word_map in x_mentions.values():
                for entry in word_map:
                    matches.add(entry)

        if x_entities:
            for word_map in x_entities.values():
                for entries in word_map:
                    for entry in entries:
                        matches.add(entry)


        return matches

    # Create a new "matches" column with both mentions and entities
    results_df['matches'] = results_df.apply(merge_matches_in_list, axis=1)

    # As we are are using the matches column to aggregare mentions and entities, we do not need them anymore
    results_df.drop(['mentions', 'entities'], axis=1, inplace=True)

    return results_df


def plot_all_signal(results, metric):
    results.loc[:, 'ewm_sentiments'] = results[metric].ewm(span=7).mean()
    fig = px.line(results, x="extract_day",
                  y='ewm_sentiments', color='entity', title='')
    fig.show()


def plot_indicators(results, entity, agg_metric='mean', indicators_list=['anger',
                                                                         'anticipation',
                                                                         'fear',
                                                                         'joy',
                                                                         'sadness',
                                                                         'surprise',
                                                                         'trust',
                                                                         'negative',
                                                                         'neutral',
                                                                         'positive']):
    """
    Plot results as a TimeSeries.
    Useful to plot a list of indicators using an aggregation function on a given entity
    """

    metrics = [agg_metric + '_' + i for i in indicators_list]
    if results is not None:
        if entity:
            results = results.query(f"entity=='{entity}'")
        agg_object = {i: 'mean' for i in metrics}
        agg_object.update({'volume_document': 'sum'})
        filtered_columns = ["extract_day", "volume_document"]
        filtered_columns.extend(metrics)
        results = results[filtered_columns].drop_duplicates()\
            .groupby(["extract_day"])\
            .agg(agg_object)\
            .reset_index()
        results[metrics] = results[metrics].ewm(span=7).mean()

        volume_bar_chart = go.Bar(x=results['extract_day'],
                                  y=results['volume_document'],
                                  yaxis='y',
                                  marker_color='rgba(150, 200, 250, 0.4)',
                                  name="volume_document")
        data = [volume_bar_chart]

        for metric in metrics:
            sent_scatter = go.Scatter(x=results['extract_day'],
                                      y=results[metric],
                                      yaxis='y2',
                                      name=metric)
            data.append(sent_scatter)

        rangeselector = dict(visible=True,
                             x=0,
                             y=0.9,
                             bgcolor='rgba(150, 200, 250, 0.4)',
                             font=dict(size=13),
                             buttons=list([
                                     dict(count=1,
                                          label='reset',
                                          step='all'),
                                     dict(count=1,
                                          label='1yr',
                                          step='year',
                                          stepmode='backward'),
                                     dict(count=1,
                                          label='1 mo',
                                          step='month',
                                          stepmode='backward'),
                                     dict(step='all')
                             ]))

        layout = go.Layout(xaxis=dict(rangeselector=rangeselector),
                           yaxis=dict(domain=[0, 0.2], showticklabels=False),
                           yaxis2=dict(domain=[0.2, 0.8]),
                           plot_bgcolor='rgb(250, 250, 250)'
                           )
        data.reverse()
        fig = go.Figure(data=data,
                        layout=layout)

        fig.update_layout()
        fig.show()
    else:
        print("No Data retrieved. Please try again")


def plot_concept_risks(results, entity, concepts_list=['esg']):
    """
    Plot results as a TimeSeries.
    Useful to plot a list of indicators using an aggregation function on a given entity
    """

    metrics = concepts_list
    if results is not None:
        if entity:
            filtered_results = results.query(f"entity=='{entity}'").copy()
        filtered_results[metrics] = filtered_results[metrics].ewm(
            span=7).mean()

        volume_bar_chart = go.Bar(x=filtered_results['extract_day'],
                                  y=filtered_results['volume_document'],
                                  yaxis='y',
                                  marker_color='rgba(150, 200, 250, 0.4)',
                                  name="volume_document")
        data = [volume_bar_chart]

        for metric in metrics:
            sent_scatter = go.Scatter(x=filtered_results['extract_day'],
                                      y=filtered_results[metric],
                                      yaxis='y2',
                                      name=metric)
            data.append(sent_scatter)

        rangeselector = dict(visible=True,
                             x=0,
                             y=0.9,
                             bgcolor='rgba(150, 200, 250, 0.4)',
                             font=dict(size=13),
                             buttons=list([
                                     dict(count=1,
                                          label='reset',
                                          step='all'),
                                     dict(count=1,
                                          label='1yr',
                                          step='year',
                                          stepmode='backward'),
                                     dict(count=1,
                                          label='1 mo',
                                          step='month',
                                          stepmode='backward'),
                                     dict(step='all')
                             ]))

        layout = go.Layout(xaxis=dict(rangeselector=rangeselector),
                           yaxis=dict(domain=[0, 0.2], showticklabels=False),
                           yaxis2=dict(domain=[0.2, 0.8]),
                           plot_bgcolor='rgb(250, 250, 250)'
                           )
        data.reverse()
        fig = go.Figure(data=data,
                        layout=layout)

        fig.update_layout()
        fig.show()
    else:
        print("No Data retrieved. Please try again")


def plot_signal_mts(results, metric, entity):
    """
    Plot results as a TimeSeries.
    Useful to signal for a given entity on a given indicator
    """
    if results is not None:

        if entity:
            results = results.query(f"entity=='{entity}'")

        results = results[["extract_day", "volume_document", metric]].drop_duplicates()\
            .groupby(["extract_day"])\
            .agg({metric: "mean", 'volume_document': 'sum'})\
            .reset_index()
        results.loc[:, 'ewm_sentiments'] = results[metric].ewm(span=7).mean()

        sent_scatter = go.Scatter(x=results['extract_day'],
                                  y=results[metric],
                                  yaxis='y2',
                                  name=metric)

        sent_scatter_ewm = go.Scatter(x=results['extract_day'],
                                      y=results['ewm_sentiments'],
                                      yaxis='y2',
                                      name="Moving Average - 7 days period"
                                      )

        volume_bar_chart = go.Bar(x=results['extract_day'],
                                  y=results['volume_document'],
                                  yaxis='y',
                                  marker_color='rgba(150, 200, 250, 0.4)',
                                  name="Volume")

        data = [sent_scatter, sent_scatter_ewm, volume_bar_chart]

        rangeselector = dict(visible=True,
                             x=0,
                             y=0.9,
                             bgcolor='rgba(150, 200, 250, 0.4)',
                             font=dict(size=13),
                             buttons=list([
                                 dict(count=1,
                                      label='reset',
                                      step='all'),
                                 dict(count=1,
                                      label='1yr',
                                      step='year',
                                      stepmode='backward'),
                                 dict(count=1,
                                      label='1 mo',
                                      step='month',
                                      stepmode='backward'),
                                 dict(step='all')
                             ]))

        layout = go.Layout(xaxis=dict(rangeselector=rangeselector),
                           yaxis=dict(domain=[0, 0.2], showticklabels=False),
                           yaxis2=dict(domain=[0.2, 0.8]),
                           plot_bgcolor='rgb(250, 250, 250)'
                           )

        fig = go.Figure(data=data,
                        layout=layout)

        fig.update_layout()
        fig.show()
    else:
        print("No Data retrieved. Please try again")


if __name__ == '__main__':  # pragma: no cover
    pass
