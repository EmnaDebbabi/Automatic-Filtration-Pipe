import json
import os
import pandas as pd
import plotly.graph_objects as go

from typing import List

def dedup_dicts(items: List[dict]):
    dedupped = [ json.loads(i) for i in set(json.dumps(item, sort_keys=True) for item in items)]
    return dedupped

def count_matches(doc):
    matche_count = sum([len(x['matches']) for x in doc])
    return matche_count

def parse_results(results):
    """
    Parse results from Sanity Check route to have a pandas dataframe
    """
    try: 
        results = dedup_dicts(results)
        results_map = map(lambda x: flatten_json_iterative_solution(x), results)
        results_df = pd.DataFrame.from_records(results_map)
        results_df['datetime'] = pd.to_datetime(results_df['extract_date'], utc=True)
        results_df = results_df[["title","text","site","site_type","extract_date", "language","country",   "mean_anger", "mean_anticipation", "mean_fear", "mean_joy", "mean_negative","mean_neutral", "mean_positive","mean_surprise","mean_trust"]]
        if not os.path.exists('./data/'):
            os.makedirs('./data/')
        results_df.to_csv("./data/sanity_check_df.csv", sep=";", header=True)
        return results_df
    except Exception as e:
        print(e)

def flatten_json_iterative_solution(dictionary):
    """
    Flatten a nested json file
    """
    if dictionary is not None and not isinstance(dictionary.get('title'), list):
        dictionary_nested = dictionary["thread"]
        dictionary.pop("thread")
        dictionary.update(dictionary_nested)
        return dictionary

def plot_signal(results, metric):
    """
    Plot results as a TimeSeries
    """
    if results is not None:

        results = results[["extract_day","volume",metric]].drop_duplicates()\
                            .groupby(["extract_day"])\
                            .agg({metric:"mean", 'volume': 'sum'})\
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
                                  y=results['volume'],
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
