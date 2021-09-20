import pandas as pd
import requests
import numpy as np
import os 
import json
from getpass import getpass
import json 
   
    
def existInKg1(keyword_of_interest,host,token,src,max_res=10):
    """Check if a keyword_of_interest exist in the kg"""

    """max_results size must be less than or equal to: [10000]"""
    
    dfkg = pd.DataFrame()
    try:
        endpoint = f'{host}/api/2.0/kg/candidates'
        source = src

        payload = '{"keyword_of_interest":"'+keyword_of_interest + \
            '", "max_results":'+ str(max_res) +', "mode": "pagerank","source": "'+source+'"}'
            

        headers = {'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + token['access_token']}

        response = requests.request(
            "POST", endpoint, headers=headers, data=payload)

        kglist = response.json()

        dfkg = pd.DataFrame(kglist)
        dfkg['company name'] = keyword_of_interest
    except:
        pass

    return dfkg

def searchInKg1(listkeyword_of_interest,host,token,src,**kwargs):

    finaldf = pd.DataFrame(columns = ['company','itemDescription', 'item_id','label','keywords','website','FIGI','ISIN','aliases','labels','chief executive officer','founded by','subsidiary'])

    for keyword_of_interest in listkeyword_of_interest:
        dfkg = existInKg1(keyword_of_interest,host,token,src,**kwargs)
        print(type(dfkg))
        if not (dfkg.empty):
            """Keeping just the'itemDescription'and'label' related to the searched 'keyword_of_interest'"""
            wordsinclude = ["producer", "company", "corporate",
                    "establishment", "group", "business", "corporation",
                    "association", "firm", "venture", "enterprise",
                    "Inc", "services", "maker", "provider", "multinational", "supplier",
                    "establishment", "fellowship", "manufacturer", "venture-capital", "alliance", "partnership"]

            data = list()
            for i in range(len(dfkg)):

                datadict = dict()
                try:
                    datadict['itemDescription'] = dfkg['itemDescription'][i]
                except:
                    continue
                datadict['item_id'] = dfkg['item_id'][i]
                datadict['label'] = dfkg['label'][i]
                try:
                    datadict['ISIN'] = dfkg['ISIN'][i]
                except:
                    continue
                try:    
                    datadict['FIGI'] = dfkg['FIGI'][i]
                except:
                    continue
                datadict['company'] = keyword_of_interest

                data.append(datadict)

            dfdata = pd.DataFrame(data)
            dataToSave = list()
            for i in range(len(dfdata)):
                for word in wordsinclude:
                    if word in dfdata['itemDescription'][i]:
                        dataToSave.append(i)
            dfdata = dfdata.iloc[dataToSave]

            dfdata = dfdata.reset_index(drop=True)
            if not (dfkg.empty):
                finaldf = finaldf.append(dfdata, ignore_index=True)
            else:
                finaldf = finaldf.append(pd.DataFrame([{'item_id':' ','label':' ','itemDescription':' ','company':keyword_of_interest}]), ignore_index=True)
                
        else:
            finaldf = finaldf.append(pd.DataFrame([{'item_id':' ','label':' ','itemDescription':' ','company':keyword_of_interest}]), ignore_index=True)
            
    return finaldf