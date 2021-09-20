#Download Spacy
#!python -m spacy download en_core_web_sm
import spacy #load spacy
import requests
import pandas as pd
import os
import json
import re
from time import sleep
import warnings
warnings.filterwarnings("ignore")
import nltk
from textblob import TextBlob
from tqdm import tqdm
from settings import host,token
from spacy.lang.en.stop_words import STOP_WORDS
import string
import numpy as np
nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])


def existInKg(keyword_of_interest):
    '''Check if a keyword_of_interest exist in the kg'''
    
    #max_results size must be less than or equal to: [10000]
    dfkg=pd.DataFrame()
    try:
        endpoint = f'{host}/api/2.0/kg/candidates'
        
         
        payload ='{"keyword_of_interest":"'+keyword_of_interest+'", "max_results": 20,  "source": "general_kg"}'

        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token['access_token']}

        response = requests.request("POST", endpoint, headers=headers, data = payload)


        kglist=response.json()
        
        dfkg=pd.DataFrame(kglist)
        dfkg['company name']= keyword_of_interest
    except:
        pass
    
    return dfkg



def normalize(comment, lowercase, remove_stopwords):
        
    #stops =stopwords.words("english")
    if lowercase:
        comment = comment.lower()
    # convert the text to a spacy document
    comment = nlp(comment) # all spacy documents are tokenized. You can access them using document[i]
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in STOP_WORDS):
                lemmatized.append(lemma)
    return " ".join(lemmatized)



def cleanComment(dfkg):
    
    """Remove stopWords and punctuation"""
    #Remove StopWords
    dfkg['ItemDescription_After_Clean'] = dfkg['itemDescription'].apply(normalize, lowercase=True, remove_stopwords=True)
    #Remove pnuctuation
    dfkg["ItemDescription_After_Clean"] = dfkg['ItemDescription_After_Clean'].apply(lambda text: 
                                      " ".join(token.lemma_ for token in nlp(text) 
                                               if not token.is_punct))
    return dfkg


def searchInKg(listkeyword_of_interest):
     
    """ Search entities in kg by its company name"""    
    finaldf= pd.DataFrame()
    for keyword_of_interest in tqdm(listkeyword_of_interest):   
        dfkg=existInKg(keyword_of_interest)
        
        if dfkg.empty:
            print('empty',keyword_of_interest)
            datadict=dict()
            datadict['company name']= keyword_of_interest
            dfdata=pd.DataFrame([datadict])
            pass
        else:
            dfkg=cleanComment(dfkg)
            dfkg=dfkg.reset_index(drop=True)

        #Keeping just the'itemDescription'and'label' related to the searched "keyword_of_interest"
        wordsinclude=["producer", "company", "corporate", "establishment", "group",
               "association", "firm", "partnership", "venture", "entreprise", "inc", "services","team","charity",
               "constitution","establishement","creation","endowment","guild","inaugration","institute","institution",
               "organization","plantation","settlement","setup","society","trusteeship","business","subsidiaries","subsidiary"]
        wordsexclude=["film","music","fans","book","unincorporated","beatles","pie","juice","Beatles","cultivar","sea","eat","eating","table","plants","album","actor","play","trial","family","football","rapper","DJ","single","song","girl","boy","girls","boys"]

        data=list()
        
        for i in range(len(dfkg)):

            datadict=dict()
            
            if dfkg['label'][i].lower()==keyword_of_interest.lower():
                    datadict['itemDescription']=dfkg['itemDescription'][i]
                    datadict['item_id']=dfkg['item_id'][i]
                    datadict['label']=dfkg['label'][i]
                    datadict['company name']= keyword_of_interest

                    data.append(datadict) 
            else:
                doc=nlp(dfkg['ItemDescription_After_Clean'][i])

                for token in doc:

                    if (token.text.lower() in wordsinclude):       
                        datadict['itemDescription']=dfkg['itemDescription'][i]
                        datadict['item_id']=dfkg['item_id'][i]
                        datadict['label']=dfkg['label'][i]
                        datadict['company name']= keyword_of_interest
                if datadict !={} :           
                    data.append(datadict) 
          
        dfdata=pd.DataFrame(data)
        
        indexToDrop=list()
        if 'itemDescription' in dfdata.columns:
            for i in range(len(dfdata)):    
                for word in wordsexclude:
                    if word in dfdata['itemDescription'][i]:
                        indexToDrop.append(i)                           
        dfdata=dfdata.drop(dfdata.index[indexToDrop]) 
        dfdata=dfdata.reset_index(drop=True)

        finaldf=finaldf.append(dfdata,ignore_index=True)    
    return finaldf

def cleanName(lstring):
    """
    Clean list of input companies names
    """   
    #remove [] and ()
    lstring=lstring.apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x))
    stopfile="Preprocess/stop_list.csv"
    dfstop=pd.read_csv(stopfile)
    stop=[x.lower() for x in dfstop["stop"]]
    lstring=lstring.apply(lambda x: x.lower())
    lstring=lstring.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    lstring=lstring.apply(lambda x: x.title())
    lstring = lstring.str.strip().str.replace('[-,\/()]', ' ', regex=True).str.replace(' +', ' ', regex=True).str.replace('.', '', regex=False)
    lstring = lstring.str.strip().str.replace("'", '', regex=False)
    lstring = lstring.str.strip().str.replace(":", '', regex=False)
    lstring = lstring.str.strip()
    lstring.replace(np.nan,"",inplace=True)
    return lstring

def searchListCompaniesInKg(df):
    
    """
    Search list of companies names in KG
    """
    #drop duplicates in df
    df.drop_duplicates(keep=False, inplace=True)
    finalresult=pd.DataFrame()
    df['name']=cleanName(df['name'])
    listkeyword_of_interest=list(df['name'])
    finalresult=searchInKg(listkeyword_of_interest)
    missingcompanies=[]
    for keyword in listkeyword_of_interest:
        if keyword not in list(finalresult['company name']):
            missingcompanies.append(keyword)
    dfmissed=pd.DataFrame({'company name':missingcompanies})
    final=pd.concat([finalresult,dfmissed],axis=0).reset_index(drop=True)
    final=final.sort_values(by='company name')
    final=final[['company name','label','item_id','itemDescription']]
    final=final.fillna('')
    return final

 
def availableEntitiesProperties(listentities_ids):

    endpoint = f'{host}/api/2.0/kg/entities/bulk'

    payload = '{"entities_ids": '+str(json.dumps(listentities_ids))+',"source": "general_kg","language":"english"}'
    #payload = '{"entities_ids": '+str(json.dumps(listentities_ids))+',"properties_list": ["chief_executive_officer","domain","founders","legal_name","subsidiaries","ticker"]}'
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token['access_token']}

    response = requests.request("POST", endpoint, headers=headers, data = payload)

    mrequest = response.json()
    return mrequest   


def DataProcess(df):
    """Preprocessing name, website and description"""
    
    #Name preprocessing
    lnames=[]
    k=[]
    for i in tqdm(range(len(df['name']))):
        punctuation=re.compile(r'[?.!)°]|]')
        df['name'][i]=punctuation.sub("",df['name'][i])
        l=re.split('; |\(|\|\[|:|;|,|_|-|/ |\*|\n',df['name'][i])
          
        namelist=[]
        for i in range(len(l)):
            namelist.append('name'+str(i))

        data={}    
        for mykey,myvalue in zip(namelist, l):
            data[mykey]=myvalue
        k.append(data)  
    
    dfnames=pd.DataFrame(k)
    
    #Website preprocessing
    lwebsites=[]
    k=[]
    for i in range(len(df['website'])):
        
        try:
            
            l=re.split('&|&&|:|;|,',df['website'][i])
        except:
            pass
        websitelist=[]
        for i in range(len(l)):
            websitelist.append('website'+str(i))

        data={}    
        for mykey,myvalue in zip(websitelist, l):
            data[mykey]=myvalue        
        k.append(data)  
    
    dfwebsites=pd.DataFrame(k)
    
    #Description preproessing   
    for colname in dfnames.columns:
        for i in range(len(dfnames)):
            lnames=[]
            lrelated=["producer", "company", "corporate", "establishment", "group",
               "association", "firm", "partnership", "venture", "entreprise", "inc", "services","team","charity",
               "constitution","establishement","creation","endowment","guild","inaugration","institute","institution",
               "organization","plantation","settlement","setup","society","trusteeship","business","subsidiaries","subsidiary"]
            try:
                lnames=dfnames[colname][i].split(" ")
                for j in lnames:
                    if j.lower() not in lrelated and j.lower() in df['description'][i].lower():
                        df['description'][i]=df['description'][i].lower().replace(j.lower(),'')
            except:
                pass
    dfdescription= pd.DataFrame(df['description'])
    
    #Extracting nouns from descriptions
    lnouns=[]
    k=[]
    for i in range(len(df['description'])):
        
        try:
            #remove punctuation from description
            sent=df['description'][i].translate(str.maketrans('', '', string.punctuation.replace('.','')))
            #Extracting all nouns from description
            l=list(TextBlob(sent).noun_phrases)
        except:
            pass
        nounslist=[]
        for i in range(len(l)):
            nounslist.append('noun'+str(i))

        data={}    
        for mykey,myvalue in zip(nounslist, l):

            data[mykey]=myvalue
        
        k.append(data)  
    
    dfnouns=pd.DataFrame(k)
    
    dfprocess=pd.concat([df['id'],dfnames,dfwebsites,dfnouns,dfdescription,df['crunchbase_id']],axis=1)
    return dfprocess
    
def CreateSearchedCompanies(file_input_df):
  
    #drop duplicates in dataframe
    file_input_df.drop_duplicates(keep=False, inplace=True)
    
    #df0: dataframe after data preprocessing
    df0=DataProcess(file_input_df)
    
    #Prepare the keywords list
    listkeywords=[]
    for colname in df0.columns: 
        
        if 'name' in colname or 'website' in colname or 'noun' in colname:
            listkeywords.append(df0[colname].astype(str))
    
        
    #print(type(pd.Series(df0[colname].dropna().unique()).astype(str)))  
    #listkeywords =listkeywords.dropna().unique().tolist()
    dfkeywords = (pd.concat(listkeywords, axis=1)
              .apply(lambda x: ', '.join(x.dropna()), axis=1))
    #Building the request with the required parameters for each entity(entity_of_interest,keywords,context)
    entities_df=pd.DataFrame()
    entities_df = entities_df.assign(entity_of_interest=df0['id'],keywords =dfkeywords,context=df0['description'])
    
    
    entities_df['keywords'] = pd.Series(entities_df['keywords'].str.strip().str.split(','))
    
    for i in range(len(entities_df)):
        entities_df['keywords'][i]=[x for x in (entities_df['keywords'][i]) if (str(x) != ' nan')]
        
    entities_df['context'] = entities_df['context'].str.strip()
    return entities_df

def displayBy_SearchKey1(totaldata,df,searchkey):
    displaydf=pd.DataFrame()
    try:
        # result is the data of the companies found in kg (the output of the function searchListCompaniesInKg(df) )
        result=searchListCompaniesInKg(df)

        listcompaniesids=[]
        for i in range(len(totaldata)):
            #ch1=json.loads(totaldata['entities'][i].replace("'","\"")) # use this if reading totaldata from csv 
            # if reading totaldata directly from display_mts(output_path)
            ch1=totaldata['entities'][i]
            #it could be "item_id"(with the method search in kg) or "entity_of_interest"(with the method search without kg)

            for key, value in ch1.items():
                listcompaniesids.append(key)
        totaldata["company_id"]=listcompaniesids
        dfmerge=totaldata.merge(result, left_on='company_id', right_on='item_id', how='left', indicator=False)
        displaydf=dfmerge[dfmerge['company name'].str.lower()==searchkey.lower()]
        if displaydf.empty:
            displaydf=dfmerge[dfmerge['company_id']==searchkey]
            if displaydf.empty:
                print("The entred search key does not exist")

    except:  
        pass     
    return displaydf

def displayBySearchKey(totaldata,searchkey):
    """
    Display data by company name or company_id
    """
    displaydf=pd.DataFrame()
    listcompaniesnames=[]
    listcompaniesids=[]
    for i in range(len(totaldata)):
    #ch1=json.loads(totaldata['entities'][i].replace("'","\"")) # use this if reading totaldata from csv 
    # if reading totaldata directly from display_mts(output_path)
        ch1=totaldata['entities'][i]
    #it could be "item_id"(with the method search in kg) or "entity_of_interest"(with the method search without kg)

        for key, value in ch1.items():
            listcompaniesids.append(key)
            listcompaniesnames.append((pd.DataFrame(value).columns)[0])
                
    totaldata["company_id"]=listcompaniesids
    totaldata["company name"]=listcompaniesnames
    
        #dfmerge=totaldata
    displaydf=totaldata[totaldata['company name'].str.lower()==searchkey.lower()]
    if displaydf.empty:
        displaydf=totaldata[totaldata['company_id']==searchkey]
        if displaydf.empty:
            print("The entred search key does not exist")

       
    return displaydf 

def displayBy_SearchKey2(totaldata,df,searchkey):
    displaydf=pd.DataFrame()
    try:
       
        listcompaniesids=[]
        for i in range(len(totaldata)):
            #ch1=json.loads(totaldata['entities'][i].replace("'","\"")) # use this if reading totaldata from csv             
            ch1=totaldata['entities'][i] # if reading totaldata directly from display_mts(output_path)
           

            for key, value in ch1.items():
                listcompaniesids.append(key)
        totaldata["company_id"]=listcompaniesids
        dfmerge=totaldata.merge(df, left_on='company_id', right_on='id', how='left', indicator=False)
        displaydf=dfmerge[dfmerge['name'].str.lower()==searchkey.lower()]
        if displaydf.empty:
            displaydf=dfmerge[dfmerge['company_id']==searchkey]
            if displaydf.empty:
                    print("The entred search key does not exist")

    except:  
        pass     
    return displaydf