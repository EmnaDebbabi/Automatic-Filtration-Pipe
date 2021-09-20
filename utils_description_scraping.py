from googlesearch import search
import requests
import re
from bs4 import BeautifulSoup
import googlesearch
from itertools import chain
from nltk.corpus import wordnet
import string
from tqdm import tqdm
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
import sys
import pandas as pd
import numpy as np
import time
import random
import datetime 
import logging
import requests
import urllib.request

#scraping for one company
def compDescScrap(searched_company):
    """Scraping one company description"""
    
    lfinalsent=[]
    searched_company=searched_company.strip()
    searched_company=searched_company.capitalize()
    searched_company=str(searched_company)
    print('searched_company',searched_company)
    #linkElements=googlesearch.search(searched_company,num_results=50, lang="en")
    #print('***heere',linkElements)
    linkElements=[]
    query = searched_company
    for url in search(searched_company, stop=30):
        linkElements.append(url)
    linkElements=list(set(linkElements))
    print('****',linkElements)
    print(linkElements)
    for url in tqdm(linkElements):
        try:
            result=requests.get(url)
            #result= urllib.request.urlopen(url)
            #scraping data
            #doc=BeautifulSoup(result.text,"html.parser")
            doc = BeautifulSoup(result.text, "html.parser")
            #doc=doc.title.text
            #print('-----',doc)
            #search for the sentences that contain the pattern of the searched company
            tags=[]
            tags1=doc.find_all(text=re.compile(searched_company+" is [\w.-]+"))
            tags2=doc.find_all(text=re.compile(searched_company+" [\w.-]+ is [\w.-]+"))

            if tags1:
                for sent in tags1:
                    if sent not in tags:
                        tags.append(sent)
            if tags2:
                for sent in tags1:
                    if sent not in tags:
                        tags.append(sent)
                    
            #the texts collected
            sentences = []
            for x in tags:
                sentences.append(str(x))

            #Text preprocessing
            #Split the otained texts into sentences
            lsentsplit=[]
            for sent in sentences:
                if sent.lower().startswith(searched_company.lower()):
                    lsentsplit.append([sent])
                else:    
                    lsentsplit.append(sent.split("."))
            sentsplit=[item.lower() for sublist in lsentsplit for item in sublist]
            # Check if the sentence is about a company
            #Dictionary of company synonyms
            complist = ["producer", "company", "corporate", "establishment", "group","corporation","multinational"," produce",
            "association", "firm", "partnership", "venture", "entreprise", "inc", "services","team","charity",
            "constitution","establishement","creation","endowment","guild","inaugration","institute","institution",
            "organization","plantation","settlement","setup","society","trusteeship","business","subsidiaries","subsidiary"]
            listsyn=[]
            t=[]
            t.append(complist)
            for syn in complist:
                #wordnet to search other synonymes
                synonyms = wordnet.synsets(syn)
                t.append(list(set(chain.from_iterable([word.lemma_names() for word in synonyms]))))
            listsyn = set([item.lower() for sublist in t for item in sublist])
            #sentencescomp contains only sentences about companies and its related synonymes
            sentencescomp=[]
            for i in range(len(sentences)):
                for word in (sentences[i].split()):
                    if word.strip().lower() in listsyn:
                        #strip to remove white spaces
                        sentencescomp.append(sentences[i].strip())           
            # remove punctuation and capitalize the sentence first letter
            sentcompclean=[]
            for sent in sentencescomp:
                sentcompclean.append(sent.translate(str.maketrans('', '', string.punctuation.replace('.',''))).capitalize())
            #keeping only the sentences that starts with the company name
            finalsent=[]
            
            for sent in sentcompclean:    
                if sent.startswith(searched_company):
                    finalsent.append(sent)
                    #print(finalsent)       
            if finalsent:
                #Automated proofreading and grammar checking
                # keeping only grammaticaly correct sentences
                
                for i in range(len(finalsent)):
                    text=finalsent[i]
                    
                    matches = tool.check(text)
                    if not matches:
                        
                        lfinalsent.append(text)
                    else:
                        
                        lfinalsent.append(tool.correct(text))
            else:
                finalsent=['']         
                
                
            time.sleep(1)
        except Exception as e: 
            print(e)
            
            continue
    #the final description of a one given company
    #lfinalsent1=[item for sublist in lfinalsent for item in sublist]
    descpcomp1='.'.join(lfinalsent)
    return descpcomp1
def DescriptionScraping (df0):
    """
    Scraping empty companies descriptions list if it  exists 

    Parameters
    ----------
    input_file : dataframe 
        original data with or without companies descriptions

    Returns
    -------
    new_file: dataframe
        new_file: data with full descriptions if exists 
    """

    df=df0.copy().reset_index(drop=True)
    df=df.fillna('') 
    count=0
    for i in range(len(df)):
        try:
            if(df['description'][i]==""):
                print(i)
                df['description'][i]=compDescScrap(df['name'][i])
                count=count+1
                time.sleep(1)
        except:
            
            continue
    return df  
    
def  PPDescScraping(df,description_scrap=True):
    if description_scrap==True:
         return DescriptionScraping(df) 
    else:
        return df