import sys
import pandas as pd
import re
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import time
# generate the matrix of TF-IDF values for each n-grams
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import json
import requests
import pandas as pd
import ipywidgets as widgets
import plotly.express as px
from utils_mts import *
from utils1 import *
from utils_preprocess import *
from getpass import getpass
from tqdm import tqdm


class PPFiltration():
    def __init__(self, names, new_file_path, KeepMostSimilarDescription=False):
        """
        Search and filter companies within the knowlege graph.
        Parameters
        ----------
        files :  
            files :  file/input_file_path: contains all companies to search within the knowledge graph.

        Returns
        -------
        str
            file/new_file_path: contains all companies found or not within the knowledge graph and filtred based on their most similar descriptions
        """
        result = searchListCompaniesInKg(names).sort_values(
            by='company name').reset_index(drop=True)
        if KeepMostSimilarDescription:
            # PPFiltration.DescriptionFiltration(input_file_path,new_file_path)
            if (len(result) == 1):
                return result
            else:

                names1 = self.addNameOnTop(result, names)
                #names1=names.groupby(["company name"]).apply(lambda x :addNameOnTop(searchListCompaniesInKg(x),x))

                try:
                    result = names1.groupby(["company name"]).apply(
                        lambda x: self.KeepMostSimilarDes(x))
                    result = result.reset_index(drop=True)
                    result.to_csv(new_file_path)
                except Exception as e:
                    print(e)
                    result.to_csv(new_file_path)
        else:
            result.to_csv(new_file_path)

    # cleans a string and generates all n-grams in this string
    def ngrams(self, string, n=3):
        string = re.sub(r'[,-./]|\sBD', r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    def awesome_cossim_top(self, A, B, ntop, lower_bound=0):
        # force A and B as a CSR matrix.
        # If they have already been CSR, there is no overhead
        A = A.tocsr()
        B = B.tocsr()
        M, _ = A.shape
        _, N = B.shape

        idx_dtype = np.int32

        nnz_max = M*ntop

        indptr = np.zeros(M+1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)

        return csr_matrix((data, indices, indptr), shape=(M, N))

    # The following code unpacks the resulting sparse matrix.
    # As it is a bit slow, an option to look at only the first n values is added.
    def get_matches_df(self, sparse_matrix, name_vector, top=7):
        non_zeros = sparse_matrix.nonzero()

        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]

        if top:
            nr_matches = top
        else:
            nr_matches = sparsecols.size

        left_side = np.empty([nr_matches], dtype=object)
        right_side = np.empty([nr_matches], dtype=object)
        ind = np.empty([nr_matches], dtype=object)
        similairity = np.zeros(nr_matches)

        for index in range(0, nr_matches):
            
            left_side[index] = name_vector[sparserows[index]]
            right_side[index] = name_vector[sparsecols[index]]
            similairity[index] = sparse_matrix.data[index]
            ind[index] = sparsecols[index]

        return pd.DataFrame({'left_side': left_side,
                            'right_side': right_side,
                             'similairity': similairity, 'ind': ind})

    def KeepMostSimilarDes(self, names):

        names = names.reset_index(drop=True)
        names = names.fillna("")

        if len(names['itemDescription']) <= 1:
            names0 = names
        elif ((len(list(names['itemDescription'])) == 2) and ('' in list(names['itemDescription']))):

            names0 = names.drop([0]).reset_index(drop=True)

        else:

            nullindex = []
            nullindex = list(names[names['itemDescription'] == ''].index)

            dfnullindex = pd.DataFrame()
            if nullindex:

                dfnullindex = names.iloc[nullindex].reset_index(drop=True)
                names = names.drop(nullindex).reset_index(drop=True)

            if (len(names) == 2):

                names1 = names.drop([0]).reset_index(drop=True)
                names0 = pd.concat([names1, dfnullindex],
                                   axis=0).reset_index(drop=True)
            else:

                company_names = names['itemDescription']

                # generate the matrix of TF-IDF values for each n-grams
                vectorizer = TfidfVectorizer(min_df=1, analyzer=self.ngrams)
                tf_idf_matrix = vectorizer.fit_transform(company_names)
                # The following code runs the optimized cosine similarity function.
                # It only stores the top 20(10) most similar items, and only items with a similarity above 0 (0.8)
                t1 = time.time()
                matches = self.awesome_cossim_top(
                    tf_idf_matrix, tf_idf_matrix.transpose(), 20, 0)
                t = time.time()-t1
                print("SELFTIMED:", t)
                matches_df = self.get_matches_df(matches, company_names, top=7)

                # Remove all exact matches
                matches_df = matches_df[matches_df['similairity'] < 0.99999]

                # matches_df.sample(20)
                matches_df = matches_df.sort_values(
                    ['similairity'], ascending=False)
                matches_df = matches_df.reset_index(drop=True)

                lindex = []
                for i in range(len(matches_df)):

                    if (matches_df['left_side'][i] not in names['itemDescription'][0] and matches_df['right_side'][i] not in names['itemDescription'][0]):
                        lindex.append(i)
                if lindex:

                    matches_df = matches_df.drop(lindex).reset_index(drop=True)

                names1 = names.drop([0]).reset_index(drop=True)

                lindex = []
                for i in range(len(names1)):

                    if (names1['itemDescription'][i] not in list(matches_df['left_side'])+list(matches_df['right_side'])):
                        lindex.append(i)
                if lindex:

                    names1 = names1.drop(lindex).reset_index(drop=True)

                if len(names1) > 1:

                    lindex = []
                    for i in range(len(names1)):

                        if ((names1['itemDescription'][i] != matches_df['left_side'][0]) and (names1['itemDescription'][i] != matches_df['right_side'][0])):
                            lindex.append(i)
                    if lindex:

                        names1 = names1.drop(lindex).reset_index(drop=True)

                if nullindex:

                    names0 = pd.concat([names1, dfnullindex],
                                       axis=0).reset_index(drop=True)
                else:

                    names0 = names1

        return names0

    def addNameOnTop(self, result, df1):

        df1['company name'] = df1['name']
        df1 = df1.rename(
            columns={"id": "item_id", "description": "itemDescription", "website": "label"})
        result = result.reset_index(drop=True)
        df1 = df1.reset_index(drop=True)

        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
        data_all = []
        listnames = []
        for i in range(len(df1)):
            for j in range(len(result)):
                if ((df1['company name'][i].strip() == result['company name'][j].strip()) and (len(result) > 1) and (df1['company name'][i] not in listnames)):

                    initialname = df1[df1['company name'] ==
                                      df1['company name'][i]].reset_index(drop=True)
                    listnames.append(df1['company name'][i])
                    resultname = result[result['company name'] ==
                                        result['company name'][j]].reset_index(drop=True)

                    df2 = pd.concat([initialname, resultname],
                                    axis=0).reset_index(drop=True)

                    df2 = df2[['item_id', 'company name',
                               'itemDescription', 'name', 'label']]
                    df2.fillna('')
                    df3 = pd.concat([df3, df2], axis=0)

                elif (len(result) == 1):

                    df3 = result.reset_index(drop=True)

        return df3.reset_index(drop=True)

    def existInKg(self, keyword_of_interest):
        '''Check if a keyword_of_interest exist in the kg'''

        # max_results size must be less than or equal to: [10000]
        dfkg = pd.DataFrame()
        try:
            endpoint = f'{host}/api/2.0/kg/candidates'

            payload = '{"keyword_of_interest":"'+keyword_of_interest + \
                '", "max_results": 20,  "source": "general_kg"}'

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

    def searchInKg(self, listkeyword_of_interest):
        """ Search entities in kg by its company name"""

        finaldf = pd.DataFrame()
        for keyword_of_interest in tqdm(listkeyword_of_interest):

            dfkg = existInKg(keyword_of_interest)

            if dfkg.empty:
                print('empty', keyword_of_interest)
                datadict = dict()
                datadict['company name'] = keyword_of_interest
                datadict['label'] = ''
                datadict['item_id'] = ''
                datadict['itemDescription'] = ''
                #datadict.columns=({'company name','label','item_id','itemDescription'})
                dfdata = pd.DataFrame([datadict])

                pass

            else:
                dfkg = cleanComment(dfkg)
                dfkg = dfkg.reset_index(drop=True)

                # Keeping just the'itemDescription'and'label' related to the searched "keyword_of_interest"
                wordsinclude = ["producer", "company", "corporate", "establishment", "group",
                                "association", "firm", "partnership", "venture", "entreprise", "inc", "services", "team", "charity",
                                "constitution", "establishement", "creation", "endowment", "guild", "inaugration", "institute", "institution",
                                "organization", "plantation", "settlement", "setup", "society", "trusteeship", "business", "subsidiaries", "subsidiary"]
                wordsexclude = ["film", "music", "fans", "band", "book", "unincorporated", "beatles", "pie", "juice", "Beatles", "cultivar", "sea", "eat", "eating",
                                "table", "plants", "album", "actor", "play", "trial", "family", "football", "rapper", "DJ", "single", "song", "girl", "boy", "girls", "boys"]

                data = list()

                for i in range(len(dfkg)):

                    datadict = dict()

                    if dfkg['label'][i].lower() == keyword_of_interest.lower():
                        datadict['itemDescription'] = dfkg['itemDescription'][i]
                        datadict['item_id'] = dfkg['item_id'][i]
                        datadict['label'] = dfkg['label'][i]
                        datadict['company name'] = keyword_of_interest

                        data.append(datadict)
                    else:
                        doc = nlp(dfkg['ItemDescription_After_Clean'][i])

                        for token in doc:

                            if (token.text.lower() in wordsinclude):
                                datadict['itemDescription'] = dfkg['itemDescription'][i]
                                datadict['item_id'] = dfkg['item_id'][i]
                                datadict['label'] = dfkg['label'][i]
                                datadict['company name'] = keyword_of_interest
                        if datadict != {}:
                            data.append(datadict)

                dfdata = pd.DataFrame(data)

                indexToDrop = list()
                if 'itemDescription' in dfdata.columns:
                    for i in range(len(dfdata)):
                        for word in wordsexclude:
                            if word in dfdata['itemDescription'][i]:
                                indexToDrop.append(i)
                dfdata = dfdata.drop(dfdata.index[indexToDrop])
                dfdata = dfdata.reset_index(drop=True)

            finaldf = finaldf.append(dfdata, ignore_index=True)
        return finaldf

    def cleanName(self, lstring):
        """
        Clean list of input companies names
        """
        # remove [] and ()
        lstring = lstring.apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x))
        stopfile = "Preprocess/stop_list.csv"
        dfstop = pd.read_csv(stopfile)
        stop = [x.lower() for x in dfstop["stop"]]
        lstring = lstring.apply(lambda x: x.lower())
        lstring = lstring.apply(lambda x: ' '.join(
            [word for word in x.split() if word not in (stop)]))
        lstring = lstring.apply(lambda x: x.title())
        lstring = lstring.str.strip().str.replace(
            '[-,\/()]', ' ', regex=True).str.replace(' +', ' ', regex=True).str.replace('.', '', regex=False)
        lstring = lstring.str.strip().str.replace("'", '', regex=False)
        lstring = lstring.str.strip().str.replace(":", '', regex=False)
        lstring = lstring.str.strip()
        lstring.replace(np.nan, "", inplace=True)
        return lstring

    def searchListCompaniesInKg(df):
        """
        Search list of companies names in KG
        """
        # drop duplicates in df
        df.drop_duplicates(keep=False, inplace=True)
        finalresult = pd.DataFrame()
        df['name'] = cleanName(df['name'])
        listkeyword_of_interest = list(df['name'])

        finalresult = searchInKg(listkeyword_of_interest)
        
        missingcompanies = []

        for keyword in listkeyword_of_interest:
            if keyword not in list(finalresult['company name']):
                missingcompanies.append(keyword)
        dfmissed = pd.DataFrame({'company name': missingcompanies})
        final = pd.concat([finalresult, dfmissed],
                          axis=0).reset_index(drop=True)
        final = final.sort_values(by='company name')
        try:
            final = final[['company name', 'label',
                           'item_id', 'itemDescription']]
        except:
            final['label'] = ''
            final['item_id'] = ''
            final['itemDescription'] = ''
            final.columns = (
                {'company name', 'label', 'item_id', 'itemDescription'})
        final = final.fillna('')
        return final


def main():
    # total arguments
    n = len(sys.argv)
    print("Total arguments passed:", n)
    if (len(sys.argv) == 0):
        logging.error("Error: require 8 arguments, the elasticsearch config "
                      "file, the starting date, the ending date, the ESG risk queries file, "
                      "the company queries file, query params file and the score config file.")
        sys.exit(1)
    if (len(sys.argv) == 1):
        logging.error("Warining: require 8 arguments, the elasticsearch config "
                      "file, the starting date, the ending date, the ESG risk queries file, "
                      "the company queries file, query params file and the score config file.")
        sys.exit(1)
    if len(sys.argv) == 3:
        input_file_path = sys.argv[1]
        new_file_path = sys.argv[2]
    else:
        input_file_path = None
        new_file_path = None

    # Arguments passed
    print("\nName of Python script:", sys.argv[0])
    # Type Arguments passed
    print("\nInput file path:", sys.argv[1])
    # Type Arguments passed
    print("\nOutput file path:", sys.argv[2])
    df = pd.read_csv(input_file_path)
    names = df.copy()
    names['company name'] = names['name']
    names = names.sort_values(by='company name').reset_index(drop=True)
    PPFiltration(names, new_file_path, KeepMostSimilarDescription=True)


if __name__ == '__main__':
    main()
