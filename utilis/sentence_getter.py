class SentenceGetter(object):

    '''
    Create a list of sentences, where each sentence is a list of tokens/words in a document. 
    Parameters
    -----------------
    data: Pandas dataframe. 

    Perform groupby operation on Document Name to group tokens belonging to same document. 
    '''

    def __init__(self,data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s : [(w,t) for w,t in zip(s['token'].values.tolist(),s['category'].values.tolist())]

        self.grouped = self.data.groupby("Doc_name").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence {}".format(self.n_sent)]
            self.n_sent += 1
            return s

        except:
            return None

