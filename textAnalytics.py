'''
Luke Moles (lmm8fb@virginia.edu) 
DS 5001
11 May 2021
'''


import textAnalytics as ta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import scipy
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from gensim.models import word2vec
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px


''' Tokenize text
INPUT
    docs: fulltext from json, already split into paragraphs 
    OHCO: hierarchy of text
    text_col: name of column that contains text
    remove_pos_tuple: remove tuple after extraction
    ws: use nltk whitespace tokenizer
OUTPUT
    tokenized dataframe
'''
def tokenize(docs, OHCO, text_col='text', remove_pos_tuple=False, ws=False):
    
    # full text to paragraphs
    df = docs.explode(text_col)
    df['para_id'] = df.groupby(OHCO[:1]).cumcount()
    df = df.reset_index().set_index(OHCO[:2])
    
    # paragraphs to sentences
    df = df.text \
        .apply(lambda x: pd.Series(nltk.sent_tokenize(x))) \
        .stack() \
        .to_frame() \
        .rename(columns={0:'sent_str'})
    df.index.names = OHCO[:3]
    
    # sentences to tokens
    def word_tokenize(x):
        if ws:
            s = pd.Series(nltk.pos_tag(nltk.WhitespaceTokenizer().tokenize(x)))
        else:
            s = pd.Series(nltk.pos_tag(nltk.word_tokenize(x)))
        return s
    
    df = df.sent_str \
        .apply(word_tokenize) \
        .stack() \
        .to_frame() \
        .rename(columns={0:'pos_tuple'})
    df.index.names = OHCO[:4]
    
    # info from tuple
    df['pos'] = df.pos_tuple.apply(lambda x: x[1])
    df['token_str'] = df.pos_tuple.apply(lambda x: x[0])
    if remove_pos_tuple:
        df = df.drop('pos_tuple', 1)
        
    # term_str
    df['term_str'] = df['token_str'].str.lower().str.replace('[\W_]', '')
    
    return df

''' Create Vocabulary from Tokens
INPUT
    tokens: token table with OHCO
OUTPUT
    vocabulary table with [counts, stopword, stem, max_part_of_speech]
'''
def vocabularize(tokens):
    
    # extract vocab
    vocab = tokens.term_str.value_counts().to_frame('n') \
        .reset_index().rename(columns={'index':'term_str'})
    vocab.index.name = 'term_id'
    
    # mark stop words
    sw = pd.DataFrame(nltk.corpus.stopwords.words('english'), columns=['term_str'])
    sw = sw.reset_index().set_index('term_str')
    sw.columns = ['dummy']
    sw.dummy = 1
    
    vocab['stop'] = vocab.term_str.map(sw.dummy)
    vocab['stop'] = vocab['stop'].fillna(0).astype('int')
    
    # add stems
    stemmer = PorterStemmer()
    vocab['p_stem'] = vocab.term_str.apply(stemmer.stem)
    
    # max part of speech
    ct = pd.crosstab(tokens.term_str, tokens.pos)
    ct = ct.apply(np.argmax, axis=1) \
        .map(lambda x: ct.columns[x]) \
        .rename('max_pos')
    vocab = vocab.merge(ct, on='term_str')
    
    return vocab




''' Compute TFIDF
INPUT
    tok: tokens table
    voc: vocab table
    bag: desired bag for BOW
    method: tfidf method
    item: name of token level item
    alpha: factor for max tfidf
OUTPUT
    BOW
    vocab modified with [df, idf, dfidf, tfidf_sum]
'''
def compute_tfidf(tok, voc, bag, method='max', item='term_str', alpha=0.4):
    
    # build BOW
    BOW = tok.groupby(bag + ['term_str'])[item].count().to_frame('n').copy()
    BOW['c'] = 1
    # copy vocab while working
    vocab = voc.copy()
    
    # calculate desired tfidf
    D = BOW.groupby(bag).n
    if method == 'n':
        BOW['tf'] = BOW.n
    elif method == 'sum':
        BOW['tf'] = D.apply(lambda x: x / x.sum())
    elif method == 'l2':
        BOW['tf'] = D.apply(lambda x: x / np.sqrt((x**2).sum()))
    elif method == 'max':
        BOW['tf'] = D.apply(lambda x: alpha + (1-alpha) * (x / x.max()))
    elif method == 'log':
        BOW['tf'] = D.apply(lambda x: np.log2(1 + x))
    elif method == 'bool':
        BOW['tf'] = BOW.c
        
    # calculate df and idf
    df = BOW.groupby('term_str').n.count().rename('df')
    vocab = vocab.join(df, on='term_str')
    n_docs = len(D.groups)
    vocab['idf'] = np.log2(n_docs / vocab.df)
    
    # calculate tfidf
    BOW = BOW.reset_index().merge(vocab[['term_str','idf']], on='term_str') \
        .set_index(bag + ['term_str'])
    BOW['tfidf'] = BOW.tf * BOW.idf
    
    # add tfidf aggregate and dfidf to vocab
    colname = f'{method}_tfidf_sum'
    tfidf_sum = BOW.groupby(item)['tfidf'].sum().rename(colname)
    vocab = vocab.merge(tfidf_sum, on='term_str')
    vocab['dfidf'] = vocab.df*vocab.idf
    
    return BOW, vocab





''' Compute metrics from a cluster index
INPUT
    x: cluster index
    model: skl clustering model
    labels: df with indices to match cluster model and year col
OUTPUT
    formatted string for scipy.hierarchy.dendrogram labels
'''
def mean_year(x, model, labels):
    
    # initialize starting values
    trace = [x]
    n_samples = len(model.labels_)
    df = pd.DataFrame(model.children_)
    
    # add parent nodes:
    # node is parent if index >= n_samples
    i = 0
    while i < len(trace):
        node = trace[i]
        if node >= n_samples:
            for parent in df.iloc[trace[i]-n_samples,].values:
                trace.append(parent)
        i += 1
    
    # pull out original obs. and compute metrics
    parent_ids = [parent for parent in trace if parent < n_samples]
    avg_year = np.round(labels.iloc[parent_ids,].year.mean(), 2)
    std = np.round(labels.iloc[parent_ids,].year.std(), 2)
    
    return f'{avg_year} | ({len(parent_ids)})'


''' Plot dendrogram of clustering model
INPUT
    model: skl clustering model
    p: max number of leaf nodes to show
    labels: location of labels
    color_thresh: see scipy.hierarchy.dendrogram
    figsize: see plt.figure
OUTPUT
    None
Extended from skl documentation:
  https://scikit-learn.org/stable/auto_examples/cluster/
  plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-
  cluster-plot-agglomerative-dendrogram-py
''' 
def plot_dendrogram(model, lib, p=30,
                    color_thresh=None, figsize=(10,10)):
    
    # initialize starting values
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    # get counts for each cluster merge
    for i,row in enumerate(model.children_):
        for child in row:
            if child < n_samples:
                counts[i] += 1
            else:
                counts[i] += counts[child-n_samples]
    
    # generate linkage matrix
    linkage_mat = np.column_stack([
        model.children_,
        model.distances_,
        counts
    ])
    
    # plot dendrogram
    fig, ax = plt.subplots(figsize=figsize)
    hierarchy.dendrogram(
        linkage_mat,
        p=p, 
        truncate_mode='lastp', 
        orientation='left',
        color_threshold=color_thresh,
        leaf_label_func=lambda x: mean_year(x, model, lib)
    )
    
    # text key
    plt.text(
        0.02, 0.92, 
        'Key:\n\nAvg. Year | Count', 
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.5)
    )
    
    return None

    
    
    
    
'''
Class to hold tables associated with PCA
'''
class PCA_Transformer():
    
    ''' init
    INPUT
        n_comps: number of top PCA components
        max_words: number of top words to use
        metric: for determining top words
        tfidf_name: name of column with tfidf
    '''
    def __init__(self, n_comps=10, max_words=4000, metric='dfidf', tfidf_name='tfidf'):
        
        # store parameters
        self.n_comps = n_comps
        self.max_words = max_words
        self.metric = metric
        self.tfidf_name = tfidf_name
        
        # placeholders
        self.top_words = None
        self.TFIDF = None
        self.COV = None
        self.EIG_PAIRS = None
        self.COMPS = None
        self.LOADINGS = None
        self.DCM = None
        
    ''' Perform PCA computations on tables
    INPUT
        vocab: vocab table
        bow: bow table by desired bag
        lib: library with metadata
        cols: columns from library table to use
    OUTPUT
        None
    '''
    def transform(self, vocab, bow, lib=None, cols=None):
        
        # get top words
        self.top_words = vocab.sort_values(self.metric, ascending=False) \
            .head(self.max_words) \
            .term_str
        
        # build tfidf matrix
        TFIDF = bow[self.tfidf_name].unstack(fill_value=0)
        TFIDF = TFIDF[self.top_words.values]
        TFIDF = TFIDF.apply(lambda x: x / np.sqrt(np.square(x).sum()), axis=1)
        TFIDF = TFIDF - TFIDF.mean()
        self.TFIDF = TFIDF
        
        # calculate cov matrix
        self.COV = TFIDF.T.dot(TFIDF) / (TFIDF.shape[0] - 1)
        
        # get eigenvalues and eigenvectors
        eig_vals, eig_vecs = scipy.linalg.eigh(self.COV)
        EIG_VEC = pd.DataFrame(eig_vecs, index=self.COV.index, columns=self.COV.index)
        EIG_VAL = pd.DataFrame(eig_vals, index=self.COV.index, columns=['eig_val'])
        EIG_VAL.index.name = 'term_str'
        EIG_PAIRS = EIG_VAL.join(EIG_VEC.T)
        EIG_PAIRS['exp_var'] = np.round((EIG_PAIRS.eig_val / EIG_PAIRS.eig_val.sum()) * 100, 2)
        self.EIG_PAIRS = EIG_PAIRS
    
        # get top components by explained variance
        COMPS = EIG_PAIRS.sort_values('exp_var', ascending=False).head(10).reset_index(drop=True)
        COMPS.index.name = 'comp_id'
        COMPS.index = [f'PC{i}' for i in COMPS.index.tolist()]
        COMPS.index.name = 'pc_id'
        self.COMPS = COMPS
        
        # get loadings
        self.LOADINGS = COMPS[self.COV.index].T
        self.LOADINGS.index.name = 'term_str'
        
        # calculate DCM with merged library data if provided
        self.DCM = TFIDF.dot(COMPS[self.COV.index].T)
        if (cols is not None) and (lib is not None):
            self.DCM = self.DCM.merge(lib[cols], on='doc_id')
            self.DCM['doc'] = self.DCM.apply(lambda x: f'{x.top_cat} - {x.year} - {x.title}', 1)
            
        return None
    
    ''' Modify a vocab table to include components
    INPUT
        vocab: vocabulary table
        merge_on: column on which to merge vocab with comps
    '''
    def modify_vocab(self, vocab, merge_on='term_str'):
        
        # make sure transform has been called
        if self.COMPS is None:
            return None
        # arrange components and merge
        comps = self.COMPS.T.drop('exp_var')
        comps.index.name = merge_on
        return vocab.join(comps, on=merge_on, how='left')
    
    
    # function to visualize results
    def vis_pcs(self, a, b, label='top_cat', hover_name='doc', symbol=None, size=None):
        fig = px.scatter(self.DCM, f'PC{a}', f'PC{b}', color=label, hover_name=hover_name, 
                         symbol=symbol, size=size,
                         marginal_x='box', height=800)
        fig.show()
        
        return None

    
'''
Class to perform topic modeling with LDA
'''
class TopicModel():
    
    ''' init
    INPUT
        vectorizer: sklearn CountVectorizer object
        lda: sklearn LDA object
        docs: docs table formatted for vectorizer
    '''
    def __init__(self, vectorizer, lda, docs=None):
        self.docs = docs
        # set up vectorizer
        self.vectorizer =vectorizer
        # set up lda
        self.lda = lda
        
    # build docs table from tokens and bag
    def docs_from_tokens(self, tokens, bag):
        self.docs = tokens[tokens.pos.str.match(r'^NNS?$')] \
            .groupby(bag).term_str \
            .apply(lambda x: ' '.join(x)) \
            .to_frame() \
            .rename(columns={'term_str':'doc_str'})
        
    # fit models and get important results
    def fit(self):
        self.term_counts = self.vectorizer.fit_transform(self.docs.doc_str)
        self.terms = self.vectorizer.get_feature_names()
        self.theta = pd.DataFrame(
            self.lda.fit_transform(self.term_counts), 
            index=self.docs.index
        )
        self.theta.columns.name = 'topic_id'
        self.phi = pd.DataFrame(self.lda.components_, columns=self.terms)
        self.phi.index.name = 'topic_id'
        self.phi.columns.name = 'term_str'
        
    # take top terms from topics
    def topics(self, n_top_terms):
        topics = self.phi.stack().to_frame().rename(columns={0:'topic_weight'})\
            .groupby('topic_id')\
            .apply(lambda x: 
                x.sort_values('topic_weight', ascending=False)\
                       .head(n_top_terms)\
                       .reset_index()\
                       .drop('topic_id',1)['term_str'])
        return topics.copy()
    
    # merge library table with metadata
    def merge_lib(self, lib, on='doc_id'):
        merged = self.theta.reset_index() \
            .set_index(on) \
            .merge(lib, on=on)
        return merged.copy()
    


    
    
''' Class to perform various word embedding tasks
'''
class WordEmbedding():
    
    # requires tokens, bag specification, and args for word2vec
    def __init__(self, tokens, vocab, bag, window=5, min_count=50):
        
        self.tokens = tokens
        self.vocab = vocab
        
        # build docs from tokens
        self.docs = tokens.groupby(bag) \
            .term_str.apply(lambda  x:  x.tolist()) \
            .reset_index()['term_str'].tolist()
        self.docs = [d for d in self.docs if len(d) > 1]
        
        # build w2v model
        self.model = word2vec.Word2Vec(
            self.docs,
            window=window,
            min_count=min_count
        )
        
        # placeholder for coords
        self.coords = None
        
    # get coordinates from model and merge with features
    def get_coords(self, perp=40, n_comp=2, n_iter=2500, random_state=19):
        
        # extract coords from model
        coords = pd.DataFrame(
            dict(
                vector = [self.model.wv.get_vector(w) for w in self.model.wv.key_to_index.keys()], 
                term_str = self.model.wv.key_to_index.keys()
        )).set_index('term_str')
        
        # tsne for plotting
        tsne = TSNE(
            perplexity=perp, 
            n_components=n_comp, 
            init='pca', 
            n_iter=n_iter, 
            random_state=random_state
        )
        tsne_model = tsne.fit_transform(coords.vector.to_list())
        
        # set up x, y
        coords['x'] = tsne_model[:,0]
        coords['y'] = tsne_model[:,1]
        
        # join to vocab for useful information
        coords = coords.merge(self.vocab, on='term_str')
        self.coords = coords
        return coords
    
    # non-interactive plotting with matplotlib
    def plot_coords(self, hue=None):
        
        # if still no coords then compute default
        if self.coords is None:
            self.get_coords()
        
        # plot figure
        pos = ['NN','NNS','VB','VBG','VBD','VBN','VBP','VBZ']
        plot_df = self.coords.reset_index().dropna()
        plot_df = plot_df[plot_df.max_pos.isin(pos)]
        fig = plt.figure(figsize=(10,8))
        g = sns.scatterplot(x='x', y='y', hue=hue, palette='tab10',
                            s=80, data=plot_df);
        
    # interactive px plotting
    def px_coords(self, hue):
    
        # if still no coords then compute default
        if self.coords is None:
            self.get_coords()
            
        # px plotting
        pos = ['NN','NNS','VB','VBG','VBD','VBN','VBP','VBZ']
        plot_df = self.coords.reset_index().dropna()
        plot_df = plot_df[plot_df.max_pos.isin(pos)]
        return px.scatter(plot_df, 'x', 'y', 
           text=None, 
           color=hue, 
           hover_name='term_str',          
           size='max_tfidf_sum',
           height=1000).update_traces(
                mode='markers+text', 
                textfont=dict(color='black', size=14, family='Arial'),
                textposition='top center')
        
    # complete analogy with w2v model
    def complete_analogy(self, A, B, C, n=2):
        try:
            cols = ['term', 'sim']
            return pd.DataFrame(self.model.wv.most_similar(positive=[B, C], negative=[A])[0:n], columns=cols)
        except KeyError as e:
            print('Error:', e)
            return None

    # return most similar with w2v model
    def get_most_similar(self, positive, negative=None):
        return pd.DataFrame(self.model.wv.most_similar(positive, negative), columns=['term', 'sim'])
    

