import sys
import pandas as pd
from SIF import data_io, params, SIF_embedding,tree
from gensim.scripts.glove2word2vec import glove2word2vec
from scipy.spatial import distance
import jieba
from gensim.models import KeyedVectors
import numpy as np
from itertools import product,count
from heapq import nlargest
import math
import random

path= '/bigdata/xiaoma/data/news_chinese_sqlResult_1558435.csv'
glove_file = 'vectors.txt'
tmp_file = 'glove_2_wordvec.txt'
word_freauency_path = 'vocab.txt'
glove2word2vec(glove_file, tmp_file)



def cut_sentences_iter(sentence):
    sign = '。！？'
    s = []
    for ch in sentence:
        if ch == '\n' or ch == '\\n':
            continue
        s.append(ch)
        if ch in sign:
            yield ''.join(s)
            s = []
    yield ''.join(s)


def create_stopwords():
    stop_list = [line.strip() for line in open("stopWords.txt", 'r', encoding='utf-8').readlines()]
    return stop_list

weight_params = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
(Word2Indx, Word2vector) = data_io.getWordmap(glove_file)
Word2Weight = data_io.getWordWeight(word_freauency_path, weight_params)
Index2Weight= data_io.getWeight(Word2Indx, Word2Weight)

def cosine_distance_by_sentence_vector(s1,s2):
    word_idx_seq_of_sentence, mask = data_io.sentences2idx([' '.join(s1),' '.join(s2)], Word2Indx) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location

    word_weight_of_sentence = data_io.seq2weight(word_idx_seq_of_sentence, mask, Index2Weight) # get word weights
    # set parameters
    param = params.params()
    param.rmpc = rmpc
    embedding = SIF_embedding.SIF_embedding(Word2vector, word_idx_seq_of_sentence, word_weight_of_sentence, param)
    s1_embed = embedding[0]
    s2_embed = embedding[1]

    return cosine_similarity(s1_embed,s2_embed)

def cosine_similarity(vec1, vec2):
    x = np.array(vec1)
    y = np.array(vec2)
    frac = np.sum(x * y)
    dem = float(np.sqrt(sum(x ** 2)) * np.sqrt(sum(y ** 2)))   
    return frac / dem

def create_graph(sentence_2word_list):
    sentence_count = len(sentence_2word_list)
    sentence_graph = np.zeros((sentence_count,sentence_count),dtype = np.float)
    for i,j in product(range(sentence_count),repeat=2):#00,01,02,03;10,11,12,13...
        if i != j:
            sentence_graph[i][j] = cosine_distance_by_sentence_vector(sentence_2word_list[i],sentence_2word_list[j])
    print(sentence_graph)
    return sentence_graph

def calculate_score(sentence_weight_graph, scores):
    """
    计算句子在图中的分数
    pagerank:
    pr(Vi) = (1-D)/n + D * sum(pr(Vj) / L(Vj))
                             j
    times n:
     n* pr(Vi) = (1-d) + d * sum(n*pr(Vj) / L(Vj))

    L(j) nums of link out
    n    all pages nums

    textrank:
    S(Vi) = (1-d) + d*sum(wji / sum(wjk) * S(Vj))
                      j         k
    Reference:https://zhuanlan.zhihu.com/p/41091116
    """
    n = len(sentence_weight_graph)
    d = 0.85
    pr_score = 0.0
    for i in range(n):
        for j in range(n):
            denominator = 1e-8
            # do not process i==j because weight[i][j]=0
            fraction = sentence_weight_graph[j][i] * scores[j]
            # 计算分母
            for k in range(n):
                denominator += sentence_weight_graph[j][k]
            pr_score += fraction / denominator
        scores[i] = (1 - d) + d * pr_score
    return scores

def get_val_by_index(generate_handle,n):
    if n == 0:
        return next(generate_handle)
    for _ in range(n):
        next(generate_handle)
    return next(generate_handle)
    
def text_sumary(text,n_lines):
    sentence_generate = cut_sentences_iter(text)
    sentence_2word_list = []
    sentences = []
    count=0
    for sentence in sentence_generate:
        if not sentence.strip():
            continue
        sentence_2word_list.append([str(w) for w in jieba.cut(sentence) if (w not in stop_word_list) and (w in model) ])

    sentence_graph = create_graph(sentence_2word_list)
    print('------')
    scores = np.zeros((len(sentence_2word_list),1))
    old_scores = scores.copy()
    for i in range(len(sentence_2word_list)):
        old_scores[i] = 0.1        
   
    while not ((abs(scores - old_scores) < 1e-3).all()):   
        old_scores = scores.copy()
        scores = calculate_score(sentence_graph,scores)
        count+=1
    print('itera calac counts:'+str(count))
    print(scores)
         
    selected_id_list = nlargest(n_lines,zip(scores,count()),lambda x:x[0])
    for i in range(len(selected_id_list)):
        sentence = get_val_by_index(cut_sentences_iter(text),selected_id_list[i][1])
        sentences.append(sentence)
        print(sentence)
model = KeyedVectors.load_word2vec_format(tmp_file)
df = pd.read_csv(path,error_bad_lines=False,header=None)
text = df.iloc[:,3].apply(lambda x:''.join(str(x).split('\\n'))).to_list()
stop_word_list = create_stopwords()    
text_sumary(text[24],3)
