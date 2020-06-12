#--------ライブラリ・関数系--------
#(ただし、Max_size=10のまま実行する場合)
#(そうでなくす場合は、その都度Tensor作成ファイルの実行が必要)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import tensorflow as tf
import keras

import time
import statistics

from torch.utils.data import TensorDataset, DataLoader

import collections

import gensim
import gc

train = pd.read_csv("Part6_Result/train.txt",sep="\t",header=None)
valid = pd.read_csv("Part6_Result/valid.txt",sep="\t",header=None)
test = pd.read_csv("Part6_Result/test.txt",sep="\t",header=None)

X_train = train.iloc[:,0]
X_valid = valid.iloc[:,0]
X_test = test.iloc[:,0]


#関数その1 : 学習等に使う多次元配列の生成を行う関数、もちろん小文字に統一
def get_word_list_True(X_df):
    ans = []
    Ans = []
    for i,text in enumerate(X_df):
        if i == 0: #データの1行目にラベル名を残してしまったので、その部分は飛ばす
            continue
        else :
            words = text.split(" ")
            ans.append(words)
            
    #単語を分解して、各々大文字を小文字に直す
    for words in ans:
        word_low = []
        for word in words:
            word_low.append(word.lower())
        Ans.append(word_low)
    
    return Ans


#関数その2 : DataFrameを受け取り、Title部分をword単位に分割したリストにする関数
#ついでに、小文字に統一
#この時作られる配列は、すべてが1行にまとまったものなので、辞書用に使うものとする
def get_word_list(X_df):
    ans = []
    Ans = []
    for i,text in enumerate(X_df):
        if i == 0:
            continue
        else :
            ans.extend(text.split(" "))
    #単語を分解して、各々大文字を小文字に直す
    for word in ans:
        Ans.append(word.lower())
        
    return Ans

#関数その3 : Title部分をword単位に分割したリストから、出現頻度の回数順にしたリストにする関数
def get_counter_list(X_df):
    target = get_word_list(X_df)
    target = collections.Counter(target)
    Ans = target.most_common()
    return Ans

#関数その4 : 出現頻度の回数のリストから、idを割り振るリストにする関数
def get_counter_list_translate(X_df):
    target = get_counter_list(X_df)
        
    id_n = 1
    
    for i in range(len(target)): 
        #要素変更のため、tupleをlist化
        target[i] = list(target[i])
        #基本処理は、elem[1](出現回数)による場合分け
        #ただし、例外処理として①同じ回数の場合、同じid番号を付与する
        #                      ⇒0番目は1で確定
        #                      ⇒それ以降は、ひとつ前のelem[1]と同じかどうかで判断
        #                      ②elem[1] < 2 の場合は0にする
        if target[i][1] < 2:
            target[i][1] = 0
        else :
            if i == 0:
                target[i][1] = id_n
            else :
                if target[i-1][1] == target[i][1]:
                    target[i][1] = id_n
                else :
                    target[i][1] = id_n + 1
                    id_n = id_n + 1
        
        #一応タプルに戻す
        target[i] = tuple(target[i])
    
    return target
                    
#関数その5 : idを割り振るリストを、辞書化する関数
def get_counter_dict(X_df):
    target = get_counter_list_translate(X_df)
    Ans = dict(target)
    return Ans

#ここまでのまとめ
#X_trainなどのデータに対して、get_counter_dictを適用すると
#単語にID番号を付与した辞書を取得できる

#ここからやりたいこと
#与えた文字列を、空白で区切り
#各要素(単語に対して、辞書と比較しID番号に変えていく)
#dはdict,xは探したい文字列⇒その番号を返す

#関数その6 : 辞書内から指定した要素のvalue、すなわちidを取ってくる関数
def search_dict(d,x): 
    for k,v in d.items():
        if k == x:
            return v

#関数その6.5 : パディングするときに使う関数で、idの中央値を取得する
def search_dict_median(d):
    Ans = []
    for k,v in d.items():
        Ans.append(v)
    k = statistics.median(Ans)
    return k


#関数その6.8 : パディングするときに使う関数で、idの調和平均を取得する
def search_dict_harmonic_mean(d):
    Ans = []
    for k,v in d.items():
        Ans.append(v)
    k = statistics.harmonic_mean(Ans)
    return k

#関数その6.9 : RNN学習時に使う関数・辞書の単語ベクトルの種類をカウント(0が多くあるため)
def search_dict_count_vector(d):
    Ans = []
    for k,v in d.items():
        if v in Ans:
            continue
        else:
            Ans.append(v)
    k = len(Ans)
    return k
    

#関数その7 : X_dfから得られる辞書を使って、str内の各単語に対してidを割り振り、そのリストを返す関数
def Change_id(str_target,X_df):
    #辞書の作成
    id_dict = get_counter_dict(X_df)
    Ans = []
    text = str_target.split(" ")
    for i in text:
        if i in id_dict.keys():
            ans = search_dict(id_dict,i)
        else:
            ans = -1
        Ans.append(ans)
    return Ans

#関数その8 : X_dfから得られる辞書を使って、X_df内の全ての文字列に対してidを割り振り、そのリストを返す関数
def Change_id_df(X_df):
    id_dict = get_counter_dict(X_df)
    Ans = []
    word_list = get_word_list_True(X_df)
    
    for words in word_list:
        id_num = []
        for word in words:
            ans = search_dict(id_dict,word)
            id_num.append(ans)
        Ans.append(id_num)
    
    return Ans



#1時間くらいかかるので、要注意
#ファイルの読み込み・変換
#trainデータの読み込み
max_size = 10

def Change_Tensor(X_df,size): #データフレームと、1行の要素のサイズを引数に指定
    vec_matrix = Change_id_df(X_df) #求めたい多次元配列
    id_dict = get_counter_dict(X_df) #X_dfの辞書
    Ans = []
    for vec in vec_matrix: #1行において
        if len(vec) > size: #sizeよりも大きかったら、そこに部分までを切り取り
            vec = vec[:size]
        else: #size以下なら、足りない分を中央値でパディング　⇒ 終わった後に分かったけど、0にした方がいいかも(過学習気味なので)
            k = search_dict_median(id_dict)
            vec += [k] * (size - len(vec))
        Ans.append(vec)
    return torch.tensor(Ans,dtype=torch.int64)

#-----------------------------規定サイズ(max_size)にパディングした引数の作成・保存------------------------
'''
#validデータの読み込み
t1 = time.time()

valid = pd.read_csv("Part6_Result/valid.txt",sep="\t",header=None)
X_valid = valid.iloc[:,0]
Y_valid = np.loadtxt("Part8_Result/Y_valid.txt",encoding="utf-8_sig",skiprows=1)

t2 = time.time()
print(t2-t1)

X_valid_torch = Change_Tensor(X_valid,max_size)

t3 = time.time()
print(t3-t1)

Y_valid_torch = torch.tensor(Y_valid,dtype=torch.int64)
t4 = time.time()
print(t4-t1)

torch.save(X_valid_torch, "Part9_Result/X_valid_torch.pt")
torch.save(Y_valid_torch, "Part9_Result/Y_valid_torch.pt")

#testデータの読み込み
test = pd.read_csv("Part6_Result/test.txt",sep="\t",header=None)
X_test = test.iloc[:,0]
Y_test = np.loadtxt("Part8_Result/Y_test.txt",encoding="utf-8_sig",skiprows=1)

X_test_torch = Change_Tensor(X_test,max_size)

t5 = time.time()
print(t5-t4)

Y_test_torch = torch.tensor(Y_test,dtype=torch.int64)

torch.save(X_test_torch, "Part9_Result/X_test_torch.pt")
torch.save(Y_test_torch, "Part9_Result/Y_test_torch.pt")

#trainデータの読み込み
train = pd.read_csv("Part6_Result/train.txt",sep="\t",header=None)
X_train = train.iloc[:,0]
Y_train = np.loadtxt("Part8_Result/Y_train.txt",encoding="utf-8_sig",skiprows=1)

X_train_torch = Change_Tensor(X_train,max_size)
Y_train_torch = torch.tensor(Y_train,dtype=torch.int64)

torch.save(X_train_torch, "Part9_Result/X_train_torch.pt")
torch.save(Y_train_torch, "Part9_Result/Y_train_torch.pt")
'''

X_train_torch = torch.load("Part9_Result/X_train_torch.pt")
Y_train_torch = torch.load("Part9_Result/Y_train_torch.pt")

X_valid_torch = torch.load("Part9_Result/X_valid_torch.pt")
Y_valid_torch = torch.load("Part9_Result/Y_valid_torch.pt")

X_test_torch = torch.load("Part9_Result/X_test_torch.pt")
Y_test_torch = torch.load("Part9_Result/Y_test_torch.pt")

#---------------------------あるニュース見出しの記事だけ抽出した引数の作成------------------------------
#---------------------------num = {"b":0,"t":1,"e":2,"m":3}の数値-------------------------------------

def Pull_Tensor(X_tensor,Y_tensor,num):
    Ans_X = []
    Ans_Y = []

    X_tensor = X_tensor.data.cpu().numpy()
    Y_tensor = Y_tensor.data.cpu().numpy()

    for xx,yy in zip(X_tensor, Y_tensor):
        if yy == num:
            Ans_X.append(xx)
            Ans_Y.append(yy)
    return Ans_X, Ans_Y

X_train_torch_Pull_b = torch.tensor(Pull_Tensor(X_train_torch, Y_train_torch, 0)[0],dtype=torch.int64)
Y_train_torch_Pull_b = torch.tensor(Pull_Tensor(X_train_torch, Y_train_torch, 0)[1],dtype=torch.int64)

X_valid_torch_Pull_b = torch.tensor(Pull_Tensor(X_valid_torch, Y_valid_torch, 0)[0],dtype=torch.int64)
Y_valid_torch_Pull_b = torch.tensor(Pull_Tensor(X_valid_torch, Y_valid_torch, 0)[1],dtype=torch.int64)

X_test_torch_Pull_b = torch.tensor(Pull_Tensor(X_test_torch, Y_test_torch, 0)[0],dtype=torch.int64)
Y_test_torch_Pull_b = torch.tensor(Pull_Tensor(X_test_torch, Y_test_torch, 0)[1],dtype=torch.int64)

#----------------------------以降、BERT用の引数作成--------------------------------------------------
from transformers import BertJapaneseTokenizer, BertForMaskedLM
from transformers import BertTokenizer

#bart-base-japaneseと
#bert-base-japanese-whole-word-maskingと
#bert-base-japanese-charと
#bert-base-japanese-char-whole-word-maskingがある
#masking版の方が、fine-tuningした時の精度が高い
#データが、そもそもJapaneseじゃなかった......

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def Change_BERT_id(X_df):
    Ans = []
    word_list = get_word_list_True(X_df) #wordのリスト

    for word in word_list:
        new_tokens = tokenizer.convert_tokens_to_ids(word)
        Ans.append(new_tokens)

    return Ans

def Change_BERT_Tensor(X_df,size): #データフレームと、1行の要素のサイズを引数に指定
    vec_matrix = Change_BERT_id(X_df) #求めたい多次元配列
    Ans = []
    for vec in vec_matrix: #1行において
        if len(vec) > size: #sizeよりも大きかったら、そこに部分までを切り取り
            vec = vec[:size]
        else: #size以下なら、足りない分を1でパディング　⇒ 終わった後に分かったけど、0にした方がいいかも(過学習気味なので)
            k = 1
            vec += [k] * (size - len(vec))
        Ans.append(vec)
    return torch.tensor(Ans,dtype=torch.int64)

#ここを変更しても、bert用のtensorの長さは変えられるし
#notebookの方でも変えられる
max_size = 10

X_train_torch_bert = Change_BERT_Tensor(X_train,max_size)
X_valid_torch_bert = Change_BERT_Tensor(X_valid,max_size)
X_test_torch_bert = Change_BERT_Tensor(X_test,max_size)