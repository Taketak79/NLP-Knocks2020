## 言語処理100本ノックとは
言語処理100本ノックとは、言語処理に関わる技術の（楽しい）習得を目的とし 
東北大学乾研究室にて、作成されたものです。
[言語処理100本ノック2020](https://nlp100.github.io/ja/)

## 実行環境
以下の環境で実行しました。
(本来は、Linux超推奨です。Windowsの場合はWLSでLinuxを出来るようにすると便利！)

Windows10
Python 3.7.3
Anaconda 3

使用GPU : RTX2080 (8GB)
        : Google ColabolatoryのGPU(8GB～16GB)

使用ライブラリ : time,statistics,numpy,pandas,re,matplotlib,seaborn,mecab,cabocha,tensorflow,keras,pytorch,gensim

## 実行における注意事項

### コードを見るだけ
コードと結果を見るだけであれば、つらつら見てもらえれば問題ありません。

### 第6章以降の注意事項
Github上には、データセットやそれを変換したtxtファイルはいれていません。
(25MB以上の為)

基本的には、100本ノックのサイトからダウンロード・解凍し使ってください。
(実行Errorが出る場合は8割方、パスの適切な設定がなされていないことが考えられます。)

そこそこのGPU性能が要ります。
10GB以上あると、楽しい実行が出来ると思います。

[TaketakのGoogle Drive](https://drive.google.com/drive/folders/1VV-51LXMQPy1pHfpjRiCgic1B5zzdwOK?usp=sharing)

### Google Colabolatory(以下、Colab)
Colabは誰でも使えるGPUサービスです。
ここでは気を付けるべきことを紹介します。
詳しいことは調べてみてください。

1.Driveの容量に気を付けること
　
  データセットをDriveに入れると、Driveの容量を圧迫します。
  注意してくださいね。
　
2.使えるGPUの容量は時間帯によるということ

　使えるGPUは時間帯によります。(Colab Pro出ない場合)
  やってみた感じでは、昼間・夜6時くらいは少なく(7GBくらい)
  朝の3～6時は一番多いです。(15GBくらい)
 

### 第1章
特にありません。

### 第2章
Linux環境を構築できなかったので、行えませんでした。

### 第3章 正規表現探索
reが必要ですが、condaならすでに入ってます。

### 第4章 Mecabを使った形態素解析
Mecabのインストールが必要です。
気を付けるべき点は2つ

1. 構築するbit数に気を付けること

   Mecabは32bit版と、64bit版(有志作成)があります。
   後に、Cabochaを使う場合には、32bitでMecabを入れる必要があります
   Anacondaを使う場合には、Anaconda自体も32bitでなければいけません。

2. Colabの場合は、以下のコマンドを打つこと

   `!apt install aptitude`
   `!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y`
   `pip install mecab-python3==0.7`
   
### 第5章 Cabochaを使った係受け解析
Cabochaのインストールが必要です。
ここら辺を参考に、入れてみてください。
[Cabochaのインストールその1](https://qiita.com/ayuchiy/items/17a2d48116b2da7535eb)
[Cabochaのインストールその2](https://qiita.com/mima_ita/items/161cd869648edb30627b)
Cabochaは、Colabを使わないほうがいいかもしれません。

### 第6章 機械学習
tensorflow + keras もしくは pytorchのインストールが必要です。
pipコマンドか、condaコマンドで出来るはず。
 `pip install ○○`
諸々、データセットはダウンロードしてください。

層構造のレイヤーを組むのであれば、kerasがやはり便利。
簡単に機械学習がしたい！等であればpytorchが良いかな～
という感じがしました。

### 第7章 単語ベクトル・機械学習
gensimのインストールが必要です。
pipコマンドで出来るはず。
諸々、データセットはダウンロードしてください。
第6章の結果を使います。

### 第8章 RNN・CNN・Bertの転移学習
第6章・第7章の結果を使います。
諸々、データセットはダウンロードしてください。

### 第9章 機械翻訳
やっていません。

### Special Thanks

1.[u++の備忘録](https://upura.hatenablog.com/entry/2020/04/14/024948)

2.[駆け出しエンジニア塾](https://kakedashi-engineer.appspot.com/nlp100/)

3.[素人の言語処理100本ノックまとめ](https://qiita.com/segavvy/items/fb50ba8097d59475f760)

4.[Pytorch公式リファレンス](https://pytorch.org/docs/stable/index.html)

5.[TensorFlow公式リファレンス](https://www.tensorflow.org/api_docs/python/tf/lite)

他、python・機械学習・pytorch・Tensorflowに関する関連サイト