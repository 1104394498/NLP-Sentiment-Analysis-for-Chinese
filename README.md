# NLP-Sentiment-Analysis-for-Chinese
## About
This repos implements Chinese sentiment analysis, using Word2Vec, RNN (LSTM or GRU) and FullyConnection, etc. 

## How to use
```
python main.py config/config.json -s 'The Chinese sentence you want to analyze'
```
Then the score will be printed out. 
The score $\in [0, 1]$. `1` represents absolutely positive, and `0` represents absolutely negative. The larger the score, the more positive the sentiment is. 

Example:
Input:
```
python main.py config/config.json -s '这个菜超级好吃！！非常棒！！'
```
Output:
```
sentence: 这个菜超级好吃！！非常棒！！
<generator object Tokenizer.cut at 0x1a25a44408>
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/vm/j8zz2_qd0kjf9_c9wmhk49jm0000gn/T/jieba.cache
Loading model cost 0.671 seconds.
Prefix dict has been built succesfully.
菜 超级 好吃 棒
sentiment score:  0.995
```
`sentiment score = 0.995` means that this sentence's sentiment is very positive. 

