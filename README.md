# NLP-Sentiment-Analysis-for-Chinese
## About
This repos implements Chinese sentiment analysis, using Word2Vec, RNN (LSTM or GRU) and FullyConnection, etc. 

## How to use
```
python main.py config/config.json -s 'The Chinese sentence you want to analyze'
```
Then the score will be printed out. 
The score $\in [0, 1]$. `1` represents absolutely positive, and `0` represents absolutely negative. The larger the score, the more positive the sentiment is. 
