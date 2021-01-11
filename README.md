How to structure your Python code
https://docs.python-guide.org/writing/structure/#structure-of-the-repository

## Directory Structure
```
.
├── README.md
├── base
│   ├── base_model.py
│   └── base_train.py
├── config
│   ├── Bilstm.json
│   ├── att_bilstm.json
│   └── cnn.json
├── data_collection
│   ├── create_label.py
│   ├── scrape_news.py
│   ├── stock_price.py
│   └── tickers.py
├── data_loader
│   └── data_generator.py
├── main.py
├── models
│   ├── attention_Bilstm.py
│   ├── cnn.py
│   └── lstm.py
├── saved_models
│   ├── lstm_model_fast_text.h5
│   ├──  
├── trainers
│   └── model_trainer.py
└── utils
    ├── __init__.py
    ├── config.py
    └── util.py
