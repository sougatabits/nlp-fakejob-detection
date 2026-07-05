fake-job-nlp-project/
│
├── data/
│   └── fake_job_postings.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│   └── 02_preprocessing.ipynb
│   └── 03_model_tfidf.ipynb
│   └── 04_model_word2vec.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── utils.py
│
├── models/
│   ├── tfidf_model.pkl
│   ├── w2v_model.pkl
│   ├── vectorizer.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── wordcloud.png
│   ├── class_report.txt
│
├── app/
│   └── app.py   # (Streamlit / Flask optional)
│
├── requirements.txt
├── README.md
└── main.py