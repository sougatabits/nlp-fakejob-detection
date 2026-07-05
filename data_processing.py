import pandas as pd
import numpy as np
import re
import nltk



from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#read CSV

synthetic_indian_jobs = pd.read_csv('./data/raw/synthetic_indian_jobs.csv')

# Raw Data shape head and isnull

# print(synthetic_indian_jobs.shape)
# print(synthetic_indian_jobs.isna().sum())
# print(synthetic_indian_jobs.head)

##################DATA CLEANING###############


synthetic_indian_jobs = synthetic_indian_jobs[['title','description','label']]

#dropping columns not required
synthetic_indian_jobs.dropna(inplace=True)
#print(synthetic_indian_jobs.head())


##########





