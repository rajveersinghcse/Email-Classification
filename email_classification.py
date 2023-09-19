#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Define a function to perform email classification
def classify_email(content):
    # Load the dataset
    data = pd.read_csv("emails.csv")
    
    # Drop rows with missing values
    data.dropna(inplace=True)
    
    # Encode the 'Class' column using LabelEncoder
    encoder = LabelEncoder()
    data['Class'] = encoder.fit_transform(data['Class'])
    
    # Limit the dataset to the first 20,000 rows for demonstration purposes
    data1 = data.head(20000)
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
    
    # Fit and transform the vectorizer on the email content
    X = vectorizer.fit_transform(data1['content'])
    
    # Create a Logistic Regression classifier
    classifier = LogisticRegression(solver='liblinear', penalty='l1')
    
    # Fit the classifier on the data
    classifier.fit(X, data1['Class'])
    
    # Transform the input email content
    df = vectorizer.transform([content])
    
    # Make a prediction
    prediction = classifier.predict(df)
    
    # Return the prediction
    return 'ABUSIVE' if prediction == 0 else 'NON ABUSIVE'

# Streamlit app
st.title('Email Classification')


# Get user input
content = st.text_input("Enter an email")

# Add a "Predict" button
if st.button("Predict"):
    if content:
        # Call the classification function
        result = classify_email(content)
        
        # Display the result
        st.subheader('Prediction')
        st.write(result)

