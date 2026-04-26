# Cancer Risk Prediction Analysis

## Project Overview

This project develops and evaluates machine learning models to predict cancer risk scores based on lifestyle, environmental, and genetic factors. The analysis compares multiple regression models and provides uncertainty quantification for predictions.

## Reproduction

1. Open the Model_Analysis jupyter notebook
2. Download the listed dataset found in the notebook. If you are unable to find the link, it is also here: https://www.kaggle.com/datasets/tarekmasryo/cancer-risk-factors-dataset
3. After extracting the dataset, you should obtain the file "cancer-risk-factors.csv". If using Google Colab, upload the file into the notebook. If not, ensure that you upload the file into the same folder as the notebook
4. After uploading the file, download all relevant dependecies. All libraries used can be found on the top of the notebook. If you are still missing some dependecies, look to the requirements.txt and ensure that you have all dependecies listed there.  
5. Click the Run All button in the notebook. A model.pkl file should have been created. This file will be used in app.py to run the streamlit dashboard.
6. Next ensure that you have all dependencies downloaded on top of the app.py file. If not, download the dependencies.
7. Go to your terminal and run this command: streamlit run app.py
8. The command should run the streamlit dashboard on your local host, providing you with how the dashboard should look like. 


Thank you for exploring our Cancer Risk Prediction Analysis Project.
