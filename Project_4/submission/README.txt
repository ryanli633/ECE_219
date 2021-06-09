Project 4: Regression Analysis

Kenny Huynh: 805436804
Shu-Yun Ku: 405515493
Ryan Li: 704142455
Osama Hassen: 105644793

For the Video Transcoding Time Dataset:
1. The code for this dataset is contained in the Jupyter Notebook named:
   Video_Transcoding.ipynb
   Open up this Jupyter Notebook with Python 3.
2. To execute the code for this dataset, first update the variable 'path' at
   the top of the file to where you have the data file, 'transcoding_mesurment.tsv'
   in your directory.
3. You can then run all the cells using the built-in button in the Jupyter
   Notebook toolbar: Cell > Run All
   Or, you can run each cell individually and manually in order.
4. When working on this dataset, we were initially using the latest version of
   scikit-learn: 0.24.2. This worked for all the models and questions up until
   we had to use BayesSearchCV in Questions 24-26. The latest version of
   scikit-optimize: 0.8.1 had an issue with the variable iid=True, which has 
   been deprecated in scikit-learn.0.24.2. In order to get BayesSearchCV working
   we had to revert to older versions of software. We ended up using:
   scikit-learn: 0.23.2
   scikit-optimize: 0.8.0
   You can revert your version of scikit-learn using the commmand:
   python -m pip install scikit-learn==0.23.2

If you run into any issues, it may be that you need to install certain packages
(scikit-learn, nltk, numpy, pandas, etc.) in your python environment before 
executing the code.