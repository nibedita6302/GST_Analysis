# Hyper-parametrized Logistic Regression for Binary Classification of GST Data

### About Project
Analysis of Goods and Services Tax (GST) data is crucial for identifying patterns related to tax compliance or potential fraud by analysing taxpayer behaviour. Machine learning models offers means to automate this process hence, providing efficient monitoring and reducing manual checks.  This project aims to develop an optimize logistic regression model for the binary classification of GST data. The dataset is pre-processed to ensure data quality and reliability, followed by construction of logistic regression model. Further, hyper-parameter tuning is performed by fine-tuning parameters such as regularization strength, penalty type, max iterations and solver. The trained model is then evaluated using test dataset, achieving an overall accuracy of 96.85% and log-loss of 0.094. 

### Tech-Stack: ![Static Badge](https://img.shields.io/badge/Python-ebd31f) ![Static Badge](https://img.shields.io/badge/Pandas-eb1fae) ![Static Badge](https://img.shields.io/badge/Numpy-eb6d1f) ![Static Badge](https://img.shields.io/badge/Scikit%20Learn-3a1feb) ![Static Badge](https://img.shields.io/badge/Matplotlib-4b9e3b) ![Static Badge](https://img.shields.io/badge/Seaborn-1f9deb) ![Static Badge](https://img.shields.io/badge/Joblib-ebd7a1) ![Static Badge](https://img.shields.io/badge/Git-ae104a)

<hr>

## Steps to Execute 
1. Fork the remote repository using [https://github.com/nibedita6302/GST_Analysis.git](https://github.com/nibedita6302/GST_Analysis.git)
2. Clone the repository or download the ZIP.
3. Create a virtual environment -
```
python -m venv venv
```
4. Activate the virtual environment -
```
venv\Scripts\activate
```
5. Install Libraries in terminal or command prompt:
```
pip install -r requirements.txt
```
6. Execute predict.py for prediction. Provide path for Input data and Target CSV files.
```
python predict.py
```
> [!NOTE]  
> predict.py uses the trained model _logistic_regression_parameterized_v1o3o1.joblib_.
-------
- To train the hyper-parameterized model with new training data execute -
```
python model.py
```
- To perform hyper-parameter tuning execute -
```
python hyperparameter_tuning.py
```
> [!Important]  
> Use the new model saved to predict test data.
------
#### Hackathon Team Members:
<table>
  <tr>
    <td style="text-align: center; padding: 10px;">
      <a href="https://github.com/AshisSardar">
        <img src="https://github.com/AshisSardar.png" alt="Ashis Sardar" style="width: 100px; height: 100px;">
      </a><br>
      &nbsp; Ashis Sardar
    </td>
    <td style="text-align: center; padding: 10px;">
      <a href="https://github.com/nibedita6302">
        <img src="https://github.com/nibedita6302.png" alt="Nibedita Chakraborty" style="width: 100px; height: 100px;">
      </a><br>
      &nbsp;&nbsp; Nibedita <br>
      &nbsp; Chakraborty
    </td>
  </tr>
</table>
