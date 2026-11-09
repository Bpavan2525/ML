# Machine Learning Notebooks

Exploratory data analysis and supervised learning in Python with **pandas, Matplotlib, Seaborn and scikit-learn**. Each notebook opens on GitHub with its outputs.

## Highlights

### Used-bike price analysis ([EDA.ipynb](EDA.ipynb))

1,061 Indian used-bike listings (name, price, year, seller type, owner, km driven, ex-showroom price).

- Found 435 missing `ex_showroom_price` values and set out a fill-or-drop strategy
- Removed `km_driven` outliers with the IQR rule: 1,061 → 1,022 rows, max km from 880,000 to 86,000, std from 51.6k to 19.6k
- Selling prices are right-skewed: most bikes sell cheap, a long tail of premium models
- Newer bikes sell for more and high-mileage bikes for less; seller type and number of owners also move price
- One-hot encoded `seller_type` and built a correlation heatmap to rank the drivers of price

### Decision trees ([Decision_Tree_Assignment.ipynb](Decision_Tree_Assignment.ipynb))

- Iris classifier: 100% test accuracy; petal length carries 91% of feature importance
- A depth-3 tree matches the fully grown tree, so the simpler model wins
- `GridSearchCV` over `max_depth` and `min_samples_split` (best: depth 4, split 2)
- Boston housing regressor: MSE 10.4; rooms per dwelling (`RM`, 60%) and `LSTAT` (19%) dominate

### Regression ([Supervised_Learning_Regression…](Supervised_Learning_Regression_Models_and_Performance_Metrics_%20Assignment.ipynb), [Regression.ipynb](Regression.ipynb))

Simple, multiple and polynomial regression with scikit-learn; MAE, MSE, RMSE and R² explained and computed.

### Feature engineering ([Feature_Engineering.ipynb](Feature_Engineering.ipynb))

Notes on parameters, encoding, scaling and feature selection.

## Run

```bash
pip install pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

The bike dataset is the public "Bike Details" CSV (Kaggle); place it next to `EDA.ipynb` and change the path in the first cell.

## Related projects

- [Spreadsheet Cleaner](https://github.com/Bpavan2525/Data-Toolkit): pandas data cleaning as a web app and CLI
- [Shop Orders API](https://github.com/Bpavan2525/restapi_flask): Flask REST API with tests
