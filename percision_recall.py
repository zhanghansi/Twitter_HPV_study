import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

df = pd.read_csv('./intermediate_data/analysis/annotation_for_rule_based_categorization/result_sheet.csv',encoding='ISO-8859-1')
y_true = np.array(df['hansi'])
y_pred = np.array(df['new'])
print(precision_recall_fscore_support(y_true, y_pred, average='weighted'))