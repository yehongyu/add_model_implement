#coding:utf-8
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import pandas


d_path1 = 'ph_0806_3d.keyphrase_0801_30d_batch256.all.csv'
d_path1 = 'ph_0806_3d.cmt_texts_0801_30d_batch256_v10.all.csv'
d_path1 = 'ph_0806_3d.keyphrase_0801_30d_only_text_batch256.all.csv'
df1 = pandas.read_csv(d_path1, sep='\t', names=['label', 'score', 'cid', 'project_id', 'level'], header=1)
print(df1.head())
scores1 = df1['score'].values
print(type(scores1[0]), scores1[:10])
print(type(scores1))
labels1 = df1['label'].values
print(type(labels1[0]), labels1[:10])
print(type(labels1))

plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
#y_true为样本实际的类别，y_scores为样本为正例的概率

#y_true = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
#y_scores = np.array([0.9, 0.75, 0.86, 0.47, 0.55, 0.56, 0.74, 0.62, 0.5, 0.86, 0.8, 0.47, 0.44, 0.67, 0.43, 0.4, 0.52, 0.4, 0.35, 0.1])


precision, recall, thresholds = precision_recall_curve(labels1, scores1)
#print(precision)
#print(recall)
#print(thresholds)
plt.plot(recall,precision)
plt.show()


