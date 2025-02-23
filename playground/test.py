# 导入必要库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# 加载内置的乳腺癌数据集
data = datasets.load_breast_cancer()
X = data.data  # 特征数据
y = data.target  # 目标变量（0: 恶性，1: 良性）
feature_names = data.feature_names

# 数据预处理
# 划分训练集和测试集（8:2比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建SVM分类器（使用RBF核）
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# 训练模型
svm_classifier.fit(X_train, y_train)

# 预测测试集
y_pred = svm_classifier.predict(X_test)

# 评估模型性能
print("测试集准确率: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Malignant', 'Benign'], rotation=45)
plt.yticks(tick_marks, ['Malignant', 'Benign'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 可视化特征空间（使用PCA降维到2D）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y_train==0, 0], X_pca[y_train==0, 1], color='red', label='Malignant')
plt.scatter(X_pca[y_train==1, 0], X_pca[y_train==1, 1], color='green', label='Benign')
plt.title('PCA Visualization of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()