import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.impute import SimpleImputer
from scipy import stats


data = pd.read_csv('C:/Users/hiennpd3/Desktop/Mô hình Scorecard/hmeq.csv', header=0, sep=',')

# Biểu đồ histogram
def _plot_hist_subplot(x, fieldname, bins=10, use_kde=True):
    x = x.dropna()
    xlabel = '{} bins tickers'.format(fieldname)
    ylabel = 'Count obs in each bin'.format(fieldname)
    title = 'Histogram plot of {} with {} bins'.format(fieldname, bins)
    ax = sns.histplot(x, bins=bins, kde=use_kde)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax

# Biểu đồ barchart
def _plot_barchart_subplot(x, fieldname):
    xlabel = 'Group of {}'.format(fieldname)
    ylabel = 'Count obs in each bin'.format(fieldname)
    title = 'Barchart plot of {}'.format(fieldname)
    x = x.fillna('Missing')
    df_summary = x.value_counts(dropna=False)
    y_values = df_summary.values
    x_index = df_summary.index
    ax = sns.barplot(x=x_index, y=y_values, order=x_index)
    labels = list(set(x))
    for label, p in zip(y_values, ax.patches):
        ax.annotate(label, (p.get_x()+0.25, p.get_height()+0.15))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return ax

# # Điều chỉnh histogram và bar chart subplot
# fig, axes = plt.subplots(4, 3, figsize=(18, 16))
# fig.subplots_adjust(hspace=0.5, wspace=0.2)

# for ax, (fieldname, dtype) in zip(axes.flatten(), zip(data.columns, data.dtypes.values)):
#     if dtype in ['float64', 'int64']:
#         _plot_hist_subplot(data[fieldname], fieldname=fieldname)  # Loại bỏ đối số ax
#     else:
#         _plot_barchart_subplot(data[fieldname], fieldname=fieldname)  # Loại bỏ đối số ax

# fig.suptitle('Visualization all fields')
# plt.show()

# Hàm binning và tính WOE
def _bin_table(data, colname, n_bins=10, qcut=None):
    X = data[[colname, 'BAD']]
    X = X.sort_values(colname)
    coltype = X[colname].dtype

    if coltype in ['float64', 'int64']:
        if qcut is None:
            try:
                bins, thres = pd.qcut(X[colname], q=n_bins, retbins=True, duplicates='drop')
                bins = pd.cut(X[colname], bins=thres, include_lowest=True)
                X['bins'] = bins
            except ValueError:
                print('Error: Too few unique values to bin.')
        else:
            bins, thres = pd.cut(X[colname], bins=qcut, retbins=True, include_lowest=True)
            X['bins'] = bins
    elif coltype == 'object':
        X['bins'] = X[colname]

    df_GB = pd.pivot_table(X,
                           index=['bins'],
                           values=['BAD'],
                           columns=['BAD'],
                           aggfunc={'BAD': np.size},
                           fill_value=0)

    df_Count = pd.pivot_table(X,
                              index=['bins'],
                              values=['BAD'],
                              aggfunc={'BAD': np.size},
                              fill_value=0)

    if coltype in ['float64', 'int64']:
        df_Thres = pd.DataFrame({'Thres': thres[1:]}, index=df_GB.index)
    elif coltype == 'object':
        df_Thres = pd.DataFrame(index=df_GB.index)
        thres = None
    df_Count.columns = ['No_Obs']
    df_GB.columns = ['#BAD', '#GOOD']
    df_summary = df_Thres.join(df_Count).join(df_GB)
    return df_summary, thres

def _WOE(data, colname, n_bins=None, min_obs=100, qcut=None):
    df_summary, thres = _bin_table(data, colname, n_bins=n_bins, qcut=qcut)
    df_summary['#BAD'] = df_summary['#BAD'].replace({0: 1})

    if qcut is not None:
        exclude_ind = np.where(df_summary['No_Obs'] <= min_obs)[0]
        if exclude_ind.shape[0] > 0:
            new_thres = np.delete(thres, exclude_ind)
            print('Auto combine {} bins into {} bins'.format(n_bins, new_thres.shape[0] - 1))
            df_summary, thres = _bin_table(data, colname, qcut=new_thres)

    new_thres = thres
    df_summary['GOOD/BAD'] = df_summary['#GOOD'] / df_summary['#BAD']
    df_summary['%BAD'] = df_summary['#BAD'] / df_summary['#BAD'].sum()
    df_summary['%GOOD'] = df_summary['#GOOD'] / df_summary['#GOOD'].sum()
    df_summary['WOE'] = np.log(df_summary['%GOOD'] / df_summary['%BAD'])
    df_summary['IV'] = (df_summary['%GOOD'] - df_summary['%BAD']) * df_summary['WOE']
    df_summary['COLUMN'] = colname
    IV = df_summary['IV'].sum()
    # print('Information Value of {} column: {}'.format(colname, IV))
    return df_summary, IV, new_thres

# Khởi tạo WOE_dict và tính toán WOE cho từng cột
WOE_dict = {}
nbins = {'LOAN': 10, 'MORTDUE': 10, 'VALUE': 10, 'YOJ': 10, 'CLAGE': 10, 'NINQ': 2, 'CLNO': 10, 'DEBTINC': 7}
for col, bins in nbins.items():
    df_summary, IV, thres = _WOE(data, colname=col, n_bins=bins)
    WOE_dict[col] = {'table': df_summary, 'IV': IV}

'''
Do các biến DEROG, DELINQ có xu hướng là biến thứ bậc hơn là biến liên tục nên áp dụng 
cách phân chia theo quantile sẽ tạo ra những khoảng bins có độ dài bằng 0.
Do đó chúng ta sẽ phân chia theo ngưỡng cutpoint.
'''

# Define MIN_VAL and MAX_VAL
for col in ['DEROG', 'DELINQ']:
  MIN_VAL = data[col].min()
  MAX_VAL = data[col].max()  
  df_summary, IV, thres = _WOE(data, colname=col, n_bins=5, qcut=[MIN_VAL, 2, MAX_VAL])
  WOE_dict[col] = {'table':df_summary, 'IV':IV}

# Tiếp theo ta sẽ tính toán IV cho các biến category là REASON và JOB.

for col in ['REASON', 'JOB']:
  df_summary, IV, thres = _WOE(data, colname=col)
  WOE_dict[col] = {'table':df_summary, 'IV':IV}

'''
2.4.3. Xếp hạng các biến theo sức mạnh dự báo
Dựa trên giá trị IV đã tính toán ở bước trước, ta sẽ xếp hạng các biến này như bên dưới.
'''

columns = []
IVs = []
for col in data.columns:
  if col != 'BAD':
    columns.append(col)
    IVs.append(WOE_dict[col]['IV'])
df_WOE = pd.DataFrame({'column': columns, 'IV': IVs})

def _rank_IV(iv):
  if iv <= 0.02:
    return 'Useless'
  elif iv <= 0.1:
    return 'Weak'
  elif iv <= 0.3:
    return 'Medium'
  elif iv <= 0.5:
    return 'Strong'
  else:
    return 'suspicious'

df_WOE['rank']=df_WOE['IV'].apply(lambda x: _rank_IV(x))
df_WOE.sort_values('IV', ascending=False)

'''
Như vậy trong các biến trên, biến REASON không có tác dụng trong việc phân loại hồ sơ nợ xấu. Các biến còn lại đều có tác dụng hỗ trợ một phần phân loại hồ sơ. 
Trong đó các biến có sức mạnh nhất là DELINQ, DEBTINC. 
Tiếp theo CLAGE, DEROG, LOAN, VALUE, JOB  là các biến có sức mạnh trung bình. 
Các biến còn lại bao gồm NINQ, YOJ, CLNO và MORTDUE cũng có sức mạnh phân loại nhưng yếu hơn. DELINQ là biến có tương quan rất lớn đến việc phân loại nên chúng ta cần phải review lại giá trị của biến.
'''

'''
2.4.4. Hồi qui logistic
Phương trình hồi qui logistic trong credit scorecard sẽ không 
hồi qui trực tiếp trên các biến gốc mà thay vào đó giá trị WOE ở từng biến sẽ được sử dụng thay thế để làm đầu vào. 
Ta sẽ tính toán các biến WOE bằng cách map mỗi khoảng bin tương ứng với giá trị WOE của nó như sau:
'''

for col in WOE_dict.keys():
    try:
        key = list(WOE_dict[col]['table']['WOE'].index)
        woe = list(WOE_dict[col]['table']['WOE'])
        d = dict(zip(key, woe))
        col_woe = col+'_WOE'
        data[col_woe] = data[col].map(d)
    except Exception as e:
        print(col, e)

# Gán giá trị input là các biến WOE và biến mục tiêu là data[‘BAD’].
X = data.filter(like='_WOE', axis=1)
y = data['BAD']

# Phân chia tập train/test
ids = np.arange(X.shape[0])
X_train_1, X_test_1, y_train, y_test, id_train, id_test = train_test_split(X, y, ids, test_size=0.2, stratify=y, shuffle=True, random_state=123)

# Xử lý giá trị NaN
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train_1)
X_test = imputer.transform(X_test_1)

# Đào tạo mô hình Logistic Regression
logit_model = LogisticRegression(solver='lbfgs', max_iter=1000, fit_intercept=True, tol=0.0001, C=1, penalty='l2')
logit_model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred_train = logit_model.predict(X_train)
acc_train = accuracy_score(y_pred_train, y_train)
y_pred_test = logit_model.predict(X_test)
acc_test = accuracy_score(y_pred_test, y_test)

##Đường cong ROC trên tập test
y_pred_prob_test = logit_model.predict_proba(X_test)[:, 1]
fpr, tpr, thres = roc_curve(y_test, y_pred_prob_test)
roc_auc = auc(fpr, tpr)

# def _plot_roc_curve(fpr, tpr, thres, auc):
#     plt.figure(figsize = (10, 8))
#     plt.plot(fpr, tpr, 'b-', color='darkorange', lw=2, linestyle='--', label='ROC curve (area = %0.2f)'%auc)
#     plt.plot([0, 1], [0, 1], '--')
#     plt.axis([0, 1, 0, 1])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend(loc='lower right')
#     plt.title('ROC Curve')

# _plot_roc_curve(fpr, tpr, thres, roc_auc)

'''
Chỉ số AUC (area under curve) đo lường phần diện tích nằm dưới đường cong ROC cho biết 
khả năng phân loại của các hợp đồng GOOD/BAD của mô hình hồi qui logistic là mạnh hay yếu. 
AUC thuộc khoản [0,1], giá trị của nó càng lớn thì mô hình càng tốt. 
Đối với mô hình hồi qui logistic này, AUC = 0.87 là khá cao, cho thấy khả năng dự báo của 
mô hình tốt và có thể áp dụng mô hình vào thực tiễn.
# '''
# #Đường cong precision và recall trên tập test
precision, recall, thres = precision_recall_curve(y_test, y_pred_prob_test)

# def _plot_prec_rec_curve(prec, rec, thres):
#     plt.figure(figsize = (10, 8))
#     plt.plot(thres, prec[:-1], 'b--', label = 'Precision')
#     plt.plot(thres, rec[:-1], 'g-', label = 'Recall')
#     plt.xlabel('Threshold')
#     plt.ylabel('Probability')
#     plt.title('Precsion vs Recall Curve')
#     plt.legend()

# _plot_prec_rec_curve(precision, recall, thres)

'''
Đường cong precision, recall giúp lựa chọn ngưỡng xác suất phù hợp để mang lại 
độ chính xác cao hơn trong precision hoặc recall. 
Precision cho ta biết tỷ lệ dự báo chính xác trong số các hồ sơ được dự báo là GOOD (tức nhãn là 1). 
Recall đo lường tỷ lệ dự báo chính xác các hồ sơ GOOD trên thực tế. 
Luôn có sự đánh đổi giữa 2 tỷ lệ này, nên ta cần phải dựa vào biểu đồ 
của 2 đường precision vs recall để tìm ra ngưỡng tối ưu. 
Thông thường sẽ dựa trên kì vọng về precision hoặc recall từ trước để lựa chọn ngưỡng threshold. 
Chẳng hạn trước khi bước vào dự án ta kì vọng tỷ lệ dự báo đúng hồ sơ GOOD là 70% 
thì cần thiết lập các threshold để recall >= 70%. 
Trường hợp khác, ta kì vọng tỷ lệ dự báo đúng 
trong số các hồ sơ được dự báo là GOOD là 70% 
thì cần lựa chọn các threshold để precision >= 70%. 
Rất khó để nói ngưỡng threshold nào nên được lựa chọn là tốt nhất. 
Điều này phụ thuộc vào mục tiêu của mô hình là ưu tiên phân loại đúng hồ sơ GOOD hay hồ sơ BAD hơn.
'''

'''
Kiểm định Kolmogorov-Smirnov:

Đây là kiểm định đo lường sự khác biệt trong phân phối giữa GOOD và BAD theo các tỷ lệ ngưỡng threshold. 
Nếu mô hình có khả năng phân loại GOOD và BAD tốt thì đường cong phân phối xác suất tích lũy 
(cumulative distribution function - cdf) giữa GOOD và BAD phải có sự tách biệt lớn. 
Trái lại, nếu mô hình rất yếu và kết quả dự báo của nó chỉ ngang bằng một phép lựa chọn ngẫu nhiên. 
Khi đó đường phân phối xác suất tích lũy của GOOD và BAD sẽ nằm sát nhau và tiệm cận đường chéo 45 độ. 
Kiểm định Kolmogorov-Smirnov sẽ kiểm tra giả thuyết Ho là hai phân phối xác suất GOOD và BAD không có sự khác biệt. 
Khi P-value < 0.05 bác bỏ giả thuyết Ho.
'''
##Tính toán phân phối xác suất tích lũy của GOOD và BAD
def _KM(y_pred, n_bins):
  _, thresholds = pd.qcut(y_pred, q=n_bins, retbins=True)
  cmd_BAD = []
  cmd_GOOD = []
  BAD_id = set(np.where(y_test == 0)[0])
  GOOD_id = set(np.where(y_test == 1)[0])
  total_BAD = len(BAD_id)
  total_GOOD = len(GOOD_id)
  for thres in thresholds:
    pred_id = set(np.where(y_pred <= thres)[0])
    # Đếm % số lượng hồ sơ BAD có xác suất dự báo nhỏ hơn hoặc bằng thres
    per_BAD = len(pred_id.intersection(BAD_id))/total_BAD
    cmd_BAD.append(per_BAD)
    # Đếm % số lượng hồ sơ GOOD có xác suất dự báo nhỏ hơn hoặc bằng thres
    per_GOOD = len(pred_id.intersection(GOOD_id))/total_GOOD
    cmd_GOOD.append(per_GOOD)
  cmd_BAD = np.array(cmd_BAD)
  cmd_GOOD = np.array(cmd_GOOD)
  return cmd_BAD, cmd_GOOD, thresholds

cmd_BAD, cmd_GOOD, thresholds = _KM(y_pred_prob_test, n_bins=20)

# #Biểu đồ phân phối xác suất tích lũy của GOOD và BAD
# def _plot_KM(cmd_BAD, cmd_GOOD, thresholds):
#   plt.figure(figsize = (10, 8))
#   plt.plot(thresholds, cmd_BAD, 'y-', label = 'BAD')
#   plt.plot(thresholds, cmd_GOOD, 'g-', label = 'GOOD')
#   plt.plot(thresholds, cmd_BAD-cmd_GOOD, 'b--', label = 'DIFF')
#   plt.xlabel('% observation')
#   plt.ylabel('% total GOOD/BAD')
#   plt.title('Kolmogorov-Smirnov Curve')
#   plt.legend()

# _plot_KM(cmd_BAD, cmd_GOOD, thresholds)

#Kiểm định Kolmogorov-Smirnov test:
stats.ks_2samp(cmd_BAD, cmd_GOOD)

'''
p-value < 0.05 cho thấy phân phối tích lũy giữa tỷ lệ BAD và GOOD là khác biệt nhau. Do đó mô hình có ý nghĩa trong phân loại hồ sơ.
'''

'''
2.4.5. Tính điểm credit score cho mỗi feature
Bước cuối cùng là tính ra điểm tín nhiệm (credit scorecard) của mỗi khách hàng bằng cách tính điểm số cho mỗi feature 
(feature ở đây là một khoảng bin của biến liên tục hoặc một class của biến category). 
Điểm sẽ được scale theo công thức sau:
<Công thức trong file>
'''
def _CreditScore(beta, alpha, woe, n = 12, odds = 1/4, pdo = -50, thres_score = 600):
  factor = pdo/np.log(2)
  offset = thres_score - factor*np.log(odds)
  score = (beta*woe+alpha/n)*factor+offset/n
  return score

_CreditScore(beta = 0.5, alpha = -1, woe = 0.15, n = 12)

X_train_df = pd.DataFrame(X_train)
betas_dict = dict(zip(list(X_train_df.columns), logit_model.coef_[0]))
alpha = logit_model.intercept_[0]


cols = []
features = []
woes = []
betas = []
scores = []

for col in columns:
  for feature, woe in WOE_dict[col]['table']['WOE'].to_frame().iterrows():
      cols.append(col)
      # Add feature
      feature = str(feature)
      features.append(feature)    
      # Add woe
      woe = woe.values[0]
      woes.append(woe)
      # Add beta
      col_woe = col+'_WOE'
      beta = betas_dict[col_woe]
      betas.append(beta)
      # Add score
      score = _CreditScore(beta = beta, alpha = alpha, woe = woe, n = 12)
      scores.append(score)

df_WOE = pd.DataFrame({'Columns': cols, 'Features': features, 'WOE': woes, 'Betas':betas, 'Scores':scores})
df_WOE.head()

# '''
# Như vậy ta đã hoàn thiện bảng tính điểm số cho mỗi features. 
# Từ điểm số này ta có thể suy ra điểm tín nhiệm của mỗi một hồ sơ bằng cách 
# tính tổng điểm số của toàn bộ các features của hồ sơ đó. 
# Bên dưới ta sẽ thực hành tính điểm tín nhiệm cho một hồ sơ ngẫu nhiên:

# '''
# ###
# ### Viết hàm tính điểm cho một hồ sơ
# ###
# test_obs = data[columns].iloc[0:1, :]

# def _search_score(obs, col):
#   feature = [str(inter) for inter in list(WOE_dict[col]['table'].index) if obs[col].values[0] in inter][0]
#   score = df_WOE[(df_WOE['Columns'] == col) & (df_WOE['Features'] == feature)]['Scores'].values[0]
#   return score

# # Tính điểm cho trường 'LOAN' của bộ hồ sơ test
# score = _search_score(test_obs, 'LOAN')
# score