# =====================================================
#                 DATA PRE-PROCESSING
# =====================================================

# import pandas as pd
# pd.set_option('display.max_rows', 10)
#
# # files reading
# dfh = pd.read_csv('./hosts.csv')
# dfm = pd.read_csv('./medals.csv')
# dfa = pd.read_csv('./athletes.csv')
# dfp = pd.read_csv('./programs.csv', encoding='ISO-8859-1')
#
#
# # data filtering
# dfh = dfh[dfh['Year'] >= 1960]
# dfm = dfm[dfm['Year'] >= 1960]
# dfa = dfa[dfa['Year'] >= 1960]
# dfa = dfa[dfa['Medal'] != 'No medal']
# dfag = dfa.groupby(['Team', 'Sport'])
# dfh['Host'] = dfh['Host'].apply(lambda x: x.split(',')[1][1:])
# dfm = dfm.rename(columns={'NOC': 'Team'})
# # print(dfag.get_group(('China', 'Football')))
#
#
# # # country modifying
# dfm = dfm[dfm['Team'] != 'Soviet Union']
# dfm = dfm[dfm['Team'] != 'East Germany']
# dfm = dfm[dfm['Team'] != 'West Germany']
# # dfm['Team'] = dfm['Team'] \
# #     .replace({'Yugoslavia': 'Serbia and Montenegro'}) \
# #     .replace({'Soviet Union': 'Russia'}) \
# #     .replace({'North Yemen': 'Yemen'}) \
# #     .replace({'South Yemen': 'Yemen'}) \
# #     .replace({'West Germany': 'Germany', 'East Germany': 'Germany'})
# # dfm = dfm.groupby('Team').agg({
# #     'Rank': 'first',
# #     'Gold': 'sum',
# #     'Silver': 'sum',
# #     'Bronze': 'sum',
# #     'Total': 'sum',
# #     'Year': 'first'
# # }).reset_index()
#
#
# # hosts
# print('每年的东道主：\n', dfh)
#
#
# # medals / (year, country)
# medals_per_year = dfm.groupby(['Team', 'Year'])[['Gold', 'Silver', 'Bronze', 'Total']].sum()
# print('每年各国的奖牌数：\n', medals_per_year)
#
#
# # athletes / (year, country)
# athletes_per_year = dfa.groupby(['Year', 'Team'])['Team'].nunique().reset_index(name='num_participants')
# print('每年各国的运动员数：\n', athletes_per_year)
# # for index, row in athletes_per_year.iterrows():
#     # print(f"{row['Year']} {row['NOC']} {row['num_participants']}")
#
#
# # sports / (year, country)
# sports_per_year = dfa.groupby(['Year', 'Team'])['Sport'].nunique().reset_index(name='num_events')
# print('每年各国的项目数：\n', sports_per_year)
# # for index, row in sports_per_year.iterrows():
#     # print(f"{row['Year']} {row['NOC']} {row['num_events']}")
#
#
#
# # [COMMON SPORTS IN CHN/USA/RUS]
# df = dfa
# df_filtered = df[df['NOC'].isin(['CHN', 'USA', 'RUS'])]
# grouped = df_filtered.groupby(['Year', 'Sport'])['NOC'].nunique()
# result = grouped[grouped == 3].reset_index()
# print('每年中美俄三国重合项目：\n', result)
#
#
#
# # [TOP PERCENTAGE SPORTS]
# dftmp = dfa
# # dftmp = dftmp[dftmp['Team'].isin(['China', 'United States', 'Russia'])]
# dftmp['Medal_Count'] = 1
# country_year_stats = dftmp.groupby(['Year', 'Team', 'Sport']).agg({'Medal_Count': 'sum'}).reset_index()
# # calc total medals
# total_medals_per_country_year = dftmp.groupby(['Year', 'Team']).agg({'Medal_Count': 'sum'}).reset_index()
# # merging
# merged = pd.merge(country_year_stats, total_medals_per_country_year, on=['Year', 'Team'], suffixes=('_sport', '_total'))
# merged['Percentage'] = (merged['Medal_Count_sport'] / merged['Medal_Count_total']) * 100
# # filtering top 5
# top_sports = merged.groupby(['Year', 'Team']).apply(lambda x: x.nlargest(5, 'Percentage')).reset_index(drop=True)
#
# top_sports = top_sports.groupby(['Year', 'Team']).apply(
#     lambda x: {sport: perc for sport, perc in zip(x['Sport'], x['Percentage'])}
# ).reset_index(name='Sport_Percentage')
#
# print('每年各国奖牌贡献项目占比：\n', top_sports)
# # # Output
# # for year in top_sports['Year'].unique():
# #     for country in top_sports['NOC'].unique():
# #         print(f"\n{year} {country}:")
# #         sports = top_sports[(top_sports['Year'] == year) & (top_sports['NOC'] == country)]
# #         for _, row in sports.iterrows():
# #             print(f"  - {row['Sport']}: {row['Percentage']:.2f}%")
#
#
#
# # [K AWARDS IN A ROW]
# K = 5
# dftmp2 = dfa
# dftmp2 = dftmp2.sort_values(by=['Name', 'Year'])
# consecutive_winners = []
# for name, group in dftmp2.groupby('Name'):
#     years = group['Year'].values
#     for i in range(len(years) - (K - 1)):
#         if years[i + K - 1] == years[i] + (4 * (K - 1)):
#             consecutive_winners.append(name)
#             break
# print(f"至少连续{K}届奥运会得奖的运动员:")
# for winner in set(consecutive_winners):
#     print(winner)






# =====================================================
#                 K-MOST-IMPORTANT GAMES
# =====================================================

# pd.set_option('display.max_rows', 10)
# merged_df = pd.merge(sports_per_year, athletes_per_year , on=['Year', 'Team'])
# merged_df = pd.merge(medals_per_year, merged_df , on=['Year', 'Team'])
# merged_df = pd.merge(top_sports, merged_df , on=['Year', 'Team'])
# merged_df = pd.merge(merged_df, dfh, on='Year', how='left')
# merged_df['is_host'] = (merged_df['Host'] == merged_df['Team']).astype(int)
# merged_df.drop('Host', axis=1, inplace=True)
# print(merged_df)
# print("==================================================")
# from statistics import LinearRegression
#
# u = pd.read_csv('medals.csv')
#
# u['weight'] = u['Gold'] * 100 + u['Silver'] * 5 + u['Bronze'] * 2
#
# pivot_df = u.pivot_table(values='weight', index='Year', columns='Team', aggfunc='sum')
# pivot_df = pivot_df.fillna(0)
# print(pivot_df)
# pivot_df.to_csv('u.csv')
#
#
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# s = pd.read_csv('s.csv')
# t = pd.read_csv('t.csv')
#
# s.set_index('Year', inplace=True)
# t.set_index('Year', inplace=True)
#
# X = s.drop(columns=['Total disciplines', 'Total events', 'Total sports'])
# y = t
#
# important_projects_entropy = {}
#
# for country in y.columns:
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X, y[country])
#     feature_importances = model.feature_importances_
#     total_importance = np.sum(feature_importances)
#     normalized_importance = feature_importances / total_importance
#     entropy_values = -normalized_importance * np.log(normalized_importance + 1e-10)
#     entropy = np.sum(entropy_values)
#     entropy_weight = (1 - entropy) / np.sum(1 - entropy)
#
#     sorted_indices = np.argsort(feature_importances)[::-1]
#     sorted_features = X.columns[sorted_indices]
#     sorted_importances = feature_importances[sorted_indices]
#
#     important_projects_entropy[country] = []
#     for i in range(5):
#         feature = sorted_features[i]
#         importance = sorted_importances[i]
#         entropy = entropy_values[i]
#         entropy_weight = (1 - entropy) / np.sum(1 - entropy_values)
#         important_projects_entropy[country].append((feature, importance, entropy_weight))
#
# for country, important_features in important_projects_entropy.items():
#     if country in ['China', 'United States', 'Cuba', 'Norway']:
#         print(f"{country} 最重要的5个比赛项目及其熵权：")
#         for feature, importance, entropy_weight in important_features:
#             print(f"  - {feature}: 重要性 = {importance:.4f}, 熵权 = {entropy_weight:.4f}")
#         print("\n")








# import pandas as pd
#
# df_medals = pd.read_csv('medals.csv')
# df_scores = pd.read_csv('t.csv')
#
# def calculate_weighted_score(row):
#     return row['Gold'] * 10 + row['Silver'] * 5 + row['Bronze'] * 2
#
# for _, row in df_medals.iterrows():
#     year = row['Year']
#     team = row['Team']
#     score = calculate_weighted_score(row)
#
#     if year not in df_scores['Year'].values:
#         df_scores = df_scores.append({'Year': year, team: score}, ignore_index=True)
#     else:
#         df_scores.loc[df_scores['Year'] == year, team] = score
#
# print(df_scores)
# df_scores.to_csv('t.csv', index=False)








# v = pd.read_csv('athletes.csv')
#
#
# award_teams = v[v['Medal'].isin(['Gold', 'Silver', 'Bronze'])]
# award_teams_list = award_teams['Team'].unique()
# teams_list = v['Team'].unique()
# no_award_teams_list = set(teams_list) - set(award_teams_list)
# print(no_award_teams_list)
#
#
# no_award_teams_df = v[v['Team'].isin(no_award_teams_list)]
# team_counts = no_award_teams_df['Team'].value_counts()
# print(team_counts)
#
#
# no_award_teams_df = no_award_teams_df.sort_values(by=['Team', 'Year'])
# year_diff = no_award_teams_df.groupby('Team').agg(
#     first_year=('Year', 'min'),
#     last_year=('Year', 'max')
# )
# year_diff['year_diff'] = year_diff['last_year'] - year_diff['first_year']
# print(year_diff[['year_diff']])
#
#
# year_diff.to_csv('year_diff.csv')
# team_counts_df = team_counts.reset_index()
# team_counts_df.columns = ['Team', 'count']
# merged_df = pd.merge(team_counts_df, year_diff, on='Team', how='inner')
# merged_df = merged_df.drop(columns=['first_year', 'last_year'])
# print(merged_df)
# merged_df.to_csv('q.csv')






# =====================================================
#         FIRST-MEDAL PREDICTING - PRE-PROCESSING
# =====================================================


# award_teams_df = award_teams
# award_teams_df = award_teams_df.sort_values(by=['Team', 'Year'])
# award_teams_df['participation_count_before_award'] = award_teams_df.groupby('Team').cumcount()
# award_teams_df.to_csv('award_teams.csv')
#
# first_award_df = award_teams_df.drop_duplicates('Team', keep='first')
# first_participation_year = df.groupby('Team')['Year'].min()
# result = pd.merge(first_award_df[['Team', 'Year', 'participation_count_before_award']],
#                   first_participation_year,
#                   left_on='Team',
#                   right_index=True,
#                   how='inner')
#
# print(result)
#
# result['year_diff'] = result['Year_x'] - result['Year_y']
# result = result[['Team', 'participation_count_before_award', 'Year_x', 'year_diff']]
# result.columns = ['Team', 'participation_count_before_award', 'award_year', 'first_participation_year_diff']
# # result.to_csv('award_teams.csv')
# print(result)








# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
#
# df = pd.read_csv('q.csv')
#
# X = df[['count', 'year_diff']]
# y = df['tag']
#
# if y.nunique() < 2:
#     print("ERROR!")
# else:
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     rf = RandomForestClassifier(random_state=42)
#     rf.fit(X_scaled, y)
#     probs = rf.predict_proba(X_scaled[y == 0])[:, 1]
#     top_10_indices = probs.argsort()[-15:][::-1]
#     tag_0_indices = df.index[y == 0]
#     pd.set_option('display.max_rows', None)
#     print("top 10 with probability:")
#
#     result = df.iloc[tag_0_indices[top_10_indices]].copy()
#     result['probability'] = probs[top_10_indices]
#     print(result[['Team', 'probability']])
#

#     feature_importances = rf.feature_importances_
#     feature_importance_df = pd.DataFrame({
#         'feature': X.columns,
#         'importance': feature_importances
#     })
#     feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
#     print("特征重要性顺序：")
#     print(feature_importance_df)






# =====================================================
#                 DID-TESTING
# =====================================================

# import pandas as pd
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
#
# # 假设 df 是你的数据框
# # 示例数据
# data = {
#     'country': ['USA', 'USA', 'USA', 'China', 'China', 'China', 'Russia', 'Russia', 'Russia'],
#     'year': [2016, 2020, 2024, 2016, 2020, 2024, 2016, 2020, 2024],
#     'sport': ['Volleyball', 'Volleyball', 'Volleyball', 'Gymnastics', 'Gymnastics', 'Gymnastics', 'Track', 'Track', 'Track'],
#     'medals': [40, 50, 55, 25, 30, 35, 20, 22, 24],
#     'coach_change': [0, 1, 1, 0, 1, 1, 0, 0, 0],  # 假设 2020 和 2024 年教练发生变动
#     'treatment': [1, 1, 1, 1, 1, 1, 0, 0, 0]  # 处理组: 美国和中国，未发生变动的是俄罗斯
# }
#
# # 创建 DataFrame
# df = pd.DataFrame(data)
#
# # 设置一个“前后差分”变量，表示某年是否在教练变动之后
# df['after_change'] = df['year'].apply(lambda x: 1 if x >= 2020 else 0)
#
# # 创建DID交互项
# df['DID'] = df['after_change'] * df['treatment']
#
# # 查看数据
# print(df)
#
# # 使用差分中差分模型（DID回归）
# model = smf.ols('medals ~ after_change + treatment + DID', data=df).fit()
#
# # 输出结果
# print(model.summary())






# =====================================================
#                 RDD-TESTING
# =====================================================

# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.sandbox.regression.gmm import IV2SLS
#
# # 数据
# data = {
#     'year': [2012, 2016, 2020, 2024, 2012, 2016, 2020, 2024, 2012, 2016, 2020, 2024],
#     'country': ['USA', 'USA', 'USA', 'USA', 'China', 'China', 'China', 'China', 'Russia', 'Russia', 'Russia', 'Russia'],
#     'medals': [40, 45, 50, 55, 25, 30, 35, 40, 20, 22, 24, 26],
#     'coach_change': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
#     'sport': ['Volleyball', 'Volleyball', 'Volleyball', 'Volleyball', 'Gymnastics', 'Gymnastics', 'Gymnastics', 'Gymnastics', 'Track', 'Track', 'Track', 'Track']
# }
#
# df = pd.DataFrame(data)
# print(df)
#
# # 处理变量 - Sharp RDD: 以2020年为断点
# df['after_change'] = (df['year'] >= 2020).astype(int)
#
# # Sharp RDD 模型：假设教练更替在2020年发生，检验教练更替对奖牌数的影响
# model_sharp = ols('medals ~ after_change * coach_change', data=df).fit()
#
# # 输出 Sharp RDD 模型结果
# print("Sharp RDD 模型结果:")
# print(model_sharp.summary())
#
# # Fuzzy RDD 模型：使用工具变量（instrumental variable, IV）进行处理
# # 假设教练更替受到一些外部因素影响，使用 'after_change' 作为工具变量
#
# # Fuzzy RDD 模型：工具变量回归
# df['instrument'] = (df['year'] >= 2020).astype(int)  # 工具变量（instrument）
#
# model_fuzzy = IV2SLS.from_formula('medals ~ coach_change + after_change | instrument', data=df).fit()
#
# # 输出 Fuzzy RDD 模型结果
# print("\nFuzzy RDD 模型结果:")
# print(model_fuzzy.summary())







# =====================================================
#                 2008, 2012 ..., 2024
# =====================================================

# import pandas as pd
#
# df = pd.read_csv('athletes.csv')
# medal_points = {'No medal': 0, 'Bronze': 2, 'Silver': 5, 'Gold': 10}
# df['Score'] = df['Medal'].map(medal_points)
# df_filtered = df[df['Year'].isin([2008, 2012, 2016, 2020, 2024])]
#
# sports = {'Badminton', 'Equestrianism', 'Fencing', 'Gymnastics', 'Judo', 'Shooting', 'Swimming', 'Tennis', 'Volleyball'}
# df_filtered = df_filtered[df_filtered['Sport'].isin(sports)]
# total_scores = df_filtered.groupby(['Sport', 'Team'])['Score'].sum().reset_index()
# top_teams = total_scores.groupby('Sport').apply(lambda x: x.nlargest(3, 'Score')).reset_index(drop=True)
#
# top_teams.to_csv('w.csv')
# print(top_teams)








# import pandas as pd
#
# df = pd.read_csv('f.csv')
# medal_points = {'No medal': 0, 'Bronze': 2, 'Silver': 5, 'Gold': 10}
# df['Score'] = df['Medal'].map(medal_points)
#
# gold_count = df[df['Medal'] == 'Gold'].groupby(['Year', 'Team', 'Sport']).size().reset_index(name='Gold')
# medal_count = df[df['Medal'].isin(['Gold', 'Silver', 'Bronze'])].groupby(['Year', 'Team', 'Sport']).size().reset_index(name='Medal Count')
# df_bronze = df[df['Medal'] == 'Bronze'].groupby(['Year', 'Team', 'Sport']).size().reset_index(name='Bronze')
# df_silver = df[df['Medal'] == 'Silver'].groupby(['Year', 'Team', 'Sport']).size().reset_index(name='Silver')
# df_gold = df[df['Medal'] == 'Gold'].groupby(['Year', 'Team', 'Sport']).size().reset_index(name='Gold')
#
# result = gold_count.merge(medal_count, on=['Year', 'Team', 'Sport'], how='outer')
# result = result.merge(df_bronze, on=['Year', 'Team', 'Sport'], how='outer')
# result = result.merge(df_silver, on=['Year', 'Team', 'Sport'], how='outer')
# result['Custom Score'] = 2 * result['Bronze'].fillna(0) + 5 * result['Silver'].fillna(0) + 10 * result['Gold'].fillna(0)
#
# result = result.fillna(0)
# result.to_csv('c.csv', index=False)
# print(result)






# =====================================================
#                 SHARP-RDD TESTING
# =====================================================

# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.discrete.discrete_model import Logit
#
# # 读取数据
# data = pd.read_csv('c.csv')
#
# # 打印数据的前几行，确认数据结构
# print(data.head())
#
# # 1. Sharp RDD模型：假设Change为0/1，分析教练变化的影响
# # 在一个模型中同时考虑Treatment和Change的交互项
# data['post_treatment'] = data['Change'] * data['Treatment']  # 交互项
#
# # 使用OLS回归进行Sharp RDD分析
# X_sharp = sm.add_constant(data[['Treatment', 'Change', 'post_treatment']])  # 添加常数项
# y_sharp = data['Total']  # 假设Gold为因变量
#
# model_sharp = sm.OLS(y_sharp, X_sharp).fit()
# print("Sharp RDD模型分析")
# print(model_sharp.summary())
#
# # 2. Fuzzy RDD模型：估计Change的发生概率（使用Logit回归）
# # 假设使用Score来预测Change的概率
# X = sm.add_constant(data['Total'])  # 选择一个特征来预测Change的概率
# y = data['Change']
#
# logit_model = Logit(y, X).fit()
# print("Fuzzy RDD模型（Logit回归估计Change概率）")
# print(logit_model.summary())
#
# # 3. DID模型：通过Treatment和Change的交互项分析教练变化的效应
# data['post_treatment'] = data['Change'] * data['Treatment']
#
# # DID模型：回归分析，包括Treatment、Change和交互项
# X_did = sm.add_constant(data[['Treatment', 'Change', 'post_treatment']])
# y_did = data['Total']  # 或者Score、Total等
#
# model_did = sm.OLS(y_did, X_did).fit()
# print("DID模型分析")
# print(model_did.summary())








# import pandas as pd
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
#
# # # 假设 df 是你的数据框
# # # 示例数据
# # data = {
# #     'country': ['USA', 'USA', 'USA', 'China', 'China', 'China', 'Russia', 'Russia', 'Russia'],
# #     'year': [2016, 2020, 2024, 2016, 2020, 2024, 2016, 2020, 2024],
# #     'sport': ['Volleyball', 'Volleyball', 'Volleyball', 'Gymnastics', 'Gymnastics', 'Gymnastics', 'Track', 'Track', 'Track'],
# #     'medals': [20, 50, 24, 20, 50, 24, 20, 20, 20],  # 变化幅度增大
# #     'coach_change': [0, 1, 0, 0, 1, 0, 0, 0, 0],  # 教练变动标识
# #     'treatment': [1, 1, 1, 1, 1, 1, 0, 0, 0]  # 处理组与对照组
# # }
# # # 创建 DataFrame
# # df = pd.DataFrame(data)
#
# df = pd.read_csv('c.csv')
#
# # 创建DID交互项，直接利用coach_change来标记是否发生教练变动
# df['DID'] = df['Change'] * df['Treatment']
#
# # 查看数据
# print(df)
#
# # 使用差分中差分模型（DID回归）
# model = smf.ols('Total ~ Change + Treatment + DID', data=df).fit()
#
# # 输出结果
# print(model.summary())







# =====================================================
#         LINEAR REG FOR GREAT COACHES, USELESS
# =====================================================

# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.impute import SimpleImputer
#
# # 假设你已经有了一个DataFrame
# data = pd.read_csv("e.csv")  # 替换为你的数据文件路径
#
# # 选择特征和目标变量
# X = data[['Coach', 'Host', 'Events', 'Athletes', 'GDP']]
# y = data['Score']
#
# # 使用 SimpleImputer 填充缺失值
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)
#
# # 归一化（Min-Max归一化）
# scaler = StandardScaler()
# X_imputed = scaler.fit_transform(X_imputed)
#
# # 将数据分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
#
# # 线性回归建模
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # 预测
# y_pred = model.predict(X_test)
#
# # 输出各个特征的系数（影响程度）
# features = X.columns  # 获取特征的名称
# coefficients = model.coef_  # 获取回归系数
#
# for feature, coef in zip(features, coefficients):
#     print(f"{feature}: {coef}")






# =====================================================
#                       ENTROPY
# =====================================================

# import pandas as pd
# import numpy as np
# from scipy.stats import entropy
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import mean_squared_error, r2_score
#
# df = pd.read_csv('p.csv')
# def calculate_entropy(df):
#     entropies = {}
#     for column in df.columns:
#         if column != 'Coach':
#             normalized_column = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
#             value_counts = normalized_column.value_counts(normalize=True)
#             entropies[column] = entropy(value_counts)
#     return entropies
#
# entropy_values = calculate_entropy(df)
# print(f"Column-wise Entropy values: {entropy_values}")
#
# total_entropy = sum([1 - value for value in entropy_values.values()])
# entropy_weight = {column: (1 - value) / total_entropy for column, value in entropy_values.items()}
# entropy_weight_series = pd.Series(entropy_weight, index=df.drop(columns=['Coach', 'Score']).columns)
#
# weighted_features = df.drop(columns=['Coach', 'Score']).multiply(entropy_weight_series, axis=1)
# weighted_features['Coach'] = df['Coach']
# weighted_features['Score'] = df['Score']
#
#
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# X = weighted_features.drop(columns=['Score'])
# y = weighted_features['Score']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# imputer = SimpleImputer(strategy='mean')
# X_train = imputer.fit_transform(X_train)
# X_test = imputer.transform(X_test)
#
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# coefficients = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
# print(coefficients)
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Root Mean Squared Error (RMSE): {rmse}")
# r2 = r2_score(y_test, y_pred)
# print(f"R² (R-squared): {r2}")






# =====================================================
#                 TUNING PARAM K
# =====================================================

# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
#
# data = {
#     'Gold': [1, 2, 1, 3, 0],
#     'Total': [5, 10, 3, 7, 8],
#     'Score': [80, 90, 70, 85, 75],
#     'Host': [1, 0, 1, 0, 1],
#     'Events': [20, 30, 15, 25, 28],
#     'Athletes': [50, 100, 80, 60, 55],
#     'GDP': [2000, 5000, 3000, 4000, 2500],
#     'Coach': [5, 7, 6, 5, 6],
# }
#
# # df = pd.DataFrame(data)
# df = pd.read_csv('p.csv')
#
# entropy_weights = { # to be changed
#     'Gold': 2.2614717525046624,
#     'Total': 3.1019693773510175,
#     'Score': 4.386350853115759,
#     'Host': 0.175364662501861,
#     'Events': 2.875821983722058,
#     'Athletes': 4.874462777559124,
#     'GDP': 4.132099866102211
# }
#
# weighted_df = df.drop(columns=['Coach', 'Gold', 'Score', 'Total'])
#
# for column in weighted_df.columns:
#     weighted_df[column] = weighted_df[column] * entropy_weights[column]
#
# k = 1
#
# weighted_df['Coach_weighted'] = df['Coach'] * k
# X = weighted_df
# y = df['Total']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# imputer = SimpleImputer(strategy='mean')
#
# X_train = imputer.fit_transform(X_train)
# X_test = imputer.transform(X_test)
#
# model = GradientBoostingRegressor()
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
#
# print(f"RMSE: {rmse}")
# print(f"R²: {r2}")
#
# best_k = k
# best_rmse = rmse
# best_r2 = r2
#
# k_values = np.linspace(0.1, 5.0, 50)
#
# for k in k_values:
#     weighted_df['Coach_weighted'] = df['Coach'] * k
#     X = weighted_df
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     current_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     current_r2 = r2_score(y_test, y_pred)
#
#     if current_rmse < best_rmse:
#         best_rmse = current_rmse
#         best_r2 = current_r2
#         best_k = k
#
# print(f"Best k: {best_k}")
# print(f"Best RMSE: {best_rmse}")
# print(f"Best R²: {best_r2}")








# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# from xgboost import XGBRegressor
#
# df = pd.read_csv('h.csv')
#
#
#
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # 假设df是你的DataFrame
# # 设置熵权（假设你已经有了这三个熵权值，分别是Athletes、Events和GDP的熵权）
# entropy_weights = {
#     # 'Coach': 0.36905,
#     # 'Host': 0.47857,
#     # 'Events': 0.04029,
#     # 'Athletes': 0.01086,
#     # 'GDP': 0.10123,
#
#     'Events': 0.26440,
#     'Athletes': 0.07127,
#     'GDP': 0.66433,
#
#     # 'Events': 2.875821983722058,
#     # 'Athletes': 4.874462777559124,
#     # 'GDP': 4.132099866102211,
# }
#
# # # 计算熵权的总和
# # total_weight = sum(entropy_weights.values())
# #
# # # 归一化熵权，使总和为1
# # for key in entropy_weights:
# #     entropy_weights[key] /= total_weight
#
#
# # 定义一个函数计算weighted，自变量
# def compute_weighted(df, p, q):
#     return (
#             df['Athletes'] * entropy_weights['Athletes'] +
#             df['Events'] * entropy_weights['Events'] +
#             df['GDP'] * entropy_weights['GDP'] +
#             df['Coach'] * p +
#             df['Host'] * q
#     )
#
#
# # 对 Score 列进行差分操作
# df['Score_diff'] = df.groupby(['Team', 'Sport'])['Score'].diff()
#
# # 删除差分后的 NaN 行（第一个年份的差分会是 NaN）
# df = df.dropna(subset=['Score_diff'])
#
# # 设置自变量（X）和目标变量（y）
# X = df[['Athletes', 'Events', 'GDP', 'Coach', 'Host']]  # 自变量，包含所有必要的列
# y = df['Score_diff']  # 目标变量是差分后的 Score
#
#
# # # 设置目标变量
# # X = df[['Athletes', 'Events', 'GDP', 'Coach', 'Host']]  # 自变量，包含所有必要的列
# # y = df['Score']  # 目标变量
#
# # 使用均值填充缺失值
# imputer = SimpleImputer(strategy='mean')
# X = imputer.fit_transform(X)
# X = pd.DataFrame(X, columns=['Athletes', 'Events', 'GDP', 'Coach', 'Host'])
# print(X)
#
#
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
# X = pd.DataFrame(X, columns=['Athletes', 'Events', 'GDP', 'Coach', 'Host'])
# print(X)
#
# vif_data = pd.DataFrame()
# vif_data["Variable"] = X.columns
# vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#
# print(vif_data)
#
# # 初始化模型
# model = XGBRegressor()
# PROMPT = 'XGB Regressor'
#
# # model = GradientBoostingRegressor()
# # PROMPT = 'Gradient Boosting Regressor'
#
# # model = SVR(kernel='rbf', C=1000, epsilon=0.01)
# # PROMPT = 'SVM Regression (kernel=\'rbf\', C=1000, epsilon=0.01)'
#
# # model = KNeighborsRegressor(n_neighbors=3)
# # PROMPT = 'KNN Regression (n_neighbors=3)'
#
# # model = RandomForestRegressor(n_estimators=100, random_state=0)
# # PROMPT = 'Random Forest Regressor (n_estimators=100)'
#
# # 定义一个GridSearchCV来优化p和q
# def custom_score(X, y, p, q):
#     # 计算weighted
#     X['weighted'] = compute_weighted(X, p, q)
#     # 使用GBR模型进行拟合和预测
#     model.fit(X[['weighted']], y)
#     y_pred = model.predict(X[['weighted']])
#
#     # 计算MSE
#     mse = mean_squared_error(y, y_pred)
#
#     # 计算R²
#     r2 = r2_score(y, y_pred)
#
#     return mse, r2
#
#
# # 进行网格搜索
# best_mse = float('inf')
# best_r2 = float('-inf')
# best_p = None
# best_q = None
#
# for p_value in np.linspace(0, 4.2, 42):
#     for q_value in np.linspace(0, 4.2, 42):
#         mse, r2 = custom_score(X, y, p_value, q_value)
#         if mse < best_mse:
#             best_mse = mse
#             best_r2 = r2
#             best_p = p_value
#             best_q = q_value
#
# # for p_value in np.linspace(4.097560975609756, 4.097560975609757, 1):
# #     for q_value in np.linspace(1.2341463414634146, 1.2341463414634146, 1):
# #         mse, r2 = custom_score(X, y, p_value, q_value)
# #         if mse < best_mse:
# #             best_mse = mse
# #             best_r2 = r2
# #             best_p = p_value
# #             best_q = q_value
#
# print(PROMPT)
# print(f"Best p: {best_p}, Best q: {best_q}")
# print(f"Best MSE: {best_mse}")
# print(f"Best R²: {best_r2}")






# =====================================================
#                       2028
# =====================================================

# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from xgboost import XGBRegressor
#
# coefs = {
#     'Coach': 1.12683,
#     'Host': 0.20488,
#     'Events': 0.26440,
#     'Athletes': 0.07127,
#     'GDP': 0.66433,
# }
#
# data = pd.read_csv('x.csv')
#
# #     scaler = StandardScaler()
# #     X_scaled = scaler.fit_transform(X)
#
# data['Coach'] *= coefs['Coach']
# data['Host'] *= coefs['Host']
# data['Events'] *= coefs['Events']
# data['Athletes'] *= coefs['Athletes']
# data['GDP'] *= coefs['GDP']
# data['weighed'] = data['Coach'] + data['Host'] + data['Events'] + data['Athletes'] + data['GDP']
#
# train_data = data[data['Year'] < 2028]
# test_data = data[data['Year'] == 2028]
#
# features = ['Coach', 'Host', 'Events', 'Athletes', 'GDP']
# target = ['Gold', 'Total', 'Score']
#
# train_data_encoded = pd.get_dummies(train_data[features], drop_first=True)
# test_data_encoded = pd.get_dummies(test_data[features], drop_first=True)
#
# train_data_encoded, test_data_encoded = train_data_encoded.align(test_data_encoded, join='left', axis=1)
#
# train_data_encoded.fillna(0, inplace=True)
# test_data_encoded.fillna(0, inplace=True)
#
#
# # scaler = MinMaxScaler()
# #
# # train_data_encoded[train_data_encoded.columns] = scaler.fit_transform(train_data_encoded[train_data_encoded.columns])
# # test_data_encoded[test_data_encoded.columns] = scaler.transform(test_data_encoded[test_data_encoded.columns])
#
# # model_gold = XGBRegressor(max_depth=5, n_estimators=100, random_state=42)
# # model_total = XGBRegressor(max_depth=5, n_estimators=100, random_state=42)
# # model_score = XGBRegressor(max_depth=5, n_estimators=100, random_state=42)
# model_gold = GradientBoostingRegressor()
# model_total = GradientBoostingRegressor()
# model_score = GradientBoostingRegressor()
#
# model_gold.fit(train_data_encoded, train_data['Gold'])
# model_total.fit(train_data_encoded, train_data['Total'])
# model_score.fit(train_data_encoded, train_data['Score'])
#
# real_2024_data = train_data[['Team', 'Sport', 'Gold', 'Total', 'Score']]
# predictions = []
#
# for index, row in test_data.iterrows():
#     modified_test_data = test_data.copy()
#     modified_test_data.loc[modified_test_data['Year'] == 2028, 'Coach'] = 0
#     modified_test_data.loc[index, 'Coach'] = row['Events']
#     modified_test_data_encoded = pd.get_dummies(modified_test_data[features], drop_first=True)
#     modified_test_data_encoded, _ = modified_test_data_encoded.align(test_data_encoded, join='left', axis=1)
#     modified_test_data_encoded.fillna(0, inplace=True)
#     print(modified_test_data_encoded)
#
#     predicted_gold = model_gold.predict(modified_test_data_encoded.loc[[index]])
#     predicted_total = model_total.predict(modified_test_data_encoded.loc[[index]])
#     predicted_score = model_score.predict(modified_test_data_encoded.loc[[index]])
#
#     # predictions.append({
#     #     'Team': row['Team'],
#     #     'Sport': row['Sport'],
#     #     'Predicted_Gold': predicted_gold[0],
#     #     'Predicted_Total': predicted_total[0],
#     #     'Predicted_Score': predicted_score[0]
#     # })
#
#     real_values_2024 = real_2024_data[
#         (real_2024_data['Team'] == row['Team']) & (real_2024_data['Sport'] == row['Sport'])]
#
#     # delta
#     if not real_values_2024.empty:
#         real_gold_2024 = real_values_2024['Gold'].values[0]
#         real_total_2024 = real_values_2024['Total'].values[0]
#         real_score_2024 = real_values_2024['Score'].values[0]
#
#         diff_gold = predicted_gold[0] - real_gold_2024
#         diff_total = predicted_total[0] - real_total_2024
#         diff_score = predicted_score[0] - real_score_2024
#     else:
#         diff_gold = np.nan
#         diff_total = np.nan
#         diff_score = np.nan
#
#     predictions.append({
#         'Team': row['Team'],
#         'Sport': row['Sport'],
#         'Diff_Gold': diff_gold,
#         'Diff_Total': diff_total,
#         'Diff_Score': diff_score
#     })
#
# predictions_df = pd.DataFrame(predictions)
# predictions_df.to_csv('z.csv')
# print(predictions_df)






# =====================================================
#                       BACKUP
# =====================================================

# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# import numpy as np
# from xgboost import XGBRegressor
#
# # 读取数据
# data = pd.read_csv('x.csv')
#
# # 处理数据：过滤出2024年及以前的数据进行训练
# train_data = data[data['Year'] < 2028]
#
# # 用 2028 年的数据作为测试集
# test_data = data[data['Year'] == 2028]
#
# # 特征和目标变量
# features = ['Coach', 'Host', 'Events', 'Athletes', 'GDP']
# target = ['Gold', 'Total', 'Score']
#
# # 对目标变量进行差分
# train_data['Gold_diff'] = train_data.groupby(['Team', 'Sport'])['Gold'].diff()
# train_data['Total_diff'] = train_data.groupby(['Team', 'Sport'])['Total'].diff()
# train_data['Score_diff'] = train_data.groupby(['Team', 'Sport'])['Score'].diff()
#
# # 对目标变量进行差分后，移除 NaN（因为第一年的差分值是NaN）
# train_data = train_data.dropna(subset=['Gold_diff', 'Total_diff', 'Score_diff'])
#
# # 对特征进行处理，因变量是离散或需要编码的（例如：Coach, Host等）
# train_data_encoded = pd.get_dummies(train_data[features], drop_first=True)
# test_data_encoded = pd.get_dummies(test_data[features], drop_first=True)
#
# # 确保训练和测试数据的特征一致
# train_data_encoded, test_data_encoded = train_data_encoded.align(test_data_encoded, join='left', axis=1)
#
# # 填充缺失值
# train_data_encoded.fillna(0, inplace=True)
# test_data_encoded.fillna(0, inplace=True)
#
# # 使用 RandomForestRegressor 训练模型
# model_gold = XGBRegressor(max_depth=5, n_estimators=100, random_state=42)
# model_total = XGBRegressor(max_depth=5, n_estimators=100, random_state=42)
# model_score = XGBRegressor(max_depth=5, n_estimators=100, random_state=42)
#
# # 分别训练 Gold_diff, Total_diff, Score_diff 的回归模型
# model_gold.fit(train_data_encoded, train_data['Gold_diff'])
# model_total.fit(train_data_encoded, train_data['Total_diff'])
# model_score.fit(train_data_encoded, train_data['Score_diff'])
#
# # 获取2024年的真实数据，便于后续计算差值
# real_2024_data = train_data[['Team', 'Sport', 'Gold', 'Total', 'Score']]
#
# # 为2028年的每一行进行一次回归预测
# predictions = []
#
# for index, row in test_data.iterrows():
#     # 对2028年数据进行处理：每次将该行的Coach列设置为Events列的值
#     modified_test_data = test_data.copy()
#     modified_test_data.loc[modified_test_data['Year'] == 2028, 'Coach'] = 0
#     modified_test_data.loc[index, 'Coach'] = row['Events']
#
#     # 对修改后的数据进行编码
#     modified_test_data_encoded = pd.get_dummies(modified_test_data[features], drop_first=True)
#
#     # 确保对齐
#     modified_test_data_encoded, _ = modified_test_data_encoded.align(test_data_encoded, join='left', axis=1)
#
#     # 填充缺失值
#     modified_test_data_encoded.fillna(0, inplace=True)
#
#     # 进行差分预测
#     predicted_gold_diff = model_gold.predict(modified_test_data_encoded.loc[[index]])
#     predicted_total_diff = model_total.predict(modified_test_data_encoded.loc[[index]])
#     predicted_score_diff = model_score.predict(modified_test_data_encoded.loc[[index]])
#
#     # 获取2024年的真实值
#     real_values_2024 = real_2024_data[
#         (real_2024_data['Team'] == row['Team']) & (real_2024_data['Sport'] == row['Sport'])]
#
#     # 计算差值
#     if not real_values_2024.empty:
#         real_gold_2024 = real_values_2024['Gold'].values[0]
#         real_total_2024 = real_values_2024['Total'].values[0]
#         real_score_2024 = real_values_2024['Score'].values[0]
#
#         # 计算2028年的预测差值与2024年实际差值之和
#         predicted_gold_2028 = real_gold_2024 + predicted_gold_diff[0]
#         predicted_total_2028 = real_total_2024 + predicted_total_diff[0]
#         predicted_score_2028 = real_score_2024 + predicted_score_diff[0]
#     else:
#         # 如果没有找到2024年对应的数据，可以设置为NaN或者0，取决于需求
#         predicted_gold_2028 = np.nan
#         predicted_total_2028 = np.nan
#         predicted_score_2028 = np.nan
#
#     # 保存差值
#     predictions.append({
#         'Team': row['Team'],
#         'Sport': row['Sport'],
#         'Predicted_Gold_2028': predicted_gold_2028,
#         'Predicted_Total_2028': predicted_total_2028,
#         'Predicted_Score_2028': predicted_score_2028
#     })
#
# # 将预测结果保存到 DataFrame 中
# predictions_df = pd.DataFrame(predictions)
#
# # 查看预测结果
# print(predictions_df)
# predictions_df.to_csv('j.csv')
