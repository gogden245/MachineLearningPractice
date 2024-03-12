# These are just some helpful functions

# PATH TO IMPORT:
  # import sys
  # sys.path.append('functions.py') # use actual file path
  # import functions as fun


# UNIVARIATE
def univariate(df):
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  df_output = pd.DataFrame(columns=['type', 'missing', 'unique', 'min', 'q1', 'median',
                                    'q3', 'max', 'mode', 'mean', 'std', 'skew', 'kurt'])

  for col in df:
    # Features that apply to all dtypes
    missing = df[col].isna().sum()
    unique = df[col].nunique()
    mode = df[col].mode()[0]
    if pd.api.types.is_numeric_dtype(df[col]):
      # Features for numeric only
      min = df[col].min()
      q1 = df[col].quantile(0.25)
      median = df[col].median()
      q3 = df[col].quantile(0.75)
      max = df[col].max()
      mean = df[col].mean()
      std = df[col].std()
      skew = df[col].skew()
      kurt = df[col].kurt()
      df_output.loc[col] = ["numeric", missing, unique, min, q1, median, q3, max, mode,
                            round(mean, 2), round(std, 2), round(skew, 2), round(kurt, 2)]
      sns.histplot(data=df, x=col)
      plt.show()
    else:
      df_output.loc[col] = ["categorical", missing, unique, '-', '-', '-', '-', '-',
                            mode, '-', '-', '-', '-']
      sns.countplot(data=df, x=col)
      plt.show()
  return df_output


# SCATTERPLOT
def scatterplot(df, feature, label, roundto=4, linecolor='darkorange'):
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from scipy import stats

  # Create the plot
  sns.regplot(x=df[feature], y=df[label], line_kws={"color":linecolor})

  # Calculate the regression line
  m, b, r, p, err = stats.linregress(df[feature], df[label])
  tau, tp = stats.kendalltau(df[feature], df[label])
  rho, rp = stats.spearmanr(df[feature], df[label])
  fskew = round(df[feature].skew(), roundto)
  lskew = round(df[label].skew(), roundto)

  # Add all of those values into the plot
  textstr = f'y = {round(m, roundto)}x + {round(b, roundto)}\n'
  textstr += f'r = {round(r, roundto)}, p = {round(p, roundto)}\n'
  textstr += f'τ = {round(tau, roundto)}, p = {round(tp, roundto)}\n'
  textstr += f'ρ = {round(rho, roundto)}, p = {round(rp, roundto)}\n'
  textstr += f'{feature} skew = {round(fskew, roundto)}\n'
  textstr += f'{label} skew = {round(lskew, roundto)}'

  plt.text(.95, 0.2, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()
  

# BAR CHART
def bar_chart(df, feature, label, roundto=4, p_threshold=0.05, sig_ttest_only=True):
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from scipy import stats

  # Make sure that the feature is categorical and label is numeric
  if pd.api.types.is_numeric_dtype(df[feature]):
    num = feature
    cat = label
  else:
    num = label
    cat = feature

  # Create the bar chart
  sns.barplot(x=df[cat], y=df[num])

  # Create the numeric lists needed to calcualte the ANOVA
  groups = df[cat].unique()
  group_lists = []
  for g in groups:
    group_lists.append(df[df[cat] == g][num])

  f, p = stats.f_oneway(*group_lists) # <- same as (group_lists[0], group_lists[1], ..., group_lists[n])

  # Calculate individual pairwise t-test for each pair of groups
  ttests = []
  for i1, g1 in enumerate(groups):
    for i2, g2 in enumerate(groups):
      if i2 > i1:
        list1 = df[df[cat]==g1][num]
        list2 = df[df[cat]==g2][num]
        t, tp = stats.ttest_ind(list1, list2)
        ttests.append([f'{g1} - {g2}', round(t, roundto), round(tp, roundto)])

  # Make a Bonferroni correction -> adjust the p-value threshold to be 0.05 / n of ttests
  bonferroni = p_threshold / len(ttests)

  # Create textstr to add statistics to chart
  textstr = f'F: {round(f, roundto)}\n'
  textstr += f'p: {round(p, roundto)}\n'
  textstr += f'Bonferroni p: {round(bonferroni, roundto)}'
  for ttest in ttests:
    if sig_ttest_only:
      if ttest[2] <= bonferroni:
        textstr +=f'\n{ttest[0]}: t:{ttest[1]}, p:{ttest[2]}'
    else:
      textstr +=f'\n{ttest[0]}: t:{ttest[1]}, p:{ttest[2]}'

  # If there are too many feature groups, print x labels vertically
  if df[feature].nunique() > 7:
    plt.xticks(rotation=90)

  plt.text(.95, 0.10, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()
  

# CROSSTAB
def crosstab(df, feature, label, roundto=4):
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from scipy import stats
  import numpy as np

  # Generate the crosstab
  crosstab = pd.crosstab(df[feature], df[label])

  # Calculate the statistics
  X, p, dof, contingency_table = stats.chi2_contingency(crosstab)

  textstr = f'X2: {X}\n'
  textstr += f'p: {p}'
  plt.text(.95, 0.2, textstr, fontsize=12, transform=plt.gcf().transFigure)

  ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), columns=crosstab.columns, index=crosstab.index)
  sns.heatmap(ct_df, annot=True, fmt='d', cmap='coolwarm')
  plt.show()
  

# BIVARIATE
def bivariate(df, label, roundto=4):
  import pandas as pd
  from scipy import stats

  output_df = pd.DataFrame(columns=['missing %', 'skew', 'type', 'unique', 'p', 'r', 'τ', 'ρ', 'y = m(x) + b', 'F', 'X2'])

  for feature in df:
    if feature != label:
      # Calculate stats that apply to all data types
      df_temp = df[[feature, label]].copy()
      df_temp = df_temp.dropna().copy()
      missing = round((df.shape[0] - df_temp.shape[0]) / df.shape[0], roundto) * 100
      dtype = df_temp[feature].dtype
      unique = df_temp[feature].nunique()
      if pd.api.types.is_numeric_dtype(df[feature]) and pd.api.types.is_numeric_dtype(df[label]):
        # Process N2N relationships
        m, b, r, p, err = stats.linregress(df_temp[feature], df_temp[label])
        tau, tp = stats.kendalltau(df_temp[feature], df_temp[label])
        rho, rp = stats.spearmanr(df_temp[feature], df_temp[label])
        skew = round(df[feature].skew(), roundto)
        output_df.loc[feature] = [f'{missing}%', skew, dtype, unique, round(p, roundto), round(r, roundto), round(tau, roundto),
                                  round(rho, roundto), f"y = {round(m, roundto)}x + {round(b, roundto)}", '-', '-']
        scatterplot(df_temp, feature, label)
      elif not pd.api.types.is_numeric_dtype(df_temp[feature]) and not pd.api.types.is_numeric_dtype(df_temp[label]):
        # Process C2C relationships
        contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
        X2, p, dof, expected = stats.chi2_contingency(contingency_table)
        output_df.loc[feature] = [f'{missing}%', '-', dtype, unique, p, '-', '-', '-', '-', '-', X2]
        crosstab(df_temp, feature, label)
      else:
        # Process C2N and N2C relationships
        if pd.api.types.is_numeric_dtype(df_temp[feature]):
          skew = round(df[feature].skew(), roundto)
          num = feature
          cat = label
        else:
          skew = '-'
          num = label
          cat = feature

        groups = df_temp[cat].unique()
        group_lists = []
        for g in groups:
          group_lists.append(df_temp[df_temp[cat] == g][num])

        f, p = stats.f_oneway(*group_lists) # <- same as (group_lists[0], group_lists[1], ..., group_lists[n])
        output_df.loc[feature] = [f'{missing}%', skew, dtype, unique, round(p, roundto), '-', '-', '-', '-', round(f, roundto), '-']
        bar_chart(df_temp, cat, num)


  return output_df.sort_values(by=['p'], ascending=True)


# DATA WRANGLING
def basic_wrangling(df, features=[], missing_threshold = 0.95, unique_threshold=0.95, messages = True):
    # remove columns with too many unique values
    # remove columns with too much missing data
    # remove columns with single values
    import pandas as pd 

    if len(features) == 0: features = df.columns

    for feat in features:
       if feat in df.columns:
          missing = df[feat].isna().sum()
          unique = df[feat].nunique()
          rows = df.shape[0]

          if missing / rows >= missing_threshold:
             if messages: print (f"Too much missing ({missing} out of {rows}, {round(missing/rows, 0)}) for {feat}")
             df.drop(columns=[feat], inplace=True)
          elif unique / rows >= unique_threshold:
             if df[feat].dtype in ['int64', 'object']:
                if messages: print(f"Too many unique values ({unique} out of {rows}, {round(unique/rows, 0)}) for {feat}")
                df.drop(columns=[feat], inplace=True)
          elif unique == 1:
              if messages: print(f"Only one value ({df[feat].unique()[0]}) for {feat}")
              df.drop(columns=[feat], inplace=True)
          else:
             if messages: print(f"The feature \"{feat}\" doesn't exist as spelled in the DataFrame provided")

    return df
    import pandas as pd

    for col in df:
      missing = df[col].isna().sum()
      unique = df[col].nunique()
      rows = df.shape[0]
      
      if missing / rows >= missing_threshold:
        df.drop(columns = [col], inplace = True)
        if messages: print(f'Column {col} dropped because of too much missing data ({round(missing/rows, 2)*100}%)')
      elif unique / rows >= unique_threshold:
        if df[col].dtype in ['object', 'int64']:
          df.drop(columns = [col], inplace = True)
          if messages: print(f'Column {col} dropped because of too many unique values ({round(unique/rows, 2)*100}%)')
      elif unique == 1:
        df.drop(columns = [col], inplace = True)
        if messages: print(f'Column {col} dropped because of only one value ({df[col].unique()[0]})')

    return df


# VIF
def vif(df):

  import pandas as pd
  from sklearn.linear_model import LinearRegression

  df_vif = pd.DataFrame(columns=['VIF'])

  for col in df:
    y = df[col]
    X = df.drop(columns=[col])
    r_squared = LinearRegression().fit(X, y).score(X, y)
    vif = 1 / (1 - r_squared) # VIF = 3 is the cutoff
    df_vif.loc[col] = vif

  return df_vif.sort_values(by=['VIF'], ascending=False)


# PARSE DATE
def parse_date(df, features=[], days_since_today=False, drop_date=True, messages=True):
  import pandas as pd
  from datetime import datetime as dt

  for feat in features:
    if feat in df.columns:
      df[feat] = pd.to_datetime(df[feat])
      df[f'{feat}_year'] = df[feat].dt.year
      df[f'{feat}_month'] = df[feat].dt.month
      df[f'{feat}_day'] = df[feat].dt.day
      df[f'{feat}_weekday'] = df[feat].dt.day_name()

      if days_since_today: df[f'{feat}_days_until_today'] = (dt.today() - df[feat]).dt.days
      if drop_date: df.drop(columns=[feat], inplace=True)
    else:
      if messages: print(f'{feat} does not exist in the DataFrame provided. No work performed')

  return df


# BIN CATEGORIES
def bin_categories(df, features=[], cutoff=0.05, replace_with='Other', messages=True):
  import pandas as pd

  if len(features) == 0: features = df.columns

  for feat in features:
    if feat in df.columns:
      if not pd.api.types.is_numeric_dtype(df[feat]):
        other_list = df[feat].value_counts()[df[feat].value_counts() / df.shape[0] < cutoff].index
        df.loc[df[feat].isin(other_list), feat] = replace_with
        if messages: print(f'{feat} has been binned by setting {other_list} to {replace_with}')
    else:
      if messages: print(f'{feat} not found in the DataFrame provided. No binning performed')

  return df


# CLEAN OUTLIER
def clean_outlier(df, features=[], method="remove", messages=True, skew_threshold=1):
  import pandas as pd, numpy as np

  for feat in features:
    if feat in df.columns:
      if pd.api.types.is_numeric_dtype(df[feat]):
        if df[feat].nunique() != 1:
          if not all(df[feat].value_counts().index.isin([0, 1])):
            skew = df[feat].skew()
            if skew < (-1 * skew_threshold) or skew > skew_threshold: # Tukey boxplot rule: < 1.5*IQR < is an outlier
              q1 = df[feat].quantile(0.25)
              q3 = df[feat].quantile(0.75)
              min = q1 - (1.5 * (q3 - q1))
              max = q3 + (1.5 * (q3 - q1))
            else:  # Empirical rule: any value > 3 std from the mean (or < 3) is an outlier
              min = df[feat].mean() - (df[feat].std() * 3)
              max = df[feat].mean() + (df[feat].std() * 3)

            min_count = df.loc[df[feat] < min].shape[0]
            max_count = df.loc[df[feat] > max].shape[0]
            if messages: print(f'{feat} has {max_count} values above max={max} and {min_count} below min={min}')

            if min_count > 0 or max_count > 0:
              if method == "remove": # Remove the rows with outliers
                df = df[df[feat] > min]
                df = df[df[feat] < max]
              elif method == "replace":   # Replace the outliers with the min/max cutoff
                df.loc[df[feat] < min, feat] = min
                df.loc[df[feat] > max, feat] = max
              elif method == "impute": # Impute the outliers by deleting them and then prediting the values based on a linear regression
                df.loc[df[feat] < min, feat] = np.nan
                df.loc[df[feat] > max, feat] = np.nan

                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                imp = IterativeImputer(max_iter=10)
                df_temp = df.copy()
                df_temp = bin_categories(df_temp, features=df_temp.columns, messages=False)
                df_temp = basic_wrangling(df_temp, features=df_temp.columns, messages=False)
                df_temp = pd.get_dummies(df_temp, drop_first=True)
                df_temp = pd.DataFrame(imp.fit_transform(df_temp), columns=df_temp.columns, index=df_temp.index, dtype='float')
                df_temp.columns = df_temp.columns.get_level_values(0)
                df_temp.index = df_temp.index.astype('int64')

                # Save only the column from df_temp that we are iterating on in the main loop because we may not want every new column
                df[feat] = df_temp[feat]
              elif method == "null":
                df.loc[df[feat] < min, feat] = np.nan
                df.loc[df[feat] > max, feat] = np.nan
          else:
            if messages: print(f'{feat} is a dummy code (0/1) and was ignored')
        else:
          if messages: print(f'{feat} has only one value ({df[feat].unique()[0]}) and was ignored')
      else:
        if messages: print(f'{feat} is categorical and was ignored')
    else:
      if messages: print(f'{feat} is not found in the DataFrame provided')

  return df


# CLEAN OUTLIERS
def clean_outliers(df, messages=True, drop_percent=0.02, distance='manhattan', min_samples=5):
  import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
  from sklearn.cluster import DBSCAN
  from sklearn import preprocessing

  # Clean the dataset first
  if messages: print(f"{df.shape[1] - df.dropna(axis='columns').shape[1]} columns were dropped first due to missing data")
  df.dropna(axis='columns', inplace=True)
  if messages: print(f"{df.shape[0] - df.dropna().shape[0]} rows were dropped first due to missing data")
  df.dropna(inplace=True)
  df_temp = df.copy()
  df_temp = bin_categories(df_temp, features=df_temp.columns, messages=False)
  df_temp = basic_wrangling(df_temp, features=df_temp.columns, messages=False)
  df_temp = pd.get_dummies(df_temp, drop_first=True)
  # Normalize the dataset
  df_temp = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df_temp), columns=df_temp.columns, index=df_temp.index)

  # Calculate the number of outliers based on a range of eps values
  outliers_per_eps = []
  outliers = df_temp.shape[0]
  eps = 0

  if df_temp.shape[0] < 500:
    iterator = 0.01
  elif df_temp.shape[0] < 2000:
    iterator = 0.05
  elif df_temp.shape[0] < 10000:
    iterator = 0.1
  elif df_temp.shape[0] < 25000:
    iterator = 0.2

  while outliers > 0:
    eps += iterator
    db = DBSCAN(metric=distance, min_samples=min_samples, eps=eps).fit(df_temp)
    outliers = np.count_nonzero(db.labels_ == -1)
    outliers_per_eps.append(outliers)
    if messages: print(f'eps: {round(eps, 2)}, outliers: {outliers}, percent: {round((outliers / df_temp.shape[0])*100, 3)}%')

  drops = min(outliers_per_eps, key=lambda x:abs(x-round(df_temp.shape[0] * drop_percent)))
  eps = (outliers_per_eps.index(drops) + 1) * iterator
  db = DBSCAN(metric=distance, min_samples=min_samples, eps=eps).fit(df_temp)
  df['outlier'] = db.labels_

  if messages:
    print(f"{df[df['outlier'] == -1].shape[0]} outlier rows removed from the DataFrame")
    sns.lineplot(x=range(1, len(outliers_per_eps) + 1), y=outliers_per_eps)
    sns.scatterplot(x=[eps/iterator], y=[drops])
    plt.xlabel(f'eps (divide by {iterator})')
    plt.ylabel('Number of Outliers')
    plt.show()

  # Drop rows that are outliers
  df = df[df['outlier'] != -1]
  return df


# SKEW CORRECT
def skew_correct(df, feature, max_power=100, messages=True):
  import pandas as pd, numpy as np
  import seaborn as sns, matplotlib.pyplot as plt

  # In case the dataset is too big, we can resample it down to a reasonable size to find the best transformation
  df_temp = df.copy()
  if df_temp.shape[0] > 10000:
    df_temp = df.sample(frac=round(10000 / df.shape[0], 2))

  i = 1 # Starting
  skew = df[feature].skew()
  if messages: print(f'Starting skew: {round(skew, 3)}')
  while round(skew, 2) != 0 and i <= max_power:
    if skew > 0:
      skew = np.power(df_temp[feature], 1/i).skew()
    else:
      skew = np.power(df_temp[feature], i).skew()
    i += 0.01
  if messages: print(f'Final skew: {round(skew, 3)}')

  if skew > -1 and skew < 1:  # If we were able to correct it:
    if skew > 0:
      corrected = np.power(df[feature], 1/round(i, 3))
      name = f'{feature}^1/{round(i, 3)}'
    else:
      corrected = np.power(df[feature], round(i, 3))
      name = f'{feature}^{round(i, 3)}'
    df[name] = corrected  # Add the new corrected column to the DataFrame
  else:  # If we weren't able to correct it, then turn it into 0/1
    name = f'{feature}_binary'
    df[name] = df[feature]
    df.loc[df[name] == df[name].value_counts().index[0], name] = 0
    df.loc[df[name] != df[name].value_counts().index[0], name] = 1
    if messages:
      print(f'The feature {feature} could not be transformed into a normal distribution.')
      print(f'Instead, it has been converted to a binary (0/1) where {df[feature].value_counts().index[0]} = 0 and all other values = 1')

  if messages:
    f, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    sns.despine(left=True)
    sns.histplot(df_temp[feature], color="b", ax=axes[0], kde=True)
    if skew > -1 and skew < 1:
      if skew > 0:
        corrected = np.power(df_temp[feature], 1/round(i, 3))
      else:
        corrected = np.power(df_temp[feature], round(i, 3))
      df_temp[name] = corrected
      sns.histplot(df_temp[name], color="g", ax=axes[1], kde=True)
    else:
      df_temp[name] = df[feature]
      df_temp.loc[df_temp[name] == df_temp[name].value_counts().index[0], name] = 0
      df_temp.loc[df_temp[name] != df_temp[name].value_counts().index[0], name] = 1
      sns.countplot(data=df_temp, x=name, color="g", ax=axes[1])
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    plt.show()

  return df


# MISSING DROP
def missing_drop(df, label="", features=[], messages=True, row_threshold=2):
  import pandas as pd
  
  start_count = df.count().sum()

  # Drop the obvious columns and rows
  df.dropna(axis=0, how='all', inplace=True)            # Drop rows where every value is missing
  df.dropna(axis=0, thresh=row_threshold, inplace=True) # Drop rows that are missing more than the threshold allowed
  df.dropna(axis=1, how='all', inplace=True)            # Drop columns where every value is missing
  if label != "": df.dropna(axis=0, how='all', subset=[label], inplace=True)  # Drop rows where the label is missing

# Get the number of missing values for each feature and calculate the remaining non-nan cells if the column vs rows were dropped
  def generate_missing_table():
    df_results = pd.DataFrame(columns=['Missing', 'column', 'rows'])
    for feat in df: 
      missing = df[feat].isna().sum()
      if missing > 0: # Only do the work if there is missing data
        # Compare the non-nan cells remaining after dropping either the column or entire rows that are missing that column
        memory_cols = df.drop(columns=[feat]).count().sum()
        memory_rows = df.dropna(subset=[feat]).count().sum()
        df_results.loc[feat] = [missing, memory_cols, memory_rows]
    return df_results

  df_results = generate_missing_table()
  while df_results.shape[0] > 0: # If any missing data were found
    max = df_results[['column', 'rows']].max(axis=1)[0]
    max_axis = df_results.columns[df_results.isin([max]).any()][0]
    df_results.sort_values(by=[max_axis], ascending=False, inplace=True)
    if messages: print('\n', df_results)
    if max_axis == 'rows':
      df.dropna(axis=0, subset=[df_results.index[0]], inplace=True)
    else:
      df.drop(columns=[df_results.index[0]], inplace=True)
    df_results = generate_missing_table()

    if messages: print(f'{round(df.count().sum()/start_count * 100, 2)}% ({df.count().sum()} / {start_count}) of non-null cells were kept.')
  return df