import statsmodels.api as sm
import scipy

def ChowTest(data, y_variable_name, subgroup_variable_name):
    '''data: a dataframe containing all the data including the y variable
       y_variable_name: the label of the y variable
       subgroup_variable_name: the dummy variable representing the subgroups in question'''
    X = data.drop(y_variable_name,axis=1)
    X = sm.add_constant(X)
    y = data[y_variable_name]
    X_1 = X[data[subgroup_variable_name]==1].drop(subgroup_variable_name,axis=1)
    y_1 = data[data[subgroup_variable_name]==1][y_variable_name]
    X_2 = X[data[subgroup_variable_name]==0].drop(subgroup_variable_name,axis=1)
    y_2 = data[data[subgroup_variable_name]==0][y_variable_name]
    J=X.shape[1]
    k=X_1.shape[1]
    N1 = X_1.shape[0]
    N2 = X_2.shape[0]
    model_dummy = sm.OLS(y,X).fit()
    RSSd = model_dummy.ssr
    model_1 = sm.OLS(y_1,X_1).fit()
    RSS1 = model_1.ssr
    model_2 = sm.OLS(y_2,X_2).fit()
    RSS2 = model_2.ssr
    chow = ((RSSd-(RSS1+RSS2))/J)/((RSS1+RSS2)/(N1+N2-2*k))
    p = scipy.stats.f.cdf(chow, J, N1+N2+2*k)
    print('p-value = ', p)
    return p
