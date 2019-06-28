import pandas as pd
from fido import factors, query_var, ordered_hidden_var_list, p_col_name
pd.set_option('mode.chained_assignment', None)

# ----------------------
# Input format:#
# factors_list = python list [ ... ]
#   Each entry is a factor.
#   Either a Pandas dataframe or numpy ndarray like shown below.
#   last column contains probabilites (float)
#   Rest are variables in the factor. (int) 1: true, 0: false
#
# query_var = (str) variable name
# ordered_hidden_var_list = python list [ ... ]. Of variable names (str
# evidence_list = python dict. key: variable name (str), value: (int) evidence 1 or 0
#
# Factor looks like this. Refer to holmes.py to see how it is defined
# +---+---+--------+
# | R | E |  Prob  |
# +---+---+--------+
# | 0 | 0 | 0.9998 |
# | 0 | 1 |    0.1 |
# | 1 | 0 | 0.0002 |
# | 1 | 1 |    0.9 |
# +---+---+--------+
#
# Output can numpy array or Pandas frame. Two colomns:
# First column contains query variable values (int)
# Last column has the values after normalize() (float)
#     B      Prob
# 0  1  0.001597
# 1  0  0.998403
#
# ----------------------

def restrict(factor, variable, value):
    if DEBUG:
        print("Restricted {}:\n".format(variable), factor)
    query = factor[variable]==value
    result = factor.loc[query]
    result = result.drop(variable, axis=1)
    if DEBUG:
        print("\nResult:\n", result, "\n--------------\n")
    return result

def sumout(factor,variable):
    if DEBUG:
        print("Sumout {} from:\n".format(variable), factor)
    # Remove coloumn corresponding to given variable
    columns = list(factor)
    columns.remove(variable)
    subset = factor[columns]
    deleted_indices = []
    for index, row in subset.iterrows():
        row = row.drop(p_col_name)
        var_values = subset.loc[:, subset.columns != p_col_name]
        matches = ((var_values == row) | (var_values.isnull() & row.isnull())).all(1)

        if index >= len(matches):
            break
        elif index in deleted_indices:
            continue

        # sum matching rows
        subset.at[index, p_col_name] = subset[matches][p_col_name].sum(axis=0)

        # remove summed up rows except first one
        res = next((i for i, j in enumerate(matches) if j), None)
        matches.iloc[res] = False
        to_delete_indices = matches.index[matches].tolist()
        deleted_indices += to_delete_indices
        subset.drop(to_delete_indices, inplace=True)

    if DEBUG:
        print("\nResult:\n", subset, "\n--------------\n")
    return subset

def multiply(factor1, factor2):
    if DEBUG:
        print("Multipying:\n", factor1, "\n", factor2)
    f1_columns =  list(factor1)
    f1_columns.remove(p_col_name)
    f2_columns = list(factor2)
    f2_columns.remove(p_col_name)
    common_vars = list(set(f1_columns).intersection(f2_columns))

    merged_factors = pd.merge(factor1, factor2, how='left', left_on=common_vars, right_on=common_vars)
    f1_p_col_name = p_col_name + '_x'
    f2_p_col_name = p_col_name + '_y'
    merged_factors[p_col_name] = merged_factors[f1_p_col_name] * merged_factors[f2_p_col_name]
    merged_factors.drop(f1_p_col_name, axis=1, inplace=True)
    merged_factors.drop(f2_p_col_name, axis=1, inplace=True)

    if DEBUG:
        print("\nResult:\n", merged_factors, "\n--------------\n")
    return merged_factors

def normalize(factor):
    if DEBUG:
        print("Normalize:\n", factor)

    sum = factor[p_col_name].sum(axis=0)

    factor[p_col_name] = factor[p_col_name].apply(lambda x: x / float(sum))

    if DEBUG:
        print("\nResult:\n", factor, "\n--------------\n")
    return factor


def get_var_to_factor_mapping(factor_list):
    vars_dict = {}
    for index, factor in enumerate(factor_list):
        if factor is not None:
            var_list = list(factor)
            var_list.remove(p_col_name)
            for var in var_list:
                if var in vars_dict:
                    vars_dict[var].append(index)
                else:
                    vars_dict[var] = [index]

    if DEBUG:
        print ("New mapping:")
        print(vars_dict)
    return vars_dict

def inference(factor_list_full, query_var, ordered_hidden_var_list_full, evidence_list):
    print ("Inferring with evidence: {}".format(evidence_list))
    factor_list = factor_list_full.copy()
    ordered_hidden_var_list =  ordered_hidden_var_list_full.copy()
    var_to_factor_mapping = get_var_to_factor_mapping(factor_list)

    # Restrict factors according to evidence
    for var, evidence in evidence_list.items():
        for factor_index in var_to_factor_mapping[var]:
            result = restrict(factor_list[factor_index],var,evidence)
            if len(result.columns) <= 1:
                if DEBUG:
                    print("Removed above factor from factor list: {}".format(var))
                factor_list[factor_index] = None
            else:
                factor_list[factor_index] = result
        var_to_factor_mapping = get_var_to_factor_mapping(factor_list)
        ordered_hidden_var_list.remove(var)

    ordered_hidden_var_list.remove(query_var)

    # handle hidden vars
    for hidden_var in ordered_hidden_var_list:
        var_factors_indices = var_to_factor_mapping[hidden_var]

        if len(var_factors_indices) > 0:
            # multiply if there are more than 1 factors that contain the hidden var
            if len(var_factors_indices) > 1:
                multiplied_factors = factor_list[var_factors_indices[0]]
                factor_list[var_factors_indices[0]] = None
                for factor_index in var_factors_indices[1:]:
                    multiplied_factors = multiply(multiplied_factors, factor_list[factor_index])
                    factor_list[factor_index] = None
            else:
                multiplied_factors = factor_list[var_factors_indices[0]]
                factor_list[var_factors_indices[0]] = None

            # sumout each hidden variable
            factor_list.append(sumout(multiplied_factors, hidden_var))
            var_to_factor_mapping = get_var_to_factor_mapping(factor_list)

    # multiply remaining factors
    var_factors_indices = var_to_factor_mapping[query_var]
    if len(var_factors_indices) == 0:
        raise KeyError("Query variable factors not found")
    if len(var_factors_indices) > 1:
        query_factor = factor_list[var_factors_indices[0]]
        factor_list[var_factors_indices[0]] = None
        for factor_index in var_factors_indices[1:]:
            query_factor = multiply(query_factor, factor_list[factor_index])
            factor_list[factor_index] = None
    else:
        query_factor = factor_list[var_factors_indices[0]]
        factor_list[var_factors_indices[0]] = None

    query_result = normalize(query_factor)
    print (query_result)
    print("\n")
    return query_result

DEBUG = False

evidence_list = {'M': 1, 'FH':1}
inference(factors, query_var, ordered_hidden_var_list, evidence_list)

evidence_list = {'M': 1, 'FH':1, 'B':1}
inference(factors, query_var, ordered_hidden_var_list, evidence_list)

evidence_list = {'M': 1, 'FH':1, 'B':1, 'NA':1}
inference(factors, query_var, ordered_hidden_var_list, evidence_list)








