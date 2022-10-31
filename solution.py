import numpy as np
import pandas as pd
import sklearn.preprocessing as pre

SAMPLE_INPUT_1 = [[1,2,0],[0,1,1],[5,6,5]]
SAMPLE_SOLUTION_1 = [[-1.224744871391589, -0.4629100498862757, 0],
                     [0, -0.9258200997725514, -1.22474487],
                     [1.224744871391589, 1.38873015, 1.22474487]
                     ]


def data_imputation(input_matrix):
    # aim - impute zero values with mean of columns

    input_dataframe = pd.DataFrame(input_matrix, columns=['A','B','C'])
    output_dataframe = input_dataframe.copy()
    
    for col in input_dataframe.columns:
        arr = input_dataframe[col].values
        arr_nonzero_mean = np.mean(arr[arr!=0])
        output_dataframe[col] = output_dataframe[col].replace(0,arr_nonzero_mean)

    return output_dataframe

def data_transformation(input_dataframe):
    input_array = input_dataframe.values

    # aim - standardize the values
    scaler = pre.StandardScaler()
    transformed_array = scaler.fit_transform(input_array)
    return np.round(transformed_array, 5)

def assembled_transfomation(input_matrix):
    imputed_dataframe = data_imputation(input_matrix)
    transformed_array = data_transformation(imputed_dataframe)
    return transformed_array
    
def validation_with_sample(input_matrix, actual_output_matrix):
    output_array = assembled_transfomation(input_matrix)

    actual_output_array = np.array(actual_output_matrix)
    actual_output_array = np.round(actual_output_array,5)

    assert((output_array - actual_output_array).all()==0)

input_arr = [[1,3,2],[4,0,1],[1,4,0]]

class __main__():
    validation_with_sample(SAMPLE_INPUT_1, SAMPLE_SOLUTION_1)
    output_arr = assembled_transfomation(input_arr)
    print(output_arr)