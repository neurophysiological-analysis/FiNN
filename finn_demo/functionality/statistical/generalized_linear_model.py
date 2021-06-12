'''
Created on Jun 2, 2020

@author: voodoocode
'''

import numpy as np

import finn.statistical.glmm as glmm

data_size = 100000
random_factor_count = 20
nested_random_factor_count = 2

data_01 = np.random.normal(0, 3, int(data_size/2)); data_02 = np.random.normal(1, 2, int(data_size/2)); data_0 = np.concatenate((data_01, data_02)); data_0 = np.expand_dims(data_0, axis = 0)

data_11 = np.random.binomial(1, 0.1, int(data_size/2)); data_12 = np.random.binomial(1, 0.9, int(data_size/2)); data_1 = np.concatenate((data_11, data_12)); data_1 = np.expand_dims(data_1, axis = 0)
data_21 = np.random.binomial(1, 0.5, int(data_size/2))*2-2; data_22 = np.random.binomial(1, 0.5, int(data_size/2))*2-2; data_2 = np.concatenate((data_21, data_22)); data_2 = np.expand_dims(data_2, axis = 0)
data_31 = np.random.normal(0, 1, int(data_size/2)); data_32 = np.random.normal(0.25, 1, int(data_size/2)); data_3 = np.concatenate((data_31, data_32)); data_3 = np.expand_dims(data_3, axis = 0)
data_4 = np.repeat(np.arange(0, random_factor_count), data_size/random_factor_count); np.random.shuffle(data_4); data_4 = np.expand_dims(data_4, axis = 0)
data_5 = np.repeat(np.arange(0, nested_random_factor_count), data_size/nested_random_factor_count); np.random.shuffle(data_5); data_5 = np.expand_dims(data_5, axis = 0)

data = np.concatenate((data_0, data_1, data_2, data_3, data_4, data_5), axis = 0).transpose()
data_label = ["measured_variable", "categorical_factor_A", "categorical_factor_B", "continous_factor_A", "random_effect_A", "nested_random_effect_A"]
glm_formula = "measured_variable ~ categorical_factor_A + categorical_factor_B + continous_factor_A + categorical_factor_A:continous_factor_A + (1|random_effect_A) + (1|random_effect_A:nested_random_effect_A)"
glm_factor_types = ["continuous", "categorical",  "categorical",   "continuous",  "categorical",  "categorical"]
glm_contrasts = "list(categorical_factor_A = contr.sum, categorical_factor_B = contr.sum, continous_factor_A = contr.sum, random_effect_A = contr.sum, nested_random_effect_A = contr.sum)"
glm_model_type = "gaussian"

stat_results = glmm.run(data = data, label_name= data_label, factor_type = glm_factor_types, formula = glm_formula, contrasts = glm_contrasts, data_type = glm_model_type)

print("Demo may return a singular fit since the naive applied data generation of this example\n does not guarantee sufficient observations for any random factor/nested random factor.\n")

for (factor_idx, factor_name) in enumerate(stat_results[5]):
    print("factor: %s | p-value: %2.2f | effect size: %2.2f | std error %2.2f" % (factor_name, stat_results[2][factor_idx], 
                                                                                  stat_results[3][factor_idx], stat_results[4][factor_idx]))
    
    
    
    
    
    
    
    
    
    
    
    