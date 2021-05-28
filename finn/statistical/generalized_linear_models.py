'''
Created on Jun 12, 2018

@author: voodoocode
'''

import numpy as np
import warnings
import gc

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import rpy2.rinterface_lib.embedded


def check_formula(formula):
    """
    Extracts fixed and random factors for the purpose of validating these prior to analysis
    
    @param formula: The formula to be applied
    
    @return: fixed factors, random factors - as lists
    """
    
    return get_factors(formula)

def run(data, label_name, factor_type, formula, contrasts, data_type = "gaussian"):
    """
    Applies generalized linear mixed models (GLM). As the formula needs to be parsed in order to align significances
    with effect sizes, it is highly recommended to check prior to execution once whether the formula is understood correctly
    via running check_formula(formula). The formula will be parsed (scrubbed and unpacked), and fixed and random factors extracted. 
    
    @param data: Input data as a single matrix (Axis 0 => samples; Axis 1 data per sample). (Samples x Values).
    @param label_name: Names of the individual columns; have to align with the names specified in the formula.
    @param factor_type: Categorical vs. continuous factors.
    Be aware:
     - Categorical variables: These get ordered and the lowest value becomes the reference. The returned effect size is the
    average effect size of all levels vs. the reference effect size, i.e. the levels are '0' and '1'. In this case does '0' determine
    the point of reference whereas '1' determines the target value. Any effect size will correspond to the effect of changing 
    said factor from '0' to '1', not the other way around.
     - Continuous variables: Effect sizes for this kind of variable are scaled towards 1 functionality of this factor.
     Therefore the effect size corresponds to the effect of changing said factor from '0' to '1'.
    @param formula: Formula to be applied by the GLM. The formula must not contain a digit within the factor names.
    @param contrasts: Contrast values for the GLM. Contrast names have to align with the names specified in the formula.
    @param data_type: Model type for the glm, default is gaussian.
    
    @return: (scores, df, p-values, coefficients, std_error, factor names)
    
    How to write lmer models:
    
    
    Variables are continuous by default. Factor(var) turns them into categorical variables.
    var0 ~ var1 + var2 | Evaluate effect of var1, var2 und var0.
    var0 ~ var1 * var2 == var0 ~ var1 + var2 + var1:var2 | Evaluate effect of var1, var2 and the interaction of var1 and var2.
    var0 ~ var1 + (1|var2) | Model var1 as a random effect.
    """
    
    if (digit_in_formula(formula)):
        raise AssertionError("Formula is malformed and may not include digits as characters within the factor names")
    
    if (type(data) != np.ndarray):
        data = np.asarray(data)
    
    pre_sanity = pre_sanity_checks(data, formula)
    if (pre_sanity != True):
        return pre_sanity
    
    rpy2.robjects.numpy2ri.activate()
    
    
    if ("|" in formula):
        random_effect = True
    else:
        random_effect = False
    
    #Check if R dependencies are installed
    check_r_dependencies()

    #Copy glm data to R
    process_glm_data(data, label_name, factor_type)
    
    execution_successful = execute_glm(data_type, formula, random_effect)
    if (execution_successful[0] == False):
        return execution_successful[1]
    try:
        ro.r("res = Anova(lm0, type = 3, contrasts=" + contrasts + ")")
    except rpy2.rinterface_lib.embedded.RRuntimeError:
        
        warnings.warn("Cannot evalute data, forcing normalization ([0, 1]) to solve failure due to numerical inaccuracy", RuntimeWarning)
        data[:, 0] -= np.nanmin(data[:, 0])
        data[:, 0] /= np.nanmax(data[:, 0])
        
        process_glm_data(data, label_name, factor_type)
        
        execution_successful = execute_glm(data_type, formula, random_effect)
        if (execution_successful[0] == False):
            return execution_successful[1]
        
        ro.r("res = Anova(lm0, type = 3, contrasts=" + contrasts + ")")
        
    (scores, p_values, df, anova_factors, coefficients, coeff_names, coeff_std_error) = collect_results()
    
    try:
        (final_coeff_names, final_coefficients, final_std_error) = sync_meta_info(coeff_names, coefficients, coeff_std_error, formula)
        (final_anova_factors, final_scores, final_df, final_p_values) = sync_anova_info(anova_factors, scores, df, p_values, formula)
    except:
        return overwrite_negative_result(formula, "A coefficient was dropped. Cannot evaluate formula due to insufficient data.")
         
    result_sanity_check(final_coeff_names, final_coefficients, final_std_error, final_df, final_anova_factors, final_scores, final_p_values)
    
    ro.r('rm(list=ls(all=TRUE))')
    rpy2.robjects.numpy2ri.deactivate()
    gc.collect()
    
    return (final_scores, final_df, final_p_values, final_coefficients, final_std_error, final_anova_factors)

def digit_in_formula(formula):
    """
    """
    
    (fixed_factors, random_factors) = get_factors(formula)
    
    for fixed_factor in fixed_factors:
        if (any(char.isdigit() for char in fixed_factor)):
            return True
    for random_factor in random_factors:
        random_factor_parts = random_factor.split("|")
        
        for (char_idx, char) in enumerate(random_factor_parts[0]):
            if ( #In case there is a char left to the digit, check whether it is a meta char
                (((char_idx - 1) >= 0) and (formula[char_idx - 1] not in ["(", ")", "+", ":", "*", "/"]))
                or #In case there is a char right to the digit, check whether it is a meta char
                (((char_idx + 1) < len(random_factor_parts[0])) and (formula[char_idx + 1] not in ["(", ")", "+", ":", "*", "/"]))
                ): #If either is not the case, the char is part of a factor name
                return True
        if (any(char.isdigit() for char in random_factor_parts[1])):
            return True

    return False

def pre_sanity_checks(data, formula):
    """
    Checks whether the input data is all zeros or all equal.
    
    @param data: Input data samples x values
    @param formula: Used to generate a negative result in case of an unexploreable data set.
    
    @return: Returns all negative results in case of a failed check, True otherwise
    """
    
    if (len(data) == 0):
        return overwrite_negative_result(formula, "No samples within matrix")
    
    # In case all values are equal, the R-based GLM will crash.
    # Therefore, this is caught here, returning P = 1 and matching values.
    if ((data[:, 0] == data[0, 0]).all()):
        return overwrite_negative_result(formula, "All dependent values are equal in the input data for the GLM")
    
    return True

def overwrite_negative_result(formula, error_msg):
    """
    Returns negative results in case of an error
    
    @param formula: The formula applied is used to generate the correct amount of negative results (deducing the number and type of factors)
    @param error_msg: The error message to be communicated
    
    @return: A all negative result for any evaluated fixed factor
    """
    
    fixed_factors = get_factors(formula)[0] + ["(Intercept)"]
    factor_cnt = len(fixed_factors)
    negative_result = (np.zeros((factor_cnt)), np.zeros((factor_cnt)), np.ones((factor_cnt)), np.zeros((factor_cnt)), np.zeros((factor_cnt)), fixed_factors)
    warnings.warn(error_msg, UserWarning)
    return negative_result

def execute_glm(data_type, formula, random_effect):
    """
    Estimates the glm defined in formula using the data already copied to R (variable name 'data').
    
    @param data_type: The distribution of the observed variable. Either gaussian, binomial or poisson.
    @param formula: The formula to be applied to the glm.
    @param random_effect: Flag whether or not the formula includes random effects
    
    @return True in case of success, False and a negative result otherwise
    """
    
    try:
        if (random_effect == True and (data_type is None or data_type == "gaussian")):
            ro.r("lm0 = lmer(" + formula + ", data)")
        elif(random_effect == True and data_type == "poisson"):
            ro.r('lm0 = glmer(' + formula + ', data, family=poisson(link="log"), nAGQ=0, control=glmerControl(calc.derivs = FALSE, optimizer = "nloptwrap"))')
        elif(random_effect == True and data_type == "binomial"):
            ro.r('lm0 = glmer(' + formula + ', data, family=binomial(link="logit"), nAGQ=0, control=glmerControl(calc.derivs = FALSE, optimizer = "nloptwrap"))')
        elif(random_effect == False and (data_type is None or data_type == "gaussian")):
            ro.r("lm0 = lm(" + formula + ", data)")
        else:
            raise AssertionError("Distribution not implemented")
        
        return (True, )
    except rpy2.rinterface_lib.embedded.RRuntimeError as e:
        if (len(e.args) > 0):
            if (e.args[0] == 'Error: grouping factors must have > 1 sampled level\n'):
                return (False, overwrite_negative_result(formula, "Grouping factors must have > 1 sampled level"))
            elif (e.args[0] == 'Error: number of levels of each grouping factor must be < number of observations\n'):
                return (False, overwrite_negative_result(formula, "Number of levels of each grouping factor must be < number of observations"))
            else:
                return (False, overwrite_negative_result(formula, "Unknown error in input data matrix"))

def process_glm_data(data, label_name, factor_type):
    glm_data = anova_container(data, label_name, factor_type)
    
    for idx in range(0, len(glm_data.label_name)):
        tmp = ro.r.matrix(glm_data.data[:, idx], nrow = len(glm_data.data[:, idx]), ncol = 1)
        ro.r.assign(glm_data.label_name[idx], tmp)
        
    glm_data.send_labels()

def check_r_dependencies():
    """
    Checks whether all R dependncies for the glm are installied, raises an exception otherwise
    """
    
    r_dependencies = ["Matrix", "lme4", "carData", "car"]
    for r_dependency in r_dependencies:
        if (list(ro.r("nzchar(system.file(package = '" + r_dependency + "'))"))[0] == False):
            raise Exception("Error: R dependency " + r_dependency + " no installed.")
        ro.r("library(" + r_dependency + ")")

def collect_results():
    """
    Collects the results from R and transfers them back to python. As the anova call and the glm call
    may return the factors with slightly different names and in different order, the results are
    synchronized for a coherent return.
    
    @return: scores, p-values, degrees of freedom, anova factor names, coefficients, coefficient names and standard errors of the coefficients
    """
    scores        = list()
    p_values      = list()
    df            = list()
    anova_factors = list()
    if (ro.r('length(grep("Sum Sq", names(res), value = TRUE))')[0] > 0):
        for anova_factor in list(ro.r("row.names(res)")):
            if (anova_factor == "Residuals"):
                continue
            
            scores.append(ro.r('res["' + anova_factor + '","F value"]')[0])
            p_values.append(ro.r('res["' + anova_factor + '","Pr(>F)"]')[0])
            df.append(ro.r('res["' + anova_factor + '","Df"]')[0])
            anova_factors.append(anova_factor)
    elif(ro.r('length(grep("Chisq", names(res), value = TRUE))')[0] > 0):
        for anova_factor in list(ro.r("row.names(res)")):
            scores.append(ro.r('res["' + anova_factor + '","Chisq"]')[0])
            p_values.append(ro.r('res["' + anova_factor + '","Pr(>Chisq)"]')[0])
            df.append(ro.r('res["' + anova_factor + '","Df"]')[0])
            anova_factors.append(anova_factor)
    else:
        raise TypeError("Cannot extract score and p_values")
    
    coefficients = list()
    coeff_names = list()
    coeff_std_error = list()
    for coeffName in list(ro.r('rownames(coefficients(summary(lm0)))')):
        coeff_names.append(coeffName)
        coefficients.append(ro.r('coefficients(summary(lm0))["' + coeffName + '","Estimate"]')[0])
        coeff_std_error.append(ro.r('coefficients(summary(lm0))["' + coeffName + '","Std. Error"]')[0])
    
    return (scores, p_values, df, anova_factors, coefficients, coeff_names, coeff_std_error)

def result_sanity_check(final_coeff_names, final_coefficients, final_std_error,
                        final_df, final_anova_factors, final_scores, final_p_values):
    """
    Checks whether the results from the anova and the glm call returned the same number of arguments.
    This should *theoretically* always be the case.
    Raises an error in case the sanity check fails.
    
    @param final_coeff_names: Names of the coefficients
    @param final_coefficients: Coefficients from the glm call
    @param final_std_error: Standard errors from the glm call
    @param final_df: Degrees of freedom from the anova call
    @param final_anova_factors: Name of factors from the anova call
    @param final_scores: Scores from the anova call
    @param final_p_values: P-values from the anova call
    
    """    
    
    if (len(final_df) != len(final_coeff_names)
        or len(final_df) != len(final_coefficients)
        or len(final_df) != len(final_std_error)
        or len(final_df) != len(final_anova_factors)
        or len(final_df) != len(final_scores)
        or len(final_df) != len(final_p_values)):
        raise AssertionError("Something went wrong with the ANOVA.")

def sync_meta_info(names, values, values_2, formula, skip_missing = False):
    """
    Synchronizes the order of names to the order defined in formula
    
    @param names: Names' order is synchronized to their appearance in the formula.
    @param values: Values which are synchronized in regards to the synchronization implied by names and the formula.
    @param values_2: Additional values which are synchronized in regards to the synchronization implied by names and the formula.
    @param formula: A reference for the order of the factors named in names.
    @param skip_missing: If flagged, missing values can be skipped.
    
    @return: names, values, values_2 as consistently sorted variants
    """
    
    #formula_names = get_fixed_factors(formula) + ["(Intercept)"]
    formula_names = get_factors(formula)[0] + ["(Intercept)"]
    corr_names = list()
    for name in names:
        for refVal in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ","]:
            name = name.replace(refVal, "")
        corr_names.append(name)
    
    final_names = [None for _ in range(len(formula_names))]
    final_vals = [None for _ in range(len(formula_names))]
    final_vals_2 = [None for _ in range(len(formula_names))]
    for (tgtIdx, formula_name) in enumerate(formula_names):
        
        src_idx_list = list()
        for (corr_idx, corrName) in enumerate(corr_names):
            if (set(formula_name.split(":")) == set(corrName.split(":"))):
                src_idx_list.append(corr_idx)
        src_idx_list = np.asarray(src_idx_list)
        
        if (len(src_idx_list) > 1):
            src_idx_list = np.sort(src_idx_list)
            
            final_names[tgtIdx] = corr_names[src_idx_list[0]]
            final_vals[tgtIdx] = np.mean(np.asarray(values)[src_idx_list])
            final_vals_2[tgtIdx] = np.mean(np.asarray(values_2)[src_idx_list])
        elif(len(src_idx_list) == 1):
            final_names[tgtIdx] = corr_names[src_idx_list[0]]
            final_vals[tgtIdx] = values[src_idx_list[0]]
            final_vals_2[tgtIdx] = values_2[src_idx_list[0]]
        elif(len(src_idx_list) == 0):
            if (skip_missing == True):
                final_vals[tgtIdx] = None
                final_vals_2[tgtIdx] = None
                final_names[tgtIdx] = None
            else:
                raise AssertionError("Cannot synchronize component")
    
    return (final_names, final_vals, final_vals_2)
    
def sync_anova_info(anova_factors, scores, df, p_values, formula):
    """
    Synchronizes the order of factors to the order defined in the formula
    @param anova_factors: Names of the factors.
    @param scores: Scores of the factors.
    @param df: Degrees of freedom of the factors.
    @param p_values: P-values of the factors.
    @param formula: Formula which is used as a reference order for the factors.
    
    @return: Names of the anova factors, scores, degrees of freedom, p-values, all ordered according
    to the order reference in the formula.
    """
    
    #formula_names = get_fixed_factors(formula) + ["(Intercept)"]
    formula_names = get_factors(formula)[0] + ["(Intercept)"]
    
    corr_anova_factors = list()
    for name in anova_factors:
        for refVal in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            name = name.replace(refVal, "")
        corr_anova_factors.append(name)
    
    final_anova_factors = [None for _ in range(len(formula_names))]
    final_score = [None for _ in range(len(formula_names))]
    final_df = [None for _ in range(len(formula_names))]
    final_p_palue = [None for _ in range(len(formula_names))]
    for (tgtIdx, formula_name) in enumerate(formula_names):
        
        src_idx_list = list()
        for (corr_anova_idx, corranova_factor) in enumerate(corr_anova_factors):
            if (set(formula_name.split(":")) == set(corranova_factor.split(":"))):
                src_idx_list.append(corr_anova_idx)
        src_idx_list = np.asarray(src_idx_list)
        
        if (len(src_idx_list) == 1):
            final_anova_factors[tgtIdx] = corr_anova_factors[src_idx_list[0]]
            final_score[tgtIdx] = scores[src_idx_list[0]]
            final_df[tgtIdx] = df[src_idx_list[0]]
            final_p_palue[tgtIdx] = p_values[src_idx_list[0]]
        else:
            src_idx_list = np.sort(src_idx_list)
            
            final_anova_factors[tgtIdx] = corr_anova_factors[src_idx_list[0]]
            final_score[tgtIdx] = np.mean(np.asarray(scores)[src_idx_list])
            final_df[tgtIdx] = np.mean(np.asarray(df)[src_idx_list])
            final_p_palue[tgtIdx] = np.mean(np.asarray(p_values)[src_idx_list])
    
    return (final_anova_factors, final_score, final_df, final_p_palue)

#----------------------------------------------- def get_fixed_factors(formula):
    #----------------------------------------------------------------------- """
    #------------------------------- Extracts the fixed factors from the formula
#------------------------------------------------------------------------------ 
    #--------------------------------------- @param formula: The formula string.
#------------------------------------------------------------------------------ 
    #------------------------------------------ @return: Extracted fixed factors
    #----------------------------------------------------------------------- """
    #------------------------------------------ formula = clean_formula(formula)
    #-------------------------------------- formula = rm_random_effects(formula)
    #----------------------- formula = formula.replace("(", "").replace(")", "")
#------------------------------------------------------------------------------ 
    #---------------------------------------------- factors = formula.split("+")
    #------------------------------------------------------------ factors.sort()
#------------------------------------------------------------------------------ 
    #------------------------------------------------------------ return factors

def get_factors(formula):
    formula = clean_formula(formula)
    (fixed_effects, random_effects) = split_effects(formula)
     
    return (fixed_effects, random_effects)

def split_effects(formula):
    """
    Splits the effects specified in the formula into fixed effects and random effects
    
    @param formula: The formula to be parsed.
    
    @return: (fixed effects, random effects) - as two lists
    """
 
    fixed_effects = list()
    random_effects = list()
     
    while(True):
        idx = formula.find("|")
        if (idx == -1):
            for (reIdx, random_effect) in enumerate(random_effects):
                random_effects[reIdx] = random_effect.replace("(", "").replace(")", "")
            fixed_effects = (formula.replace("(", "").replace(")", "")).split("+")
             
            return (fixed_effects, random_effects)
         
        l_idx = idx;
        r_idx = idx if (formula[idx + 1] != "|") else idx + 1
         
        left_idx = find_left_idx(formula, l_idx, operators = [])
        right_idx = find_right_idx(formula, r_idx, operators = [])
         
        pre_form = formula[:(left_idx-1)]
        post_form = formula[(right_idx+1):]
         
        rand_factor = formula[(left_idx-1):(right_idx+1)]
        random_effects.append(rand_factor)
         
        if (len(post_form) > 0 and len(pre_form) > 0):
            formula = pre_form + post_form[1:] if (post_form[0] == "+") else pre_form + post_form    #Avoid double '+'
        else:
            if(len(post_form) > 0):
                formula = post_form[1:] if (post_form[0] == "+") else post_form
            elif(len(pre_form) > 0):
                formula = pre_form[:-1] if (pre_form[-1] == "+") else pre_form
            else:
                return (fixed_effects, random_effects)

def clean_formula(formula):
    """
    Preprocesses the formula prior to parsing it by removing space and simplifying operators.
    
    @param formula: The formula to be preprocessed for parsing.
    
    @return: The refined formula
    """
    formula = rm_empty_space(formula)
    formula = replace_operator(formula)
    formula = restore_interactions(formula)
    
    return formula

def rm_empty_space(formula):
    """
    Removes empty parts in the formula
    
    @param formula: The formula to be parsed.
    
    @return The cleaned formula.
    """
    
    #rm spaces and tabs
    formula = formula.replace(" ", "")
    formula = formula.replace("\t", "")
    #rm dependent variable
    formula = formula.split("~")[1]
    
    return formula

def replace_operator(formula):
    """
    Replaces convoluted operators in formulas with their elongated versions for parsing. The operators
    '*' (crossed) and '/' (nested) get replaced with A*B -> A+B+A:B and A/B -> A+A:B respectively.
    
    @param formula: The formula to be parsed
    
    @return: formula without the operators '*' (crossed) and '/' (nested).
    """
    
    while(True):

        idx = find_idx(formula)
        if (idx == -1):
            return formula
        
        left_idx = find_left_idx(formula, idx)
        right_idx = find_right_idx(formula, idx)
        
        pre_form = formula[:left_idx]
        post_form = formula[(right_idx):]
        
        left_arg = formula[left_idx:idx]
        right_arg = formula[(idx + 1):right_idx]

        if (formula[idx] == "*"):
            formula = pre_form + "(" + left_arg + "+" + right_arg + "+" + left_arg + ":" + right_arg + ")" + post_form
        else: #(formula[idx] == "/")
            formula = pre_form + left_arg + "+" + left_arg + ":" + right_arg + post_form
    
    return formula

def find_idx(formula):
    """
    Finds the next occurance of the '*' (cross) or '/' (nested) operator in
    a given formula.
    
    @param formula: The formula to be parsed.
    
    @return The index of the '*' (cross) or '/' (nested) operator.
    """
    
    idx = formula.find("*")
    if (idx >= 0):
        return idx
    else:
        return formula.find("/")

def find_left_idx(formula, idx, operators = ["+", ":", "|", "*", "/"]):
    """
    In case an operator is part of a bracketed equation, the left bracket is looked for. In case a
    closeing right side bracked is found, it is recognized and a left bracket is skipped for each additional
    right side bracket.
    
    @param formula: The formula to be parsed.
    @param idx: The index within the formula string on where to start the search for the left side bracket
    @param operators: List of operators to be distinguished from brackets (aka. factor name terminating objects).
    
    @return index of the corresponding closing left bracket.
    """
    
    brack_cnt = 0
    while(True):
        idx-=1
        
        if (formula[idx] == "("):
            brack_cnt -= 1
        if (formula[idx] == ")"):
            brack_cnt += 1
        if (formula[idx] in operators and brack_cnt == 0):
            return idx+1
        
        # In case more brackets have been closed than opened
        if (brack_cnt < 0):
            return idx+1
        
        if (idx == 0):
            return idx
        
def find_right_idx(formula, idx, operators = ["+", ":", "|", "*", "/"]):
    """
    In case an operator is part of a bracketed equation, the right bracket is looked for. In case a
    closeing left side bracked is found, it is recognized and a right bracket is skipped for each additional
    left side bracket.
    
    @param formula: The formula to be parsed.
    @param idx: The index within the formula string on where to start the search for the right side bracket
    @param operators: List of operators to be distinguished from brackets (aka. factor name terminating objects).
    
    @return index of the corresponding closing right bracket.
    """
    
    brack_cnt = 0
    while(True):
        idx+=1
        
        if (formula[idx] == ")"):
            brack_cnt -= 1
        if (formula[idx] == "("):
            brack_cnt += 1
        if (formula[idx] in operators and brack_cnt == 0):
            return idx
        
        # In case more brackets have been closed than opened
        if (brack_cnt < 0):
            return idx
        
        if (idx == (len(formula) - 1)):
            return idx+1

def restore_interactions(formula):
    """
    Replaces any occurance of (term1):(term2) with term1:term2
    
    @param formula: The formula to be parsed.
    
    @return: The cleaned formula.
    """
    
    while(True):
        idx = formula.find("):")
        if (idx == -1):
            idx = formula.find(":(")
        else:
            idx += 1
        
        if (idx == -1):
            return formula
        
        left_idx = find_left_idx(formula, idx)
        right_idx = find_right_idx(formula, idx)
        
        pre_form = formula[:left_idx]
        post_form = formula[right_idx:]
        left_arg = formula[left_idx:idx]
        right_arg = formula[(idx + 1):right_idx]
        
        if(len(right_arg) == 1):
#            middle_form = "(" + "+".join([factor + ":" + right_arg for factor in left_arg[1:-1].split("+")]) + ")"
            middle_form = "+".join([factor + ":" + right_arg for factor in left_arg[1:-1].split("+")])
        else:
            #middle_form = "(" + "+".join([factor + ":" + left_arg for factor in right_arg[1:-1].split("+")]) + ")"
            middle_form = "+".join([factor + ":" + left_arg for factor in right_arg[1:-1].split("+")])
        
        formula = pre_form + middle_form + post_form
            
    return formula

#----------------------------------------------- def rm_random_effects(formula):
    #------------------------------------------------------------- while (True):
        #----------------------------------------------- idx = formula.find("|")
        #------------------------------------------------------- if (idx == -1):
            #---------------------------------------------------- return formula
#------------------------------------------------------------------------------ 
        #---------------------------------------------------------- l_idx = idx;
        #----------------- r_idx = idx if (formula[idx + 1] != "|") else idx + 1
#------------------------------------------------------------------------------ 
        #-------------- left_idx = find_left_idx(formula, l_idx, operators = [])
        #------------ right_idx = find_right_idx(formula, r_idx, operators = [])
#------------------------------------------------------------------------------ 
        #------------------------------------- pre_form = formula[:(left_idx-1)]
        #----------------------------------- post_form = formula[(right_idx+1):]
#------------------------------------------------------------------------------ 
        #------------------------ if (len(post_form) > 0 and len(pre_form) > 0):
            # formula = pre_form + post_form[1:] if (post_form[0] == "+") else pre_form + post_form    #Avoid double '+'
        #----------------------------------------------------------------- else:
            #------------------------------------------- if(len(post_form) > 0):
                # formula = post_form[1:] if (post_form[0] == "+") else post_form
            #------------------------------------------- else:#len(pre_form) > 0
                # formula = pre_form[:-1] if (pre_form[-1] == "+") else pre_form
                
#-------------- def get_sub_formula(f_effects, r_effects, rm_f_effect, formula):
    # sub_formula = formula.split("~")[0].replace(" ","").replace("\\","") + "~"
#------------------------------------------------------------------------------ 
    #------------------------------------------------ for f_effect in f_effects:
        #----------------------------------------- if (rm_f_effect in f_effect):
            #---------------------------------------------------------- continue
        #----------------------------------------- sub_formula += f_effect + "+"
#------------------------------------------------------------------------------ 
    #------------------------------------------------ for r_effect in r_effects:
        #----------------------------- sub_formula += "(" + r_effect + ")" + "+"
    # sub_formula = sub_formula[:-1] if (sub_formula[-1] == "+") else sub_formula
#------------------------------------------------------------------------------ 
    #-------------------------------------------------------- return sub_formula

class anova_container():

    label_name   = None
    factor_type  = None
    data        = None
    
    label_comm   = None
    

    def __init__(self, raw_wata, label_name, factor_type):
        self.label_name  = label_name
        self.factor_type = factor_type
        
        self.gen_labels()
        self.data       = np.asarray(raw_wata)

    def gen_labels(self):
        self.label_comm = "data = data.frame(" + self.label_name[0]
        
        for nIdx in range(1, len(self.label_name)):
            if (self.factor_type[nIdx] == "categorical"):
                self.label_comm += ", " + self.label_name[nIdx] + " = factor(" + self.label_name[nIdx] + ")"
            elif (self.factor_type[nIdx] == "continuous"):
                self.label_comm += ", " + self.label_name[nIdx]
            else:
                raise TypeError("Only categorical and continuous variables are allowed")
        
        self.label_comm += ")"
        
    def send_labels(self):
        ro.r(self.label_comm)
