import numpy as np
import pandas as pd
import networkx as nx
import graphviz
import matplotlib.pyplot as plt
import os
import sys

from dowhy import CausalModel
import dowhy.datasets
import pydot
import argparse

# Structure learning library
from rpy2.robjects import r as R

from cdt.causality.graph import LiNGAM, PC, GES

# Avoid printing dataconversion warnings from sklearn
import warnings
from sklearn.exceptions import DataConversionWarning
from sympy import O
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from econml.inference import BootstrapInference

# Config dict to set the logging level
import logging.config
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'WARN',
        },
    }
}

import matplotlib.pyplot as plt
from datetime import datetime

SMALL_SIZE = 8
MEDIUM_SIZE = 26
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_treatment_outcome(treatment, outcome, time_var):
    fig, ax = plt.subplots()
    tline = ax.plot(time_var, treatment, 'o', label="Treatment")
    oline = ax.plot(time_var, outcome, 'r^', label="Outcome")

    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
    plt.xlabel("Time")
    fig.set_size_inches(8, 6)
    fig.savefig("obs_data" + datetime.now().strftime("%H-%M-%S") + ".png",
                bbox_inches="tight")


def plot_causal_effect(estimate, treatment, outcome, treatment_name, outcome_name, prefix="effect-of_"):
    fig, ax = plt.subplots()
    x_min = 0
    x_max = max(treatment)
    y_min = estimate.params["intercept"]
    y_max = y_min + estimate.value * (x_max - x_min)
    ax.scatter(treatment, outcome, c="gray", marker="o", label="Observed data")
    ax.plot([x_min, x_max], [y_min, y_max], c="black", ls="solid", lw=4,
            label="Causal variation")
    ax.set_ylim(0, max(outcome))
    ax.set_xlim(0, x_max)

    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax.text(10.8, 1, r"DoWhy estimate $\rho$ (slope) = " + str(round(estimate.value, 2)),
            ha="right", va="bottom", size=20, bbox=bbox_props)
    ax.legend(loc="upper left")
    plt.xlabel("Treatment")
    plt.ylabel("Outcome")

    fig.set_size_inches(8, 6)
    fig.savefig(prefix + treatment_name + "-on_" + outcome_name + "-" + datetime.now().strftime("%H-%M-%S") + ".png",
                bbox_inches='tight')


logging.config.dictConfig(DEFAULT_LOGGING)
ATTACKS = ['Oak17Acc', 'Top3MLLeakAcc', 'Top3MLLeakLAcc', 'MLLeakAcc', 'MLLeakLAcc', 'ThresholdAcc']
LOSSES = ['ce', 'mse']
FEATURES = [
    'TrainBias2', 'TestBias2',
    'TrainAcc', 'TestAcc',
    'TrainLoss', 'TestLoss',
    'TrainVar', 'TestVar',
    'LossDiff',
    'AccDiff',
    'NumParams', 'TrainSize',
    'CentroidDistance.origin.', 'CentroidDistance.sorted_3.']
TREATMENT_VARS = {
    'ThresholdAcc': [x for x in FEATURES if x not in ['CentroidDistance.sorted_3.']],
    'Oak17Acc': [x for x in FEATURES if x != 'CentroidDistance.sorted_3.'],
    'Top3MLLeakAcc': [x for x in FEATURES if x not in ['CentroidDistance.origin.']],
    'Top3MLLeakLAcc': [x for x in FEATURES if x not in ['CentroidDistance.origin.']],
    'MLLeakAcc': [x for x in FEATURES if x not in ['CentroidDistance.sorted_3.']],
    'MLLeakLAcc': [x for x in FEATURES if x not in ['CentroidDistance.sorted_3.']],
}
GRAPH_DIR = 'dowhy-graphs'

QUERIES_OF_INTEREST = {
    'Oak17Acc': ['AccDiff', 'TrainVar', 'TestVar', 'TrainSize', 'NumParams'],
    'Top3MLLeakAcc': ['AccDiff', 'TrainVar', 'TestVar', 'TrainSize', 'NumParams', 'CentroidDistance.sorted_3.'],
    'Top3MLLeakLAcc': ['AccDiff', 'TrainVar', 'TestVar', 'TrainSize', 'NumParams', 'CentroidDistance.sorted_3.'],
    'MLLeakAcc': ['AccDiff', 'TrainVar', 'TestVar', 'TrainSize', 'NumParams', 'CentroidDistance.origin.'],
    'MLLeakLAcc': ['AccDiff', 'TrainVar', 'TestVar', 'TrainSize', 'NumParams', 'CentroidDistance.origin.'],
    'ThresholdAcc': ['AccDiff', 'TrainVar', 'TestVar', 'TrainSize', 'NumParams']
}

import keras

def run_dml_estimate(model, identified_estimand, control_value, treatment_value):
    """
    Non-linear estimator
    """
    dml_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.econml.dml.DML",
                                         control_value=control_value,
                                         treatment_value=treatment_value,
                                         confidence_intervals=False,
                                         method_params={"init_params": {
                                             'model_y': GradientBoostingRegressor(),
                                             'model_t': GradientBoostingRegressor(),
                                             "model_final": LassoCV(fit_intercept=False),
                                             'featurizer': PolynomialFeatures(degree=1, include_bias=False)},
                                             "fit_params": {}})
    print("DML Causal estimate is ")
    print(dml_estimate)
    print('='*80)
    print('================================ DML Varying estimator =============================')
    if dml_estimate.value > 0:
        interpretation = dml_estimate.interpret(method_name="textual_effect_interpreter")
        print(interpretation)
        random_refute_results = model.refute_estimate(identified_estimand, dml_estimate,
                                                      method_name="random_common_cause")
        print(random_refute_results)

        # random subset validation
        subset_refute_results = model.refute_estimate(
            identified_estimand,
            dml_estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8
        )
        print(subset_refute_results)

    return dml_estimate

def run_example_1_paper(dataset, wd):
    df = pd.read_csv(dataset, skip_blank_lines=True)
    df = df.loc[df['WeightDecay'] == wd]
    df.rename(columns={'CentroidDistance(origin)': 'CentroidDistance.origin.',
                       'CentroidDistance(sorted_3)': 'CentroidDistance'},
              inplace=True)
    loss_df = df.loc[df['Loss'] == "ce"]
    # draw graph
    attack = 'Oak17Acc'
    nodes = ['AccDiff', 'NumParams'] + [attack]
    graph1 = """
        digraph {
            AccDiff;
            NumParams;
            Oak17Acc;
            NumParams -> AccDiff;
            NumParams -> Oak17Acc;
            AccDiff -> Oak17Acc;
        }
        """
    VARS = nodes
    current_df  = loss_df[nodes].copy()
    current_df[["NumParams"]] = current_df[["NumParams"]].apply(pd.to_numeric)
    scaler = sklearn.preprocessing.StandardScaler()
    current_df[VARS] = scaler.fit_transform(current_df[VARS])

    # normalize the data
    scaler = sklearn.preprocessing.MinMaxScaler()
    current_df[VARS] = scaler.fit_transform(current_df[VARS])

    print(current_df)
    model1 = CausalModel(
        data=current_df,
        treatment='AccDiff',
        treatment_is_binary=False,
        outcome=attack,
        graph=graph1.replace("\n", " "))

    #model1.view_model() #file_name='dowhy-{}-causal_model.png'.format(attack))
    model1.view_model(file_name='example_1_1-{}-causal_model'.format(attack))

    # II. Identify causal effect and return target estimands
    identified_estimand = model1.identify_effect(proceed_when_unidentifiable=True)
    print('Identified estimand: {}'. format(identified_estimand))

    print('='*60)
    # III. Estimate the target estimand using a statistical method.
    # propensity_score only for binary treatment variables
    causal_estimate = model1.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression",
                                            control_value=min(current_df['AccDiff']),
                                            treatment_value=max(current_df['AccDiff']),
                                            confidence_intervals=True,
                                            test_significance=True)
    print("Causal Estimate is")
    print(causal_estimate)
    graph2 = """
        digraph {
            AccDiff;
            NumParams;
            Oak17Acc;
            NumParams -> AccDiff;
            NumParams -> Oak17Acc;
            AccDiff -> Oak17Acc;
        }
        """
    VARS = nodes
    current_df  = loss_df[nodes].copy()
    current_df[["NumParams"]] = current_df[["NumParams"]].apply(pd.to_numeric)
    scaler = sklearn.preprocessing.StandardScaler()
    current_df[VARS] = scaler.fit_transform(current_df[VARS])

    # normalize the data
    scaler = sklearn.preprocessing.MinMaxScaler()
    current_df[VARS] = scaler.fit_transform(current_df[VARS])

    print(current_df)
    model1 = CausalModel(
        data=current_df,
        treatment='NumParams',
        treatment_is_binary=False,
        outcome=attack,
        graph=graph1.replace("\n", " "))

    #model1.view_model() #file_name='dowhy-{}-causal_model.png'.format(attack))
    model1.view_model(file_name='example_1_2-{}-causal_model'.format(attack))

    # II. Identify causal effect and return target estimands
    identified_estimand = model1.identify_effect(proceed_when_unidentifiable=True)
    print('Identified estimand: {}'. format(identified_estimand))

    print('='*60)
    # III. Estimate the target estimand using a statistical method.
    # propensity_score only for binary treatment variables
    causal_estimate = model1.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression",
                                            control_value=min(current_df['NumParams']),
                                            treatment_value=max(current_df['NumParams']),
                                            confidence_intervals=True,
                                            test_significance=True)
    print("Causal Estimate is")
    print(causal_estimate)




def run_example_2_paper(dataset, wd):
    df = pd.read_csv(dataset, skip_blank_lines=True)
    df = df.loc[df['WeightDecay'] == wd]
    df.rename(columns={'CentroidDistance(origin)': 'CentroidDistance.origin.',
                       'CentroidDistance(sorted_3)': 'CentroidDistance'},
              inplace=True)
    loss_df = df.loc[df['Loss'] == "ce"]
    # draw graph
    attack = 'Top3MLLeakAcc'
    nodes = ['AccDiff', 'NumParams', 'CentroidDistance'] + [attack]
    # Graph with TrainSize influencing the Oak17Acc
    graph1 = """
        digraph {
            AccDiff;
            NumParams;
            CentroidDistance;
            Top3MLLeakAcc;
            NumParams -> AccDiff;
            NumParams -> Top3MLLeakAcc;
            CentroidDistance -> AccDiff;
            CentroidDistance -> Top3MLLeakAcc;
            AccDiff -> Top3MLLeakAcc;
        }
        """
    VARS = nodes
    current_df  = loss_df[nodes].copy()
    current_df[["NumParams"]] = current_df[["NumParams"]].apply(pd.to_numeric)
    scaler = sklearn.preprocessing.StandardScaler()
    current_df[VARS] = scaler.fit_transform(current_df[VARS])

    # normalize the data
    scaler = sklearn.preprocessing.MinMaxScaler()
    current_df[VARS] = scaler.fit_transform(current_df[VARS])

    print(current_df)
    model1 = CausalModel(
        data=current_df,
        treatment='AccDiff',
        treatment_is_binary=False,
        outcome=attack,
        graph=graph1.replace("\n", " "))

    #model1.view_model() #file_name='dowhy-{}-causal_model.png'.format(attack))
    model1.view_model(file_name='example_2_1-{}-causal_model'.format(attack))

    # II. Identify causal effect and return target estimands
    identified_estimand = model1.identify_effect(proceed_when_unidentifiable=True)
    print('Identified estimand: {}'. format(identified_estimand))

    print('='*60)
    # III. Estimate the target estimand using a statistical method.
    # propensity_score only for binary treatment variables
    causal_estimate = model1.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression",
                                            control_value=min(current_df['AccDiff']),
                                            treatment_value=max(current_df['AccDiff']),
                                            confidence_intervals=True,
                                            test_significance=True)
    print("Causal Estimate is")
    print(causal_estimate)

    current_df  = loss_df[nodes].copy()
    current_df[["NumParams"]] = current_df[["NumParams"]].apply(pd.to_numeric)
    scaler = sklearn.preprocessing.StandardScaler()
    current_df[VARS] = scaler.fit_transform(current_df[VARS])

    # normalize the data
    scaler = sklearn.preprocessing.MinMaxScaler()
    current_df[VARS] = scaler.fit_transform(current_df[VARS])

    # Graph with TrainSize not influencing the Oak17Acc
    graph2 = """
        digraph {
            AccDiff;
            NumParams;
            Top3MLLeakAcc;
            CentroidDistance;
            NumParams -> AccDiff;
            CentroidDistance -> AccDiff;
            NumParams -> Top3MLLeakAcc;
            AccDiff -> Top3MLLeakAcc;
        }
        """

    model2 = CausalModel(
        data=current_df,
        treatment='AccDiff',
        treatment_is_binary=False,
        outcome=attack,
        graph=graph2.replace("\n", " "))

    model2.view_model(file_name='example_2_2-{}-causal_model'.format(attack))

    # II. Identify causal effect and return target estimands
    identified_estimand = model2.identify_effect(proceed_when_unidentifiable=True)
    print('Identified estimand: {}'. format(identified_estimand))

    print('='*60)
    # III. Estimate the target estimand using a statistical method.
    # propensity_score only for binary treatment variables
    causal_estimate = model2.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression",
                                            control_value=min(current_df['AccDiff']),
                                            treatment_value=max(current_df['AccDiff']),
                                            confidence_intervals=True,
                                            test_significance=True)
    print("Causal Estimate is")
    print(causal_estimate)


def run_example_3_paper(dataset, wd):
    df = pd.read_csv(dataset, skip_blank_lines=True)
    df.rename(columns={'CentroidDistance(origin)': 'CentroidDistance',
                       'CentroidDistance(sorted_3)': 'CentroidDistance.sorted_3.'},
              inplace=True)
    df = df.loc[df['WeightDecay'] == wd]

    loss_df = df.loc[df['Loss'] == "ce"]
    # draw graph
    attack = 'Oak17Acc'
    nodes = ['AccDiff', 'NumParams', 'Oak17Acc', 'TrainSize', 'CentroidDistance']
    graph = """
        digraph {
            AccDiff;
            NumParams;
            Oak17Acc;
            TrainSize;
            CentroidDistance;
            AccDiff -> CentroidDistance;
            NumParams -> AccDiff;
            NumParams -> Oak17Acc;
            TrainSize -> AccDiff;
            TrainSize -> Oak17Acc;
            CentroidDistance -> Oak17Acc;
        }
        """
    VARS = ['NumParams', 'TrainSize', 'AccDiff', 'CentroidDistance']
    current_df  = loss_df[nodes].copy()
    current_df.loc[current_df["TrainSize"] == "5k", "TrainSize"] = 5000.0
    current_df.loc[current_df["TrainSize"] == "1k", "TrainSize"] = 1000.0
    current_df[["TrainSize"]] = current_df[["TrainSize"]].apply(pd.to_numeric)
    current_df[["NumParams"]] = current_df[["NumParams"]].apply(pd.to_numeric)
    scaler = sklearn.preprocessing.StandardScaler()
    current_df[VARS] = scaler.fit_transform(current_df[VARS])

    # normalize the data
    scaler = sklearn.preprocessing.MinMaxScaler()
    current_df[VARS] = scaler.fit_transform(current_df[VARS])

    print(current_df)
    model = CausalModel(
        data=current_df,
        treatment='CentroidDistance',
        treatment_is_binary=False,
        outcome=attack,
        graph=graph.replace("\n", " "))

    model.view_model() #file_name='dowhy-{}-causal_model.png'.format(attack))

    # II. Identify causal effect and return target estimands
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print('Identified estimand: {}'. format(identified_estimand))

    print('='*60)
    # III. Estimate the target estimand using a statistical method.
    # propensity_score only for binary treatment variables
    causal_estimate = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression",
                                            control_value=min(current_df['CentroidDistance']),
                                            treatment_value=max(current_df['CentroidDistance']),
                                            confidence_intervals=True,
                                            #evaluate_effect_strength=True,
                                            test_significance=True)
    print("Causal Estimate is")
    print(causal_estimate)


def causal_w_bnlearn_graph(df, graph_dot_dict, output_csv_filename, run_refuters, run_poly_estimator=False, plot=False):
    causal_estimates_pd = pd.DataFrame(columns=['Loss', 'Attack', 'Feature',
                                                'ATE Mean value',
                                                'ATE p-value',
                                                'Random Cause Refuter Estimated Effect',
                                                'Random Cause Refuter New Effect',
                                                'Placebo Treatment Refuter',
                                                'Placebo Treatment p_value',
                                                'Subset Refuter',
                                                'Subset Refuter p_value'])
    ce_queries_pd = pd.DataFrame(columns=['Attack', 'Feature', 'ATE', 'p-value'])
    mse_queries_pd = pd.DataFrame(columns=['Attack', 'Feature', 'ATE', 'p-value'])

    for loss in LOSSES:
        loss_df = df.loc[df['Loss'] == loss]
        for attack in ATTACKS:
            if attack not in graph_dot_dict[loss]:
                continue
            print('Reading graph from {}'.format(graph_dot_dict[loss][attack]))
            graphs = pydot.graph_from_dot_file(graph_dot_dict[loss][attack])
            graph = graphs[0]

            graph = nx.drawing.nx_pydot.from_pydot(graph)
            pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
            nx.draw_networkx(graph, pos=pos)
            if not os.path.exists(GRAPH_DIR):
                os.mkdir(GRAPH_DIR)
            plt.savefig('{}/{}-{}-bnlearn_graph.pdf'.format(GRAPH_DIR, loss, attack), format='pdf')
            plt.clf()

            current_df = loss_df[TREATMENT_VARS[attack] + [attack]].copy()
            # Not nice, hard-coded preprocessing
            current_df.loc[current_df["TrainSize"] == "5k", "TrainSize"] = 5000.0
            current_df.loc[current_df["TrainSize"] == "1k", "TrainSize"] = 1000.0
            current_df[["TrainSize"]] = current_df[["TrainSize"]].apply(pd.to_numeric)
            current_df[["NumParams"]] = current_df[["NumParams"]].apply(pd.to_numeric)
            # standardise the data
            scaler = sklearn.preprocessing.StandardScaler()
            current_df[TREATMENT_VARS[attack]] = scaler.fit_transform(current_df[TREATMENT_VARS[attack]])

            # normalize the data
            scaler = sklearn.preprocessing.MinMaxScaler()
            current_df[TREATMENT_VARS[attack]] = scaler.fit_transform(current_df[TREATMENT_VARS[attack]])

            from sklearn.model_selection import train_test_split

            train, test = train_test_split(current_df, test_size=0.2)

            for treatment in TREATMENT_VARS[attack]:
                print('====== Treatment: {}; Outcome: {} ======'.format(treatment,
                                                                        attack))
                print('Observed variables: {}'.format(train.columns.tolist()))
                #print('Observed variables: {}'.format(df.columns.tolist()))
                print('='*80)
                print('Treatment Variable is {}'.format(treatment))
                model = CausalModel(
                    data=train,
                    treatment=treatment,
                    treatment_is_binary=False,
                    outcome=attack,
                    graph=graph_dot_dict[loss][attack])

                model.view_model() #file_name='dowhy-{}-causal_model.png'.format(attack))

                # II. Identify causal effect and return target estimands
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
                print('Identified estimand: {}'. format(identified_estimand))

                print('='*80)
                if run_poly_estimator:
                    run_dml_estimate(model, identified_estimand, min(current_df[treatment]), max(current_df[treatment]))

                causal_estimate = model.estimate_effect(identified_estimand,
                                                        method_name="backdoor.linear_regression",
                                                        control_value=min(current_df[treatment]),
                                                        treatment_value=max(current_df[treatment]),
                                                        confidence_intervals=True,
                                                        #evaluate_effect_strength=True,
                                                        test_significance=True)
                range_ate_mean = causal_estimate.value
                range_ate_p_value = 0
                if range_ate_mean != 0:
                    range_ate_p_value = causal_estimate.test_stat_significance()['p_value'][0]

                    # print("What is the effect on {} of intervening on {} by setting {} to {}?".format(attack, treatment, attack, treatment, 0))
                    interpretation = causal_estimate.interpret(method_name="textual_effect_interpreter")
                    print(interpretation)
                    if plot:
                        plot_causal_effect(causal_estimate, test[treatment], test[attack], treatment,
                                        attack, prefix="{}-effect-of_".format(loss))
                    print('='*80)
                    print('================================ Varying Estimator =============================')
                    interpretation = causal_estimate.interpret(method_name="textual_effect_interpreter")
                    print(interpretation)

                range_random_est_effect = 0
                range_random_new_effect = 0
                range_subset_new_effect = 0
                range_subset_p_value = 0
                if run_refuters and range_ate_mean != 0:
                    print("========================= Running refuters (min to max) ===================================")
                    range_random_refuter = model.refute_estimate(identified_estimand, causal_estimate,
                                                                 method_name="random_common_cause",
                                                                 num_simulations=10)
                    print(range_random_refuter)
                    range_random_est_effect = range_random_refuter.estimated_effect
                    range_random_new_effect = range_random_refuter.new_effect

                    # checks whether the estimator returns an estimate value of 0 when the action variable
                    # is replaced by a random variable, independent of all other variables.
                    # cate_placebo_refute_results = model.refute_estimate(identified_estimand, causal_estimate,
                    #                                                     method_name="placebo_treatment_refuter")
                    # print(cate_placebo_refute_results)

                    # random subset validation
                    range_subset_refuter = model.refute_estimate(
                        identified_estimand,
                        causal_estimate,
                        method_name="data_subset_refuter",
                        subset_fraction=0.8,
                        num_simulations=10
                    )
                    print(range_subset_refuter)
                    range_subset_new_effect = range_subset_refuter.new_effect
                    range_subset_p_value = range_subset_refuter.refutation_result['p_value']

                causal_estimates_pd = causal_estimates_pd.append({'Loss': loss, 'Attack': attack,
                                                                  'Feature': treatment,
                                                                  'ATE Mean value': range_ate_mean,
                                                                  'ATE p-value': range_ate_p_value,
                                                                  'ATE Random Cause Refuter Estimated Effect': range_random_est_effect,
                                                                  'ATE Random Cause Refuter New Effect': range_random_new_effect,
                                                                  'ATE Subset Refuter': range_subset_new_effect,
                                                                  'ATE Subset Refuter p_value': range_subset_p_value,
                                                                  },
                                                                 ignore_index=True)
                if treatment in QUERIES_OF_INTEREST[attack]:
                    if loss == 'ce':
                        ce_queries_pd = ce_queries_pd.append({'Attack': attack, 'Feature': treatment,
                                                              'ATE': range_ate_mean, 'p-value': range_ate_p_value},
                                                              ignore_index=True)
                    else:
                        mse_queries_pd = mse_queries_pd.append({'Attack': attack, 'Feature': treatment,
                                                                'ATE': range_ate_mean, 'p-value': range_ate_p_value},
                                                                ignore_index=True)

        causal_estimates_pd.to_csv(output_csv_filename)

        # Saving files for paper
        formats = {'ATE': '{:10.4f}', 'p-value': '{:.2E}'}

        sorted_df = ce_queries_pd.sort_values(by=["Attack", "Feature"])
        for col, f in formats.items():
            sorted_df[col] = sorted_df[col].map(lambda x: f.format(x))

        print(sorted_df)
        # Print the estimates for the paper
        sorted_df['Attack'] = sorted_df['Attack'].map({'MLLeakAcc': '\mlleakacc',
                                                       'MLLeakLAcc': '\mlleaklacc',
                                                       'Top3MLLeakAcc': '\mlleaktopacc',
                                                       'Top3MLLeakLAcc': '\mlleaktoplacc',
                                                       'Oak17Acc': '\oakacc',
                                                       'ThresholdAcc': '\\threshacc'})
        sorted_df['Feature'] = sorted_df['Feature'].map({'TrainVar': '\\trainvar',
                                                         'TestVar': '\\testvar',
                                                         'AccDiff': '\\accdiff',
                                                         'TrainSize': '\\trainsize',
                                                         'TrainBias2': '\\trainbias',
                                                         'TestBias2': '\\testbias',
                                                         'NumParams': '\\numparams',
                                                         'CentroidDistance.origin.': '\centroid',
                                                         'CentroidDistance.sorted_3.': '\centroid'})
        print(sorted_df)
        paper_res_filename = os.path.splitext(os.path.basename(output_csv_filename))[0]
        sorted_df.to_csv(paper_res_filename + ".ce.csv")

        sorted_df = mse_queries_pd.sort_values(by=["Attack", "Feature"])
        for col, f in formats.items():
            sorted_df[col] = sorted_df[col].map(lambda x: f.format(x))

        print(sorted_df)
        sorted_df['Attack'] = sorted_df['Attack'].map({'MLLeakAcc': '\mlleakacc',
                                                       'MLLeakLAcc': '\mlleaklacc',
                                                       'Top3MLLeakAcc': '\mlleaktopacc',
                                                       'Top3MLLeakLAcc': '\mlleaktoplacc',
                                                       'Oak17Acc': '\oakacc',
                                                       'ThresholdAcc': '\\threshacc'})
        sorted_df['Feature'] = sorted_df['Feature'].map({'TrainVar': '\\trainvar',
                                                         'TestVar': '\\testvar',
                                                         'AccDiff': '\\accdiff',
                                                         'TrainSize': '\\trainsize',
                                                         'NumParams': '\\numparams',
                                                         'CentroidDistance.origin.': '\centroid',
                                                         'CentroidDistance.sorted_3.': '\centroid'})
        print(sorted_df)
        sorted_df.to_csv(paper_res_filename + ".mse.csv")



def main(dataset, graph_dot_dict, wd, output_csv_filename, run_refuters):
    # This checks that the graph is initialized from the dot file
    #nx.draw_networkx(graph)
    #plt.show()
    df = pd.read_csv(dataset, skip_blank_lines=True)
    df = df.loc[df['WeightDecay'] == wd]
    df.rename(columns={'CentroidDistance(origin)': 'CentroidDistance.origin.',
                       'CentroidDistance(sorted_3)': 'CentroidDistance.sorted_3.'},
              inplace=True)
    # Ignore missing values
    if df.isnull().values.any():
        print('Warning! Dataset contains missing values!')
        df = df.dropna()
    print('Using the bnlearn graphs to do causal analysis')
    causal_w_bnlearn_graph(df, graph_dot_dict, output_csv_filename, run_refuters)

if __name__ == "__main__":
    graph_dot_dict = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset path')
    parser.add_argument('graph_folder', type=str, help='folder with the graphs saved as dot files')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='select the weight decay of 5e-4')
    parser.add_argument('--run_refuters', default=False, action='store_true', help='Run the random and subset refuters.')
    parser.add_argument('--filename', type=str,  default=None)
    parser.add_argument('--run_example', choices=['1', '2', '3', '0'], default=0, help='run the example')
    args = parser.parse_args()

    print(args.run_example)
    if args.run_example == '1':
        # Other numbers for example 1 from the paper is in script `statistical_analysis.py`
        # this example 1 assumes that we have a very simple graph
        # in the paper, we choose to present the values ultimately coming after the ATE analysis
        run_example_1_paper(args.dataset, args.wd)
        exit()
    if args.run_example == '2':
        run_example_2_paper(args.dataset, args.wd)
        exit()
    if args.run_example == '3':
        run_example_3_paper(args.dataset, args.wd)
        exit()

    for loss in LOSSES:
        graph_dot_dict[loss] = {}
        for attack in ATTACKS:
            filename = "{}/{}-{}-cv_avg_net_discovery.dot".format(args.graph_folder, loss, attack)
            if os.path.exists(filename):
                graph_dot_dict[loss][attack] = filename

    if args.filename is None:
        output_csv_filename = 'dowhy_causal_estimates-{}.csv'.format(args.wd)
    else:
        output_csv_filename = args.filename
    main(args.dataset, graph_dot_dict, args.wd, output_csv_filename, args.run_refuters)
