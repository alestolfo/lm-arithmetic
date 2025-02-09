import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.number_utils import convert_to_words, is_int

sns.set()
pd.set_option('display.max_columns', 50)


def plot_distributions(distrib_base, distrib_alt):
    x = np.arange(len(distrib_base))
    plt.plot(x, distrib_base)
    plt.plot(x, distrib_alt, 'red')
    plt.show()


def plot_distribution(distrib_base, color='blue', figsize=(10, 5)):
    x = np.arange(len(distrib_base))
    fig = plt.figure(figsize=figsize)
    plt.plot(x, distrib_base, color, figure=fig)
    plt.xlabel('$n$')
    plt.ylabel('$P(n)$')
    plt.show()


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def top_k_pred(distrib, k):
    return np.argpartition(distrib, -k)[-k:][::-1]


def plot_distributions_of_example(df, example_index):
    base_string = df.iloc[example_index]['base_string']
    alt_string = df.iloc[example_index]['alt_string']
    distrib_base = df.iloc[example_index]['distrib_base']
    distrib_alt = df.iloc[example_index]['distrib_alt']
    # distrib_base = np.array([float(x) for x in distrib_base[1:-1].replace(',', '').split()])
    # distrib_alt = np.array([float(x) for x in distrib_alt[1:-1].replace(',', '').split()])

    print('base_string: ', base_string)
    print('alt_string', alt_string)

    if 'res' in df.columns:
        res = df.iloc[example_index]['res']
        print('res', res)
        res_prob_base = distrib_base[int(res)]
        res_prob_alt = distrib_alt[int(res)]
        old_rel_change_prob = np.abs(res_prob_base - res_prob_alt) / res_prob_alt
        d1 = (res_prob_base - res_prob_alt) / res_prob_alt
        d2 = (res_prob_alt - res_prob_base) / res_prob_base
        rel_change_prob = (d1 + d2) / 2
        print('rel_change_prob', rel_change_prob)
    elif 'res_base' in df.columns:
        res_base = df.iloc[example_index]['res_base']
        res_alt = df.iloc[example_index]['res_alt']
        print('res_base', res_base)
        print('res_alt', res_alt)
        res_prob_base = distrib_base[int(res_base)]
        res_alt_prob_base = distrib_base[int(res_alt)]
        print('res_alt_prob_base', res_alt_prob_base)
        res_prob_alt = distrib_alt[int(res_alt)]
        res_base_prob_alt = distrib_alt[int(res_base)]
        print('res_base_prob_alt', res_base_prob_alt)

    print('res_prob_base: ', res_prob_base)
    print('res_prob_alt: ', res_prob_alt)

    base_top_5_pred = top_k_pred(distrib_base, 5)
    alt_top_5_pred = top_k_pred(distrib_alt, 5)
    print('base_top_5_pred', base_top_5_pred)
    print('base_top_5_prob', distrib_base[base_top_5_pred])
    print('alt_top_5_pred', alt_top_5_pred)
    print('alt_top_5_prob', distrib_alt[alt_top_5_pred])

    plot_distributions(distrib_base, distrib_alt)
    return distrib_base, distrib_alt


def compute_effects_alt(df):
    # outdated, the original way of computing effects
    df['base_correctness'] = df.res_alt_base_prob / df.res_base_base_prob
    df['alt_correctness'] = df.res_base_alt_prob / df.res_alt_alt_prob
    df['yz'] = df.res_alt_prob / df.res_base_prob
    df['effect'] = df.yz / df.base_correctness - 1
    df['total_effect'] = 1 / (df.alt_correctness * df.base_correctness) - 1


def compute_res_probs_full_dist(df, representation='arabic'):
    # sets res_base_prob and res_alt_prob for df with full distributions
    res_base_probs = []
    res_alt_probs = []
    words_to_n = {convert_to_words(str(i)): i for i in range(300 + 1)}
    for i, r in df.iterrows():
        if representation == 'words':
            res_base = int(words_to_n[r.res_base])
            res_alt = int(words_to_n[r.res_alt])
        elif representation == 'arabic':
            res_base = int(r.res_base)
            res_alt = int(r.res_alt)
        else:
            raise ValueError('representation must be arabic or words')

        res_base_prob = r.distrib_alt[res_base]
        res_alt_prob = r.distrib_alt[res_alt]
        res_base_probs.append(res_base_prob)
        res_alt_probs.append(res_alt_prob)
    df['res_base_prob'] = res_base_probs
    df['res_alt_prob'] = res_alt_probs


def compute_effects(df, representation='arabic'):
    if 'res_base_prob' not in df.columns:
        print('computing res_base_prob and res_alt_prob from the distributions')
        compute_res_probs_full_dist(df, representation)
    df['base_correctness'] = df.res_alt_base_prob / df.res_base_base_prob
    df['alt_correctness'] = df.res_base_alt_prob / df.res_alt_alt_prob
    df['yz'] = df.res_alt_prob / df.res_base_prob
    p_prime_g = df.res_base_prob
    p_prime_g_prime = df.res_alt_prob
    p_g = df.res_base_base_prob
    p_g_prime = df.res_alt_base_prob
    df['effect'] = ((p_g - p_prime_g) / p_prime_g + (p_prime_g_prime - p_g_prime) / p_g_prime) / 2
    df['effect_relative_increase_p_res_alt'] = (p_prime_g_prime - p_g_prime) / p_g_prime
    df['effect_average_diff'] = ((p_g - p_prime_g) + (p_prime_g_prime - p_g_prime))
    df['effect_increase_p_res_alt'] = (p_prime_g_prime - p_g_prime)
    df['effect_diff_int2'] = abs((p_g - p_prime_g))
    df['total_effect_old'] = 1 / (df.alt_correctness * df.base_correctness) - 1
    df['total_effect'] = ((p_g - df.res_base_alt_prob) / df.res_base_alt_prob + (
            df.res_alt_alt_prob - p_g_prime) / p_g_prime) / 2


def compute_prediction_change(df, type='all'):
    def make_int(x):
        if isinstance(x, str):
            if is_int(x):
                return int(x)
            else:
                return words_to_n[x]
        else:
            return x

    words_to_n = {convert_to_words(str(i)): i for i in range(300 + 1)}
    pred_changes = []
    for i, r in df.iterrows():
        pred_base = r.pred_base
        if r.distrib_alt is None:
            raise ValueError('distrib_alt is None')
        d_alt = r.distrib_alt
        pred_alt = np.argmax(d_alt)
        pred_change = pred_alt != pred_base
        res_base = make_int(r.res_base)
        res_alt = make_int(r.res_alt)
        if type == 'desired':
            pred_change = pred_change and pred_alt == res_alt
        elif type == 'undesired':
            pred_change = pred_change and pred_base == res_base
        pred_changes.append(int(pred_change))
    df['pred_change'] = pred_changes


def plot_pred_change(df, type='all', figsize=(15, 6)):
    cols = [
        'neuron', 'layer',
        'accuracy',
        'pred_change',
    ]

    if type == 'both':
        compute_prediction_change(df, 'desired')

        layer_aggregated = df[cols].groupby(['layer']).agg(['mean', 'std', 'sem'])
        layer_aggregated.columns = ['_'.join(col) for col in layer_aggregated.columns]

        fig = plt.figure(figsize=figsize)

        t = layer_aggregated.sort_values('layer').reset_index().copy()
        effect_desired_mean = t.pred_change_mean
        effect_desired_std = t.pred_change_sem

        compute_prediction_change(df, 'undesired')

        layer_aggregated = df[cols].groupby(['layer']).agg(['mean', 'std', 'sem'])
        layer_aggregated.columns = ['_'.join(col) for col in layer_aggregated.columns]

        fig = plt.figure(figsize=figsize)

        t = layer_aggregated.sort_values('layer').reset_index()
        effect_undesired_mean = t.pred_change_mean
        effect_undesired_std = t.pred_change_sem

        colors = ['teal', 'orange']
        title = 'Prediction Change (Last Token)'


        for mean, std, color, label in [(effect_desired_mean, effect_desired_std, 'teal', 'Desired'),
                                        (effect_undesired_mean, effect_undesired_std, 'orange', 'Undesired')]:
            line, = plt.plot(
                df.layer.unique(),
                mean,
                figure=fig,
                label=label, color=color, linestyle='-', alpha=0.5,
            )
            plt.fill_between(
                df.layer.unique(),
                mean + std,
                mean - std,
                figure=fig,
                alpha=0.1, color=color
            )
        # ax.set_yscale('log')
        plt.legend(loc='upper left')
        plt.title(title)
        plt.xlabel('Layer')
        plt.ylabel('% of Prediction Change')
        # plt.ylim([-0.1, 1.1])
        plt.hlines(0, 0, len(mean), color='black', alpha=0.5, linestyle='dotted')
        plt.tight_layout()
        # plt.show()
    else:
        compute_prediction_change(df, 'undesired')

        compute_prediction_change(df, type)

        layer_aggregated = df[cols].groupby(['layer']).agg(['mean', 'std', 'sem'])
        layer_aggregated.columns = ['_'.join(col) for col in layer_aggregated.columns]

        fig = plt.figure(figsize=figsize)

        df = layer_aggregated.sort_values('layer').reset_index()
        effect_mean = df.pred_change_mean
        effect_std = df.pred_change_sem

        if type == 'desired':
            color = 'teal'
            title = 'Desired Change in Prediction'
        elif type == 'undesired':
            color = 'orange'
            title = 'Undesired Change in Prediction'
        else:
            color = 'orange'
            title = 'Prediction change'

        ax = fig.add_subplot(2, 1, 1)
        line, = ax.plot(
            df.layer.unique(),
            effect_mean,
            label=f'prediction change', color=color, linestyle='-', alpha=0.5,
        )
        ax.fill_between(
            df.layer.unique(),
            effect_mean + effect_std,
            effect_mean - effect_std,
            alpha=0.1, color=color
        )
        # ax.set_yscale('log')
        # plt.legend()
        plt.title(title)
        plt.xlabel('Layer')
        plt.ylabel('% of Prediction Change')
        # plt.ylim([-0.1, 1.1])
        plt.hlines(0, 0, len(effect_mean), color='black', alpha=0.5, linestyle='dotted')
        plt.tight_layout()
        # plt.show()


def load_df(path, split_operators=False, accuracy_filter=None, representation='arabic', mult_by_1_filter=False):
    cols = [
        'neuron', 'layer',
        'effect', 'accuracy',
        'total_effect', 'effect_average_diff', 'effect_diff_int2', 'effect_increase_p_res_alt',
        'effect_relative_increase_p_res_alt'
    ]

    df = pd.read_feather(path)
    if accuracy_filter is not None:
        df = df[df.accuracy == accuracy_filter]

    if mult_by_1_filter:
        mult_by_1 = []
        for i, r in df.iterrows():
            if r.operation == '*':
                if '1' in r.base_string.split() or '1' in r.alt_string.split():
                    mult_by_1.append(True)
                else:
                    mult_by_1.append(False)
            else:
                mult_by_1.append(False)
        df['mult_by_1'] = mult_by_1
        df = df[df.mult_by_1 == False]

    compute_effects(df, representation)

    layer_aggregated = df[cols].groupby(['layer', 'neuron']).agg(['mean', 'max', 'std', 'sem'])
    layer_aggregated.columns = ['_'.join(col) for col in layer_aggregated.columns]

    acc = df.accuracy.mean()
    print(f'accuracy\t\t| {acc}')

    if split_operators:
        op_agg = {}
        overall_acc = 0
        for op in ['+', '-', '*', '/']:
            op_agg[op] = df[df.equation == f'x {op} y'][cols].groupby(['layer', 'neuron']).agg(
                ['mean', 'max', 'std', 'sem'])
            op_agg[op].columns = ['_'.join(col) for col in op_agg[op].columns]

            acc = op_agg[op].accuracy_mean.mean()
            overall_acc += acc
            print(f'accuracy on op {op} | {acc}')

        print(f'overall accuracy | {overall_acc / 4}')

        layer_aggregated = op_agg

    return df, layer_aggregated


def plot_layer_aggregated(layer_aggregated, split_operators=False, operator=None, ylim=None, figsize=(15, 6),
                          color='black', title='Indirect Effects by layer', type='relative_diff', log=False):
    if split_operators:
        plot_layer_aggregated_split(layer_aggregated, figsize=figsize, ylim=ylim)
    else:
        fig = plt.figure(figsize=figsize)
        df = layer_aggregated.sort_values('layer').reset_index()
        if type == 'diff_int2':
            effect_mean = df.effect_diff_int2_mean
            effect_std = df.effect_diff_int2_sem
        elif type == 'p_res_increase':
            effect_mean = df.effect_increase_p_res_alt_mean
            effect_std = df.effect_increase_p_res_alt_sem
        elif type == 'absolute_diff':
            effect_mean = df.effect_average_diff_mean
            effect_std = df.effect_average_diff_sem
        elif type == 'relative_diff':
            effect_mean = df.effect_mean
            effect_std = df.effect_sem

        label = f'{operator}' if operator else 'overall'
        line, = plt.plot(
            df.layer.unique(),
            effect_mean,
            figure=fig,
            label=label, color=color, linestyle='-', alpha=0.5,
        )
        plt.fill_between(
            df.layer.unique(),
            effect_mean + effect_std,
            effect_mean - effect_std,
            figure=fig,
            alpha=0.1, color=color
        )
        if log:
            plt.set_yscale('log')
        plt.title(title)
        plt.xlabel('Layer')
        plt.ylabel('IE')
        if ylim:
            plt.ylim(ylim)
        plt.hlines(0, 0, len(effect_mean), color='black', alpha=0.5, linestyle='dotted')
        plt.tight_layout()


def plot_effects_across_tokens(df, figsize=(15, 6), title='Indirect Effect'):
    position_classes = []
    for i, r in df.iterrows():
        last_pos = df[df.base_string == r.base_string].position.max()
        if r.position < r.op1_pos:
            position_classes.append('early')
        elif r.position == r.op1_pos:
            position_classes.append('op1')
        elif r.position == r.op2_pos:
            position_classes.append('op2')
        elif r.position == r.operator_pos:
            position_classes.append('operator')
        elif r.position == last_pos:
            position_classes.append('last')
        elif r.position > r.op2_pos:
            position_classes.append('late')
        else:
            position_classes.append('middle')

    df['position_class'] = position_classes

    for pos_class in df.position_class.unique():
        df_pos = df[df.position_class == pos_class]
        plot_layer_aggregated(df_pos, figsize=figsize, title=f'{title} ({pos_class})')


def plot_layer_aggregated_split(layer_aggregated, ylim=None, figsize=(15, 6)):
    colors = {
        '+': 'black',
        '-': 'red',
        '*': 'blue',
        '/': 'green'
    }
    titles = {
        '+': 'Addition',
        '-': 'Subtraction',
        '*': 'Multiplication',
        '/': 'Division'
    }

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=False, figsize=figsize)

    for (x, y), op in zip([(0, 0), (0, 1), (1, 0), (1, 1)], ['+', '-', '*', '/']):
        df = layer_aggregated[op].sort_values('layer').reset_index()
        effect_mean = df.effect_mean
        effect_std = df.effect_sem

        ax = axs[x, y]

        color = colors[op]

        line, = ax.plot(
            df.layer.unique(),
            effect_mean,
            label='None', color=color, linestyle='-', alpha=0.5,
        )
        ax.fill_between(
            df.layer.unique(),
            effect_mean + effect_std,
            effect_mean - effect_std,
            alpha=0.1, color=color
        )
        # ax.set_yscale('log')
        # plt.legend()
        ax.set_title(titles[op])
        if ylim:
            plt.ylim(ylim)
        plt.hlines(0, 0, len(effect_mean), color='black', alpha=0.5, linestyle='dotted')
        plt.tight_layout()

        ax.set(ylabel='IE')

        if op == '*':
            ax.set(xlabel='Layer')

    ax.set(xlabel='Layer')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # ax.label_outer()


def compute_overlap(l1, l2):
    set1 = set(l1)
    set2 = set(l2)
    overlap = len(set1.intersection(set2))
    overlap_ratio = overlap / len(l1)
    return overlap_ratio


def compute_neuron_overlap_single(layer_aggregated1, layer_aggregated2, top_k):
    top_neurons1 = layer_aggregated1.sort_values('effect_mean')[-top_k:].index
    top_neurons2 = layer_aggregated2.sort_values('effect_mean')[-top_k:].index

    overlap_ratio = compute_overlap(top_neurons1, top_neurons2)
    print(f'overlap_ratio: {overlap_ratio}')


def compute_neuron_overlap(layer_aggregated1, layer_aggregated2, top_k):
    top_neurons = {}
    for op in ['+', '-', '*', '/']:
        top_neurons[op] = layer_aggregated2[op].sort_values('effect_mean')[-top_k:].index

    top_neurons1 = layer_aggregated1.sort_values('effect_mean')[-top_k:].index

    overlap_ratios = {}
    for op1 in ['+', '-', '*', '/']:
        overlap_ratio = compute_overlap(top_neurons1, top_neurons[op1])
        overlap_ratios[f'{op1}'] = overlap_ratio
        print(f'overlap_ratio with {op1}: {overlap_ratio}')

    return overlap_ratios


def compute_neuron_overlap_for_operations(layer_aggregated, top_k):
    top_neurons = {}
    for op in ['+', '-', '*', '/']:
        top_neurons[op] = layer_aggregated[op].sort_values('effect_mean')[-top_k:].index

    overlap_ratios = {}
    for op1 in ['+', '-', '*', '/']:
        for op2 in ['+', '-', '*', '/']:
            overlap_ratio = compute_overlap(top_neurons[op1], top_neurons[op2])
            overlap_ratios[f'{op1} {op2}'] = overlap_ratio
            print(f'overlap_ratio {op1} {op2}: {overlap_ratio}')
        print('=====================')

    return overlap_ratios


def plot_neurons(layer_aggregated, sort=False, split_operators=False, operator=None, ylim=None):
    if split_operators:
        for op in ['+', '-', '*', '/']:
            plot_neurons(layer_aggregated[op], sort=sort, split_operators=False, operator=op, ylim=ylim)
        return
    colors = {
        '+': 'black',
        '-': 'red',
        '*': 'blue',
        '/': 'green'
    }
    fig = plt.figure(figsize=(15, 6))
    df = layer_aggregated
    if sort:
        print('sorting')
        df = df.sort_values('effect_mean').reset_index()
    effect_mean = df.effect_mean
    effect_std = df.effect_sem

    ax = fig.add_subplot(2, 1, 1)
    label = f'{operator}' if operator else 'overall'
    color = colors[operator] if operator else 'black'
    line, = ax.plot(
        df.index.unique(),
        effect_mean,
        label=label, color=color, linestyle='-', alpha=0.5,
    )
    ax.fill_between(
        df.index.unique(),
        effect_mean + effect_std,
        effect_mean - effect_std,
        alpha=0.1, color=color
    )
    # ax.set_yscale('log')
    plt.legend()
    plt.title('Indirect effects of single neurons')
    plt.xlabel('Neuron')
    plt.ylabel('Indirect effect')
    if ylim:
        plt.ylim(ylim)
    plt.hlines(0, 0, len(effect_mean), color='black', alpha=0.5, linestyle='dotted')
    plt.show()
