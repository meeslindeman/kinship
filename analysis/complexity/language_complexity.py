import os
import itertools
import pandas as pd 
from pandas import isna
from collections import defaultdict
from itertools import product
from math import log2
import json
from matplotlib import pyplot as plt
import numpy as np

    
# node mapping
NODES = [
    'MM', 'MF', 'MZy', 'MBy', 'M', 'MZe', 'MBe',
    'FM', 'FF', 'FZy', 'FBy', 'F', 'FZe', 'FBe',
    'Zy', 'By', 'Ego', 'Ze', 'Be', 'ZyD', 'ZyS',
    'ByD', 'ByS', 'D', 'S', 'ZeD', 'ZeS', 'BeD', 'BeS',
    'DD', 'DS', 'SD', 'SS'
]

N_SKIPED_EPOCHS = 50


def estimate_prob_given_count(counts):
    # collect counts with conditions
    count_u = defaultdict(lambda: 1e-10)  # for p(u)
    count_w_u = defaultdict(lambda: defaultdict(lambda: 1e-10))  # for p(w|u)
    count_u_w = defaultdict(lambda: defaultdict(lambda: 1e-10))  # for p(u|w)
    
    for us, w, c in counts:
        for u in us:
            count_u[u] += c / len(us)
            count_w_u[u][w] += 1
            count_u_w[w][u] += c / len(us)
    
    # estimate prob p(u), p(w|u), p(u|w)
    p_u = defaultdict(lambda: 1e-10, {u:count_u[u] / sum(count_u.values()) for u in count_u.keys()})
    
    p_w_u = defaultdict(lambda: defaultdict(lambda: 1e-10), {
        u:defaultdict(lambda: 1e-10, {
            w:count_w_u[u][w] / sum(count_w_u[u].values())
            for w in count_w_u[u].keys()
        })
        for u in count_w_u.keys()})
    
    p_u_w = defaultdict(lambda: defaultdict(lambda: 1e-10), {
        w:defaultdict(lambda: 1e-10, {
            u:count_u_w[w][u] / sum(count_u_w[w].values())
            for u in count_u_w[w].keys()
        })
        for w in count_u_w.keys()})
    
    return p_u, p_w_u, p_u_w


def compute_complexity_infoloss_accuracy(
    all_u, all_w, 
    p_u, 
    p_sender_w_u, p_sender_u_w, 
    p_receiver_u_w
):
    complexity = 0  # I(W,U) = sum_u_w p(u) x p(w|u) x log2( p(u|w) / p(u) ) 
    for u, w in product(all_u, all_w):
        complexity += (
            p_u[u] * 
            p_sender_w_u[u][w] * 
            log2(p_sender_u_w[w][u] / p_u[u])
        )
        
    info_loss = 0  # -sum_u_w p(u) x p(w|u) x log2(p(u|w))
    for u, w in product(all_u, all_w):
        info_loss += -p_u[u] * p_sender_w_u[u][w] * log2(p_receiver_u_w[w][u])
    
    acc = 0  # 
    for u, w in product(all_u, all_w): 
        acc += p_u[u] * p_sender_w_u[u][w] * p_receiver_u_w[w][u]
    
    return {
        'complexity': complexity,
        'info loss': info_loss,
        'accuracy': acc
    }


# ## Complexity, information loss and accuracy for Dutch

def compute_metrics_nl(natural_language_file: str):
    df = pd.read_excel(natural_language_file)

    counts = []
    all_u = set()
    all_w = set()

    # collect counts
    for _, row in df.iterrows():
        w = row['Word']
        us = row['Target group']
        if isna(us):
            counts[-1][2] += row['Count']
            continue # us = counts[-1][0] -- discard synonyms 
        else: 
            us = [u.strip() for u in us.split(',')]
        counts.append([us, row['Word'], row['Count']])
        all_u.update(set(us))
        all_w.add(w)
        

    p_u, p_w_u, p_u_w = estimate_prob_given_count(counts)
    return compute_complexity_infoloss_accuracy(
        all_u, all_w,
        p_u, 
        p_w_u, 
        p_u_w, 
        p_u_w
    ), p_u


def compute_metrics_el(path: str, need_prob, ego, thr=1e-4):
    p_u = need_prob
    all_df = pd.read_csv(path)
    eme_lang = []
    
    for epoch in range(0, 10000, N_SKIPED_EPOCHS):
        df = all_df[(all_df.Epoch == epoch) & (all_df["Ego Node"] == ego)]
        if df.shape[0] == 0:
            continue
        
        # estimate counts with weights 
        counts_sender = []
        all_u = set()
        all_w = set()
        for _, row in df.iterrows(): 
            u, w = row['Target Node'], row['Message']
            counts_sender.append(([u], w, p_u[u]))  # assume that a target has only one name
            all_u.add(u)
            all_w.add(w)
        p_sender_u, p_sender_w_u, p_sender_u_w = estimate_prob_given_count(counts_sender)

        # make sure that p_u == p_sender_u
        for u, p in p_u.items():
            assert abs(p_sender_u[u] - p) < thr, f"{[u, p, p_sender_u[u]]}"
        for u, p in p_sender_u.items():
            assert abs(p_u[u] - p) < thr, f"{[u, p, p_u[u]]}"

        p_receiver_u_w = defaultdict(lambda: defaultdict(lambda: 1e-10))
        outputs = {}
        for _, row in df.iterrows(): 
            idx, u, w = row['Target Node Idx'], row['Target Node'], row['Message']
            receiver_output = json.loads(row['Receiver Output'])
            if isinstance(receiver_output[0], list):
                receiver_output = receiver_output[0]

            if w not in outputs:
                outputs[w] = receiver_output
            else: 
                assert all(abs(o-r)<thr for o, r in zip(outputs[w], receiver_output)), \
                    [abs(o-r) for o, r in zip(outputs[w], receiver_output)]
            
            for uidx in range(len(receiver_output)):
                p_receiver_u_w[w][NODES[uidx]] = max(1e-10, receiver_output[uidx])
                        
        eme_lang.append({
                'epoch': epoch,
                'metrics': compute_complexity_infoloss_accuracy(
                    all_u, all_w,
                    p_u, 
                    p_sender_w_u, 
                    p_sender_u_w, 
                    p_receiver_u_w
                )
            }
        )
    return eme_lang


def plot_all(
    natural_language_file, 
    natural_language_name, 
    emerged_languages_files, 
    ego,
    cplx_infoloss_plot_file,
    acc_plot_file,
    run_info,
    out_path=""
):
    # compute metrics for natural language
    nl_metrics, need_prob = compute_metrics_nl(natural_language_file)

    # compute metrics for emerged languages
    el_metrics = {}
    for name, file in emerged_languages_files.items(): 
        el_metrics[name] = compute_metrics_el(file, need_prob, ego)

    fig, ax = plt.subplots()

    # draw optimal boundary
    # information loss = - complexity + entropy(u)
    p_u = need_prob
    entropy_u = sum(-p_u[u] * log2(p_u[u]) for u in p_u.keys())
    plt.plot([0, entropy_u], [entropy_u, 0], '--b')

    # plot natural language
    plt.scatter([nl_metrics['complexity']], [nl_metrics['info loss']],
                marker='D',
                color='black')

    # plot emerged languages
    def get_distinguishable_colors(n, cmap_name="tab10"):
        cmap = plt.get_cmap(cmap_name)
        return [cmap(i / n) for i in range(n)]

    colors=get_distinguishable_colors(len(emerged_languages_files))
    
    for el, color in zip(el_metrics.values(), colors):
        base_color = color[0:3]
        alphas = np.linspace(0.2, 1, len(el))
        color_range = np.column_stack(
            [np.tile(base_color, (len(el), 1)), alphas]
        )  # (N, 4) shape
        ax = plt.scatter(
            [x['metrics']['complexity'] for x in el], 
            [x['metrics']['info loss'] for x in el], 
            color=color_range
        )


    for s, el in enumerate(el_metrics.values()):
        for i in range(len(el)):
            l1, l2 = el[i-1]['metrics'], el[i]['metrics']
            if i == 1:
                #replace marker initial state
                plt.scatter(l1['complexity'], l1['info loss'],
                            marker='o', facecolors='none', edgecolors=colors[s], s=100)  # Square marker, larger size
            #add arrows
            if i > 0:
                plt.arrow(
                    l1['complexity'], l1['info loss'], 
                    l2['complexity'] - l1['complexity'],
                    l2['info loss'] - l1['info loss'],
                    shape='full', lw=0.1, length_includes_head=True, head_width=.05
                )
            if i == len(el)-1:
                #replace marker final state
                plt.scatter(l2['complexity'], l2['info loss'],
                            marker='D', facecolors='none', edgecolors=colors[s], s=100)

    plt.title(f"Ego:{ego} \n({run_info})".replace("_"," "))
    plt.xlabel('Complexity (bits)')
    plt.ylabel('Information loss (bits)')
    plt.legend(['optimal', natural_language_name] + list(el_metrics.keys()))
    plt.grid()
    plt.savefig(os.path.join(out_path,cplx_infoloss_plot_file))
    plt.close()


    # plot accuracy

    fig, ax = plt.subplots()
    # plot natural language
    plt.axhline(y=nl_metrics['accuracy'], color='b', linestyle='--', linewidth=2)

    # plot emerged language
    for el, color in zip(el_metrics.values(), colors):
        plt.plot(
            [x['epoch'] for x in el], 
            [x['metrics']['accuracy'] for x in el], 
            color=color
        )

    plt.title(f"Ego:{ego} \n({run_info})".replace("_"," "))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend([natural_language_name] + list(el_metrics.keys()))
    plt.grid()
    plt.savefig(os.path.join(out_path,acc_plot_file))
    plt.close()


def example():
    plot_all(
        natural_language_file='/home/phongle/workspace/kinship/kinship/kinship_dutch.xlsx',
        natural_language_name='dutch', 
        emerged_languages_files={
            '100-distractor': '/home/phongle/workspace/kinship/kinship/results/uniform42/evaluation.csv',
        }, 
        ego='Alice',
        cplx_infoloss_plot_file='cplx_infoloss.png',
        acc_plot_file='acc.png',
        run_info='vocab_32'
    )

    
def main():
    natural_language_file = '../../kinship_dutch.xlsx'
    natural_language_name = 'dutch'
    sweep_name="SeriousSweep20250205_170552"
    seeds=[51, 52, 53, 54, 55]
    vocab_sizes=[15,32,64,100]
    max_lens=[1,2,3]


    for v,l in itertools.product(vocab_sizes, max_lens):
        run_info=f"vocab_size_{v}_max_len_{l}"
        eval_path=f"../../results/uniform/outputs_{sweep_name}/{run_info}/"
        emerged_languages_files = { f"seed{s}":
                                    os.path.join(eval_path,f"evaluation_{run_info}_seed_{s}.csv")
                                for s in seeds }
        for ego in ('Alice', 'Bob'):
            plot_all(
                natural_language_file=natural_language_file,
                natural_language_name=natural_language_name,
                emerged_languages_files=emerged_languages_files,
                ego=ego,
                cplx_infoloss_plot_file=f'cplx_infoloss{run_info}_{ego}.png',
                acc_plot_file=f'acc{run_info}_{ego}.png',
                run_info=run_info,
                out_path=f"../../results/uniform/outputs_{sweep_name}"
            )

if __name__ == "__main__":
   # example()
    main()