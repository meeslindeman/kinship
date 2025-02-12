import os, itertools
import wandb
import fetch_runs_wandb as fw
import language_complexity as ibplot

if __name__ == '__main__':


    sweep_name = "SeriousSweep20250205_170552"
    sweep_id = "b2zly68n"
    sweep_path=f"koala-lab/kinship/{sweep_name}/{sweep_id}"
    vocab_sizes=[15,32,64,100]
    max_lens=[1,2,3]

    # 1. fetch runs from wandb
    download_dir=f"outputs_{sweep_name}"

    for v,l in itertools.product(vocab_sizes, max_lens):
        params = {"vocab_size" : v, "max_len" : l}
        artifacts = fw.fetch_artifacts_from_sweep(sweep_path, params, download_dir=download_dir)

    os.rename(download_dir, os.path.join(os.getcwd(),f"../../results/uniform/{download_dir}"))

    #2.create infoloss-complexity plots for a specific language
    natural_language_file = '../../kinship_dutch.xlsx'
    natural_language_name = 'dutch'
    seeds=[51, 52, 53, 54, 55]


    for v,l in itertools.product(vocab_sizes, max_lens):
        run_info=f"vocab_size_{v}_max_len_{l}"
        eval_path=f"../../results/uniform/outputs_{sweep_name}/{run_info}/"
        emerged_languages_files = { f"seed{s}":
                                        os.path.join(eval_path,f"evaluation_{run_info}_seed_{s}.csv")
                                    for s in seeds }
        for ego in ('Alice', 'Bob'):
            ibplot.plot_all(
                natural_language_file=natural_language_file,
                natural_language_name=natural_language_name,
                emerged_languages_files=emerged_languages_files,
                ego=ego,
                cplx_infoloss_plot_file=f'cplx_infoloss{run_info}_{ego}.png',
                acc_plot_file=f'acc{run_info}_{ego}.png',
                run_info=run_info,
                out_path=f"../../results/uniform/outputs_{sweep_name}"
            )
