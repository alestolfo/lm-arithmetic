import os
import hydra
import json
import random
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer, BloomTokenizerFast, GPTNeoXTokenizerFast, LlamaTokenizer
from intervention_models.intervention_model import load_model
from interventions.intervention import get_data


@hydra.main(config_path='conf', config_name='config')
def run_experiment(args: DictConfig):
    print(OmegaConf.to_yaml(args))
    print("Model:", args.model)

    print('args.intervention_type', args.intervention_type)

    if 'llama_models_hf/7B' in args.model:
        model_str = 'llama7B'
    elif 'llama_models_hf/13B' in args.model:
        model_str = 'llama13B'
    elif 'llama_models_hf/30B' in args.model:
        model_str = 'llama30B'
    elif 'alpaca' in args.model:
        model_str = 'alpaca'
    else:
        model_str = args.model

    # initialize logging
    log_directory = args.output_dir
    log_directory += f'/{model_str}'
    if args.model_ckpt:
        ckpt_name = '_'.join(args.model_ckpt.split('/')[5:9])
        log_directory += f'_from_ckpt_{ckpt_name}'
    log_directory += f'/n_operands{args.n_operands}'
    log_directory += f'/template_type{args.template_type}'
    log_directory += f'/max_n{args.max_n}'
    log_directory += f'/n_shots{args.n_shots}'
    log_directory += f'/examples_n{args.examples_per_template}'
    log_directory += f'/seed{args.seed}'
    print(f'log_directory: {log_directory}')
    os.makedirs(log_directory, exist_ok=True)
    wandb_name = ('random-' if args.random_weights else '')
    wandb_name += f'{model_str}'
    wandb_name += f' -p {args.template_type}'
    wandb.init(project='mathCMA', name=wandb_name, notes='', dir=log_directory,
               settings=wandb.Settings(start_method='fork'), mode=args.wandb_mode)
    args_to_log = dict(args)
    args_to_log['out_dir'] = log_directory
    print("\n" + json.dumps(str(args_to_log), indent=4) + "\n")
    wandb.config.update(args_to_log)
    del args_to_log

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize Model and Tokenizer
    model = load_model(args)
    tokenizer_class = (GPT2Tokenizer if model.is_gpt2 or model.is_gptneo or model.is_opt else
                       BertTokenizer if model.is_bert else
                       AutoTokenizer if model.is_gptj or model.is_flan or model.is_pythia else
                       BloomTokenizerFast if model.is_bloom else
                       GPTNeoXTokenizerFast if model.is_neox else
                       LlamaTokenizer if model.is_llama else
                       None)
    if not tokenizer_class:
        raise Exception(f'Tokenizer for model {args.model} not found')

    if 'goat' in args.model:
        tokenizer_id = 'decapoda-research/llama-7b-hf'
    else:
        tokenizer_id = args.model

    tokenizer = tokenizer_class.from_pretrained(tokenizer_id, cache_dir=args.transformers_cache_dir)
    model.create_vocab_subset(tokenizer, args)

    intervention_list = get_data(tokenizer, args)

    if args.debug_run:
        intervention_list = intervention_list[:2]

    print('================== INTERVENTIONS ==================')
    for intervention in intervention_list[:5]:
        print(f'BASE: {intervention.base_string} {intervention.res_base_string}')
        print(f'ALT: {intervention.alt_string} {intervention.res_alt_string}')

    if args.intervention_loc.startswith('attention_'):
        attention_int_loc = '_'.join(args.intervention_loc.split('_')[1:])
        results = model.attention_experiment(interventions=intervention_list,
                                             effect_type=args.effect_type,
                                             intervention_loc=attention_int_loc,
                                             get_full_distribution=args.get_full_distribution,
                                             all_tokens=args.all_tokens)
    else:
        results = model.intervention_experiment(interventions=intervention_list,
                                                effect_type=args.effect_type,
                                                intervention_loc=args.intervention_loc,
                                                get_full_distribution=args.get_full_distribution,
                                                all_tokens=args.all_tokens)

    df_results = model.process_intervention_results(intervention_list, model.word_subset, results, args)

    random_w = 'random_' if args.random_weights else ''
    f_name: str = f'{random_w}intervention_{args.intervention_type}'
    f_name += f'_{args.representation}'
    f_name += f'_{args.effect_type}'
    f_name += f'_{args.intervention_loc}'
    f_name += '_all_tokens' if args.all_tokens else ''
    f_name += '_int8' if args.int8 else ''
    out_path = os.path.join(log_directory, f_name + ".feather")
    print('out_path: ', out_path)
    df_results.to_feather(out_path)


if __name__ == "__main__":
    run_experiment()
