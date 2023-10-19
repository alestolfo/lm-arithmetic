import json
import os
from interventions import Intervention
import random

from utils.number_utils import check_same_length

RELS_SUBSET = ['P36', 'P1376', 'P279', 'P19', 'P103', 'P20']


def load_files(dirname, excluded_files=[]):
    data = {}
    for filename in os.listdir(dirname):
        rel_data = []
        if filename in excluded_files:
            continue
        with open(dirname + '/' + filename, "r") as f:
            for line in f.readlines():
                rel_data.append(json.loads(line))
        data[filename.replace('.jsonl', '')] = rel_data
    return data


def get_relations_dict(lama_path):
    rels = {}
    with open(lama_path + '/relations.jsonl') as fp:
        for line in fp:
            d = json.loads(line)
            rels[d['relation']] = d
    return rels


def get_lama_vocab_subset(tokenizer, args):
    vocab = {}
    rel_to_data_dict = load_files(args.lama_path + '/TREx')

    for rel_id, ex_list in rel_to_data_dict.items():
        if rel_id not in RELS_SUBSET:
            continue
        for ex in ex_list:
            tok_label = tokenizer.tokenize('a ' + ex['obj_label'])[1:]
            if len(tok_label) > 1:
                continue
            tok_label = tokenizer.convert_tokens_to_ids(tok_label)
            vocab[tok_label[0]] = ex['obj_label']

    return vocab


def get_lama_data(tokenizer, args):

    relations_dict = get_relations_dict(args.lama_path)
    rel_to_data_dict = load_files(args.lama_path + '/TREx')

    attention_intervention = args.intervention_loc.startswith('attention_')

    intervention_list = []
    for rel_id, ex_list in rel_to_data_dict.items():
        if rel_id not in RELS_SUBSET:
            continue
        template = relations_dict[rel_id]['template']
        template = template.split('[Y]')[0].strip()
        no_space_before_sub = template.startswith('[X]')

        few_shots = ''
        for _ in range(args.n_shots):
            ex = random.sample(ex_list, 1)[0]
            sub1 = ex['sub_label']
            string = template.replace('[X]', sub1)
            obj = ex['obj_label']
            shot = f'{string} {obj}\n'
            few_shots += shot

        for _ in range(args.examples_per_template):
            while 1:
                ex1, ex2 = random.sample(ex_list, 2)
                obj1 = ex1['obj_label']
                obj2 = ex2['obj_label']

                tok_obj1 = tokenizer.tokenize('a ' + obj1)
                tok_obj2 = tokenizer.tokenize('a ' + obj2)
                if len(tok_obj1) > 2 or len(tok_obj2) > 2:
                    continue

                sub1 = ex1['sub_label']
                sub2 = ex2['sub_label']
                prefix = '\n' if no_space_before_sub else 'a '
                tok_sub1 = tokenizer.tokenize(prefix + sub1)
                tok_sub2 = tokenizer.tokenize(prefix + sub2)
                if len(tok_sub1) != len(tok_sub2):
                    continue

                base_string = template.replace('[X]', sub1)
                alt_string = template.replace('[X]', sub2)

                if obj1 != obj2 and (not attention_intervention or check_same_length(base_string, alt_string, tokenizer)):
                    break

            if attention_intervention:
                assert check_same_length(base_string, alt_string, tokenizer), f'{base_string} {alt_string}'

            intervention = Intervention(tokenizer,
                                        template_type=args.template_type,
                                        base_string=base_string,
                                        alt_string=alt_string,
                                        equation=f'LAMA {rel_id}',
                                        few_shots='',
                                        n_vars=0)
            intervention.set_results(obj1, obj2)
            intervention.set_position_of_tokens_lama(sub1, sub2, no_space_before_sub=no_space_before_sub)

            intervention_list.append(intervention)

    return intervention_list

