import json

from interventions import Intervention
import random
from tqdm import tqdm
from utils.number_utils import convert_to_words, check_same_length


def generate_instance(template, args, entity_dict, tokenizer, get_entities=False):
    attention_intervention = args.intervention_loc.startswith('attention_')
    x, y = random.sample(range(args.max_n + 1), 2)
    x_plural = 0 if x == 1 else 1
    y_plural = 0 if y == 1 else 1
    class_base, class_alt = random.sample(entity_dict.keys(), 2)
    e1 = random.sample(entity_dict[class_base], 1)[0][x_plural]
    e2 = random.sample(entity_dict[class_alt], 1)[0][y_plural]
    if args.intervention_type == 11 and not check_same_length('a ' + e1, 'a ' + e2, tokenizer):
        while 1:
            class_base, class_alt = random.sample(entity_dict.keys(), 2)
            e1 = random.sample(entity_dict[class_base], 1)[0][x_plural]
            e2 = random.sample(entity_dict[class_alt], 1)[0][y_plural]
            if check_same_length('a ' + e1, 'a ' + e2, tokenizer):
                break
    x = str(x)
    y = str(y)

    if args.representation == 'words':
        x = convert_to_words(x)
        y = convert_to_words(y)

    t_string = template.replace('{x}', x).replace('{y}', y).replace('{entity1}', e1).replace('{entity2}', e2)
    if args.intervention_type == 10:
        base_string = t_string.replace('{class}', class_base)
        alt_string = t_string.replace('{class}', class_alt)
    elif args.intervention_type == 11:
        base_string = t_string.replace('{class}', e1)
        alt_string = t_string.replace('{class}', e2)
    else:
        raise Exception(f'Invalid intervention type: {args.intervention_type}')
    res_base = x
    res_alt = y

    ret = base_string, alt_string, res_base, res_alt
    if get_entities:
        ret += (e1, e2)
    return ret


def get_entity_data(tokenizer, args):
    templates = ['Q: I have {x} {entity1} and {y} {entity2}. How many {class} do I have? A:',
                 'Q: Paul has {x} {entity1} and {y} {entity2}. How many {class} does Paul have? A:',
                 'Q: I can see {x} {entity1} and {y} {entity2}. How many {class} are there? A:',
                 'Q: I have {x} {entity1} and {y} {entity2}. How many {class} are there? A:',
                 'Q: Alice received {x} {entity1} and {y} {entity2}. How many {class} did she receive? A:',
                 'Q: Mark bought {x} {entity1} and {y} {entity2}. How many {class} did he buy? A:',
                 'Q: Jasmin sold {x} {entity1} and {y} {entity2}. How many {class} did she sell? A:',
                 'Q: Sarah owns {x} {entity1} and {y} {entity2}. How many {class} does she own? A:',
                 ]

    with open(args.path_to_entity_dict, 'r') as fp:
        entity_dict = json.load(fp)

    intervention_list = []

    progress = tqdm(total=len(templates) * args.examples_per_template)
    for t_id, t in enumerate(templates):
        few_shots = ''
        for _ in range(args.n_shots):
            base_string, _, res_base, _ = generate_instance(t, args, entity_dict, tokenizer)
            shot = f'{base_string} {res_base}\n'
            few_shots += shot

        for _ in range(args.examples_per_template):
            base_string, alt_string, res_base, res_alt, e1, e2 = generate_instance(t, args, entity_dict, tokenizer, get_entities=True)

            intervention = Intervention(tokenizer,
                                        template_type=t_id,
                                        base_string=base_string,
                                        alt_string=alt_string,
                                        equation='x or y',
                                        few_shots=few_shots,
                                        n_vars=2)
            intervention.set_results(res_base, res_alt)
            intervention.set_position_of_tokens_int11(e1, e2)
            intervention_list.append(intervention)

            progress.update()

    return intervention_list
