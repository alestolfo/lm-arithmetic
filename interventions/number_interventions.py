from numerical_utils import convert_to_words
from interventions import Intervention
import random
from number_generator import NumberGenerator
from tqdm import tqdm

RANDOM_WORDS_BASE = ['chair', 'rabbit', 'apple', 'jump', 'large', 'window', 'bright', 'whole', 'sky', 'dance']
RANDOM_WORDS_ALT = ['drink', 'old', 'fact', 'world', 'dangerous', 'sister', 'useless']


def get_vars_and_result(eq, n_vars, number_generator, args):
    if args.n_operands == 2:
        numbers_instance = number_generator.gen_numbers(eq, n_vars, args.max_n, n_changed_vars="two", same_result=True)
    elif args.n_operands == 3:
        numbers_instance = number_generator.gen_numbers_two_step(eq, n_vars, args.max_n, same_result=True)
    vars_base, vars_alt, res = numbers_instance.values()
    res = str(res)
    vars_base = [str(i) for i in vars_base]
    vars_alt = [str(i) for i in vars_alt]
    if args.representation == 'words':
        vars_base = [convert_to_words(i) for i in vars_base]
        vars_alt = [convert_to_words(i) for i in vars_alt]
        res = convert_to_words(res)

    return vars_base, vars_alt, res


def get_vars_and_results(eq, n_vars, n_changed_vars, number_generator, args):
    if args.n_operands == 2:
        numbers_instance = number_generator.gen_numbers(eq, n_vars, args.max_n, n_changed_vars=n_changed_vars)
    elif args.n_operands == 3:
        numbers_instance = number_generator.gen_numbers_two_step(eq, n_vars, args.max_n)
    vars_base = numbers_instance['vars_base']
    vars_alt = numbers_instance['vars_alt']
    res_base = numbers_instance['res_base']
    res_alt = numbers_instance['res_alt']

    res_base = str(res_base)
    res_alt = str(res_alt)
    vars_base = [str(i) for i in vars_base]
    vars_alt = [str(i) for i in vars_alt]

    if args.representation == 'words':
        vars_base = [convert_to_words(i) for i in vars_base]
        vars_alt = [convert_to_words(i) for i in vars_alt]
        res_base = convert_to_words(res_base)
        res_alt = convert_to_words(res_alt)

    return vars_base, vars_alt, res_base, res_alt


def change_numbers_interventions(templates, tokenizer, args, control=False, n_changed_vars='one', change_result=True):
    interventions = []
    all_count = 0
    template_id = 0

    number_generator = NumberGenerator(args.max_n, args.seed, args.n_operands)

    for t_dict in tqdm(templates, desc='generating interventions'):
        template_id += 1

        number_generator.reset_triple_lists()

        for _ in range(args.examples_per_template):
            all_count += 1
            n_vars = t_dict['n_vars']
            eq = t_dict['eq']
            second_part = 'statement' if 'statement' in args.prompt else 'question'

            try:
                if change_result:
                    vars_base, vars_alt, res_base, res_alt = get_vars_and_results(eq, n_vars, n_changed_vars, number_generator, args)
                else:
                    vars_base, vars_alt, res = get_vars_and_result(eq, n_vars, number_generator, args)
            except Exception as e:
                if str(e) == 'Ran out of number triples':
                    print('Ran out of number triples but continuing')
                    break
                else:
                    raise e

            temp1 = (t_dict['body'] + ' ' + t_dict[second_part])
            temp2 = (t_dict['body'] + ' ' + t_dict[second_part])

            if control:
                [rand_word_base] = random.sample(RANDOM_WORDS_BASE, 1)
                idx_to_switch = int(eq.split(' ')[1].replace('number', ''))
                vars_base[idx_to_switch] = rand_word_base
                [rand_word_alt] = random.sample(RANDOM_WORDS_ALT, 1)
                vars_alt[idx_to_switch] = rand_word_alt

            for i in range(n_vars):
                var_name = 'number' + str(i)
                temp1 = temp1.replace(var_name, vars_base[i])
                temp2 = temp2.replace(var_name, vars_alt[i])

            intervention = Intervention(
                tokenizer,
                str(t_dict['id']),
                temp1,
                temp2,
                n_vars=n_vars,
                equation=eq,
                vars_base=vars_base,
                vars_alt=vars_alt,
                multitoken=(args.representation == 'words'),
                device=args.device)

            if change_result:
                intervention.set_results(res_base, res_alt)
            else:
                intervention.set_result(res)

            interventions.append(intervention)

    print(f"\t Number of templates used: {template_id}")
    print(f"\t Total interventions generated: {all_count}")
    random.shuffle(interventions)
    return interventions
