from interventions import Intervention
import json
import random
from tqdm import tqdm
from utils.number_utils import is_int, convert_to_words
from interventions.entity_intervention import get_entity_data
from interventions.lama_intervention import get_lama_data

INTERVENTION_TYPES_TWO_RESULTS = [1, 3, 10, 11]
INTERVENTION_TYPES_SINGLE_RESULT = [2, 20]
INTERVENTION_TYPES = INTERVENTION_TYPES_TWO_RESULTS + INTERVENTION_TYPES_SINGLE_RESULT


def get_data(tokenizer, args):
    if args.intervention_type in [10, 11]:
        return get_entity_data(tokenizer, args)
    elif args.intervention_type == 20:
        return get_lama_data(tokenizer, args)
    elif args.intervention_type in [1, 2] and args.n_operands == 2:
        return get_arithmetic_data_two_operands(tokenizer, args)
    elif args.intervention_type in [1, 2] and args.n_operands == 3:
        return get_arithmetic_data_three_operands(tokenizer, args)
    elif args.intervention_type == 3:
        return get_arithmetic_data_two_operations(tokenizer, args)
    else:
        raise Exception(f'Intervention type not recognized {args.intervention_type}')


def generate_operands_pair(args, op, keep_result):
    llama_setting = False
    if args.max_n == 9:
        llama_setting = True
        op_max = 300
        res_max = 9
    else:
        op_max = args.max_n
        res_max = args.max_n
    while 1:
        x_base = str(random.randint(1, op_max))
        y_base = str(random.randint(1, op_max))
        res_base = eval(f'{x_base} {op} {y_base}')
        if is_int(res_base) and res_base in range(1, res_max):
            break
    res_base = str(int(res_base))

    num_retries = 0
    while 1:
        x_alt = str(random.randint(1, op_max))
        y_alt = str(random.randint(1, op_max))

        res_alt = eval(f'{x_alt} {op} {y_alt}')
        if is_int(res_alt) and res_alt in range(1, res_max):
            if llama_setting:
                if len(str(x_base)) != len(str(x_alt)) or len(str(y_base)) != len(str(y_alt)):
                    continue
                if not keep_result and int(res_alt) == int(res_base):
                    continue

            if not keep_result:
                break
            elif int(res_alt) == int(res_base):
                break
            else:
                num_retries += 1
            if num_retries > 100000:
                raise Exception(
                    f'Could not find a pair of operands with the same result: {x_base} {op} {y_base} = {res_base}')

    res_alt = str(int(res_alt))

    return [(x_base, y_base, res_base), (x_alt, y_alt, res_alt)]


def generate_operands_pair_two_ops(args, op1, op2):
    llama_setting = False
    if args.max_n == 9:
        llama_setting = True
        op_max = 300
        res_max = 9
    else:
        op_max = args.max_n
        res_max = args.max_n

    num_retries = 0
    while 1:
        x = str(random.randint(1, op_max))
        y = str(random.randint(1, op_max))
        res_base = eval(f'{x} {op1} {y}')
        res_alt = eval(f'{x} {op2} {y}')

        if is_int(res_alt) and res_alt in range(1, res_max) and is_int(res_base) and res_base in range(1, res_max):
            break
        num_retries += 1
        if num_retries > 100000:
            raise Exception(
                f'Could not find a pair of operands: {op1} {op2} ')

    res_alt = str(int(res_alt))
    res_base = str(int(res_base))

    return x, y, res_base, res_alt


def generate_operands_triple(args, equation, keep_result):
    llama_setting = False
    if args.max_n == 9:
        llama_setting = True
        op_max = 50
        res_max = 9
    else:
        op_max = args.max_n
        res_max = args.max_n

    num_retries = 0
    while 1:
        x_base = str(random.randint(1, op_max))
        y_base = str(random.randint(1, op_max))
        z_base = str(random.randint(1, op_max))
        try:
            res_base = eval(equation.replace('{x}', x_base).replace('{y}', y_base).replace('{z}', z_base))
        except ZeroDivisionError:
            continue
        if not is_int(res_base) or not res_base in range(1, res_max):
            continue

        res_base = str(int(res_base))
        res_alt = None

        while 1:
            x_alt = str(random.randint(1, op_max))
            y_alt = str(random.randint(1, op_max))
            z_alt = str(random.randint(1, op_max))
            num_retries += 1

            try:
                res_alt = eval(equation.replace('{x}', x_alt).replace('{y}', y_alt).replace('{z}', z_alt))
            except ZeroDivisionError:
                continue
            if is_int(res_alt) and res_alt in range(1, res_max):
                if llama_setting:
                    if len(str(x_base)) != len(str(x_alt)) or len(str(y_base)) != len(str(y_alt)) or len(
                            str(z_base)) != len(str(z_alt)):
                        continue
                    if not keep_result and int(res_alt) == int(res_base):
                        continue

                if not keep_result:
                    break
                elif int(res_alt) == int(res_base):
                    break

                if num_retries % 10000 == 0:
                    break
                if num_retries > 1000000000:
                    raise Exception(
                        f'Could not find a pair of operands with the same result:\
                        {equation.replace("{x}", x_base).replace("{y}", y_base).replace("{z}", z_base)} = {res_base}')

        if res_alt is not None:
            break

    res_alt = str(int(res_alt))

    return [(x_base, y_base, z_base, res_base), (x_alt, y_alt, z_alt, res_alt)]


def get_arithmetic_data_three_operands(tokenizer, args):
    with open('interventions/three_operand_questions.json') as fp:
        three_operand_questions = json.load(fp)

    keep_result = args.intervention_type in INTERVENTION_TYPES_SINGLE_RESULT

    intervention_list = []
    progress = tqdm(total=len(three_operand_questions) * args.examples_per_template)

    for equation, template in three_operand_questions.items():
        few_shots = ''
        if args.n_shots > 0:
            for _ in range(args.n_shots):
                base_tuple, alt_tuple = generate_operands_triple(args, equation, keep_result)
                x_base, y_base, z_base, res_base = base_tuple
                shot = template.replace('{x}', x_base).replace('{y}', y_base).replace('{z}', z_base) + f' {res_base}\n'
                few_shots += shot

        for _ in range(args.examples_per_template):
            base_tuple, alt_tuple = generate_operands_triple(args, equation, keep_result)
            x_base, y_base, z_base, res_base = base_tuple
            x_alt, y_alt, z_alt, res_alt = alt_tuple

            if args.representation == 'words':
                x_base = convert_to_words(x_base)
                y_base = convert_to_words(y_base)
                z_base = convert_to_words(z_base)
                x_alt = convert_to_words(x_alt)
                y_alt = convert_to_words(y_alt)
                z_alt = convert_to_words(z_alt)
                res_base = convert_to_words(res_base)
                res_alt = convert_to_words(res_alt)

            base_string = template.replace('{x}', x_base).replace('{y}', y_base).replace('{z}', z_base)
            alt_string = template.replace('{x}', x_alt).replace('{y}', y_alt).replace('{z}', z_alt)

            intervention = Intervention(tokenizer,
                                        template_type='-',
                                        base_string=base_string,
                                        alt_string=alt_string,
                                        equation=equation,
                                        few_shots=few_shots,
                                        n_vars=2)
            intervention.set_results(res_base, res_alt)
            intervention.set_position_of_tokens_three_operands((x_base, y_base, z_base), (x_alt, y_alt, z_alt))
            intervention_list.append(intervention)

            progress.update()

    return intervention_list


def get_arithmetic_data_two_operands(tokenizer, args):
    addition_templates = ['Q: How much is {x} plus {y}? A:',
                          'Q: What is {x} plus {y}? A:',
                          'Q: What is the result of {x} plus {y}? A:',
                          'Q: What is the sum of {x} and {y}? A:',
                          'The sum of {x} and {y} is',
                          '{x} + {y} =']
    subtraction_templates = ['Q: How much is {x} minus {y}? A:',
                             'Q: What is {x} minus {y}? A:',
                             'Q: What is the result of {x} minus {y}? A:',
                             'Q: What is the difference between {x} and {y}? A:',
                             'The difference between {x} and {y} is',
                             '{x} - {y} =']
    multiplication_templates = ['Q: How much is {x} times {y}? A:',
                                'Q: What is {x} times {y}? A:',
                                'Q: What is the result of {x} times {y}? A:',
                                'Q: What is the product of {x} and {y}? A:',
                                'The product of {x} and {y} is',
                                '{x} * {y} =']
    division_templates = ['Q: How much is {x} over {y}? A:',
                          'Q: What is {x} over {y}? A:',
                          'Q: What is the result of {x} over {y}? A:',
                          'Q: What is the ratio between {x} and {y}? A:',
                          'The ratio of {x} and {y} is',
                          '{x} / {y} =']
    operator_word_indices = [5, 4, 7, 4, 1, 1]

    keep_result = args.intervention_type in INTERVENTION_TYPES_SINGLE_RESULT

    ops = ['+', '-', '*', '/']
    all_templates = [addition_templates, subtraction_templates, multiplication_templates, division_templates]
    if args.representation == 'words':
        all_templates = [tl[:-1] for tl in all_templates]
    elif args.template_type != 'all':
        all_templates = [[tl[args.template_type]] for tl in all_templates]
    print(f'Using templates: {all_templates}')

    template_dict = {op: ts for op, ts in zip(ops, all_templates)}

    intervention_list = []

    progress = tqdm(total=len(all_templates[0]) * len(all_templates) * args.examples_per_template)
    for op in template_dict:
        for t_id, t in enumerate(template_dict[op]):
            few_shots = ''
            if args.n_shots > 0:
                for _ in range(args.n_shots):
                    base_tuple, alt_tuple = generate_operands_pair(args, op, keep_result)
                    x_base, y_base, res_base = base_tuple
                    shot = t.replace('{x}', x_base).replace('{y}', y_base) + f' {res_base}\n'
                    few_shots += shot

            for _ in range(args.examples_per_template):
                base_tuple, alt_tuple = generate_operands_pair(args, op, keep_result)
                x_base, y_base, res_base = base_tuple
                x_alt, y_alt, res_alt = alt_tuple

                if args.representation == 'words':
                    x_base = convert_to_words(x_base)
                    y_base = convert_to_words(y_base)
                    x_alt = convert_to_words(x_alt)
                    y_alt = convert_to_words(y_alt)
                    res_base = convert_to_words(res_base)
                    res_alt = convert_to_words(res_alt)

                base_string = t.replace('{x}', x_base).replace('{y}', y_base)
                alt_string = t.replace('{x}', x_alt).replace('{y}', y_alt)

                intervention = Intervention(tokenizer,
                                            template_type=t_id,
                                            base_string=base_string,
                                            alt_string=alt_string,
                                            equation=f'x {op} y',
                                            few_shots=few_shots,
                                            n_vars=2)
                intervention.set_results(res_base, res_alt)
                operator_word_idx = operator_word_indices[t_id]
                operator_word = t.split(' ')[operator_word_idx]
                intervention.set_position_of_tokens((x_base, y_base), (x_alt, y_alt), operator_word,
                                                    no_space_before_op1=t_id == 5)

                intervention_list.append(intervention)

                progress.update()

    return intervention_list


def get_arithmetic_data_two_operations(tokenizer, args):
    addition_templates = ['Q: How much is {x} plus {y}? A:',
                          'Q: What is {x} plus {y}? A:',
                          'Q: What is the result of {x} plus {y}? A:',
                          'Q: What is the sum of {x} and {y}? A:',
                          'The sum of {x} and {y} is',
                          '{x} + {y} =']
    subtraction_templates = ['Q: How much is {x} minus {y}? A:',
                             'Q: What is {x} minus {y}? A:',
                             'Q: What is the result of {x} minus {y}? A:',
                             'Q: What is the difference between {x} and {y}? A:',
                             'The difference between {x} and {y} is',
                             '{x} - {y} =']
    multiplication_templates = ['Q: How much is {x} times {y}? A:',
                                'Q: What is {x} times {y}? A:',
                                'Q: What is the result of {x} times {y}? A:',
                                'Q: What is the product of {x} and {y}? A:',
                                'The product of {x} and {y} is',
                                '{x} * {y} =']
    division_templates = ['Q: How much is {x} over {y}? A:',
                          'Q: What is {x} over {y}? A:',
                          'Q: What is the result of {x} over {y}? A:',
                          'Q: What is the ratio between {x} and {y}? A:',
                          'The ratio of {x} and {y} is',
                          '{x} / {y} =']
    operator_word_indices = [5, 4, 7, 4, 1, 1]

    keep_result = args.intervention_type in INTERVENTION_TYPES_SINGLE_RESULT

    ops = ['+', '-', '*', '/']
    all_templates = [addition_templates, subtraction_templates, multiplication_templates, division_templates]
    if args.representation == 'words':
        all_templates = [tl[:-1] for tl in all_templates]
    elif args.template_type != 'all':
        all_templates = [[tl[args.template_type]] for tl in all_templates]
    print(f'Using templates: {all_templates}')

    template_dict = {op: ts for op, ts in zip(ops, all_templates)}

    intervention_list = []

    progress = tqdm(total=len(all_templates[0]) * len(all_templates) * args.examples_per_template)
    for op in template_dict:
        for t_id, t in enumerate(template_dict[op]):
            for _ in range(args.examples_per_template):
                # sample a different op
                op2 = random.choice([o for o in ops if o != op])

                t2 = template_dict[op2][t_id]

                few_shots_t1 = ''
                few_shots_t2 = ''
                if args.n_shots > 0:
                    for _ in range(args.n_shots):
                        base_tuple, _ = generate_operands_pair(args, op, keep_result)
                        base_tuple2, _ = generate_operands_pair(args, op2, keep_result)
                        x_base, y_base, res_base = base_tuple
                        x_base2, y_base2, res_base2 = base_tuple
                        shot = t.replace('{x}', x_base).replace('{y}', y_base) + f' {res_base}\n'
                        shot2 = t2.replace('{x}', x_base2).replace('{y}', y_base2) + f' {res_base2}\n'
                        few_shots_t1 += shot
                        few_shots_t2 += shot2

                x, y, res_base, res_alt = generate_operands_pair_two_ops(args, op, op2)

                if args.representation == 'words':
                    x = convert_to_words(x)
                    y = convert_to_words(y)
                    res_base = convert_to_words(res_base)
                    res_alt = convert_to_words(res_alt)

                base_string = t.replace('{x}', x).replace('{y}', y)
                alt_string = t2.replace('{x}', x).replace('{y}', y)

                intervention = Intervention(tokenizer,
                                            template_type=t_id,
                                            base_string=base_string,
                                            alt_string=alt_string,
                                            equation=f'x {op} y - x {op2} y',
                                            few_shots=few_shots_t1,
                                            few_shots_t2=few_shots_t2,
                                            n_vars=2)
                intervention.set_results(res_base, res_alt)
                operator_word_idx = operator_word_indices[t_id]
                operator_word = t.split(' ')[operator_word_idx]
                intervention.set_position_of_tokens((x, y), (x, y), operator_word,
                                                    no_space_before_op1=t_id == 5)

                intervention_list.append(intervention)

                progress.update()

    return intervention_list
