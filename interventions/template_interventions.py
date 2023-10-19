import random
from numerical_utils import convert_to_words
from interventions import Intervention
from number_generator import NumberGenerator
from tqdm import tqdm

RANDOM_SENTENCES = ['the sky is blue today .', 'america is great .', 'the sea is deep .', 'everybody likes ice cream .', 'everybody likes pizza .', 'lionel messi plays for psg .']
#RANDOM_SENTENCES_WITH_NUMBER = ['there are {} birds in the sky .', 'the sea is {} feet deep .', 'lebron james scored {} points last night .', 'that cars is {} years old .', 'lionel messi wears the shirt number {} .']
#RANDOM_SENTENCES_WITHOUT_NUMBER = ['there are some birds in the sky .', 'the sea is many feet deep .', 'lebron james scored several points last night .', 'that cars is multiple years old .', 'lionel messi wears the shirt  number of maradona .']
RANDOM_WORDS = [ 'the', 'hello', 'car', 'sky', 'red', 'large', 'more', 'attention', 'help', 'take', 'average', 'dog' ]

RANDOM_SENTENCES_WITH_NUMBER = ['I have {} brothers and sisters .',
                                'I live at {} Elm Street .',
                                'there are {} types of bears .',
                                'the average car gets about {} miles per gallon of gasoline .',
                                'the average car can reach speeds of about {} miles per hour .',
                                'the los angeles lakers have won {} NBA championships .',
                                'I have {} cats .',
                                'there are {} birds in the sky .'
                                ]

RANDOM_SENTENCES_WITHOUT_NUMBER = ['I have no brothers and sisters .',
                                'I live at Elm Street .',
                                'there are multiple types of bears .',
                                'the average car gets about few miles per gallon of gasoline .',
                                'the average car can reach speeds of about several miles per hour .',
                                'the los angeles lakers have won some NBA championships .',
                                'I have some cats .',
                                'there are some birds in the sky .'
                                ]


def build_strings_and_add_intervention(temp1, temp2, vars, tokenizer, n_vars, eq, t1_id, t2_id, args):
    for i in range(n_vars):
        var_name = 'number' + str(i)
        temp1 = temp1.replace(var_name, vars[i])
        temp2 = temp2.replace(var_name, vars[i])

    id = str(t1_id) + '-' + str(t2_id) if t2_id else str(t1_id)

    intervention = Intervention(
        tokenizer=tokenizer,
        template_id=id,
        base_temp=temp1,
        alt_temp=temp2,
        n_vars=n_vars,
        equation=eq,
        vars_base=vars,
        vars_alt=vars,
        multitoken=(args.representation == 'words'),
        device=args.device)

    return intervention


def add_tokens_same_result_interventions(templates, tokenizer, args, add_number=False, tokens_n=1):
    interventions = []

    all_count = 0
    used_count = 0
    template_id = 0

    random.seed(args.seed)

    number_generator = NumberGenerator(args.max_n, args.seed)

    for t_dict in tqdm(templates, desc='generating interventions'):
        template_id += 1

        number_generator.reset_triple_lists()

        for _ in range(args.examples_per_template):
            all_count += 1
            n_vars = t_dict['n_vars']
            eq = t_dict['eq']
            second_part = 'statement' if 'statement' in args.prompt else 'question'
            numbers_instance = number_generator.gen_numbers(eq, n_vars, args.max_n)
            vars = numbers_instance['vars_base']
            res = numbers_instance['res_base']

            res = str(res)
            vars = [str(i) for i in vars]

            if args.representation == 'words':
                raise Exception('representation words nor supported yet')

            if add_number:
                #random_numbers = [ str(random.randint(1, args.max_n)) for _ in range(tokens_n) ]
                random_numbers = ['100' for _ in range(tokens_n)]
                random_sentence = ' '.join(random_numbers)
            else:
                random_words = random.choices(RANDOM_WORDS, k=tokens_n)
                random_sentence = ' '.join(random_words)

            temp1 = (t_dict['body'] + ' ' + t_dict[second_part])

            if t_dict['body'][-1] != '.':
                temp2 = (random_sentence + ' ' + t_dict['body'] + ' ' + t_dict[second_part])
            else:
                random_sentence = ' ' + random_sentence
                temp2 = (t_dict['body'] + random_sentence + ' ' + t_dict[second_part])

            intervention = build_strings_and_add_intervention(temp1, temp2, vars, tokenizer, n_vars, eq,
                                                              t_dict['id'], None, args)
            intervention.set_result(res)
            interventions.append(intervention)
            used_count += 1

    print(f"\t Number of templates used: {template_id}")
    print(f"\t Only used {used_count}/{all_count} examples due to tokenizer")
    random.shuffle(interventions)
    return interventions


def add_sentence_same_result_interventions(templates, tokenizer, args, add_number=False, add_number_same_sentence=False):
    interventions = []

    all_count = 0
    used_count = 0
    template_id = 0

    number_generator = NumberGenerator(args.max_n, args.seed)

    for t_dict in tqdm(templates, desc='generating interventions'):
        template_id += 1

        number_generator.reset_triple_lists()

        for _ in range(args.examples_per_template):
            all_count += 1
            n_vars = t_dict['n_vars']
            eq = t_dict['eq']
            second_part = 'statement' if 'statement' in args.prompt else 'question'
            numbers_instance = number_generator.gen_numbers(eq, n_vars, args.max_n)
            vars = numbers_instance['vars_base']
            res = numbers_instance['res_base']

            res = str(res)
            vars = [str(i) for i in vars]
            rand_number = str(random.randint(1, args.max_n))
            while rand_number == res:
                rand_number = str(random.randint(1, args.max_n))

            if args.representation == 'words':
                vars = [convert_to_words(i) for i in vars]
                res = convert_to_words(res)
                rand_number = convert_to_words(rand_number)

            if add_number_same_sentence:
                zipped = list(zip(RANDOM_SENTENCES_WITH_NUMBER, RANDOM_SENTENCES_WITHOUT_NUMBER))
                [(random_sentence_w_num, random_sentence_wo_num)] = random.sample(zipped, 1)
                random_sentence_w_num = random_sentence_w_num.format(rand_number)

                if t_dict['body'][-1] != '.':
                    temp1 = (random_sentence_wo_num + ' ' + t_dict['body'] + ' ' + t_dict[second_part])
                    temp2 = (random_sentence_w_num + ' ' + t_dict['body'] + ' ' + t_dict[second_part])
                else:
                    random_sentence_wo_num = ' ' + random_sentence_wo_num
                    random_sentence_w_num = ' ' + random_sentence_w_num
                    temp1 = (t_dict['body'] + random_sentence_wo_num + ' ' + t_dict[second_part])
                    temp2 = (t_dict['body'] + random_sentence_w_num + ' ' + t_dict[second_part])
            else:
                if add_number:
                    [random_sentence] = random.sample(RANDOM_SENTENCES_WITH_NUMBER, 1)
                    random_sentence = random_sentence.format(rand_number)
                else:
                    [random_sentence] = random.sample(RANDOM_SENTENCES_WITHOUT_NUMBER, 1)

                temp1 = (t_dict['body'] + ' ' + t_dict[second_part])

                if t_dict['body'][-1] != '.':
                    temp2 = (random_sentence + ' ' + t_dict['body'] + ' ' + t_dict[second_part])
                else:
                    random_sentence = ' ' + random_sentence
                    temp2 = (t_dict['body'] + random_sentence + ' ' + t_dict[second_part])

            intervention = build_strings_and_add_intervention(temp1, temp2, vars, tokenizer, n_vars, eq,
                                                              t_dict['id'], None, args)
            intervention.set_result(res)
            if add_number_same_sentence or add_number:
                intervention.set_irrelevant_number(rand_number)
            interventions.append(intervention)
            used_count += 1

    print(f"\t Number of templates used: {template_id}")
    print(f"\t Only used {used_count}/{all_count} examples due to tokenizer")
    random.shuffle(interventions)
    return interventions


def change_template_different_result_interventions(templates, tokenizer, args):
    # group templates by number of variables
    vars_dict = {}
    for t_dict in templates:
        n_vars = t_dict['n_vars']
        if n_vars not in vars_dict:
            vars_dict[n_vars] = [t_dict]
        else:
            vars_dict[n_vars].append(t_dict)

    number_generator = NumberGenerator(args.max_n, args.seed)

    all_count = 0
    used_count = 0
    interventions = []
    for _ in tqdm(range(args.examples_per_template), desc='generating interventions'):
        number_generator.reset_triple_lists()

        for n_vars, list_templates in vars_dict.items():
            list_t = list_templates.copy()

            while len(list_t) >= 2:
                all_count += 1

                t1_dict = list_t.pop(0)
                t2_dict = None
                random.shuffle(list_t)
                for i in range(len(list_t)):
                    possible_t2 = list_t[i]
                    if t1_dict['eq'] != possible_t2['eq'] and t1_dict['eq'][1:] == possible_t2['eq'][1:]:
                        t2_dict = list_t.pop(i)
                        break
                if not t2_dict:
                    continue

                if args.n_operands == 2:
                    numbers_instance = number_generator.gen_numbers_different_result(t1_dict['eq'], t2_dict['eq'], n_vars, args.max_n)
                elif args.n_operands == 3:
                    numbers_instance = number_generator.gen_numbers_two_step(eq, n_vars, args.max_n, same_result=True)

                vars, res_base, res_alt = numbers_instance.values()
                vars = [str(i) for i in vars]
                res_base = str(res_base)
                res_alt = str(res_alt)

                second_part = 'statement' if 'statement' in args.prompt else 'question'
                temp1 = (t1_dict['body'] + ' ' + t1_dict[second_part])
                temp2 = (t2_dict['body'] + ' ' + t2_dict[second_part])

                eq = t1_dict['eq'] + ' | ' + t2_dict['eq']
                intervention = build_strings_and_add_intervention(temp1, temp2, vars, tokenizer, n_vars, eq,
                                                                  t1_dict['id'], t2_dict['id'], args)
                intervention.set_results(res_base, res_alt)
                interventions.append(intervention)
                used_count += 1

    print(f"\t Only used {used_count}/{all_count} examples")
    random.shuffle(interventions)
    return interventions


def change_template_same_result_interventions(templates, tokenizer, args):
    eq_dict = {}
    for s in templates:
        eq = s['eq']
        if eq not in eq_dict:
            eq_dict[eq] = [s]
        else:
            eq_dict[eq].append(s)

    all_count = 0
    used_count = 0
    interventions = []

    number_generator = NumberGenerator(args.max_n, args.seed)

    for _ in tqdm(range(args.examples_per_template), desc='generating interventions'):
        number_generator.reset_triple_lists()

        for eq, list_templates in eq_dict.items():
            list_s = list_templates.copy()

            while len(list_s) >= 2:
                all_count += 1

                random.shuffle(list_s)
                t1_dict = list_s.pop(0)
                t2_dict = list_s.pop(0)

                n_vars = max(t1_dict['n_vars'], t2_dict['n_vars'])
                numbers_instance = number_generator.gen_numbers(eq, n_vars, args.max_n)
                vars = numbers_instance['vars_base']
                res = numbers_instance['res_base']
                vars = [str(i) for i in vars]
                res = str(res)

                second_part = 'statement' if 'statement' in args.prompt else 'question'
                temp1 = (t1_dict['body'] + ' ' + t1_dict[second_part])
                temp2 = (t2_dict['body'] + ' ' + t2_dict[second_part])

                intervention = build_strings_and_add_intervention(temp1, temp2, vars, tokenizer, n_vars, eq, t1_dict['id'], t2_dict['id'], args)
                intervention.set_result(res)
                interventions.append(intervention)
                used_count += 1

    print(f"\t Only used {used_count}/{all_count} examples due to tokenizer")
    random.shuffle(interventions)
    return interventions
