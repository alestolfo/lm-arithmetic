import torch
import pdb

class Intervention:
    """
    Wrapper for all the possible interventions
    """

    def __init__(self,
                 tokenizer,
                 template_type,
                 base_string: str,
                 alt_string: str,
                 equation: str,
                 n_vars,
                 few_shots='',
                 few_shots_t2=None,
                 multitoken=False,
                 device='cpu'):
        self.op3_pos = None
        self.operator_word = None
        self.operands_alt = None
        self.operands_base = None
        self.operator_pos = None
        self.op2_pos = None
        self.op1_pos = None
        self.res_alt_tok = None
        self.res_base_tok = None
        self.res_string = None
        self.res_base_string = None
        self.res_alt_string = None
        self.device = device
        self.multitoken = multitoken
        self.is_llama = False

        self.template_id = template_type
        self.n_vars = n_vars

        # All the initial strings
        self.base_string = base_string
        self.alt_string = alt_string
        self.few_shots = few_shots
        if few_shots_t2 is not None:
            self.few_shots_t2 = few_shots_t2
        else:
            self.few_shots_t2 = few_shots

        self.equation = equation

        self.enc = tokenizer

        if self.enc is not None:
            self.is_llama = ('llama' in self.enc.name_or_path or 'alpaca' in self.enc.name_or_path)
            add_sp_tokens = True if 'google/flan' in self.enc.name_or_path else False
            if self.is_llama:
                base_string += ' '
                alt_string += ' '
            self.len_few_shots = len(self.enc.encode(self.few_shots))
            self.len_few_shots_t2 = len(self.enc.encode(self.few_shots_t2))
            self.base_string_tok_list = self.enc.encode(self.few_shots + base_string, add_special_tokens=add_sp_tokens)
            self.alt_string_tok_list = self.enc.encode(self.few_shots_t2 + alt_string, add_special_tokens=add_sp_tokens)

            self.base_string_tok = torch.LongTensor(self.base_string_tok_list).to(device).unsqueeze(0)
            self.alt_string_tok = torch.LongTensor(self.alt_string_tok_list).to(device).unsqueeze(0)

            assert len(self.base_string_tok_list) == len(self.alt_string_tok_list), '{} vs {}'.format(
                self.base_string, self.alt_string)

    def set_results(self, res_base, res_alt):
        self.res_base_string = res_base
        self.res_alt_string = res_alt

        if self.enc is not None:
            if 'google/flan' in self.enc.name_or_path:
                self.res_base_tok = ['▁' + res_base]
                self.res_alt_tok = ['▁' + res_alt]
                if not self.multitoken:
                    assert len(self.enc.convert_tokens_to_ids(self.res_base_tok)) == 1, '{} - {}'.format(
                        self.enc.tokenize(self.res_base_tok), res_base)
                    assert len(self.enc.convert_tokens_to_ids(self.res_alt_tok)) == 1, '{} - {}'.format(
                        self.enc.tokenize(self.res_alt_tok), res_alt)

            else:
                # 'a ' added to input so that tokenizer understands that first word
                # follows a space.
                if self.is_llama:
                    prefix = ''
                else:
                    prefix = 'a '
                self.res_base_tok = self.enc.tokenize(prefix + res_base)[1:]
                self.res_alt_tok = self.enc.tokenize(prefix + res_alt)[1:]
                if not self.multitoken:
                    assert len(self.res_base_tok) == 1, '{} - {}'.format(self.res_base_tok, res_base)
                    assert len(self.res_alt_tok) == 1, '{} - {}'.format(self.res_alt_tok, res_alt)

            self.res_base_tok = self.enc.convert_tokens_to_ids(self.res_base_tok)
            self.res_alt_tok = self.enc.convert_tokens_to_ids(self.res_alt_tok)

    @staticmethod
    def index_last_occurrence(lst, item):
        return len(lst) - lst[::-1].index(item) - 1

    def set_position_of_tokens(self, operands_base, operands_alt, operator_word, no_space_before_op1=False):
        self.operands_base = ' '.join(operands_base)
        self.operands_alt = ' '.join(operands_alt)
        self.operator_word = operator_word
        x_base, y_base = operands_base
        if self.is_llama:
            # todo for llama take the last token of the number
            x_base_tok = self.enc.tokenize(x_base)[-1:]
            y_base_tok = self.enc.tokenize(y_base)[-1:]
        else:
            prefix = '\n' if no_space_before_op1 else 'a '
            x_base_tok = self.enc.tokenize(prefix + x_base)[1:]
            y_base_tok = self.enc.tokenize('a ' + y_base)[1:]
        assert len(x_base_tok) == 1, '{} - {}'.format(x_base_tok, x_base)
        assert len(y_base_tok) == 1, '{} - {}'.format(y_base_tok, y_base)
        x_base_tok = self.enc.convert_tokens_to_ids(x_base_tok)[0]
        y_base_tok = self.enc.convert_tokens_to_ids(y_base_tok)[0]

        operator_word_tok = self.enc.tokenize('a ' + operator_word)[1:]
        assert len(operator_word_tok) == 1, '{} - {}'.format(operator_word_tok, operator_word)
        operator_word_tok = self.enc.convert_tokens_to_ids(operator_word_tok)[0]
        self.op2_pos = self.index_last_occurrence(self.base_string_tok_list, y_base_tok)
        self.op1_pos = self.index_last_occurrence(self.base_string_tok_list[:self.op2_pos], x_base_tok)

        self.operator_pos = self.index_last_occurrence(self.base_string_tok_list, operator_word_tok)

    def set_position_of_tokens_three_operands(self, operands_base, operands_alt):
        self.operands_base = ' '.join(operands_base)
        self.operands_alt = ' '.join(operands_alt)
        x_base, y_base, z_base = operands_base
        x_alt, y_alt, z_alt = operands_alt
        if self.is_llama:
            # todo for llama take the last token of the number
            x_base_tok = self.enc.tokenize(x_base)[-1:]
            y_base_tok = self.enc.tokenize(y_base)[-1:]
            z_base_tok = self.enc.tokenize(z_base)[-1:]
            z_alt_tok = self.enc.tokenize(z_alt)[-1:]
        else:
            prefix = 'a '
            x_base_tok = self.enc.tokenize(prefix + x_base)[1:]
            y_base_tok = self.enc.tokenize('a ' + y_base)[1:]
            z_base_tok = self.enc.tokenize('a ' + z_base)[1:]
            z_alt_tok = self.enc.tokenize('a ' + z_alt)[1:]
        assert len(x_base_tok) == 1, '{} - {}'.format(x_base_tok, x_base)
        assert len(y_base_tok) == 1, '{} - {}'.format(y_base_tok, y_base)
        assert len(z_base_tok) == 1, '{} - {}'.format(z_base_tok, z_base)
        x_base_tok = self.enc.convert_tokens_to_ids(x_base_tok)[0]
        y_base_tok = self.enc.convert_tokens_to_ids(y_base_tok)[0]
        z_base_tok = self.enc.convert_tokens_to_ids(z_base_tok)[0]
        z_alt_tok = self.enc.convert_tokens_to_ids(z_alt_tok)[0]

        self.op3_pos = self.index_last_occurrence(self.base_string_tok_list, z_base_tok)
        self.op2_pos = self.index_last_occurrence(self.base_string_tok_list[:self.op3_pos], y_base_tok)
        self.op1_pos = self.index_last_occurrence(self.base_string_tok_list[:self.op2_pos], x_base_tok)

        assert self.op1_pos < self.op2_pos < self.op3_pos
        assert self.op3_pos == self.index_last_occurrence(self.alt_string_tok_list, z_alt_tok)

    def set_position_of_tokens_lama(self, subj_base, subj_alt, no_space_before_sub=False):
        self.operands_base = subj_base
        self.operands_alt = subj_alt

        prefix = '\n' if no_space_before_sub else 'a '
        sub_base_tok = self.enc.tokenize(prefix + subj_base)[1:]
        sub_alt_tok = self.enc.tokenize(prefix + subj_alt)[1:]

        sub_base_last_token = self.enc.convert_tokens_to_ids(sub_base_tok)[-1]
        sub_base_first_token = self.enc.convert_tokens_to_ids(sub_base_tok)[0]
        sub_alt_last_token = self.enc.convert_tokens_to_ids(sub_alt_tok)[-1]
        sub_alt_first_token = self.enc.convert_tokens_to_ids(sub_alt_tok)[0]

        self.op1_pos = self.index_last_occurrence(self.base_string_tok_list, sub_base_first_token)
        self.op2_pos = self.index_last_occurrence(self.base_string_tok_list, sub_base_last_token)

        assert self.op1_pos == self.index_last_occurrence(self.alt_string_tok_list, sub_alt_first_token), \
            f'{self.op1_pos} - {self.index_last_occurrence(self.alt_string_tok_list, sub_alt_first_token)}'
        assert self.op2_pos == self.index_last_occurrence(self.alt_string_tok_list, sub_alt_last_token), \
            f'{self.op2_pos} - {self.index_last_occurrence(self.alt_string_tok_list, sub_alt_last_token)}'

    def set_position_of_tokens_int11(self, e1, e2):
        self.operands_base = e1
        self.operands_alt = e2

        prefix = 'a '
        e1_tok = self.enc.tokenize(prefix + e1)[1:]
        e2_tok = self.enc.tokenize(prefix + e2)[1:]

        e1_last_token = self.enc.convert_tokens_to_ids(e1_tok)[-1]
        e1_first_token = self.enc.convert_tokens_to_ids(e1_tok)[0]
        e2_last_token = self.enc.convert_tokens_to_ids(e2_tok)[-1]
        e2_first_token = self.enc.convert_tokens_to_ids(e2_tok)[0]

        self.e1_first_pos = self.index_last_occurrence(self.alt_string_tok_list, e1_first_token)
        self.e1_last_pos = self.index_last_occurrence(self.alt_string_tok_list, e1_last_token)
        self.e2_first_pos = self.index_last_occurrence(self.base_string_tok_list, e2_first_token)
        self.e2_last_pos = self.index_last_occurrence(self.base_string_tok_list, e2_last_token)

        self.entity_q_first = self.index_last_occurrence(self.alt_string_tok_list, e2_first_token)
        self.entity_q_last = self.index_last_occurrence(self.alt_string_tok_list, e2_last_token)

        assert self.entity_q_first == self.index_last_occurrence(self.base_string_tok_list, e1_first_token)
        assert self.entity_q_last == self.index_last_occurrence(self.base_string_tok_list, e1_last_token), f'{e1} - {e2}'

    def set_result(self, res):
        self.res_string = res

        if self.enc is not None and ('llama' not in self.enc.name_or_path and 'alpaca' not in self.enc.name_or_path):
            self.res_tok = self.enc.tokenize('a ' + res)[1:]
            if not self.multitoken:
                assert (len(self.res_tok) == 1)
