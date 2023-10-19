from copy import deepcopy
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AutoModelForCausalLM, BertForMaskedLM, GPTNeoForCausalLM, BloomForCausalLM, \
    OPTForCausalLM, GPTNeoXForCausalLM, AutoModelForSeq2SeqLM, LlamaForCausalLM
from functools import partial
from interventions.lama_intervention import get_lama_vocab_subset
from utils.number_utils import batch, convert_to_words
from intervention_models.attention_modules import OverrideGPTJAttention, OverrideGPTNeoXAttention
from intervention_models import BaseModel
from peft import PeftModel


def load_model(args):
    return Model(device=args.device, random_weights=args.random_weights, model_version=args.model,
                 model_ckpt=args.model_ckpt, transformers_cache_dir=args.transformers_cache_dir, int8=args.int8)


class Model(BaseModel):

    def __init__(self, device='cpu', random_weights=False, model_version='gpt2', model_ckpt=None,
                 transformers_cache_dir=None, int8=False):

        super().__init__(model_version)
        assert (self.is_gpt2 or self.is_bert or self.is_gptj or self.is_gptneo or self.is_bloom
                or self.is_opt or self.is_neox or self.is_flan or self.is_pythia or self.is_llama)

        self.device = device

        model_class = (GPT2LMHeadModel if self.is_gpt2 else
                       AutoModelForCausalLM if self.is_gptj else
                       BertForMaskedLM if self.is_bert else
                       GPTNeoForCausalLM if self.is_gptneo else
                       BloomForCausalLM if self.is_bloom else
                       OPTForCausalLM if self.is_opt else
                       GPTNeoXForCausalLM if self.is_neox else
                       AutoModelForSeq2SeqLM if self.is_flan else
                       GPTNeoXForCausalLM if self.is_pythia else
                       LlamaForCausalLM if self.is_llama else
                       None)
        if self.is_llama and 'goat' in model_version:
            model = LlamaForCausalLM.from_pretrained(
                'decapoda-research/llama-7b-hf',
                load_in_8bit=int8,
                torch_dtype=torch.float16,
                cache_dir=transformers_cache_dir,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(
                model,
                'tiedong/goat-lora-7b',
                torch_dtype=torch.float16,
                cache_dir=transformers_cache_dir,
                device_map={'': 0},
            )
        elif model_ckpt:
            self.model = model_class.from_pretrained(model_ckpt)
            self.model.to(device)
        else:
            self.model = model_class.from_pretrained(
                model_version,
                load_in_8bit=int8,
                device_map='balanced',
                cache_dir=transformers_cache_dir,
                output_attentions=False
            )
            if hasattr(self.model, 'hf_device_map'):
                print('hf_device_map:', self.model.hf_device_map)
            else:
                self.model.to(device)

        if random_weights:
            print('Randomizing weights')
            self.model.init_weights()

        # store model details
        self.num_layers = self.model.config.num_hidden_layers
        self.num_neurons = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads

        if self.is_gpt2 or self.is_gptj:
            self.get_attention_layer = lambda layer: self.model.transformer.h[layer].attn
            self.word_emb_layer = self.model.transformer.wte
            self.get_neuron_layer = lambda layer: self.model.transformer.h[layer].mlp
        elif self.is_neox:
            self.get_attention_layer = lambda layer: self.model.gpt_neox.layers[layer].attention
            self.word_emb_layer = self.model.gpt_neox.embed_in
            self.get_neuron_layer = lambda layer: self.model.gpt_neox.layers[layer].mlp
        elif self.is_flan:
            self.get_attention_layer = lambda layer: (self.model.encoder.block + self.model.decoder.block)[layer].layer[
                0]
            self.word_emb_layer = self.model.encoder.embed_tokens
            self.get_neuron_layer = lambda layer: (self.model.encoder.block + self.model.decoder.block)[layer].layer[
                1 if layer < len(self.model.encoder.block) else 2]
        elif self.is_pythia:
            self.get_attention_layer = lambda layer: self.model.gpt_neox.layers[layer].attention
            self.word_emb_layer = self.model.gpt_neox.embed_in
            self.get_neuron_layer = lambda layer: self.model.gpt_neox.layers[layer].mlp
        elif self.is_llama:
            if 'goat' in model_version:
                self.get_attention_layer = lambda layer: self.model.model.model.layers[layer].self_attn
                self.word_emb_layer = self.model.model.model.embed_tokens
                self.get_neuron_layer = lambda layer: self.model.model.model.layers[layer].mlp
            else:
                self.get_attention_layer = lambda layer: self.model.model.layers[layer].self_attn
                self.word_emb_layer = self.model.model.embed_tokens
                self.get_neuron_layer = lambda layer: self.model.model.layers[layer].mlp
        else:
            raise Exception(f'Model version not supported: {model_version}')

    def create_vocab_subset(self, tokenizer, args):
        if args.intervention_type == 20:
            vocab_subset = get_lama_vocab_subset(tokenizer, args)
            ids, tokens = zip(*vocab_subset.items())
            self.vocab_subset = ids
            self.word_subset = tokens
        elif self.is_flan:
            if args.max_n > 55:
                raise Exception('flan-t5\'s tokenizer splits some numbers over 55')
            if args.representation == 'arabic':
                self.vocab_subset = tokenizer.convert_tokens_to_ids(['▁' + str(i) for i in range(args.max_n + 1)])
            else:
                raise Exception('Representation unknown: {}'.format(args.representation))
        else:
            if self.is_opt:
                prefix = 'a '
                start_idx = 2
            elif self.is_llama:
                start_idx = 2
                prefix = ''
            else:
                prefix = 'a '
                start_idx = 1

            if args.representation == 'arabic':
                self.vocab_subset = [tokenizer.encode(prefix + str(i))[start_idx:] for i in range(args.max_n + 1)]
            elif args.representation == 'words':
                self.vocab_subset = [tokenizer.encode(prefix + convert_to_words(str(i)))[start_idx:] for i in
                                     range(args.max_n + 1)]
            else:
                raise Exception('Representation unknown: {}'.format(args.representation))

    def intervention_experiment(self, interventions, effect_type, intervention_loc, multitoken=False,
                                get_full_distribution=False, all_tokens=False):
        intervention_results = {}
        progress = tqdm(total=len(interventions), desc='performing interventions')
        for idx, intervention in enumerate(interventions):
            intervention_results[idx] = self.neuron_intervention_single_experiment(intervention=intervention,
                                                                                   effect_type=effect_type,
                                                                                   intervention_loc=intervention_loc,
                                                                                   get_full_distribution=get_full_distribution,
                                                                                   all_tokens=all_tokens)
            progress.update()

        return intervention_results

    def get_representations(self, context, position=-1, is_attention=False):
        # Hook for saving the representation
        def extract_representation_hook(module,
                                        input,
                                        output,
                                        position,
                                        representations,
                                        layer):
            representations[layer] = output[(0, position)]

        def extract_representation_hook_attn(module,
                                             input,
                                             output,
                                             position,
                                             representations,
                                             layer):
            representations[layer] = output[0][0, position]

        handles = []
        representation = {}
        with torch.no_grad():
            # construct all the hooks
            # word embeddings will be layer -1
            if not is_attention:
                handles.append(self.word_emb_layer.register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=-1)))
            # hidden layers
            for layer_n in range(self.num_layers):
                if is_attention:
                    handles.append(self.get_attention_layer(layer_n).register_forward_hook(
                        partial(extract_representation_hook_attn,
                                position=position,
                                representations=representation,
                                layer=layer_n)))
                else:
                    handles.append(self.get_neuron_layer(layer_n).register_forward_hook(
                        partial(extract_representation_hook,
                                position=position,
                                representations=representation,
                                layer=layer_n)))
            if self.is_flan:
                self.model(context.to(self.device), decoder_input_ids=torch.tensor([[0]]).to(self.device))
            else:
                self.model(context.to(self.device))
            for h in handles:
                h.remove()
        return representation

    def get_probability_for_example(self, context, res_tok):
        """Return probabilities of single-token candidates given context"""
        if len(res_tok) > 1:
            raise ValueError(f"Multiple tokens not allowed: {res_tok}")

        if self.is_flan:
            decoder_input_ids = torch.tensor([[0]] * context.shape[0]).to(self.device)
            logits = self.model(context.to(self.device), decoder_input_ids=decoder_input_ids)[0].to(
                'cpu')
        else:
            logits = self.model(context.to(self.device))[0].to('cpu')
        logits = logits[:, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        probs_res = probs[:, res_tok[0]].squeeze().tolist()
        return probs_res

    def get_distribution_for_example(self, context):
        if self.is_flan:
            logits = self.model(context.to(self.device), decoder_input_ids=torch.tensor([[0]]).to(self.device))[0].to(
                'cpu')
        else:
            logits = self.model(context.to(self.device))[0].to('cpu')
        logits = logits[:, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        probs_subset = probs[:, self.vocab_subset].squeeze()
        return probs_subset

    def neuron_intervention_single_experiment(self,
                                              intervention,
                                              effect_type,
                                              bsize=200,
                                              intervention_loc='all',
                                              get_full_distribution=False,
                                              all_tokens=False):
        """
        run one full neuron intervention experiment
        """

        with torch.no_grad():

            # Probabilities without intervention (Base case)
            distrib_base = self.get_distribution_for_example(intervention.base_string_tok)
            distrib_alt = self.get_distribution_for_example(intervention.alt_string_tok)
            distrib_base = distrib_base.numpy()
            distrib_alt = distrib_alt.numpy()

            if effect_type == 'total':
                return distrib_base, distrib_alt, None, None

            context = intervention.base_string_tok

            positions = list(range(intervention.len_few_shots, len(context.squeeze()))) if all_tokens else [-1]

            res_base_probs = {}
            res_alt_probs = {}

            for position in positions:
                # assumes effect_type is indirect
                rep = self.get_representations(intervention.alt_string_tok, position=position)

                if intervention_loc == 'all':
                    # Now intervening on potentially biased example
                    dimensions = (self.num_layers + 1, self.num_neurons)
                    res_base_probs[position] = torch.zeros(dimensions)
                    res_alt_probs[position] = torch.zeros(dimensions)
                    first_layer = -1
                    for layer in tqdm(range(first_layer, self.num_layers)):
                        layers_to_search = [layer]
                        for neurons in batch(range(self.num_neurons), bsize):
                            neurons_to_search = [[i] for i in neurons]

                            probs = self.neuron_intervention(
                                context=context,
                                res_base_tok=intervention.res_base_tok,
                                res_alt_tok=intervention.res_alt_tok,
                                rep=rep,
                                layers=layers_to_search,
                                neurons=neurons_to_search,
                                position=-1,
                                intervention_loc=intervention_loc,
                                get_full_distribution=get_full_distribution)

                            for neuron, (p1, p2) in zip(neurons, probs):
                                res_base_probs[position][layer + 1][neuron] = p1
                                res_alt_probs[position][layer + 1][neuron] = p2

                elif intervention_loc == 'layer':
                    dimensions = (self.num_layers + 1, 1)
                    if get_full_distribution:
                        dimensions = (self.num_layers + 1, 1, len(self.vocab_subset))
                    res_base_probs[position] = torch.zeros(dimensions)
                    res_alt_probs[position] = torch.zeros(dimensions)
                    first_layer = -1
                    for layer in range(first_layer, self.num_layers):
                        neurons = list(range(self.num_neurons))
                        neurons_to_search = [neurons]
                        layers_to_search = [layer]

                        probs = self.neuron_intervention(
                            context=context,
                            res_base_tok=intervention.res_base_tok,
                            res_alt_tok=intervention.res_alt_tok,
                            rep=rep,
                            layers=layers_to_search,
                            neurons=neurons_to_search,
                            position=position,
                            intervention_loc=intervention_loc,
                            get_full_distribution=get_full_distribution)
                        p1, p2 = probs
                        res_base_probs[position][layer + 1][0] = torch.tensor(p1)
                        res_alt_probs[position][layer + 1][0] = torch.tensor(p2)

                elif intervention_loc.startswith('single_layer_'):
                    layer = int(intervention_loc.split('_')[-1])
                    dimensions = (1, self.num_neurons)
                    res_base_probs[position] = torch.zeros(dimensions)
                    res_alt_probs[position] = torch.zeros(dimensions)
                    layers_to_search = [layer]
                    for neurons in batch(range(self.num_neurons), bsize):
                        neurons_to_search = [[i] for i in neurons]

                        probs = self.neuron_intervention(
                            context=context,
                            res_base_tok=intervention.res_base_tok,
                            res_alt_tok=intervention.res_alt_tok,
                            rep=rep,
                            layers=layers_to_search,
                            neurons=neurons_to_search,
                            position=position,
                            intervention_loc=intervention_loc,
                            get_full_distribution=get_full_distribution)

                        for neuron, (p1, p2) in zip(neurons, probs):
                            res_base_probs[position][0][neuron] = p1
                            res_alt_probs[position][0][neuron] = p2

                else:
                    raise Exception(f'intervention_loc not defined: {intervention_loc}')

        return distrib_base, distrib_alt, res_base_probs, res_alt_probs

    def neuron_intervention(self,
                            context,
                            res_base_tok,
                            res_alt_tok,
                            rep,
                            layers,
                            neurons,
                            position,
                            intervention_loc,
                            get_full_distribution=False,
                            is_attention=False):
        # Hook for changing representation during forward pass
        def intervention_hook(module,
                              input,
                              output,
                              position,
                              neurons,
                              intervention,
                              is_attention=False):
            # Get the neurons to intervene on
            if is_attention:
                o = output[0]
            else:
                o = output
            out_device = o.device
            neurons = torch.LongTensor(neurons).to(out_device)
            # First grab the position across batch
            # Then, for each element, get correct index w/ gather
            base_slice = (slice(None), position, slice(None))
            base = o[base_slice].gather(1, neurons)
            intervention_view = intervention.view_as(base)
            base = intervention_view

            # Overwrite values in the output
            # First define mask where to overwrite
            scatter_mask = torch.zeros_like(o, dtype=torch.bool)
            for i, v in enumerate(neurons):
                scatter_mask[(i, position, v)] = 1
            # Then take values from base and scatter
            o.masked_scatter_(scatter_mask, base.flatten())

        # Set up the context as batch
        batch_size = len(neurons)
        context = context.repeat(batch_size, 1)
        handle_list = []
        for layer in set(layers):
            if intervention_loc == 'all' or intervention_loc.startswith('single_layer_'):
                neuron_loc = np.where(np.array(layers) == layer)[0]
                n_list = []
                for n in neurons:
                    unsorted_n_list = [n[i] for i in neuron_loc]
                    n_list.append(list(np.sort(unsorted_n_list)))
            elif intervention_loc == 'layer':
                n_list = neurons
            else:
                raise Exception(f'intervention_loc not defined: {intervention_loc}')
            m_list = n_list
            intervention_rep = rep[layer][m_list]
            if layer == -1:
                if is_attention:
                    raise Exception('Attention not implemented for word embedding intervention')

                handle_list.append(self.word_emb_layer.register_forward_hook(
                    partial(intervention_hook,
                            position=position,
                            neurons=n_list,
                            intervention=intervention_rep)))
            else:
                if is_attention:
                    module = self.get_attention_layer(layer)
                else:
                    module = self.get_neuron_layer(layer)
                handle_list.append(module.register_forward_hook(
                    partial(intervention_hook,
                            position=position,
                            neurons=n_list,
                            intervention=intervention_rep,
                            is_attention=is_attention)))

        if get_full_distribution:
            # new distribution is a list of tensors
            new_distrib = self.get_distribution_for_example(context)
            probs = new_distrib, [0. for _ in range(len(new_distrib))]
        else:
            new_base_probability = self.get_probability_for_example(context, res_base_tok)
            new_alt_probability = self.get_probability_for_example(context, res_alt_tok)

            if type(new_base_probability) is not list:
                probs = new_base_probability, new_alt_probability
            else:
                probs = zip(new_base_probability, new_alt_probability)

        for handle in handle_list:
            handle.remove()

        return probs

    def attention_experiment(self, interventions, effect_type, intervention_loc, multitoken=False,
                             get_full_distribution=False, all_tokens=False):
        intervention_results = {}
        progress = tqdm(total=len(interventions), desc='performing interventions')
        for idx, intervention in enumerate(interventions):
            intervention_results[idx] = self.attention_intervention_single_experiment(intervention=intervention,
                                                                                      effect_type=effect_type,
                                                                                      intervention_loc=intervention_loc,
                                                                                      get_full_distribution=get_full_distribution,
                                                                                      all_tokens=all_tokens)
            progress.update()

        return intervention_results

    def attention_intervention_single_experiment(self, intervention, intervention_loc,
                                                 get_full_distribution=False, all_tokens=False):
        """
        Run one full attention intervention experiment
        measuring indirect effect.
        """

        with torch.no_grad():
            # Probabilities without intervention (Base case)
            distrib_base = self.get_distribution_for_example(intervention.base_string_tok)
            distrib_alt = self.get_distribution_for_example(intervention.alt_string_tok)
            distrib_base = distrib_base.numpy()
            distrib_alt = distrib_alt.numpy()

            # E.g. 4 plus 5 is...
            x = intervention.base_string_tok[0]
            # E.g. 1 plus 2 is...
            x_alt = intervention.alt_string_tok[0]

            input = x_alt  # Get attention for x_alt
            context = x

            intervention_on_output = len(intervention_loc.split('_')) > 1 and intervention_loc.split('_')[1] == 'output'

            batch_size = 1
            seq_len = len(x)
            seq_len_alt = len(x_alt)

            positions = list(range(intervention.len_few_shots, len(context.squeeze()))) if all_tokens else [-1]

            res_base_probs = {}
            res_alt_probs = {}

            for position in positions:
                if intervention_on_output:
                    attention_override = self.get_representations(input.unsqueeze(0), position=position,
                                                                  is_attention=True)
                    assert attention_override[0].shape[0] == self.num_neurons, \
                        f'attention_override[0].shape: {attention_override[0].shape} vs {self.num_neurons}'
                else:
                    batch = input.clone().detach().unsqueeze(0).to(self.device)
                    model_output = self.model(batch, output_attentions=True)
                    attention_override = model_output[-1]
                    assert attention_override[0].shape == (batch_size, self.num_heads, seq_len, seq_len), \
                        f'attention_override[0].shape: {attention_override[0].shape} vs ({batch_size}, {self.num_heads}, {seq_len}, {seq_len})'

                assert seq_len == seq_len_alt, f'x: [{x}] vs x_alt: [{x_alt}]'
                assert len(attention_override) == self.num_layers

                # basically generate the mask for the layers_to_adj and heads_to_adj
                if intervention_loc == 'head':
                    candidate1_probs_head = torch.zeros((self.num_layers, self.num_heads))
                    candidate2_probs_head = torch.zeros((self.num_layers, self.num_heads))

                    for layer in range(self.num_layers):
                        layer_attention_override = attention_override[layer]

                        # one head at a time
                        for head in range(self.num_heads):
                            attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)

                            # Set mask to 1 for single head only
                            # this should have shape (1, n_heads, seq_len, seq_len)
                            attention_override_mask[0][head] = 1

                            head_attn_override_data = [{
                                'layer': layer,
                                'attention_override': layer_attention_override,
                                'attention_override_mask': attention_override_mask
                            }]

                            candidate1_probs_head[layer][head], candidate2_probs_head[layer][
                                head] = self.attention_intervention(
                                context=context.unsqueeze(0),
                                res_base_tok=intervention.res_base_tok,
                                res_alt_tok=intervention.res_alt_tok,
                                attn_override_data=head_attn_override_data)

                elif intervention_loc.startswith('layer'):
                    dimensions = (self.num_layers, 1)
                    res_base_probs[position] = torch.zeros(dimensions)
                    res_alt_probs[position] = torch.zeros(dimensions)

                    for layer in range(self.num_layers):
                        if intervention_on_output:
                            probs = self.neuron_intervention(
                                context=context.unsqueeze(0),
                                res_base_tok=intervention.res_base_tok,
                                res_alt_tok=intervention.res_alt_tok,
                                rep=attention_override,
                                layers=[layer],
                                neurons=[list(range(self.num_neurons))],
                                position=position,
                                intervention_loc='layer',
                                get_full_distribution=get_full_distribution,
                                is_attention=True)
                            p1, p2 = probs
                            res_base_probs[position][layer][0] = torch.tensor(p1)
                            res_alt_probs[position][layer][0] = torch.tensor(p2)
                        else:
                            layer_attention_override = attention_override[layer]

                            # set all the head_masks in layer to 1
                            attention_override_mask = torch.ones_like(layer_attention_override, dtype=torch.bool)

                            head_attn_override_data = [{
                                'layer': layer,
                                'attention_override': layer_attention_override,
                                'attention_override_mask': attention_override_mask
                            }]

                            candidate1_probs_head[layer][0], candidate2_probs_head[layer][0] = \
                                self.attention_intervention(
                                    context=context.unsqueeze(0),
                                    res_base_tok=intervention.res_base_tok,
                                    res_alt_tok=intervention.res_alt_tok,
                                    attn_override_data=head_attn_override_data)

                else:
                    raise ValueError(f"Invalid intervention_loc: {intervention_loc}")

        return distrib_base, distrib_alt, res_base_probs, res_alt_probs

    def attention_intervention(self,
                               context,
                               res_base_tok,
                               res_alt_tok,
                               attn_override_data):
        """ Override attention values in specified layer
        Args:
            context: context text
            attn_override_data: list of dicts of form:
                {
                    'layer': <index of layer on which to intervene>,
                    'attention_override': <values to override the computed attention weights.
                           Shape is [batch_size, num_heads, seq_len, seq_len]>,
                    'attention_override_mask': <indicates which attention weights to override.
                                Shape is [batch_size, num_heads, seq_len, seq_len]>
                }
        """

        def intervention_hook(module, input, kwargs, outputs, attn_override, attn_override_mask):
            # attention_override_module = AttentionOverride(
            #    module, attn_override, attn_override_mask)
            attention_override_module_class = (OverrideGPTJAttention if self.is_gptj else
                                               OverrideGPTNeoXAttention if self.is_pythia else
                                               None)
            if attention_override_module_class is None:
                raise ValueError("Invalid model type")

            attention_override_module = attention_override_module_class(
                module, attn_override, attn_override_mask
            )

            attention_override_module.to(self.device)

            return attention_override_module(*input, **kwargs)

        with torch.no_grad():
            hooks = []
            for d in attn_override_data:
                attn_override = d['attention_override']
                attn_override_mask = d['attention_override_mask']
                layer = d['layer']
                hooks.append(self.get_attention_layer(layer).register_forward_hook(
                    partial(intervention_hook,
                            attn_override=attn_override,
                            attn_override_mask=attn_override_mask), with_kwargs=True))

            # new probabilities are scalar
            new_base_probabilities = self.get_probability_for_example(
                context,
                res_base_tok)

            new_alt_probabilities = self.get_probability_for_example(
                context,
                res_alt_tok)

            for hook in hooks:
                hook.remove()

            return new_base_probabilities, new_alt_probabilities

    @staticmethod
    def process_intervention_results(interventions, list_of_words, intervention_results, args):
        results = []
        for example in intervention_results:
            distrib_base, distrib_alt, \
                res_base_probs, res_alt_probs = intervention_results[example]

            intervention = interventions[example]

            if args.intervention_type == 20:
                res_base = intervention.res_base_string
                res_alt = intervention.res_alt_string
                res_base_idx = list_of_words.index(res_base)
                res_alt_idx = list_of_words.index(res_alt)

                res_base_base_prob = distrib_base[res_base_idx]
                res_alt_base_prob = distrib_base[res_alt_idx]
                res_base_alt_prob = distrib_alt[res_base_idx]
                res_alt_alt_prob = distrib_alt[res_alt_idx]

                pred_base_idx = np.argmax(distrib_base)
                pred_alt_idx = np.argmax(distrib_alt)
                pred_base = list_of_words[pred_base_idx]
                pred_alt = list_of_words[pred_alt_idx]

            else:
                if args.representation == 'arabic':
                    res_base = int(intervention.res_base_string)
                    res_alt = int(intervention.res_alt_string)
                else:
                    words_to_n = {convert_to_words(str(i)): i for i in range(args.max_n + 1)}
                    res_base = int(words_to_n[intervention.res_base_string])
                    res_alt = int(words_to_n[intervention.res_alt_string])

                res_base_base_prob = distrib_base[res_base]
                res_alt_base_prob = distrib_base[res_alt]
                res_base_alt_prob = distrib_alt[res_base]
                res_alt_alt_prob = distrib_alt[res_alt]

                pred_base = np.argmax(distrib_base)
                pred_alt = np.argmax(distrib_alt)

            # accuracy_10 = int(res_base in top_10_preds_base) * 0.5 + int(res_alt in top_10_preds_alt) * 0.5
            accuracy = int(pred_base == res_base) * 0.5 + int(pred_alt == res_alt) * 0.5

            metric_dict = {
                'example': example,
                'template_id': intervention.template_id,
                'n_vars': intervention.n_vars,
                'base_string': intervention.base_string,
                'alt_string': intervention.alt_string,
                'few_shots': intervention.few_shots,
                'equation': intervention.equation,
                'res_base': intervention.res_base_string,
                'res_alt': intervention.res_alt_string,
                # base probs
                'res_base_base_prob': float(res_base_base_prob),
                'res_alt_base_prob': float(res_alt_base_prob),
                'res_base_alt_prob': float(res_base_alt_prob),
                'res_alt_alt_prob': float(res_alt_alt_prob),
                # distribs
                'distrib_base': distrib_base,
                'distrib_alt': distrib_alt,
                # preds
                'pred_base': pred_base,
                'pred_alt': pred_alt,
                'accuracy': accuracy,
                # operands
                'operands_base': intervention.operands_base,
                'operands_alt': intervention.operands_alt
            }

            if args.all_tokens:
                if args.intervention_type == 11:
                    metric_dict.update({
                        'e1_first_pos': intervention.e1_first_pos,
                        'e1_last_pos': intervention.e1_last_pos,
                        'e2_first_pos': intervention.e2_first_pos,
                        'e2_last_pos': intervention.e2_last_pos,
                        'entity_q_first': intervention.entity_q_first,
                        'entity_q_last': intervention.entity_q_last,
                    })
                else:
                    if args.n_operands == 2:
                        metric_dict.update({
                            'op1_pos': intervention.op1_pos,
                            'op2_pos': intervention.op2_pos,
                            'operator_pos': intervention.operator_pos,
                            'operation': intervention.equation.split()[1],
                        })
                    elif args.n_operands == 3:
                        operations = intervention.equation.replace('{x}', '').replace('{y}', '').replace('{z}', '')
                        operations = operations.replace('(', '').replace(')', '')
                        metric_dict.update({
                            'op1_pos': intervention.op1_pos,
                            'op2_pos': intervention.op2_pos,
                            'op3_pos': intervention.op3_pos,
                            'operation': operations,
                        })
                    else:
                        raise NotImplementedError

            if res_base_probs is None:
                results.append(metric_dict)
            else:
                for position in res_base_probs.keys():
                    for layer in range(res_base_probs[position].size(0)):
                        if args.intervention_loc.startswith('single_layer_'):
                            layer_number = int(args.intervention_loc.split('_')[-1])
                        else:
                            layer_number = layer
                        for neuron in range(res_base_probs[position].size(1)):
                            c1_prob, c2_prob = res_base_probs[position][layer][neuron], res_alt_probs[position][layer][
                                neuron]
                            results_single = deepcopy(metric_dict)
                            results_single.update({
                                'position': position,
                                'layer': layer_number,
                                'neuron': neuron})
                            if args.get_full_distribution:
                                results_single['distrib_alt'] = c1_prob.numpy()
                            else:
                                results_single.update({  # strings
                                    # intervention probs
                                    'res_base_prob': float(c1_prob),
                                    'res_alt_prob': float(c2_prob),
                                })
                                if 'distrib_base' in metric_dict:
                                    metric_dict.pop('distrib_base')
                                    metric_dict.pop('distrib_alt')
                            if 'few_shots' in metric_dict:
                                metric_dict.pop('few_shots')

                            results.append(results_single)

        return pd.DataFrame(results)
