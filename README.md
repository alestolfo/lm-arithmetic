# Mechanistically Interpreting Arithmetic in LLMs

This repository contains the code for the EMNLP 2023 paper "A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis".

<img src="arithmetic_cma.png" width="300">

## Setup
The requirements are listed in `requirements.txt`. To install them, run:

```pip install -r requirements.txt```

The configuration of the parameters is handled with Hydra.
The configuration files are located in `conf/`.
The default configuration is `conf/config.yaml`. 

### Parameters

- `intervention_type`: defines how the two input prompts differ. 
  - `1` -> the two prompts differ for the value of the operands. For example, `2 + 3 =` and `4 + 5 =`.
  - `2` -> the two prompts differ for the value of the operands, but the result is the same. For example, `2 + 3 =` and `4 + 1 =`.
  - `3` -> the two prompts differ for the operation. For example, `3 + 1 =` and `3 - 1 =`.
  - `11` -> number retrieval synthetic task. 
  - `20` -> factual knowledge queries. For this task set the `lama_path` parameter to the path to the locally downloaded LAMA dataset.
- `intervention_loc`: defines the type of components on which the interventions take place. Use `layer` for MLPs and `attention_layer_output` for the attention modules.
- `model` : `EleutherAI/gpt-j-6B`, `EleutherAI/pythia-2.8b-deduped` or `goat`. For LLaMA, set this parameter to the path to the locally downloaded model weights.
- `model_ckpt` : path to a fine-tuned version of one of the models above. Can be `null`.
- `n_operands`: number of operands in the input prompts.
- `examples_per_template`: number of prompt pairs generated per template.
- `n_shots`: number of exemplars included in the prompts.
- `max_n`: maximum value that the operands and the results can attain. (Experiments with LLaMA and Goat require `max_n=9`. However, in this case the constraint applies only to the value of the result, for example, `164 - 159 =` is a valid prompt.)
- `representation`: `arabic` or `words`. Defines the representation used for the numbers in the input prompts.
- `all_tokens`: if `true`, carry out the interventions on the components at each position of the input sequence. If `false`, carry out the interventions only on the components at the last position of the input sequence.
- `output_dir`: path to the directory where the results will be saved.


## Run
To run the code with the default configuration, run:

```python math_cma.py```

## Results
The results are saved in the directory specified by the `output_dir` parameter. 
The results are saved as `.feather` files.
In the `notebooks/` directory, we provide some notebooks that can be used to visualize the results.

## Citation
```
@article{stolfo2023understanding,
  title={Understanding Arithmetic Reasoning in Language Models using Causal Mediation Analysis},
  author={Stolfo, Alessandro and Belinkov, Yonatan and Sachan, Mrinmaya},
  journal={arXiv preprint arXiv:2305.15054},
  year={2023}
}
```
