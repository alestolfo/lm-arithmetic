class BaseModel:
    def __init__(self, model_version):
        self.word_subset = None
        self.vocab_subset = None
        self.is_gpt2 = model_version.startswith('gpt2') or model_version.startswith('distilgpt2')
        self.is_gptj = model_version.startswith('EleutherAI/gpt-j')
        self.is_bert = model_version.startswith('bert')
        self.is_neox = model_version.startswith('EleutherAI/gpt-neox')
        self.is_gptneo = model_version.startswith('EleutherAI/gpt-neo')
        self.is_openai_chat = model_version.startswith('gpt-3.5') or model_version.startswith('gpt-4')
        self.is_gpt3 = model_version.startswith('gpt3')
        self.is_bloom = model_version.startswith('bigscience/bloom')
        self.is_opt = model_version.startswith('facebook/opt')
        self.is_llama = 'llama' in model_version or 'alpaca' in model_version or 'goat' in model_version
        self.is_flan = model_version.startswith('google/flan-t5')
        self.is_pythia = model_version.startswith('EleutherAI/pythia')
