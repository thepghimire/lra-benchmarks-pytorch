import unicodedata
import torch
import ml_collections
from train_utils import create_learning_rate_scheduler


# helper fns
def make_char_tokenizer(allowed_chars, lowercase_input=False):
    # make distinct
    #allowed_chars = list cha re
    allowed_chars = ['·', '0', 'T', 's', '\x8a', 'ä', '\x01', '\xa0', 'v', '\x12', '\x96', 'ò', 'B', 'È', '&', '®', '{', '\t', '|', '\x82', 't', '¼', 'i', 'É', '\x15', 'ô', 'ü', '«', '\x0b', '\x93', '\x9a', '?', '×', 'Ä', '´', 'ó', '~', 'H', '\x92', 'l', 'r', '\x00', '_', '\x8d', 'Ú', 'g', 'j', '\x88', 'Ö', 'ï', '(', 'G', '\x03', '\\', ';', 'Ô', '\x8c', 'e', 'ì', 'ý', '<', 'F', '\x94', '\xad', '`', 'Ï', '½', '\x11', 'w', '\x19', 'U', '°', 'ù', 'À', 'V', '\x80', 'u', '\x05', '4', '\x99', '\x9b', '\x8f', 'q', '\x14', '\x13', '1', '9', ',', '"', 'h', '³', '–', '*', '\x9c', 'ê', 'R', 'Z', 'O', '$', '\x0e', '¿', 'L', '#', '\r', 'd', '%', 'î', '\x0c', '!', 'm', '\x95', '\x1b', 'z', '\x98', '\x1d', '¤', 'D', '[', '\x08', 'S', 'à', 'ø', ' ', ']', 'C', '¸', '\x86', "'", '¢', 'I', '±', '3', 'Ì', 'Î', 'M', '\x1c', '\x9d', 'Ó', '\x02', '\x17', '+', '¾', '¯', '¦', '\x9f', 'í', 'á', '©', 'y', '¹', 'Ø', 'æ', '\x1a', 'Q', '2', '§', 'J', 'E', 'Á', '\x1e', '\x06', ':', '8', 'Û', '^', 'õ', '>', '\x18', '@', '\x90', 'ß', 'ç', '\x0f', '\x8e', '\x9e', 'X', 'Ê', ')', '£', 'þ', 'Ë', 'ÿ', '¶', '\x8b', 'P', '\x91', '\x1f', 'Ã', '\x84', '/', '\x07', '»', 'ñ', 'Y', 'ð', '¡', 'Æ', 'ö', 'º', 'x', '\x85', 'k', '}', 'Ç', 'Ð', 'â', '÷', 'f', 'Ü', '\n', 'Ñ', 'è', 'µ', 'ú', 'Ò', 'Þ', 'ª', 'Â', '²', '\x89', 'W', '\x10', '\x83', '\x87', 'ë', '\x81', '¬', 'Í', '\x16', 'é', 'Ù', 'Ý', '\x97', '5', '\x04', 'N', 'ã', '=', 'Å', '.', 'c', '¥', 'n', 'Õ', '\x7f', 'K', 'b', 'å', 'o', '7', '6', '¨', 'û', 'p', 'a', 'A', '-']
    allowed_chars = list(set(allowed_chars))
    # print(allowed_chars)
    # allowed_chars.append('–')
    # allowed_chars.append('…')
    # # print(len(allowed_chars))
    # print(allowed_chars)
    

    def _tokenizer(x, max_length):
        # note: x is not batched
        x = x[:max_length]
        if lowercase_input:
            x = x.lower()
        n = len(x)
        mask = ([1] * n) + ([0] * (max_length - n))
        error_items = ['—', '…', '‘','’','’','י','–','“','”','″','ג','Ż','א']
        for err in error_items:
            allowed_chars.append(err)        
        ids = list(map(lambda c: allowed_chars.index(c) + 1, x)) + ([0] * (max_length - n))
        return {'input_ids': torch.LongTensor([ids]), 'attention_mask': torch.LongTensor([mask])}

    _tokenizer.vocab_size = len(allowed_chars) + 1
    return _tokenizer


def make_word_tokenizer(allowed_words, lowercase_input=False, allow_unk=True):
    # make distinct
    allowed_words = list(set(allowed_words))
    PAD_TOKEN = 0
    UNK_TOKEN = 1

    def _tokenizer(x_str, max_length):
        # note: x_str is not batched
        if lowercase_input:
            x_str = x_str.lower()

        x = x_str.split()
        x = x[:max_length]
        n = len(x)
        mask = ([1] * n) + ([0] * (max_length - n))
        ids = list(map(lambda c: allowed_words.index(c) + 2 if c in allowed_words else UNK_TOKEN, x)) + \
                  ([PAD_TOKEN] * (max_length - n))
        if not allow_unk:
            assert UNK_TOKEN not in ids, "unknown words are not allowed by this tokenizer"
        return {'input_ids': torch.LongTensor([ids]), 'attention_mask': torch.LongTensor([mask])}

    _tokenizer.vocab_size = len(allowed_words) + 2
    return _tokenizer


def pixel_tokenizer(x, max_length):
    # note: x is not batched
    x = x.flatten()
    x = x[:max_length]
    n = len(x)
    ids = list(map(lambda a: a+1, x)) + ([0] * (max_length-n))
    mask = ([1] * n) + ([0] * (max_length-n))
    return {'input_ids': torch.LongTensor([ids]), 'attention_mask': torch.LongTensor([mask])}

pixel_tokenizer.vocab_size = 256 + 1

# ascii_tokenizer = make_char_tokenizer(''.join(chr(i) for i in range(256))) #TODO: Professor is investigating
#another way
for i in range(256):
    try:
        ascii_tokenizer = make_char_tokenizer(''.join(chr(i)))
    except ValueError as e:
        print(e)

# configs
def get_listops_config():
    config = ml_collections.ConfigDict()
    config.batch_size = 4
#     config.gradient_accumulation_steps = 8
    config.eval_frequency = 50
    config.total_eval_samples = 640
    config.total_train_samples = 160000
    config.learning_rate = 0.005
    config.weight_decay = 1e-1
    config.warmup_steps = 1000
    config.tied_weights = False
    config.max_length = 2000
    config.tokenizer = make_word_tokenizer(list('0123456789') + ['[', ']', '(', ')', 'MIN', 'MAX', 'MEDIAN', 'SUM_MOD'])
    #make_char_tokenizer(set('0123456789 MIN MAX MEDIAN SUM_MOD [ ] ( )'))
    config.lr_scheduler = create_learning_rate_scheduler("constant * linear_warmup * rsqrt_decay", config)

    model_config = ml_collections.ConfigDict()    
    model_config.max_position_embeddings = config.max_length
    model_config.num_attention_heads = 8
    model_config.num_hidden_layers = 6
    model_config.hidden_size = 512
    model_config.intermediate_size = 2048
    model_config.num_labels = 10
    model_config.vocab_size = config.tokenizer.vocab_size
    
    return config, model_config


def get_text_classification_config(num_labels=2):
    config = ml_collections.ConfigDict()
    config.batch_size = 4
    config.eval_frequency = 100
    config.total_train_samples = 640000
    config.total_eval_samples = -1
    config.learning_rate = 0.05
    config.weight_decay = 1e-1
    config.warmup_steps = 8000
    config.tokenizer = ascii_tokenizer
    config.tied_weights = False
    config.max_length = 1000
    config.lr_scheduler = create_learning_rate_scheduler("constant * linear_warmup * rsqrt_decay", config)

    model_config = ml_collections.ConfigDict()
    model_config.max_position_embeddings = config.max_length
    model_config.num_attention_heads = 4
    model_config.num_hidden_layers = 4
    model_config.hidden_size = 256
    model_config.intermediate_size = 1024
    model_config.num_labels = num_labels
    model_config.vocab_size = config.tokenizer.vocab_size

    return config, model_config


def get_cifar10_config():
    NUM_EPOCHS = 200
    TRAIN_EXAMPLES = 45000
    VALID_EXAMPLES = 10000
    
    config = ml_collections.ConfigDict()
    config.batch_size = 256
    config.tokenizer = pixel_tokenizer
    config.eval_frequency = TRAIN_EXAMPLES // config.batch_size
    config.total_eval_samples = VALID_EXAMPLES
    config.total_train_samples = TRAIN_EXAMPLES * NUM_EPOCHS
    config.weight_decay = 0.
    config.learning_rate = .0005
    config.warmup_steps = (TRAIN_EXAMPLES // config.batch_size) * 1
    config.tied_weights = False
    # 32 x 32 pics (which we "gray-scaled"..)
    config.max_length = 1024
    config.steps_per_cycle = (TRAIN_EXAMPLES // config.batch_size) * NUM_EPOCHS
    config.lr_scheduler = create_learning_rate_scheduler("constant * linear_warmup * cosine_decay", config)
    
    # model params
    model_config = ml_collections.ConfigDict()
    model_config.max_position_embeddings = config.max_length
    model_config.hidden_size = 32
    model_config.num_attention_heads = 1
    model_config.num_hidden_layers = 1
    model_config.intermediate_dim = 64
    model_config.hidden_dropout_prob = 0.3
    model_config.attention_probs_dropout_prob = 0.2
    model_config.num_labels = 10
    model_config.vocab_size = config.tokenizer.vocab_size
    
    return config, model_config
