from kerMIT import tree
from kerMIT.dtk import DT
from tqdm import tqdm
import re
from kerMIT.tree_encode import parse as parse_tree
import torch
import pickle
import copy
import transformers
from torchtext import data as datx
from torch import nn
from torch import optim
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#if torch.cuda.is_available(): torch.cuda.manual_seed_all(10)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(3)

def get_sentence(sentence, calculator):
    # genero la forma parentetica
    tree_sentence = parse_tree(sentence)
    tree_sentence = re.sub("\("," (",tree_sentence)
    tree_sentence = tree_sentence[1:]
    # prendo i token di BERT
    bert_sentence = get_token_BERT(sentence)
    # calcolo il DTK
    alberoCompleto = tree.Tree(string=tree_sentence)
    dtk_sentence = calculator.dt(alberoCompleto).reshape(1,4000)
    dtk_sentence = torch.from_numpy(dtk_sentence).float().cuda()
    return tree_sentence, dtk_sentence, bert_sentence

def get_two_sentences(sentence1, sentence2, calculator):
    
    # genero la forma parentetica s1
    tree_sentence1 = parse_tree(sentence1)
    tree_sentence1 = re.sub("\("," (",tree_sentence1)
    tree_sentence1 = tree_sentence1[1:]

    # genero la forma parentetica s2
    tree_sentence2 = parse_tree(sentence2)
    tree_sentence2 = re.sub("\("," (",tree_sentence2)
    tree_sentence2 = tree_sentence2[1:]

    # prendo i token di BERT
    bert_sentence = get_token_BERT(f'{sentence1}[SEP]{sentence2}')

    # calcolo il DTK s1
    alberoCompleto1 = tree.Tree(string=tree_sentence1)
    dtk_sentence1 = calculator.dt(alberoCompleto1).reshape(1,4000)
    dtk_sentence1 = torch.from_numpy(dtk_sentence1).float().cuda()

    # calcolo il DTK s2
    alberoCompleto2 = tree.Tree(string=tree_sentence2)
    dtk_sentence2 = calculator.dt(alberoCompleto2).reshape(1,4000)
    dtk_sentence2 = torch.from_numpy(dtk_sentence2).float().cuda()

    dtk_sentence = torch.cat((dtk_sentence1, dtk_sentence2),1)
    tree_sentence = f'(S {tree_sentence1} {tree_sentence2} )'

    return tree_sentence, dtk_sentence, bert_sentence

#inizializzo tokenizzatore
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# bert prende in input un tensore di dim (51,). Nella segunete funzione codifico la sentence in input e poi sommo il tensore risultante con un tensore di dim (51,) inizializzato a 0
#NB ---> ORA(se tensore di encoding è maggiore di (51,) non lo considero e uso un tensore di (51,0) nullo) 
#TODO ---> se tensore di encoding è maggiore di (51,) si dovrebbero prendere solo i primi (51,)
def get_token_BERT(sentence):
    
    input_ids1 = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True))

    input_ids = torch.nn.functional.pad(input_ids1, (0, (51-input_ids1.shape[0]))).unsqueeze(0).cuda()
    
    if (input_ids.shape[0]>51):
        print('Bert problem')
        x_sem = torch.zeros(1, 51).cuda()
        x_sem = torch.tensor(x_sem).to(torch.int64)
        input_ids = x_sem
        
    return input_ids

###### Function BertForSequenceClassification

def get_sentence2(sentence, calculator):
    # genero la forma parentetica
    tree_sentence = parse_tree(sentence)
    tree_sentence = re.sub("\("," (",tree_sentence)
    tree_sentence = tree_sentence[1:]
    # prendo i token di BERT
    bert_sentence = input_to_bert([sentence],128,1)
    # calcolo il DTK
    alberoCompleto = tree.Tree(string=tree_sentence)
    dtk_sentence = calculator.dt(alberoCompleto).reshape(1,4000)
    dtk_sentence = torch.from_numpy(dtk_sentence).float().cuda()
    return tree_sentence, dtk_sentence, bert_sentence
    
def input_to_bert(sentences, max_len, b_s):
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
    
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    MAX_LEN = max_len

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []
  
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    batch_size = b_s 
    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    return prediction_dataloader

