import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np


class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, peephole=False, forget_bias=0.0):
        super().__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.peephole = peephole
        self.W = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size * 4))
        self.peep_i = nn.Parameter(torch.Tensor(hidden_size))
        self.peep_f = nn.Parameter(torch.Tensor(hidden_size))
        self.peep_o = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.forget_bias = forget_bias
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, sequence_len=None,
                init_states=None):
        """Assumes x is of shape (sequence, batch, feature)"""
        if sequence_len is None:
            seq_sz, bs, _ = x.size()
        else:
            seq_sz = sequence_len.max()
            _, bs, _ = x.size()
        hidden_seq = []
        if init_states is None:
            c_t, h_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            c_t, h_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[t, :, :]
            if sequence_len is not None:
                if sequence_len.min() <= t+1:
                    old_c_t = c_t.clone().detach()
                    old_h_t = h_t.clone().detach()
            # batch the computations into a single matrix multiplication
            lstm_mat = torch.cat([x_t, h_t], dim=1)
            if self.peephole:
                gates = lstm_mat @ self.W + self.bias
            else:
                gates = lstm_mat @ self.W + self.bias
                g_t = torch.tanh(gates[:, HS * 2:HS * 3])

            if self.peephole:
                i_t, j_t, f_t, o_t = (
                    (gates[:, :HS]),  # input
                    (gates[:, HS:HS * 2]),  # new input
                    (gates[:, HS * 2:HS * 3]),   # forget
                    (gates[:, HS * 3:])   # output
                )
            else:
                i_t, f_t, o_t = (
                    torch.sigmoid(gates[:, :HS]),  # input
                    torch.sigmoid(gates[:, HS:HS * 2]),# + self.forget_bias),  # forget
                    torch.sigmoid(gates[:, HS * 3:])  # output
                )

            if self.peephole:
                c_t = torch.sigmoid(f_t + self.forget_bias + c_t * self.peep_f) * c_t \
                      + torch.sigmoid(i_t + c_t * self.peep_i) * torch.tanh(j_t)
                h_t = torch.sigmoid(o_t + c_t * self.peep_o) * torch.tanh(c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

            out = h_t.clone()
            if sequence_len is not None:
                if sequence_len.min() <= t:
                    c_t = torch.where(torch.tensor(sequence_len).to(c_t.device) <= t, old_c_t.T, c_t.T).T
                    h_t = torch.where(torch.tensor(sequence_len).to(h_t.device) <= t, old_h_t.T, h_t.T).T
                    out = torch.where(torch.tensor(sequence_len).to(out.device) <= t, torch.zeros(out.shape).to(out.device).T, out.T).T

            hidden_seq.append(out.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)

        return hidden_seq, (c_t, h_t)

def train_gmu_opp(model, batch, optimiser, epoch_loss, params):
    optimiser.zero_grad()  # free the optimizer from previous gradients
    gt_description = batch['L_fw'][1:]
    gt_action = batch['B_fw'][1:]
    ran_sig = torch.randint(3, (1,))
    opp = 0
    if ran_sig == 0:
        rep_sig = torch.randint(3, (1,))
        if rep_sig == 0:
            signal = 'repeat action'
            opp = torch.randint(2, (1,))
            gt_description = gt_description[-1].unsqueeze(0)
        elif rep_sig == 1:
            signal = 'repeat both'
            opp = torch.randint(2, (1,))
        else:
            signal = 'repeat language'
            gt_action = batch['B_fw'][0].repeat(len(gt_action), 1, 1)* batch["B_bin"][1:]
    elif ran_sig ==1:
        signal = 'describe'
        opp = torch.randint(2, (1,))
        if opp == 0:
            gt_action = batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
        else:
            gt_action = batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
    else:
        signal = 'execute'
        gt_description = gt_description[-1].unsqueeze(0)
    output = model(batch, signal, opp)
    L_loss, B_loss, batch_loss = loss_gmu(output, gt_description, gt_action, batch["B_bin"], signal, params)  # compute loss
    batch_loss.backward()  # compute gradients
    optimiser.step()  # update weights
    epoch_loss.append(batch_loss.item())  # record the batch loss

    return L_loss, B_loss, batch_loss, signal  # return the losses

def validate_gmu_opp(model, batch, epoch_loss, params):
    with torch.no_grad():
        gt_description = batch['L_fw'][1:]
        gt_action = batch['B_fw'][1:]
        opp = 0
        ran_sig = torch.randint(3, (1,))
        if ran_sig == 0:
            rep_sig = torch.randint(3, (1,))
            if rep_sig == 0:
                signal = 'repeat action'
                gt_description = gt_description[-1].unsqueeze(0)
                opp = torch.randint(2, (1,))
            elif rep_sig == 1:
                signal = 'repeat both'
                opp = torch.randint(2, (1,))
            else:
                signal = 'repeat language'
                gt_action = batch['B_fw'][0].repeat(len(gt_action), 1, 1)* batch["B_bin"][1:]
        elif ran_sig == 1:
            signal = 'describe'
            opp = torch.randint(2, (1,))
            if opp == 0:
                gt_action = batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
            else:
                gt_action = batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
        else:
            signal = 'execute'
            gt_description = gt_description[-1].unsqueeze(0)
        output = model(batch, signal, opp)
        L_loss, B_loss, batch_loss = loss_gmu(output, gt_description, gt_action, batch["B_bin"], signal, params)  # compute loss
        epoch_loss.append(batch_loss.item())  # record the batch loss

    return L_loss, B_loss, batch_loss, signal # return the losses

def train_gmu(model, batch, optimiser, epoch_loss, params, unimodal_ratio, vis_out = False):
    optimiser.zero_grad()  # free the optimizer from previous gradients
    gt_description = batch['L_fw'][1:]
    if vis_out:
        gt_action = torch.cat((batch['V_fw'][1:],batch['B_fw'][1:]), dim=-1)
    else:
        gt_action = batch['B_fw'][1:]
    ran_sig = torch.randint(100, (1,))
    if ran_sig < unimodal_ratio:
        rep_sig = torch.randint(2, (1,))  # torch.randint(3, (1,))
        if rep_sig == 0:
            signal = 'repeat action'
            gt_description = gt_description[-1].unsqueeze(0)
        else:
            signal = 'repeat language'
            if vis_out:
                gt_action = torch.cat((batch['V_fw'][0].repeat(len(gt_action), 1, 1) *
                                       batch["B_bin"][1:].repeat(1, 1, int(batch["V_fw"].shape[-1] / batch["B_bin"].shape[-1])),
                                       batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]), dim=-1)
            else:
                gt_action = batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
    else:
        supervised_sig = torch.randint(3, (1,))
        if supervised_sig == 0:
            signal = 'describe'
            if vis_out:
                gt_action = torch.cat((batch['V_bw'][0].repeat(len(gt_action), 1, 1) *
                                       batch["B_bin"][1:].repeat(1, 1, int(batch["V_bw"].shape[-1] / batch["B_bin"].shape[-1])),
                                       batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]), dim=-1)
            else:
                gt_action = batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
        elif supervised_sig == 1:
            signal = 'execute'
            gt_description = gt_description[-1].unsqueeze(0)
        else:
            signal = 'repeat both'
    output = model(batch, signal)
    L_loss, B_loss, batch_loss = loss_gmu(output, gt_description, gt_action, batch["B_bin"], signal, params, vis_out)  # compute loss
    batch_loss.backward()  # compute gradients
    optimiser.step()  # update weights
    epoch_loss.append(batch_loss.item())  # record the batch loss

    return L_loss, B_loss, batch_loss, signal  # return the losses

def validate_gmu(model, batch, epoch_loss, params, unimodal_ratio, vis_out=False):
    with torch.no_grad():
        gt_description = batch['L_fw'][1:]
        if vis_out:
            gt_action = torch.cat((batch['V_fw'][1:], batch['B_fw'][1:]), dim=-1)
        else:
            gt_action = batch['B_fw'][1:]
        ran_sig = torch.randint(100, (1,))
        if ran_sig < unimodal_ratio:
            rep_sig = torch.randint(2, (1,))
            if rep_sig == 0:
                signal = 'repeat action'
                gt_description = gt_description[-1].unsqueeze(0)

            else:
                signal = 'repeat language'
                if vis_out:
                    gt_action = torch.cat((batch['V_fw'][0].repeat(len(gt_action), 1, 1) *
                                           batch["B_bin"][1:].repeat(1, 1, int(batch["V_fw"].shape[-1] / batch["B_bin"].shape[-1])),
                                           batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]), dim=-1)
                else:
                    gt_action = batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
        else:
            supervised_sig = torch.randint(3, (1,))  # torch.randint(3, (1,))
            if supervised_sig == 0:
                signal = 'describe'
                if vis_out:
                    gt_action = torch.cat((batch['V_bw'][0].repeat(len(gt_action), 1, 1) *
                                           batch["B_bin"][1:].repeat(1, 1, int(batch["V_bw"].shape[-1] / batch["B_bin"].shape[-1])),
                                           batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]), dim=-1)
                else:
                    gt_action = batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
            elif supervised_sig == 1:
                signal = 'execute'
                gt_description = gt_description[-1].unsqueeze(0)
            else:
                signal = 'repeat both'
        output = model(batch, signal)
        L_loss, B_loss, batch_loss = loss_gmu(output, gt_description, gt_action, batch["B_bin"], signal, params, vis_out)  # compute loss
        epoch_loss.append(batch_loss.item())  # record the batch loss

    return L_loss, B_loss, batch_loss, signal # return the losses

def loss_gmu(output, gt_description, gt_action, B_bin, signal, net_conf, vis_out=False):
    if signal == 'repeat both':
        [L_output, B_output] = output
        if vis_out:
            B_output = B_output * B_bin[1:].repeat(1, 1, int(B_output.shape[-1] / B_bin.shape[-1]))
        else:
            B_output = B_output * B_bin[1:]
        # L_loss = torch.mean(-torch.sum(torch.tensor([0.1048, 0.1048, 1.2571, 1.2571, 1.2571, 1.2571, 1.2571, 1.2571, 1.2571,
        # 0.8381, 0.8381, 0.8381, 0.8381, 0.8381, 1.2571, 0.8381, 1.2571, 1.2571,
        # 0.8381, 0.8381, 1.2571, 1.2571]).to('cuda') *gt_description * torch.log(L_output), 2))
        L_loss = torch.mean(-torch.sum(torch.tensor([0.1168, 1.4010, 1.4010, 1.4010, 1.4010, 0.4670, 1.4010, 1.4010, 1.4010,
        0.7005, 0.7005, 0.7005, 0.7005, 1.4010, 0.7005, 0.4670, 1.4010, 1.4010, 0.7005, 0.4670, 0.4670, 1.4010, 1.4010]).to('cuda')
                                      *gt_description * torch.log(L_output), 2))  # description loss
        #L_loss = torch.mean(-torch.sum(torch.tensor([0.0051, 0.0609, 0.0609, 0.0609, 0.0609, 0.0203, 0.0609, 0.0609, 0.0609,
        #0.0305, 0.0305, 0.0305, 0.0305, 0.0609, 0.0305, 0.0203, 0.0609, 0.0609, 0.0305, 0.0203, 0.0203, 0.0609, 0.0609]).to('cuda')
        #                              *gt_description * torch.log(L_output)*(torch.ones(L_output.size()).to('cuda')-L_output)**2, 2))  # description loss
        B_loss = torch.mean(torch.square(B_output - gt_action))  # action loss (MSE)
    elif signal == 'describe' or signal == 'repeat language':
        [L_output, B_output] = output
        if vis_out:
            B_output = B_output * B_bin[1:].repeat(1, 1, int(B_output.shape[-1] / B_bin.shape[-1]))
        else:
            B_output = B_output * B_bin[1:]
        # L_loss = torch.mean(-torch.sum(torch.tensor([0.1048, 0.1048, 1.2571, 1.2571, 1.2571, 1.2571, 1.2571, 1.2571, 1.2571,
        # 0.8381, 0.8381, 0.8381, 0.8381, 0.8381, 1.2571, 0.8381, 1.2571, 1.2571,
        # 0.8381, 0.8381, 1.2571, 1.2571]).to('cuda') *gt_description * torch.log(L_output), 2))
        L_loss = torch.mean(-torch.sum(torch.tensor([0.1168, 1.4010, 1.4010, 1.4010, 1.4010, 0.4670, 1.4010, 1.4010, 1.4010,
        0.7005, 0.7005, 0.7005, 0.7005, 1.4010, 0.7005, 0.4670, 1.4010, 1.4010, 0.7005, 0.4670, 0.4670, 1.4010, 1.4010]).to('cuda')
                                      *gt_description * torch.log(L_output), 2))  # description loss
        #L_loss = torch.mean(-torch.sum(torch.tensor([0.0051, 0.0609, 0.0609, 0.0609, 0.0609, 0.0203, 0.0609, 0.0609, 0.0609,
        #0.0305, 0.0305, 0.0305, 0.0305, 0.0609, 0.0305, 0.0203, 0.0609, 0.0609, 0.0305, 0.0203, 0.0203, 0.0609, 0.0609]).to('cuda')
        #                              *gt_description * torch.log(L_output)*(torch.ones(L_output.size()).to('cuda')-L_output)**2, 2))
        B_loss = torch.mean(torch.square(B_output - gt_action))  # action loss (MSE)
    else:
        [L_output, B_output] = output
        if vis_out:
            B_output = B_output * B_bin[1:].repeat(1, 1, int(B_output.shape[-1] / B_bin.shape[-1]))
        else:
            B_output = B_output * B_bin[1:]
        # L_loss = torch.mean(-torch.sum(torch.tensor([0.1048, 0.1048, 1.2571, 1.2571, 1.2571, 1.2571, 1.2571, 1.2571, 1.2571,
        # 0.8381, 0.8381, 0.8381, 0.8381, 0.8381, 1.2571, 0.8381, 1.2571, 1.2571,
        # 0.8381, 0.8381, 1.2571, 1.2571]).to('cuda') *gt_description * torch.log(L_output), 2))
        L_loss = torch.mean(-torch.sum(torch.tensor([0.1168, 1.4010, 1.4010, 1.4010, 1.4010, 0.4670, 1.4010, 1.4010, 1.4010,
        0.7005, 0.7005, 0.7005, 0.7005, 1.4010, 0.7005, 0.4670, 1.4010, 1.4010, 0.7005, 0.4670, 0.4670, 1.4010, 1.4010]).to('cuda')
                                      *gt_description * torch.log(L_output), 2))
        #L_loss = torch.mean(-torch.sum(torch.tensor([0.0051, 0.0609, 0.0609, 0.0609, 0.0609, 0.0203, 0.0609, 0.0609, 0.0609,
        #0.0305, 0.0305, 0.0305, 0.0305, 0.0609, 0.0305, 0.0203, 0.0609, 0.0609, 0.0305, 0.0203, 0.0203, 0.0609, 0.0609]).to('cuda')
        #                              *gt_description * torch.log(L_output)*(torch.ones(L_output.size()).to('cuda')-L_output)**2, 2))
        B_loss = torch.mean(torch.square(B_output - gt_action))  # action loss (MSE)
    loss = net_conf.L_weight * L_loss + net_conf.B_weight * B_loss
    return L_loss, B_loss, loss

# Gated Multimodal Unit (Arevalo et al., 2017) - this is not used as part of PVAE or PVAE-BERT; maybe used as an alternative to VAE
class GatedMultimodalUnit(nn.Module):
    def __init__(self, params, bert=False):
        super(GatedMultimodalUnit, self).__init__()
        self.params = params
        if bert:
            self.lang_h_linear = nn.Linear(768, self.params.hidden_dim)
        else:
            self.lang_h_linear = nn.Linear(self.params.L_num_units * self.params.L_num_layers * 2, self.params.hidden_dim)
        self.act_h_linear = nn.Linear(self.params.VB_num_units * self.params.VB_num_layers * 2, self.params.hidden_dim)
        self.tanh = nn.Tanh()
        if bert:
            self.z_linear = nn.Linear(self.params.VB_num_units * self.params.VB_num_layers * 2 +
                                      768, self.params.hidden_dim)
        else:
            self.z_linear = nn.Linear(self.params.VB_num_units * self.params.VB_num_layers * 2 +
                               self.params.L_num_units * self.params.L_num_layers * 2, self.params.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(p=0.1)

    def forward(self, act_features, lang_features):
        h_act = self.tanh(self.act_h_linear(act_features))
        h_lang = self.tanh(self.lang_h_linear(lang_features))
        z = self.sigmoid(self.z_linear(torch.cat([act_features, lang_features], dim=-1)))
        h = z * h_act + (1 - z) * h_lang
        return h

# Word Embedding Layer
class Embedder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
    def forward(self, x):
        return self.embed(x)

# Language Model-Based Language Encoder
class LanguageModel(nn.Module):
    def __init__(self, params, language_model='bert-base-uncased'):
        super(LanguageModel, self).__init__()
        self.params = params
        self.language_model = language_model
        if language_model == 'roberta':
            from transformers import RobertaTokenizer, RobertaModel
            self.tokeniser = RobertaTokenizer.from_pretrained('roberta-base')
            self.encoder = RobertaModel.from_pretrained('roberta-base')      # batch, seq, hidden
        elif language_model == 'distilbert':
            from transformers import DistilBertTokenizer, DistilBertModel
            self.tokeniser = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')      # batch, seq, hidden
        elif language_model == 'albert-base':
            from transformers import AlbertTokenizer, AlbertModel
            self.tokeniser = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')      # batch, seq, hidden
        elif language_model == 't5':
            from transformers import T5Tokenizer, T5Model
            self.tokeniser = T5Tokenizer.from_pretrained('t5-small')
            self.encoder = T5Model.from_pretrained('t5-small')      # batch, seq, hidden
        elif language_model == 'sentence-lm':
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')      # batch, seq, hidden
        elif language_model == 'sentence-t5':
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer.from_pretrained('sentence-transformers/sentence-t5-base')      # batch, seq, hidden
        else:
            from transformers import BertTokenizer, BertModel
            self.tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = BertModel.from_pretrained("bert-base-uncased")  # batch, seq, hidden
        # Uncomment below if no finetuning desired
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, inp):
        # Use the vocabulary to feed descriptions
        file = open('../vocabulary.txt', 'r')
        vocab = file.read().splitlines()
        file.close()
        t = inp[:, :, :].argmax(axis=-1)
        descriptions = []
        for i in range(inp.shape[1]):
            sentence = ''
            for k in range(0, inp.shape[0]):
                sentence += vocab[t[k, i]] + ' '
            descriptions.append(sentence.replace('<BOS/EOS>', '')[:-1])
            # Different versions of the description:
            #descriptions.append(sentence.split(' ')[-2] + ' ' + sentence.split(' ')[0] + ' ' + sentence.split(' ')[1])
            #descriptions.append('could you ' + sentence[:-1] + '?')
            #descriptions.append(sentence.split(' ')[0] + ' ' + sentence.split(' ')[-2] + ' ' + sentence.split(' ')[1] + ' ' + sentence.split(' ')[2] + ' ' + sentence.split(' ')[3] + ' ' + sentence.split(' ')[4])
            #descriptions.append(sentence.split(' ')[-2] + ' ' + sentence.split(' ')[1] + ' ' + sentence.split(' ')[2] + ' ' + sentence.split(' ')[3] + ' ' + sentence.split(' ')[4] + ' ' + sentence.split(' ')[0])
            #descriptions.append(sentence.split(' ')[1] + ' ' + sentence.split(' ')[2] + ' ' + sentence.split(' ')[3] + ' ' + sentence.split(' ')[4] +' ' + sentence.split(' ')[5] + ' ' + sentence.split(' ')[0])
        if self.language_model=='sentence-lm' or self.language_model=='sentence-t5':
            v=self.encoder.encode(descriptions)
        else:
            encoded_input = self.tokeniser(descriptions, return_tensors='pt', padding=True)
            output = self.encoder(**encoded_input.to(self.encoder.device))
            #v = self.mean_pooling(output, encoded_input['attention_mask'])     # use this when descriptions have different number of subwords to ignore padding
            v = output.last_hidden_state.mean(1)
        return v

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Encoder(nn.Module):
    def __init__(self, params, lang=False):
        super(Encoder, self).__init__()
        self.params = params
        if lang:
            self.enc_cells = torch.nn.Sequential()
            for i in range(self.params.L_num_layers):
                if i == 0:
                    self.enc_cells.add_module("ellstm"+str(i), PeepholeLSTM(input_size=self.params.L_input_dim+5,
                                                                            hidden_size=self.params.L_num_units,
                                                                            peephole=True, forget_bias=0.8))
                    #self.enc_cells.add_module("ellstm"+str(i), PeepholeLSTM(input_size=50,
                                                                            #hidden_size=self.params.L_num_units,
                                                                            #peephole=True, forget_bias=0.8))
                else:
                    self.enc_cells.add_module("ellstm"+str(i), PeepholeLSTM(input_size=self.params.L_num_units,
                                                                            hidden_size=self.params.L_num_units,
                                                                            peephole=True, forget_bias=0.8))
        else:
            self.enc_cells = torch.nn.Sequential()
            for i in range(self.params.VB_num_layers):
                if i == 0:
                    self.enc_cells.add_module("ealstm" + str(i), PeepholeLSTM(input_size=self.params.VB_input_dim,
                                                                              hidden_size=self.params.VB_num_units,
                                                                              peephole=True, forget_bias=0.8))
                else:
                    self.enc_cells.add_module("ealstm" + str(i), PeepholeLSTM(input_size=self.params.VB_num_units,
                                                                              hidden_size=self.params.VB_num_units,
                                                                              peephole=True, forget_bias=0.8))

    def forward(self, inp, sequence_length, lang=False):
        if lang:
            num_of_layers = self.params.L_num_layers
        else:
            num_of_layers = self.params.VB_num_layers
        layer_input = inp
        states = []
        for l in range(num_of_layers):
            enc_cell = self.enc_cells.__getitem__(l)
            hidden_seq, (cn, hn) = enc_cell(layer_input.float().to('cuda'), sequence_len=sequence_length)
            layer_input = hidden_seq
            states.append((cn, hn))
        states = tuple(map(torch.stack, zip(*states)))
        final_state = torch.stack(states, dim=1)    # n_layers, 2, batch_size, n_units
        final_state = final_state.permute(2,0,1,3)  # transpose to batchsize, n_layers, 2, n_units
        final_state = torch.reshape(final_state, (int(final_state.shape[0]), -1))
        return final_state#

class Decoder(nn.Module):
    def __init__(self, params, lang=False, vis_out=False):
        super(Decoder, self).__init__()
        self.params = params
        self.vis_out = vis_out
        if lang:
            self.dec_cells = torch.nn.Sequential()
            for i in range(self.params.L_num_layers):
                if i == 0:
                    self.dec_cells.add_module("dllstm"+str(i), PeepholeLSTM(self.params.L_input_dim,
                                                                            self.params.L_num_units, True, forget_bias=0.8).to('cuda'))
                else:
                    self.dec_cells.add_module("dllstm"+str(i), PeepholeLSTM(self.params.L_num_units,
                                                                            self.params.L_num_units, True, forget_bias=0.8).to('cuda'))
            self.linear = nn.Linear(self.params.L_num_units, self.params.L_input_dim)
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.dec_cells = torch.nn.Sequential()
            for i in range(self.params.VB_num_layers):
                if i == 0:
                    self.dec_cells.add_module("dalstm"+str(i), PeepholeLSTM(input_size=self.params.VB_input_dim,
                                                                            hidden_size=self.params.VB_num_units,
                                                                            peephole=True, forget_bias=0.8).to('cuda'))
                else:
                    self.dec_cells.add_module("dalstm"+str(i), PeepholeLSTM(input_size=self.params.VB_num_units,
                                                                            hidden_size=self.params.VB_num_units,
                                                                            peephole=True, forget_bias=0.8).to('cuda'))
            if vis_out:
                self.linear = nn.Linear(self.params.VB_num_units, self.params.VB_input_dim)
            else:
                self.linear = nn.Linear(self.params.VB_num_units, self.params.B_input_dim)
            self.tanh = nn.Tanh()

    def forward(self, input, length, initial_state=None, lang=False, teacher_forcing=True):
        y = []

        if lang:
            initial_state = initial_state.view(initial_state.size()[0], self.params.L_num_layers, 2, self.params.L_num_units)
            initial_state = initial_state.permute(1, 2, 0, 3)
            for i in range(length - 1):
                dec_states = []
                layer_input = input.unsqueeze(0)
                if i == 0:
                    for j in range(self.params.L_num_layers):
                        dec_cell = self.dec_cells.__getitem__(j)
                        dec_state = (initial_state[j][0].float().to('cuda'), initial_state[j][1].float().to('cuda'))
                        output, (cx, hx) = dec_cell(layer_input.float().to('cuda'), init_states=dec_state)
                        dec_state = (cx, hx)
                        dec_states.append(dec_state)
                        layer_input = output
                else:
                    layer_input = out
                    for j in range(self.params.L_num_layers):
                        dec_cell = self.dec_cells.__getitem__(j)
                        dec_state = prev_dec_states[j]
                        output, (cx, hx) = dec_cell(layer_input, init_states=dec_state)
                        dec_state = (cx, hx)
                        dec_states.append(dec_state)
                        layer_input = output
                prev_dec_states = dec_states
                linear = self.linear(layer_input)
                out = self.softmax(linear)
                y.append(out.squeeze())
        else:
            initial_state = initial_state.view(initial_state.size()[0], self.params.VB_num_layers, 2, self.params.VB_num_units)
            initial_state = initial_state.permute(1, 2, 0, 3)
            for i in range(length - 1):
                if self.vis_out == False or teacher_forcing == True:
                    current_V_in = input[0][i]
                dec_states = []
                if i == 0:
                    if self.vis_out and teacher_forcing == False:
                        current_V_in = input[0]
                    current_B_in = input[-1]
                    layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)
                    for j in range(self.params.VB_num_layers):
                        dec_state = (initial_state[j][0].float(), initial_state[j][1].float())
                        dec_cell = self.dec_cells.__getitem__(j)
                        output, (cx, hx) = dec_cell(layer_input.float(), init_states=dec_state)
                        dec_state = (cx, hx)
                        dec_states.append(dec_state)
                        layer_input = output
                else:
                    if self.vis_out and teacher_forcing == False:
                        layer_input = out
                    else:
                        if self.vis_out and teacher_forcing == True:
                            current_B_in = out[:,:,30:].squeeze(dim=0)
                        else:
                            current_B_in = out.squeeze(dim=0)
                        layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)
                    for j in range(self.params.VB_num_layers):
                        dec_cell = self.dec_cells.__getitem__(j)
                        dec_state = prev_dec_states[j]
                        output, (cx, hx) = dec_cell(layer_input.float(), init_states=dec_state)
                        dec_state = (cx, hx)
                        dec_states.append(dec_state)
                        layer_input = output
                prev_dec_states = dec_states
                linear = self.linear(layer_input)
                out = self.tanh(linear)
                y.append(out.squeeze())
        y = torch.stack(y, dim=0)
        return y


class PGAE(nn.Module):
    def __init__(self, params):
        super(PGAE, self).__init__()
        self.params = params

        self.lang_encoder = Encoder(self.params, True)
        self.action_encoder = Encoder(self.params, False)

        self.hidden = GatedMultimodalUnit(self.params)

        self.initial_lang = nn.Linear(self.params.hidden_dim, self.params.L_num_units*self.params.L_num_layers*2)
        self.initial_act = nn.Linear(self.params.hidden_dim, self.params.VB_num_units*self.params.VB_num_layers*2)

        self.lang_decoder = Decoder(self.params, True)
        self.action_decoder = Decoder(self.params, False, False)

    def forward(self, inp, signal):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim+4] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim+2] = 1.0
            #l_fw_ndim = torch.zeros((inp['L_fw'].size()[0], inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5)).to('cuda')
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:, self.params.L_input_dim] = 1.0
            #l_fw_ndim = torch.zeros((inp['L_fw'].size()[0], inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5)).to('cuda')
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim+1] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input=torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) *\
                     inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        z = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(z)
        VB_dec_init_state = self.initial_lang(z)
        if signal == 'repeat both':
            VB_input_f = inp['VB_fw']
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state, teacher_forcing=True)
        elif signal == 'describe':
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']),1,1), inp["B_bw"][0, :, :]]
            #VB_input_f = [inp["V_bw"][0, :, :], inp["B_bw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state, teacher_forcing=True)
        elif signal == 'repeat language':
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']),1,1), inp["B_fw"][0, :, :]]
            #VB_input_f = [inp["V_fw"][0, :, :], inp["B_fw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state, teacher_forcing=True)
        else:
            VB_input_f = inp['VB_fw']
            L_output = self.lang_decoder(inp['L_fw'][0], 2, L_dec_init_state, True)
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state, teacher_forcing=True)
        return L_output, B_output

    def extract_representations(self, inp, signal):
        if signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:, self.params.L_input_dim] = 1.0
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        elif signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim+4] = 1.0
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim+2] = 1.0
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim + 1] = 1.0
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)

        h = self.hidden(encoded_act, encoded_lang)
        return h

    def language_to_action(self, inp):
        l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to(
            'cuda')
        signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
        signalrow[0, :, self.params.L_input_dim + 1] = 1.0
        # VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
        VB_input = torch.cat(
            (inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                   inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
        lang_in_length = inp['L_len'].int().numpy()

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())
        VB_input_f = inp['VB_fw']

        h = self.hidden(encoded_act, encoded_lang)
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        return B_output

    def action_to_language(self, inp):
        signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to('cuda')
        VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
        signalrow[0, :, self.params.L_input_dim] = 1.0
        # l_fw_ndim = torch.zeros((inp['L_fw'].size()[0], inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5)).to('cuda')
        l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to('cuda')
        lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(h)
        L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])
        return one_hot

    def reproduce_actions(self, inp):

        signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to('cuda')
        VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
        signalrow[0, :, self.params.L_input_dim + 2] = 1.0

        l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to('cuda')
        lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)
        VB_input_f = inp['VB_fw']
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)

        return B_output

    def reproduce_lang(self, inp):
        l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to(
            'cuda')
        signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
        signalrow[0, :, self.params.L_input_dim + 4] = 1.0

        VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                   inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
        lang_in_length = inp['L_len'].int().numpy()

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(h)
        L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot

    def inference(self, inp, signal):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim + 4] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_fw"][0, :, :]]
            #VB_input_f = [inp["V_fw"][0, :, :], inp["B_fw"][0, :, :]]
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim + 2] = 1.0
            #l_fw_ndim = torch.zeros((inp['L_fw'].size()[0], inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5)).to('cuda')
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:, self.params.L_input_dim] = 1.0
            #l_fw_ndim = torch.zeros((inp['L_fw'].size()[0], inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5)).to('cuda')
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
            VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
            #VB_input_f = [inp["V_bw"][0, :, :], inp["B_bw"][0, :, :]]
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim + 1] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy()

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)


        h = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(h)
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)
        if signal == 'execute' or signal == 'repeat action':
            L_output = self.lang_decoder(inp['L_fw'][0], 2, L_dec_init_state, True)
        else:
            L_output = self.lang_decoder(inp['L_fw'][0], int(inp['L_len'].item()), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])
        return one_hot, B_output

class PGAEBERT(nn.Module):
    def __init__(self, params):
        super(PGAEBERT, self).__init__()
        self.params = params

        self.lang_encoder = LanguageModel(self.params, 'bert-base')
        self.action_encoder = Encoder(self.params, False)

        self.hidden = GatedMultimodalUnit(self.params, bert=True)

        self.initial_lang = nn.Linear(self.params.hidden_dim, self.params.L_num_units*self.params.L_num_layers*2)
        self.initial_act = nn.Linear(self.params.hidden_dim, self.params.VB_num_units*self.params.VB_num_layers*2)

        self.lang_decoder = Decoder(self.params, True)
        self.action_decoder = Decoder(self.params, False)

    def forward(self, inp, signal):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim + 4] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim + 2] = 1.0
            #l_fw_ndim = torch.zeros((inp['L_fw'].size()[0], inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5)).to('cuda')
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:, self.params.L_input_dim] = 1.0
            #l_fw_ndim = torch.zeros((inp['L_fw'].size()[0], inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5)).to('cuda')
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim + 1] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input=torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) *\
                     inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)
        if signal == 'repeat both':
            VB_input_f = inp['VB_fw']
            L_dec_init_state = self.initial_lang(h)
            VB_dec_init_state = self.initial_act(h)
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        elif signal == 'describe':
            L_dec_init_state = self.initial_lang(h)
            VB_dec_init_state = self.initial_act(h)
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']),1,1), inp["B_bw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        elif signal == 'repeat language':
            L_dec_init_state = self.initial_lang(h)
            VB_dec_init_state = self.initial_act(h)
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']),1,1), inp["B_fw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        else:
            VB_input_f = inp['VB_fw']
            VB_dec_init_state = self.initial_act(h)
            L_dec_init_state = self.initial_lang(h)
            L_output = self.lang_decoder(inp['L_fw'][0], 2, L_dec_init_state, True)
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        return L_output, B_output

    def extract_representations(self, inp, signal):
        if signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:, self.params.L_input_dim] = 1.0
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
        elif signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim+4] = 1.0
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim+2] = 1.0
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim + 1] = 1.0
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)

        h = self.hidden(encoded_act, encoded_lang)
        return h

    def language_to_action(self, inp):
        l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to('cuda')
        signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
        signalrow[0, :, self.params.L_input_dim + 1] = 1.0
        # VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
        VB_input = torch.cat(
            (inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                   inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())
        VB_input_f = inp['VB_fw']

        h = self.hidden(encoded_act, encoded_lang)
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        return B_output

    def action_to_language(self, inp):
        signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to('cuda')
        VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
        signalrow[0, :, self.params.L_input_dim] = 1.0

        l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(h)
        L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])
        return one_hot

    def reproduce_actions(self, inp):
        signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to('cuda')
        VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
        signalrow[0, :, self.params.L_input_dim + 2] = 1.0

        l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to('cuda')
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        encoded_lang = self.lang_encoder(lang_inp)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)
        VB_input_f = inp['VB_fw']
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)

        return B_output

    def reproduce_lang(self, inp):
        l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to(
            'cuda')
        signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
        signalrow[0, :, self.params.L_input_dim + 4] = 1.0

        VB_input = torch.cat(
            (inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                   inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(h)
        L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot

    def inference(self, inp, signal):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            VB_input_f = inp['VB_fw']
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim + 4] = 1.0

            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_fw"][0, :, :]]
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim + 2] = 1.0

            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
            VB_input_f = inp['VB_fw']
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:, self.params.L_input_dim] = 1.0

            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
            VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim + 1] = 1.0

            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = inp['VB_fw']

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)

        h = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(h)
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)
        if signal=='execute' or signal == 'repeat action':
            L_output = self.lang_decoder(inp['L_fw'][0], 2, L_dec_init_state, True)
        else:
            L_output = self.lang_decoder(inp['L_fw'][0], int(inp['L_len'].item()), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])
        return one_hot, B_output

class PGAEOPP(nn.Module):
    def __init__(self, params):
        super(PGAEOPP, self).__init__()
        self.params = params

        self.lang_encoder = Encoder(self.params, True)
        self.action_encoder = Encoder(self.params, False)

        self.hidden = GatedMultimodalUnit(self.params)

        self.initial_lang = nn.Linear(self.params.hidden_dim, self.params.L_num_units*self.params.L_num_layers*2)
        self.initial_act = nn.Linear(self.params.hidden_dim, self.params.VB_num_units*self.params.VB_num_layers*2)

        self.lang_decoder = Decoder(self.params, True)
        self.action_decoder = Decoder(self.params, False)

    def forward(self, inp, signal, opp):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            lang_in_length = inp['L_len'].int().numpy()
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)*inp["B_bin"]], dim=2)
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim+4] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim+2] = 1.0
            #l_fw_ndim = torch.zeros((inp['L_fw'].size()[0], inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5)).to('cuda')
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]], dim=2)
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim] = 1.0
            #l_fw_ndim = torch.zeros((inp['L_fw'].size()[0], inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5)).to('cuda')
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]], dim=2)
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim+1] = 1.0

            VB_input=torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) *\
                     inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp, lang_in_length+1, True)

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        z = self.hidden(encoded_act, encoded_lang)
        if signal == 'repeat both':
            VB_input_f = inp['VB_fw']
            L_dec_init_state = self.initial_lang(z)
            VB_dec_init_state = self.initial_act(z)
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        elif signal == 'describe':
            L_dec_init_state = self.initial_lang(z)
            VB_dec_init_state = self.initial_act(z)
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            if opp == 0:
                VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
            else:
                VB_input_f = [inp["V_opp_bw"][0].repeat(len(inp['V_opp_bw']), 1, 1), inp["B_fw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        elif signal == 'repeat language':
            L_dec_init_state = self.initial_lang(z)
            VB_dec_init_state = self.initial_act(z)
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']),1,1), inp["B_fw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        else:
            VB_input_f = inp['VB_fw']
            VB_dec_init_state = self.initial_act(z)
            L_dec_init_state = self.initial_lang(z)
            L_output = self.lang_decoder(inp['L_fw'][0], 2, L_dec_init_state, True)
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        return L_output, B_output

    def extract_representations(self, inp, signal, opp=0):
        if signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to(
                'cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
            signalrow[0, :, self.params.L_input_dim] = 1.0
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        elif signal == 'repeat both':
            l_fw_ndim = torch.cat(
                (inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat(
                (inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim + 4] = 1.0
            VB_input = torch.cat(
                (inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to(
                'cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
            signalrow[0, :, self.params.L_input_dim + 2] = 1.0
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        else:
            l_fw_ndim = torch.cat(
                (inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim + 1] = 1.0
            VB_input = torch.cat(
                (inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)

        h = self.hidden(encoded_act, encoded_lang)
        return h

    def language_to_action(self, inp):
        l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to(
            'cuda')
        signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
        signalrow[0, :, self.params.L_input_dim + 1] = 1.0
        # VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
        VB_input = torch.cat(
            (inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                   inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
        lang_in_length = inp['L_len'].int().numpy()

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())
        VB_input_f = inp['VB_fw']

        h = self.hidden(encoded_act, encoded_lang)
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        return B_output

    def action_to_language(self, inp, opp):
        signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to('cuda')
        if opp == 0:
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
        else:
            VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]], dim=2)
        signalrow[0, :, self.params.L_input_dim] = 1.0
        # l_fw_ndim = torch.zeros((inp['L_fw'].size()[0], inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5)).to('cuda')
        l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to('cuda')
        lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(h)
        L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])
        return one_hot

    def reproduce_actions(self, inp, opp):
        signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to('cuda')
        if opp == 0:
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
        else:
            VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]], dim=2)
        signalrow[0, :, self.params.L_input_dim + 2] = 1.0

        l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to('cuda')
        lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)
        VB_input_f = inp['VB_fw']
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)

        return B_output

    def reproduce_lang(self, inp):
        l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to('cuda')
        signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
        signalrow[0, :, self.params.L_input_dim + 4] = 1.0

        VB_input = torch.cat(
            (inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                   inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
        lang_in_length = inp['L_len'].int().numpy()

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(h)
        L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot

    def inference(self, inp, signal, opp):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)*inp["B_bin"]], dim=2)
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim + 4] = 1.0

            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_fw"][0, :, :]]
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim + 2] = 1.0

            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]], dim=2)
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim] = 1.0

            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
                VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]], dim=2)
                VB_input_f = [inp["V_opp_bw"][0].repeat(len(inp['V_opp_bw']), 1, 1), inp["B_fw"][0, :, :]]
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim + 1] = 1.0

            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy()

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())


        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)

        h = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(h)
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)
        if signal=='execute' or signal == 'repeat action':
            L_output = self.lang_decoder(inp['L_fw'][0], 2, L_dec_init_state, True)
        else:
            L_output = self.lang_decoder(inp['L_fw'][0], int(inp['L_len'].item()), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])
        return one_hot, B_output

class PGAEBERTOPP(nn.Module):
    def __init__(self, params):
        super(PGAEBERTOPP, self).__init__()
        self.params = params

        self.lang_encoder = LanguageModel(self.params, 'bert-base')
        self.action_encoder = Encoder(self.params, False)

        self.hidden = GatedMultimodalUnit(self.params, bert=True)

        self.initial_lang = nn.Linear(self.params.hidden_dim, self.params.L_num_units*self.params.L_num_layers*2)
        self.initial_act = nn.Linear(self.params.hidden_dim, self.params.VB_num_units*self.params.VB_num_layers*2)

        self.lang_decoder = Decoder(self.params, True)
        self.action_decoder = Decoder(self.params, False)

    def forward(self, inp, signal, opp):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim + 4] = 1.0

            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
            signalrow[0,:,self.params.L_input_dim + 2] = 1.0

            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
            signalrow[0,:, self.params.L_input_dim] = 1.0

            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim + 1] = 1.0

            VB_input=torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) *\
                     inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        z = self.hidden(encoded_act, encoded_lang)
        if signal == 'repeat both':
            VB_input_f = inp['VB_fw']
            L_dec_init_state = self.initial_lang(z)
            VB_dec_init_state = self.initial_act(z)
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        elif signal == 'describe':
            L_dec_init_state = self.initial_lang(z)
            VB_dec_init_state = self.initial_act(z)
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            if opp == 0:
                VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
            else:
                VB_input_f = [inp["V_opp_bw"][0].repeat(len(inp['V_opp_bw']), 1, 1), inp["B_fw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        elif signal == 'repeat language':
            L_dec_init_state = self.initial_lang(z)
            VB_dec_init_state = self.initial_act(z)
            L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']),1,1), inp["B_fw"][0, :, :]]
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        else:
            VB_input_f = inp['VB_fw']
            VB_dec_init_state = self.initial_act(z)
            L_dec_init_state = self.initial_lang(z)
            L_output = self.lang_decoder(inp['L_fw'][0], 2, L_dec_init_state, True)
            B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        return L_output, B_output

    def extract_representations(self, inp, signal, opp):
        if signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to(
                'cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
            signalrow[0, :, self.params.L_input_dim] = 1.0
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
        elif signal == 'repeat both':
            l_fw_ndim = torch.cat(
                (inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat(
                (inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim + 4] = 1.0
            VB_input = torch.cat(
                (inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to(
                'cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
            signalrow[0, :, self.params.L_input_dim + 2] = 1.0
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
        else:
            l_fw_ndim = torch.cat(
                (inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim + 1] = 1.0
            VB_input = torch.cat(
                (inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)

        h = self.hidden(encoded_act, encoded_lang)
        return h

    def language_to_action(self, inp):
        l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to('cuda')
        signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
        signalrow[0, :, self.params.L_input_dim + 1] = 1.0
        # VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
        VB_input = torch.cat(
            (inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                   inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())
        VB_input_f = inp['VB_fw']

        h = self.hidden(encoded_act, encoded_lang)
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
        return B_output

    def action_to_language(self, inp, opp):
        signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to('cuda')
        if opp == 0:
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
        else:
            VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]], dim=2)
        signalrow[0, :, self.params.L_input_dim] = 1.0

        l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),axis=-1).to('cuda')
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(h)
        L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])
        return one_hot

    def reproduce_actions(self, inp, opp):
        signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to('cuda')
        if opp == 0:
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
        else:
            VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]], dim=2)
        signalrow[0, :, self.params.L_input_dim + 2] = 1.0

        l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to('cuda')

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)
        VB_input_f = inp['VB_fw']
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)

        return B_output

    def reproduce_lang(self, inp):
        l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to(
            'cuda')
        signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
        signalrow[0, :, self.params.L_input_dim + 4] = 1.0

        VB_input = torch.cat(
            (inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                   inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)
        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())

        h = self.hidden(encoded_act, encoded_lang)

        L_dec_init_state = self.initial_lang(h)
        L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot

    def inference(self, inp, signal, opp):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            VB_input_f = inp['VB_fw']
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim + 4] = 1.0

            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_fw"][0, :, :]]
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
            signalrow[0,:,self.params.L_input_dim + 2] = 1.0

            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
            VB_input_f = inp['VB_fw']
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim] = 1.0

            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                                  axis=-1).to('cuda')
            if opp == 0:
                VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
                VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
            else:
                VB_input = torch.cat([inp['V_opp_fw'], inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1) * inp["B_bin"]],
                                     dim=2)
                VB_input_f = [inp["V_opp_bw"][0].repeat(len(inp['V_opp_bw']), 1, 1), inp["B_fw"][0, :, :]]
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim + 1] = 1.0

            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = inp['VB_fw']

        encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())


        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')
        encoded_lang = self.lang_encoder(lang_inp)

        h = self.hidden(encoded_act, encoded_lang)
        L_dec_init_state = self.initial_lang(h)
        VB_dec_init_state = self.initial_act(h)
        B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)
        if signal=='execute' or signal == 'repeat action':
            L_output = self.lang_decoder(inp['L_fw'][0], 2, L_dec_init_state, True)
        else:
            L_output = self.lang_decoder(inp['L_fw'][0], int(inp['L_len'].item()), L_dec_init_state, True)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])
        return one_hot, B_output

