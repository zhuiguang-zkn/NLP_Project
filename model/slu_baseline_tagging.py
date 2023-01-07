# coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class SLUTagging(nn.Module):
    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(
            config.embed_size,
            config.hidden_size // 2,
            num_layers=config.num_layer,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.word_embed(input_ids)
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        """ Trains on one batch.

        Args:
            label_vocab (LabelVocab): the vocabulary
            batch (Batch): the batch for training

        Returns:
            predictions (list[ list[str] ]): each of its items is a list of tags of one sentence in the batch
            labels (list[ list[str] ]): each of its items is a list of slot-value pairs of one sentence in the batch
            loss (tensor): the loss on this batch (cross entropy loss computed by TaggingFNNDecoder.forward)
        """        
        batch_size = len(batch)
        labels = batch.labels  # list of semantics of each sentence
        prob, loss = self.forward(batch)  # compute the prob of each tag for each word in each sentence
        predictions = []
        for i in range(batch_size):  
            # process the i-th sentence in the batch
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()  # the most likely tags for the words
            # `pred_tuple` contains all slot-value pairs in this sentence.
            pred_tuple = []
            # For each slot-value pair, `idx_buff` and `tag_buff` contains 
            # positions and tags of corresponding words.
            # For this sentence, `pred_tags` contains all tags.
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[: len(batch.utt[i])]  # discard the padding
            for idx, tid in enumerate(pred):  
                # process one word in the sentence
                tag = label_vocab.convert_idx_to_tag(tid)  # the most likely tag for this word
                pred_tags.append(tag)
                if (tag == "O" or tag.startswith("B")) and len(tag_buff) > 0:
                    # add one slot-value pair since it has been collected
                    slot = "-".join(tag_buff[0].split("-")[1:])
                    value = "".join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f"{slot}-{value}")
                    if tag.startswith("B"):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith("I") or tag.startswith("B"):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                # add the last slot-value pair
                slot = "-".join(tag_buff[0].split("-")[1:])
                value = "".join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f"{slot}-{value}")
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):
    """ Contains a fully connected layer and a softmax layer.
    Input: a batch of hidden vectors of RNN, 
    whose shape is 'size of batch' x 'length of the longest sequence' x 'dim of hidden vector'.
    Output: a batch of probabilities of tags, 
    whose shape is 'size of batch' x 'length of the longest sequence' x 'num of tags`.
    """    
    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob
