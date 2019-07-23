import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertConfig


class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=6, hidden_size=768, num_layers=3, dropout=0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size)
                , nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        for i in range(self.num_layers-1):
            x = F.relu(self.hidden_layers[i](x))
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob


class DomainQA(nn.Module):
    def __init__(self, bert_name_or_config, num_classes=6, hidden_size=768,
                 num_layers=3, dropout=0.1, dis_lambda=0.5):
        super(DomainQA, self).__init__()
        if isinstance(bert_name_or_config, BertConfig):
            self.bert = BertModel(bert_name_or_config)
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.qa_outputs = nn.Linear(hidden_size, 2)
        # init weight
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()

        self.discriminator = DomainDiscriminator(num_classes, hidden_size, num_layers, dropout)
        self.num_classes = num_classes
        self.dis_lambda = dis_lambda

    # only for prediction
    def forward(self, input_ids, token_type_ids, attention_mask,
                start_positions=None, end_positions=None, labels=None, dtype=None):
        if dtype == "qa":
            qa_loss = self.forward_qa(input_ids, token_type_ids, attention_mask, start_positions, end_positions)
            return qa_loss

        elif dtype == "dis":
            assert labels is not None
            dis_loss = self.forward_discriminator(input_ids, token_type_ids, attention_mask, labels)
            return dis_loss

        else:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            return start_logits, end_logits

    def forward_qa(self, input_ids, token_type_ids, attention_mask, start_positions, end_positions):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        hidden = sequence_output[:, 0]  # [b, d] : [CLS] representation
        log_prob = self.discriminator(hidden)
        targets = torch.ones_like(log_prob) * (1 / self.num_classes)
        # As with NLLLoss, the input given is expected to contain log-probabilities
        # and is not restricted to a 2D Tensor. The targets are given as probabilities
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        kld = self.dis_lambda * kl_criterion(log_prob, targets)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        qa_loss = (start_loss + end_loss) / 2
        total_loss = qa_loss + kld
        return total_loss

    def forward_discriminator(self, input_ids, token_type_ids, attention_mask, labels):
        with torch.no_grad():
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            hidden = sequence_output[:, 0]  # [b, d] : [CLS] representation
        log_prob = self.discriminator(hidden.detach())
        criterion = nn.NLLLoss()
        loss = criterion(log_prob, labels)

        return loss
