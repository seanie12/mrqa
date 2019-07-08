import math

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", bert_config=None, pretrained=True):
        super(FeatureExtractor, self).__init__()
        if pretrained:
            self.bert = BertModel.from_pretrained(bert_model)
        else:
            assert bert_config is not None, print("bert config file should be given")
            config = BertConfig(bert_config)
            self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       output_all_encoded_layers=False)
        return sequence_output


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, features, start_positions, end_positions):
        # features : [b,t,d]
        logits = self.linear(features)
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
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)
        # start_positions.clamp_(0, ignored_index)
        # end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss


class Critic(nn.Module):
    def __init__(self, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    @staticmethod
    def gelu(x):
        """Implementation of the gelu activation function.
            For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, feature):
        # feature : [b,t,d]
        # extract [CLS]
        pooled = feature[:, 0]
        # output should be scalar and non-negative value
        hidden = self.gelu(self.fc1(pooled))
        hidden = F.softplus(self.fc2(hidden))
        score = torch.mean(hidden)

        return score


class DGLearner(nn.Module):
    def __init__(self, bert_config):
        super(DGLearner, self).__init__()
        self.feat_ext = FeatureExtractor()
        self.classifier = Classifier(hidden_size=768)
        self.temp_old_feature_extractor_network = FeatureExtractor(bert_config=bert_config,
                                                                   pretrained=False)
        self.temp_new_feature_extractor_network = FeatureExtractor(bert_config=bert_config,
                                                                   pretrained=False)
        self.critic = Critic(hidden_size=768) 

    def forward(self, train_batch, test_batch, lr, theta_opt, phi_opt):
        input_ids, input_mask, seg_ids, start_positions, end_positions = train_batch
        features = self.feat_ext(input_ids, seg_ids, input_mask)
        loss_main = self.classifier(features, start_positions, end_positions)
        theta_opt.zero_grad()
        phi_opt.zero_grad()
        loss_main.backward(retain_graph=True)

        loss_dg = self.critic(features)
        grad_theta = [theta_i.grad for theta_i in self.feat_ext.parameters()]
        theta_updated_old = dict()

        num_grad = 0
        for i, (k, v) in enumerate(self.feat_ext.state_dict().items()):
            if grad_theta[num_grad] is None:
                num_grad += 1
                theta_updated_old[k] = v
            else:
                theta_updated_old[k] = v - lr * grad_theta[num_grad]
                num_grad += 1
        loss_dg.backward(create_graph=True)

        grad_theta = [theta_i.grad for theta_i in self.feat_ext.parameters()]
        theta_updated_new = {}
        num_grad = 0
        for i, (k, v) in enumerate(self.feat_ext.state_dict().items()):

            if grad_theta[num_grad] is None:
                num_grad += 1
                theta_updated_new[k] = v
            else:
                theta_updated_new[k] = v - lr * grad_theta[num_grad]
                num_grad += 1

        self.fix_nn(self.temp_new_feature_extractor_network, theta_updated_new)
        self.temp_new_feature_extractor_network.train()

        self.temp_old_feature_extractor_network.load_state_dict(theta_updated_old)
        self.temp_old_feature_extractor_network.train()
        input_ids, input_mask, seg_ids, start_positions, end_positions = test_batch
        with torch.no_grad():
            old_features = self.temp_old_feature_extractor_network(input_ids, seg_ids, input_mask)
        old_features = old_features.detach()

        new_features = self.temp_new_feature_extractor_network(input_ids, seg_ids, input_mask)

        loss_main_old = self.classifier(old_features, start_positions, end_positions)
        loss_main_new = self.classifier(new_features, start_positions, end_positions)

        reward = loss_main_old - loss_main_new
        utility = torch.tanh(reward)
        loss_held_out = - utility.sum()

        return loss_main, loss_dg, loss_held_out

    @staticmethod
    def fix_nn(model, theta):
        def k_param_fn(tmp_model, name=None):
            if len(tmp_model._modules) != 0:
                for (k, v) in tmp_model._modules.items():
                    if name is None:
                        k_param_fn(v, name=str(k))
                    else:
                        k_param_fn(v, name=str(name + '.' + k))
            else:
                for (k, v) in tmp_model._parameters.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    tmp_model._parameters[k] = theta[str(name + '.' + k)]

        k_param_fn(model)
        return model
