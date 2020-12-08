import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoCtxTransformer(nn.Module):
    def __init__(self, depth=4, model_dim=512, expansion=4, dropout=0.1):
        super().__init__()
        self.transformer = nn.Sequential(*[
            NoCtxTransformerLayer(
                model_dim=model_dim,
                expansion=expansion,
                dropout=dropout,)
            for _ in range(depth)]
        )

    def forward(self, hidden_states):
        output_states = self.transformer(hidden_states)
        return output_states


class SimpleTransformer(nn.Module):
    def __init__(self, depth=4, model_dim=512, expansion=4, attention_heads=8, multihead=True, attention_type='self', dropout=0.1):
        super().__init__()
        assert attention_type in ('self', 'joint')
        self.transformer = nn.Sequential(*[
            TransformerLayer(
                model_dim=model_dim,
                expansion=expansion,
                attention_heads=attention_heads,
                self_attention=True,
                multihead=multihead,
                dropout=dropout,)
            for _ in range(depth)]
        )

    def forward(self, hidden_states):
        output_states = self.transformer(hidden_states)
        return output_states


class DualTransformer(nn.Module):
    def __init__(self, depth=2, model_dim=512, expansion=4, attention_heads=8, multihead=True, attention_type='self', dropout=0.1):
        super().__init__()
        self_attention = attention_type in ('self', 'joint')
        self.transformers = nn.ModuleList([
            nn.ModuleList([
                TransformerLayer(
                    model_dim=model_dim,
                    expansion=expansion,
                    attention_heads=attention_heads,
                    self_attention=self_attention,
                    multihead=multihead,
                    dropout=dropout,)
                for _ in range(2)])
            for _ in range(depth)]
        )

    def forward(self, video_states, audio_states):
        for vt, at in self.transformers:
            video_states_n = vt(video_states, audio_states)
            audio_states_n = at(audio_states, video_states)
            video_states, audio_states = video_states_n, audio_states_n
        return video_states, audio_states


class NoCtxTransformerLayer(nn.Module):
    def __init__(self, model_dim=512, expansion=4, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(model_dim, model_dim*expansion),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim*expansion, model_dim),
        )
        self.dense1_dropout = nn.Dropout(dropout, inplace=True)
        self.dense1_norm = nn.LayerNorm(model_dim)

        self.dense2 = nn.Sequential(
            nn.Linear(model_dim, model_dim*expansion),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim*expansion, model_dim),
        )
        self.dense2_dropout = nn.Dropout(dropout, inplace=True)
        self.dense2_norm = nn.LayerNorm(model_dim)
        self.out_activ = nn.ReLU(inplace=True)

    def forward(self, hidden_states, other_hidden_states=None):
        dense1_output = self.dense1(hidden_states)
        dense1_output = self.dense1_dropout(dense1_output)
        dense1_output = self.dense1_norm(dense1_output + hidden_states)
        dense1_output = self.out_activ(dense1_output)

        dense2_output = self.dense2(dense1_output)
        dense2_output = self.dense2_dropout(dense2_output)
        dense2_output = self.dense2_norm(dense2_output + dense1_output)
        dense2_output = self.out_activ(dense2_output)
        return dense2_output


class TransformerLayer(nn.Module):
    def __init__(self, model_dim=512, expansion=4, attention_heads=8, multihead=True, self_attention=True, dropout=0.1):
        super().__init__()
        if multihead:
            self.attention = MultiHeadAttention(model_dim=model_dim, attention_heads=attention_heads, self_attention=self_attention, dropout=dropout)
        else:
            self.attention = SimplifiedAttention(model_dim=model_dim, self_attention=self_attention)
        self.attention_dropout = nn.Dropout(dropout, inplace=True)
        self.attention_norm = nn.LayerNorm(model_dim)

        self.dense = nn.Sequential(
            nn.Linear(model_dim, model_dim*expansion),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim*expansion, model_dim),
        )
        self.dense_dropout = nn.Dropout(dropout, inplace=True)
        self.dense_norm = nn.LayerNorm(model_dim)
        self.out_activ = nn.ReLU(inplace=True)

    def forward(self, hidden_states, other_hidden_states=None):
        attention_output = self.attention(hidden_states, other_hidden_states=other_hidden_states)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_norm(attention_output + hidden_states)

        dense_output = self.dense(attention_output)
        dense_output = self.dense_dropout(dense_output)
        dense_output = self.dense_norm(dense_output + attention_output)
        return self.out_activ(dense_output)


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, attention_heads=8, self_attention=True, dropout=0.1):
        super().__init__()
        if model_dim % attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (model_dim, attention_heads)
            )
        self.self_attention = self_attention
        self.attention_heads = attention_heads
        self.attention_head_size = int(model_dim / attention_heads)
        self.all_head_size = self.attention_heads * self.attention_head_size

        self.query = nn.Linear(model_dim, self.all_head_size)
        self.key = nn.Linear(model_dim, self.all_head_size)
        self.value = nn.Linear(model_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(model_dim, model_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, other_hidden_states=None):
        if self.self_attention:
            query_all_heads = self.query(hidden_states)
        else:
            query_all_heads = self.query(other_hidden_states)
        key_all_heads = self.key(hidden_states)
        value_all_heads = self.value(hidden_states)

        queries = self.transpose_for_scores(query_all_heads)
        keys = self.transpose_for_scores(key_all_heads)
        values = self.transpose_for_scores(value_all_heads)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(*new_context_layer_shape)

        return self.dense(context_layer)


class SimplifiedAttention(nn.Module):
    def __init__(self, model_dim=512, self_attention=True):
        super().__init__()
        self.self_attention = self_attention
        self.model_dim = model_dim

        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)

        self.dense = nn.Linear(model_dim, model_dim)

    def forward(self, hidden_states, other_hidden_states=None):
        if self.self_attention:
            queries = self.query(hidden_states)
        else:
            queries = self.query(other_hidden_states)
        keys = self.key(hidden_states)
        values = self.value(hidden_states)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.model_dim)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, values)

        return self.dense(context_layer)