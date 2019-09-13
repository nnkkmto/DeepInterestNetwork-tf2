
from collections import OrderedDict

import tensorflow as tf


class AttentionBlock(tf.keras.Model):

    def __init__(self):

        super(AttentionBlock, self).__init__(self)

        self.dense_1 = tf.keras.layers.Dense(80, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(40, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        """
        query(inputs[0]): candidateアイテムのembedding
        keys(inputs[1]): Embeddingした行動系列
        behavior_input(inputs[2]): inputの行動系列

        :param inputs:
        :return:
        """

        # B: バッチサイズ, T: 系列長, H: embedding_dim
        # query shape: (B,H)
        # keys shape: (B,T,H)
        # behavior_input shape: (B,T)
        query, keys, behavior_input = inputs[0], inputs[1], inputs[2]

        # 行動系列の長さ分繰り返す shape: (B,H) -> (B,T,H)
        query = tf.keras.backend.repeat_elements(query, keys.get_shape()[1], 1)

        # フィードフォワードネットワークへの入力
        # 論文ではquery - keysは入力としていないが、実装ではしているのでそれを採用
        attention_input = tf.keras.layers.Concatenate(axis=-1)(
            [query, keys, query - keys, query * keys])  # shape: (B,T,H*4)

        attention_weight = self.dense_1(attention_input)  # shape: (B,T,dense_units)
        attention_weight = self.dense_2(attention_weight)  # shape: (B,T,dense_units)
        attention_weight = self.dense_3(attention_weight)  # shape: (B,T,1)

        attention_weight = tf.transpose(attention_weight, (0, 2, 1))  # shape: (B,T,1) => (B,1,T)

        # パディング用マスキング
        mask = tf.equal(behavior_input, 0)  # shape: (B,T)
        mask = tf.expand_dims(mask, 1)  # shape: (B,1,T)
        padding = tf.ones_like(attention_weight) * (-2 ** 32 + 1)
        attention_weight = tf.where(mask, attention_weight, padding)  # maskはpad部分のみが1, 他は0

        # scaling
        attention_weight = attention_weight / (keys.get_shape()[-1] ** 0.5)
        attention_weight = tf.nn.softmax(attention_weight)

        attention_output = tf.matmul(attention_weight, keys)

        return attention_output


def build_input(features_info: dict, seq_max_len: int) -> dict:

    inputs = OrderedDict()
    inputs['candidate'] = OrderedDict()
    inputs['behavior'] = OrderedDict()
    inputs['category'] = OrderedDict()

    for feature in features_info:
        if feature['type'] == 'behavior':
            inputs[feature['type']][feature['name']] = tf.keras.layers.Input(
                shape=(seq_max_len, ), name=feature['name'])
        else:
            inputs[feature['type']][feature['name']] = tf.keras.layers.Input(
                shape=(1,), name=feature['name'])

    return inputs


def build_embedding_layer(features_info: dict, embedding_dim: int) -> dict:

    embedding_dict = OrderedDict()

    for feature in features_info:
        if feature['type'] == 'behavior':
            embedding_dict[feature['name']] = tf.keras.layers.Embedding(
                feature['dim'], embedding_dim, name='emb_'+feature['name'], mask_zero=True)
        else:
            embedding_dict[feature['name']] = tf.keras.layers.Embedding(
                feature['dim'], embedding_dim, name='emb_'+feature['name'])

    return embedding_dict


def embedding(inputs: dict, embedding_dict: dict, features_info: dict) -> dict:

    embeddings = OrderedDict()

    embeddings['candidate'] = []
    embeddings['behavior'] = []
    embeddings['category'] = []

    for feature in features_info:
        embeddings = embedding_dict[feature['name']](inputs[feature['type']][feature['name']])
        embeddings[feature['type']].append(embedding)

    # feature type ごとに結合
    for key in embeddings.keys():
        if len(embeddings[key]) >= 2:
            embeddings[key] = tf.keras.layers.Concatenate()(embeddings[key])
        else:
            embeddings[key] = embeddings[key][0]

    return embeddings


def deep_interest_network(features_info, seq_max_len: int = 100, dropout_rate: float = 0.5, embedding_dim: int = 100):

    # build input
    inputs = build_input(features_info, seq_max_len=seq_max_len)

    # build embeddings layer
    embedding_dict = build_embedding_layer(features_info, embedding_dim=embedding_dim)

    # embedding
    embeddings = embedding(inputs, embedding_dict, features_info)

    # local activation unit 等
    behavior_attention_embeddings = AttentionBlock()(
        [embeddings['behavior'], embeddings['candidate'], list(inputs['behavior'].values())[0]])

    # MLPへの入力
    embeddings = tf.keras.layers.Concatenate()(
        [embeddings['category'], embeddings['candidate'], behavior_attention_embeddings]
    )
    embeddings = tf.keras.layers.Flatten()(embeddings)
    embeddings = tf.keras.layers.Dropout(dropout_rate)(embeddings)

    output = tf.keras.layers.Dense(80, activation='relu')(embeddings)
    output = tf.keras.layers.Dropout(dropout_rate)(output)
    output = tf.keras.layers.Dense(40, activation='relu')(output)
    output = tf.keras.layers.Dropout(dropout_rate)(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model

















