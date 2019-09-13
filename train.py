
import numpy as np
import tensorflow as tf

from lib import preprocesser
from lib import data_handler
from model import deep_interest_network


def get_unique_counts(token_encoders, cols):

    # ユニーク数取得
    unique_counts = {}
    for col in cols:
        unique_counts[col] = len(token_encoders[col].get_params()['mapping'][0]['mapping'])

    return unique_counts


def build_features_info(unique_counts, behavior_cols, category_cols):

    features_info = []
    for category_col in category_cols:
        features_info.append(
            {'name': 'category_' + category_col, 'dim': unique_counts[category_col], 'type': 'category'})
    for behavior_col in behavior_cols:
        features_info.append(
            {'name': 'behavior_' + behavior_col, 'dim': unique_counts[behavior_col], 'type': 'behavior'})
        features_info.append(
            {'name': 'candidate_' + behavior_col, 'dim': unique_counts[behavior_col], 'type': 'candidate'})

    return features_info


def build_tf_dataset(dataset_dict: dict, seq_max_len: int, category_cols, behavior_cols) -> [dict, np.array]:

    inputs = {}
    for category_col in category_cols:
        inputs['category_' + category_col] = np.array(dataset_dict['category_' + category_col])
    for behavior_col in behavior_cols:
        inputs['behavior_' + behavior_col] = tf.keras.preprocessing.sequence.pad_sequences(
            dataset_dict['behavior_' + behavior_col], padding='post', truncating='post', maxlen=seq_max_len)
        inputs['candidate_' + behavior_col] = np.array(dataset_dict['candidate_' + behavior_col])

    output = np.array(dataset_dict['label'])

    return inputs, output


def main():
    seq_max_len = 100

    # load data
    filepath = 'data/unpacked/data.tsv'
    use_columns = ['user_id', 'order_number', 'add_to_cart_order', 'order_dow', 'order_hour_of_day',
                   'days_since_prior_order', 'product_name', 'aisle', 'department']
    df = data_handler.load_tsv(filepath, use_columns)

    # encode
    encode_cols = ['user_id', 'product_name', 'aisle', 'department', 'order_dow',
                   'order_hour_of_day', 'days_since_prior_order']
    df, ordinal_encoders = preprocesser.token_col_encode(df, encode_cols)

    # make dataset
    user_col = 'user_id'
    behavior_key_col = 'product_id'
    behavior_category_cols = ['aisle', 'department']
    sort_cols = ['order_number', 'add_to_cart_order']
    context_cols = ['order_dow', 'order_hour_of_day', 'days_since_prior_order']
    dataset_dict = preprocesser.aggregate_features(df, user_col=user_col, sort_cols=sort_cols,
                                                   behavior_key_col=behavior_key_col,
                                                   behavior_category_cols=behavior_category_cols,
                                                   seq_max_len=seq_max_len,
                                                   user_category_cols=None, context_cols=context_cols)

    # get counts
    behavior_cols = ['product_id', 'aisle', 'department']
    category_cols = ['user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']
    unique_counts = get_unique_counts(ordinal_encoders, behavior_cols+category_cols)

    # build feature info
    features_info = build_features_info(unique_counts, behavior_cols, category_cols)

    # build dataset
    inputs, output = build_tf_dataset(dataset_dict, seq_max_len, category_cols, behavior_cols)

    # build model
    model = deep_interest_network(features_info, seq_max_len=seq_max_len)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.9)
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x=inputs, y=output, epochs=30, validation_split=0.2, batch_size=10000)


if __name__ == '__main__':
    main()







