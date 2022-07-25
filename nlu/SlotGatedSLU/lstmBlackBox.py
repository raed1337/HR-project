import os
import logging
import sys

# import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell_impl
from .utils import createVocabulary
from .utils import loadVocabulary
from .utils import computeF1Score
from .utils import DataProcessor
import subprocess

# tf.disable_v2_behavior()


class lstmBlackBox:
    def lstm_black_box(self, text):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        layer_size = 64
        model_type = "full"
        batch_size = 16
        dataset = "Airudi"
        model_path = "model"
        vocab_path = "vocab"
        train_data_path = "train"
        test_data_path = "test"
        valid_data_path = "valid"
        input_file = "seq.in"
        slot_file = "seq.out"
        intent_file = "label"

        add_final_state_to_intent = True
        remove_slot_attn = False

        full_train_path = os.path.join("./nlu/SlotGatedSLU/data", dataset, train_data_path)
        full_test_path = os.path.join("./nlu/SlotGatedSLU/data", dataset, test_data_path)
        full_valid_path = os.path.join("./nlu/SlotGatedSLU/data", dataset, valid_data_path)

        in_vocab = loadVocabulary(os.path.join("./nlu/SlotGatedSLU/", vocab_path, "in_vocab"))
        slot_vocab = loadVocabulary(os.path.join("./nlu/SlotGatedSLU/", vocab_path, "slot_vocab"))
        intent_vocab = loadVocabulary(os.path.join("./nlu/SlotGatedSLU/", vocab_path, "intent_vocab"))

        with open(full_valid_path + "/label", "w") as labelfile:
            labelfile.write("int")
        yo = []
        with open(full_valid_path + "/seq.in", "w") as seqinfile:
            text = text.lower()
            dn = text.split()
            yo = dn
            seqinfile.write(text)

        with open(full_valid_path + "/seq.out", "w") as seqoutfile:
            str = "o "
            for i in range(0, len(yo) - 1):
                str += "o "
            seqoutfile.write(str)

        def createModel(
            input_data, input_size, sequence_length, slot_size, intent_size, layer_size=128, isTraining=True
        ):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(layer_size)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(layer_size)

            if isTraining == True:
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.5, output_keep_prob=0.5)
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=0.5, output_keep_prob=0.5)

            embedding = tf.get_variable("embedding", [input_size, layer_size])
            inputs = tf.nn.embedding_lookup(embedding, input_data)
            state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs, sequence_length=sequence_length, dtype=tf.float32
            )

            final_state = tf.concat([final_state[0][0], final_state[0][1], final_state[1][0], final_state[1][1]], 1)
            state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2)
            state_shape = state_outputs.get_shape()

            with tf.variable_scope("attention"):
                slot_inputs = state_outputs
                if remove_slot_attn == False:
                    with tf.variable_scope("slot_attn"):
                        attn_size = state_shape[2].value
                        origin_shape = tf.shape(state_outputs)
                        hidden = tf.expand_dims(state_outputs, 1)
                        hidden_conv = tf.expand_dims(state_outputs, 2)
                        # hidden shape = [batch, sentence length, 1, hidden size]
                        k = tf.get_variable("AttnW", [1, 1, attn_size, attn_size])
                        hidden_features = tf.nn.conv2d(hidden_conv, k, [1, 1, 1, 1], "SAME")
                        hidden_features = tf.reshape(hidden_features, origin_shape)
                        hidden_features = tf.expand_dims(hidden_features, 1)
                        v = tf.get_variable("AttnV", [attn_size])

                        slot_inputs_shape = tf.shape(slot_inputs)
                        slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])
                        y = rnn_cell_impl._linear(slot_inputs, attn_size, True)
                        y = tf.reshape(y, slot_inputs_shape)
                        y = tf.expand_dims(y, 2)
                        s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [3])
                        a = tf.nn.softmax(s)
                        # a shape = [batch, input size, sentence length, 1]
                        a = tf.expand_dims(a, -1)
                        slot_d = tf.reduce_sum(a * hidden, [2])
                else:
                    attn_size = state_shape[2].value
                    slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])

                intent_input = final_state
                with tf.variable_scope("intent_attn"):
                    attn_size = state_shape[2].value
                    hidden = tf.expand_dims(state_outputs, 2)
                    k = tf.get_variable("AttnW", [1, 1, attn_size, attn_size])
                    hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
                    v = tf.get_variable("AttnV", [attn_size])

                    y = rnn_cell_impl._linear(intent_input, attn_size, True)
                    y = tf.reshape(y, [-1, 1, 1, attn_size])
                    s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [2, 3])
                    a = tf.nn.softmax(s)
                    a = tf.expand_dims(a, -1)
                    a = tf.expand_dims(a, -1)
                    d = tf.reduce_sum(a * hidden, [1, 2])

                    if add_final_state_to_intent == True:
                        intent_output = tf.concat([d, intent_input], 1)
                    else:
                        intent_output = d

                with tf.variable_scope("slot_gated"):
                    intent_gate = rnn_cell_impl._linear(intent_output, attn_size, True)
                    intent_gate = tf.reshape(intent_gate, [-1, 1, intent_gate.get_shape()[1].value])
                    v1 = tf.get_variable("gateV", [attn_size])
                    if remove_slot_attn == False:
                        slot_gate = v1 * tf.tanh(slot_d + intent_gate)
                    else:
                        slot_gate = v1 * tf.tanh(state_outputs + intent_gate)
                    slot_gate = tf.reduce_sum(slot_gate, [2])
                    slot_gate = tf.expand_dims(slot_gate, -1)
                    if remove_slot_attn == False:
                        slot_gate = slot_d * slot_gate
                    else:
                        slot_gate = state_outputs * slot_gate
                    slot_gate = tf.reshape(slot_gate, [-1, attn_size])
                    slot_output = tf.concat([slot_gate, slot_inputs], 1)

            with tf.variable_scope("intent_proj"):
                intent = rnn_cell_impl._linear(intent_output, intent_size, True)

            with tf.variable_scope("slot_proj"):
                slot = rnn_cell_impl._linear(slot_output, slot_size, True)

            outputs = [slot, intent]
            return outputs

        #
        # # # # Create Training Model
        input_data = tf.placeholder(tf.int32, [None, None], name="inputs")
        sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
        global_step = tf.Variable(0, trainable=False, name="global_step")
        slots = tf.placeholder(tf.int32, [None, None], name="slots")
        slot_weights = tf.placeholder(tf.float32, [None, None], name="slot_weights")
        intent = tf.placeholder(tf.int32, [None], name="intent")

        with tf.variable_scope("model"):
            inference_outputs = createModel(
                input_data,
                len(in_vocab["vocab"]),
                sequence_length,
                len(slot_vocab["vocab"]),
                len(intent_vocab["vocab"]),
                layer_size=layer_size,
                isTraining=False,
            )

        inference_slot_output = tf.nn.softmax(inference_outputs[0], name="slot_output")
        inference_intent_output = tf.nn.softmax(inference_outputs[1], name="intent_output")

        inference_outputs = [inference_intent_output, inference_slot_output]
        inference_inputs = [input_data, sequence_length]

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(
                os.path.join("nlu/SlotGatedSLU/", model_path, "_step_17550_epochs_130.ckpt.meta")
            )
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join("nlu/SlotGatedSLU/", model_path)))

            epochs = 0
            loss = 0.0
            data_processor = None
            line = 0
            num_loss = 0
            step = 0
            no_improve = 0

            # variables to store highest values among epochs, only use 'valid_err' for now
            valid_slot = 0
            test_slot = 0
            valid_intent = 0
            test_intent = 0
            valid_err = 0
            test_err = 0

            while True:

                def valid(in_path, slot_path, intent_path):
                    data_processor_valid = DataProcessor(
                        in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab
                    )

                    pred_intents = []
                    correct_intents = []
                    slot_outputs = []
                    correct_slots = []
                    input_words = []

                    # used to gate
                    gate_seq = []
                    while True:
                        (
                            in_data,
                            slot_data,
                            slot_weight,
                            length,
                            intents,
                            in_seq,
                            slot_seq,
                            intent_seq,
                        ) = data_processor_valid.get_batch(batch_size)
                        feed_dict = {input_data.name: in_data, sequence_length.name: length}
                        ret = sess.run(inference_outputs, feed_dict)
                        for i in ret[0]:
                            pred_intents.append(np.argmax(i))
                        for i in intents:
                            correct_intents.append(i)

                        pred_slots = ret[1].reshape((slot_data.shape[0], slot_data.shape[1], -1))
                        for p, t, i, l in zip(pred_slots, slot_data, in_data, length):
                            p = np.argmax(p, 1)
                            tmp_pred = []
                            tmp_correct = []
                            tmp_input = []
                            for j in range(l):
                                tmp_pred.append(slot_vocab["rev"][p[j]])
                                tmp_correct.append(slot_vocab["rev"][t[j]])
                                tmp_input.append(in_vocab["rev"][i[j]])

                            slot_outputs.append(tmp_pred)
                            correct_slots.append(tmp_correct)
                            input_words.append(tmp_input)

                        if data_processor_valid.end == 1:
                            break

                    for key, value in intent_vocab["vocab"].items():
                        if value == pred_intents[0]:
                            slot_outputs[0].append(key)
                    return slot_outputs[0]

                slot_out = valid(
                    os.path.join(full_valid_path, input_file),
                    os.path.join(full_valid_path, slot_file),
                    os.path.join(full_valid_path, intent_file),
                )
                break
        tf.get_variable_scope().reuse_variables()
        tf.reset_default_graph()

        return slot_out
