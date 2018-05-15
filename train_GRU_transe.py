import tensorflow as tf
import numpy as np
import time
import datetime
import os
from sklearn import linear_model, decomposition, datasets, tree, svm
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.externals import joblib
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

os.chdir('/opt/work/wikipedia/nre')
import network_transe as network


FLAGS = tf.app.flags.FLAGS

HIDDEN_EMBEDDINGS_SIZE = 10;

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')

# change the name to who you want to send
#tf.app.flags.DEFINE_string('wechat_name', 'Tang-24-0325','the user you want to send info to')
# tf.app.flags.DEFINE_string('wechat_name', 'filehelper',
#                           'the user you want to send info to')

# if you want to try itchat, please set it to True
itchat_run = False
if itchat_run:
    import itchat

def expand_probs(classes, probs, all_classes):
    res = np.zeros((probs.shape[0], all_classes));
    for i in range(0, probs.shape[0]):
        for j in range(0, len(probs[i])):
            res[i, classes[j]] = probs[i, j];
    return res;


def main(_):
    # the path to save models
    save_path = './model/'

    print('reading wordembedding')
    wordembedding = np.load('./data/vec.npy')

    print('reading training data')
    train_y = np.load('./data/small_y.npy')
    train_word = np.load('./data/small_word.npy')
    train_pos1 = np.load('./data/small_pos1.npy')
    train_pos2 = np.load('./data/small_pos2.npy')
    train_emb = np.load('./data/small_emb.npy');
    # validation_emb = np.load('./data/validation_emb.npy');

    train_y_scikit = np.argmax(train_y, axis=1);
    # train_y_scikit[0] = 10;
    print(train_y_scikit);

    settings = network.Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(train_y[0])

    big_num = settings.big_num

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():
            '''
            print('make predictions from embeddings');
            tree_emb = linear_model.LogisticRegression();
            # tree_emb.n_classes = settings.num_classes;
            # tree_emb.classes_ = np.array(list(range(0,settings.num_classes)));
            tree_emb.fit(train_emb, train_y_scikit);
            print('classes: ' + str(tree_emb.classes_));

            print('train_emb: ' + str(train_emb.shape));
            print('train_y: ' + str(train_y.shape));
            probs_emb = tree_emb.predict_proba(train_emb);
            print('probs_ems: ' + str(len(probs_emb))  + ': ' + str(probs_emb));
            print('probs_ems[0]: ' + str(probs_emb[0].shape) + ': ' + str(probs_emb[0]));
            probs_emb = expand_probs(tree_emb.classes_, probs_emb, settings.num_classes);
            print('probs_ems: ' + str(probs_emb.shape));
            print('probs_ems: ' + str(probs_emb[0]));
            joblib.dump(tree_emb, save_path + 'embclassifier');
            '''

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = network.GRU(
                    is_training=True, word_embeddings=wordembedding, settings=settings)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)

            # train_op=optimizer.minimize(m.total_loss,global_step=global_step)
            train_op = optimizer.minimize(
                m.final_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)

            #merged_summary = tf.summary.merge_all()
            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(
                FLAGS.summary_dir + '/train_loss', sess.graph)

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch, emb_batch):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []

                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)

                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch
                feed_dict[m.input_emb] = emb_batch

                loss, accuracy, prob = sess.run(
                    [m.loss, m.accuracy, m.prob], feed_dict)
                return prob, accuracy

            # summary for embedding
            # it's not available in tf 0.11,(because there is no embedding panel in 0.11's tensorboard) so I delete it =.=
            # you can try it on 0.12 or higher versions but maybe you should
            # change some function name at first.

            # summary_embed_writer = tf.train.SummaryWriter('./model',sess.graph)
            # config = projector.ProjectorConfig()
            # embedding_conf = config.embedding.add()
            # embedding_conf.tensor_name = 'word_embedding'
            # embedding_conf.metadata_path = './data/metadata.tsv'
            # projector.visualize_embeddings(summary_embed_writer, config)

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, emb_batch, big_num):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch
                # print(emb_batch.shape)
                feed_dict[m.input_emb] = emb_batch

                temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                accuracy = np.reshape(np.array(accuracy), (big_num))
                acc = np.mean(accuracy)
                summary_writer.add_summary(summary, step)

                if step % 50 == 0:
                    tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}".format(
                        time_str, step, loss, acc)
                    print(tempstr)
                    if itchat_run:
                        itchat.send(tempstr, FLAGS.wechat_name)

            for one_epoch in range(settings.num_epochs):
                if itchat_run:
                    itchat.send('epoch ' + str(one_epoch) +
                                ' starts!', FLAGS.wechat_name)
                print('epoch ' + str(one_epoch) + ' starts!')

                temp_order = list(range(len(train_word)))
                np.random.shuffle(temp_order)
                stop_train = False

                for i in range(int(len(temp_order) / float(settings.big_num))):

                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []
                    temp_emb = []

                    temp_input = temp_order[i *
                                            settings.big_num:(i + 1) * settings.big_num]
                    for k in temp_input:
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])
                        temp_emb.append(train_emb[k])
                    num = 0
                    for single_word in temp_word:
                        num += len(single_word)

                    if num > 1500:
                        print('out of range')
                        continue

                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)
                    temp_emb = np.array(temp_emb)

                    train_step(temp_word, temp_pos1, temp_pos2,
                               temp_y, temp_emb, settings.big_num)

                    current_step = tf.train.global_step(sess, global_step)
                    if current_step == 5800:
                        # if current_step == 50:
                        print('saving model')
                        path = saver.save(
                            sess, save_path + 'ATT_GRU_model', global_step=current_step)
                        tempstr = 'have saved model to ' + path
                        print(tempstr)
                        stop_train = True;
                        break;
                if stop_train == True:
                    break;

            if itchat_run:
                itchat.send('training has been finished!', FLAGS.wechat_name)
            print('training has been finished!')

            print('make predictions on train data');
            allprob = [];
            temp_order = list(range(len(train_word)));
            for i in range(int(len(temp_order) / float(settings.big_num))):
                temp_word = []
                temp_pos1 = []
                temp_pos2 = []
                temp_y = []
                temp_emb = []

                temp_input = temp_order[i *
                                        settings.big_num:(i + 1) * settings.big_num]
                for k in temp_input:
                    temp_word.append(train_word[k])
                    temp_pos1.append(train_pos1[k])
                    temp_pos2.append(train_pos2[k])
                    temp_y.append(train_y[k])
                    temp_emb.append(train_emb[k])
                num = 0
                for single_word in temp_word:
                    num += len(single_word)

                if num > 1500:
                    print('out of range')
                    continue

                temp_word = np.array(temp_word)
                temp_pos1 = np.array(temp_pos1)
                temp_pos2 = np.array(temp_pos2)
                temp_y = np.array(temp_y)
                temp_emb = np.array(temp_emb)

                prob, accuracy = test_step(temp_word, temp_pos1, temp_pos2, temp_y, temp_emb);
                prob = np.reshape(
                        np.array(prob), (settings.big_num, settings.num_classes));
                for single_prob in prob:
                    allprob.append(single_prob);
            allprob = np.array(allprob);


if __name__ == "__main__":
    if itchat_run:
        itchat.auto_login(hotReload=True, enableCmdQR=2)
    tf.app.run()
