# import os

# os.chdir('/home/adammer/git/TensorFlow-NRE/')

import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network_transe as network
from sklearn.metrics import average_precision_score
from sklearn import linear_model, decomposition, datasets, tree
from sklearn.externals import joblib

os.chdir('/opt/work/wikipedia/nre')

HIDDEN_EMBEDDINGS_SIZE = 10;

FLAGS = tf.app.flags.FLAGS
# change the name to who you want to send
#tf.app.flags.DEFINE_string('wechat_name', 'Tang-24-0325','the user you want to send info to')
# tf.app.flags.DEFINE_string('wechat_name', 'filehelper',
#                           'the user you want to send info to')

# global answers

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
    save_path = './model/';

    # ATTENTION: change pathname before you load your model
    pathname = "./model/ATT_GRU_model-"

    wordembedding = np.load('./data/vec.npy')

    test_settings = network.Settings()
    test_settings.vocab_size = 2268896
    # test_settings.num_classes = 53
    test_settings.num_classes = 11
    test_settings.big_num = 71

    big_num_test = test_settings.big_num

    test_emb = np.load('./data/test_emb.npy');

    '''
    emb_tree = joblib.load(save_path + 'embclassifier');
    print('predicting from embeddings');
    pred_emb = emb_tree.predict_proba(test_emb);
    pred_emb = expand_probs(emb_tree.classes_, pred_emb, test_settings.num_classes);
    '''

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

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

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch
                feed_dict[mtest.input_emb] = emb_batch

                loss, accuracy, prob = sess.run(
                    [mtest.loss, mtest.accuracy, mtest.prob], feed_dict)
                return prob, accuracy

            # evaluate p@n
            def eval_pn(test_y, test_word, test_pos1, test_pos2, test_emb, test_settings):
                allprob = []
                acc = []
                for i in range(int(len(test_word) / float(test_settings.big_num))):
                    prob, accuracy = test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num], test_pos1[i * test_settings.big_num:(
                        i + 1) * test_settings.big_num], test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num], test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num], test_emb[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    acc.append(
                        np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
                    prob = np.reshape(
                        np.array(prob), (test_settings.big_num, test_settings.num_classes))
                    for single_prob in prob:
                        allprob.append(single_prob[1:])
                allprob = np.reshape(np.array(allprob), (-1))

                eval_y = []
                for i in test_y:
                    eval_y.append(i[1:])
                allans = np.reshape(eval_y, (-1))
                order = np.argsort(-allprob)

                # print('allans shape: ' + str(allans.shape))

                print('P@100:')
                top100 = order[:100]
                correct_num_100 = 0.0
                for i in top100:
                    if allans[i] == 1:
                        correct_num_100 += 1.0
                print(correct_num_100 / 100)

                print('P@200:')
                top200 = order[:200]
                correct_num_200 = 0.0
                for i in top200:
                    if allans[i] == 1:
                        correct_num_200 += 1.0
                print(correct_num_200 / 200)

                print('P@300:')
                top300 = order[:300]
                correct_num_300 = 0.0
                for i in top300:
                    if allans[i] == 1:
                        correct_num_300 += 1.0
                print(correct_num_300 / 300)

                if itchat_run:
                    tempstr = 'P@100\n' + str(correct_num_100 / 100) + '\n' + 'P@200\n' + str(
                        correct_num_200 / 200) + '\n' + 'P@300\n' + str(correct_num_300 / 300)
                    itchat.send(tempstr, FLAGS.wechat_name)

            with tf.variable_scope("model"):
                mtest = network.GRU(
                    is_training=False, word_embeddings=wordembedding, settings=test_settings)

            saver = tf.train.Saver()

            # ATTENTION: change the list to the iters you want to test !!
            #testlist = range(9025,14000,25)
            # testlist = [10900]
            testlist = [5800]
            for model_iter in testlist:

                saver.restore(sess, pathname + str(model_iter))
                print("Evaluating P@N for iter " + str(model_iter))

                if itchat_run:
                    itchat.send("Evaluating P@N for iter " +
                                str(model_iter), FLAGS.wechat_name)

                print('Evaluating P@N for one')
                if itchat_run:
                    itchat.send('Evaluating P@N for one', FLAGS.wechat_name)

                test_y = np.load('./data/pone_test_y.npy')
                test_emb = np.load('./data/pone_test_emb.npy');
                test_word = np.load('./data/pone_test_word.npy')
                test_pos1 = np.load('./data/pone_test_pos1.npy')
                test_pos2 = np.load('./data/pone_test_pos2.npy')
                eval_pn(test_y, test_word, test_pos1, test_pos2, test_emb, test_settings)

                print('Evaluating P@N for two')
                if itchat_run:
                    itchat.send('Evaluating P@N for two', FLAGS.wechat_name)
                test_y = np.load('./data/ptwo_test_y.npy')
                test_emb = np.load('./data/ptwo_test_emb.npy');
                test_word = np.load('./data/ptwo_test_word.npy')
                test_pos1 = np.load('./data/ptwo_test_pos1.npy')
                test_pos2 = np.load('./data/ptwo_test_pos2.npy')
                eval_pn(test_y, test_word, test_pos1, test_pos2, test_emb, test_settings)

                print('Evaluating P@N for all')
                if itchat_run:
                    itchat.send('Evaluating P@N for all', FLAGS.wechat_name)
                test_y = np.load('./data/pall_test_y.npy')
                test_emb = np.load('./data/pall_test_emb.npy');
                test_word = np.load('./data/pall_test_word.npy')
                test_pos1 = np.load('./data/pall_test_pos1.npy')
                test_pos2 = np.load('./data/pall_test_pos2.npy')
                eval_pn(test_y, test_word, test_pos1, test_pos2, test_emb, test_settings)

                time_str = datetime.datetime.now().isoformat()
                print(time_str)
                print('Evaluating all test data and save data for PR curve')
                if itchat_run:
                    itchat.send(
                        'Evaluating all test data and save data for PR curve', FLAGS.wechat_name)

                test_y = np.load('./data/testall_y.npy')
                test_emb = np.load('./data/test_emb.npy');
                test_word = np.load('./data/testall_word.npy')
                test_pos1 = np.load('./data/testall_pos1.npy')
                test_pos2 = np.load('./data/testall_pos2.npy')

                allprob = []
                acc = []
                print('Len test_word: ' + str(len(test_word)));
                for i in range(int(len(test_word) / float(test_settings.big_num))):
                    prob, accuracy = test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num], test_pos1[i * test_settings.big_num:(
                        i + 1) * test_settings.big_num], test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num], test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num], test_emb[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    acc.append(
                        np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
                    prob = np.reshape(
                        np.array(prob), (test_settings.big_num, test_settings.num_classes))
                    for single_prob in prob:
                        allprob.append(single_prob)

                allprobarray = np.array(allprob)
                print('Shape allprobarray: ' + str(allprobarray.shape))

                dists = test_emb[:, :HIDDEN_EMBEDDINGS_SIZE];
                subjs = test_emb[:, HIDDEN_EMBEDDINGS_SIZE:HIDDEN_EMBEDDINGS_SIZE*2];
                objs = test_emb[:, HIDDEN_EMBEDDINGS_SIZE*2:];

                norms = np.linalg.norm(dists, axis=1, keepdims=True);

                norms = (norms / (norms.max(axis = 0) - norms.min(axis=0)) - 0.5 ) * 2;
                objs = (objs / (objs.max(axis = 0) - objs.min(axis=0)) - 0.5 ) * 2;

                print('Shape norms: ' + str(norms.shape))

#                print('predicting from ensemble');
#                ens_tree = joblib.load(save_path + 'ensclassifier');

#                pred_ens = ens_tree.predict_proba(np.hstack((allprobarray, norms, objs)));
#                pred_ens = expand_probs(ens_tree.classes_, pred_ens, test_settings.num_classes);


                '''
                filt = np.logical_and(pred_ens[:,1]>pred_ens[:,0], norms[:,0]>6.0);

                for i in range(0, pred_ens.shape[0]):
                    if filt[i]==True:
                        pred_ens[i, 0] = 1.0;
                        pred_ens[i, 1] = 0.0;

                filt = np.logical_and(pred_ens[:,0]>pred_ens[:,1], norms[:,0]<2.0);

                for i in range(0, pred_ens.shape[0]):
                    if filt[i]==True:
                        pred_ens[i, 0] = 0.0;
                        pred_ens[i, 1] = 1.0;
                '''

                # allprob_with_na = np.reshape(pred_ens, (-1));
                # allprob = np.reshape(pred_ens[:,1:], (-1));
                # print('Shape allprobarray: ' + str(allprobarray.shape))
                allprob = np.reshape(allprobarray[:, 1:], (-1));
                allprob_with_na = np.reshape(allprobarray, (-1));
                print('Shape allprob: ' + str(allprob.shape))
                print('Shape allprob_with_na: ' + str(allprob_with_na.shape))
                order = np.argsort(-allprob)

                print('saving all test result...')
                current_step = model_iter

                # ATTENTION: change the save path before you save your result
                # !!
                np.save('./out/allprob_iter_' +
                        str(current_step) + '.npy', allprob_with_na)
                allans = np.load('./data/allans.npy')

                print('Shape allans: ' + str(allans.shape))
                print('Shape allprob: ' + str(allprob.shape))

                # caculate the pr curve area
                average_precision = average_precision_score(allans, allprob)
                print('PR curve area:' + str(average_precision))

                if itchat_run:
                    itchat.send('PR curve area:' +
                                str(average_precision), FLAGS.wechat_name)

                time_str = datetime.datetime.now().isoformat()
                print(time_str)
                print('P@N for all test data:')
                print('P@100:')
                top100 = order[:100]
                correct_num_100 = 0.0
                for i in top100:
                    if allans[i] == 1:
                        correct_num_100 += 1.0
                print(correct_num_100 / 100)

                print('P@200:')
                top200 = order[:200]
                correct_num_200 = 0.0
                for i in top200:
                    if allans[i] == 1:
                        correct_num_200 += 1.0
                print(correct_num_200 / 200)

                print('P@300:')
                top300 = order[:300]
                correct_num_300 = 0.0
                for i in top300:
                    if allans[i] == 1:
                        correct_num_300 += 1.0
                print(correct_num_300 / 300)

                if itchat_run:
                    tempstr = 'P@100\n' + str(correct_num_100 / 100) + '\n' + 'P@200\n' + str(
                        correct_num_200 / 200) + '\n' + 'P@300\n' + str(correct_num_300 / 300)
                    itchat.send(tempstr, FLAGS.wechat_name)


if __name__ == "__main__":
    if itchat_run:
        itchat.auto_login(hotReload=True, enableCmdQR=2)
    tf.app.run()
