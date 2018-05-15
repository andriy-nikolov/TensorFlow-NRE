import numpy as np
import os
import sys
sys.path.insert(0, '/opt/work/wikipedia/wikidata/transR')
import post_process_transR_results as pp;

TARGET_RELATION = "P22";

os.chdir('/opt/work/wikipedia/nre')

entity2idfilename = '/git/TensorFlow-TransX/data/wikidata-family/entity2id.txt';
relation2idfilename = "/git/TensorFlow-TransX/data/wikidata-family/relation2id.txt";
entity2id, id2entity = pp.load_from_id_file(entity2idfilename);
rel2id, id2relation = pp.load_from_id_file(relation2idfilename);

# embedding the position


def pos_embed(x):
    if x < -60:
        return 0
    if x >= -60 and x <= 60:
        return x + 61
    if x > 60:
        return 122
# find the index of x in y, if x not in y, return -1


def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag

def get_q_id(q):
    qid = q;
    if q.startswith('WIKIDATA_ID'):
        qid = q[12:];
    return qid;

def get_emb_dist(q1, q2, entity2id, pair2emb):
    qid1 = get_q_id(q1);
    qid2 = get_q_id(q2);
    if qid1 not in entity2id:
        return None;
    if qid2 not in entity2id:
        return None;
    return pair2emb[(qid1, qid2)];

# loading graph embeddings
def prepare_graph_embeddings(triplesetfilename):
    pair2emb = pp.load_vec_enhanced_triple_set(triplesetfilename, id2entity, id2relation);
    return pair2emb;

# reading data


def init():

    print('reading word embedding data...')
    vec = []
    word2id = {}
    f = open('./origin_data/vec.txt')
    f.readline()
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [(float)(i) for i in content]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    dim = 50
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)

    print('reading relation to id')
    relation2id = {}
    f = open('./origin_data/relation2id.txt', 'r')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()

    # length of sentence is 70
    fixlen = 70
    # max length of position embedding is 60 (-60~+60)
    maxlen = 60

    # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
    train_sen = {}
    # {entity pair:[label1,label2,...]} the label is one-hot vector
    train_ans = {}

    train_embs = {};

    triplesetfilename = "/opt/work/wikipedia/wikidata/transR/trainset_with_emb.npy";
    train_pair2emb = prepare_graph_embeddings(triplesetfilename);

    print("train_pair2emb: " + str(len(train_pair2emb)));

    print('reading train data...')
    f = open('./origin_data/train.txt', 'r')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split('\t')
        # get entity name
        en1 = content[2]
        en2 = content[3]
        relations_str = content[4].split()
        relations = []
        for rel_str in relations_str:
            if not (TARGET_RELATION in relations_str):
                rel_str = 'NA'
            if rel_str == 'FAMILY' and rel_str != 'NA':
                continue
            if rel_str not in relation2id:
                relation = relation2id['NA']
            else:
                relation = relation2id[rel_str]
            relations.append(relation)
        # put the same entity pair sentences into a dict
        tup = (en1, en2)
        label_tag = 0
        if tup not in train_sen:
            train_sen[tup] = []
            train_sen[tup].append([])
            # y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            for y_id in relations:
                label[y_id] = 1
            train_ans[tup] = []
            train_ans[tup].append(label)
        else:
            # y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            for y_id in relations:
                label[y_id] = 1

            temp = find_index(label, train_ans[tup])
            if temp == -1:
                train_ans[tup].append(label)
                label_tag = len(train_ans[tup]) - 1
                train_sen[tup].append([])
            else:
                label_tag = temp

        # print(str(tup) + ", " + str(en1) + ", " + str(en2));
        train_embs[tup] = get_emb_dist(en1, en2, entity2id, train_pair2emb);

        sentence = content[5].split()

        en1pos = 0
        en2pos = 0

        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word

        train_sen[tup][label_tag].append(output)

    print('reading test data ...')

    triplesetfilename = "/opt/work/wikipedia/wikidata/transR/testset_with_emb.npy";
    test_pair2emb = prepare_graph_embeddings(triplesetfilename);

    test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)
    test_ans = {}
    test_embs = {}

    f = open('./origin_data/test.txt', 'r')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split('\t')
        en1 = content[2]
        en2 = content[3]
        relation = 0
        relations_str = content[4].split()
        # print(relations_str)
        relations = []
        for rel_str in relations_str:
            if not (TARGET_RELATION in relations_str):
                rel_str = 'NA'
            if rel_str == 'FAMILY' and rel_str != 'NA':
                continue
            if rel_str not in relation2id:
                relation = relation2id['NA']
            else:
                relation = relation2id[rel_str]
            relations.append(relation)
        tup = (en1, en2)

        if tup not in test_sen:
            test_sen[tup] = []
            # y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            for y_id in relations:
                label[y_id] = 1
            test_ans[tup] = label
        else:
            # y_id = relation
            for y_id in relations:
                test_ans[tup][y_id] = 1
        # print(test_ans)
        sentence = content[5].split()
        # print(sentence)
        # print(test_pair2emb)

        test_embs[tup] = get_emb_dist(en1, en2, entity2id, test_pair2emb);

        en1pos = 0
        en2pos = 0

        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word
        test_sen[tup].append(output)

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_x_emb = [];
    test_x_emb = [];

    print('organizing train data')
    f = open('./data/train_q&a.txt', 'w')
    temp = 0
    for i in train_sen:
        if len(train_ans[i]) != len(train_sen[i]):
            print('ERROR')
        lenth = len(train_ans[i])
        for j in range(lenth):
            train_x.append(train_sen[i][j])
            train_y.append(train_ans[i][j])
            train_x_emb.append(train_pair2emb[(get_q_id(i[0]), get_q_id(i[1]))]);
            f.write(str(temp) + '\t' + i[0] + '\t' + i[1] +
                    '\t' + str(np.argmax(train_ans[i][j])) + '\n')
            temp += 1
    f.close()

    print('organizing test data')
    f = open('./data/test_q&a.txt', 'w')
    temp = 0
    for i in test_sen:
        test_x.append(test_sen[i])
        test_y.append(test_ans[i])
        test_x_emb.append(test_pair2emb[(get_q_id(i[0]), get_q_id(i[1]))]);
        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + tempstr + '\n')
        temp += 1
    f.close()

    train_x = np.array(train_x)
    print("train_x: " + str(train_x.shape))
    train_y = np.array(train_y)
    print("train_y: " + str(train_y.shape))
    test_x = np.array(test_x)
    print("test_x: " + str(test_x.shape))
    test_y = np.array(test_y)
    print("test_y: " + str(test_y.shape))
    train_x_emb = np.array(train_x_emb)
    print("train_x_emb: " + str(train_x_emb.shape))
    test_x_emb = np.array(test_x_emb)
    print("test_x_emb: " + str(test_x_emb.shape))

    np.save('./data/vec.npy', vec)
    np.save('./data/train_x.npy', train_x)
    np.save('./data/train_y.npy', train_y)
    np.save('./data/testall_x.npy', test_x)
    np.save('./data/testall_y.npy', test_y)
    np.save('./data/train_emb.npy', train_x_emb);
    np.save('./data/test_emb.npy', test_x_emb);

    # get test data for P@N evaluation, in which only entity pairs with more
    # than 1 sentence exist
    print('get test data for p@n test')

    pone_test_x = []
    pone_test_y = []
    pone_test_emb = []

    ptwo_test_x = []
    ptwo_test_y = []
    ptwo_test_emb = []


    pall_test_x = []
    pall_test_y = []
    pall_test_emb = []

    for i in range(len(test_x)):
        if len(test_x[i]) > 1:

            pall_test_x.append(test_x[i])
            pall_test_y.append(test_y[i])
            pall_test_emb.append(test_x_emb[i]);

            onetest = []
            temp = np.random.randint(len(test_x[i]))
            onetest.append(test_x[i][temp])
            pone_test_x.append(onetest)
            pone_test_y.append(test_y[i])
            pone_test_emb.append(test_x_emb[i])

            twotest = []
            temp1 = np.random.randint(len(test_x[i]))
            temp2 = np.random.randint(len(test_x[i]))
            while temp1 == temp2:
                temp2 = np.random.randint(len(test_x[i]))
            twotest.append(test_x[i][temp1])
            twotest.append(test_x[i][temp2])
            ptwo_test_x.append(twotest)
            ptwo_test_y.append(test_y[i])
            ptwo_test_emb.append(test_x_emb[i]);

    pone_test_x = np.array(pone_test_x)
    pone_test_y = np.array(pone_test_y)
    ptwo_test_x = np.array(ptwo_test_x)
    ptwo_test_y = np.array(ptwo_test_y)
    pall_test_x = np.array(pall_test_x)
    pall_test_y = np.array(pall_test_y)

    pall_test_emb = np.array(pall_test_emb);
    pone_test_emb = np.array(pone_test_emb);
    ptwo_test_emb = np.array(ptwo_test_emb);

    np.save('./data/pone_test_x.npy', pone_test_x)
    np.save('./data/pone_test_y.npy', pone_test_y)
    np.save('./data/ptwo_test_x.npy', ptwo_test_x)
    np.save('./data/ptwo_test_y.npy', ptwo_test_y)
    np.save('./data/pall_test_x.npy', pall_test_x)
    np.save('./data/pall_test_y.npy', pall_test_y)
    np.save('./data/pall_test_emb.npy', pall_test_emb)
    np.save('./data/pone_test_emb.npy', pone_test_emb)
    np.save('./data/ptwo_test_emb.npy', ptwo_test_emb)


def seperate():

    print('reading training data')
    x_train = np.load('./data/train_x.npy')

    train_word = []
    train_pos1 = []
    train_pos2 = []

    print('seprating train data')
    for i in range(len(x_train)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_train[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    np.save('./data/train_word.npy', train_word)
    np.save('./data/train_pos1.npy', train_pos1)
    np.save('./data/train_pos2.npy', train_pos2)

    print('reading p-one test data')
    x_test = np.load('./data/pone_test_x.npy')
    print('seperating p-one test data')
    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)
    np.save('./data/pone_test_word.npy', test_word)
    np.save('./data/pone_test_pos1.npy', test_pos1)
    np.save('./data/pone_test_pos2.npy', test_pos2)

    print('reading p-two test data')
    x_test = np.load('./data/ptwo_test_x.npy')
    print('seperating p-two test data')
    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)
    np.save('./data/ptwo_test_word.npy', test_word)
    np.save('./data/ptwo_test_pos1.npy', test_pos1)
    np.save('./data/ptwo_test_pos2.npy', test_pos2)

    print('reading p-all test data')
    x_test = np.load('./data/pall_test_x.npy')
    print('seperating p-all test data')
    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)
    np.save('./data/pall_test_word.npy', test_word)
    np.save('./data/pall_test_pos1.npy', test_pos1)
    np.save('./data/pall_test_pos2.npy', test_pos2)

    print('seperating test all data')
    x_test = np.load('./data/testall_x.npy')

    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)

    np.save('./data/testall_word.npy', test_word)
    np.save('./data/testall_pos1.npy', test_pos1)
    np.save('./data/testall_pos2.npy', test_pos2)


def getsmall():

    print('reading training data')
    word = np.load('./data/train_word.npy')
    pos1 = np.load('./data/train_pos1.npy')
    pos2 = np.load('./data/train_pos2.npy')
    y = np.load('./data/train_y.npy')
    emb = np.load('./data/train_emb.npy')

    new_word = []
    new_pos1 = []
    new_pos2 = []
    new_y = []
    new_emb = []

    # we slice some big batch in train data into small batches in case of
    # running out of memory
    print('get small training data')
    for i in range(len(word)):
        lenth = len(word[i])
        if lenth <= 1000:
            new_word.append(word[i])
            new_pos1.append(pos1[i])
            new_pos2.append(pos2[i])
            new_y.append(y[i])
            new_emb.append(emb[i])

        if lenth > 1000 and lenth < 2000:

            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:])

            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:])

            new_y.append(y[i])
            new_y.append(y[i])
            new_emb.append(emb[i])
            new_emb.append(emb[i])

        if lenth > 2000 and lenth < 3000:
            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:2000])
            new_word.append(word[i][2000:])

            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:2000])
            new_pos1.append(pos1[i][2000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:2000])
            new_pos2.append(pos2[i][2000:])

            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_emb.append(emb[i])
            new_emb.append(emb[i])
            new_emb.append(emb[i])

        if lenth > 3000 and lenth < 4000:
            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:2000])
            new_word.append(word[i][2000:3000])
            new_word.append(word[i][3000:])

            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:2000])
            new_pos1.append(pos1[i][2000:3000])
            new_pos1.append(pos1[i][3000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:2000])
            new_pos2.append(pos2[i][2000:3000])
            new_pos2.append(pos2[i][3000:])

            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_emb.append(emb[i])
            new_emb.append(emb[i])
            new_emb.append(emb[i])
            new_emb.append(emb[i])

        if lenth > 4000:

            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:2000])
            new_word.append(word[i][2000:3000])
            new_word.append(word[i][3000:4000])
            new_word.append(word[i][4000:])

            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:2000])
            new_pos1.append(pos1[i][2000:3000])
            new_pos1.append(pos1[i][3000:4000])
            new_pos1.append(pos1[i][4000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:2000])
            new_pos2.append(pos2[i][2000:3000])
            new_pos2.append(pos2[i][3000:4000])
            new_pos2.append(pos2[i][4000:])

            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_emb.append(emb[i])
            new_emb.append(emb[i])
            new_emb.append(emb[i])
            new_emb.append(emb[i])
            new_emb.append(emb[i])

    new_word = np.array(new_word)
    new_pos1 = np.array(new_pos1)
    new_pos2 = np.array(new_pos2)
    new_y = np.array(new_y)
    new_emb = np.array(new_emb)

    np.save('./data/small_word.npy', new_word)
    np.save('./data/small_pos1.npy', new_pos1)
    np.save('./data/small_pos2.npy', new_pos2)
    np.save('./data/small_y.npy', new_y)
    np.save('./data/small_emb.npy', new_emb)

# get answer metric for PR curve evaluation


def getans():
    test_y = np.load('./data/testall_y.npy')
    eval_y = []
    print('Shape test_y: ' + str(test_y.shape))
    for i in test_y:
        eval_y.append(i[1:])
    allans = np.reshape(eval_y, (-1))
    print('Shape allans: ' + str(allans.shape))
    np.save('./data/allans.npy', allans)


def get_metadata():
    fwrite = open('./data/metadata.tsv', 'w')
    f = open('./origin_data/vec.txt')
    f.readline()
    while True:
        content = f.readline().strip()
        if content == '':
            break
        name = content.split()[0]
        fwrite.write(name + '\n')
    f.close()
    fwrite.close()


init()
seperate()
getsmall()
getans()
get_metadata()
