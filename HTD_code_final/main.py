import numpy as np
import tensorflow as tf
import sys
import time
import random
import datetime

random.seed(datetime.datetime.now())

from model import Seq2SeqModel, _START_VOCAB
try:
    from wordseg_python import Global
except:
    Global = None

question_words = []

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 20000, "vocabulary size.")
tf.app.flags.DEFINE_integer("embed_units", 100, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("layers", 4, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("batch_size", 50, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory") 
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.") 
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("check_version", 0, "The version for continuing training or for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("inference_path", "", "Set filename of inference, empty for screen input")
tf.app.flags.DEFINE_string("PMI_path", "./PMI", "PMI director.") 
tf.app.flags.DEFINE_integer("keywords_per_sentence", 20, "How many keywords will be included")
tf.app.flags.DEFINE_boolean("question_data", True, "(Deprecated, please set to True)An unused option in the final version.")
FLAGS = tf.app.flags.FLAGS

def load_data(path, fname):
    with open('%s/%s.post' % (path, fname)) as f:
        post = [line.strip().split() for line in f.readlines()]
    with open('%s/%s.response' % (path, fname)) as f:
        response = [line.strip().split() for line in f.readlines()]
    data = []
    for p, r in zip(post, response):
        data.append({'post': p, 'response': r})
    return data

def build_vocab(path, data):
    question_file = open("question_words.txt", "r")
    question_words = question_file.readline().strip().split('|')
    print len(question_words)
    question_file.close()

    if FLAGS.question_data:
        print("Creating vocabulary...")
        vocab = {}

    #It was a bad attempt, just ignore the following codes.
    #question_classifier and classified_data won't be used when running
        
        #build classifier
        question_classifier = {}
        question_classifier[question_words[0]] = 0
        question_classifier[question_words[1]] = -1
        question_classifier[question_words[2]] = 1
        question_classifier[question_words[3]] = 2
        question_classifier[question_words[4]] = 3
        question_classifier[question_words[5]] = 4
        question_classifier[question_words[6]] = 5
        question_classifier[question_words[7]] = -1
        question_classifier[question_words[8]] = -1
        question_classifier[question_words[9]] = 3
        question_classifier[question_words[10]] = 6
        question_classifier[question_words[11]] = -1
        question_classifier[question_words[12]] = 7
        question_classifier[question_words[13]] = 8
        question_classifier[question_words[14]] = 3
        question_classifier[question_words[15]] = 9
        question_classifier[question_words[16]] = -1
        question_classifier[question_words[17]] = -1
        question_classifier[question_words[18]] = -1
        question_classifier[question_words[19]] = 10
        question_classifier[question_words[20]] = 11
        question_classifier[question_words[21]] = 6
        question_classifier[question_words[22]] = 12
        question_classifier[question_words[23]] = -1
        question_classifier[question_words[24]] = -1
        question_classifier[question_words[25]] = 2
        classified_data = [[] for i in range(13)]
        for i, pair in enumerate(data):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            for token in pair['post']+pair['response']:
                if token in vocab:
                    vocab[token] += 1
                elif not token in question_words:
                    vocab[token] = 1

                #add classifiedData
                if token in question_words:
                    if question_classifier[token] != -1:
                        classified_data[question_classifier[token]].append(pair)
                        #could a pair in several chassified_data's conmponents

        vocab_list = _START_VOCAB + question_words + sorted(vocab, key=vocab.get, reverse=True) 
        if len(vocab_list) > FLAGS.symbols:
            vocab_list = vocab_list[:FLAGS.symbols]
        f1 = open("vocab.txt", 'w')
        for word in vocab_list:
            f1.write(word)
            f1.write('\n')
        print len(vocab_list)
        f1.close()
    else:
        print("loading vocab from vocab.txt...")
        vocab_list = []
        f1 = open("vocab.txt", 'r')
        for word in f1:
            vocab_list.append(word.strip())
        print len(vocab_list)

    print("Loading word vectors...")
    vectors = {}
    with open('%s/vector.txt' % "/home/data/share/wordvector") as f: #you should set your word to vector path here
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector
    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = map(float,  vectors[word].split())
        else:
            vector = np.zeros((FLAGS.embed_units), dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
    if FLAGS.question_data:
        return vocab_list, embed, question_words, classified_data
    else:
        return vocab_list, embed, question_words, data

def load_PMI(PMItype): #added by wys, PMItype = "noun", "verb", "all"
    print "loading PMI:"
    keywords_list = []
    keywords_file = open("%s/%s.txt" % (FLAGS.PMI_path, PMItype), 'r')
    for line in keywords_file:
        keywords_list.append(line.strip())
    keywords_index = {}
    keywords_file.close()
    PMI_file = open("%s/%s_PMI.txt" % (FLAGS.PMI_path, PMItype), 'r')
    for i in range(len(keywords_list)):
        keywords_index[keywords_list[i]] = i
    PMI = []
    temp = 0
    for line in PMI_file:
        if temp % 1000 == 0:
            print "    loading %d lines" % temp
        temp = temp + 1
        linePMI = map(float, line.strip().split())
        PMI.append(linePMI)
    return keywords_list, keywords_index, PMI

def gen_batched_data(data):
    encoder_len = max([len(item['post']) for item in data])+1
    decoder_len = max([len(item['response']) for item in data])+1

    posts, responses, posts_length, responses_length = [], [], [], []
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)
    #complete the list with '_PAD'.

    keyword_tensor = []
    word_type = []
    keywords = np.zeros((3, FLAGS.symbols))
    for i in range(4):
        keywords[1][i] = 1
    for i in range(4, 4 + len(question_words)):
        keywords[0][i] = 1
    for i in range(len(question_words) + 4, FLAGS.symbols):
            keywords[1][i] = 1
            keywords[2][i] = 0

    for i in range(4, 4 + len(question_words)):
        keywords[0][i] = 1

    for item in data:
        for word in item["response"]:
            if keywords_index.has_key(word):
                keywords[1][key_to_vocab[keywords_index[word]]] = 0
                keywords[2][key_to_vocab[keywords_index[word]]] = 1
                word_type.append(2)
            elif word in question_words:
                word_type.append(0)
            else:
                word_type.append(1)
        for _ in range(decoder_len - len(item["response"])):
            word_type.append(1)
        posts.append(padding(item['post'], encoder_len))
        responses.append(padding(item['response'], decoder_len))
        posts_length.append(len(item['post'])+1)
        responses_length.append(len(item['response'])+1)
        for _ in range(decoder_len):
            keyword_tensor.append(keywords)  
        for word in item["response"]:
            if keywords_index.has_key(word):
                keywords[1][key_to_vocab[keywords_index[word]]] = 1
                keywords[2][key_to_vocab[keywords_index[word]]] = 0

    batched_data = {'posts': np.array(posts),
            'responses': np.array(responses),
            'posts_length': posts_length,
            'responses_length': responses_length,
            'keyword_tensor': keyword_tensor,
            'word_type': word_type}
    return batched_data

def train(model, sess, data_train):
    selected_data = [random.choice(data_train) for i in range(FLAGS.batch_size)]
    batched_data = gen_batched_data(selected_data)
    outputs = model.step_decoder(sess, batched_data)
    print "train_output[0]", outputs[0]
    print "train_output[1]", outputs[1]
    return outputs[0], outputs[3]

def evaluate(model, sess, data_dev):
    loss = np.zeros((1, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = gen_batched_data(selected_data)
        outputs = model.step_decoder(sess, batched_data, forward_only=True)
        loss += outputs[1]
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= times
    print('    perplexity on dev set: %.2f' % np.exp(loss))

def inference(model, sess, posts):
    length = [len(p)+1 for p in posts]
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)
    batched_posts = [padding(p, max(length)) for p in posts]
    keyword_tensor = []
    keywords = np.zeros((3, FLAGS.symbols))
    for i in range(4):
        keywords[1][i] = 1
    for i in range(4, 4 + len(question_words)):
        keywords[0][i] = 1

    for item in posts:
        for i in range(len(question_words) + 4, FLAGS.symbols):
            keywords[1][i] = 1
            keywords[2][i] = 0
        tempPMI = [0] * len(keywords_list)
        for word in item:
            if keywords_index.has_key(word):
                for i in range(len(keywords_list)):
                    if PMI[keywords_index[word]][i] > 0:
                        tempPMI[i] = tempPMI[i] + PMI[keywords_index[word]][i]
        for i in range(FLAGS.keywords_per_sentence):
            pos = 0
            max_PMI = 0
            for j in range(len(keywords_list)):
                if tempPMI[j] > max_PMI:
                    pos = j
                    max_PMI = tempPMI[j]
            keywords[1][key_to_vocab[pos]] = 0
            keywords[2][key_to_vocab[pos]] = 1
            tempPMI[pos] = 0
            with open("PMI_infer.txt", 'a') as f:
                f.write(keywords_list[pos])
                f.write("\n")
            print keywords_list[pos]
        keyword_tensor.append(keywords)  
        with open("PMI_infer.txt", 'a') as f:
            f.write("\n")
    batched_data = {'posts': np.array(batched_posts),
            'posts_length': np.array(length, dtype=np.int32),
            'keyword_tensor': keyword_tensor}
    responses = model.inference(sess, batched_data)[0]
    results = []
    for response in responses:
        result = []
        for token in response:
            if token != '_EOS':
                result.append(token)
            else:
                break
        results.append(result)
    return results

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    if FLAGS.is_train:
        #load dataset: you should apply your train and test set here
        data_train = load_data(FLAGS.data_dir, 'weibo_pair_train_Q_after')
        data_dev = load_data(FLAGS.data_dir, 'weibo_pair_dev_Q')

        #test how well sample selection works:
        vocab, embed, question_words, classified_data = build_vocab(FLAGS.data_dir, data_train)
        keywords_list, keywords_index, PMI = load_PMI("all") 
        key_to_vocab = [0] * len(keywords_list)
        inverse_vocab = {}
        for i in range(len(vocab)):
            inverse_vocab[vocab[i]] = i
        print "mapping keywords to vocab..."
        for i in range(len(keywords_list)):
            if inverse_vocab.has_key(keywords_list[i]):
                key_to_vocab[i] = inverse_vocab[keywords_list[i]]
            if i % 1000 == 0:
                print "    processing line %d" % i
        model = Seq2SeqModel(
                FLAGS.symbols,
                len(question_words),
                FLAGS.embed_units,
                FLAGS.units,
                FLAGS.layers,
                is_train=True,
                vocab=vocab,
                embed=embed,
                question_data=FLAGS.question_data)
        if FLAGS.log_parameters:
            model.print_parameters()

        if FLAGS.check_version > 0:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.check_version)
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(sess, model_path)
            model.symbol2index.init.run()
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            model.symbol2index.init.run()

        loss_step, time_step, log_perplexity_step = np.zeros((1, )), 0, np.zeros((1, ))
        previous_losses = [1e15]*3
        while True:
            if model.global_step.eval() % FLAGS.per_checkpoint == 0:
                show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
                print("global step %d learning rate %.4f step-time %.2f perplexity %s"
                        % (model.global_step.eval(), model.learning_rate.eval(),
                            time_step, show(np.exp(log_perplexity_step))))
                print "loss_step:", loss_step
                model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir,
                        global_step=model.global_step)
                evaluate(model, sess, data_dev)
                if np.sum(loss_step) > max(previous_losses):
                    sess.run(model.learning_rate_decay_op)
                previous_losses = previous_losses[1:]+[np.sum(loss_step)]
                loss_step, time_step, log_perplexity_step = np.zeros((1, )), 0, np.zeros((1, ))
            print model.global_step.eval()
            start_time = time.time()
            per_loss, per_log_perplexity = train(model, sess, data_train)
            loss_step += per_loss / FLAGS.per_checkpoint
            log_perplexity_step += per_log_perplexity / FLAGS.per_checkpoint
            time_step += (time.time() - start_time) / FLAGS.per_checkpoint
    else:
        data_train = load_data(FLAGS.data_dir, 'weibo_pair_train_Q_after')
        vocab, embed, question_words, _ = build_vocab(FLAGS.data_dir, data_train)
        keywords_list, keywords_index, PMI = load_PMI("all") 
        key_to_vocab = [0] * len(keywords_list)
        inverse_vocab = {}
        for i in range(len(vocab)):
            inverse_vocab[vocab[i]] = i
        print "mapping keywords to vocab..."
        for i in range(len(keywords_list)):
            if inverse_vocab.has_key(keywords_list[i]):
                key_to_vocab[i] = inverse_vocab[keywords_list[i]]
            if i % 1000 == 0:
                print "    processing line %d" % i
        model = Seq2SeqModel(
                FLAGS.symbols,
                len(question_words),
                FLAGS.embed_units,
                FLAGS.units,
                FLAGS.layers,
                is_train=False,
                vocab=vocab,
                embed=embed)

        if FLAGS.check_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.check_version)
        print('restore from %s' % model_path)
        model.saver.restore(sess, model_path)
        model.symbol2index.init.run()

        def split(sent):
            if Global == None:
                return sent.split()

            sent = sent.decode('utf-8', 'ignore').encode('gbk', 'ignore')
            tuples = [(word.decode("gbk").encode("utf-8"), pos)
                    for word, pos in Global.GetTokenPos(sent)]
            return [each[0] for each in tuples]

        if FLAGS.inference_path == '':
            while True:
                sys.stdout.write('post: ')
                sys.stdout.flush()
                post = split(sys.stdin.readline())
                response = inference(model, sess, [post])[0]
                print('response: %s' % ''.join(response))
                sys.stdout.flush()
        else:
            posts = []
            with open(FLAGS.inference_path) as f:
                for line in f:
                    sent = line.strip().split('\t')[0]
                    posts.append(split(sent))

            responses = []
            st, ed = 0, FLAGS.batch_size
            tot = 0
            while st < len(posts):
                responses += inference(model, sess, posts[st: ed])
                st, ed = ed, ed+FLAGS.batch_size
                tot = tot + 1
                print tot
            with open(FLAGS.inference_path+'.out'+str(FLAGS.check_version), 'w') as f:
                for p, r in zip(posts, responses):
                    f.writelines('%s\n' % ' '.join(r))
