from gensim.models import word2vec

def train(model_dir, data_path):

    num_features = 100    # Word vector dimensionality
    min_word_count = 1   # Minimum word count
    num_workers = 16       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    sentences = word2vec.Text8Corpus(data_path)

    model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sg = 1, sample = downsampling)
    model.init_sims(replace=True)
    # 保存模型，供日後使用
    model.save(model_dir)

    # 可以在加载模型之后使用另外的句子来进一步训练模型
    # model = gensim.models.Word2Vec.load('/tmp/mymodel')
    # model.train(more_sentences)

def test_model(model_dir):
    model = word2vec.Word2Vec.load(model_dir)      #模型讀取方式
    sim_to_word = model.most_similar(positive=['美国', '孩子'], negative=['中国']) #根据给定的条件推断相似词
    print('sim to word:', sim_to_word)
    #doesnt_match = model.doesnt_match("breakfast cereal dinner lunch".split()) #寻找离群词
    #print('doesnt_match:', doesnt_match)
    print('sim of woman and man:', model.similarity('国家', '中国')) #计算两个单词的相似度
    print('sim of woman and man:', model.similarity('美国', '中国')) #计算两个单词的相似度
    print('computer:', model['孩子']) #获取单词的词向量

if __name__ == "__main__":
    data_path = "toutiao_cmt_cut_text_0101_0811.1w.data"
    model_dir = "cmt_cut_w2v"
    train(model_dir, data_path)
    test_model(model_dir)
