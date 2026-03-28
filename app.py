import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import re
import gensim
from gensim.models import Word2Vec, FastText
import gensim.downloader as api

# 全局页面配置
st.set_page_config(page_title="语义分析综合测试平台", layout="wide", page_icon="📊")

# 依赖检查与下载
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    st.warning(f"NLTK数据下载失败: {e}")

# 内置默认语料
DEFAULT_TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.

AI applications include advanced web search engines (e.g., Google), recommendation systems (used by YouTube, Amazon and Netflix), understanding human speech (such as Siri and Amazon Alexa), self-driving cars (e.g., Tesla), automated decision-making and competing at the highest level in strategic game systems (such as chess and Go). As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.

The various sub-fields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects. General intelligence (the ability to solve an arbitrary problem) is among the field's long-term goals. To solve these problems, AI researchers have adapted and integrated a wide range of problem-solving techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, probability and economics. AI also draws upon computer science, psychology, linguistics, philosophy, and many other fields.

The field was founded on the assumption that human intelligence "can be so precisely described that a machine can be made to simulate it". This raises philosophical arguments about the mind and the ethics of creating artificial beings endowed with human-like intelligence, issues which have been explored by myth, fiction and philosophy since antiquity. Computer scientists and philosophers have since suggested that AI may become an existential risk to humanity if its rational capacities are not steered towards beneficial goals.
"""


# 预处理函数
@st.cache_data
def preprocess_text(text):
    """预处理文本：分句、分词、去停用词、去标点"""
    try:
        # 分句
        sentences = sent_tokenize(text)

        # 获取停用词
        try:
            stop_words = set(stopwords.words("english"))
        except:
            stop_words = set()

        # 标点符号
        punctuation = set(string.punctuation)

        processed_sentences = []
        for sentence in sentences:
            # 分词
            words = word_tokenize(sentence.lower())
            # 过滤停用词和标点
            filtered_words = [
                word
                for word in words
                if word not in stop_words
                and word not in punctuation
                and word.isalpha()
                and len(word) > 1
            ]
            if filtered_words:
                processed_sentences.append(" ".join(filtered_words))

        return processed_sentences, sentences
    except Exception as e:
        st.error(f"文本预处理失败: {e}")
        return [], []


# 侧边栏配置
with st.sidebar:
    st.title("📊 语义分析综合测试平台")
    st.markdown("---")

    # 全局语料输入
    st.subheader("📝 全局语料输入")
    default_text = st.text_area(
        "请输入英文文本（用于所有模块）：",
        value=DEFAULT_TEXT,
        height=300,
        help="请输入英文文本，建议500-1000词",
    )

    st.markdown("---")

    # Tab 2 参数配置
    st.subheader("⚙️ Word2Vec 参数配置")
    architecture = st.radio("训练架构：", ["CBOW (sg=0)", "Skip-Gram (sg=1)"], index=0)
    window_size = st.slider(
        "上下文窗口大小：",
        min_value=2,
        max_value=10,
        value=5,
        help="上下文窗口大小，范围2-10",
    )

# 主界面标签页
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 传统统计模型", "🔤 Word2Vec", "🌐 GloVe 词类比", "🚀 FastText & Sent2Vec"]
)

# Tab 1: 传统统计模型 (TF-IDF与LSA)
with tab1:
    with st.expander("📝 算法原理与实验观察任务", expanded=False):
        st.markdown("""
        **TF-IDF (Term Frequency-Inverse Document Frequency)**:
        - 衡量词在文档中的重要性
        - TF: 词频，IDF: 逆文档频率
        - 公式: TF-IDF = TF × IDF
        
        **LSA (Latent Semantic Analysis)**:
        - 基于SVD的降维技术
        - 将高维词向量降维到2D空间
        - 发现词与词之间的潜在语义关系
        
        **实验观察**:
        - 观察TF-IDF权重最高的关键词
        - 观察词向量在2D空间中的分布模式
        """)

    st.header("Tab 1: 传统统计模型 (TF-IDF与LSA)")

    if not default_text.strip():
        st.warning("请先在侧边栏输入文本内容")
    else:
        try:
            with st.spinner("正在处理文本..."):
                processed_sentences, raw_sentences = preprocess_text(default_text)

                if not processed_sentences:
                    st.error("文本处理失败，请检查输入内容")
                else:
                    st.success(f"成功处理 {len(processed_sentences)} 个句子")

                    # TF-IDF分析
                    st.subheader("📊 TF-IDF 分析")

                    with st.spinner("正在计算TF-IDF矩阵..."):
                        vectorizer = TfidfVectorizer(max_features=1000)
                        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
                        feature_names = vectorizer.get_feature_names_out()

                        # 计算每个词的总TF-IDF权重
                        tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
                        word_scores = pd.DataFrame(
                            {"word": feature_names, "tfidf_score": tfidf_scores}
                        ).sort_values("tfidf_score", ascending=False)

                        # 显示Top 5关键词
                        st.write("**TF-IDF权重最高的5个关键词:**")
                        top_words = word_scores.head(5)
                        st.dataframe(top_words, use_container_width=True)

                    # LSA降维与可视化
                    st.subheader("🔍 LSA 降维与可视化")

                    with st.spinner("正在进行LSA降维..."):
                        # 使用TruncatedSVD进行降维
                        svd = TruncatedSVD(n_components=2, random_state=42)
                        lsa_result = svd.fit_transform(tfidf_matrix.T)  # 转置以词为维度

                        # 创建词向量数据框
                        word_vectors = pd.DataFrame(
                            {
                                "word": feature_names,
                                "x": lsa_result[:, 0],
                                "y": lsa_result[:, 1],
                                "tfidf_score": tfidf_scores,
                            }
                        )

                        # 选取前50-100个词进行可视化
                        top_n = min(80, len(word_vectors))
                        top_words_for_viz = word_vectors.nlargest(top_n, "tfidf_score")

                        # 创建2D散点图
                        fig = px.scatter(
                            top_words_for_viz,
                            x="x",
                            y="y",
                            text="word",
                            size="tfidf_score",
                            color="tfidf_score",
                            color_continuous_scale="Viridis",
                            title="LSA 2D 词向量可视化 (Top 80 高权重词)",
                            labels={"x": "LSA Component 1", "y": "LSA Component 2"},
                        )

                        # 调整文本位置避免重叠
                        fig.update_traces(
                            textposition="top center", marker=dict(size=10, opacity=0.7)
                        )
                        fig.update_layout(
                            height=600,
                            showlegend=False,
                            xaxis_title="LSA Component 1",
                            yaxis_title="LSA Component 2",
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # 显示解释方差比
                        explained_variance = svd.explained_variance_ratio_
                        st.info(
                            f"LSA前两个主成分解释的方差比例: {explained_variance[0]:.2%} + {explained_variance[1]:.2%} = {sum(explained_variance):.2%}"
                        )

        except Exception as e:
            st.error(f"处理过程中出现错误: {e}")

# Tab 2: Word2Vec训练与对比
with tab2:
    with st.expander("📝 算法原理与实验观察任务", expanded=False):
        st.markdown("""
        **Word2Vec**:
        - 将词映射到稠密向量空间
        - 两种架构: CBOW和Skip-Gram
        - CBOW: 用上下文预测中心词
        - Skip-Gram: 用中心词预测上下文
        
        **余弦相似度**:
        - 衡量两个向量的相似程度
        - 范围: -1到1，1表示最相似
        
        **实验观察**:
        - 比较CBOW和Skip-Gram的结果差异
        - 观察窗口大小对词向量的影响
        """)

    st.header("Tab 2: Word2Vec 训练与对比")

    if not default_text.strip():
        st.warning("请先在侧边栏输入文本内容")
    else:
        try:
            # 预处理文本为Word2Vec格式
            @st.cache_data
            def prepare_word2vec_data(text):
                """准备Word2Vec训练数据"""
                sentences = sent_tokenize(text)
                tokenized_sentences = []

                try:
                    stop_words = set(stopwords.words("english"))
                except:
                    stop_words = set()

                punctuation = set(string.punctuation)

                for sentence in sentences:
                    words = word_tokenize(sentence.lower())
                    filtered_words = [
                        word
                        for word in words
                        if word not in stop_words
                        and word not in punctuation
                        and word.isalpha()
                        and len(word) > 1
                    ]
                    if filtered_words:
                        tokenized_sentences.append(filtered_words)

                return tokenized_sentences

            with st.spinner("正在准备训练数据..."):
                training_data = prepare_word2vec_data(default_text)

                if not training_data:
                    st.error("训练数据准备失败")
                else:
                    st.success(f"准备了 {len(training_data)} 个训练句子")

                    # 训练Word2Vec模型
                    st.subheader("🔧 模型训练")

                    with st.spinner("正在训练Word2Vec模型..."):
                        # 根据选择设置参数
                        sg_param = 1 if "Skip-Gram" in architecture else 0

                        model_w2v = Word2Vec(
                            sentences=training_data,
                            vector_size=100,
                            window=window_size,
                            min_count=1,
                            workers=4,
                            sg=sg_param,
                            epochs=10,
                        )

                        st.success(
                            f"Word2Vec模型训练完成 (架构: {architecture}, 窗口: {window_size})"
                        )

                    # 词义检索
                    st.subheader("🔍 词义相似度检索")

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        search_word = st.text_input(
                            "输入目标单词：",
                            value="artificial",
                            help="输入一个英文单词，查找最相似的词",
                        )
                    with col2:
                        search_button = st.button("🔍 检索", type="primary")

                    if search_button and search_word:
                        try:
                            with st.spinner(
                                f"正在查找与 '{search_word}' 最相似的词..."
                            ):
                                if search_word.lower() in model_w2v.wv:
                                    similar_words = model_w2v.wv.most_similar(
                                        search_word.lower(), topn=5
                                    )

                                    # 转换为DataFrame
                                    similar_df = pd.DataFrame(
                                        similar_words, columns=["word", "similarity"]
                                    )
                                    similar_df["similarity"] = similar_df[
                                        "similarity"
                                    ].round(4)

                                    st.write(f"**与 '{search_word}' 最相似的5个词:**")
                                    st.dataframe(similar_df, use_container_width=True)

                                    # 可视化相似度
                                    fig = px.bar(
                                        similar_df,
                                        x="word",
                                        y="similarity",
                                        title=f"与 '{search_word}' 的余弦相似度",
                                        color="similarity",
                                        color_continuous_scale="Viridis",
                                    )
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)

                                else:
                                    st.warning(f"单词 '{search_word}' 不在词表中")

                        except Exception as e:
                            st.error(f"检索过程中出现错误: {e}")

                    # 模型信息
                    st.subheader("📊 模型信息")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("词表大小", len(model_w2v.wv))
                    with col2:
                        st.metric("向量维度", model_w2v.wv.vector_size)
                    with col3:
                        st.metric("训练句子数", len(training_data))

        except Exception as e:
            st.error(f"Word2Vec处理过程中出现错误: {e}")

# Tab 3: 预训练模型与词类比 (GloVe)
with tab3:
    with st.expander("📝 算法原理与实验观察任务", expanded=False):
        st.markdown("""
        **GloVe (Global Vectors for Word Representation)**:
        - 基于全局词共现统计的词向量方法
        - 结合了矩阵分解和局部上下文窗口的优点
        - 预训练模型包含丰富的语义知识
        
        **词类比 (Word Analogy)**:
        - 经典示例: king - man + woman ≈ queen
        - 利用向量运算进行语义推理
        - 公式: Result = Vector(A) - Vector(B) + Vector(C)
        
        **实验观察**:
        - 观察词类比的准确性
        - 比较不同词对的相似度
        """)

    st.header("Tab 3: 预训练模型与词类比 (GloVe)")

    # 加载预训练GloVe模型
    @st.cache_resource
    def load_glove_model():
        """加载预训练GloVe模型"""
        with st.spinner("正在下载/加载GloVe模型，请稍候..."):
            try:
                # 加载轻量级GloVe模型
                model = api.load("glove-twitter-25")
                return model
            except Exception as e:
                st.error(f"模型加载失败: {e}")
                return None

    glove_model = load_glove_model()

    if glove_model:
        st.success("GloVe模型加载成功！")

        # 功能1：词类比
        st.subheader("🔄 词类比 (Word Analogy)")
        st.markdown("经典示例: king - man + woman ≈ queen")

        col1, col2, col3 = st.columns(3)
        with col1:
            word_a = st.text_input("词 A:", value="king", key="word_a")
        with col2:
            word_b = st.text_input("词 B:", value="man", key="word_b")
        with col3:
            word_c = st.text_input("词 C:", value="woman", key="word_c")

        analogy_button = st.button("🧮 计算类比", type="primary")

        if analogy_button:
            try:
                with st.spinner("正在进行词类比计算..."):
                    # 检查所有词是否在词表中
                    missing_words = []
                    for word in [word_a, word_b, word_c]:
                        if word.lower() not in glove_model:
                            missing_words.append(word)

                    if missing_words:
                        st.warning(f"以下单词不在词表中: {', '.join(missing_words)}")
                    else:
                        # 执行词类比: A - B + C
                        result = glove_model.most_similar(
                            positive=[word_a.lower(), word_c.lower()],
                            negative=[word_b.lower()],
                            topn=3,
                        )

                        st.write(f"**词类比结果: {word_a} - {word_b} + {word_c} ≈ ?**")

                        # 显示结果
                        result_df = pd.DataFrame(result, columns=["word", "similarity"])
                        result_df["similarity"] = result_df["similarity"].round(4)
                        st.dataframe(result_df, use_container_width=True)

                        # 可视化
                        fig = px.bar(
                            result_df,
                            x="word",
                            y="similarity",
                            title=f"词类比结果: {word_a} - {word_b} + {word_c}",
                            color="similarity",
                            color_continuous_scale="Viridis",
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"词类比计算失败: {e}")

        st.markdown("---")

        # 功能2：词义相似度
        st.subheader("📏 词义相似度")

        col1, col2 = st.columns(2)
        with col1:
            word1 = st.text_input("单词 1:", value="artificial", key="word1")
        with col2:
            word2 = st.text_input("单词 2:", value="intelligence", key="word2")

        similarity_button = st.button("📊 计算相似度", type="primary")

        if similarity_button:
            try:
                with st.spinner("正在计算相似度..."):
                    # 检查单词是否在词表中
                    missing = []
                    for w in [word1, word2]:
                        if w.lower() not in glove_model:
                            missing.append(w)

                    if missing:
                        st.warning(f"以下单词不在词表中: {', '.join(missing)}")
                    else:
                        # 计算相似度
                        similarity = glove_model.similarity(
                            word1.lower(), word2.lower()
                        )

                        # 显示结果
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("相似度分数", f"{similarity:.4f}")
                        with col2:
                            # 解释相似度
                            if similarity > 0.7:
                                interpretation = "高度相似"
                            elif similarity > 0.4:
                                interpretation = "中度相似"
                            elif similarity > 0.1:
                                interpretation = "低度相似"
                            else:
                                interpretation = "几乎不相似"
                            st.metric("相似度等级", interpretation)

                        # 可视化相似度
                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=similarity,
                                domain={"x": [0, 1], "y": [0, 1]},
                                title={"text": f"'{word1}' 与 '{word2}' 的相似度"},
                                gauge={
                                    "axis": {"range": [-1, 1]},
                                    "bar": {"color": "darkblue"},
                                    "steps": [
                                        {"range": [-1, 0], "color": "red"},
                                        {"range": [0, 0.5], "color": "yellow"},
                                        {"range": [0.5, 1], "color": "green"},
                                    ],
                                    "threshold": {
                                        "line": {"color": "red", "width": 4},
                                        "thickness": 0.75,
                                        "value": similarity,
                                    },
                                },
                            )
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"相似度计算失败: {e}")

        # 模型信息
        st.markdown("---")
        st.subheader("📊 模型信息")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("词表大小", len(glove_model))
        with col2:
            st.metric("向量维度", glove_model.vector_size)

    else:
        st.error("无法加载GloVe模型，请检查网络连接")

# Tab 4: 子词特征与句向量 (FastText & Sent2Vec)
with tab4:
    with st.expander("📝 算法原理与实验观察任务", expanded=False):
        st.markdown("""
        **FastText**:
        - 基于子词(subword)的词向量方法
        - 将词分解为字符n-gram
        - 能处理未登录词(OOV)问题
        
        **Sent2Vec (基于平均池化)**:
        - 将句子中所有词向量取平均
        - 得到句子的整体向量表示
        - 简单有效的句向量方法
        
        **OOV鲁棒性**:
        - Word2Vec: 遇到未登录词会报错
        - FastText: 通过子词信息生成向量
        
        **实验观察**:
        - 比较Word2Vec和FastText对OOV的处理
        - 观察句向量相似度的合理性
        """)

    st.header("Tab 4: FastText & Sent2Vec")

    if not default_text.strip():
        st.warning("请先在侧边栏输入文本内容")
    else:
        try:
            # 准备训练数据
            @st.cache_data
            def prepare_fasttext_data(text):
                """准备FastText训练数据"""
                sentences = sent_tokenize(text)
                tokenized_sentences = []

                try:
                    stop_words = set(stopwords.words("english"))
                except:
                    stop_words = set()

                punctuation = set(string.punctuation)

                for sentence in sentences:
                    words = word_tokenize(sentence.lower())
                    filtered_words = [
                        word
                        for word in words
                        if word not in stop_words
                        and word not in punctuation
                        and word.isalpha()
                        and len(word) > 1
                    ]
                    if filtered_words:
                        tokenized_sentences.append(filtered_words)

                return tokenized_sentences

            with st.spinner("正在准备训练数据..."):
                training_data = prepare_fasttext_data(default_text)

                if not training_data:
                    st.error("训练数据准备失败")
                else:
                    st.success(f"准备了 {len(training_data)} 个训练句子")

                    # 训练FastText模型
                    st.subheader("🔧 FastText 模型训练")

                    with st.spinner("正在训练FastText模型..."):
                        model_ft = FastText(
                            sentences=training_data,
                            vector_size=100,
                            window=5,
                            min_count=1,
                            workers=4,
                            epochs=10,
                            min_n=3,  # 最小n-gram长度
                            max_n=6,  # 最大n-gram长度
                        )

                        st.success("FastText模型训练完成！")

                    # 功能1：OOV鲁棒性测试
                    st.subheader("🧪 OOV 鲁棒性测试")
                    st.markdown("比较Word2Vec和FastText对未登录词的处理能力")

                    oov_word = st.text_input(
                        "输入一个带有拼写错误的词（如 'computeer'）：",
                        value="computeer",
                        help="测试FastText对未登录词的处理能力",
                    )

                    oov_button = st.button("🧪 测试OOV处理", type="primary")

                    if oov_button and oov_word:
                        try:
                            with st.spinner("正在测试OOV处理..."):
                                col1, col2 = st.columns(2)

                                # 测试Word2Vec (使用Tab2的模型)
                                with col1:
                                    st.write("**Word2Vec 结果:**")
                                    try:
                                        # 重新训练Word2Vec模型用于测试
                                        sg_param = (
                                            1 if "Skip-Gram" in architecture else 0
                                        )
                                        test_w2v = Word2Vec(
                                            sentences=training_data,
                                            vector_size=100,
                                            window=window_size,
                                            min_count=1,
                                            workers=4,
                                            sg=sg_param,
                                            epochs=10,
                                        )

                                        if oov_word.lower() in test_w2v.wv:
                                            similar_w2v = test_w2v.wv.most_similar(
                                                oov_word.lower(), topn=3
                                            )
                                            st.dataframe(
                                                pd.DataFrame(
                                                    similar_w2v,
                                                    columns=["word", "similarity"],
                                                ),
                                                use_container_width=True,
                                            )
                                        else:
                                            st.error("Word2Vec 报错：未登录词 (OOV)")
                                    except KeyError:
                                        st.error("Word2Vec 报错：未登录词 (OOV)")
                                    except Exception as e:
                                        st.error(f"Word2Vec 错误: {e}")

                                # 测试FastText
                                with col2:
                                    st.write("**FastText 结果:**")
                                    try:
                                        if oov_word.lower() in model_ft.wv:
                                            similar_ft = model_ft.wv.most_similar(
                                                oov_word.lower(), topn=3
                                            )
                                            st.dataframe(
                                                pd.DataFrame(
                                                    similar_ft,
                                                    columns=["word", "similarity"],
                                                ),
                                                use_container_width=True,
                                            )
                                            st.success("FastText 成功处理OOV！")
                                        else:
                                            st.warning("FastText 也无法识别该词")
                                    except Exception as e:
                                        st.error(f"FastText 错误: {e}")

                        except Exception as e:
                            st.error(f"OOV测试失败: {e}")

                    st.markdown("---")

                    # 功能2：基于Average Pooling的Sent2Vec
                    st.subheader("📝 句向量相似度 (Sent2Vec)")
                    st.markdown("使用平均池化方法计算句子相似度")

                    col1, col2 = st.columns(2)
                    with col1:
                        sentence1 = st.text_area(
                            "句子 1:",
                            value="Artificial intelligence is transforming the world.",
                            height=100,
                            key="sentence1",
                        )
                    with col2:
                        sentence2 = st.text_area(
                            "句子 2:",
                            value="Machine learning is changing our daily lives.",
                            height=100,
                            key="sentence2",
                        )

                    sent_sim_button = st.button("📊 计算句子相似度", type="primary")

                    if sent_sim_button:
                        try:
                            with st.spinner("正在计算句子相似度..."):
                                # 预处理句子
                                def get_sentence_vector(sentence, model):
                                    """获取句子的向量表示（平均池化）"""
                                    try:
                                        stop_words = set(stopwords.words("english"))
                                    except:
                                        stop_words = set()

                                    punctuation = set(string.punctuation)
                                    words = word_tokenize(sentence.lower())
                                    filtered_words = [
                                        word
                                        for word in words
                                        if word not in stop_words
                                        and word not in punctuation
                                        and word.isalpha()
                                        and len(word) > 1
                                    ]

                                    if not filtered_words:
                                        return None

                                    # 获取词向量
                                    word_vectors = []
                                    for word in filtered_words:
                                        if word in model.wv:
                                            word_vectors.append(model.wv[word])

                                    if not word_vectors:
                                        return None

                                    # 平均池化
                                    sentence_vector = np.mean(word_vectors, axis=0)
                                    return sentence_vector

                                # 获取句子向量
                                vec1 = get_sentence_vector(sentence1, model_ft)
                                vec2 = get_sentence_vector(sentence2, model_ft)

                                if vec1 is None or vec2 is None:
                                    st.error("无法生成句子向量，请检查输入")
                                else:
                                    # 计算余弦相似度
                                    similarity = 1 - cosine(vec1, vec2)

                                    # 显示结果
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("相似度分数", f"{similarity:.4f}")
                                    with col2:
                                        # 解释相似度
                                        if similarity > 0.8:
                                            interpretation = "高度相似"
                                        elif similarity > 0.6:
                                            interpretation = "中高度相似"
                                        elif similarity > 0.4:
                                            interpretation = "中度相似"
                                        elif similarity > 0.2:
                                            interpretation = "低度相似"
                                        else:
                                            interpretation = "几乎不相似"
                                        st.metric("相似度等级", interpretation)

                                    # 可视化相似度
                                    fig = go.Figure(
                                        go.Indicator(
                                            mode="gauge+number",
                                            value=similarity,
                                            domain={"x": [0, 1], "y": [0, 1]},
                                            title={"text": "句子相似度"},
                                            gauge={
                                                "axis": {"range": [0, 1]},
                                                "bar": {"color": "darkblue"},
                                                "steps": [
                                                    {"range": [0, 0.4], "color": "red"},
                                                    {
                                                        "range": [0.4, 0.7],
                                                        "color": "yellow",
                                                    },
                                                    {
                                                        "range": [0.7, 1],
                                                        "color": "green",
                                                    },
                                                ],
                                                "threshold": {
                                                    "line": {
                                                        "color": "red",
                                                        "width": 4,
                                                    },
                                                    "thickness": 0.75,
                                                    "value": similarity,
                                                },
                                            },
                                        )
                                    )
                                    fig.update_layout(height=300)
                                    st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"句子相似度计算失败: {e}")

                    # 模型信息
                    st.markdown("---")
                    st.subheader("📊 模型信息")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("词表大小", len(model_ft.wv))
                    with col2:
                        st.metric("向量维度", model_ft.wv.vector_size)

        except Exception as e:
            st.error(f"FastText处理过程中出现错误: {e}")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 14px;">
        📊 语义分析综合测试平台 | 基于传统统计模型到分布式词向量的演进演示
    </div>
    """,
    unsafe_allow_html=True,
)
