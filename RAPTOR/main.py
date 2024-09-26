import os
import sys
sys.path.append(os.getcwd())
import umap
import argparse
import numpy as np
import pandas as pd

from uuid import uuid4
from bs4 import BeautifulSoup
from sklearn.mixture import GaussianMixture
from typing import Optional, Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

from util.prompt import PromptHelper
from util.chain import return_chain


# Init
URL = "https://namu.wiki/w/%ED%94%84%EB%9E%91%ED%82%A4"
emb = OpenAIEmbeddings()
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
vectorstore = Chroma(collection_name = "rag-chroma4", persist_directory = "/data/dev/Advanced-RAG/data/doc4", embedding_function = emb)


def global_cluster_embeddings(embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine") -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    
    return umap.UMAP(n_neighbors = n_neighbors ,n_components = dim, metric = metric).fit_transform(embeddings)
    
def local_cluster_embeddings(embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine") -> np.ndarray:
    
    return umap.UMAP(n_neighbors = num_neighbors, n_components = dim, metric = metric).fit_transform(embeddings)

def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, ramdom_state: int = 123) -> int:

    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    
    bics = []
    for n in n_clusters:
        gmm = GaussianMixture(n, random_state = ramdom_state)
        gmm.fit(embeddings)
        bics.append(gmm.bic(embeddings))
    
    return n_clusters[np.argmin(bics)]

def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gmm = GaussianMixture(n_components=n_clusters, random_state = random_state)
    gmm.fit(embeddings)
    probs = gmm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    
    return labels, n_clusters

def perform_clustering(embeddings: np.ndarray, dim: int, threshold: float) -> List[np.ndarray]:
    if len(embeddings) <= dim + 1:
        # avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]
    
    # global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # global clustering
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)
    
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0
    
    # iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # extractg embeddings belonging to the current global cluster
        global_cluster_embeddings_ = embeddings[np.array([i in gc for gc in global_clusters])]
        
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # local dimensionality reduction and clustering
            reduced_embedding_local = local_cluster_embeddings(global_cluster_embeddings_, dim)
            local_clusters, n_local_clusters = GMM_cluster(reduced_embedding_local, threshold)
            
        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[np.array([j in lc for lc in local_clusters])]
            indices = np.where((embeddings == local_cluster_embeddings_[:, None]).all(-1))[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], j+total_clusters)
                
        total_clusters += n_local_clusters
        
    return all_local_clusters

def embed(texts) -> np.ndarray:
    text_embeddings = emb.embed_documents(texts)
    arr_text_embeddings = np.array(text_embeddings)
    return arr_text_embeddings

def embed_cluster_texts(texts) -> pd.DataFrame:
    arr_text_embeddings = embed(texts)
    cluster_labels = perform_clustering(arr_text_embeddings, dim = 10, threshold = 0.1)
    
    df = pd.DataFrame()
    df['text'] = texts
    df['embed'] = list(arr_text_embeddings)
    df['cluster'] = cluster_labels
    
    return df

def fmt_txt(df: pd.DataFrame) -> str:
    unique_txt = df['text'].tolist()
    return "--- --- \n --- --- ".join(unique_txt)

def embed_cluster_summarize_texts(texts: List[str], level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # embed and cluster the texts, resulting in a DataFrame with 'text', 'embed', 'cluster' columns
    df_clusters = embed_cluster_texts(texts)
    
    # prepare to expand the DataFrame for easier manipulation of clusters
    expanded_list = []
    
    # expand DataFrame entries to doc-cluster pairings for straightforward processing
    for idx, row in df_clusters.iterrows():
        for cluster in row['cluster']:
            expanded_list.append(
                {'text': row['text'], 'embed': row['embed'], 'cluster': cluster}
            )
            
    # create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)
    
    # retrieve unique cluster indentifiers for processing
    all_clusters = expanded_df['cluster'].unique()
    
    print(f"--Generated {len(all_clusters)} clusters--")
    
    # Summarization - Apply into PromptHelper template
    sum_template = PromptHelper.get_prompt("raptor_summarization")
    sum_prompt = ChatPromptTemplate.from_template(sum_template)
    
    sum_chain = return_chain(
        prompt = sum_prompt, 
        llm = llm, 
        parser = StrOutputParser()
    )
    
    # format text within each cluster for summarization
    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df['cluster'] == i]
        formatted_txt = fmt_txt(df_cluster)
        summaries.append(sum_chain.invoke({"documentation": formatted_txt}))
        
    # create a DataFrame to store summaries with their corresponding cluster and level
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters)
        }
    )
    
    return df_clusters, df_summary

def recursive_embed_cluster_summarize(texts: List[str], level: int = 1, n_levels: int = 3) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    
    results = {}    # Dictionary to store results at each level
    
    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)
    
    # store the results of the current level
    results[level] = (df_clusters, df_summary)
    
    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary['cluster'].nunique()
    
    if level < n_levels and unique_clusters > 1:
        # use summaries as the input texts for the next level of recursion
        new_texts = df_summary['summaries'].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level+1, n_levels
        )
        
        # merge the results from the next into the current results dictionary
        results.update(next_level_results)
        
    return results

def simplify_collapsed_tree_retrieval(doc: List[str], query: str, level: int = 1, n_levels: int = 3) -> None:
    
    # generating summarization
    results = recursive_embed_cluster_summarize(doc, level = level, n_levels = n_levels)
    
    new_texts = []
    
    for level in sorted(results.keys()):
        summaries = results[level][1]['summaries'].tolist()
        # all_texts.extend(summaries)
        new_texts.extend(summaries)
        
    # Create New Vectorstore
    new_text = [
        Document(
            metadata = {"source": "RAPTOR"},
            page_content = txt
        ) for txt in new_texts
    ]
    
    old_text = [
        Document(
            metadata = {"source": "db"},
            page_content = txt
        ) for txt in vectorstore.get().get("documents")
    ]
    
    old_text.extend(new_text)
    
    new_vectorstore = Chroma.from_documents(
        documents = old_text, embedding = emb, collection_name = "RAPTOR-DB"
    )
    retriever = new_vectorstore.as_retriever(search_kwargs = {"k": 3})
    
    rag_template = PromptHelper.get_prompt("generation_system")
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    
    rag_chain = return_chain(
        prompt = rag_prompt,
        llm = llm,
        parser = StrOutputParser(),
        retrieval_dict = {"context": retriever, "question": RunnablePassthrough()}
    )
    
    print(rag_chain.invoke(query))
    
    

def main(args):
    
    # loader = RecursiveUrlLoader(
    #     url = URL,
    #     max_depth = 2,
    #     extractor = lambda x: BeautifulSoup(x, "html.parser").text
    # )
    # doc = loader.load()
    # doc_texts = [d.page_content for d in doc]
    
    doc_texts = vectorstore.get().get("documents")
    
    if not doc_texts:
        raise ValueError("vectorstore data is empty!")
    
    simplify_collapsed_tree_retrieval(doc = doc_texts, query = args.query)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type = str)
    args = parser.parse_args()
    
    main(args)