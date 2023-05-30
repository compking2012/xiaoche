from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from configs.model_config import *
import datetime
from textsplitter import ChineseTextSplitter, AliTextSplitter
from typing import List, Tuple, Dict
from langchain.docstore.document import Document
import numpy as np
from utils import torch_gc
from tqdm import tqdm
from pypinyin import lazy_pinyin
from loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import models.shared as shared
from agent import bing_search
from langchain.docstore.document import Document


def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template: str = PROMPT_TEMPLATE, ) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
) -> List[Tuple[Document, float]]:
    scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
    docs = []
    id_set = set()
    store_len = len(self.index_to_docstore_id)
    for j, i in enumerate(indices[0]):
        if i == -1 or 0 < self.score_threshold < scores[0][j]:
            # This happens when not enough docs are returned.
            continue
        _id = self.index_to_docstore_id[i]
        doc = self.docstore.search(_id)
        if not self.chunk_conent:
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc.metadata["score"] = int(scores[0][j])
            docs.append(doc)
            continue
        id_set.add(i)
        docs_len = len(doc.page_content)
        for k in range(1, max(i, store_len - i)):
            break_flag = False
            for l in [i + k, i - k]:
                if 0 <= l < len(self.index_to_docstore_id):
                    _id0 = self.index_to_docstore_id[l]
                    doc0 = self.docstore.search(_id0)
                    if docs_len + len(doc0.page_content) > self.chunk_size:
                        break_flag = True
                        break
                    elif doc0.metadata["source"] == doc.metadata["source"]:
                        docs_len += len(doc0.page_content)
                        id_set.add(l)
            if break_flag:
                break
    if not self.chunk_conent:
        return docs
    if len(id_set) == 0 and self.score_threshold > 0:
        return []
    id_list = sorted(list(id_set))
    id_lists = seperate_list(id_list)
    for id_seq in id_lists:
        for id in id_seq:
            if id == id_seq[0]:
                _id = self.index_to_docstore_id[id]
                doc = self.docstore.search(_id)
            else:
                _id0 = self.index_to_docstore_id[id]
                doc0 = self.docstore.search(_id0)
                doc.page_content += " " + doc0.page_content
        if not isinstance(doc, Document):
            raise ValueError(f"Could not find document for id {_id}, got {doc}")
        doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
        doc.metadata["score"] = int(doc_score)
        docs.append(doc)
    torch_gc()
    return docs

class LocalLlmQA:
    llm: BaseAnswer = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_conent: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD

    def init_cfg(self,
                 llm_model: BaseAnswer = None,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 ):
        self.llm = llm_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k

    def get_knowledge_based_answer(self, query, vs_path, chat_history=[], streaming: bool = STREAMING):
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_conent = self.chunk_conent
        vector_store.score_threshold = self.score_threshold
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        torch_gc()
        prompt = generate_prompt(related_docs_with_score, query)

        # print("prompt=", prompt)
        # print("history=", chat_history)
        
        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": related_docs_with_score}
            yield response, history
