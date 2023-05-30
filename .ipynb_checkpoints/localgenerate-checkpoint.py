from configs.model_config import *
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from utils import torch_gc
from tqdm import tqdm
import datetime
from pypinyin import lazy_pinyin
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from textsplitter import ChineseTextSplitter, AliTextSplitter
from loader import UnstructuredPaddleImageLoader, UnstructuredPaddlePDFLoader
from langchain.vectorstores import FAISS

class LocalGenerate:
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K

    def init(self,
            embedding_model: str = EMBEDDING_MODEL,
            embedding_device= EMBEDDING_DEVICE,
            filepath: str="",
            top_k=VECTOR_SEARCH_TOP_K,
            ):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k
        self.init_knowledge_vector_store(filepath)

    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],
                                    sentence_size=SENTENCE_SIZE):
        vs_path: str = None
        loaded_files = []
        failed_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = self.load_file(filepath, sentence_size)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
                    return
            elif os.path.isdir(filepath):
                docs = []
                for file in tqdm(os.listdir(filepath), desc="加载文件"):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += self.load_file(fullfilepath, sentence_size)
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        logger.error(e)
                        failed_files.append(file)

                if len(failed_files) > 0:
                    logger.info("以下文件未能成功加载：")
                    for file in failed_files:
                        logger.info(f"{file}\n")

        else:
            docs = []
            for file in filepath:
                try:
                    docs += self.load_file(file)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
        if len(docs) > 0:
            logger.info("文件加载完毕，正在生成向量库")
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                vector_store.add_documents(docs)
                torch_gc()
            else:
                if not vs_path:
                    vs_path = os.path.join(VS_ROOT_PATH,
                                           f"""{"".join(lazy_pinyin(os.path.splitext(file)[0]))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
                vector_store = FAISS.from_documents(docs, self.embeddings)  # docs 为Document列表
                torch_gc()

            vector_store.save_local(vs_path)
            print(vs_path, loaded_files)
        else:
            logger.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
        
        return

    def load_file(self, filepath, sentence_size=SENTENCE_SIZE):
        if filepath.lower().endswith(".md"):
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()
        elif filepath.lower().endswith(".txt"):
            loader = TextLoader(filepath, autodetect_encoding=True)
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(textsplitter)
        elif filepath.lower().endswith(".pdf"):
            loader = UnstructuredPaddlePDFLoader(filepath)
            textsplitter = AliTextSplitter(pdf=True)
            docs = loader.load_and_split(textsplitter)
        elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
            loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(text_splitter=textsplitter)
        else:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
            docs = loader.load_and_split(text_splitter=textsplitter)
        self.write_check_file(filepath, docs)
        return docs

    def write_check_file(self, filepath, docs):
        folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        fp = os.path.join(folder_path, 'load_file.txt')
        with open(fp, 'a+', encoding='utf-8') as fout:
            fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
            fout.write('\n')
            for i in docs:
                fout.write(str(i))
                fout.write('\n')
            fout.close()
        
if __name__ == "__main__":
    
    localgen = LocalGenerate()
    localgen.init(filepath = "../pdf/Audi-A6.pdf")