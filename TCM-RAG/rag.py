import chromadb
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from indexer import build_from_json_files

def top_answer(question, n=5):
    """检索相关知识库内容，返回最相关的答案"""
    try:
        client = chromadb.PersistentClient(path="./tcm_chroma")
        coll = client.get_collection("tcm_json")
        hits = coll.query(query_texts=[question], n_results=n)
        
        if hits["metadatas"] and hits["metadatas"][0]:
            return hits["metadatas"][0][0]["output"]
        else:
            return "未找到相关信息"
    except Exception as e:
        print(f"查询过程中出错：{e}")
        return "查询过程中出现错误"

# 初始化LLM和提示模板
llm = ChatZhipuAI(model="glm-4.5", api_key="Your_key", timeout=400, max_retries=2)
prompt = PromptTemplate.from_template(
    "现在你是一名专业的中医药国医大师，已知中医药知识：{kb}\n\n请回答用户问题：{q}"
)
chain = LLMChain(prompt=prompt, llm=llm)

if __name__ == "__main__":
    # 构建向量数据库，使用TCM_SD文件夹
    # json_directory = "./TCM_SD"
    # build_from_json_files(json_directory)
    
    # 测试查询
    question = "某女患者月经经期错乱，经色紫暗，夹有血块，且少腹冷痛，形寒肢冷，舌紫暗，属于什么证候？解释一下该证候。"
    knowledge_base = top_answer(question)
    
    # 生成回答
    result = chain.invoke({"kb": knowledge_base, "q": question})
    print(f"问题：{question}")
    print(f"回答：{result['text']}")
