import chromadb
import json
import os
from tqdm import tqdm

def build_from_json_files(json_dir):
    # 创建客户端和集合
    client = chromadb.PersistentClient(path="./tcm_chroma")
    if "tcm_json" in {c.name for c in client.list_collections()}:
        print("删除已存在的集合...")
        client.delete_collection("tcm_json")
    
    coll = client.create_collection("tcm_json")
    print("创建新的向量库集合")
    
    if not os.path.exists(json_dir):
        print(f"错误：目录 {json_dir} 不存在")
        return

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        print(f"警告：目录 {json_dir} 中没有找到JSON文件")
        return

    print(f"找到 {len(json_files)} 个JSON文件: {json_files}")

    total_count = 0
    batch_size = 1000  # 分批处理，避免内存问题

    for json_file in json_files:
        file_path = os.path.join(json_dir, json_file)
        print(f"\n处理文件：{json_file}")
        
        file_count = 0
        docs, ids, metas = [], [], []

        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(tqdm(lines, desc=f"处理 {json_file}")):
                try:
                    line = line.strip()
                    if not line:
                        continue
                        
                    item = json.loads(line)
                    
                    if json_file == "syndrome_diag.json":
                        if not item.get('lcd_name') or not item.get('syndrome'):
                            print(f"警告：第 {line_num} 行缺少必要字段")
                            continue
                            
                        query_text = f"疾病：{item.get('lcd_name', '')}，证候：{item.get('syndrome', '')}，主诉：{item.get('chief_complaint', '')}"
                        meta_info = {
                            "output": f"疾病：{item.get('lcd_name', '')}\n证候：{item.get('syndrome', '')}\n主诉：{item.get('chief_complaint', '')}\n描述：{item.get('description', '')}\n四诊摘要：{item.get('detection', '')}",
                            "type": "diagnosis",
                            "lcd_name": item.get('lcd_name', ''),
                            "syndrome": item.get('syndrome', ''),
                            "source_file": json_file
                        }
                        doc_id = f"diag_{total_count}"

                    elif json_file == "syndrome_knowledge.json":
                        if not item.get('Name'):
                            print(f"警告：第 {line_num} 行缺少证候名称")
                            continue
                            
                        query_text = f"证候：{item.get('Name', '')}，定义：{item.get('Definition', '')[:100]}..."
                        meta_info = {
                            "output": f"证候名称：{item.get('Name', '')}\n定义：{item.get('Definition', '')}\n典型表现：{item.get('Typical_performance', '')}\n常见疾病：{item.get('Common_isease', '')}",
                            "type": "knowledge",
                            "syndrome_name": item.get('Name', ''),
                            "source_file": json_file
                        }
                        doc_id = f"knowledge_{total_count}"

                    else:
                        query_text = str(item)[:200]  # 截断避免过长
                        meta_info = {
                            "output": str(item),
                            "type": "other",
                            "source_file": json_file
                        }
                        doc_id = f"other_{total_count}"

                    docs.append(query_text)
                    metas.append(meta_info)
                    ids.append(doc_id)
                    total_count += 1
                    file_count += 1

                    # 分批写入
                    if len(docs) >= batch_size:
                        coll.add(documents=docs, metadatas=metas, ids=ids)
                        docs, ids, metas = [], [], []
                        print(f"已写入 {batch_size} 条记录")

                except json.JSONDecodeError as e:
                    print(f"JSON解析错误，文件 {json_file} 第 {line_num} 行: {e}")
                    continue
                except Exception as e:
                    print(f"处理文件 {json_file} 第 {line_num} 行时出错: {e}")
                    continue
            if docs:
                coll.add(documents=docs, metadatas=metas, ids=ids)
                print(f"写入剩余 {len(docs)} 条记录")

            print(f"文件 {json_file} 处理完成，共 {file_count} 条记录")

        except Exception as e:
            print(f"打开或读取文件 {json_file} 时出错：{e}")
            continue

    # 最终统计
    print(f"\n向量库构建完成！总共处理了 {total_count} 条记录")
    
    # 验证集合中的记录数量
    collection_count = coll.count()
    print(f"向量库中实际存储的记录数量: {collection_count}")
    
    if total_count != collection_count:
        print(f"警告：处理记录数({total_count})与向量库记录数({collection_count})不一致")