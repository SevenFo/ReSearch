from typing import List
import requests
import torch
import logging
from transformers import T5ForSequenceClassification
from transformers import T5Tokenizer

from flashrag.retriever.retriever import retry
import gc  # 导入垃圾回收模块，虽然主要靠empty_cache


class CRAGEvaluator:
    def __init__(self, evaluator_path, device=None, refine=False):
        self.tokenizer = T5Tokenizer.from_pretrained(evaluator_path)
        self.model = T5ForSequenceClassification.from_pretrained(
            evaluator_path, num_labels=1
        )
        self.device = (
            (torch.device(device) if torch.cuda.is_available() else torch.device("cpu"))
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        self.refine = refine

    def extract_strips_from_psg(self, psg, mode="selection"):
        """将文档分解成更小的片段"""
        # 基于scripts/internal_knowledge_preparation.py中的实现
        if mode == "selection":
            return [psg]  # 最简单的模式，直接返回原始文档
        elif mode == "fixed_num":
            # 按固定单词数分解
            final_strips = []
            window_length = 50
            words = psg.split(" ")
            buf = []
            for w in words:
                buf.append(w)
                if len(buf) == window_length:
                    final_strips.append(" ".join(buf))
                    buf = []
            if buf:
                if len(buf) < 10:
                    final_strips[-1] += " " + " ".join(buf)
                else:
                    final_strips.append(" ".join(buf))
            return final_strips
        elif mode == "excerption":
            # 按句子分解
            strips = []
            # 实现省略，可参考internal_knowledge_preparation.py
            return strips

    def select_relevants(self, strips, query, top_n=5):
        """选择最相关的文档片段"""
        strips_data = []
        for i, p in enumerate(strips):
            if len(p.split()) < 4:
                scores = -1.0
            else:
                input_content = query + " [SEP] " + p
                inputs = self.tokenizer(
                    input_content,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                )
                try:
                    with torch.no_grad():
                        outputs = self.model(
                            inputs["input_ids"].to(self.device),
                            attention_mask=inputs["attention_mask"].to(self.device),
                        )
                    scores = float(outputs["logits"].cpu())
                except:
                    scores = -1.0
            strips_data.append((scores, p, i))

        # 按分数排序
        sorted_results = sorted(strips_data, key=lambda x: x[0], reverse=True)
        selected_strips = [s[1] for s in sorted_results[:top_n]]
        return "; ".join(selected_strips)

    def refine_internal_knowledge(self, query, docs):
        """内部知识精炼"""
        all_strips = []
        for doc in docs:
            all_strips.extend(self.extract_strips_from_psg(doc, mode="selection"))
        refined_knowledge = self.select_relevants(all_strips, query, top_n=5)
        return refined_knowledge

    def evaluate_retrieval(self, query, docs):
        """评估检索到的文档相关性"""
        scores = []

        # 在开始处理一批 docs 前清理缓存
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        for doc in docs:
            input_text = query + " [SEP] " + doc
            inputs = self.tokenizer(
                input_text, return_tensors="pt", padding="max_length", max_length=512
            )
            with torch.no_grad():
                outputs = self.model(
                    inputs["input_ids"].to(self.device),
                    attention_mask=inputs["attention_mask"].to(self.device),
                )
            score = float(outputs["logits"].cpu())
            scores.append(score)
        print(f"Query: {query}, scores: {scores}")
        # 可选：在每次迭代后清理
        del inputs, outputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return scores


class RemoteCRAGEvaluator:
    def __init__(self, config: dict):
        """
        初始化远程评估器。

        Args:
            config (dict): 配置字典，必须包含键 "remote_evaluator_url"。
        """
        self.remote_url = config["remote_evaluator_url"]

    @retry(max=5, sleep=1)
    def _evaluate(self, query: str, docs: List[str]) -> List[float]:
        """
        向远程服务发送评估请求。

        Args:
            query (str): 用户查询。
            docs (List[str]): 待评估的文档列表。

        Returns:
            List[float]: 每个文档的评分。
        """
        url = f"{self.remote_url}/evaluate"
        response = requests.post(url, json={"query": query, "docs": docs})
        response.raise_for_status()  # 如果响应码不是 200，抛出异常
        return response.json()["scores"]

    def evaluate_retrieval(self, query: str, docs: List[str]) -> List[float]:
        """
        评估给定文档与查询的相关性。

        Args:
            query (str): 用户查询。
            docs (List[str]): 待评估的文档列表。

        Returns:
            List[float]: 每个文档的评分。
        """
        scores = self._evaluate(query, docs)
        # print(f"Query: {query}, scores: {scores}")
        return scores


if __name__ == "__main__":
    # 测试代码
    remote_crag = RemoteCRAGEvaluator(
        {"remote_evaluator_url": "http://localhost:10098"}
    )
    query = "What is the capital of France?"
    docs = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome.",
    ]
    scores = remote_crag.evaluate_retrieval(query, docs)
    print(scores)
