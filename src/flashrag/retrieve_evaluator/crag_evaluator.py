from typing import List
import requests
import torch
from transformers import T5ForSequenceClassification
from transformers import T5Tokenizer

from flashrag.retriever.retriever import retry


class CRAGEvaluator:
    def __init__(self, evaluator_path, device):
        self.tokenizer = T5Tokenizer.from_pretrained(evaluator_path)
        self.model = T5ForSequenceClassification.from_pretrained(
            evaluator_path, num_labels=1
        )
        self.device = (
            torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

    def evaluate_retrieval(self, query, docs):
        """评估检索到的文档相关性"""
        scores = []
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
        return scores


class RemoteCRAGEvaluator:
    def __init__(self, config: dict):
        """
        初始化远程评估器。

        Args:
            config (dict): 配置字典，必须包含键 "remote_evaluator_url"。
        """
        self.remote_url = config["remote_evaluator_url"]

    @retry(max_retries=10, delay=1)
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
        print(f"Query: {query}, scores: {scores}")
        return scores
