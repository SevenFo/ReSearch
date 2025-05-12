from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import T5Tokenizer, T5ForSequenceClassification

from flashrag.retrieve_evaluator.crag_evaluator import CRAGEvaluator

# 初始化 FastAPI 应用
app = FastAPI()


# 定义请求体模型
class EvaluateRequest(BaseModel):
    query: str
    docs: List[str]


# # 加载模型和分词器
# class EvaluatorModel:
#     def __init__(self, model_path: str):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = T5Tokenizer.from_pretrained(model_path)
#         self.model = T5ForSequenceClassification.from_pretrained(
#             model_path, num_labels=1
#         )
#         self.model.to(self.device)
#         self.model.eval()  # 设置为评估模式

#     def evaluate(self, query: str, docs: List[str]) -> List[float]:
#         scores = []
#         for doc in docs:
#             input_text = query + " [SEP] " + doc
#             inputs = self.tokenizer(
#                 input_text,
#                 return_tensors="pt",
#                 padding="max_length",
#                 max_length=512,
#                 truncation=True,
#             )
#             with torch.no_grad():
#                 outputs = self.model(
#                     input_ids=inputs["input_ids"].to(self.device),
#                     attention_mask=inputs["attention_mask"].to(self.device),
#                 )
#                 score = float(outputs.logits.cpu().item())
#                 scores.append(score)
#         return scores


# 实例化模型
model = CRAGEvaluator("path/to/your/t5-evaluator-model")


# 定义接口
@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    try:
        scores = model.evaluate(request.query, request.docs)
        return {"scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 启动服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=1008)
