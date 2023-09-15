from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
import models.shared as shared
from models.loader.args import parser
from models.loader import LoaderCheckPoint
import os
import torch

# print(torch.cuda.is_a)


# local_doc_qa = LocalDocQA()
# print(local_doc_qa)


# 10.735
# 5346

from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig, LoraConfig

tokenizer = AutoTokenizer.from_pretrained("/code/chatglm2-6b", trust_remote_code=True)
peft_config = LoraConfig(r=8,
                        lora_alpha=32,
                        target_modules=["query_key_value"],  # lora的目标位置，具体有哪些可选项可打印出源码中的key_list 注意不同的模型中的定义名称不同
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        )

model = AutoModel.from_pretrained("/code/chatglm2-6b", trust_remote_code=True, device_map="auto")
model = PeftModel(model, "./out")
model = model.eval()

# model = AutoModel.from_pretrained("/code/chatglm2-6b", load_in_8bit=True, trust_remote_code=True, device_map="auto")