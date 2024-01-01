import sys
import os
from tqdm import tqdm
import numpy as np
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import pytorch_lightning as pl

import streamlit as st

BEST_PATH = "checkpoints_pl/Swallow_jojo_bestmodel.ckpt"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(100)

DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")

def build_prompt(user_query, responses="", inputs="", sep="\n\n### "):
    # sys_msg = "以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。"
    sys_msg = "以下に、あるタスクを説明する指示があります。リクエストを適切に完了するための回答を記述してください。"
    p = sys_msg
    roles = ["指示", "応答"]
    msgs = [":\n" + user_query, ":"]
    if responses:
        msgs[-1] = ":\n" + responses
    if inputs:
        roles.insert(1, "入力")
        msgs.insert(1, ":\n" + inputs)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p

class LLMTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            "tokyotech-llm/Swallow-7b-instruct-hf", 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Swallow-7b-instruct-hf")
        self.tokenizer.add_special_tokens({'pad_token': '<|padding|>'})

    def generate(self, ids, mask):

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        output = self.tokenizer.decode(output_ids[0][0], skip_special_tokens=True)

        return output

@st.cache_resource
def load_model():
    
    LLM_Module = LLMTrainer().to(DEVICE)

    checkpoint_path = BEST_PATH
    tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Swallow-7b-instruct-hf")
    tokenizer.add_special_tokens({'pad_token': '<|padding|>'})

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    LLM_Module.load_state_dict(checkpoint['state_dict'])
    return LLM_Module, tokenizer

def main():

    LLM_Module, tokenizer = load_model()
    st.title("ジョジョ言い換え")
    init_text = "やれやれです"
    text = st.text_area("入力したテキストをちょっぴりジョジョっぽく言い換えます．文章を入力してください．", value=init_text, height=150)

    if st.button("言い換え"):
        user_inputs = {
            "user_query": "文「"+text+"」をジョジョっぽく言い換えてください。",
            "responses": "",
            "inputs": "",
        }
        text = build_prompt(**user_inputs)
        token_ids = tokenizer.encode_plus(text, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            output = LLM_Module.generate(token_ids["input_ids"].to(DEVICE), token_ids["attention_mask"].to(DEVICE))
        st.write("回答: ")
        st.write(output.replace(text, ""))

if __name__ == "__main__":
    main()