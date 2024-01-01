import torch
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM

import random
import copy
import datetime

from tqdm import tqdm
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchinfo import summary

from pytorch_lightning.loggers import TensorBoardLogger
import wandb
wandb.init(project="Swallow_jojo_tuning")

# 乱数シード設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(100)

DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")
SEQUENCE = 512
dt = datetime.datetime.today()  # ローカルな現在の日付と時刻を取得
date = f'{dt.year}.'+f'{dt.month}.'+f'{dt.day}.'+f'{dt.hour+9}.'+f'{dt.minute}'

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

class CreateTokenID(Dataset):
    def __init__(self, FT_path,):
        self.FT_path = FT_path
        self.inputs = []
        self.targets = []
        self.tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Swallow-7b-instruct-hf")
        self.tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"input_ids": source_ids, "attention_mask": source_mask,
                "labels": target_ids, "decoder_attention_mask": target_mask}

    def _build(self):
        src_list, tgt_list = [], []
        spath, tpath ="_src", "_tgt"
        with open(self.FT_path+spath, "r", encoding="utf-8") as f:
            src_list = f.read().split("\n")
            src_list.pop(-1)
        with open(self.FT_path+tpath, "r", encoding="utf-8") as f:
            tgt_list = f.read().split("\n")
            tgt_list.pop(-1)
        for i, line in tqdm(enumerate(src_list)):

            src_tab = src_list[i]
            tgt_tab = tgt_list[i]
            inputs = {
                "user_query": "文「"+src_tab+"」をジョジョっぽく言い換えてください。",
                "responses": "",
                "inputs": "",
            }
            src = build_prompt(**inputs)
            source_tokenized = self.tokenizer(src, add_special_tokens=False, padding="longest", max_length=SEQUENCE, return_tensors="pt", return_length=True,)
            source_len = source_tokenized["length"][0]
            
            inputs["responses"] = "「"+tgt_tab+"」"
            tgt = build_prompt(**inputs) + self.tokenizer.eos_token
            source_tokenized = self.tokenizer(tgt, add_special_tokens=False, padding="longest", max_length=SEQUENCE, return_tensors="pt")
        
            targets_tokenized = copy.deepcopy(source_tokenized)
            targets_tokenized["input_ids"][0][:source_len] = -100

            self.inputs.append(source_tokenized)
            self.targets.append(targets_tokenized)


class LLMDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, FT_path):
        super().__init__()
        self.batch_size = batch_size
        self.FT_path = FT_path
    
    def get_dataset(self, FT_path):
        """データセットを作成する"""
        return CreateTokenID(
            FT_path,
            )

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            self.train_dataset = self.get_dataset(FT_path=self.FT_path+"train")
            # self.val_dataset = self.get_dataset(FT_path=self.FT_path+"dev")
        if stage == 'test':
            self.test_dataset = self.get_dataset(FT_path=self.FT_path+"test")

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          drop_last=True, shuffle=True, 
                          num_workers=4,
                          )

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size, 
                          num_workers=4,
                          )
    
    def test_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=4,
                          )
    
class LLMTrainer(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            "tokyotech-llm/Swallow-7b-instruct-hf", 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Swallow-7b-instruct-hf")
        self.tokenizer.add_special_tokens({'pad_token': '<|padding|>'})

        self.lr = lr
        self.training_step_outputs = []


    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, _ = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True)
        wandb.log({"train_step_loss": loss})
        self.training_step_outputs.append(loss.item())
        return loss
    
    def on_train_epoch_end(self):
        epoch_train_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
        wandb.log({"train_loss": epoch_train_loss})
        print('-------- Current Epoch {} --------'.format(self.current_epoch + 1))
        print('train Loss: {:.4f}'.format(epoch_train_loss))
        self.training_step_outputs.clear()
        
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

def main():

    FT_path = "jojo_"
    batch_size = 1
    learning_rate = 1e-4
    num_epochs = 50

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints_pl",
        filename="bestloss_Swallow_jojo-{epoch:02d}-{train_loss:.2f}-"+date,
        monitor="train_loss",  
        save_last=True,
        save_weights_only=True,
        mode="min",
    )

    data_module = LLMDataModule(batch_size, FT_path)
    LLM_Module = LLMTrainer(learning_rate).to(DEVICE)

    trainer = pl.Trainer(accelerator="gpu", max_epochs=num_epochs, callbacks=[checkpoint_callback])
    trainer.fit(LLM_Module, data_module)

if __name__ == "__main__":
    main()
