from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,TrainingArguments
from peft import LoraConfig
import torch
from datasets import Dataset
from trl import SFTTrainer
import os
import gdown
import re

class AYA:
    def __init__(self, 
                 device="cuda",
                 use_grad_ckpt=True,
                 train_bs=8,
                 train_max_seq_length=256,
                 grad_acc_steps=4,
                 finetuned_ckpt_path=None
                 ):
        self._device = device
        self._use_grad_ckpt = use_grad_ckpt
        self._train_bs = train_bs
        self._train_max_seq_length = train_max_seq_length
        self._grad_acc_steps = grad_acc_steps
        
        self._fined_ckpt_path = os.path.join(os.getcwd(), "results") if finetuned_ckpt_path is None else finetuned_ckpt_path


        if self._device == "cuda" and not(torch.cuda.is_available()):
            raise RuntimeError("There is no CUDA available but the device is set to 'cuda'")


        print("Model is initialising ...")
        self._model, self._tokenizer = self._initialize_model_n_tokenizer() 
        print("Model is successfully initialised ...")


    def _initialize_model_n_tokenizer(self):
        root_to_ckpt = os.path.join(os.getcwd(), "weights")
        if os.path.isdir(root_to_ckpt):
            try:    
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    root_to_ckpt,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                )
                model = model.to(self._device)
                tokenizer = AutoTokenizer.from_pretrained(root_to_ckpt)
                return model, tokenizer
            except Exception as e:
                print("There is sth wrong with the provided checkpoint.")
                print("The complete log of error is: ")
                print(e)

    
    def _get_dataset(self):
        default_data_path = os.path.join(os.getcwd(), "EN_GE_PE_DATASET.txt")

        if not(os.path.isfile(default_data_path)):
            print("Downloading data to root dir ...")
            data_url="https://drive.google.com/uc?id=1T7B_OhwFDU9Td846as_eC47ZTYsSOsCs"
            gdown.download(data_url, default_data_path, quiet=False)

        inputs = []
        ge_targets = []
        pe_targets = []
        with open(default_data_path, "r", encoding='utf-8') as f:
            content = [l.strip() for l in f.readlines()]


        for c in content:
            en = c.split("|||")[0]
            ge = c.split("|||")[1]
            pe = c.split("|||")[2]

            inputs.append(en)
            ge_targets.append(ge)
            pe_targets.append(pe)

        return Dataset.from_dict({
                                "inputs": inputs,
                                "ge_targets": ge_targets,
                                "pe_targets": pe_targets,
                                })

    @staticmethod
    def _format_func(record):
        output_texts = []
        for i in range(len(record['inputs'])):
            text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Translate {record['inputs'][i]} to German and Persian<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>German: {record['ge_targets'][i]} Persian: {record['pe_targets'][i]}"
            output_texts.append(text)
        return output_texts

    def finetune(self):
        train_dataset = self._get_dataset()

        training_arguments = TrainingArguments(
            output_dir=self._fined_ckpt_path,
            num_train_epochs=1, 
            per_device_train_batch_size=self._train_bs,
            gradient_accumulation_steps=self._grad_acc_steps,
            gradient_checkpointing=self._use_grad_ckpt,
            optim="paged_adamw_32bit",
            save_steps=50,
            logging_steps=10,
            learning_rate=1e-3,
            weight_decay=0.001,
            fp16=False,
            bf16=True,
            warmup_ratio=0.05,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none"
        )

        peft_config = LoraConfig(
            lora_alpha=32,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj"]
        )

        trainer = SFTTrainer(
            model=self._model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            max_seq_length=self._train_max_seq_length,
            tokenizer=self._tokenizer,
            args=training_arguments,
            formatting_func=self._format_func
        )

        # Train the model
        print("Starting Training")
        trainer.train()

        # Save the model to disk
        print("Saving Model")
        trainer.model.save_pretrained(save_directory='trans_ge_to_ge_pe')

    @staticmethod
    def _postprocess(response):
        response = re.sub(r"<.*?>", "", response)
        
        german_match = re.search(r"German:\s*(.*?)(?=\s*Persian:|$)", response)
        german_translation = german_match.group(1).strip() if german_match else ""
        
        persian_match = re.search(r"Persian:\s*(.*?)(?=\s*German:|$)", response)
        persian_translation = persian_match.group(1).strip() if persian_match else ""
        
        german_translation = german_translation.replace('"', '')
        persian_translation = persian_translation.replace('«', '').replace('»', '').replace('"', '')

        return german_translation, persian_translation


    def infer(self, content):
        try:
            self._model.load_adapter(r"C:\Users\ASUS\Desktop\github_projects\Aya-Expanse-Translation\Aya-Expanse-Finetuning\trans_ge_to_ge_pe")
        except:
            pass

        prompts = [{"role": "user", "content": f'Translate "{content}" to German and Persian'}]

        input_ids = self._tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self._device)
        gen_tokens = self._model.generate(
            input_ids, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.3,
            )

        gen_text = self._tokenizer.decode(gen_tokens[0])
        german, persian = self._postprocess(gen_text)

        return german, persian

if __name__ == "__main__":
    aya = AYA()
    # aya.finetune()
    german, persian = aya.infer("Hi I want to become a movie star")

    print(german, persian)
