from datasets import load_dataset, Dataset

def make_culturaX():
    import huggingface_hub
    huggingface_hub.login(token="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")
    dataset = load_dataset("uonlp/CulturalX", "vi", split="train")
    return dataset

def make_dummy_culturaX():
    data = open("database/dummy_culturalX.txt", "r").readlines()
    data_dict = {"text": [], "timestamp": [], "url": [], "source": []}
    for i in range(int(len(data)/4)):
        data_dict["text"].append("".join(data[i*4].split("\t\n")))
        data_dict["timestamp"].append("".join(data[i*4+1].split("\t\n")))
        data_dict["url"].append("".join(data[i*4+2].split("\t\n")))
        data_dict["source"].append("".join(data[i*4+3].split("\t\n")))
    dataset = Dataset.from_dict(data_dict)
    return dataset

# dataset = make_dummy_culturaX()
# print(dataset[0]) 
class CulturaXDataset:
    def __init__(self) -> None:
        self.hgf_dataset = make_dummy_culturaX()
    
    def __getitem__(self, index):
        content = self.hgf_dataset[index]["text"]
        ratio_human = 0.1
        words = content.split(" ")
        # idx_split = len(" ".join(words[:int(ratio_human*len(words))]))
        idx_split = len(" ".join(words[:10]))
        human_part = content[:idx_split]
        conversation = [
            {
                "from": "human",
                "value": "Hoàn thành đoạn văn sau: " + human_part
            },
            {
                "from": "gpt",
                "value": content
            }
        ]
        return conversation
