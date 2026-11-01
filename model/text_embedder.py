import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

class TextEmbedder(nn.Module):
    def __init__(self, cache_dir='./cache', max_seq_len=512, model_name='google-bert/bert-large-uncased'):
        super(TextEmbedder, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_len = max_seq_len
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = BertModel.from_pretrained(self.model_name, cache_dir=cache_dir, 
                                               max_position_embeddings=max_seq_len).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
    def get_embedding(self, texts: list[str]):
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors='pt', max_length=self.max_seq_len, 
                                    padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            embedding = outputs[0]
            key_padding_mask = (inputs['attention_mask'] == 0)
            
            return embedding, key_padding_mask

if __name__ == '__main__':
    text_embedder = TextEmbedder()
    texts = ['Hello, world!', 'This is a longer test that should be truncated']
    
    embedding, key_padding_mask = text_embedder.get_embedding(texts)
    print(embedding.shape)
    print(embedding)
    print(key_padding_mask.shape)
    print(key_padding_mask)