from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class T5AutoEncoder:
    def __init__(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.model_name = "flan-t5-base"
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

        self.rules = [
            "Ball one will be bounced off and Ball two will move in the previous Ball one's direction if Ball one's mass is less than Ball two's.",
            "Ball one and ball two will move in the same direction after the crash if Ball one's mass is larger than Ball two.",
            "Ball one will to stay static if its mass is equal to Ball two, Ball two will start moving in the previous Ball one’s moving direction after the crash.",
            "If a ball hits the platform it will be bounced off.",
            "A ball will fall due to gravity in the case of free fall.",
        ]

        self.max_length = 40
        self.rule_embeddings = self.encode(self.rules)
        self.decoded_rules = self.decode(self.rule_embeddings)
        self.rule_embeddings = self.rule_embeddings.reshape(self.rule_embeddings.shape[0], -1).detach().cpu().numpy()
        self.max_length = 512
        self.rule_max_length = 40

    def encode(self, texts):
        inputs = self.tokenizer(texts, max_length=self.max_length, return_tensors="pt", padding="max_length", truncation=True).to(self.device)
        encoder_hidden_states = self.model.encoder(inputs.input_ids)
        return encoder_hidden_states.last_hidden_state

    def decode(self, embeddings):
        decoder_texts = ["Repeat the provided text." for _ in range(embeddings.shape[0])]
        decoder_inputs = self.tokenizer(decoder_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        outputs = self.model.generate(
            input_ids=decoder_inputs.input_ids,
            encoder_outputs=(embeddings,),
            max_length=self.max_length
        )

        outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
        return outputs


if __name__ == "__main__":
    window_size = 160
    autoencoder = T5AutoEncoder()
    for a, b in zip(autoencoder.rules, autoencoder.decoded_rules):
        print("#" * window_size)
        print(a)
        print("-" * window_size)
        print(b)
    print("#" * window_size)
