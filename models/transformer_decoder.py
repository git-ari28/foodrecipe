from transformers import T5ForConditionalGeneration

class TransformerDecoder:
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')

    def generate(self, inputs_embeds, tokenizer, max_len=128):
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_len,
            num_beams=4,
            early_stopping=True
        )
        return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=(encoder_hidden_states,)
        )
