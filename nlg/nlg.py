from transformers import pipeline
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch, os
import pandas as pd


class WordInserter:
    """
    This class is designed to insert relevent words in between the words in a sentence.
    Use this ONLY with the sentences that you have already tagged in NLU's dataset.
    """

    def __init__(self) -> None:
        """
        We will be using the BERT model for this task.
        """
        self.model_name = "bert-base-uncased"
        self.bert_tokeniser = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = BertForMaskedLM.from_pretrained(self.model_name)

    def format_model_input(self, text: str, tokeniser, insert_mask_at_idx: int):
        """
        Prepare the input to be fed into the model.

        Args:
            text (str): The text to be masked.
            tokeniser (_type_): the tokenizer initialized in the constructor.
            insert_mask_at_idx (int): where in sentence to insert the mask.

        Returns:
            a tokenized tensor of the input text.
        """
        tokens = tokeniser.tokenize(f"[CLS] {text} [SEP]")
        tokens_with_mask = tokens[:insert_mask_at_idx] + ["[MASK]"] + tokens[insert_mask_at_idx:]
        return torch.tensor([tokeniser.convert_tokens_to_ids(tokens_with_mask)])

    def format_model_output(self, model_output, token_idxs, tokeniser, masked_idx):
        tokens = tokeniser.convert_ids_to_tokens(token_idxs.tolist()[0])
        tokens[masked_idx] = tokeniser.convert_ids_to_tokens([torch.argmax(model_output[0, masked_idx]).item()])[0]
        return " ".join(tokens[1:-1]).replace("##", "")

    def insert_mask_and_predict(self, sentence, model, tokeniser, masked_idx):
        tokens_with_mask_inserted = self.format_model_input(
            text=sentence,
            tokeniser=tokeniser,
            insert_mask_at_idx=masked_idx,
        )
        segment_ids = torch.tensor([[0] * len(tokens_with_mask_inserted)])
        with torch.no_grad():
            return self.format_model_output(
                model_output=model(tokens_with_mask_inserted, segment_ids),
                tokeniser=tokeniser,
                token_idxs=tokens_with_mask_inserted,
                masked_idx=masked_idx,
            )

    def insert_words(self, example):
        new_examples = [example]
        idx = 1
        try:
            while True:
                new_examples.append(
                    self.insert_mask_and_predict(
                        sentence=example,
                        model=self.bert_model,
                        tokeniser=self.bert_tokeniser,
                        masked_idx=idx,
                    )
                )
                idx += 1
        except:
            new_examples.pop()
            return new_examples


class QuestionAnswerer:
    """
    This class uses a pre-trained BERT model to answer questions.
    Since we still have problem with not enough data, use this to extract information from pre-defined scenarios to extract information.
    """

    def __init__(self) -> None:
        self.question_answerer_agent = pipeline("question-answering")

    def question_answering(self, question: str, context_path: str) -> str:
        context = open(context_path, "r").read()
        context = r"{}".format(context)
        result = self.question_answerer_agent(question=question, context=context)
        return result["answer"]


if __name__ == "__main__":
    q_a = QuestionAnswerer()
    print(q_a.question_answering("did he have permission?", os.path.join(os.getcwd(), "nlg", "scenarios", "18a.txt")))
