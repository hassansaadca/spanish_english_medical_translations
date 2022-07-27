# w266_final_project
Spanish &lt;-> English Medical Translations

## Model
We address the process of building a model to translate Spanish to English in the biomedical domain. We start with an existing model sourced from HuggingFace and we train on a text corpus from the Institute of Formal and Applied Linguistics (UFAL). We report the evaluation metrics both in terms of the corpus BLEU scores and a comparison to translations completed by a professional Spanish-English medical interpreter.

![Model Architecture](w266_final_project/images/model_architecture.png)

The MarianMT transformer consists of an encoder-decoder stack with 6 layers in each component. Similar to T5, it uses a self attention layer on the encoder side and a masked self-attention layer on the decoder side to act as an autoregressive language model when converting from embeddings to text. Where T5 is trained on various different tasks including question-answering, translation, etc., our fine-tuning of course only focused on a translation-based training strategy. 

![Model Preprocessing](w266_final_project/images/model_preprocessing.png)
