
Usage
=====

Characteristics of Sentence Transformer (a.k.a bi-encoder) models:

1. Calculates a **fixed-size vector representation (embedding)** given **texts, images, audio, video, or combinations thereof** (depending on the model).
2. Embedding calculation is often **efficient**, embedding similarity calculation is **very fast**.
3. Applicable for a **wide range of tasks**, such as semantic textual similarity, semantic search, clustering, classification, paraphrase mining, and more.
4. Often used as a **first step in a two-step retrieval process**, where a Cross-Encoder (a.k.a. reranker) model is used to re-rank the top-k results from the bi-encoder.

Once you have `installed <../../installation.html>`_ Sentence Transformers, you can easily use Sentence Transformer models:

.. sidebar:: Documentation

   1. :class:`SentenceTransformer <sentence_transformers.sentence_transformer.model.SentenceTransformer>`
   2. :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`
   3. :meth:`SentenceTransformer.encode_query <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query>`
   4. :meth:`SentenceTransformer.encode_document <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document>`
   5. :meth:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>`
   6. :doc:`Speeding up Inference <efficiency>`

::

   from sentence_transformers import SentenceTransformer

   # 1. Load a pretrained Sentence Transformer model
   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

   # The sentences to encode
   sentences = [
       "The weather is lovely today.",
       "It's so sunny outside!",
       "He drove to the stadium.",
   ]

   # 2. Calculate embeddings by calling model.encode()
   embeddings = model.encode(sentences)
   print(embeddings.shape)
   # [3, 384]

   # 3. Calculate the embedding similarities
   similarities = model.similarity(embeddings, embeddings)
   print(similarities)
   # tensor([[1.0000, 0.6660, 0.1046],
   #         [0.6660, 1.0000, 0.1411],
   #         [0.1046, 0.1411, 1.0000]])

Detecting truncated text inputs
-------------------------------

On a checkout that includes truncation-count support, ``encode(..., output_value=None)`` returns
one feature dictionary per input. For the text tokenizer path, each dictionary includes
``num_truncated_tokens``: zero means tokenization kept the full input, while a positive integer
is the number of tokens removed from that input. The count includes prompt and special tokens
that are part of the tokenized sequence. It counts tokens, not documents.

.. code-block:: python

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   texts = ["A short input.", "A longer input " * 200]
   features = model.encode(
       texts,
       output_value=None,
       batch_size=2,
       processing_kwargs={"text": {"max_length": 64}},
   )

   for text, feature in zip(texts, features):
       embedding = feature["sentence_embedding"]
       count = feature.get("num_truncated_tokens")
       if count is None:
           print("No truncation count was reported for this input path.")
       elif count.item() == 0:
           print("The full tokenized input was kept.")
       else:
           print(f"{count.item()} tokens were removed.")
       print(text[:30], tuple(embedding.shape))

   # The default call still returns only sentence embeddings.
   embeddings = model.encode(texts)
   print(embeddings.shape)

``output_value=None`` also returns other features such as token embeddings, so it can use more
memory than the default ``encode`` call. The count is currently available on the text tokenizer
path; multimodal processors and other processor types may not report it. A missing key therefore
means that no count was reported, not that zero tokens were removed. This example describes the
proposed feature on a checkout containing the change; it does not imply availability in a released
PyPI version.

Some Sentence Transformer models support inputs beyond text, such as images, audio, or video. Use :attr:`model.modalities <sentence_transformers.sentence_transformer.model.SentenceTransformer.modalities>` and :meth:`model.supports() <sentence_transformers.sentence_transformer.model.SentenceTransformer.supports>` to check modality support. See :doc:`../../input_formats` for accepted representations, metadata, and batching.

The following example loads a multimodal model and computes similarities between text and image embeddings:

.. sidebar:: Modality Support

   .. code-block:: python

      from sentence_transformers import SentenceTransformer
   
      model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")
   
      # List all supported modalities
      print(model.modalities)
      # ['text', 'image', 'video', 'message']
   
      # Check for a specific modality
      print(model.supports("image"))
      # True
      print(model.supports("audio"))
      # False

.. code-block:: python

   from sentence_transformers import SentenceTransformer

   # 1. Load a model that supports both text and images
   model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")

   # 2. Encode images from URLs
   img_embeddings = model.encode([
       "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
       "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
   ])

   # 3. Encode text queries (one matching + one hard negative per image)
   text_embeddings = model.encode([
       "A green car parked in front of a yellow building",
       "A red car driving on a highway",
       "A bee on a pink flower",
       "A wasp on a wooden table",
   ])

   # 4. Compute cross-modal similarities
   similarities = model.similarity(text_embeddings, img_embeddings)
   print(similarities)
   # tensor([[0.5115, 0.1078],
   #         [0.1999, 0.1108],
   #         [0.1255, 0.6749],
   #         [0.1283, 0.2704]])

For retrieval tasks, :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query` and :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document` are the recommended methods. Many embedding models use different prompts or instructions for queries vs. documents, and these methods handle that automatically:

- :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query` uses the model's ``"query"`` prompt (if available) and sets ``task="query"``.
- :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document` uses the first available prompt from ``"document"``, ``"passage"``, or ``"corpus"``, and sets ``task="document"``.

These methods accept all the same input types as :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode` (text, images, URLs, multimodal dicts, etc.) and pass through all the same parameters. For models without specialized query/document prompts, they behave identically to :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode`.

.. code-block:: python

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")

   # Encode text queries with the query prompt
   query_embeddings = model.encode_query([
       "Find me a photo of a vehicle parked near a building",
       "Show me an image of a pollinating insect",
   ])

   # Encode document screenshots with the document prompt
   doc_embeddings = model.encode_document([
       "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
       "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
   ])

   # Compute similarities
   similarities = model.similarity(query_embeddings, doc_embeddings)
   print(similarities)
   # tensor([[0.3907, 0.1490],
   #         [0.1235, 0.4872]])

.. toctree::
   :maxdepth: 1
   :caption: Tasks and Advanced Usage

   ../../input_formats
   ../../../examples/sentence_transformer/applications/computing-embeddings/README
   semantic_textual_similarity
   ../../../examples/sentence_transformer/applications/semantic-search/README
   ../../../examples/sentence_transformer/applications/retrieve_rerank/README
   ../../../examples/sentence_transformer/applications/clustering/README
   ../../../examples/sentence_transformer/applications/paraphrase-mining/README
   ../../../examples/sentence_transformer/applications/parallel-sentence-mining/README
   ../../../examples/sentence_transformer/applications/image-search/README
   ../../../examples/sentence_transformer/applications/embedding-quantization/README
   custom_models
   mteb_evaluation
