# Why VLMs are built wrong and why agentic VLMs are the solution

Give a vision language model an image and a question. It can describe the scene, read text, and explain a chart. But ask it to count a few crossing lines or follow a route, and it can confidently miss something right in front of it.

My suspicion is that part of the problem happens before the language model starts reasoning. We give it a question about pixels, but it reasons over an encoded representation of those pixels. The component making that representation usually does not know what we are looking for.

That is what made me interested in models that crop an image and look again. The model can use the question to decide where to look next. I wanted to try that with a small experiment, using a crop tool and questions with answers we could check.

Cropping is one possible tool. The broader idea is to let an agent act on the image and inspect the result. That loop can mitigate the problem while the underlying vision architecture still needs fixing.

## What a VLM actually receives

A typical VLM combines a vision encoder with a language model. The encoder divides an image into patches and converts them into numerical vectors. These vectors carry visual features such as shapes, colours, and relationships between nearby regions.

The question takes a different path. A tokenizer splits the text into tokens. An embedding lookup maps each token to a vector. These are often called word embeddings, although a token can be only part of a word.

Both streams enter the language model as embeddings. The language model uses the text and visual features together to generate its response.

![Image patches and question tokens follow separate paths, then enter the same language model](assets/vlm-architecture.png)

Our diagram simplifies the [Gemma 4 architecture](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4). Its vision transformer encodes image patches. Pooling merges nearby patches, then a projection maps the visual features into the language model's embedding space.

Look at where the two paths meet. The vision encoder has already processed the image before the language model combines its output with the question. Putting the question earlier in the chat does not make that encoder question-aware.

## What gets squeezed out?

The image representation has a limited capacity. Resizing can remove detail before encoding. Pooling can merge small features. The encoder also learns which visual features are useful during training. The language model gets the resulting vectors, rather than direct access to the original pixels.

This could affect more than tiny text. Imagine an encoder trained mostly on pictures with blue skies. It might represent the presence of a sky while preserving less information about its exact colour. Now give it a picture with a green sky. If that unusual colour is missing from the representation, the language model may have no reliable way to recover it.

That is a thought experiment, not a failure we measured. It illustrates the difference between having the pixels available and having the relevant information available to the language model.

A longer conversation can change how the language model interprets the embeddings. It cannot add a feature that those embeddings do not contain. Sending the identical image through a deterministic, question-independent encoder again would produce the same representation. More reasoning does not automatically mean more visual evidence.

## Let the model choose another view

A crop tool gives the model a way to request new visual evidence. It chooses a rectangle using the question and the image it has already seen. Code cuts that region from the original pixels and returns the crop as a tool result. The same VLM then processes that new image inside the same conversation.

![Three steps show the model requesting a crop, the tool cutting pixels, and the next model call receiving the crop in the same chat](assets/crop-loop.png)

The encoder still compresses the image. But now it processes a region selected for the question. A letter or line can occupy more patches, and irrelevant surroundings can take up less of the representation.

An agent could also adjust contrast, isolate colours, or write its own image processing code and tools. Each action could produce new evidence for the next model call. We used a fixed crop tool here to keep the experiment small and the comparison clear.

Of course zooming also helps a person read small text. The useful question here is whether the model can choose a useful region itself and answer more accurately after re-encoding it. A crop can also miss the target or remove context needed to answer. We need to measure both outcomes.

## Give it something we can check

We used three tasks from [VLMs Are Blind](https://vlmsareblind.github.io). Each has a short answer that we can compare with ground truth.

In the first task, the model counts intersections between red and blue lines. In the second, it identifies the letter inside a red ring. In the third, it follows subway lines and counts routes connecting the stations named in the question.

![Benchmark examples of crossing lines, an enlarged circled letter, and a subway map](assets/task-gallery.png)

We selected 20 images per task, giving 60 images in total. GPT-5.4 and GPT-4o each answered every image under three conditions.

- See the full image once and answer
- See the full image and optionally request it again
- Make at least one crop before answering, with further crops allowed

The tool conditions allowed five image views, including the original. Every crop used coordinates chosen by the model. The tool always cut from the original image. It did not choose a region, read text, or tell the model whether an answer was correct.

The models received image pixels and questions. Ground truth and dataset metadata stayed outside their context. We required the first crop because neither model chose to crop in our initial pilot. That means this tests their ability to use a crop tool when instructed, rather than their ability to decide whether to use one.

## What happened

With cropping, GPT-5.4 reached **90% accuracy**. GPT-4o reached **60%**. Those are different models, so the useful comparison is how each changed from its own single-view result.

GPT-5.4 improved from **44/60 to 54/60**, or **73% to 90%**. GPT-4o improved from **34/60 to 36/60**, or **57% to 60%**.

![Results for both models in all three conditions](assets/results.png)

The biggest gain came from GPT-5.4 on subway maps. It went from **6/20 to 14/20**. It already answered all 20 line intersection questions correctly. Circled letters improved from **18/20 to 20/20**.

Here is one of those letter examples. In its single-view run, GPT-5.4 answered **U**. In the crop run, it selected the region around the text and answered **i**, which is correct. The rectangle on the left shows the coordinates the model requested.

![The full image and actual model-selected crop, with their separate answers](assets/poster-example.png)

GPT-4o was less reliable. Cropping fixed nine answers that were wrong in its single-view runs, but seven previously correct answers became wrong. A new view helps only if the model selects and understands it.

This run shows why selecting the crop is part of the problem. GPT-4o's first two regions missed the red ring. Its third included the circled **y**, and it answered correctly. Its separate single-view run had answered **o**.

![Three crops selected by GPT-4o, with the target reached on the third attempt](assets/three-crops.png)

## What this experiment leaves open

The repeat condition allowed another full-image view, but every agent answered after its first view. Its scores were **47/60** for GPT-5.4 and **33/60** for GPT-4o. It therefore did not test whether repeated full-image views help. A follow-up should require the same number of views in both tool conditions.

We ran each image once per condition, giving 360 episodes at a nominal API cost of about **$2.06**. This is a small experiment. It does not identify which features the encoder lost or prove the green-sky hypothesis. It shows that model-selected crops helped GPT-5.4 substantially on this sample, while GPT-4o gained little overall.

I still think VLMs are built wrong when their visual representation is created without knowing what the question needs. An agentic loop is one way to work around that design, and cropping is one tool inside that loop. It gives the model some control over what gets encoded next, but the same bottleneck remains on every pass.

I see this as a mitigation. A more direct solution would let the question guide visual encoding itself. This experiment does not tell us which architecture would do that best. It gives us a small, practical example of why letting the model gather more relevant visual evidence is worth pursuing.

*Images come from the [XAI VLMs Are Blind dataset](https://huggingface.co/datasets/XAI/vlmsareblind), by Rahmanzadehgervi, Bolton, Taesiri, and Nguyen. See the [evaluation notes](../evaluation-results.md) for the full results.*
