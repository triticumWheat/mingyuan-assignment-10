from flask import Flask, render_template, request
import torch
from PIL import Image as PILImage
from open_clip import create_model_and_transforms, get_tokenizer
import torch.nn.functional as F
import pandas as pd

app = Flask(__name__)

df = pd.read_pickle('image_embeddings.pickle')
model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = get_tokenizer('ViT-B-32')
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text_query = request.form.get('text_query')
        image_query = request.files.get('image_query')
        weight = float(request.form.get('weight', 0.8))

        try:
            if text_query:
                text_query = tokenizer([text_query])
                text_embedding = F.normalize(model.encode_text(text_query))
            else:
                text_embedding = None

            if image_query:
                image = PILImage.open(image_query)
                image_tensor = preprocess(image).unsqueeze(0)
                image_embedding = F.normalize(model.encode_image(image_tensor))
            else:
                image_embedding = None

            if text_embedding is not None and image_embedding is not None:
                query_embedding = F.normalize(weight * text_embedding + (1 - weight) * image_embedding)
            elif text_embedding is not None:
                query_embedding = text_embedding
            elif image_embedding is not None:
                query_embedding = image_embedding
            else:
                return render_template('index.html', error="Please provide text or image query.", results=None)

            embeddings = torch.tensor(df['embedding'].to_list())
            embeddings = F.normalize(embeddings, dim=1)
            cosine_similarities = torch.mm(query_embedding, embeddings.T).squeeze(0)

            top5_indices = torch.topk(cosine_similarities, 5).indices.tolist()
            top5_scores = torch.topk(cosine_similarities, 5).values.tolist()

            results = []
            for idx, score in zip(top5_indices, top5_scores):
                impath = 'coco_images_resized/' + df.iloc[idx]['file_name']
                results.append({'path': impath, 'score': round(score, 4)})

            return render_template('index.html', error=None, results=results)

        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}", results=None)

    return render_template('index.html', error=None, results=None)

if __name__ == '__main__':
    app.run(debug=True)
