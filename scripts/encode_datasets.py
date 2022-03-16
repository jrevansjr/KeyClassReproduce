import argparse
from os.path import join
from utils import fetch_data

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/zfsauton/project/public/chufang/classes/',
                    help='dataset directory')
parser.add_argument('--dataset', default='imdb',
                    help='dataset')
parser.add_argument('--model_name', default='all-mpnet-base-v2'
                    help='Sentence Encoder model to use')
args = parser.parse_args()

model = Encoder(model_name=args.model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

for split in ['train', 'test']:
    text = fetch_data(dataset=args.dataset, split=split, path=args.data_path)
    embeddings = model.get_embeddings(text=text, batch_size=128)
    with open(join(data_path, dataset, f'{split}_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)