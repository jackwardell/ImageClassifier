"""
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
    Basic usage: python predict.py /path/to/image checkpoint
    Options:
        Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
        Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
        Use GPU for inference: python predict.py input checkpoint --gpu
"""


import click
from train_pred_funcs import get_dataloaders, get_model, train_model, save_checkpoint, load_checkpoint, predict

@click.command()
@click.argument('filename')
@click.argument("checkpoint", default="checkpoint.pth")
@click.option("topk", "--topk", default=5)
@click.option("category_names", "--category_names", required=False)
@click.option("gpu", '--gpu', is_flag=True, default=True)
def do_predict(filename, checkpoint, topk, category_names, gpu):
    model = load_checkpoint(checkpoint_path=checkpoint)
    prediction = predict(filename, model, topk=topk)
    click.echo(prediction)
    
    
if __name__ == '__main__':
    do_predict()