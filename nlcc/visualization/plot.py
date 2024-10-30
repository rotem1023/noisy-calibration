import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import os

def _get_cur_file_path():
    return f'{os.path.dirname(os.path.abspath(__file__))}'

def _get_plots_dir():
    return f'{_get_cur_file_path()}/plots'

def _get_plots_dir(dataset):
    plots_dir  = _get_plots_dir()
    os.makedirs(plots_dir, exist_ok=True)
    dataset_dir = f'{plots_dir}/{dataset}'
    os.makedirs(plots_dir, exist_ok=True)  
    return plots_dir

def create_tsne_plot(dataset, x, y):
    tsne = TSNE(n_components=2, random_state=0)
    x_tsne = tsne.fit(x)
    
    df = pd.DataFrame(x_tsne, columns = ['TSNE1', 'TSNE2'])
    df['label'] = y
    
    plt.figure(figsize=(10,6))
    scater =plt.scatter(df['TSNE1'], df['TSNE2'], c = df['label'], cmap='viridis')
    plt.colorbar(scater, label='label')
    
    plots_dir  = _get_plots_dir(dataset)
    
    plt.savefig(f"{plots_dir}/tsne.png", format='png', dpi = 300)  
    plt.show()
    
    plt.close()
    

    