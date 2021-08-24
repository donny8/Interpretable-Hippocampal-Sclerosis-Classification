import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from reduction_func import *

modelHS = DR_expr_sett()
intermediates, color_multi, listName, idxMulti= last_conv_features(modelHS)

if(SELECT=='tSNE'):
    tsne = TSNE(n_components=2, random_state=KERNEL_SEED, perplexity = PERP, learning_rate=EPSI,n_iter=STEP)
    intermediates_tsne = tsne.fit_transform(intermediates)
elif(SELECT=='UMAP'):
    reducer = umap.UMAP(n_components=2, random_state=KERNEL_SEED, n_neighbors=PERP, min_dist=mini)
    intermediates_tsne = reducer.fit_transform(intermediates)
figure_2D(color_multi, idxMulti, intermediates_tsne)

excel_save(listName, intermediates_tsne, color_multi)

if(SELECT=='tSNE'):
    tsne = TSNE(n_components=3, random_state=KERNEL_SEED, perplexity = PERP, learning_rate=EPSI,n_iter=STEP*10)
    intermediates_tsne = tsne.fit_transform(intermediates)
elif(SELECT=='UMAP'):
    reducer = umap.UMAP(n_components=3, random_state=KERNEL_SEED, n_neighbors=PERP, min_dist=mini)
    intermediates_tsne = reducer.fit_transform(intermediates)
figure_3D(color_multi,intermediates_tsne)