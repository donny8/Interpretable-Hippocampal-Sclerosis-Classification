import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from reduction_func import *


start = time.time()

dir_path = './%s' %(select)
if(not(os.path.isdir(dir_path))): os.mkdir(dir_path)

kernel_path = './%s/K%dP%dE%dS%d' %(select,KERNEL_SEED,perp,epsi,step)
if(not(os.path.isdir(kernel_path))): os.mkdir(kernel_path)
        
layer_path = './%s/K%dP%dE%dS%d/layer%d' %(select,KERNEL_SEED,perp,epsi,step,layer_of_interest)
if(not(os.path.isdir(layer_path))): os.mkdir(layer_path)

#[FUL27hflip]HSCLRM_D60{F1K14}[best].pt


model_path = os.getcwd()+'/[%s%d%s]HS%s_D%d{F%dK%d}[best].pt'%(SETT,TRIAL,AUG,CONTROLTYPE,DATATYPE,FOLD_SEED,KERNEL_SEED)
net = HSCNN(ksize=4)
net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load(model_path))
net.to(device)
net.eval()

print(net)

storage = './%s_Storage.txt' %(select)
fwSt=open(storage,'a')
fwSt.write('\n\n[KERNEL %d Layer %d]\n'%(KERNEL_SEED, layer_of_interest))
ms, bs, pr, roc, sens, spec,f1_scores, B_check,idxMulti,idxBinary = balance(CONTROLTYPE,SETT,TRIAL,AUG,KERNEL_SEED,iters, imgCountNO_4, imgCountNO_7,imgCountNO_0,imgCountYES_1,imgCountYES_2,num1,num2,num3, fwSt)

df_color = 1
#traintest(test,idxMulti,24,epsi,step,layer_path,intermediate_tensor_function,pca_dim,df_color)
#traintest(not(test),idxMulti,50,epsi,step,layer_path,intermediate_tensor_function,pca_dim,df_color)

dirDataSet = "../../dataset/75_add_new4_from71"
MODE='TEST'
zerocut=86
# [Arbitraty]
# TEST
#CM0 = '#fea993' 
#CM1 = '#a2cffe'
#CM2 = '#9be5aa'

# TRAIN
#DM0 = '#e50000'
#DM1 = '#030aa7'
#DM2 = '#0a5f38'

#[Epilepsia]
# TEST
CM0 = '#ce8080'
CM1 = '#bacfec'
CM2 = '#abb47d'

# TRAIN
DM0 = '#a30234' 
DM1 = '#0076c0'
DM2 = '#67771a'






CB0 = CM0
CB1 = CM1


idxMulti = idxMulti-zerocut
idxMulti=np.array(sorted(idxMulti))

inputX, inputY, listFile1 = data_single(categories,dirDataSet)
inputX = inputX[zerocut:] ; inputY = inputY[zerocut:] ; listFile1 = listFile1[zerocut:]
inputX = torch.tensor(inputX) ; inputY = np.array(inputY) ; listFile1 = np.array(listFile1)
inputX = inputX[idxMulti] ; inputY = inputY[idxMulti] ; listFile1 = listFile1[idxMulti]
inputX = inputX.reshape(-1,160,200,170,1)
print('Test tensor Done : %d'%(len(inputX)))

intermediates = [] ; color_binary = [] ; color_multi = []
temp_intermediates = [] ; temp_color_binary = [] ; temp_color_multi = []


for i in range(len(inputX)):
    if((i+1)%25==0): print('Test data %d / %d'%(i, len(inputX)))
    output_class = np.argmax(inputY[i])
    if(output_class == 0):
        temp_color_multi.append(CM0)
        temp_color_binary.append(CB0)
    elif(output_class == 1):
        temp_color_multi.append(CM1)
        temp_color_binary.append(CB1)
    elif(output_class == 2):
        temp_color_multi.append(CM2)
        temp_color_binary.append(CB1)
            
    with torch.no_grad():
        images = inputX[i].view(-1,1,imgRow,imgCol,imgDepth)
        images = images.float()
        images = images.to(device)
        if(0):
            intermediate_tensor = net.module.extractor(images)
            intermediate_tensor = intermediate_tensor.view(35640)
        else:
            intermediate_tensor = net.module.extractor(images)
            intermediate_tensor = intermediate_tensor.view(-1,35640)
            intermediate_tensor = net.module.classifier[0](intermediate_tensor)
            intermediate_tensor = intermediate_tensor.view(64)
        intermediate_tensor = intermediate_tensor.detach().cpu().numpy()
        temp_intermediates.append(intermediate_tensor)
        images = images.detach().cpu()
        del images, intermediate_tensor


del inputX, inputY
torch.cuda.empty_cache()
import gc
gc.collect()

temp_intermediates = np.array(temp_intermediates)
temp_color_multi = np.array(temp_color_multi)
temp_color_binary = np.array(temp_color_binary)



for idx in range(len(idxMulti)):
    for_idx = idxMulti[idx]
    intermediates.append(temp_intermediates[idx])
    color_multi.append(temp_color_multi[idx])
    color_binary.append(temp_color_binary[idx])
    
    
dirDataSet = "../../dataset/60_Obl_160_LPI_160EAM"
MODE='TRAIN'
zerocut=94



inputX, inputY, listFile2 = data_single(categories,dirDataSet)
inputX = inputX[zerocut:] ; inputY = inputY[zerocut:] ; listFile2 = listFile2[zerocut:]
inputX = torch.tensor(inputX) ; listFile2 = np.array(listFile2)
inputX = inputX.reshape(-1,160,200,170,1)
listFileW = np.hstack([listFile1,listFile2])
print('\n Train tensor Done : %d'%(len(inputX)))


listName = []
for i in range(len(listFileW)):
    listName.append(listFileW[i][-19:-14])
listName = np.array(listName)


for i in range(len(inputX)):
    if((i+1)%35==0): print('Train data %d / %d'%(i, len(inputX)))

    output_class = np.argmax(inputY[i])
    if(output_class == 0):
        color_multi.append(DM0)
    elif(output_class == 1):
        color_multi.append(DM1)
    elif(output_class == 2):
        color_multi.append(DM2)

    with torch.no_grad():
        images = inputX[i].view(-1,1,imgRow,imgCol,imgDepth)
        images = images.float()
        images = images.to(device)
        if(0):
            intermediate_tensor = net.module.extractor(images)
            intermediate_tensor = intermediate_tensor.view(35640)
        else:
            intermediate_tensor = net.module.extractor(images)
            intermediate_tensor = intermediate_tensor.view(-1,35640)
            intermediate_tensor = net.module.classifier[0](intermediate_tensor)
            intermediate_tensor = intermediate_tensor.view(64)
        intermediate_tensor = intermediate_tensor.detach().cpu().numpy()
        intermediates.append(intermediate_tensor)
        images = images.detach().cpu()
        del images, intermediate_tensor

del inputX, inputY
torch.cuda.empty_cache()
import gc
gc.collect()


if(select=='tSNE'):
    tsne = TSNE(n_components=2, random_state=KERNEL_SEED, perplexity = perp, learning_rate=epsi,n_iter=step)
    intermediates_tsne = tsne.fit_transform(intermediates)
    if(layer_of_interest==8):
        pca = PCA(n_components = pca_dim, random_state=1)
        Principals = pca.fit_transform(intermediates)
        intermediates_PCA = tsne.fit_transform(Principals)
        intermediates_PCA = np.array(intermediates_PCA)

elif(select=='UMAP'):
    reducer = umap.UMAP(n_components=2, random_state=KERNEL_SEED, n_neighbors=perp, min_dist=mini)
    intermediates_tsne = reducer.fit_transform(intermediates)
    
color_multi = np.array(color_multi)
color_binary = np.array(color_binary)

dict_ = {'File':listName,'X':intermediates_tsne[:,0],'Y':intermediates_tsne[:,1],'color':color_multi}
excel = pd.DataFrame(dict_)
excel.to_csv('%s_%d_result.csv' %(select,KERNEL_SEED))

figure_single('M', 'T&TM', color_multi, idxMulti, intermediates_tsne, layer_path, select, KERNEL_SEED,pca_dim,df_color,CM0,CM1,CM2,DM0,DM1,DM2)


print('\n=========================================== \n 3D_projection_initiate \n===========================================')

zero = [0] * 66 ; one = [1] * 100 ; two = [2] * 60
label = zero + one + two
zero2 = [0] * 25 ; one2 = [1] * 25 ; two2 = [2] * 25
label2 = zero2 + one2 + two2
label3 = label2 + label
label = np.array(label3)




fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

if(select=='tSNE'):
    tsne = TSNE(n_components=3, random_state=KERNEL_SEED, perplexity = perp, learning_rate=epsi,n_iter=step*10)
    intermediates_tsne = tsne.fit_transform(intermediates)
    if(layer_of_interest==8):
        pca = PCA(n_components = pca_dim, random_state=1)
        Principals = pca.fit_transform(intermediates)
        intermediates_PCA = tsne.fit_transform(Principals)
        intermediates_PCA = np.array(intermediates_PCA)

elif(select=='UMAP'):
    reducer = umap.UMAP(n_components=3, random_state=KERNEL_SEED, n_neighbors=perp, min_dist=mini)
    intermediates_tsne = reducer.fit_transform(intermediates)

color3D = color_multi
SETT ='M'
color_list = [DM0,DM1,DM2,CM0,CM1,CM2]
#    color_list = ["#e50000","#ff6f52","#030aa7", "#4984b8", "#0a5f38", '#65ab7c']

for color_idx in color_list:
    idx = color_multi==color_idx
    if(df_color):
        if(color_idx==DM0):
            label_col = 'Train No HS'
        elif(color_idx==CM0):
            label_col = 'Test No HS'
                
        elif(color_idx==DM1):
            if(SETT=='M')or(SETT=='P'):
                label_col = 'Train Left HS'
            elif(SETT=='B'):
                label_col = 'HS'
        elif(color_idx==CM1):
            if(SETT=='M')or(SETT=='P'):
                label_col = 'Test Left HS'
            elif(SETT=='B'):
                label_col = 'HS'
                    
        elif(color_idx==DM2):
            label_col = 'Train Right HS'
        elif(color_idx==CM2):
            label_col = 'Test Right HS'
    
    if(color_idx==CM0)or(color_idx==CM1)or(color_idx==CM2): # TEST
#        ax.scatter(intermediates_tsne[:,0][idx], intermediates_tsne[:,1][idx], intermediates_tsne[:,2][idx], zdir='z', s=20, c=color_idx,label=label_col, depthshade=False) #1
#        ax.scatter(intermediates_tsne[:,0][idx], intermediates_tsne[:,1][idx], intermediates_tsne[:,2][idx], zdir='z', s=20, c=color_idx,label=label_col, depthshade=False, edgecolor='black', linewidth=0.5) #2
#        ax.scatter(intermediates_tsne[:,0][idx], intermediates_tsne[:,1][idx], intermediates_tsne[:,2][idx], zdir='z', s=20, c=color_idx,label=label_col, depthshade=False, marker='^' ) #3
        ax.scatter(intermediates_tsne[:,0][idx], intermediates_tsne[:,1][idx], intermediates_tsne[:,2][idx],
                   zdir='z', s=p_size, c=color_idx,label=label_col, depthshade=False, marker='^' , edgecolor='black', linewidth=l_width) #4
    
    elif(color_idx==DM0)or(color_idx==DM1)or(color_idx==DM2): # TRAIN
        ax.scatter(intermediates_tsne[:,0][idx], intermediates_tsne[:,1][idx], intermediates_tsne[:,2][idx],
                   zdir='z', s=30, c=color_idx,label=label_col, depthshade=False)
#        ax.scatter(intermediates_tsne[:,0][idx], intermediates_tsne[:,1][idx], intermediates_tsne[:,2][idx], zdir='z', s=20, c=color_idx,label=label_col, depthshade=False, edgecolor='black', linewidth=0.5) #0

#ax.set_xlabel('$X$', fontsize=10, rotation=0)
#ax.set_ylabel('$Y$', fontsize=10)
#ax.set_zlabel(r'Z', fontsize=10, rotation=0)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.legend(prop={'size': 15})
#ax.figure

ax.figure.savefig('./%s/K%dP%dE%dS%d/layer%d/%s3D_stop_%s_K%d_%d.png'%(select,KERNEL_SEED,perp,epsi,step,layer_of_interest,select,MODE,KERNEL_SEED,layer_of_interest))
ax.view_init(azim=99)
ax.figure.savefig('./%s/K%dP%dE%dS%d/layer%d/%s3D_stop_%s_K%d_%d.png'%(select,KERNEL_SEED,perp,epsi,step,layer_of_interest,select,MODE,KERNEL_SEED,layer_of_interest))


angle = 3
def rotate(angle):
     ax.view_init(azim=angle)
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save('./%s/K%dP%dE%dS%d/layer%d/%s3D_%s_K%d_%d.gif' %(select,KERNEL_SEED,perp,epsi,step,layer_of_interest,select,MODE,KERNEL_SEED, layer_of_interest), writer=animation.PillowWriter(fps=20))



finish = time.time() - start
hour = int(finish // 3600)
minute = int((finish - hour * 3600) // 60)
second = int(finish - hour*3600 - minute*60)
timetime =str(hour) +'h '+ str(minute)+'m ' + str(second)+'s'
temp_log = "\n\n Time Elapse: %s \n\n\n" %(timetime)
print(temp_log)
fwSt.write(temp_log)
fwSt.close()
