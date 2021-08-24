import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from reduction_func import *

start = time.time()

dir_path = './%s' %(SELECT)
kernel_path = dir_path + '/K%dP%dE%dS%d' %(KERNEL_SEED,PERP,EPSI,STEP)
layer_path = kernel_path + '/layer%d' %(layer_of_interest)

if(not(os.path.isdir(dir_path))):
        os.mkdir(dir_path)
if(not(os.path.isdir(kernel_path))):
        os.mkdir(kernel_path)        
if(not(os.path.isdir(layer_path))):
        os.mkdir(layer_path)



SETT = 'SIG' ; AUG = 'hflip' ; CONTROLTYPE = 'CLRM'
TRIAL = 13 ; loadmodelNum = 6020 ; FOLD_SEED = 1 ; 

imgCountNO_0 =  168 ; imgCountNO_7 = 69 ; imgCountNO_4 = 15
imgCountYES_1 = 25 ; imgCountYES_2 = 25 ; imgCountYES_3 = 0
categories = ["C047_no","C1_left_yes","C2_right_yes"]
iters = 100; pca_dim = 200; mini = 0.1
num1=0 ; num2=0 ; num3=50



## model load ##
modelName = '../saveModel/[%s%d%s]modelHS%s_%d{F%dK%d}.json' %(SETT,TRIAL,AUG,CONTROLTYPE,loadmodelNum,FOLD_SEED,KERNEL_SEED)
if os.path.isfile(modelName):
    json_file = open(modelName, "r") 
    loaded_model_json = json_file.read() 
    json_file.close() 
    modelHS = model_from_json(loaded_model_json)
else:
    print('\n!!!warning!!! \n load model file not exist!!')
            
weightFileName = '../saveModel/[%s%d%s]HS%s_%d{F%dK%d}[0](best).h5' %(SETT,TRIAL,AUG,CONTROLTYPE,loadmodelNum,FOLD_SEED,KERNEL_SEED)
print('weightFileName %s' %(weightFileName))
if os.path.isfile(weightFileName):
    modelHS.load_weights(weightFileName)
else:
    print('\n!!!warning!!! \n load Weight file not exist!!')
            
## Compiling the model
learning_rate = 1e-2; nb_epochs = 50
decay_rate = learning_rate / nb_epochs
adam=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
modelHS.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
intermediate_tensor_function = K.function([modelHS.layers[0].input],[modelHS.layers[layer_of_interest].output])


storage = './%s_Storage.txt' %(SELECT)
fwSt=open(storage,'a')
fwSt.write('\n\n[KERNEL %d Layer %d]\n'%(KERNEL_SEED, layer_of_interest))
idxMulti = balance(CONTROLTYPE,SETT,TRIAL,AUG,KERNEL_SEED,iters, imgCountNO_4, imgCountNO_7,imgCountNO_0,imgCountYES_1,imgCountYES_2,num1,num2,num3)

test=1


dirDataSet = "../../dataset/75_add_new4_from71"
MODE='TEST'
zerocut=86
# TEST
CM0 = '#fea993' 
CM1 = '#a2cffe'
CM2 = '#9be5aa'
# TRAIN
DM0 = '#e50000'
DM1 = '#030aa7'
DM2 = '#0a5f38'
        
idxMulti = idxMulti-zerocut
idxMulti=np.array(sorted(idxMulti))

inputX, inputY, listFile1 = data_single(categories,dirDataSet)
inputX = inputX[zerocut:] ; inputY = inputY[zerocut:] ; listFile1 = listFile1[zerocut:]
inputX = np.array(inputX) ; inputY = np.array(inputY) ; listFile1 = np.array(listFile1)
inputX = inputX[idxMulti] ; inputY = inputY[idxMulti] ; listFile1 = listFile1[idxMulti]
inputX = inputX.reshape(-1,160,200,170,1)


intermediates = [] ; color_multi = []
temp_intermediates = [] ; temp_color_multi = []


for i in range(len(inputX)):
    output_class = np.argmax(inputY[i])
    if(output_class == 0):
        temp_color_multi.append(CM0)
    elif(output_class == 1):
        temp_color_multi.append(CM1)
    elif(output_class == 2):
        temp_color_multi.append(CM2)            
    intermediate_tensor = intermediate_tensor_function([inputX[i].reshape(1,160,200,170,1)])[0][0]
    temp_intermediates.append(intermediate_tensor)

temp_intermediates = np.array(temp_intermediates)
temp_color_multi = np.array(temp_color_multi)

for idx in range(len(idxMulti)):
    intermediates.append(temp_intermediates[idx])
    color_multi.append(temp_color_multi[idx])
    

    
dirDataSet = "../../dataset/60_Obl_160_LPI_160EAM"
MODE='TRAIN'
zerocut=94

inputX, inputY, listFile2 = data_single(categories,dirDataSet)
inputX = inputX[zerocut:] ; inputY = inputY[zerocut:] ; listFile2 = listFile2[zerocut:]
inputX = np.array(inputX) ; listFile2 = np.array(listFile2)
inputX = inputX.reshape(-1,160,200,170,1)
listFile = np.hstack([listFile1,listFile2])

listName = []
for i in range(len(listFile)):
    listName.append(listFile[i][-19:-14])
listName = np.array(listName)


for i in range(len(inputX)):
    output_class = np.argmax(inputY[i])
    if(output_class == 0):
        color_multi.append(DM0)
    elif(output_class == 1):
        color_multi.append(DM1)
    elif(output_class == 2):
        color_multi.append(DM2)
    intermediate_tensor = intermediate_tensor_function([inputX[i].reshape(1,160,200,170,1)])[0][0]
    intermediates.append(intermediate_tensor)    
    
if(SELECT=='tSNE'):
    tsne = TSNE(n_components=2, random_state=KERNEL_SEED, perplexity = PERP, learning_rate=EPSI,n_iter=STEP)
    intermediates_tsne = tsne.fit_transform(intermediates)

elif(SELECT=='UMAP'):
    reducer = umap.UMAP(n_components=2, random_state=KERNEL_SEED, n_neighbors=PERP, min_dist=mini)
    intermediates_tsne = reducer.fit_transform(intermediates)
    
color_multi = np.array(color_multi)

dict_ = {'File':listName,'X':intermediates_tsne[:,0],'Y':intermediates_tsne[:,1],'color':color_multi}
excel = pd.DataFrame(dict_)
excel.to_csv('%s_%d_result.csv' %(SELECT,KERNEL_SEED))

figure_single('M', 'T&TM', color_multi, idxMulti, intermediates_tsne, layer_path, SELECT, KERNEL_SEED,pca_dim,df_color,CM0,CM1,CM2,DM0,DM1,DM2)


print('\n=========================================== \n 3D_projection_initiate \n===========================================')

zero = [0] * 66 ; one = [1] * 100 ; two = [2] * 60
label = zero + one + two
zero2 = [0] * 25 ; one2 = [1] * 25 ; two2 = [2] * 25
label2 = zero2 + one2 + two2
label3 = label2 + label
label = np.array(label3)




fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

if(SELECT=='tSNE'):
    tsne = TSNE(n_components=3, random_state=KERNEL_SEED, perplexity = PERP, learning_rate=EPSI,n_iter=STEP*10)
    intermediates_tsne = tsne.fit_transform(intermediates)

elif(SELECT=='UMAP'):
    reducer = umap.UMAP(n_components=3, random_state=KERNEL_SEED, n_neighbors=PERP, min_dist=mini)
    intermediates_tsne = reducer.fit_transform(intermediates)

color3D = color_multi
SETT ='M'
color_list = [DM0,DM1,DM2,CM0,CM1,CM2]

for color_idx in color_list:
    idx = color_multi==color_idx
    if(color_idx==DM0):
        label_col = 'Train No HS'
    elif(color_idx==CM0):
        label_col = 'Test No HS'
            
    elif(color_idx==DM1):
        label_col = 'Train Left HS'
    elif(color_idx==CM1):
        label_col = 'Test Left HS'
                
    elif(color_idx==DM2):
        label_col = 'Train Right HS'
    elif(color_idx==CM2):
        label_col = 'Test Right HS'
    
    if(color_idx==CM0)or(color_idx==CM1)or(color_idx==CM2): # TEST
        ax.scatter(intermediates_tsne[:,0][idx], intermediates_tsne[:,1][idx], intermediates_tsne[:,2][idx], zdir='z', s=p_size, c=color_idx,label=label_col, depthshade=False, marker='^' , edgecolor='black', linewidth=l_width) #4
    elif(color_idx==DM0)or(color_idx==DM1)or(color_idx==DM2): # TRAIN
        ax.scatter(intermediates_tsne[:,0][idx], intermediates_tsne[:,1][idx], intermediates_tsne[:,2][idx], zdir='z', s=30, c=color_idx,label=label_col, depthshade=False)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.legend(prop={'size': 15})
#ax.figure

ax.figure.savefig('./%s/K%dP%dE%dS%d/layer%d/%s3D_stop_%s_K%d_%d.png'%(SELECT,KERNEL_SEED,PERP,EPSI,STEP,layer_of_interest,SELECT,MODE,KERNEL_SEED,layer_of_interest))
ax.view_init(azim=99)
ax.figure.savefig('./%s/K%dP%dE%dS%d/layer%d/%s3D_stop_%s_K%d_%d.png'%(SELECT,KERNEL_SEED,PERP,EPSI,STEP,layer_of_interest,SELECT,MODE,KERNEL_SEED,layer_of_interest))


angle = 3
def rotate(angle):
     ax.view_init(azim=angle)
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save('./%s/K%dP%dE%dS%d/layer%d/%s3D_%s_K%d_%d.gif' %(SELECT,KERNEL_SEED,PERP,EPSI,STEP,layer_of_interest,SELECT,MODE,KERNEL_SEED, layer_of_interest), writer=animation.PillowWriter(fps=20))


fin_time(start,fwSt)
fwSt.close()