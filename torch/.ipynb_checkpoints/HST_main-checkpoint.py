from HST_common import *
from HST_util import *

def main():
    Rep, Sum = log_intro()
    inputX, inputY, Y_vector, _, _ = source_load(categories, dirDataSet)
    if(SETT=='SIG'):
        CV_train(inputX,inputY,Y_vector,Rep,Sum) 
        CV_eval(inputX,inputY,Y_vector,Rep,Sum)
    elif(SETT=='FUL'):
        testX, testY, _, _ = data_single(categories,tstDataSet)
        FUL_train(inputX,inputY,testX,testY,Rep,Sum)
        FUL_eval(inputX,inputY,testX,testY,Rep,Sum)
    log_close(Rep, Sum, start)
    
if __name__ == '__main__':
    main()


