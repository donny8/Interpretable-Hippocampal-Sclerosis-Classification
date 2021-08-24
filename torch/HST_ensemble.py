from HST_util import *
from HST_util_ens import *
from HST_common import *

def main():
    fw, fwSt = log_ensemble()
    if(SETT=='FUL'):
        Y_vector = []
        inputX, inputY, _, _ = data_single(categories,tstDataSet)
    else:
        inputX, inputY, Y_vector,_,_ = source_load(categories, dirDataSet)
    MAIN_ENSB(inputX,inputY,Y_vector,fw,fwSt)
    log_close(fw, fwSt, start)

if __name__ == '__main__':
    main()
