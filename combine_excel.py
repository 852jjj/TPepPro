import pandas as pd
import os
#下面这句是包含excel文件的位置
for k in range(1,6):
    dir = r'./data/predict/'+'第'+str(k)+'折'
    filenames=[]
    for i in range(70):
         filenames.append(str(i)+'.xls')
    print(filenames)
    #index = 0
    dfs = []
    for name in filenames:
        #print(index)
        dfs.append(pd.read_excel(os.path.join(dir,name),sheet_name=0))
        #index += 1 #为了查看合并到第几个表格了
    df = pd.concat(dfs)
    df=df.dropna(axis=0, how='all')
    df=pd.DataFrame(df)
#保存到桌面的文件名
    df.to_excel(r'./data/'+'第'+str(k)+'折'+'TBTPpepI_yeast.xls')




