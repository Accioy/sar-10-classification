import os, random, shutil
def moveFile(fileDir,tarDir,rate):
    pathDir = os.listdir(fileDir)    #取图片的原始路径
    filenumber=len(pathDir)
    picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
    print (sample)
    for name in sample:
        filepath=os.path.join(fileDir,name)
        tarpath=os.path.join(tarDir,name)
        shutil.move(filepath, tarpath)
    return

if __name__ == '__main__':
    rootdir=os.path.abspath("./")
    orign_data_Dir = os.path.join(rootdir,'train')    #源图片文件夹路径
    target_Dir = os.path.join(rootdir,'validation')    #移动到新的文件夹路径
    rate=0.15    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    class_list=os.listdir(orign_data_Dir)
    for i in class_list:
        filesdir=os.path.join(orign_data_Dir,i)
        tarDir=os.path.join(target_Dir,i)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        moveFile(filesdir,tarDir,rate)


