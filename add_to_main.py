import csv
import sys
import emoji

def Load_dataset(DocPath,lDataPath,mask) :


    #re-writing file labeldata.csv
    f = open(lDataPath, "w+")
    f.truncate()

    with open(DocPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        col_data = [None]*(len(mask)+1)
        coloumn_label = 0;

        for row in csv_reader:

            ## Selecting masked columns
            count = 0
            for pos in mask:
                hashtag_count = 0
                col_data[count] = row[pos]
                count = count + 1;

                if pos == 8:
                   if coloumn_label == 1 :
                    hashtag_count =  count_hashtag('#',row[pos])
                    col_data[len(mask)] = hashtag_count
                   else:
                       coloumn_label  = 1
                       col_data[len(mask)] = "Num_Hashtag"


            ##Opening labeldata.csv
            with open(lDataPath, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(col_data)

def count_hashtag(char,text):
    if len(text)== 0:
        return 0
    count = 1 if text[0] == char else 0
    return count + count_hashtag(char, text[1:])


def main():
    sys.setrecursionlimit(6000)
    DocPath = "/Users/sid/Desktop/gender-classifier-DFE-791531.csv" ##"Dataset/gender-classifier-DFE-791531.csv"
    lDataPath = "/Users/sid/Desktop/labeldata.csv" ##"Dataset/labeldata.csv"
    mask = [0,1,2,4,5,7,8,10] # columns chosen from the ender-classifier-DFE-791531.csv to be represented in labeldata.csv
    Load_dataset(DocPath,lDataPath,mask)
if __name__== "__main__":
  main()




