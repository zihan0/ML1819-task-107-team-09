import csv

def Load_dataset(DocPath,lDataPath,mask) :

    #re-writing file labeldata.csv
    f = open(lDataPath, "w+")
    f.truncate()

    with open(DocPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        col_data = [None]*len(mask)


        for row in csv_reader:

            ## Selecting masked columns
            count = 0
            for pos in mask:
                col_data[count] = row[pos]
                count = count + 1;

            ##Opening labeldata.csv
            with open(lDataPath, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(col_data)


def main():

    DocPath = "/Users/sid/Desktop/gender-classifier-DFE-791531.csv"
    lDataPath = "/Users/sid/Desktop/labeldata.csv"
    mask = [0,1,2,3,7] # columns chosen from the ender-classifier-DFE-791531.csv to be represented in labeldata.csv
    Load_dataset(DocPath,lDataPath,mask)

if __name__== "__main__":
  main()




