import csv
with open('training.csv') as csvfile:
    datareader = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
    i = 0
    CNNcount = 0
    GISTcount = 0
    data = {}
    final_data = {}
    for row in datareader:
        data[i] = ', '.join(row)
        #print (', '.join(row))
        i += 1
    for item in data[0].split(","):
        #print(item)
        if item == "CNN":
            CNNcount += 1
        elif item == "GIST":
            GISTcount += 1
    newData = {}
    j = 1
    for row in data[j].split(","):
        i = 0
        newRow = []
        tempRow = []
        CNN = []
        GIST = []
        print("row length: ",len(row))
        while i < CNNcount+GISTcount:
            if i < CNNcount:
                CNN.append(row[i])
            elif i >= CNNcount and i != CNNcount+GISTcount+2 and i < CNNcount+GISTcount:
                print("i:",i)
                GIST.append(row[i])
            i+=1
        tempRow.append(CNN)
        tempRow.append(GIST)
        newData[j] = tempRow
        j+=1
        
    print(newData[0])