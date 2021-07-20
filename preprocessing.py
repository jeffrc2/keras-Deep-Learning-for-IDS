import pandas
import math

from sklearn.preprocessing import MinMaxScaler


# truthdf = label.copy()

def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')


def logNumericize(df):
    for col_index in range(df.shape[1]):
        print("Column \t" + str(col_index))
        if col_index == 1 or col_index == 2 or col_index == 3 or col_index == 41:
            print("numericizing. \n")
            counter = 0
            items = dict()
            for row_index in range(df.shape[0]):
                currval = df.loc[row_index, col_index]
                if currval in items:
                    df.loc[row_index, col_index] = items[currval];
                else:
                    counter+=1;
                    items[currval] = counter;
                    df.loc[row_index, col_index] = counter;
                progressBar(row_index, df.shape[0])
        curr_column = df[col_index]
        print("checking for log scaling.")
        curr_max_value = curr_column.max()
        curr_min_value = curr_column.min()
        print("min: " + str(curr_min_value))
        print("max: " + str(curr_max_value))
        if (curr_max_value - curr_min_value > 100):
            print("log scaling. \n")
            for row_index in range(df.shape[0]):
                currval = df.loc[row_index, col_index]
                df.loc[row_index, col_index] = math.log1p(currval)
                progressBar(row_index, df.shape[0])

df = pandas.read_csv('KDDTrain+.csv', header=None)

df.drop(df.columns[42], axis=1, inplace=True)

label = df[[df.columns[41]]]

logNumericize(df);

df.to_csv(index=False, path_or_buf='KDDTrain+processed.csv', header=False)


train = pandas.read_csv('KDDTrain+processed.csv', header=None)

   

    # pen_column = df[col_index]
    # max_val = pen_column.max()
    # min_val = pen_column.min()
    # print("normalizing. \n")
    # print(min_val)
    # print("\n")
    # print(max_val)
    # if (max_val - min_val > 1):
        # for row_index in range(df.shape[0]):
            # currval = df.loc[row_index, col_index]
            # df.loc[row_index, col_index] = (currval - min_val)/(max_val - min_val)
            # progressBar(row_index, df.shape[0])
    
	
