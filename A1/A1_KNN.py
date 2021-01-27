import pandas as pd
import sys
import random


def pickUnseenData(df, num):
    # ramdom shuffle the original data in order to randomly get num instances as unseen_instances
    index_li = [x for x in range(df.shape[0])]
    random.shuffle(index_li)
    unseen_df = pd.DataFrame(columns=df.columns, dtype=int)
    drop_li = []
    for i in range(num):
        unseen_df = unseen_df.append(df.iloc[index_li[i]])
        # drop_li.append(df.index[index_li[i]])
    df = df.drop(drop_li)
    return (df, unseen_df)

# caculate distance between instance_1 and instance_2
# return distance and class


def getDistance(ins1, ins2):
    dis = (ins1-ins2).apply(lambda x: x**2)[:-1].sum()
    return(dis, ins1["Class"])


def knn(seen_df, unseen_df, k):
    ans = pd.DataFrame(columns=['Class'], index=unseen_df.index)
    for unseen_i in range(unseen_df.shape[0]):
        dis_df = pd.DataFrame(columns=['Distance', 'Class'])
        unseen_instance = unseen_df.iloc[unseen_i]
        for seen_i in range(seen_df.shape[0]):
            seen_instance = seen_df.iloc[seen_i]
            dis, seen_class = getDistance(seen_instance, unseen_instance)
            dis_df = dis_df.append(
                {'Distance': dis, 'Class': seen_class}, ignore_index=True)
        # get k near neibours
        k_df = dis_df.sort_values(by='Distance')[:k]
        if (k_df[k_df['Class'] == 0].shape[0] > k_df[k_df['Class'] == 1].shape[0]):
            ans.at[unseen_instance.name, 'Class'] = 0
        else:
            ans.at[unseen_instance.name, 'Class'] = 1
    return ans


if __name__ == "__main__":
    if(len(sys.argv) < 4):
        print("Useage: {} <data_file> <k> <unseen_data_num>".format(
            sys.argv[0]))
        exit(1)
    df = pd.read_csv("Data4A1.tsv", sep="\t",
                     header=0, index_col="Sequence.id")
    # print(seen_df.shape, unseen_df.shape)
    seen_df, unseen_df = pickUnseenData(df, int(sys.argv[3]))
    result = knn(seen_df, unseen_df, int(sys.argv[2]))
    print(result)
    print(unseen_df['Class'])
    diff = result-unseen_df
    print(diff.shape[0], diff[diff['Class'] == 0].shape[0])
