import pandas as pd
import sys
import math

def B(q):
    if q == 1.0 or q == 0.0:
        #the entropy is 0
        return 0
    return -(q*math.log(q, 2) + (1 - q)*math.log(1 - q, 2))

def Remainder(df, attr_name, decision_attr, p, n):
    res = 0
    for val in df[attr_name].values:
        p_k = len(df.loc[(df[attr_name] == val) & (df[decision_attr] == 'Yes')])
        n_k = len(df.loc[(df[attr_name] == val) & (df[decision_attr] == 'No')])
        res += (((p_k + n_k)/(p + n)) + B(p_k/(p_k + n_k)))
    return res

def Gain(df, attr_name, decision_attr):
    p = len(df.loc[df[decision_attr] == 'Yes'])
    n = len(df.loc[df[decision_attr] == 'No'])
    res = B(p/(p + n)) - Remainder(df, attr_name, decision_attr, p, n)
    return res

def learn(df, 

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df

def main(csv_path):
    df = load_csv(csv_path)
    #the last one
    decision_attr = list(df.keys())[-1]
    #

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: {} csv_path'.format(sys.argv[0]))
        sys.exit(1)

    main(*sys.argv[1:])
