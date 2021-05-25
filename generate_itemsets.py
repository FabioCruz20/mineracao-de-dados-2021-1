from  apyori import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import KBinsDiscretizer

# -------------------------------- SUBPROGRAMAS ---------------------------- #

def discretize(data, 
    indexes=('Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE')):
    """ Discretiza os campos numéricos do dataset """

    for i in indexes:
        data[i] = pd.qcut(data[i], q=10, duplicates='drop')

    return None


def transform(row):
    """ Transforma uma linha de dataframe em uma lista para o apyori """

    return [
        attr + '=' + str(value) for attr, value in row.items()
    ]

# ------------------------------- PRINCIPAL -------------------------------- #

file_name = 'data/ObesityDataSet_raw_and_data_sinthetic.csv'

data = pd.read_csv(file_name, header='infer')

discretize(data)

data_rows = [ transform(row) for i, row in data.iterrows() ]

print('Antes do apriori')
results = apriori(data_rows, min_support=0.05, min_confidence=0.05, min_lift=0.05)
print('Depois do apriori')

print('escrevendo arquivo')
with open('itemsets/items_{0:s}.csv'.format(datetime.now()), 'w') as arq:
    for r in results:
        row = '{:s},{:f}\n'.format(str(tuple(r.items)), r.support)
        print(row)
        arq.write(row)


# itemset_df = pd.DataFrame([ (r.items, r.support) for r in results ], columns=['itemsets', 'support'])

# print('Gerando regras')
# rules = association_rules(itemset_df, min_threshold=0.05)

# print('escrevendo arquivo')
# with open('saida.csv', 'w') as arq:
#     for r in rules:
#         for field, val in r.iterrows():
#             print(val, end=';')
#         print()
