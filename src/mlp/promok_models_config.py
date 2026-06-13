
import pandas as pd
from cashia_model.promok_model import *
import os

feature_engineering = "standardization"
# feature_engineering = "normalization"

local_notes = "Usando: "+feature_engineering+"Sin remplazar campos vacios por 'Unknown'"
zip_digits = 4

notes = "zip_digits = "+str(zip_digits)+"  "+local_notes

# Fecha máxima de los datos para el aprendizaje y evaluación
min_date =  pd.to_datetime('2023-01-01')
max_date =  pd.to_datetime('2026-05-31')
source_file_name = 'Data Set Cashia Entrenamiento.csv'
# source_file_name = 'Data Set Cashia V4.csv' #Not Used
# source_file_name = 'Data Set CashiaV3.csv'
# source_file_name = 'Valores Act Rech7.csv'
#source_file_name = 'Query Nopagos.csv'
#source_file_name = 'Query RNV Frecuencia Score.csv'

configuration_file_name = "CashIA_ConfFile.xlsx"

			

# conf_column = "NV_Agt"
# conf_column = "RNV_Agt"
# conf_column = "NV_CC"
# conf_column = "RNV_CC"
# conf_column = "NV_Agt_CS"
# conf_column = "RNV_Agt_CS"
# conf_column = "NV_CC_CS"
conf_column = "RNV_CC_CS"


stats_file_name = "Statistics.xlsx"
test_stats_file_name = "StatisticsTest.xlsx"

#model_conf, _ = os.path.splitext(configuration_file_name)
source_fn, _ = os.path.splitext(source_file_name)
#data_file_name = source_fn+model_conf+"Processed.csv"

data_file_name = source_fn+conf_column+"Processed.csv"

test_size = 0.2

models_to_test = [#ModelToTest("Random_Forest_SMOTE", {'criterion':'gini', 'bootstrap':True, 'random_state':100}, model_conf=conf_column),
                  ModelToTest("Random_Forest", {'criterion':'gini', 'bootstrap':True, 'random_state':100}, model_conf=conf_column),
                  #ModelToTest("Regresion", {'random_state':0, 'solver':'newton-cg', 'max_iter':4000}, model_conf=conf_column),
                  #ModelToTest("Naive Bayes", {}, model_conf=conf_column),
                  #ModelToTest("Ada Boost", {'criterion':'entropy', 'max_depth':1, 'random_state':1, 
                  #                          'n_estimators':500, 'learning_rate':0.1,'random_state':1}, model_conf=conf_column),
                  #ModelToTest("SVC", {'kernel':'linear'}, model_conf=conf_column),
                  #ModelToTest("KNN", {'n_neighbors':5}, model_conf=conf_column),
                  #ModelToTest("KNN", {'n_neighbors':20}, model_conf=conf_column),
                  #ModelToTest("KNN", {'n_neighbors':21}, model_conf=conf_column),
                  #ModelToTest("KNN", {'n_neighbors':22}, model_conf=conf_column),
                  #ModelToTest("KNN", {'n_neighbors':23}, model_conf=conf_column),
                  ]

MLP_RESOURCE_KEYS = {
    "inflation_table": "mlp/inputs/tablaDeInflacion.csv"
}

def get_mlp_resource_key(name: str) -> str:
    try:
        return MLP_RESOURCE_KEYS[name]
    except KeyError as e:
        available = ", ".join(sorted(MLP_RESOURCE_KEYS.keys()))
        raise KeyError(
            f"Unknown MLP resource: '{name}'. Available: {available}"
        ) from e