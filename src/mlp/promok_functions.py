import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import json
import copy
import sys

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, ConfusionMatrixDisplay

from datetime import datetime


#--- Models labraries
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

# Regresion
from sklearn.linear_model import LogisticRegression

#Naïve Bayes
from sklearn.naive_bayes import GaussianNB

# Ada Boost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# KNN 
from sklearn.neighbors import KNeighborsClassifier

# SVC
from sklearn import svm

# Split data set
from sklearn.model_selection import train_test_split

BY_SIZE = 1
BY_INDEX = 2
BY_DATE = 3

class TrainedModelStats:
    """Esta clase almacena las estadísticas del entrenamiento de un modelo
    """    
    def __init__(self):
        """Almacena as estadísticas de un modelo 
        
        """
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        self.fields = {"Date": dt_string,
                     "TrainInitDate":"Undefined",
                     "TrainEndDate":"Undefined",
                     "TestInitDate":"Undefined",
                     "TestEndDate":"Undefined",                    
                     "Method": "Unknown",
                     "Evaluation": "Unknown",
                     "Size": -1,
                     "Paid": -1,
                     "Unpaid": -1,
                     "TestSize": -1.0,
                     "AccMin": -1,
                     "Recall": -1,
                     "Precision": -1,
                     "F1": -1,
                     "Accuracy": -1,
                     "AUC": -1,
                     "Accurracy Gain": -1,
                     "Monetary Gain": -1,
                     "Reference Value": -1,
                     "Improvement" : -1,
                     "Rejection rate": -1,
                     "Confusion matrix": None,
                     "False negative rate":-1,
                     "Features": "Unknown",
                     "Parameters": "Unknown",
                     "Notes": "None",
                     "SourceFile": "None",
                     "Model": "None"}
        
    def __str__(self):
        """Imprime las estadísticas de la corrida de un modelo

        Returns:
            string: Fields of the stats
        """        
        return str(self.fields)
    
    def __repr__(self):
        """Regreas una cadena rica en información que permite recrear el objeto que contiene las estadísticas

        Returns:
            string: Descripción de las estadísticas
        """
        return str(self.fields.values())


class ModelsStats:
    """Esta clase almacena una serie de estadísticas del entrenamiento de varios modelos
    """    
    def __init__(self):
        """Crea un objeto capaz de almacenar estadísticas de entrenamiento de modelos
        """        
        self.values = []
        self.base_stats = TrainedModelStats()

    def append(self, element):
        """Agrega el conjunto de estadísticas del entrenamiento de un modelo

        Args:
            element (TrainedModelStats): Estadísticas del modelo que fue entrenado
        """        
        self.values.append(tuple(element.fields.values()))

    def add_and_print_stats(self, y_true, y_pred, method, parameters, evaluation, display=False):
        """Agrega, al conjunto de estadísticas, las estadísticas de un modelo y las imprime

        Args:
            y_true (vector): Verdaderos valores de la variable objetivo
            y_pred (vector): Valores predecidos de la variable objetivo
            method (string): Nombre del método utilizado
            parameters (dictionary): Diccionario con los parámetros del método
            evaluation (string): Características de la evaluación
        """        
        new_stats = copy.deepcopy(self.base_stats)

        new_stats.fields['Method'] = method
        new_stats.fields['Parameters'] = json.dumps(parameters)
        new_stats.fields['Evaluation'] = evaluation
        new_stats.fields["Recall"] = round(recall_score(y_true, y_pred)*100, 4)
        new_stats.fields["Precision"] = round(precision_score(y_true, y_pred, zero_division=0)*100, 4)
        new_stats.fields["F1"] = round(f1_score(y_true, y_pred, zero_division=0)*100, 4)
        new_stats.fields["Accuracy"] = round(accuracy_score(y_true, y_pred)*100, 4)

        # Cuando y_true y y_pred sólo tienen una sola clase AUC no puede ser calculado
        try:
            auc_score = roc_auc_score(y_true, y_pred)
            new_stats.fields["AUC"] = round(auc_score * 100, 4)
        except:
            new_stats.fields["AUC"] = None

        new_stats.fields['Accurracy Gain'] = round(
            (new_stats.fields['Accuracy']-new_stats.fields['AccMin'])/new_stats.fields['AccMin']*100, 4)
        
        new_stats.fields['Monetary Gain'], \
        new_stats.fields['Reference Value'], \
        new_stats.fields['Improvement'] = compute_gain(y_true, y_pred)
        new_stats.fields['Rejection rate'] = compute_rejection_rate(y_pred)
       

        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        new_stats.fields['Confusion matrix'] = str(cm)
        new_stats.fields['False negative rate'] = cm[1][0]/(cm[0][0]+cm[1][0])

        self.append(new_stats)

        if (display):
            self.print_problem_info(new_stats)
            self.print_scores(new_stats, evaluation, method)
            self.display_confusion_matrix(cm)

    def print_problem_info(self, new_stats):
        """Imprime la información del modelo, cuantos pagados, cuantos impagados y el tamaño del juego de datos

        Args:
            new_stats (TrainedModelStats): estadísticas del modelo
        """
        print(f"Pagados: {new_stats.fields['Paid']} Incobrables: {new_stats.fields['Unpaid']}") 
        print(f"Accuracy mínimo: {new_stats.fields['AccMin']}")
        print("Tamaño del data set: ", new_stats.fields['Size'])
        print("Porcentaje de impagados:", round(new_stats.fields['Unpaid']/new_stats.fields['Size']*100,4))

    def print_scores(self, stats, evaluation, method_name):
        """Imprime los diferentes scores del modelo

        Args:
            stats (TrainedModelStats): Estadísticas del modelo
        """        

        print(f"------{evaluation} scores for {method_name}------") 
        print(f"Recall: {stats.fields['Recall']}")
        print(f"Precision: {stats.fields['Precision']}")
        print(f"F1-Score: {stats.fields['F1']}")
        print(f"Accuracy score: {stats.fields['Accuracy']}")
        print(f"AUC Score: {stats.fields['AUC']}")
        print("Accurracy Gain: ", stats.fields['Accurracy Gain'])
        print("Monetary Gain: ", "${:,.0f}".format(stats.fields['Monetary Gain']))
        print("Reference Value: ", "${:,.0f}".format(stats.fields['Reference Value']))
        print("Improvement: ", "{:,.1f}%".format(stats.fields['Improvement']*100))
        print("Model rejection rate", "{:,.1f}%".format(stats.fields['Rejection rate']*100))
        print("False negative rate", "{:,.1f}%".format(stats.fields['False negative rate']*100))

    def display_confusion_matrix(self, cm):
        """Imprime la matriz de confución obtenida por el entrenaiento de un moelo

        Args:
            y_true (vector): vector con la variable objetivo real
            y_pred (vector): vector con la variable predecida por el modelo
        """        
        
        cm_display = ConfusionMatrixDisplay(cm).plot()
        plt.show()

        return cm

class ModelsTester:
    """Clase que prueba varios modelos de predicción
    """    
    def __init__(self, data, models, test_ratio=None, train_init=None, train_size=None, 
                 test_init=None, test_size=2000, train_end_date=None, test_end_date= None,
                 split=None):
        """Inicializa la clase con una lista de modelos a probar.

        Args:
            data (dataframe): Data frame que contiene los datos usados para los modelos
            models (list): _description_
        """        
        self.models = models
        self.data = data

        if split == BY_SIZE:
            x = self.data.drop("Incobrable", axis=1)
            y = self.data["Incobrable"]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, 
                                                                                    test_size = test_ratio, 
                                                                                    stratify = y, 
                                                                                    random_state = 7)
        elif split == BY_INDEX:
            print("Using selected data")
            train_end = train_init + train_size
            self.x_train = self.data.iloc[train_init:train_end].drop("Incobrable", axis=1)
            self.y_train = self.data.iloc[train_init:train_end]["Incobrable"]

            if test_init == None:
                test_init = train_end

            self.x_test = self.data.iloc[test_init:test_init+test_size].drop("Incobrable", axis=1)
            self.y_test = self.data.iloc[test_init:test_init+test_size]["Incobrable"]

        elif split == BY_DATE:
            print("Using train_end_date")
            data_train = self.data[self.data['Fecha Inicial'] <= train_end_date]
            data_test = self.data[(self.data['Fecha Inicial'] > train_end_date) & (self.data['Fecha Inicial'] <= test_end_date)]

            data_train = data_train.drop('Fecha Inicial', axis=1)
            data_test = data_test.drop('Fecha Inicial', axis=1)

            self.x_train = data_train.drop("Incobrable", axis=1)
            self.y_train = data_train["Incobrable"]

            self.x_test = data_test.drop("Incobrable", axis=1)
            self.y_test = data_test["Incobrable"]

            print(f'Size of trainig data {data_train.shape[0]}')
            print(f'Size of test data {data_test.shape[0]}')

        else:
            print("Error split option not recognized")
            sys.exit()
        
    def run_all_models(self, preprocessor=None):
        """Ejectuta todos los modelos que se hayan dado de alta en la clase.
        """        
        for model in self.models:
            print("Running:", model.name)
            model.pipeline = self.run_model(model.name, model.parameters, preprocessor)

    def run_model(self, model, parameters, preprocessor=None):
        """Ejecuta un modelo en particular

        Args:
            model (str): Nombre del modelo
            parameters (diccionario): Diccionario con los parámetros del modelo

        Returns:
            pipelene: pipeline para la ejecución del modelo
        """        
        apply_SMOTE = True

        if model == "Random_Forest_SMOTE":
            clf = RandomForestClassifier(criterion=parameters['criterion'],
                                         bootstrap=parameters['bootstrap'],
                                         random_state=parameters['random_state'])
            
        elif model == "Random_Forest":
            clf = RandomForestClassifier(criterion=parameters['criterion'],
                                         bootstrap=parameters['bootstrap'],
                                         random_state=parameters['random_state'])
            apply_SMOTE = False    

        elif model == "Regresion":
            clf = LogisticRegression(random_state=parameters['random_state'],
                                     solver=parameters['solver'],
                                     max_iter = parameters['max_iter'])
            
        elif model == "Naive Bayes":
            clf = GaussianNB()

        elif model == "Ada Boost":
            tree = DecisionTreeClassifier(criterion = parameters['criterion'],
                                          max_depth=parameters['max_depth'],
                                          random_state=parameters['random_state'])

            clf = AdaBoostClassifier(base_estimator=tree,
                                    n_estimators=parameters['n_estimators'], 
                                    learning_rate=parameters['learning_rate'],
                                    random_state=parameters['random_state'])
        elif model == "KNN":
            clf = KNeighborsClassifier(n_neighbors=parameters['n_neighbors'])
            #clf.fit(self.x_train, self.y_train)
            apply_SMOTE = False
        
        elif model == "SVC":
            clf =  svm.SVC()
            #clf.fit(self.x_train, self.y_train)
            apply_SMOTE = False

        if apply_SMOTE:
            pipeline = apply_classifier_with_SMOTE(clf, self.x_train, self.y_train, preprocessor)
        else:
            if not preprocessor:
                clf.fit(self.x_train, self.y_train)
                pipeline = clf
            else:
                pipeline = make_pipeline(preprocessor,
                                         clf)
                pipeline.fit(self.x_train, self.y_train)

        return pipeline


def plot_cat_histogram(feature, data):
    """Inprime el histograma de una característica categórica

    Args:
        feature (str): nombre de la característica
        data (DataFrame): DataFrame con los datos
    """    
    print(data[feature].value_counts())
    fig, ax = plt.subplots( figsize = (6,6) )
    bar = sns.countplot(x=feature, ax=ax, data=data)
    
    bar.set_xticklabels(bar.get_xticklabels(), rotation=45,
                        horizontalalignment='right')

    plt.show()



def find_best_threshold(predictions, y_true):
    """Encuentra el mejor umbral de probabilidad para aumentar la precisi+on y para aumentar las ganancias del modelo.

    Args:
        predictions (vector): Vector con las predicciones.
        y_true (vector): Vector con los verdaderos valores.

    Returns:
        acceptance_threshold (Lista de flot): Lista de valores del umbral. 
        accurracy_list (Lista de float): Lista de precisiónes de acuerdo al umbral.
        best_threshold (Float): Umbral con la mejor precisión.
        best_accurracy (Float): La mejor precisión obtenida.
        gain_list (Lista de float): Lista de ganacias en función de la lista valores de umbral.
        best_gthreshodl (Float): El umbral que maximisa las ganancias.
        best_gain (Float): La mejor ganacia.
    """    
    acceptance_threshold = np.linspace(
                start = 0.0005,
                stop  = 1.0,
                num   = 2000
            )

    accurracy_list =[]
    best_threshold = 0.0
    best_accurracy = 0.0

    gain_list = []
    best_gthreshodl = 0.0
    best_gain = -999_000_000.0

    for threshold in acceptance_threshold:
        y_pred = np.where(predictions > threshold, 1, 0)

        # Calcular el umbral para la mejor precisión (accurracy)
        accurracy = accuracy_score(
            y_true = y_true,
            y_pred    = y_pred,
            normalize = True)
        accurracy_list.append(accurracy)
        if accurracy > best_accurracy:
            best_accurracy = accurracy
            best_threshold = threshold

      # Calcular el umbral para la mejor ganacia monetaria
        gain, _ , _ = compute_gain(y_true, y_pred)
        gain_list.append(gain)
        if gain > best_gain:
            best_gain = gain
            best_gthreshodl = threshold
    
    return acceptance_threshold, accurracy_list, best_threshold, best_accurracy, \
           gain_list, best_gthreshodl, best_gain
 
def compute_gain(y_true, y_pred):
    """Calcula las ganacias a partir de las prediciones del modelo.

    Args:
        y_true (list): Valores verdaderos.
        y_pred (list): Valores predecidos.

    Returns:
        gain: Ganancia en pesos.
        reference: Valor de referencia de las ganancias sin modelo.
        improvement_type_: Porcentaje de mejora de la ganancia con respecto al valor de referencia.
    """    
    gain_for_loan = 1847
    lost_for_umpaind = 5884

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    gain = float((cm[0][0]-cm[0][1])*gain_for_loan-cm[1][0]*lost_for_umpaind)
    reference = float((cm[0][0]+cm[0][1])*gain_for_loan - (cm[1][0]+cm[1][1])*lost_for_umpaind)

    if reference >= 0:
        improvement = (gain-reference)/reference
    else:
        improvement = (reference-gain)/reference

    return gain, reference, improvement

def compute_rejection_rate(y_pred):
    non_rejected = np.count_nonzero(y_pred == 0)
    rejected = np.count_nonzero(y_pred == 1)

    return rejected/(non_rejected+rejected)


def apply_classifier_with_SMOTE(clf, x_train, y_train, preprocessor=None):
    """Crea un clasificador con la técnica 'Synthetic Minority Oversampling Technique' (SMOTE) 
    regresando el pipeline correspondiente.

    
    Args:
        clf (clasiicador): Clasificador al que se le agregara el SMOTE
        x_train (array): Datos de entrenamiento de los predictores de entrenamiento
        y_train (array): Datos de entrenamiento de la variable target

    Returns:
        pipeline: Pipeline con el classificador y el SMOTE
    """    
    smote_sampler = SMOTE(random_state=9, sampling_strategy=1.0)
    
    if not preprocessor:
        pipeline = Pipeline(steps = [['smote', smote_sampler],
                                     ['classifier', clf]])
    else:
        pipeline = Pipeline(steps = [['preprocessor', preprocessor],
                                     ['smote', smote_sampler],
                                     ['classifier', clf]])
    
    pipeline.fit(x_train, y_train)

    return pipeline

def plot_threshold_evolution(threshold_list, evol_list, title, xlabel, ylabel):
    """Dibuja la gráfica de la evolución del umbral con respecto a una segunda variable asociada al umbral.

    Args:
        threshold_list (list): Lista con los umbrales.
        evol_list (list): Lista con la evolución de la variable asociada al umbral.
        title (str): Título de la gráfica
        xlabel (srt): Etiqueta de las x
        ylabel (str): Etiqueta de las y
    """    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(threshold_list, evol_list, color = "#0A7BBB")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.show()


