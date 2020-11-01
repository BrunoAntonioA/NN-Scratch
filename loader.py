import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from pathlib import Path

#por ahora todo lo que tenga que ver con el segundo dataset quedará comentado

def load_csv(path_to_file):
    #carga csv con una dirección por defecto
    if path_to_file is None:
        raw_data1 = pd.read_csv(r'./Desafio_4/Fashion-DataSet-master/fashion-1.csv')
        #raw_data2 = pd.read_csv(r'./Desafio_4/Fashion-DataSet-master/fashion-2.csv')
        album = raw_data1
        return album
    
    #limpiando string para usar como argumento
    path_to_file_1 = (path_to_file + "/fashion-1.csv")
    #path_to_file_2 = (path_to_file + "/fashion-2.csv")
    clean_path_1 = Path(path_to_file_1)
    #clean_path_2 = Path(path_to_file_2)

    #cargando csv con dirección ingresada
    raw_data1 = pd.read_csv(clean_path_1)
    #raw_data2 = pd.read_csv(path_to_file)

    #album = [raw_data1, raw_data2]
    album = raw_data1
    return album

#Normalización de datos
def data_normalization(album):
    min_max_scaler = preprocessing.MinMaxScaler()
    aux = None
    df = None
    for i in range(len(album)):
        df = album[i].values
        aux = min_max_scaler.fit_transform(df)
        album[i] = pd.DataFrame(aux)

    return album

#separación de datos en entrenamiento y pruebas
def data_divider(album):                            #no olvidar ajustar al añadir dataset2

    ds1_train, ds1_test = train_test_split(album, test_size=0.3, train_size=0.7, shuffle=False)
    #ds2_train, ds2_test = train_test_split(album[1], test_size=0.3, train_size=0.7, shuffle=False)
    sets = [ds1_train, ds1_test]                      #ds2_train, ds2_test
    return sets

#recibe los grupos de entrenamiento y pruebas y les quita la primera columna correspondientes a las labels
#retorna una lista con las listas de labels correspondientes a cada grupo
def cleaner(samples):
    label1_train = samples[0].pop('label')              #arreglar esto cuando se añada dataset2
    label1_test = samples[1].pop('label')
    #label2_train = samples[2].pop('label')             #arreglar esto cuando se añada dataset2
    #label2_test = samples[3].pop('label')
    labels = [label1_train, label1_test]                #arreglar esto cuando se añada dataset2

    return labels

#recibe un dataframe y lo convierte en un arreglo/matriz con numpy
def numpyfyer(df):
    return df.to_numpy()

#recorre los dataframe para convertirlos en listas y los agrega a una única lista
def organizer(samples, labels):
    organized = []
    for i in range(len(samples)):
        samples[i] = numpyfyer(samples[i])
        organized.append(samples[i])
    for j in range(len(labels)):
        labels[j] = numpyfyer(labels[j])
        organized.append(labels[j])
    
    return organized

#función de llamada para captura de datos, el argumento no debe contener el nombre del archivo
def get_datos(path_to_file):
    data = load_csv(path_to_file)                           #carga el csv
    samples = data_divider(data)                            #divide el dataframe en muestras de entrenamiento y pruebas
    labels = cleaner(samples)                               #obtiene los labels de cada muestra
    data = data_normalization(samples)                      #normaliza los datos y retorna el/los dataframe
    final_list = organizer(data, labels)                    #junta todo en una lista final con todo listo

    return final_list


#datos = get_datos('./Fashion-DataSet-master/')

# primero dos elementos son los dataset 
# los ultimos dos son las etiquetas 



